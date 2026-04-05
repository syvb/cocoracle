"""Activation Oracle: GPT-2 + LoRA with norm-matched activation injection.

Injects hidden states from the Coconut model into placeholder tokens,
then answers questions about what those activations encode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D
from src.data_gen import ACT


class LoRAConv1D(nn.Module):
    """LoRA adapter for HuggingFace Conv1D (used in GPT-2).

    Conv1D computes y = xW + b where W is (in_feat, out_feat).
    LoRA adds: y += x @ A @ B * scaling
    """

    def __init__(self, original: Conv1D, rank=32, alpha=64, dropout=0.05):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Conv1D weight shape is (in_feat, out_feat)
        in_feat = original.weight.shape[0]
        out_feat = original.weight.shape[1]
        self.lora_A = nn.Parameter(torch.randn(in_feat, rank) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_feat))
        self.dropout = nn.Dropout(dropout)

        # Freeze original weights
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x):
        # Conv1D forward: x @ weight + bias
        base_out = self.original(x)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return base_out + lora_out


def apply_lora(model, rank=32, alpha=64, dropout=0.05):
    """Apply LoRA to all Conv1D layers in the transformer blocks."""
    lora_params = []
    for name, module in list(model.named_modules()):
        if isinstance(module, Conv1D) and "transformer.h" in name:
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]

            lora = LoRAConv1D(module, rank, alpha, dropout)
            setattr(parent, attr, lora)
            lora_params.extend([lora.lora_A, lora.lora_B])

    return lora_params


class ActivationOracle(nn.Module):
    """Activation Oracle model with injection mechanism."""

    def __init__(self, tokenizer, device="cuda", lora_rank=32, lora_alpha=64):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.act_id = tokenizer.convert_tokens_to_ids(ACT)

        # Load fresh GPT-2 and resize
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.resize_token_embeddings(len(tokenizer))

        self.d_model = self.model.config.n_embd  # 768
        self.injection_layer = 1  # inject after 2nd transformer block (0-indexed)

        # Apply LoRA
        self.lora_params = apply_lora(self.model, lora_rank, lora_alpha)

        # Freeze all params, then unfreeze LoRA
        for param in self.model.parameters():
            param.requires_grad_(False)

        lora_ids = {id(p) for p in self.lora_params}
        for param in self.model.parameters():
            if id(param) in lora_ids:
                param.requires_grad_(True)

        # Injection state (set before forward pass)
        self._injection_vectors = None  # (K, D) activation vectors to inject
        self._injection_positions = None  # list of int positions of <act> tokens

        # Register hook
        self._hook = self.model.transformer.h[self.injection_layer].register_forward_hook(
            self._injection_hook
        )

    def _injection_hook(self, module, input, output):
        """Modify residual stream at <act> positions after the injection layer."""
        if self._injection_vectors is None or self._injection_positions is None:
            return output

        # output is a tuple: (hidden_states, ...) or (hidden_states, presents, ...)
        hidden_states = output[0]  # (B, L, D)
        seq_len = hidden_states.size(1)

        # Only inject if positions are within the current sequence length
        # (during generation with KV-cache, seq_len=1 for autoregressive steps)
        positions_in_range = [
            (i, pos) for i, pos in enumerate(self._injection_positions)
            if pos < seq_len and i < len(self._injection_vectors)
        ]

        if not positions_in_range:
            return output

        hidden_states = hidden_states.clone()
        for i, pos in positions_in_range:
            h_i = hidden_states[:, pos, :]  # (B, D)
            v_i = self._injection_vectors[i].to(h_i.device)  # (D,)

            # Norm-matched addition: h' = h + ||h|| * v / ||v||
            h_norm = h_i.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, 1)
            v_norm = v_i.norm().clamp(min=1e-8)
            hidden_states[:, pos, :] = h_i + h_norm * (v_i / v_norm)

        return (hidden_states,) + output[1:]

    def set_injection(self, vectors, positions):
        """Set activation vectors and positions for injection.

        Args:
            vectors: list of (D,) tensors or (K, D) tensor
            positions: list of int token positions
        """
        if isinstance(vectors, list):
            vectors = torch.stack(vectors)
        self._injection_vectors = vectors.to(self.device)
        self._injection_positions = positions

    def clear_injection(self):
        self._injection_vectors = None
        self._injection_positions = None

    def forward(self, input_ids, labels=None, attention_mask=None):
        """Forward pass with injection active (if set).

        Args:
            input_ids: (B, L) token IDs including <act> placeholders
            labels: (B, L) target token IDs (-100 for non-target positions)
            attention_mask: (B, L) attention mask

        Returns:
            loss (if labels provided), logits
        """
        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        if labels is not None:
            return outputs.loss, outputs.logits
        return outputs.logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64):
        """Generate response with injection active."""
        generated = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Return only newly generated tokens
        return generated[:, input_ids.size(1):]

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_params())
