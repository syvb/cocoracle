"""Self-Oracle: The Coconut model interprets its own latent activations.

Instead of a separate oracle model, we fine-tune the Coconut model itself
to answer questions about its own hidden states. The model already understands
arithmetic and its own internal representations.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from src.data_gen import ACT


class SelfOracle(nn.Module):
    """Coconut model that also serves as its own activation oracle.

    Uses norm-matched activation injection, applied to the model that
    produced the activations in the first place (self-interpretation).
    """

    def __init__(self, tokenizer, device="cuda", injection_layer=1, injection_scale=2.0):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.act_id = tokenizer.convert_tokens_to_ids(ACT)

        # Load GPT-2 and resize (will load Coconut checkpoint on top)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.resize_token_embeddings(len(tokenizer))

        self.d_model = self.model.config.n_embd  # 768
        self.injection_layer = injection_layer
        self.injection_scale = injection_scale

        # Injection state
        self._injection_vectors = None
        self._injection_positions = None

        # Register hook on the injection layer
        self._hook = self.model.transformer.h[self.injection_layer].register_forward_hook(
            self._injection_hook
        )

    def _injection_hook(self, module, input, output):
        """Norm-matched activation injection after the specified layer."""
        if self._injection_vectors is None or self._injection_positions is None:
            return output

        hidden_states = output[0]
        seq_len = hidden_states.size(1)

        positions_in_range = [
            (i, pos) for i, pos in enumerate(self._injection_positions)
            if pos < seq_len and i < len(self._injection_vectors)
        ]

        if not positions_in_range:
            return output

        # Build a delta tensor (all zeros except at injection positions)
        # This avoids in-place modification entirely
        delta = torch.zeros_like(hidden_states)
        for i, pos in positions_in_range:
            h_i = hidden_states[:, pos, :]
            v_i = self._injection_vectors[i].to(h_i.device)

            h_norm = h_i.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v_norm = v_i.norm().clamp(min=1e-8)
            delta[:, pos, :] = self.injection_scale * h_norm * (v_i / v_norm)

        return (hidden_states + delta,) + output[1:]

    def set_injection(self, vectors, positions):
        if isinstance(vectors, list):
            vectors = torch.stack(vectors)
        self._injection_vectors = vectors.to(self.device)
        self._injection_positions = positions

    def clear_injection(self):
        self._injection_vectors = None
        self._injection_positions = None

    def forward(self, input_ids, labels=None, attention_mask=None):
        return self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64):
        return self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
