"""GPT-2 with Coconut (Chain of Continuous Thought) support.

The model processes a text prefix, then performs N latent thought steps where
hidden states are fed back as inputs (instead of being decoded to tokens),
then decodes the answer autoregressively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config
from src.data_gen import BOT, SEP, EOT, ACT


class CoconutGPT2(nn.Module):
    """GPT-2 wrapper with continuous thought support."""

    def __init__(self, tokenizer, device="cuda"):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer

        # Load GPT-2 and resize for special tokens
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.resize_token_embeddings(len(tokenizer))

        self.config = self.model.config
        self.d_model = self.config.n_embd  # 768
        self.n_layers = self.config.n_layer  # 12

        # Special token IDs
        self.bot_id = tokenizer.convert_tokens_to_ids(BOT)
        self.sep_id = tokenizer.convert_tokens_to_ids(SEP)
        self.eot_id = tokenizer.convert_tokens_to_ids(EOT)
        self.act_id = tokenizer.convert_tokens_to_ids(ACT)
        self.pad_id = tokenizer.pad_token_id

    def get_embeddings(self, input_ids):
        """Get token + position embeddings for input_ids. Shape: (B, L, D)."""
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        tok_emb = self.model.transformer.wte(input_ids)
        pos_emb = self.model.transformer.wpe(positions)
        return tok_emb + pos_emb

    def forward_text(self, input_ids, attention_mask=None):
        """Standard GPT-2 forward pass on text tokens. Returns outputs with past_key_values."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )

    def forward_one_latent_step(self, hidden_state, past_key_values, position_id):
        """Process one latent thought step.

        Args:
            hidden_state: (B, D) last-layer hidden state from previous step
            past_key_values: KV cache from all previous positions
            position_id: int, the position index for this thought step

        Returns:
            new_hidden: (B, D) last-layer hidden state
            new_past: updated KV cache
            all_hidden: dict mapping layer_idx -> (B, D) hidden state at that layer
        """
        # Input = previous hidden state + positional embedding
        pos_emb = self.model.transformer.wpe(
            torch.tensor([position_id], device=hidden_state.device)
        )  # (1, D)
        inputs_embeds = hidden_state.unsqueeze(1) + pos_emb.unsqueeze(0)  # (B, 1, D)

        # Build attention mask: attend to all past + this position
        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        total_len = past_len + 1
        attn_mask = torch.ones(1, total_len, device=hidden_state.device)

        outputs = self.model.transformer(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attn_mask,
            use_cache=True,
            output_hidden_states=True,
        )

        new_hidden = outputs.last_hidden_state[:, -1, :]  # (B, D)
        new_past = outputs.past_key_values

        # Collect hidden states at specific layers for activation oracle
        all_hidden = {}
        for layer_idx in [3, 6, 9, 11]:  # 25%, 50%, 75%, last
            all_hidden[layer_idx] = outputs.hidden_states[layer_idx + 1][:, -1, :]  # +1 because index 0 is input embeds

        return new_hidden, new_past, all_hidden

    def forward_answer(self, answer_ids, past_key_values, start_position):
        """Process answer tokens with KV cache from prefix + thoughts.

        Args:
            answer_ids: (B, L_answer) token IDs starting with <eot>
            past_key_values: KV cache
            start_position: position ID for first answer token

        Returns:
            logits: (B, L_answer, V) logits for answer tokens
        """
        positions = torch.arange(
            start_position, start_position + answer_ids.size(1),
            device=answer_ids.device
        )
        tok_emb = self.model.transformer.wte(answer_ids)
        pos_emb = self.model.transformer.wpe(positions)
        inputs_embeds = tok_emb + pos_emb

        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        total_len = past_len + answer_ids.size(1)
        attn_mask = torch.ones(1, total_len, device=answer_ids.device)

        outputs = self.model.transformer(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attn_mask,
            use_cache=True,
        )
        logits = self.model.lm_head(outputs.last_hidden_state)
        return logits, outputs.past_key_values

    def forward_coconut(
        self,
        prefix_ids,
        num_latent_steps,
        answer_ids=None,
        collect_hidden=False,
    ):
        """Full Coconut forward pass: text prefix -> latent thoughts -> answer.

        Args:
            prefix_ids: (B, L_prefix) token IDs for problem + any text CoT
            num_latent_steps: int, number of latent thought steps
            answer_ids: (B, L_answer) token IDs for <eot> + answer (for training)
            collect_hidden: if True, collect hidden states at each latent step

        Returns:
            dict with 'answer_logits', 'thought_hiddens', 'thought_norms',
            and optionally 'layer_hiddens'
        """
        B = prefix_ids.size(0)

        # 1. Process text prefix
        prefix_out = self.forward_text(prefix_ids)
        last_hidden = prefix_out.hidden_states[-1][:, -1, :]  # (B, D)
        past_kv = prefix_out.past_key_values
        next_pos = prefix_ids.size(1)

        # 2. Latent thought loop
        thought_hiddens = []
        thought_norms = []
        layer_hiddens = []  # list of dicts

        h = last_hidden
        for t in range(num_latent_steps):
            h, past_kv, all_h = self.forward_one_latent_step(h, past_kv, next_pos + t)
            thought_hiddens.append(h)
            thought_norms.append(h.norm(dim=-1).mean())
            if collect_hidden:
                layer_hiddens.append(all_h)

        # 3. Process answer if provided
        answer_logits = None
        if answer_ids is not None:
            answer_start_pos = next_pos + num_latent_steps
            answer_logits, _ = self.forward_answer(answer_ids, past_kv, answer_start_pos)

        return {
            "answer_logits": answer_logits,
            "thought_hiddens": thought_hiddens,
            "thought_norms": thought_norms,
            "layer_hiddens": layer_hiddens,
            "past_key_values": past_kv,
        }

    def forward_text_only(self, input_ids, labels=None):
        """Standard text-only forward (Stage 0 training). Returns loss and logits."""
        out = self.model(input_ids=input_ids, labels=labels)
        return out.loss, out.logits

    @torch.no_grad()
    def generate_answer(self, prefix_ids, num_latent_steps, max_new_tokens=20):
        """Generate answer autoregressively after latent thinking.

        Args:
            prefix_ids: (1, L) token IDs
            num_latent_steps: how many latent thought steps
            max_new_tokens: max answer tokens to generate

        Returns:
            generated_ids: list of token IDs (answer only, excluding <eot>)
            thought_hiddens: list of (D,) tensors for each thought step
            layer_hiddens: list of dicts mapping layer -> (D,) tensor
        """
        result = self.forward_coconut(
            prefix_ids, num_latent_steps, collect_hidden=True
        )
        past_kv = result["past_key_values"]
        next_pos = prefix_ids.size(1) + num_latent_steps

        # Start with <eot> token
        curr_id = torch.tensor([[self.eot_id]], device=prefix_ids.device)
        generated = []

        for i in range(max_new_tokens):
            pos = torch.tensor([next_pos + i], device=prefix_ids.device)
            tok_emb = self.model.transformer.wte(curr_id)
            pos_emb = self.model.transformer.wpe(pos)
            inputs_embeds = tok_emb + pos_emb

            past_len = past_kv[0][0].size(2)
            attn_mask = torch.ones(1, past_len + 1, device=prefix_ids.device)

            out = self.model.transformer(
                inputs_embeds=inputs_embeds,
                past_key_values=past_kv,
                attention_mask=attn_mask,
                use_cache=True,
            )
            past_kv = out.past_key_values
            logits = self.model.lm_head(out.last_hidden_state[:, -1, :])
            next_token = logits.argmax(dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated.append(next_token.item())
            curr_id = next_token.unsqueeze(0)

        return generated, result["thought_hiddens"], result["layer_hiddens"]
