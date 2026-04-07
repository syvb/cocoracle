#!/usr/bin/env python3
"""Interactive demo: watch the Coconut model think, then ask the oracle what it's thinking.

Usage:
    python scripts/interactive.py                          # uses GPT-2-large checkpoints
    python scripts/interactive.py --coconut checkpoints/large/stage3_alllatent.pt \
                                  --oracle checkpoints/large/self_oracle_alllatent.pt
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import random
import torch
from transformers import GPT2LMHeadModel

from src.data_gen import make_tokenizer, BOT, SEP, EOT, ACT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(tokenizer, ckpt_path, model_name="gpt2-large"):
    """Load a GPT-2 model with special tokens and a checkpoint."""
    model = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


class OracleHook:
    """Activation injection hook for the self-oracle."""
    def __init__(self, layer_module, scale=2.0):
        self.scale = scale
        self.vec = None
        self.pos = None
        self.handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        if self.vec is None or self.pos is None:
            return output
        hs = output[0]
        if self.pos >= hs.size(1):
            return output
        delta = torch.zeros_like(hs)
        h_i = hs[:, self.pos, :]
        v = self.vec.to(h_i.device).to(h_i.dtype)
        h_norm = h_i.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v_norm = v.norm().clamp(min=1e-8)
        delta[:, self.pos, :] = self.scale * h_norm * (v / v_norm)
        return (hs + delta,) + output[1:]

    def inject(self, vec, pos):
        self.vec = vec
        self.pos = pos

    def clear(self):
        self.vec = None
        self.pos = None


@torch.no_grad()
def coconut_think(model, tokenizer, problem_text, n_latent_steps, extract_layer=18):
    """Run the Coconut model: text prefix -> latent thoughts -> answer.

    Returns (answer_text, list_of_hidden_states, list_of_layer_hidden_states).
    """
    prefix = f"{problem_text} {BOT}"
    prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(DEVICE)

    # Forward prefix
    out = model.transformer(prefix_ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    h = out.last_hidden_state[:, -1:, :]

    thought_hiddens = []
    for t in range(n_latent_steps):
        lat = model.transformer(
            inputs_embeds=h, past_key_values=past_kv,
            use_cache=True, output_hidden_states=True
        )
        past_kv = lat.past_key_values
        h = lat.last_hidden_state
        # Extract from the target layer
        h_extract = lat.hidden_states[extract_layer + 1][0, 0, :]  # (D,)
        thought_hiddens.append(h_extract)

    # Generate answer: feed <eot> then decode
    eot_id = tokenizer.convert_tokens_to_ids(EOT)
    eos_id = tokenizer.eos_token_id
    curr = torch.tensor([[eot_id]], device=DEVICE)
    answer_tokens = []

    for _ in range(15):
        emb = model.transformer.wte(curr)
        out = model.transformer(inputs_embeds=emb, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        logits = model.lm_head(out.last_hidden_state[0, -1])
        nxt = logits.argmax().item()
        if nxt == eos_id:
            break
        answer_tokens.append(nxt)
        # Stop on non-digit
        text_so_far = tokenizer.decode(answer_tokens).strip()
        if text_so_far and not text_so_far.isdigit():
            num = ""
            for c in text_so_far:
                if c.isdigit():
                    num += c
                elif num:
                    break
            if num:
                return num, thought_hiddens
            break
        curr = torch.tensor([[nxt]], device=DEVICE)

    answer = tokenizer.decode(answer_tokens).strip()
    return answer, thought_hiddens


@torch.no_grad()
def oracle_interpret(model, hook, tokenizer, hidden_state, question=None):
    """Ask the oracle to interpret a hidden state."""
    if question is None:
        question = "What is the intermediate calculation at this reasoning step?"

    prompt = f"Layer 18: {ACT} {question}"
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    act_id = tokenizer.convert_tokens_to_ids(ACT)
    eos_id = tokenizer.eos_token_id
    special_ids = [tokenizer.convert_tokens_to_ids(t) for t in [BOT, SEP, EOT, ACT]]

    act_pos = [j for j, t in enumerate(prompt_ids[0].tolist()) if t == act_id]
    if act_pos:
        hook.inject(hidden_state, act_pos[0])

    # Generate with EOS suppression for first 3 tokens
    generated = []
    past_kv = None
    curr_ids = prompt_ids
    for s in range(30):
        out = model(curr_ids, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        logits = out.logits[0, -1].float()
        if s < 3:
            logits[eos_id] = -float('inf')
            for sid in special_ids:
                logits[sid] = -float('inf')
        nxt = logits.argmax().item()
        if nxt == eos_id:
            break
        generated.append(nxt)
        curr_ids = torch.tensor([[nxt]], device=DEVICE)

    hook.clear()
    return tokenizer.decode(generated).strip()


def main():
    parser = argparse.ArgumentParser(description="Interactive Cocoracle demo")
    parser.add_argument("--coconut", default="checkpoints/large/stage3_alllatent.pt",
                        help="Path to Coconut model checkpoint")
    parser.add_argument("--oracle", default="checkpoints/large/self_oracle_alllatent.pt",
                        help="Path to self-oracle checkpoint")
    parser.add_argument("--model-name", default="gpt2-large",
                        help="Base model name (gpt2, gpt2-medium, gpt2-large)")
    parser.add_argument("--extract-layer", type=int, default=18,
                        help="Layer to extract activations from")
    parser.add_argument("--inject-layer", type=int, default=17,
                        help="Layer to inject activations into for oracle")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = make_tokenizer()

    print(f"Loading Coconut model from {args.coconut}...")
    coconut = load_model(tokenizer, args.coconut, args.model_name)

    print(f"Loading self-oracle from {args.oracle}...")
    oracle = load_model(tokenizer, args.oracle, args.model_name)
    hook = OracleHook(oracle.transformer.h[args.inject_layer], scale=2.0)

    print()
    print("=" * 60)
    print("  COCORACLE: Watch the model think, ask what it's thinking")
    print("=" * 60)
    print()
    print("Commands:")
    print("  Enter an addition problem like: 347 + 285")
    print("  Or type 'random' for a random problem")
    print("  Or type 'quit' to exit")
    print()

    while True:
        try:
            user = input("\033[1mProblem>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            break

        if user.lower() == "random":
            nd = random.choice([2, 3, 4])
            lo, hi = 10**(nd-1), 10**nd - 1
            a, b = random.randint(lo, hi), random.randint(lo, hi)
            user = f"{a} + {b}"
            print(f"  -> {user}")

        # Parse the problem
        try:
            parts = user.replace("=", "").split("+")
            a = int(parts[0].strip())
            b = int(parts[1].strip())
        except (ValueError, IndexError):
            print("  Please enter a problem like: 347 + 285")
            continue

        gold_answer = a + b
        problem_text = f"{a} + {b} ="

        # Figure out how many CoT steps (= number of digit columns + possible carry)
        digits_a = [int(d) for d in str(a)][::-1]
        digits_b = [int(d) for d in str(b)][::-1]
        max_digits = max(len(digits_a), len(digits_b))
        # Compute ground-truth CoT steps
        carry = 0
        gold_steps = []
        for i in range(max_digits):
            da = digits_a[i] if i < len(digits_a) else 0
            db = digits_b[i] if i < len(digits_b) else 0
            total = da + db + carry
            digit = total % 10
            new_carry = total // 10
            if carry > 0:
                step = f"{da}+{db}+{carry}={total} write {digit}"
            else:
                step = f"{da}+{db}={total} write {digit}"
            if new_carry > 0:
                step += f" carry {new_carry}"
            carry = new_carry
            gold_steps.append(step)
        if carry > 0:
            gold_steps.append(f"carry {carry} write {carry}")

        n_steps = len(gold_steps)

        print()
        print(f"  \033[90mGold answer: {gold_answer}\033[0m")
        print(f"  \033[90mGold CoT: {' | '.join(gold_steps)}\033[0m")
        print()

        # Run Coconut
        print("  Running Coconut (latent reasoning)...")
        answer, hiddens = coconut_think(
            coconut, tokenizer, problem_text, n_steps, args.extract_layer
        )
        correct = answer == str(gold_answer)
        mark = "\033[32mCORRECT\033[0m" if correct else f"\033[31mWRONG\033[0m (got {answer})"
        print(f"  Model answer: {answer} {mark}")
        print()

        # Ask oracle about each thought step
        print("  Asking oracle about each latent thought:")
        for t, h in enumerate(hiddens):
            gold = gold_steps[t] if t < len(gold_steps) else "?"
            interpretation = oracle_interpret(oracle, hook, tokenizer, h)
            match = "\033[32mMATCH\033[0m" if interpretation == gold else ""
            print(f"    Step {t+1}: oracle says \033[1m{interpretation}\033[0m")
            print(f"           gold:       {gold}  {match}")

        print()


if __name__ == "__main__":
    main()
