#!/usr/bin/env python3
"""Collect hidden state activations from the trained Coconut model at each latent thought step."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tqdm import tqdm

from src.data_gen import make_tokenizer, BOT
from src.coconut_model import CoconutGPT2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data"
CKPT_DIR = "checkpoints"
BATCH_SIZE = 1  # Process one at a time for variable-length latent steps


def collect_from_checkpoint(model, data, tokenizer, max_samples=None, desc="Collecting"):
    """Collect activations from a trained Coconut model.

    Returns list of dicts, each containing:
        - problem: str
        - answer: str
        - cot_steps: list of str (ground truth)
        - num_steps: int
        - thought_hiddens: list of (D,) tensors (last-layer hidden per thought)
        - layer_hiddens: list of dicts {layer_idx: (D,) tensor}
    """
    model.eval()
    results = []
    n = len(data["problems"]) if max_samples is None else min(max_samples, len(data["problems"]))

    for i in tqdm(range(n), desc=desc):
        problem = data["problems"][i]
        cot_steps = data["cot_steps"][i]
        answer = data["answers"][i]
        num_steps = len(cot_steps)

        # Tokenize prefix (problem + <bot>)
        prefix_text = f"{problem} {BOT}"
        prefix_ids = tokenizer.encode(prefix_text, return_tensors="pt").to(DEVICE)

        # Generate with latent thoughts
        generated, thought_hiddens, layer_hiddens = model.generate_answer(
            prefix_ids, num_latent_steps=num_steps, max_new_tokens=20
        )

        # Decode generated answer
        pred_answer = tokenizer.decode(generated).strip()

        results.append({
            "problem": problem,
            "answer": answer,
            "pred_answer": pred_answer,
            "cot_steps": cot_steps,
            "num_steps": num_steps,
            "thought_hiddens": [h.squeeze(0).cpu() for h in thought_hiddens],  # list of (D,)
            "layer_hiddens": [
                {k: v.squeeze(0).cpu() for k, v in lh.items()}
                for lh in layer_hiddens
            ],  # list of {layer: (D,)}
        })

    return results


def main():
    print("Loading tokenizer and model...")
    tokenizer = make_tokenizer()

    model = CoconutGPT2(tokenizer, device=DEVICE)
    model = model.to(DEVICE)

    # Collect from the final (all-latent) checkpoint
    final_ckpt = os.path.join(CKPT_DIR, "stage3_all_latent.pt")
    if not os.path.exists(final_ckpt):
        # Fall back to whatever the latest stage is
        for stage in ["stage3_all_latent", "stage2_last2_latent", "stage1_last_latent", "stage0_text_cot"]:
            path = os.path.join(CKPT_DIR, f"{stage}.pt")
            if os.path.exists(path):
                final_ckpt = path
                break

    print(f"Loading checkpoint: {final_ckpt}")
    model.load_state_dict(torch.load(final_ckpt, map_location=DEVICE))

    # Collect from train, val, test, ood
    for split in ["train", "val", "test", "ood"]:
        data = torch.load(os.path.join(DATA_DIR, f"{split}.pt"), weights_only=False)
        max_samples = {"train": 50000, "val": 5000, "test": 5000, "ood": 1000}.get(split, 5000)

        print(f"\nCollecting activations for {split} ({max_samples} samples)...")
        results = collect_from_checkpoint(model, data, tokenizer, max_samples, desc=split)

        # Print accuracy
        correct = sum(1 for r in results if r["pred_answer"] == r["answer"])
        print(f"{split} accuracy: {correct}/{len(results)} = {correct/len(results):.4f}")

        # Save
        out_path = os.path.join(DATA_DIR, f"activations_{split}.pt")
        torch.save(results, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
