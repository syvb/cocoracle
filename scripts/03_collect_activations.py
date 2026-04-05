#!/usr/bin/env python3
"""Collect hidden state activations from the trained Coconut model at each latent thought step."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tqdm import tqdm

from src.data_gen import make_tokenizer, BOT, SEP
from src.coconut_model import CoconutGPT2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data"
CKPT_DIR = "checkpoints"
BATCH_SIZE = 1  # Process one at a time for variable-length latent steps


def collect_from_checkpoint(model, data, tokenizer, num_latent_mode, max_samples=None, desc="Collecting"):
    """Collect activations from a trained Coconut model.

    Args:
        num_latent_mode: int or "all". If int, only the last N steps are latent.
                         If "all", all steps are latent.

    Returns list of dicts, each containing:
        - problem: str
        - answer: str
        - cot_steps: list of str (ground truth)
        - num_latent: int (how many steps were latent)
        - latent_cot_steps: list of str (the CoT steps that are latent)
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
        total_steps = len(cot_steps)

        # Determine how many latent steps for this example
        if num_latent_mode == "all":
            n_latent = total_steps
        else:
            n_latent = min(num_latent_mode, total_steps)

        n_text = total_steps - n_latent

        # Build prefix: problem + <bot> + text CoT steps (if any)
        prefix_text = f"{problem} {BOT}"
        for j in range(n_text):
            prefix_text += f" {cot_steps[j]} {SEP}"

        prefix_ids = tokenizer.encode(prefix_text, return_tensors="pt").to(DEVICE)

        # The latent CoT steps (ground truth for the latent positions)
        latent_cot_steps = cot_steps[n_text:]

        # Generate with latent thoughts
        generated, thought_hiddens, layer_hiddens = model.generate_answer(
            prefix_ids, num_latent_steps=n_latent, max_new_tokens=20
        )

        pred_answer = tokenizer.decode(generated).strip()

        results.append({
            "problem": problem,
            "answer": answer,
            "pred_answer": pred_answer,
            "cot_steps": cot_steps,
            "latent_cot_steps": latent_cot_steps,
            "num_latent": n_latent,
            "num_steps": total_steps,
            "thought_hiddens": [h.squeeze(0).cpu() for h in thought_hiddens],
            "layer_hiddens": [
                {k: v.squeeze(0).cpu() for k, v in lh.items()}
                for lh in layer_hiddens
            ],
        })

    return results


def collect_for_stage(model, tokenizer, stage_name, ckpt_path, num_latent_mode):
    """Collect activations for a specific stage checkpoint."""
    print(f"\n{'='*60}")
    print(f"Collecting from {stage_name}: {ckpt_path}")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    for split in ["train", "val", "test", "ood"]:
        data = torch.load(os.path.join(DATA_DIR, f"{split}.pt"), weights_only=False)
        max_samples = {"train": 10000, "val": 2000, "test": 2000, "ood": 500}.get(split, 2000)

        print(f"\nCollecting {split} ({max_samples} samples)...")
        results = collect_from_checkpoint(model, data, tokenizer, num_latent_mode, max_samples, desc=split)

        correct = sum(1 for r in results if r["pred_answer"] == r["answer"])
        print(f"{split} accuracy: {correct}/{len(results)} = {correct/len(results):.4f}")

        out_path = os.path.join(DATA_DIR, f"activations_{stage_name}_{split}.pt")
        torch.save(results, out_path)
        print(f"Saved: {out_path}")


def main():
    print("Loading tokenizer and model...")
    tokenizer = make_tokenizer()

    model = CoconutGPT2(tokenizer, device=DEVICE)
    model = model.to(DEVICE)

    # Collect from each available stage
    stages = [
        ("stage1", "stage1_last_latent", 1),
        ("stage2", "stage2_last2_latent", 2),
        ("stage3", "stage3_all_latent", "all"),
    ]

    for stage_tag, ckpt_name, num_latent in stages:
        ckpt_path = os.path.join(CKPT_DIR, f"{ckpt_name}.pt")
        if os.path.exists(ckpt_path):
            collect_for_stage(model, tokenizer, stage_tag, ckpt_path, num_latent)

    # Create symlinks to the primary stage (stage1) for convenience
    # The AO will primarily train on stage1 activations
    for split in ["train", "val", "test", "ood"]:
        src = os.path.join(DATA_DIR, f"activations_stage1_{split}.pt")
        dst = os.path.join(DATA_DIR, f"activations_{split}.pt")
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.basename(src), dst)
            print(f"Linked {dst} -> {src}")


if __name__ == "__main__":
    main()
