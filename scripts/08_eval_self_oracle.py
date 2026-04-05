#!/usr/bin/env python3
"""Evaluate the self-oracle: can the Coconut model interpret its own latent thoughts?"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import torch
import json
from collections import defaultdict
from tqdm import tqdm

from src.data_gen import make_tokenizer, ACT
from src.self_oracle import SelfOracle

DEVICE = "cuda"
DATA_DIR = "data"
CKPT_DIR = "checkpoints"
RESULTS_DIR = "results"

SOURCE_LAYERS = [3, 6, 9]

COT_QUESTION = "What is the intermediate calculation at this reasoning step?"
ANSWER_QUESTION = "What is the final answer to this math problem?"
CONTEXT_QUESTION = "What problem is being solved?"


def compute_token_f1(pred, gold):
    pred_set = set(pred.split())
    gold_set = set(gold.split())
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    common = pred_set & gold_set
    p = len(common) / len(pred_set)
    r = len(common) / len(gold_set)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_self_oracle(model, act_data, tokenizer, desc, max_n=2000):
    """Full evaluation of self-oracle on activation data."""
    model.eval()
    n = min(len(act_data), max_n)

    cot_exact = 0
    cot_f1_sum = 0
    cot_total = 0
    ans_exact = 0
    ans_total = 0
    ctx_f1_sum = 0
    ctx_total = 0
    by_layer = {l: {"exact": 0, "total": 0} for l in SOURCE_LAYERS}

    for i in tqdm(range(n), desc=desc):
        item = act_data[i]
        cot_steps = item.get("latent_cot_steps", item["cot_steps"])
        num_latent = item.get("num_latent", item["num_steps"])
        answer = item["answer"]
        problem = item["problem"]
        layer_hiddens = item["layer_hiddens"]

        if not layer_hiddens or num_latent == 0:
            continue

        src_layer = random.choice(SOURCE_LAYERS)

        # 1. CoT recovery
        step_idx = random.randint(0, min(num_latent, len(layer_hiddens), len(cot_steps)) - 1)
        if src_layer in layer_hiddens[step_idx]:
            vec = layer_hiddens[step_idx][src_layer]
            prompt = f"Layer {src_layer}: {ACT} {COT_QUESTION}"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            act_pos = [j for j, t in enumerate(prompt_ids[0].tolist()) if t == model.act_id]

            model.set_injection([vec], act_pos)
            gen = model.generate(prompt_ids, max_new_tokens=40)
            model.clear_injection()

            pred = tokenizer.decode(gen[0, prompt_ids.size(1):], skip_special_tokens=True).strip()
            gold = cot_steps[step_idx]

            # Trim prediction to first line / sentence
            for stop in ["\n", ".", "Layer", "<"]:
                if stop in pred:
                    pred = pred[:pred.index(stop)].strip()

            cot_total += 1
            if pred == gold:
                cot_exact += 1
            cot_f1_sum += compute_token_f1(pred, gold)

            by_layer[src_layer]["total"] += 1
            if pred == gold:
                by_layer[src_layer]["exact"] += 1

        # 2. Answer prediction
        vecs = [layer_hiddens[s][src_layer]
                for s in range(min(num_latent, len(layer_hiddens)))
                if src_layer in layer_hiddens[s]]
        if vecs:
            act_str = " ".join([ACT] * len(vecs))
            prompt = f"Layer {src_layer}: {act_str} {ANSWER_QUESTION}"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            act_pos = [j for j, t in enumerate(prompt_ids[0].tolist()) if t == model.act_id]

            model.set_injection(vecs[:len(act_pos)], act_pos)
            gen = model.generate(prompt_ids, max_new_tokens=20)
            model.clear_injection()

            pred = tokenizer.decode(gen[0, prompt_ids.size(1):], skip_special_tokens=True).strip()
            # Extract number
            num = ""
            for c in pred:
                if c.isdigit():
                    num += c
                elif num:
                    break

            ans_total += 1
            if num == answer:
                ans_exact += 1

        # 3. Context prediction
        if vecs:
            prompt = f"Layer {src_layer}: {act_str} {CONTEXT_QUESTION}"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            act_pos = [j for j, t in enumerate(prompt_ids[0].tolist()) if t == model.act_id]

            model.set_injection(vecs[:len(act_pos)], act_pos)
            gen = model.generate(prompt_ids, max_new_tokens=30)
            model.clear_injection()

            pred = tokenizer.decode(gen[0, prompt_ids.size(1):], skip_special_tokens=True).strip()
            ctx_total += 1
            ctx_f1_sum += compute_token_f1(pred, problem)

    results = {
        "cot_exact_match": cot_exact / max(cot_total, 1),
        "cot_token_f1": cot_f1_sum / max(cot_total, 1),
        "answer_exact_match": ans_exact / max(ans_total, 1),
        "context_f1": ctx_f1_sum / max(ctx_total, 1),
        "by_layer": {str(l): d["exact"] / max(d["total"], 1) for l, d in by_layer.items()},
    }
    return results


def random_baseline(model, act_data, tokenizer, max_n=500):
    """Inject random activations to verify the model reads real signal."""
    model.eval()
    exact = 0
    total = 0

    for i in tqdm(range(min(len(act_data), max_n)), desc="Random baseline"):
        item = act_data[i]
        cot_steps = item.get("latent_cot_steps", item["cot_steps"])
        layer_hiddens = item["layer_hiddens"]
        if not layer_hiddens or not cot_steps:
            continue

        real_vec = layer_hiddens[0].get(6)
        if real_vec is None:
            continue
        rand_vec = torch.randn_like(real_vec) * real_vec.norm()

        prompt = f"Layer 6: {ACT} {COT_QUESTION}"
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        act_pos = [j for j, t in enumerate(prompt_ids[0].tolist()) if t == model.act_id]

        model.set_injection([rand_vec], act_pos)
        gen = model.generate(prompt_ids, max_new_tokens=40)
        model.clear_injection()

        pred = tokenizer.decode(gen[0, prompt_ids.size(1):], skip_special_tokens=True).strip()
        for stop in ["\n", ".", "Layer", "<"]:
            if stop in pred:
                pred = pred[:pred.index(stop)].strip()

        gold = cot_steps[0]
        total += 1
        if pred == gold:
            exact += 1

    return exact / max(total, 1)


def qualitative(model, act_data, tokenizer, n=20):
    """Show examples of self-oracle interpreting its own latent thoughts."""
    model.eval()
    indices = random.sample(range(len(act_data)), min(n, len(act_data)))

    for i in indices:
        item = act_data[i]
        cot_steps = item.get("latent_cot_steps", item["cot_steps"])
        num_latent = item.get("num_latent", item["num_steps"])
        layer_hiddens = item["layer_hiddens"]
        if not layer_hiddens or num_latent == 0 or not cot_steps:
            continue

        print(f"\nProblem: {item['problem']}")
        print(f"Answer: {item['answer']} (predicted: {item.get('pred_answer', '?')})")

        for step_idx in range(min(num_latent, len(layer_hiddens), len(cot_steps))):
            vec = layer_hiddens[step_idx].get(6)
            if vec is None:
                continue

            prompt = f"Layer 6: {ACT} {COT_QUESTION}"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            act_pos = [j for j, t in enumerate(prompt_ids[0].tolist()) if t == model.act_id]

            model.set_injection([vec], act_pos)
            gen = model.generate(prompt_ids, max_new_tokens=40)
            model.clear_injection()

            pred = tokenizer.decode(gen[0, prompt_ids.size(1):], skip_special_tokens=True).strip()
            for stop in ["\n", ".", "Layer", "<"]:
                if stop in pred:
                    pred = pred[:pred.index(stop)].strip()

            gold = cot_steps[step_idx]
            match = "MATCH" if pred == gold else ""
            print(f"  Step {step_idx+1}: gold=[{gold}]  pred=[{pred}]  {match}")


def main():
    random.seed(42)
    torch.manual_seed(42)

    tokenizer = make_tokenizer()

    print("Loading self-oracle...")
    model = SelfOracle(tokenizer, device=DEVICE)

    ckpt = os.path.join(CKPT_DIR, "self_oracle_best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(CKPT_DIR, "self_oracle_final.pt")
    print(f"Loading: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    # Load test activations
    test_acts = torch.load(os.path.join(DATA_DIR, "activations_test.pt"), weights_only=False)

    print("\n=== Self-Oracle Evaluation ===")
    results = evaluate_self_oracle(model, test_acts, tokenizer, "test", max_n=1000)

    print(f"\nCoT exact match:    {results['cot_exact_match']:.4f}")
    print(f"CoT token F1:       {results['cot_token_f1']:.4f}")
    print(f"Answer exact match: {results['answer_exact_match']:.4f}")
    print(f"Context F1:         {results['context_f1']:.4f}")
    print(f"By layer: {results['by_layer']}")

    print("\n=== Random Baseline ===")
    rand_acc = random_baseline(model, test_acts, tokenizer, max_n=300)
    print(f"Random CoT exact match: {rand_acc:.4f}")
    results["random_baseline"] = rand_acc

    print("\n=== Qualitative Examples ===")
    qualitative(model, test_acts, tokenizer, n=20)

    # Load previous results for comparison
    prev_path = os.path.join(RESULTS_DIR, "full_results.json")
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            prev = json.load(f)
        probe_results = prev.get("probes", {})
    else:
        probe_results = {}

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON: Self-Oracle vs Previous AO vs Linear Probes")
    print("=" * 70)
    print(f"{'Metric':<35} {'Self-Oracle':>12} {'Prev AO':>12} {'Probes':>12}")
    print("-" * 75)
    print(f"{'CoT exact match':<35} {results['cot_exact_match']:>12.4f} {'0.0000':>12} {'1.0000*':>12}")
    print(f"{'CoT token F1':<35} {results['cot_token_f1']:>12.4f} {'0.2636':>12} {'N/A':>12}")
    print(f"{'Answer exact match':<35} {results['answer_exact_match']:>12.4f} {'0.0000':>12} {probe_results.get('answer_digit_exact_match', 'N/A'):>12}")
    print(f"{'Random baseline exact':<35} {rand_acc:>12.4f} {'0.0000':>12} {'N/A':>12}")
    print(f"\n* Probes measure first-token accuracy, not full CoT text")

    # Save
    results_path = os.path.join(RESULTS_DIR, "self_oracle_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {results_path}")


if __name__ == "__main__":
    main()
