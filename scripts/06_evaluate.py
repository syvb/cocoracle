#!/usr/bin/env python3
"""Full evaluation suite for Coconut + Activation Oracle experiment."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import torch
import json
from collections import defaultdict
from tqdm import tqdm

from src.data_gen import make_tokenizer, BOT, ACT
from src.coconut_model import CoconutGPT2
from src.activation_oracle import ActivationOracle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data"
CKPT_DIR = "checkpoints"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SOURCE_LAYERS = [3, 6, 9]

# Question templates
COT_QUESTION = "What is the intermediate calculation at this reasoning step?"
COT_MULTI_QUESTION = "Describe all reasoning steps."
ANSWER_QUESTION = "What is the final answer to this math problem?"
CONTEXT_QUESTION = "What problem is being solved?"


def compute_token_f1(pred_tokens, gold_tokens):
    """Compute token-level F1 between predicted and gold token lists."""
    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    common = pred_set & gold_set
    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def simple_bleu(pred, gold, max_n=4):
    """Simple BLEU score (unigram to max_n-gram)."""
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    brevity = min(1.0, len(pred_tokens) / len(gold_tokens)) if gold_tokens else 0.0
    score = 0.0
    for n in range(1, min(max_n + 1, len(pred_tokens) + 1, len(gold_tokens) + 1)):
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
        gold_ngrams = [tuple(gold_tokens[i:i+n]) for i in range(len(gold_tokens) - n + 1)]
        gold_set = set(gold_ngrams)
        matches = sum(1 for ng in pred_ngrams if ng in gold_set)
        if len(pred_ngrams) > 0:
            score += matches / len(pred_ngrams)

    return brevity * score / max_n


def evaluate_coconut_accuracy(model, data, tokenizer, split_name, num_latent_mode=1, max_samples=1000):
    """Evaluate Coconut model accuracy on a dataset."""
    print(f"\n--- Coconut Accuracy ({split_name}, latent={num_latent_mode}) ---")
    model.eval()
    n = min(len(data["problems"]), max_samples)

    correct = 0
    total = 0
    by_num_steps = defaultdict(lambda: {"correct": 0, "total": 0})

    from src.data_gen import SEP
    for i in tqdm(range(n), desc=f"Coconut {split_name}"):
        problem = data["problems"][i]
        answer = data["answers"][i]
        cot_steps = data["cot_steps"][i]
        num_steps = len(cot_steps)

        if num_latent_mode == "all":
            n_latent = num_steps
        else:
            n_latent = min(num_latent_mode, num_steps)
        n_text = num_steps - n_latent

        prefix = f"{problem} {BOT}"
        for j in range(n_text):
            prefix += f" {cot_steps[j]} {SEP}"
        prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(DEVICE)

        generated, _, _ = model.generate_answer(prefix_ids, n_latent, max_new_tokens=20)
        pred = tokenizer.decode(generated).strip()

        is_correct = pred == answer
        correct += int(is_correct)
        total += 1
        by_num_steps[num_steps]["correct"] += int(is_correct)
        by_num_steps[num_steps]["total"] += 1

    acc = correct / max(total, 1)
    print(f"Overall: {correct}/{total} = {acc:.4f}")

    for ns in sorted(by_num_steps.keys()):
        d = by_num_steps[ns]
        a = d["correct"] / max(d["total"], 1)
        print(f"  {ns} steps: {d['correct']}/{d['total']} = {a:.4f}")

    return {"overall": acc, "by_steps": {k: v["correct"]/max(v["total"],1) for k,v in by_num_steps.items()}}


def evaluate_ao(oracle, activation_data, tokenizer, split_name, max_samples=2000):
    """Evaluate Activation Oracle on collected activations."""
    print(f"\n--- AO Evaluation ({split_name}) ---")
    oracle.eval()
    n = min(len(activation_data), max_samples)

    results = {
        "cot_recovery": {"exact": 0, "token_f1": 0, "bleu": 0, "total": 0},
        "answer_pred": {"exact": 0, "total": 0},
        "context_pred": {"bleu": 0, "total": 0},
        "by_layer": {l: {"exact": 0, "total": 0} for l in SOURCE_LAYERS},
        "by_step": defaultdict(lambda: {"exact": 0, "total": 0}),
    }

    for i in tqdm(range(n), desc=f"AO {split_name}"):
        item = activation_data[i]
        cot_steps = item.get("latent_cot_steps", item["cot_steps"])
        num_steps = item.get("num_latent", item["num_steps"])
        answer = item["answer"]
        problem = item["problem"]
        layer_hiddens = item["layer_hiddens"]

        if not layer_hiddens or num_steps == 0:
            continue

        # Pick a random source layer for this eval
        src_layer = random.choice(SOURCE_LAYERS)

        # 1. Single-step CoT recovery (random step)
        step_idx = random.randint(0, num_steps - 1)
        if src_layer in layer_hiddens[step_idx]:
            vec = layer_hiddens[step_idx][src_layer]
            prompt = f"Layer {src_layer}: {ACT} {COT_QUESTION}"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            act_pos = [j for j, t in enumerate(prompt_ids[0].tolist()) if t == oracle.act_id]

            oracle.set_injection([vec], act_pos)
            gen = oracle.generate(prompt_ids, max_new_tokens=50)
            oracle.clear_injection()

            pred = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
            gold = cot_steps[step_idx]

            results["cot_recovery"]["total"] += 1
            if pred == gold:
                results["cot_recovery"]["exact"] += 1

            gold_tokens = gold.split()
            pred_tokens = pred.split()
            results["cot_recovery"]["token_f1"] += compute_token_f1(pred_tokens, gold_tokens)
            results["cot_recovery"]["bleu"] += simple_bleu(pred, gold)

            results["by_layer"][src_layer]["total"] += 1
            if pred == gold:
                results["by_layer"][src_layer]["exact"] += 1

            results["by_step"][step_idx]["total"] += 1
            if pred == gold:
                results["by_step"][step_idx]["exact"] += 1

        # 2. Answer prediction
        if src_layer in layer_hiddens[-1]:
            vecs = [layer_hiddens[s][src_layer] for s in range(num_steps) if src_layer in layer_hiddens[s]]
            act_tokens = " ".join([ACT] * len(vecs))
            prompt = f"Layer {src_layer}: {act_tokens} {ANSWER_QUESTION}"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            act_pos = [j for j, t in enumerate(prompt_ids[0].tolist()) if t == oracle.act_id]

            oracle.set_injection(vecs[:len(act_pos)], act_pos)
            gen = oracle.generate(prompt_ids, max_new_tokens=20)
            oracle.clear_injection()

            pred = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
            results["answer_pred"]["total"] += 1
            if pred == answer:
                results["answer_pred"]["exact"] += 1

        # 3. Context (problem) prediction
        if src_layer in layer_hiddens[-1]:
            vecs = [layer_hiddens[s][src_layer] for s in range(num_steps) if src_layer in layer_hiddens[s]]
            act_tokens = " ".join([ACT] * len(vecs))
            prompt = f"Layer {src_layer}: {act_tokens} {CONTEXT_QUESTION}"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            act_pos = [j for j, t in enumerate(prompt_ids[0].tolist()) if t == oracle.act_id]

            oracle.set_injection(vecs[:len(act_pos)], act_pos)
            gen = oracle.generate(prompt_ids, max_new_tokens=30)
            oracle.clear_injection()

            pred = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
            results["context_pred"]["total"] += 1
            results["context_pred"]["bleu"] += simple_bleu(pred, problem)

    # Compute averages
    summary = {}
    cr = results["cot_recovery"]
    if cr["total"] > 0:
        summary["cot_exact_match"] = cr["exact"] / cr["total"]
        summary["cot_token_f1"] = cr["token_f1"] / cr["total"]
        summary["cot_bleu"] = cr["bleu"] / cr["total"]
    else:
        summary["cot_exact_match"] = summary["cot_token_f1"] = summary["cot_bleu"] = 0

    ap = results["answer_pred"]
    summary["answer_exact_match"] = ap["exact"] / max(ap["total"], 1)

    cp = results["context_pred"]
    summary["context_bleu"] = cp["bleu"] / max(cp["total"], 1)

    summary["by_layer"] = {}
    for l in SOURCE_LAYERS:
        d = results["by_layer"][l]
        summary["by_layer"][str(l)] = d["exact"] / max(d["total"], 1)

    summary["by_step"] = {}
    for s in sorted(results["by_step"].keys()):
        d = results["by_step"][s]
        summary["by_step"][str(s)] = d["exact"] / max(d["total"], 1)

    print(f"\nCoT exact match: {summary['cot_exact_match']:.4f}")
    print(f"CoT token F1:    {summary['cot_token_f1']:.4f}")
    print(f"CoT BLEU:        {summary['cot_bleu']:.4f}")
    print(f"Answer exact:    {summary['answer_exact_match']:.4f}")
    print(f"Context BLEU:    {summary['context_bleu']:.4f}")
    print(f"By layer: {summary['by_layer']}")
    print(f"By step:  {summary['by_step']}")

    return summary


def evaluate_random_baseline(oracle, activation_data, tokenizer, max_samples=500):
    """Ablation: inject random activations to verify AO reads real signal."""
    print("\n--- Random Activation Baseline ---")
    oracle.eval()
    n = min(len(activation_data), max_samples)
    correct = 0
    total = 0

    for i in tqdm(range(n), desc="Random baseline"):
        item = activation_data[i]
        latent_cot = item.get("latent_cot_steps", item["cot_steps"])
        num_lat = item.get("num_latent", item["num_steps"])
        if not item["layer_hiddens"] or num_lat == 0:
            continue

        step_idx = 0
        gold = latent_cot[step_idx] if step_idx < len(latent_cot) else ""
        src_layer = 6

        # Random vector with same shape and similar norm
        real_vec = item["layer_hiddens"][step_idx].get(src_layer)
        if real_vec is None:
            continue
        rand_vec = torch.randn_like(real_vec) * real_vec.norm()

        prompt = f"Layer {src_layer}: {ACT} {COT_QUESTION}"
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        act_pos = [j for j, t in enumerate(prompt_ids[0].tolist()) if t == oracle.act_id]

        oracle.set_injection([rand_vec], act_pos)
        gen = oracle.generate(prompt_ids, max_new_tokens=50)
        oracle.clear_injection()

        pred = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
        total += 1
        if pred == gold:
            correct += 1

    acc = correct / max(total, 1)
    print(f"Random baseline CoT exact match: {acc:.4f} ({correct}/{total})")
    return acc


def qualitative_examples(coconut, oracle, data, tokenizer, n=20):
    """Generate qualitative examples showing the full pipeline."""
    print(f"\n--- Qualitative Examples ---")
    coconut.eval()
    oracle.eval()
    examples = []

    indices = random.sample(range(len(data["problems"])), min(n, len(data["problems"])))

    from src.data_gen import SEP
    for i in indices:
        problem = data["problems"][i]
        answer = data["answers"][i]
        cot_steps = data["cot_steps"][i]
        num_steps = len(cot_steps)
        n_latent = 1  # Stage 1
        n_text = num_steps - n_latent

        # Run Coconut with Stage 1 setup
        prefix = f"{problem} {BOT}"
        for j in range(n_text):
            prefix += f" {cot_steps[j]} {SEP}"
        prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(DEVICE)
        generated, thought_hiddens, layer_hiddens = coconut.generate_answer(
            prefix_ids, n_latent, max_new_tokens=20
        )
        pred_answer = tokenizer.decode(generated).strip()

        # Ask AO about each latent thought step
        latent_cot = cot_steps[n_text:]  # ground truth for latent steps
        ao_interpretations = []
        for step_idx in range(n_latent):
            if not layer_hiddens or step_idx >= len(layer_hiddens):
                ao_interpretations.append("[no hidden state]")
                continue

            src_layer = 6
            if src_layer not in layer_hiddens[step_idx]:
                ao_interpretations.append("[layer not available]")
                continue

            vec = layer_hiddens[step_idx][src_layer]
            prompt = f"Layer {src_layer}: {ACT} {COT_QUESTION}"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            act_pos = [j for j, t in enumerate(prompt_ids[0].tolist()) if t == oracle.act_id]

            oracle.set_injection([vec], act_pos)
            gen = oracle.generate(prompt_ids, max_new_tokens=50)
            oracle.clear_injection()
            ao_interpretations.append(tokenizer.decode(gen[0], skip_special_tokens=True).strip())

        ex = {
            "problem": problem,
            "gold_answer": answer,
            "pred_answer": pred_answer,
            "correct": pred_answer == answer,
            "text_cot": cot_steps[:n_text],
            "latent_cot_gold": latent_cot,
            "ao_interpretations": ao_interpretations,
        }
        examples.append(ex)

        print(f"\nProblem: {problem}")
        print(f"Gold answer: {answer} | Predicted: {pred_answer} {'✓' if ex['correct'] else '✗'}")
        print(f"Text CoT: {cot_steps[:n_text]}")
        for s in range(n_latent):
            gold_step = latent_cot[s] if s < len(latent_cot) else "?"
            ao_step = ao_interpretations[s] if s < len(ao_interpretations) else "?"
            match = "✓" if gold_step == ao_step else "✗"
            print(f"  Latent step {s+1}: gold=[{gold_step}] ao=[{ao_step}] {match}")

    return examples


def main():
    random.seed(42)
    torch.manual_seed(42)

    print("Loading tokenizer and models...")
    tokenizer = make_tokenizer()

    # Load Coconut model — use Stage 1 (best latent stage)
    coconut = CoconutGPT2(tokenizer, device=DEVICE).to(DEVICE)
    ckpt = os.path.join(CKPT_DIR, "stage1_last_latent.pt")
    print(f"Loading Coconut: {ckpt}")
    coconut.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    # Load AO
    oracle = ActivationOracle(tokenizer, device=DEVICE).to(DEVICE)
    ao_ckpt = os.path.join(CKPT_DIR, "oracle_best.pt")
    if not os.path.exists(ao_ckpt):
        ao_ckpt = os.path.join(CKPT_DIR, "oracle_final.pt")
    if os.path.exists(ao_ckpt):
        print(f"Loading AO: {ao_ckpt}")
        ckpt_data = torch.load(ao_ckpt, map_location=DEVICE)
        oracle.load_state_dict(ckpt_data["full_state"])
    else:
        print("WARNING: No AO checkpoint found, using untrained oracle.")

    # Load data
    test_data = torch.load(os.path.join(DATA_DIR, "test.pt"), weights_only=False)
    test_acts = torch.load(os.path.join(DATA_DIR, "activations_test.pt"), weights_only=False)

    all_results = {}

    # 1. Coconut accuracy (Stage 1: 1 latent step)
    all_results["coconut_test_stage1"] = evaluate_coconut_accuracy(
        coconut, test_data, tokenizer, "test_stage1", num_latent_mode=1, max_samples=500
    )

    # OOD (5-digit, Stage 1)
    ood_data = torch.load(os.path.join(DATA_DIR, "ood.pt"), weights_only=False)
    all_results["coconut_ood_stage1"] = evaluate_coconut_accuracy(
        coconut, ood_data, tokenizer, "ood_5digit_stage1", num_latent_mode=1, max_samples=200
    )

    # 2. AO evaluation
    all_results["ao_test"] = evaluate_ao(oracle, test_acts, tokenizer, "test", max_samples=2000)

    # 3. Random baseline ablation
    all_results["random_baseline_acc"] = evaluate_random_baseline(oracle, test_acts, tokenizer, max_samples=500)

    # 4. Load probe results if available
    probe_results_path = os.path.join(CKPT_DIR, "probe_results.json")
    if os.path.exists(probe_results_path):
        with open(probe_results_path) as f:
            all_results["probes"] = json.load(f)
    else:
        all_results["probes"] = "not yet trained"

    # 5. Qualitative examples
    examples = qualitative_examples(coconut, oracle, test_data, tokenizer, n=20)
    all_results["qualitative_examples"] = examples

    # Save results
    results_path = os.path.join(RESULTS_DIR, "full_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<40} {'Value':>10}")
    print("-" * 55)

    ctest = all_results.get("coconut_test_stage1", {})
    print(f"{'Coconut test accuracy (Stage 1)':<40} {ctest.get('overall', 0):>10.4f}")
    for ns, acc in sorted(ctest.get("by_steps", {}).items()):
        print(f"{'  ' + str(ns) + '-step problems':<40} {acc:>10.4f}")

    cood = all_results.get("coconut_ood_stage1", {})
    print(f"{'Coconut OOD (5-digit, Stage 1)':<40} {cood.get('overall', 0):>10.4f}")

    if isinstance(all_results["ao_test"], dict):
        ao = all_results["ao_test"]
        print(f"\n{'AO CoT exact match':<40} {ao.get('cot_exact_match', 0):>10.4f}")
        print(f"{'AO CoT token F1':<40} {ao.get('cot_token_f1', 0):>10.4f}")
        print(f"{'AO CoT BLEU':<40} {ao.get('cot_bleu', 0):>10.4f}")
        print(f"{'AO answer exact match':<40} {ao.get('answer_exact_match', 0):>10.4f}")
        print(f"{'AO context BLEU':<40} {ao.get('context_bleu', 0):>10.4f}")

        print(f"\n{'Random baseline CoT exact match':<40} {all_results.get('random_baseline_acc', 0):>10.4f}")

        print(f"\n{'AO by source layer:':<40}")
        for l, acc in ao.get("by_layer", {}).items():
            print(f"{'  Layer ' + str(l):<40} {acc:>10.4f}")

    if isinstance(all_results.get("probes"), dict):
        print(f"\n{'Probe Results:':<40}")
        for k, v in all_results["probes"].items():
            print(f"{'  ' + k:<40} {v:>10.4f}")

    print("\n" + "=" * 70)
    print(f"Full results saved to: {results_path}")


if __name__ == "__main__":
    main()
