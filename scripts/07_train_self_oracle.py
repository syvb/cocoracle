#!/usr/bin/env python3
"""Train the Coconut model as its own activation oracle.

Mixes two training objectives:
1. AO tasks: interpret injected activations (CoT recovery, answer prediction)
2. Original task: standard text CoT to prevent catastrophic forgetting
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import json

from src.data_gen import make_tokenizer, ACT, BOT, SEP, EOT
from src.self_oracle import SelfOracle

DEVICE = "cuda"
DATA_DIR = "data"
CKPT_DIR = "checkpoints"

# Hyperparameters
LR = 5e-6  # Lower LR since we're full fine-tuning a pretrained model
EPOCHS = 5
GRAD_ACCUM = 16  # Effective batch = 16
MAX_SEQ_LEN = 192
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01
AO_MIX_RATIO = 0.7  # 70% AO tasks, 30% original Coconut task

SOURCE_LAYERS = [3, 6, 9]

# Question templates
COT_QUESTIONS = [
    "What is the intermediate calculation at this reasoning step?",
    "What calculation is being performed here?",
    "Describe the arithmetic operation at this step.",
    "What is the model computing at this point?",
    "What intermediate result is encoded here?",
]

ANSWER_QUESTIONS = [
    "What is the final answer to this math problem?",
    "What is the result of this calculation?",
    "What number does this computation produce?",
]

CONTEXT_QUESTIONS = [
    "What problem is being solved?",
    "What was the input to this computation?",
]


def make_ao_examples(activation_data):
    """Create AO training examples from collected activations."""
    examples = []

    for item in activation_data:
        cot_steps = item.get("latent_cot_steps", item["cot_steps"])
        num_latent = item.get("num_latent", item["num_steps"])
        problem = item["problem"]
        answer = item["answer"]
        layer_hiddens = item["layer_hiddens"]

        if not layer_hiddens or num_latent == 0:
            continue

        for source_layer in SOURCE_LAYERS:
            # Single-step CoT recovery
            for step_idx in range(min(num_latent, len(layer_hiddens), len(cot_steps))):
                if source_layer not in layer_hiddens[step_idx]:
                    continue
                vec = layer_hiddens[step_idx][source_layer]
                q = random.choice(COT_QUESTIONS)
                examples.append({
                    "type": "cot",
                    "prompt": f"Layer {source_layer}: {ACT} {q}",
                    "target": cot_steps[step_idx],
                    "vectors": [vec],
                })

            # Answer prediction (use all latent hiddens)
            vecs = [layer_hiddens[s][source_layer]
                    for s in range(min(num_latent, len(layer_hiddens)))
                    if source_layer in layer_hiddens[s]]
            if vecs:
                act_str = " ".join([ACT] * len(vecs))
                q = random.choice(ANSWER_QUESTIONS)
                examples.append({
                    "type": "answer",
                    "prompt": f"Layer {source_layer}: {act_str} {q}",
                    "target": answer,
                    "vectors": vecs,
                })

                # Context prediction
                q = random.choice(CONTEXT_QUESTIONS)
                examples.append({
                    "type": "context",
                    "prompt": f"Layer {source_layer}: {act_str} {q}",
                    "target": problem,
                    "vectors": vecs,
                })

    return examples


def make_coconut_text_examples(data, max_n=10000):
    """Create original text-CoT examples to prevent forgetting."""
    examples = []
    n = min(len(data["problems"]), max_n)
    for i in range(n):
        prob = data["problems"][i]
        cot = data["cot_steps"][i]
        ans = data["answers"][i]
        cot_text = f" {SEP} ".join(cot)
        full = f"{prob} {BOT} {cot_text} {EOT} {ans}"
        examples.append({"type": "text_cot", "text": full})
    return examples


def train_ao_step(model, item, tokenizer):
    """Process one AO training example. Returns loss."""
    prompt = item["prompt"]
    target = item["target"]
    vecs = item["vectors"]

    full_text = prompt + " " + target
    encoded = tokenizer.encode(full_text, add_special_tokens=False)
    if len(encoded) > MAX_SEQ_LEN:
        encoded = encoded[:MAX_SEQ_LEN]

    input_ids = torch.tensor([encoded], device=model.device)

    # Find <act> positions
    act_positions = [i for i, t in enumerate(encoded) if t == model.act_id]

    # Labels: -100 for prompt, real IDs for target
    prompt_encoded = tokenizer.encode(prompt + " ", add_special_tokens=False)
    prompt_len = min(len(prompt_encoded), len(encoded))
    labels = [-100] * prompt_len + encoded[prompt_len:]
    labels = labels[:len(encoded)]
    labels = torch.tensor([labels], device=model.device)

    # Inject activations
    if act_positions and vecs:
        model.set_injection(vecs[:len(act_positions)], act_positions)

    outputs = model(input_ids, labels=labels)
    loss = outputs.loss

    model.clear_injection()
    return loss


def train_text_step(model, item, tokenizer):
    """Process one text-CoT example. Returns loss."""
    encoded = tokenizer.encode(item["text"], add_special_tokens=False)
    if len(encoded) > MAX_SEQ_LEN:
        encoded = encoded[:MAX_SEQ_LEN]

    input_ids = torch.tensor([encoded], device=model.device)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    outputs = model(input_ids, labels=labels)
    return outputs.loss


def evaluate(model, val_ao, val_text, tokenizer, max_n=300):
    """Quick eval: AO loss + text loss + generation accuracy."""
    model.eval()
    ao_loss_sum = 0
    text_loss_sum = 0
    ao_count = 0
    text_count = 0

    # AO loss
    for item in val_ao[:max_n]:
        loss = train_ao_step(model, item, tokenizer)
        if torch.isfinite(loss):
            ao_loss_sum += loss.item()
            ao_count += 1

    # Text loss
    for item in val_text[:max_n]:
        loss = train_text_step(model, item, tokenizer)
        if torch.isfinite(loss):
            text_loss_sum += loss.item()
            text_count += 1

    # Generation accuracy on a few AO examples
    cot_correct = 0
    cot_total = 0
    cot_examples = [e for e in val_ao if e["type"] == "cot"][:100]

    for item in cot_examples:
        prompt = item["prompt"]
        gold = item["target"]
        vecs = item["vectors"]

        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        act_positions = [i for i, t in enumerate(prompt_ids[0].tolist()) if t == model.act_id]

        if act_positions and vecs:
            model.set_injection(vecs[:len(act_positions)], act_positions)

        gen = model.generate(prompt_ids, max_new_tokens=40)
        model.clear_injection()

        pred = tokenizer.decode(gen[0, prompt_ids.size(1):], skip_special_tokens=True).strip()

        # Check if prediction starts with the gold text
        if pred == gold or pred.startswith(gold):
            cot_correct += 1
        cot_total += 1

    model.train()
    return {
        "ao_loss": ao_loss_sum / max(ao_count, 1),
        "text_loss": text_loss_sum / max(text_count, 1),
        "cot_gen_acc": cot_correct / max(cot_total, 1),
    }


def main():
    print("Loading tokenizer...")
    tokenizer = make_tokenizer()

    print("Loading activation data...")
    train_acts = torch.load(os.path.join(DATA_DIR, "activations_train.pt"), weights_only=False)
    val_acts = torch.load(os.path.join(DATA_DIR, "activations_val.pt"), weights_only=False)

    print("Loading original training data...")
    train_data = torch.load(os.path.join(DATA_DIR, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(DATA_DIR, "val.pt"), weights_only=False)

    print("Creating training examples...")
    train_ao = make_ao_examples(train_acts)
    val_ao = make_ao_examples(val_acts)
    train_text = make_coconut_text_examples(train_data, max_n=15000)
    val_text = make_coconut_text_examples(val_data, max_n=2000)

    random.seed(42)
    random.shuffle(train_ao)
    train_ao = train_ao[:50000]
    random.shuffle(val_ao)
    val_ao = val_ao[:5000]

    print(f"AO train: {len(train_ao)}, AO val: {len(val_ao)}")
    print(f"Text train: {len(train_text)}, Text val: {len(val_text)}")

    # Total examples per epoch: pick from AO or text based on mix ratio
    total_per_epoch = len(train_ao) + len(train_text)
    n_ao_per_epoch = int(total_per_epoch * AO_MIX_RATIO)
    n_text_per_epoch = total_per_epoch - n_ao_per_epoch
    print(f"Per epoch: {n_ao_per_epoch} AO + {n_text_per_epoch} text = {total_per_epoch}")

    print("Initializing self-oracle from Coconut checkpoint...")
    model = SelfOracle(tokenizer, device=DEVICE)

    # Load the Coconut Stage 1 checkpoint
    coconut_ckpt = os.path.join(CKPT_DIR, "stage1_last_latent.pt")
    print(f"Loading: {coconut_ckpt}")
    state = torch.load(coconut_ckpt, map_location=DEVICE)
    # Map Coconut state dict to SelfOracle (CoconutGPT2.model -> SelfOracle.model)
    mapped = {}
    for k, v in state.items():
        if k.startswith("model."):
            mapped[k] = v
    model.load_state_dict(mapped, strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = total_per_epoch * EPOCHS // GRAD_ACCUM
    warmup = int(total_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    best_ao_loss = float("inf")
    log_history = []

    for epoch in range(EPOCHS):
        model.train()

        # Build shuffled mixed training sequence
        random.shuffle(train_ao)
        random.shuffle(train_text)
        ao_sample = train_ao[:n_ao_per_epoch]
        text_sample = train_text[:n_text_per_epoch]

        # Interleave: create a shuffled list of (type, example) pairs
        mixed = [(True, ex) for ex in ao_sample] + [(False, ex) for ex in text_sample]
        random.shuffle(mixed)

        epoch_ao_loss = 0
        epoch_text_loss = 0
        ao_count = 0
        text_count = 0
        step = 0

        pbar = tqdm(mixed, desc=f"Self-Oracle epoch {epoch+1}/{EPOCHS}")

        for is_ao, item in pbar:
            if is_ao:
                loss = train_ao_step(model, item, tokenizer)
                if torch.isfinite(loss):
                    (loss / GRAD_ACCUM).backward()
                    epoch_ao_loss += loss.item()
                    ao_count += 1
            else:
                loss = train_text_step(model, item, tokenizer)
                if torch.isfinite(loss):
                    (loss / GRAD_ACCUM).backward()
                    epoch_text_loss += loss.item()
                    text_count += 1

            step += 1
            if step % GRAD_ACCUM == 0:
                # NaN guard
                has_nan = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
                if not has_nan:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    scheduler.step()
                optimizer.zero_grad()

            if step % 500 == 0:
                avg_ao = epoch_ao_loss / max(ao_count, 1)
                avg_text = epoch_text_loss / max(text_count, 1)
                pbar.set_postfix(ao=f"{avg_ao:.2f}", text=f"{avg_text:.2f}")

        avg_ao = epoch_ao_loss / max(ao_count, 1)
        avg_text = epoch_text_loss / max(text_count, 1)
        print(f"Epoch {epoch+1}: AO loss={avg_ao:.4f}, Text loss={avg_text:.4f}")

        # Validate
        with torch.no_grad():
            metrics = evaluate(model, val_ao, val_text, tokenizer)
        print(f"  Val AO loss={metrics['ao_loss']:.4f}, "
              f"Val text loss={metrics['text_loss']:.4f}, "
              f"CoT gen acc={metrics['cot_gen_acc']:.4f}")
        log_history.append({
            "epoch": epoch + 1,
            "train_ao_loss": avg_ao,
            "train_text_loss": avg_text,
            **metrics,
        })

        if metrics["ao_loss"] < best_ao_loss:
            best_ao_loss = metrics["ao_loss"]
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "self_oracle_best.pt"))
            print(f"  Saved best checkpoint")

    # Save final
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "self_oracle_final.pt"))
    with open(os.path.join(CKPT_DIR, "self_oracle_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    print("\nSelf-oracle training complete!")
    print(f"Best val AO loss: {best_ao_loss:.4f}")
    for entry in log_history:
        print(f"  Epoch {entry['epoch']}: "
              f"CoT gen acc={entry['cot_gen_acc']:.4f}, "
              f"AO loss={entry['ao_loss']:.4f}, "
              f"text loss={entry['text_loss']:.4f}")


if __name__ == "__main__":
    main()
