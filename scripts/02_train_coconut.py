#!/usr/bin/env python3
"""Train Coconut GPT-2 with multi-stage curriculum.

Stage 0: Full text CoT (standard LM training)
Stage 1+: Progressively replace CoT steps with latent thoughts (right-to-left)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import json
import time

from src.data_gen import make_tokenizer, BOT, SEP, EOT
from src.coconut_model import CoconutGPT2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data"
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
GRAD_ACCUM = 2
MAX_SEQ_LEN = 256
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01
NORM_REG = 0.001  # L2 regularization on latent hidden state norms

STAGE_CONFIGS = [
    {"name": "stage0_text_cot", "epochs": 3, "lr": 5e-5, "num_latent": 0},
    {"name": "stage1_last_latent", "epochs": 2, "lr": 1e-5, "num_latent": 1},
    {"name": "stage2_last2_latent", "epochs": 2, "lr": 1e-5, "num_latent": 2},
    {"name": "stage3_all_latent", "epochs": 2, "lr": 1e-5, "num_latent": "all"},
]


def prepare_curriculum_batch(batch_problems, batch_cot_steps, batch_answers, tokenizer, num_latent, device):
    """Prepare a batch for a given curriculum stage.

    For num_latent=0: full text CoT (standard LM).
    For num_latent=K: last K CoT steps become latent.
    For num_latent='all': all CoT steps become latent.

    Returns:
        prefix_ids: (B, L_prefix) — problem text + remaining text CoT steps
        num_latent_actual: list of int — how many latent steps per example
        answer_ids: (B, L_answer) — <eot> + answer tokens
        full_text_ids: (B, L) — full text version (for stage 0)
        full_text_labels: (B, L) — labels for full text (for stage 0)
    """
    if num_latent == 0:
        # Stage 0: full text, standard LM training
        texts = []
        for prob, steps, ans in zip(batch_problems, batch_cot_steps, batch_answers):
            cot = f" {SEP} ".join(steps)
            full = f"{prob} {BOT} {cot} {EOT} {ans}"
            texts.append(full)

        encoded = tokenizer(texts, padding=True, truncation=True,
                          max_length=MAX_SEQ_LEN, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        # Labels: shift is handled internally by GPT-2
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {"mode": "text", "input_ids": input_ids, "labels": labels}

    # Latent stages: split into prefix (text) and latent + answer
    prefix_list = []
    answer_list = []
    latent_counts = []

    for prob, steps, ans in zip(batch_problems, batch_cot_steps, batch_answers):
        total_steps = len(steps)
        if num_latent == "all":
            n_latent = total_steps
        else:
            n_latent = min(num_latent, total_steps)

        n_text = total_steps - n_latent

        # Build prefix: problem + text CoT steps
        prefix = f"{prob} {BOT}"
        for i in range(n_text):
            prefix += f" {steps[i]} {SEP}"

        # Answer: <eot> + answer
        answer = f"{EOT} {ans}"

        prefix_list.append(prefix)
        answer_list.append(answer)
        latent_counts.append(n_latent)

    # Tokenize prefix and answer separately
    prefix_enc = tokenizer(prefix_list, padding=True, truncation=True,
                          max_length=MAX_SEQ_LEN, return_tensors="pt")
    answer_enc = tokenizer(answer_list, padding=True, truncation=True,
                          max_length=64, return_tensors="pt")

    return {
        "mode": "latent",
        "prefix_ids": prefix_enc["input_ids"].to(device),
        "prefix_mask": prefix_enc["attention_mask"].to(device),
        "answer_ids": answer_enc["input_ids"].to(device),
        "answer_mask": answer_enc["attention_mask"].to(device),
        "latent_counts": latent_counts,
    }


def train_stage(model, train_data, val_data, tokenizer, stage_config, prev_ckpt=None):
    """Train one curriculum stage."""
    name = stage_config["name"]
    epochs = stage_config["epochs"]
    lr = stage_config["lr"]
    num_latent = stage_config["num_latent"]

    print(f"\n{'='*60}")
    print(f"Training {name} | epochs={epochs} | lr={lr} | latent={num_latent}")
    print(f"{'='*60}")

    if prev_ckpt and os.path.exists(prev_ckpt):
        print(f"Loading checkpoint: {prev_ckpt}")
        model.load_state_dict(torch.load(prev_ckpt, map_location=DEVICE))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    n_train = len(train_data["problems"])
    steps_per_epoch = n_train // BATCH_SIZE
    total_steps = steps_per_epoch * epochs // GRAD_ACCUM
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    global_step = 0
    best_val_loss = float("inf")
    log_history = []

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n_train)
        epoch_loss = 0
        epoch_steps = 0

        pbar = tqdm(range(0, n_train - BATCH_SIZE + 1, BATCH_SIZE),
                    desc=f"{name} epoch {epoch+1}/{epochs}")

        for batch_start in pbar:
            idx = perm[batch_start:batch_start + BATCH_SIZE]
            batch_problems = [train_data["problems"][i] for i in idx]
            batch_cot = [train_data["cot_steps"][i] for i in idx]
            batch_answers = [train_data["answers"][i] for i in idx]

            batch = prepare_curriculum_batch(
                batch_problems, batch_cot, batch_answers,
                tokenizer, num_latent, DEVICE
            )

            if batch["mode"] == "text":
                loss, _ = model.forward_text_only(batch["input_ids"], batch["labels"])
            else:
                # Latent mode: we need to handle variable latent counts
                # For simplicity, use the max latent count in the batch
                max_latent = max(batch["latent_counts"])

                result = model.forward_coconut(
                    batch["prefix_ids"], max_latent, batch["answer_ids"]
                )

                # Answer loss
                logits = result["answer_logits"]  # (B, L_answer, V)
                targets = batch["answer_ids"][:, 1:]  # shift by 1
                logits = logits[:, :-1, :]  # align

                # Mask padding
                mask = batch["answer_mask"][:, 1:]  # shift
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=tokenizer.pad_token_id,
                    reduction="none"
                )
                loss = (loss * mask.reshape(-1)).sum() / mask.sum().clamp(min=1)

                # Norm regularization on thought hidden states
                if result["thought_norms"]:
                    norm_loss = sum(result["thought_norms"]) / len(result["thought_norms"])
                    loss = loss + NORM_REG * norm_loss

            loss = loss / GRAD_ACCUM
            loss.backward()

            if (epoch_steps + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * GRAD_ACCUM
            epoch_steps += 1
            pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Validation
        val_loss, val_acc = evaluate_model(model, val_data, tokenizer, num_latent)
        print(f"Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.4f}")
        log_history.append({
            "stage": name, "epoch": epoch + 1,
            "train_loss": avg_loss, "val_loss": val_loss, "val_acc": val_acc
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # Save checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"{name}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    return ckpt_path, log_history


@torch.no_grad()
def evaluate_model(model, data, tokenizer, num_latent, max_samples=500):
    """Evaluate model on a dataset. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    n = min(len(data["problems"]), max_samples)

    for i in range(0, n, BATCH_SIZE):
        end = min(i + BATCH_SIZE, n)
        batch_problems = data["problems"][i:end]
        batch_cot = data["cot_steps"][i:end]
        batch_answers = data["answers"][i:end]

        batch = prepare_curriculum_batch(
            batch_problems, batch_cot, batch_answers,
            tokenizer, num_latent, DEVICE
        )

        if batch["mode"] == "text":
            loss, logits = model.forward_text_only(batch["input_ids"], batch["labels"])
            total_loss += loss.item() * (end - i)
        else:
            max_latent = max(batch["latent_counts"])
            result = model.forward_coconut(
                batch["prefix_ids"], max_latent, batch["answer_ids"]
            )
            logits = result["answer_logits"]
            targets = batch["answer_ids"][:, 1:]
            logits_shifted = logits[:, :-1, :]
            mask = batch["answer_mask"][:, 1:]
            loss = F.cross_entropy(
                logits_shifted.reshape(-1, logits_shifted.size(-1)),
                targets.reshape(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction="none"
            )
            total_loss += (loss * mask.reshape(-1)).sum().item() / mask.sum().clamp(min=1).item() * (end - i)

        # Check exact match accuracy on answer (greedy decode)
        for j in range(end - i):
            prob = batch_problems[j]
            ans_gold = batch_answers[j]

            if num_latent == 0:
                # For text mode, accuracy check is trickier; skip
                continue

            # Quick greedy check: use the logits we already have
            if batch["mode"] == "latent" and result["answer_logits"] is not None:
                pred_ids = result["answer_logits"][j].argmax(dim=-1)
                # The answer starts after <eot>, get tokens until pad
                pred_tokens = []
                for tid in pred_ids:
                    t = tid.item()
                    if t == tokenizer.pad_token_id or t == tokenizer.eos_token_id:
                        break
                    pred_tokens.append(t)
                pred_text = tokenizer.decode(pred_tokens).strip()
                if pred_text == ans_gold:
                    correct += 1
                total += 1

    avg_loss = total_loss / max(n, 1)
    accuracy = correct / max(total, 1) if total > 0 else 0.0
    model.train()
    return avg_loss, accuracy


def main():
    print("Loading tokenizer and data...")
    tokenizer = make_tokenizer()
    train_data = torch.load(os.path.join(DATA_DIR, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(DATA_DIR, "val.pt"), weights_only=False)

    print("Initializing model...")
    model = CoconutGPT2(tokenizer, device=DEVICE)
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    all_logs = []
    prev_ckpt = None

    for stage_config in STAGE_CONFIGS:
        ckpt_path, logs = train_stage(
            model, train_data, val_data, tokenizer, stage_config, prev_ckpt
        )
        prev_ckpt = ckpt_path
        all_logs.extend(logs)

    # Save training log
    with open(os.path.join(CKPT_DIR, "training_log.json"), "w") as f:
        json.dump(all_logs, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete! Final checkpoint:", prev_ckpt)
    print("=" * 60)


if __name__ == "__main__":
    main()
