#!/usr/bin/env python3
"""Train linear probe baselines on the same hidden states."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json

from src.data_gen import make_tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data"
CKPT_DIR = "checkpoints"

# Probe hyperparameters
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 256
D_MODEL = 768
SOURCE_LAYER = 6  # Use 50% depth layer


class LinearProbe(nn.Module):
    """Simple linear probe."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    """Two-layer MLP probe."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def prepare_probe_data(activation_data, tokenizer):
    """Prepare data for probe training.

    Returns:
        step_data: list of (hidden_vec, first_token_id, step_text, step_idx) per thought step
        answer_data: list of (final_hidden_vec, answer_text) per problem
    """
    step_data = []
    answer_data = []

    for item in activation_data:
        cot_steps = item["cot_steps"]
        num_steps = item["num_steps"]
        answer = item["answer"]
        layer_hiddens = item["layer_hiddens"]

        if not layer_hiddens or num_steps == 0:
            continue

        # Per-step data
        for step_idx in range(num_steps):
            if SOURCE_LAYER not in layer_hiddens[step_idx]:
                continue
            vec = layer_hiddens[step_idx][SOURCE_LAYER]
            step_text = cot_steps[step_idx]
            first_token = tokenizer.encode(step_text, add_special_tokens=False)[0]
            step_data.append((vec, first_token, step_text, step_idx))

        # Answer data: use final thought hidden state
        if SOURCE_LAYER in layer_hiddens[-1]:
            answer_data.append((layer_hiddens[-1][SOURCE_LAYER], answer))

    return step_data, answer_data


def train_first_token_probe(step_data, vocab_size):
    """Train probe to predict first token of CoT step from hidden state."""
    print("\n--- First Token Probe ---")
    vecs = torch.stack([d[0] for d in step_data]).to(DEVICE)
    targets = torch.tensor([d[1] for d in step_data], dtype=torch.long).to(DEVICE)

    n = len(step_data)
    split = int(0.9 * n)
    train_vecs, val_vecs = vecs[:split], vecs[split:]
    train_targets, val_targets = targets[:split], targets[split:]

    probe = LinearProbe(D_MODEL, vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)

    best_val_acc = 0
    for epoch in range(EPOCHS):
        probe.train()
        perm = torch.randperm(len(train_vecs))
        epoch_loss = 0
        for i in range(0, len(train_vecs) - BATCH_SIZE + 1, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            logits = probe(train_vecs[idx])
            loss = F.cross_entropy(logits, train_targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validate
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_vecs)
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == val_targets).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(probe.state_dict(), os.path.join(CKPT_DIR, "probe_first_token.pt"))

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: val_acc={val_acc:.4f} (best={best_val_acc:.4f})")

    print(f"  Best first-token probe accuracy: {best_val_acc:.4f}")
    return best_val_acc


def train_step_identity_probe(step_data, max_steps=8):
    """Train probe to classify which reasoning step a hidden state came from."""
    print("\n--- Step Identity Probe ---")
    vecs = torch.stack([d[0] for d in step_data]).to(DEVICE)
    targets = torch.tensor([min(d[3], max_steps - 1) for d in step_data], dtype=torch.long).to(DEVICE)

    n = len(step_data)
    split = int(0.9 * n)
    train_vecs, val_vecs = vecs[:split], vecs[split:]
    train_targets, val_targets = targets[:split], targets[split:]

    probe = LinearProbe(D_MODEL, max_steps).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)

    best_val_acc = 0
    for epoch in range(EPOCHS):
        probe.train()
        perm = torch.randperm(len(train_vecs))
        for i in range(0, len(train_vecs) - BATCH_SIZE + 1, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            logits = probe(train_vecs[idx])
            loss = F.cross_entropy(logits, train_targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_preds = probe(val_vecs).argmax(dim=-1)
            val_acc = (val_preds == val_targets).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(probe.state_dict(), os.path.join(CKPT_DIR, "probe_step_id.pt"))

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: val_acc={val_acc:.4f} (best={best_val_acc:.4f})")

    print(f"  Best step-identity probe accuracy: {best_val_acc:.4f}")
    return best_val_acc


def train_answer_digit_probe(answer_data):
    """Train probe to predict answer digits from final thought hidden state."""
    print("\n--- Answer Digit Probe ---")
    # Predict first 4 digits of the answer
    max_digits = 4
    vecs = []
    digit_targets = []

    for vec, ans in answer_data:
        digits = [int(d) for d in ans[:max_digits]]
        # Pad with 10 (=no digit) if answer is shorter
        while len(digits) < max_digits:
            digits.append(10)
        vecs.append(vec)
        digit_targets.append(digits)

    vecs = torch.stack(vecs).to(DEVICE)
    digit_targets = torch.tensor(digit_targets, dtype=torch.long).to(DEVICE)  # (N, max_digits)

    n = len(vecs)
    split = int(0.9 * n)
    train_vecs, val_vecs = vecs[:split], vecs[split:]
    train_targets, val_targets = digit_targets[:split], digit_targets[split:]

    # One probe per digit position
    probes = [LinearProbe(D_MODEL, 11).to(DEVICE) for _ in range(max_digits)]  # 11 = 0-9 + no_digit
    optimizers = [torch.optim.Adam(p.parameters(), lr=LR) for p in probes]

    best_val_acc = 0
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(train_vecs))
        for p in probes:
            p.train()

        for i in range(0, len(train_vecs) - BATCH_SIZE + 1, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            batch_vecs = train_vecs[idx]

            for d in range(max_digits):
                logits = probes[d](batch_vecs)
                loss = F.cross_entropy(logits, train_targets[idx, d])
                optimizers[d].zero_grad()
                loss.backward()
                optimizers[d].step()

        # Validate: full-answer exact match
        for p in probes:
            p.eval()
        with torch.no_grad():
            all_preds = torch.stack([p(val_vecs).argmax(dim=-1) for p in probes], dim=1)  # (N, max_digits)
            exact_match = (all_preds == val_targets).all(dim=1).float().mean().item()

        if exact_match > best_val_acc:
            best_val_acc = exact_match
            for d, p in enumerate(probes):
                torch.save(p.state_dict(), os.path.join(CKPT_DIR, f"probe_digit_{d}.pt"))

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: exact_match={exact_match:.4f} (best={best_val_acc:.4f})")

    print(f"  Best answer-digit probe exact match: {best_val_acc:.4f}")
    return best_val_acc


def main():
    print("Loading data...")
    tokenizer = make_tokenizer()
    train_acts = torch.load(os.path.join(DATA_DIR, "activations_train.pt"), weights_only=False)

    # Use a subset for probes
    if len(train_acts) > 30000:
        train_acts = train_acts[:30000]

    print("Preparing probe data...")
    step_data, answer_data = prepare_probe_data(train_acts, tokenizer)
    print(f"Step data: {len(step_data)}, Answer data: {len(answer_data)}")

    results = {}
    results["first_token_acc"] = train_first_token_probe(step_data, len(tokenizer))
    results["step_identity_acc"] = train_step_identity_probe(step_data)
    results["answer_digit_exact_match"] = train_answer_digit_probe(answer_data)

    with open(os.path.join(CKPT_DIR, "probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Probe Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
