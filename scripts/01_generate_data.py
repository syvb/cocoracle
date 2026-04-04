#!/usr/bin/env python3
"""Generate and save arithmetic CoT datasets."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.data_gen import make_tokenizer, generate_dataset, tokenize_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

print("Loading tokenizer...")
tokenizer = make_tokenizer()
tokenizer.save_pretrained(os.path.join(DATA_DIR, "tokenizer"))

print("Generating datasets...")
train_raw = generate_dataset(100_000, seed=42)
val_raw = generate_dataset(5_000, seed=123)
test_raw = generate_dataset(5_000, seed=456)
# OOD: 5-digit addition (never seen in training)
ood_raw = generate_dataset(1_000, digit_dist={5: 1.0}, seed=789)

print("Tokenizing...")
train_data = tokenize_dataset(train_raw, tokenizer)
val_data = tokenize_dataset(val_raw, tokenizer)
test_data = tokenize_dataset(test_raw, tokenizer)
ood_data = tokenize_dataset(ood_raw, tokenizer)

print("Saving...")
torch.save(train_data, os.path.join(DATA_DIR, "train.pt"))
torch.save(val_data, os.path.join(DATA_DIR, "val.pt"))
torch.save(test_data, os.path.join(DATA_DIR, "test.pt"))
torch.save(ood_data, os.path.join(DATA_DIR, "ood.pt"))

# Print stats
for name, raw in [("train", train_raw), ("val", val_raw), ("test", test_raw), ("ood", ood_raw)]:
    step_counts = [len(c) for _, c, _ in raw]
    print(f"{name}: {len(raw)} problems, avg {sum(step_counts)/len(step_counts):.1f} CoT steps")

print(f"\nSample: {train_raw[0]}")
print(f"Vocab size: {len(tokenizer)}")
print("Done!")
