#!/usr/bin/env python3
"""Train the Activation Oracle on collected Coconut activations."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import json

from src.data_gen import make_tokenizer, ACT
from src.activation_oracle import ActivationOracle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data"
CKPT_DIR = "checkpoints"

# Hyperparameters
BATCH_SIZE = 16
GRAD_ACCUM = 4
LR = 1e-5
EPOCHS = 3
MAX_SEQ_LEN = 192
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01

# Source layers to sample from
SOURCE_LAYERS = [3, 6, 9]

# Question templates for each task type
COT_QUESTIONS = [
    "What is the intermediate calculation at this reasoning step?",
    "What calculation is being performed here?",
    "Describe the arithmetic operation at this step.",
    "What is the model computing at this point?",
    "What intermediate result is encoded here?",
    "Explain the reasoning step.",
    "What addition operation happens at this step?",
    "Describe this reasoning step in detail.",
]

COT_MULTI_QUESTIONS = [
    "Describe all reasoning steps.",
    "What are all the intermediate calculations?",
    "List each step of the reasoning process.",
    "Explain the full chain of reasoning.",
    "What calculations were performed?",
]

ANSWER_QUESTIONS = [
    "What is the final answer to this math problem?",
    "What is the result of this calculation?",
    "What number does this computation produce?",
    "What is the answer?",
    "What does this evaluate to?",
]

CONTEXT_BEFORE_QUESTIONS = [
    "Can you predict the text that came before this reasoning?",
    "What problem is being solved?",
    "What was the input to this computation?",
    "Predict the text before this reasoning.",
]

CONTEXT_AFTER_QUESTIONS = [
    "Can you predict what answer follows this reasoning?",
    "What comes after this reasoning?",
    "Predict the result that follows.",
]


def make_ao_examples(activation_data, tokenizer):
    """Create training examples for the Activation Oracle.

    Returns list of dicts with:
        - prompt_text: str (oracle prompt)
        - target_text: str (expected answer)
        - activation_vectors: list of (D,) tensors
        - source_layer: int
    """
    examples = []

    for item in activation_data:
        cot_steps = item["cot_steps"]
        num_steps = item["num_steps"]
        problem = item["problem"]
        answer = item["answer"]
        layer_hiddens = item["layer_hiddens"]  # list of dicts {layer: (D,)}

        # Skip if we don't have hidden states
        if not layer_hiddens or num_steps == 0:
            continue

        for source_layer in SOURCE_LAYERS:
            # Task 1a: Single-step CoT recovery (30% of data)
            for step_idx in range(num_steps):
                if source_layer not in layer_hiddens[step_idx]:
                    continue
                vec = layer_hiddens[step_idx][source_layer]
                q = random.choice(COT_QUESTIONS)
                prompt = f"Layer {source_layer}: {ACT} {q}"
                target = cot_steps[step_idx]
                examples.append({
                    "prompt_text": prompt,
                    "target_text": target,
                    "activation_vectors": [vec],
                    "source_layer": source_layer,
                })

            # Task 1b: Multi-step CoT recovery (30% of data)
            if num_steps > 1:
                vecs = [layer_hiddens[s][source_layer] for s in range(num_steps) if source_layer in layer_hiddens[s]]
                if vecs:
                    act_tokens = " ".join([ACT] * len(vecs))
                    q = random.choice(COT_MULTI_QUESTIONS)
                    prompt = f"Layer {source_layer}: {act_tokens} {q}"
                    # Format all steps
                    target = " ".join(f"Step {i+1}: {step}." for i, step in enumerate(cot_steps))
                    examples.append({
                        "prompt_text": prompt,
                        "target_text": target,
                        "activation_vectors": vecs,
                        "source_layer": source_layer,
                    })

            # Task 2: Answer prediction (20% of data)
            vecs = [layer_hiddens[s][source_layer] for s in range(num_steps) if source_layer in layer_hiddens[s]]
            if vecs:
                act_tokens = " ".join([ACT] * len(vecs))
                q = random.choice(ANSWER_QUESTIONS)
                prompt = f"Layer {source_layer}: {act_tokens} {q}"
                examples.append({
                    "prompt_text": prompt,
                    "target_text": answer,
                    "activation_vectors": vecs,
                    "source_layer": source_layer,
                })

            # Task 3: Context prediction (20% of data)
            vecs = [layer_hiddens[s][source_layer] for s in range(num_steps) if source_layer in layer_hiddens[s]]
            if vecs:
                act_tokens = " ".join([ACT] * len(vecs))
                # Before context
                q = random.choice(CONTEXT_BEFORE_QUESTIONS)
                prompt = f"Layer {source_layer}: {act_tokens} {q}"
                examples.append({
                    "prompt_text": prompt,
                    "target_text": problem,
                    "activation_vectors": vecs,
                    "source_layer": source_layer,
                })
                # After context
                q = random.choice(CONTEXT_AFTER_QUESTIONS)
                prompt = f"Layer {source_layer}: {act_tokens} {q}"
                examples.append({
                    "prompt_text": prompt,
                    "target_text": answer,
                    "activation_vectors": vecs,
                    "source_layer": source_layer,
                })

    return examples


def train_batch(model, batch, tokenizer):
    """Process a single training batch.

    Each item in batch is a dict with prompt_text, target_text, activation_vectors.
    We process them one at a time due to variable-length injection.
    """
    total_loss = 0
    count = 0

    for item in batch:
        prompt_text = item["prompt_text"]
        target_text = item["target_text"]
        vecs = item["activation_vectors"]

        # Tokenize: prompt + " " + target
        full_text = prompt_text + " " + target_text
        encoded = tokenizer.encode(full_text, add_special_tokens=False)

        if len(encoded) > MAX_SEQ_LEN:
            encoded = encoded[:MAX_SEQ_LEN]

        input_ids = torch.tensor([encoded], device=model.device)

        # Find <act> token positions
        act_positions = [i for i, t in enumerate(encoded) if t == model.act_id]

        # Create labels: -100 for prompt tokens, real IDs for target tokens
        prompt_encoded = tokenizer.encode(prompt_text + " ", add_special_tokens=False)
        prompt_len = min(len(prompt_encoded), len(encoded))
        labels = [-100] * prompt_len + encoded[prompt_len:]
        labels = labels[:len(encoded)]
        labels = torch.tensor([labels], device=model.device)

        # Set injection
        if act_positions and vecs:
            model.set_injection(vecs[:len(act_positions)], act_positions)
        else:
            model.clear_injection()

        loss, _ = model(input_ids, labels=labels)
        total_loss += loss
        count += 1

    model.clear_injection()

    if count > 0:
        return total_loss / count
    return torch.tensor(0.0, device=model.device)


def main():
    print("Loading tokenizer...")
    tokenizer = make_tokenizer()

    print("Loading activation data...")
    train_acts = torch.load(os.path.join(DATA_DIR, "activations_train.pt"), weights_only=False)
    val_acts = torch.load(os.path.join(DATA_DIR, "activations_val.pt"), weights_only=False)

    print("Creating AO training examples...")
    train_examples = make_ao_examples(train_acts, tokenizer)
    val_examples = make_ao_examples(val_acts, tokenizer)

    # Subsample to manageable size
    random.seed(42)
    if len(train_examples) > 200_000:
        random.shuffle(train_examples)
        train_examples = train_examples[:200_000]
    if len(val_examples) > 10_000:
        random.shuffle(val_examples)
        val_examples = val_examples[:10_000]

    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    print("Initializing Activation Oracle...")
    oracle = ActivationOracle(tokenizer, device=DEVICE)
    oracle = oracle.to(DEVICE)
    print(f"Trainable parameters: {oracle.num_trainable_params():,}")

    optimizer = torch.optim.AdamW(oracle.trainable_params(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_examples) // BATCH_SIZE * EPOCHS // GRAD_ACCUM
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss = float("inf")
    log_history = []

    for epoch in range(EPOCHS):
        random.shuffle(train_examples)
        oracle.train()

        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(range(0, len(train_examples) - BATCH_SIZE + 1, BATCH_SIZE),
                    desc=f"AO epoch {epoch+1}/{EPOCHS}")

        for batch_start in pbar:
            batch = train_examples[batch_start:batch_start + BATCH_SIZE]

            loss = train_batch(oracle, batch, tokenizer)
            loss = loss / GRAD_ACCUM
            loss.backward()

            if (n_batches + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(oracle.trainable_params(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM:.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Validation
        oracle.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for i in range(0, min(len(val_examples), 2000) - BATCH_SIZE + 1, BATCH_SIZE):
                batch = val_examples[i:i + BATCH_SIZE]
                loss = train_batch(oracle, batch, tokenizer)
                val_loss += loss.item()
                val_count += 1

        avg_val = val_loss / max(val_count, 1)
        print(f"Val loss: {avg_val:.4f}")
        log_history.append({"epoch": epoch+1, "train_loss": avg_loss, "val_loss": avg_val})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_path = os.path.join(CKPT_DIR, "oracle_best.pt")
            torch.save({
                "lora_state": {k: v for k, v in oracle.state_dict().items() if "lora" in k},
                "full_state": oracle.state_dict(),
            }, ckpt_path)
            print(f"Saved best: {ckpt_path}")

    # Save final
    torch.save({
        "lora_state": {k: v for k, v in oracle.state_dict().items() if "lora" in k},
        "full_state": oracle.state_dict(),
    }, os.path.join(CKPT_DIR, "oracle_final.pt"))

    with open(os.path.join(CKPT_DIR, "oracle_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    print("AO training complete!")


if __name__ == "__main__":
    main()
