#!/usr/bin/env python3
"""GPT-2-large: train Coconut through all-latent, then self-oracle.

Picks up from the Stage 1 checkpoint and trains Stages 2-3,
then collects activations, trains self-oracle, and evaluates.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2LMHeadModel, get_cosine_schedule_with_warmup
from tqdm import tqdm
import json

from src.data_gen import make_tokenizer, generate_dataset, BOT, SEP, EOT, ACT

DEVICE = "cuda"
MODEL_NAME = "gpt2-large"
DATA_DIR = "data"
CKPT_DIR = "checkpoints/large"
RESULTS_DIR = "results"
os.makedirs(CKPT_DIR, exist_ok=True)

N_LAYERS = 36
D_MODEL = 1280
EXTRACT_LAYER = 18
INJECT_LAYER = 17


def load_model(tokenizer):
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    return model


# ── Coconut Stage N latent ───────────────────────────────────────────

def train_coconut_latent(model, train_data, val_data, tokenizer, num_latent, stage_name, epochs=3):
    """Train Coconut with N latent steps (from the end)."""
    print(f"\n{'='*60}")
    print(f"COCONUT {stage_name}: {num_latent} latent step(s)")
    print(f"{'='*60}")

    LR = 1e-5
    ACCUM = 16
    n_train = min(len(train_data["problems"]), 20000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = n_train * epochs // ACCUM
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps * 0.05), steps)
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0; count = 0

        pbar = tqdm(range(n_train), desc=f"{stage_name} ep{epoch+1}")
        for idx in pbar:
            i = perm[idx].item()
            prob = train_data["problems"][i]
            cot = train_data["cot_steps"][i]
            ans = train_data["answers"][i]
            n_steps = len(cot)

            # How many latent for this example
            n_lat = min(num_latent, n_steps) if isinstance(num_latent, int) else n_steps
            n_text = n_steps - n_lat

            # Build text prefix
            prefix = f"{prob} {BOT}"
            for j in range(n_text):
                prefix += f" {cot[j]} {SEP}"
            answer_text = f"{EOT} {ans}"

            prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(DEVICE)
            answer_ids = tokenizer.encode(answer_text, add_special_tokens=False, return_tensors="pt").to(DEVICE)

            with autocast():
                # Forward prefix
                prefix_out = model.transformer(prefix_ids, use_cache=True, output_hidden_states=True)
                past_kv = prefix_out.past_key_values
                h = prefix_out.last_hidden_state[:, -1:, :]  # (1, 1, D)

                # Latent thought loop
                thought_norms = []
                for t in range(n_lat):
                    lat_out = model.transformer(inputs_embeds=h, past_key_values=past_kv, use_cache=True)
                    past_kv = lat_out.past_key_values
                    h = lat_out.last_hidden_state  # (1, 1, D)
                    thought_norms.append(h.norm(dim=-1).mean())

                # Forward answer
                answer_emb = model.transformer.wte(answer_ids)
                answer_out = model.transformer(inputs_embeds=answer_emb, past_key_values=past_kv)
                logits = model.lm_head(answer_out.last_hidden_state)

                targets = answer_ids[:, 1:]
                logits = logits[:, :-1, :]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                # Norm regularization
                if thought_norms:
                    loss = loss + 0.001 * sum(thought_norms) / len(thought_norms)

            scaler.scale(loss / ACCUM).backward()
            total_loss += loss.item()
            count += 1

            if count % ACCUM == 0:
                has_grads = any(p.grad is not None for p in model.parameters())
                if has_grads:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                optimizer.zero_grad()

            if count % 500 == 0:
                pbar.set_postfix(loss=f"{total_loss/count:.3f}")

        avg = total_loss / count
        print(f"  Epoch {epoch+1}: loss={avg:.4f}")

        # Validation accuracy
        model.eval()
        correct = 0; total_eval = 0
        with torch.no_grad():
            for vi in range(min(500, len(val_data["problems"]))):
                prob = val_data["problems"][vi]
                cot_v = val_data["cot_steps"][vi]
                ans_v = val_data["answers"][vi]
                n_s = len(cot_v)
                n_l = min(num_latent, n_s) if isinstance(num_latent, int) else n_s
                n_t = n_s - n_l

                prefix = f"{prob} {BOT}"
                for j in range(n_t):
                    prefix += f" {cot_v[j]} {SEP}"

                prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(DEVICE)
                answer_text = f"{EOT} {ans_v}"
                answer_ids = tokenizer.encode(answer_text, add_special_tokens=False, return_tensors="pt").to(DEVICE)

                prefix_out = model.transformer(prefix_ids, use_cache=True)
                past_kv = prefix_out.past_key_values
                h = prefix_out.last_hidden_state[:, -1:, :]
                for t in range(n_l):
                    lat_out = model.transformer(inputs_embeds=h, past_key_values=past_kv, use_cache=True)
                    past_kv = lat_out.past_key_values
                    h = lat_out.last_hidden_state

                answer_emb = model.transformer.wte(answer_ids)
                answer_out = model.transformer(inputs_embeds=answer_emb, past_key_values=past_kv)
                logits = model.lm_head(answer_out.last_hidden_state)
                pred_ids = logits[:, :-1].argmax(dim=-1)
                pred = tokenizer.decode(pred_ids[0]).strip()
                if pred == ans_v:
                    correct += 1
                total_eval += 1

        print(f"  Val accuracy: {correct}/{total_eval} = {correct/total_eval:.4f}")

    ckpt = os.path.join(CKPT_DIR, f"{stage_name}.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"  Saved: {ckpt}")
    return ckpt


# ── Collect activations ──────────────────────────────────────────────

@torch.no_grad()
def collect_activations(model, data, tokenizer, num_latent, max_n=10000):
    """Collect hidden states from all latent thought positions."""
    print(f"\n{'='*60}")
    print(f"COLLECTING ACTIVATIONS (n={max_n}, latent={num_latent})")
    print(f"{'='*60}")

    model.eval()
    results = []

    for i in tqdm(range(min(max_n, len(data["problems"]))), desc="Collecting"):
        prob = data["problems"][i]
        cot = data["cot_steps"][i]
        ans = data["answers"][i]
        n_s = len(cot)
        n_lat = min(num_latent, n_s) if isinstance(num_latent, int) else n_s
        n_text = n_s - n_lat

        prefix = f"{prob} {BOT}"
        for j in range(n_text):
            prefix += f" {cot[j]} {SEP}"

        prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(DEVICE)
        prefix_out = model.transformer(prefix_ids, use_cache=True, output_hidden_states=True)
        past_kv = prefix_out.past_key_values
        h = prefix_out.last_hidden_state[:, -1:, :]

        hiddens = []
        for t in range(n_lat):
            lat_out = model.transformer(
                inputs_embeds=h, past_key_values=past_kv,
                use_cache=True, output_hidden_states=True
            )
            past_kv = lat_out.past_key_values
            h = lat_out.last_hidden_state
            # Extract from target layer
            h_extract = lat_out.hidden_states[EXTRACT_LAYER + 1][0, 0, :].cpu()
            hiddens.append(h_extract)

        latent_cot = cot[n_text:]  # the CoT steps that are latent

        results.append({
            "problem": prob,
            "answer": ans,
            "latent_cot_steps": latent_cot,
            "num_latent": n_lat,
            "hiddens": hiddens,  # list of (D,) per latent step
        })

    return results


# ── Self-oracle (reuses InjectionHook from script 09) ────────────────

class InjectionHook:
    def __init__(self, layer_module, scale=2.0):
        self.scale = scale
        self.vectors = None
        self.positions = None
        self.handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        if self.vectors is None or self.positions is None:
            return output
        hs = output[0]
        seq_len = hs.size(1)
        pairs = [(i, p) for i, p in enumerate(self.positions) if p < seq_len and i < len(self.vectors)]
        if not pairs:
            return output
        delta = torch.zeros_like(hs)
        for i, pos in pairs:
            h_i = hs[:, pos, :]
            v_i = self.vectors[i].to(h_i.device).to(h_i.dtype)
            h_norm = h_i.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v_norm = v_i.norm().clamp(min=1e-8)
            delta[:, pos, :] = self.scale * h_norm * (v_i / v_norm)
        return (hs + delta,) + output[1:]

    def set(self, vectors, positions):
        if isinstance(vectors, list):
            vectors = torch.stack(vectors)
        self.vectors = vectors
        self.positions = positions

    def clear(self):
        self.vectors = None
        self.positions = None


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
]

CONTEXT_QUESTIONS = [
    "What problem is being solved?",
    "What was the input to this computation?",
]


def make_ao_examples(act_data, tokenizer):
    examples = []
    for item in act_data:
        for step_idx in range(item["num_latent"]):
            if step_idx >= len(item["hiddens"]) or step_idx >= len(item["latent_cot_steps"]):
                continue
            gold = item["latent_cot_steps"][step_idx]
            vec = item["hiddens"][step_idx]

            # CoT recovery (per step)
            q = random.choice(COT_QUESTIONS)
            examples.append({
                "prompt": f"Layer {EXTRACT_LAYER}: {ACT} {q}",
                "target": gold,
                "vec": vec,
            })

        # Answer + context (use last hidden)
        if item["hiddens"]:
            vec = item["hiddens"][-1]
            q = random.choice(ANSWER_QUESTIONS)
            examples.append({
                "prompt": f"Layer {EXTRACT_LAYER}: {ACT} {q}",
                "target": item["answer"],
                "vec": vec,
            })
            q = random.choice(CONTEXT_QUESTIONS)
            examples.append({
                "prompt": f"Layer {EXTRACT_LAYER}: {ACT} {q}",
                "target": item["problem"],
                "vec": vec,
            })

    return examples


def train_self_oracle(model, hook, train_acts, val_acts, train_text_data, tokenizer):
    print(f"\n{'='*60}")
    print("SELF-ORACLE TRAINING (all-latent)")
    print(f"{'='*60}")

    EPOCHS = 5
    LR = 5e-6
    ACCUM = 16

    train_ao = make_ao_examples(train_acts, tokenizer)
    val_ao = make_ao_examples(val_acts, tokenizer)
    random.shuffle(train_ao)
    train_ao = train_ao[:60000]
    val_ao = val_ao[:3000]

    # Text CoT for anti-forgetting
    train_text = []
    for i in range(min(15000, len(train_text_data["problems"]))):
        p = train_text_data["problems"][i]
        c = f" {SEP} ".join(train_text_data["cot_steps"][i])
        a = train_text_data["answers"][i]
        train_text.append(f"{p} {BOT} {c} {EOT} {a}")

    mixed_n = len(train_ao) + len(train_text)
    print(f"  AO: {len(train_ao)}, Text: {len(train_text)}, Total: {mixed_n}")

    act_id = tokenizer.convert_tokens_to_ids(ACT)
    eos_id = tokenizer.eos_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = mixed_n * EPOCHS // ACCUM
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.05), total_steps)
    scaler = GradScaler()

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        random.shuffle(train_ao)
        random.shuffle(train_text)

        mixed = [(True, x) for x in train_ao] + [(False, x) for x in train_text]
        random.shuffle(mixed)

        ao_loss_sum = 0; text_loss_sum = 0; ao_n = 0; text_n = 0; step = 0

        pbar = tqdm(mixed, desc=f"SelfOracle ep{epoch+1}")
        for is_ao, item in pbar:
            if is_ao:
                prompt = item["prompt"]
                target = item["target"]
                vec = item["vec"]
                full = prompt + " " + target + tokenizer.eos_token
                encoded = tokenizer.encode(full, add_special_tokens=False)[:192]
                ids = torch.tensor([encoded], device=DEVICE)

                prompt_enc = tokenizer.encode(prompt + " ", add_special_tokens=False)
                plen = min(len(prompt_enc), len(encoded))
                labels = [-100] * plen + encoded[plen:]
                labels = torch.tensor([labels[:len(encoded)]], device=DEVICE)

                act_pos = [j for j, t in enumerate(encoded) if t == act_id]
                if act_pos:
                    hook.set([vec], act_pos)

                with autocast():
                    loss = model(ids, labels=labels).loss
                hook.clear()

                if torch.isfinite(loss):
                    scaler.scale(loss / ACCUM).backward()
                    ao_loss_sum += loss.item(); ao_n += 1
            else:
                text = item
                encoded = tokenizer.encode(text, add_special_tokens=False)[:192]
                ids = torch.tensor([encoded], device=DEVICE)
                labels = ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100

                with autocast():
                    loss = model(ids, labels=labels).loss

                if torch.isfinite(loss):
                    scaler.scale(loss / ACCUM).backward()
                    text_loss_sum += loss.item(); text_n += 1

            step += 1
            if step % ACCUM == 0:
                has_grads = any(p.grad is not None for p in model.parameters())
                if has_grads:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                optimizer.zero_grad()

            if step % 1000 == 0:
                pbar.set_postfix(ao=f"{ao_loss_sum/max(ao_n,1):.2f}", text=f"{text_loss_sum/max(text_n,1):.2f}")

        al = ao_loss_sum / max(ao_n, 1); tl = text_loss_sum / max(text_n, 1)
        print(f"  Epoch {epoch+1}: AO={al:.4f}, Text={tl:.4f}")

        # Validate
        model.eval()
        vl_sum = 0; vl_n = 0
        with torch.no_grad():
            for item in val_ao[:1000]:
                full = item["prompt"] + " " + item["target"] + tokenizer.eos_token
                encoded = tokenizer.encode(full, add_special_tokens=False)[:192]
                ids = torch.tensor([encoded], device=DEVICE)
                prompt_enc = tokenizer.encode(item["prompt"] + " ", add_special_tokens=False)
                plen = min(len(prompt_enc), len(encoded))
                labels = [-100] * plen + encoded[plen:]
                labels = torch.tensor([labels[:len(encoded)]], device=DEVICE)
                act_pos = [j for j, t in enumerate(encoded) if t == act_id]
                if act_pos:
                    hook.set([item["vec"]], act_pos)
                with autocast():
                    loss = model(ids, labels=labels).loss
                hook.clear()
                if torch.isfinite(loss):
                    vl_sum += loss.item(); vl_n += 1

        vl = vl_sum / max(vl_n, 1)
        print(f"  Val AO loss: {vl:.4f}")

        if vl < best_loss:
            best_loss = vl
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "self_oracle_alllatent.pt"))
            print(f"  Saved best (loss={vl:.4f})")


# ── Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, hook, test_acts, tokenizer, max_n=1000):
    print(f"\n{'='*60}")
    print(f"EVALUATION (all-latent, n={max_n})")
    print(f"{'='*60}")

    model.eval()
    act_id = tokenizer.convert_tokens_to_ids(ACT)
    eos_id = tokenizer.eos_token_id
    special_ids = [tokenizer.convert_tokens_to_ids(t) for t in [BOT, SEP, EOT, ACT]]

    cot_exact = 0; cot_f1_sum = 0; cot_total = 0
    ans_exact = 0; ans_total = 0
    rand_exact = 0; rand_total = 0

    for i in tqdm(range(min(max_n, len(test_acts))), desc="Eval"):
        item = test_acts[i]
        if not item["hiddens"] or not item["latent_cot_steps"]:
            continue

        # Test each latent step
        for step_idx in range(min(item["num_latent"], len(item["hiddens"]), len(item["latent_cot_steps"]))):
            gold = item["latent_cot_steps"][step_idx]
            vec = item["hiddens"][step_idx]

            prompt = f"Layer {EXTRACT_LAYER}: {ACT} What is the intermediate calculation at this reasoning step?"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            act_pos = [j for j, t in enumerate(prompt_ids[0].tolist()) if t == act_id]

            hook.set([vec], act_pos)
            generated = []
            past_kv = None; curr_ids = prompt_ids
            for s in range(25):
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

            pred = tokenizer.decode(generated).strip()
            cot_total += 1
            if pred == gold:
                cot_exact += 1

            p_set = set(pred.split()); g_set = set(gold.split())
            if p_set and g_set:
                common = p_set & g_set
                prec = len(common) / len(p_set); rec = len(common) / len(g_set)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            else:
                f1 = 1.0 if (not p_set and not g_set) else 0.0
            cot_f1_sum += f1

            if i < 10:
                print(f"  Gold=[{gold}]  Pred=[{pred}]  {'MATCH' if pred == gold else ''}")

        # Answer prediction (from last hidden)
        vec = item["hiddens"][-1]
        prompt2 = f"Layer {EXTRACT_LAYER}: {ACT} What is the final answer?"
        prompt2_ids = tokenizer.encode(prompt2, return_tensors="pt").to(DEVICE)
        act_pos2 = [j for j, t in enumerate(prompt2_ids[0].tolist()) if t == act_id]
        hook.set([vec], act_pos2)
        gen2 = []
        past_kv = None; curr_ids = prompt2_ids
        for s in range(10):
            out = model(curr_ids, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[0, -1].float()
            if s < 2:
                logits[eos_id] = -float('inf')
                for sid in special_ids:
                    logits[sid] = -float('inf')
            nxt = logits.argmax().item()
            if nxt == eos_id:
                break
            gen2.append(nxt)
            curr_ids = torch.tensor([[nxt]], device=DEVICE)
        hook.clear()
        pred_ans = tokenizer.decode(gen2).strip()
        num = ""
        for c in pred_ans:
            if c.isdigit():
                num += c
            elif num:
                break
        ans_total += 1
        if num == item["answer"]:
            ans_exact += 1

        # Random baseline (every 10th)
        if i % 10 == 0 and item["hiddens"]:
            rand_vec = torch.randn_like(item["hiddens"][0]) * item["hiddens"][0].norm()
            hook.set([rand_vec], act_pos)
            gen_r = []
            past_kv = None; curr_ids = prompt_ids
            for s in range(25):
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
                gen_r.append(nxt)
                curr_ids = torch.tensor([[nxt]], device=DEVICE)
            hook.clear()
            pred_r = tokenizer.decode(gen_r).strip()
            rand_total += 1
            if pred_r == item["latent_cot_steps"][0]:
                rand_exact += 1

    results = {
        "cot_exact_match": cot_exact / max(cot_total, 1),
        "cot_token_f1": cot_f1_sum / max(cot_total, 1),
        "answer_exact_match": ans_exact / max(ans_total, 1),
        "random_cot_exact": rand_exact / max(rand_total, 1),
        "cot_total_steps_evaluated": cot_total,
    }

    print(f"\n{'='*60}")
    print("GPT-2-LARGE ALL-LATENT RESULTS")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    torch.manual_seed(42)

    tokenizer = make_tokenizer()

    print("Loading data...")
    train_data = torch.load(os.path.join(DATA_DIR, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(DATA_DIR, "val.pt"), weights_only=False)
    test_data = torch.load(os.path.join(DATA_DIR, "test.pt"), weights_only=False)

    print(f"Loading {MODEL_NAME}...")
    model = load_model(tokenizer)

    # Load Stage 1 checkpoint
    stage1_ckpt = os.path.join(CKPT_DIR, "stage1.pt")
    print(f"Loading: {stage1_ckpt}")
    model.load_state_dict(torch.load(stage1_ckpt, map_location=DEVICE))

    # Train Stages 2 and 3
    stage3_ckpt = os.path.join(CKPT_DIR, "stage3_alllatent.pt")
    if os.path.exists(stage3_ckpt):
        print(f"Loading existing: {stage3_ckpt}")
        model.load_state_dict(torch.load(stage3_ckpt, map_location=DEVICE))
    else:
        train_coconut_latent(model, train_data, val_data, tokenizer, num_latent=2, stage_name="stage2", epochs=3)
        train_coconut_latent(model, train_data, val_data, tokenizer, num_latent="all", stage_name="stage3_alllatent", epochs=3)

    # Collect activations (all-latent mode)
    train_acts = collect_activations(model, train_data, tokenizer, num_latent="all", max_n=10000)
    val_acts = collect_activations(model, val_data, tokenizer, num_latent="all", max_n=2000)
    test_acts = collect_activations(model, test_data, tokenizer, num_latent="all", max_n=2000)

    # Self-oracle
    hook = InjectionHook(model.transformer.h[INJECT_LAYER], scale=2.0)
    train_self_oracle(model, hook, train_acts, val_acts, train_data, tokenizer)

    # Reload best
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "self_oracle_alllatent.pt"), map_location=DEVICE))

    # Evaluate
    results = evaluate(model, hook, test_acts, tokenizer, max_n=1000)

    results_path = os.path.join(RESULTS_DIR, "gpt2large_alllatent_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
