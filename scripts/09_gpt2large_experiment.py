#!/usr/bin/env python3
"""End-to-end Cocoracle experiment with GPT-2-large (774M).

Runs Coconut training (stages 0-1), activation collection,
self-oracle training, and evaluation in a single script.
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

# GPT-2-large: 36 layers, d=1280
N_LAYERS = 36
D_MODEL = 1280
EXTRACT_LAYER = 18   # 50% depth
INJECT_LAYER = 17    # one before extraction

# ── Shared model helpers ─────────────────────────────────────────────

def load_model(tokenizer):
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    return model


# ── Phase 1: Coconut training ────────────────────────────────────────

def train_coconut_stage0(model, train_data, val_data, tokenizer):
    """Stage 0: full text CoT, standard LM training."""
    print("\n" + "="*60)
    print("COCONUT STAGE 0: Full text CoT")
    print("="*60)

    BATCH = 16
    ACCUM = 2
    EPOCHS = 3
    LR = 5e-5
    n_train = len(train_data["problems"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = n_train // BATCH * EPOCHS // ACCUM
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps*0.05), steps)
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0; count = 0

        pbar = tqdm(range(0, n_train - BATCH + 1, BATCH), desc=f"Stage0 ep{epoch+1}")
        for batch_start in pbar:
            idx = perm[batch_start:batch_start+BATCH]
            texts = []
            for i in idx:
                p = train_data["problems"][i]
                c = f" {SEP} ".join(train_data["cot_steps"][i])
                a = train_data["answers"][i]
                texts.append(f"{p} {BOT} {c} {EOT} {a}")

            enc = tokenizer(texts, padding=True, truncation=True, max_length=192, return_tensors="pt")
            ids = enc["input_ids"].to(DEVICE)
            labels = ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            with autocast():
                out = model(ids, labels=labels)
                loss = out.loss / ACCUM

            scaler.scale(loss).backward()
            total_loss += loss.item() * ACCUM
            count += 1

            if count % ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if count % 200 == 0:
                pbar.set_postfix(loss=f"{total_loss/count:.3f}")

        print(f"  Epoch {epoch+1}: loss={total_loss/count:.4f}")

    ckpt = os.path.join(CKPT_DIR, "stage0.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"  Saved: {ckpt}")
    return ckpt


def train_coconut_stage1(model, train_data, val_data, tokenizer):
    """Stage 1: last CoT step is latent."""
    print("\n" + "="*60)
    print("COCONUT STAGE 1: Last step latent")
    print("="*60)

    EPOCHS = 3
    LR = 1e-5
    ACCUM = 16
    n_train = min(len(train_data["problems"]), 20000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = n_train * EPOCHS // ACCUM
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps*0.05), steps)
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0; count = 0; correct = 0; total_eval = 0

        pbar = tqdm(range(n_train), desc=f"Stage1 ep{epoch+1}")
        for idx in pbar:
            i = perm[idx].item()
            prob = train_data["problems"][i]
            cot = train_data["cot_steps"][i]
            ans = train_data["answers"][i]
            n_steps = len(cot)

            # Prefix: problem + all but last CoT step
            prefix = f"{prob} {BOT}"
            for j in range(n_steps - 1):
                prefix += f" {cot[j]} {SEP}"
            answer_text = f"{EOT} {ans}"

            prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(DEVICE)
            answer_ids = tokenizer.encode(answer_text, add_special_tokens=False, return_tensors="pt").to(DEVICE)

            with autocast():
                # Forward prefix
                prefix_out = model.transformer(prefix_ids, use_cache=True, output_hidden_states=True)
                past_kv = prefix_out.past_key_values
                last_h = prefix_out.last_hidden_state[:, -1:, :]  # (1, 1, D)

                # One latent step: feed hidden state back
                latent_out = model.transformer(inputs_embeds=last_h, past_key_values=past_kv, use_cache=True)
                past_kv = latent_out.past_key_values
                latent_h = latent_out.last_hidden_state  # (1, 1, D)

                # Forward answer tokens
                answer_emb = model.transformer.wte(answer_ids)
                answer_out = model.transformer(inputs_embeds=answer_emb, past_key_values=past_kv, use_cache=True)
                logits = model.lm_head(answer_out.last_hidden_state)  # (1, L_ans, V)

                targets = answer_ids[:, 1:]
                logits = logits[:, :-1, :]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                # Norm regularization
                loss = loss + 0.001 * latent_h.norm(dim=-1).mean()

            scaler.scale(loss / ACCUM).backward()
            total_loss += loss.item()
            count += 1

            if count % ACCUM == 0:
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

        # Quick accuracy check
        model.eval()
        correct = 0; total_eval = 0
        with torch.no_grad():
            for i in range(min(500, n_train)):
                prob = val_data["problems"][i]
                cot = val_data["cot_steps"][i]
                ans = val_data["answers"][i]
                n_s = len(cot)

                prefix = f"{prob} {BOT}"
                for j in range(n_s - 1):
                    prefix += f" {cot[j]} {SEP}"

                prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(DEVICE)
                answer_text = f"{EOT} {ans}"
                answer_ids = tokenizer.encode(answer_text, add_special_tokens=False, return_tensors="pt").to(DEVICE)

                prefix_out = model.transformer(prefix_ids, use_cache=True, output_hidden_states=True)
                past_kv = prefix_out.past_key_values
                last_h = prefix_out.last_hidden_state[:, -1:, :]
                latent_out = model.transformer(inputs_embeds=last_h, past_key_values=past_kv, use_cache=True)
                past_kv = latent_out.past_key_values
                answer_emb = model.transformer.wte(answer_ids)
                answer_out = model.transformer(inputs_embeds=answer_emb, past_key_values=past_kv)
                logits = model.lm_head(answer_out.last_hidden_state)
                pred_ids = logits[:, :-1].argmax(dim=-1)
                pred = tokenizer.decode(pred_ids[0]).strip()
                if pred == ans:
                    correct += 1
                total_eval += 1

        print(f"  Val accuracy: {correct}/{total_eval} = {correct/total_eval:.4f}")

    ckpt = os.path.join(CKPT_DIR, "stage1.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"  Saved: {ckpt}")
    return ckpt


# ── Phase 2: Collect activations ─────────────────────────────────────

@torch.no_grad()
def collect_activations(model, data, tokenizer, max_n=10000):
    """Collect hidden states from Stage 1 latent thought positions."""
    print("\n" + "="*60)
    print(f"COLLECTING ACTIVATIONS (n={max_n})")
    print("="*60)

    model.eval()
    results = []

    for i in tqdm(range(min(max_n, len(data["problems"]))), desc="Collecting"):
        prob = data["problems"][i]
        cot = data["cot_steps"][i]
        ans = data["answers"][i]
        n_s = len(cot)

        prefix = f"{prob} {BOT}"
        for j in range(n_s - 1):
            prefix += f" {cot[j]} {SEP}"

        prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(DEVICE)

        prefix_out = model.transformer(prefix_ids, use_cache=True, output_hidden_states=True)
        past_kv = prefix_out.past_key_values
        last_h = prefix_out.last_hidden_state[:, -1:, :]

        latent_out = model.transformer(
            inputs_embeds=last_h, past_key_values=past_kv,
            use_cache=True, output_hidden_states=True
        )

        # Extract hidden state at the extraction layer
        # hidden_states[0] = input embeds, [1] = after layer 0, ..., [k+1] = after layer k
        h_extract = latent_out.hidden_states[EXTRACT_LAYER + 1][0, 0, :].cpu()  # (D,)

        results.append({
            "problem": prob,
            "answer": ans,
            "latent_cot_steps": [cot[-1]],  # last step was latent
            "num_latent": 1,
            "hidden": h_extract,  # (D,)
        })

    return results


# ── Phase 3: Self-oracle training ────────────────────────────────────

class InjectionHook:
    """Manages activation injection for the self-oracle."""
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
        gold = item["latent_cot_steps"][0]
        vec = item["hidden"]
        prob = item["problem"]
        ans = item["answer"]

        # CoT recovery
        q = random.choice(COT_QUESTIONS)
        examples.append({
            "prompt": f"Layer {EXTRACT_LAYER}: {ACT} {q}",
            "target": gold,
            "vec": vec,
        })
        # Answer prediction
        q = random.choice(ANSWER_QUESTIONS)
        examples.append({
            "prompt": f"Layer {EXTRACT_LAYER}: {ACT} {q}",
            "target": ans,
            "vec": vec,
        })
        # Context prediction
        q = random.choice(CONTEXT_QUESTIONS)
        examples.append({
            "prompt": f"Layer {EXTRACT_LAYER}: {ACT} {q}",
            "target": prob,
            "vec": vec,
        })

    return examples


def train_self_oracle(model, hook, train_acts, val_acts, train_text_data, tokenizer):
    """Train self-oracle with mixed AO + text-CoT objective."""
    print("\n" + "="*60)
    print("SELF-ORACLE TRAINING")
    print("="*60)

    EPOCHS = 5
    LR = 5e-6
    ACCUM = 16
    AO_RATIO = 0.7

    train_ao = make_ao_examples(train_acts, tokenizer)
    val_ao = make_ao_examples(val_acts, tokenizer)
    random.shuffle(train_ao)
    train_ao = train_ao[:50000]
    val_ao = val_ao[:3000]

    # Text examples for anti-forgetting
    train_text = []
    for i in range(min(15000, len(train_text_data["problems"]))):
        p = train_text_data["problems"][i]
        c = f" {SEP} ".join(train_text_data["cot_steps"][i])
        a = train_text_data["answers"][i]
        train_text.append(f"{p} {BOT} {c} {EOT} {a}")

    n_ao = int((len(train_ao) + len(train_text)) * AO_RATIO)
    n_text = (len(train_ao) + len(train_text)) - n_ao
    print(f"  AO examples: {len(train_ao)}, Text examples: {len(train_text)}")

    act_id = tokenizer.convert_tokens_to_ids(ACT)
    eos_id = tokenizer.eos_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (n_ao + n_text) * EPOCHS // ACCUM
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps*0.05), total_steps)
    scaler = GradScaler()

    best_ao_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        random.shuffle(train_ao)
        random.shuffle(train_text)

        mixed = [(True, x) for x in train_ao[:n_ao]] + [(False, x) for x in train_text[:n_text]]
        random.shuffle(mixed)

        ao_loss_sum = 0; text_loss_sum = 0; ao_count = 0; text_count = 0; step = 0

        pbar = tqdm(mixed, desc=f"SelfOracle ep{epoch+1}")
        for is_ao, item in pbar:
            if is_ao:
                prompt = item["prompt"]
                target = item["target"]
                vec = item["vec"]
                full = prompt + " " + target + tokenizer.eos_token
                encoded = tokenizer.encode(full, add_special_tokens=False)
                if len(encoded) > 192:
                    encoded = encoded[:192]
                ids = torch.tensor([encoded], device=DEVICE)

                prompt_enc = tokenizer.encode(prompt + " ", add_special_tokens=False)
                plen = min(len(prompt_enc), len(encoded))
                labels = [-100]*plen + encoded[plen:]
                labels = labels[:len(encoded)]
                labels = torch.tensor([labels], device=DEVICE)

                act_pos = [j for j, t in enumerate(encoded) if t == act_id]
                if act_pos:
                    hook.set([vec], act_pos)

                with autocast():
                    out = model(ids, labels=labels)
                    loss = out.loss

                hook.clear()
                if torch.isfinite(loss):
                    scaler.scale(loss / ACCUM).backward()
                    ao_loss_sum += loss.item()
                    ao_count += 1
            else:
                text = item
                encoded = tokenizer.encode(text, add_special_tokens=False)
                if len(encoded) > 192:
                    encoded = encoded[:192]
                ids = torch.tensor([encoded], device=DEVICE)
                labels = ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100

                with autocast():
                    out = model(ids, labels=labels)
                    loss = out.loss

                if torch.isfinite(loss):
                    scaler.scale(loss / ACCUM).backward()
                    text_loss_sum += loss.item()
                    text_count += 1

            step += 1
            if step % ACCUM == 0:
                # Check if any gradients exist (scaler.unscale_ fails if no backward was called)
                has_grads = any(p.grad is not None for p in model.parameters())
                if has_grads:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                optimizer.zero_grad()

            if step % 1000 == 0:
                al = ao_loss_sum/max(ao_count,1)
                tl = text_loss_sum/max(text_count,1)
                pbar.set_postfix(ao=f"{al:.2f}", text=f"{tl:.2f}")

        al = ao_loss_sum/max(ao_count,1)
        tl = text_loss_sum/max(text_count,1)
        print(f"  Epoch {epoch+1}: AO={al:.4f}, Text={tl:.4f}")

        # Validation
        model.eval()
        val_loss = 0; val_n = 0
        with torch.no_grad():
            for item in val_ao[:1000]:
                prompt = item["prompt"]
                target = item["target"]
                vec = item["vec"]
                full = prompt + " " + target + tokenizer.eos_token
                encoded = tokenizer.encode(full, add_special_tokens=False)[:192]
                ids = torch.tensor([encoded], device=DEVICE)
                prompt_enc = tokenizer.encode(prompt + " ", add_special_tokens=False)
                plen = min(len(prompt_enc), len(encoded))
                labels = [-100]*plen + encoded[plen:]
                labels = torch.tensor([labels[:len(encoded)]], device=DEVICE)
                act_pos = [j for j, t in enumerate(encoded) if t == act_id]
                if act_pos:
                    hook.set([vec], act_pos)
                with autocast():
                    out = model(ids, labels=labels)
                hook.clear()
                if torch.isfinite(out.loss):
                    val_loss += out.loss.item()
                    val_n += 1

        vl = val_loss / max(val_n, 1)
        print(f"  Val AO loss: {vl:.4f}")

        if vl < best_ao_loss:
            best_ao_loss = vl
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "self_oracle.pt"))
            print(f"  Saved best (loss={vl:.4f})")

    return best_ao_loss


# ── Phase 4: Evaluation ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, hook, test_acts, tokenizer, max_n=1000):
    """Evaluate self-oracle: CoT recovery, answer prediction."""
    print("\n" + "="*60)
    print(f"EVALUATION (n={max_n})")
    print("="*60)

    model.eval()
    act_id = tokenizer.convert_tokens_to_ids(ACT)
    eos_id = tokenizer.eos_token_id

    cot_exact = 0; cot_f1_sum = 0; cot_total = 0
    ans_exact = 0; ans_total = 0
    rand_exact = 0; rand_total = 0

    for i in tqdm(range(min(max_n, len(test_acts))), desc="Eval"):
        item = test_acts[i]
        gold_cot = item["latent_cot_steps"][0]
        gold_ans = item["answer"]
        vec = item["hidden"]

        # --- CoT recovery ---
        prompt = f"Layer {EXTRACT_LAYER}: {ACT} What is the intermediate calculation at this reasoning step?"
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        act_pos = [j for j, t in enumerate(prompt_ids[0].tolist()) if t == act_id]

        hook.set([vec], act_pos)
        # Custom generation: suppress EOS for first 3 tokens
        generated = []
        past_kv = None; curr_ids = prompt_ids
        for s in range(25):
            out = model(curr_ids, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[0, -1].float()
            if s < 3:
                logits[eos_id] = -float('inf')
                for sid in [tokenizer.convert_tokens_to_ids(t) for t in [BOT, SEP, EOT, ACT]]:
                    logits[sid] = -float('inf')
            nxt = logits.argmax().item()
            if nxt == eos_id: break
            generated.append(nxt)
            curr_ids = torch.tensor([[nxt]], device=DEVICE)
        hook.clear()

        pred_cot = tokenizer.decode(generated).strip()
        cot_total += 1
        if pred_cot == gold_cot:
            cot_exact += 1
        # Token F1
        p_set = set(pred_cot.split()); g_set = set(gold_cot.split())
        if p_set and g_set:
            common = p_set & g_set
            prec = len(common)/len(p_set); rec = len(common)/len(g_set)
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        else:
            f1 = 1.0 if (not p_set and not g_set) else 0.0
        cot_f1_sum += f1

        # --- Answer prediction ---
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
                for sid in [tokenizer.convert_tokens_to_ids(t) for t in [BOT, SEP, EOT, ACT]]:
                    logits[sid] = -float('inf')
            nxt = logits.argmax().item()
            if nxt == eos_id: break
            gen2.append(nxt)
            curr_ids = torch.tensor([[nxt]], device=DEVICE)
        hook.clear()
        pred_ans = tokenizer.decode(gen2).strip()
        # Extract digits
        num = ""
        for c in pred_ans:
            if c.isdigit(): num += c
            elif num: break
        ans_total += 1
        if num == gold_ans:
            ans_exact += 1

        # --- Random baseline (every 5th example) ---
        if i % 5 == 0:
            rand_vec = torch.randn_like(vec) * vec.norm()
            prompt_ids_r = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            act_pos_r = [j for j, t in enumerate(prompt_ids_r[0].tolist()) if t == act_id]
            hook.set([rand_vec], act_pos_r)
            gen_r = []
            past_kv = None; curr_ids = prompt_ids_r
            for s in range(25):
                out = model(curr_ids, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                logits = out.logits[0, -1].float()
                if s < 3:
                    logits[eos_id] = -float('inf')
                    for sid in [tokenizer.convert_tokens_to_ids(t) for t in [BOT, SEP, EOT, ACT]]:
                        logits[sid] = -float('inf')
                nxt = logits.argmax().item()
                if nxt == eos_id: break
                gen_r.append(nxt)
                curr_ids = torch.tensor([[nxt]], device=DEVICE)
            hook.clear()
            pred_r = tokenizer.decode(gen_r).strip()
            rand_total += 1
            if pred_r == gold_cot:
                rand_exact += 1

        # Print first 20 examples
        if i < 20:
            print(f"  Gold=[{gold_cot}]  Pred=[{pred_cot}]  {'MATCH' if pred_cot==gold_cot else ''}")

    results = {
        "cot_exact_match": cot_exact / max(cot_total, 1),
        "cot_token_f1": cot_f1_sum / max(cot_total, 1),
        "answer_exact_match": ans_exact / max(ans_total, 1),
        "random_cot_exact": rand_exact / max(rand_total, 1),
    }

    print(f"\n{'='*60}")
    print(f"GPT-2-LARGE RESULTS")
    print(f"{'='*60}")
    print(f"CoT exact match:    {results['cot_exact_match']:.4f}")
    print(f"CoT token F1:       {results['cot_token_f1']:.4f}")
    print(f"Answer exact match: {results['answer_exact_match']:.4f}")
    print(f"Random baseline:    {results['random_cot_exact']:.4f}")

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    torch.manual_seed(42)

    print("Loading tokenizer...")
    tokenizer = make_tokenizer()

    print("Loading data...")
    train_data = torch.load(os.path.join(DATA_DIR, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(DATA_DIR, "val.pt"), weights_only=False)
    test_data = torch.load(os.path.join(DATA_DIR, "test.pt"), weights_only=False)

    print(f"Loading {MODEL_NAME}...")
    model = load_model(tokenizer)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Phase 1: Coconut training (skip if checkpoints exist)
    stage1_ckpt = os.path.join(CKPT_DIR, "stage1.pt")
    if os.path.exists(stage1_ckpt):
        print(f"Loading existing Stage 1 checkpoint: {stage1_ckpt}")
        model.load_state_dict(torch.load(stage1_ckpt, map_location=DEVICE))
    else:
        train_coconut_stage0(model, train_data, val_data, tokenizer)
        train_coconut_stage1(model, train_data, val_data, tokenizer)

    # Phase 2: Collect activations
    train_acts = collect_activations(model, train_data, tokenizer, max_n=10000)
    val_acts = collect_activations(model, val_data, tokenizer, max_n=2000)
    test_acts = collect_activations(model, test_data, tokenizer, max_n=2000)

    # Phase 3: Self-oracle training
    hook = InjectionHook(model.transformer.h[INJECT_LAYER], scale=2.0)
    train_self_oracle(model, hook, train_acts, val_acts, train_data, tokenizer)

    # Reload best checkpoint
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "self_oracle.pt"), map_location=DEVICE))

    # Phase 4: Evaluate
    results = evaluate(model, hook, test_acts, tokenizer, max_n=1000)

    # Save
    results_path = os.path.join(RESULTS_DIR, "gpt2large_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
