# Cocoracle: Activation Oracles for Coconut Latent Reasoning Tokens

## Experiment Summary

This experiment investigates whether Activation Oracle (AO) techniques — originally developed for interpreting LLM activations via natural-language Q&A — can be applied to interpret the latent reasoning tokens produced by Coconut (Chain of Continuous Thought) models.

We trained a Coconut model based on GPT-2-small (124M params) to perform multi-digit addition with latent chain-of-thought, then trained both linear probes and an AO to interpret what the latent reasoning tokens encode.

## Coconut Model Results

**Task**: Multi-digit addition (2-4 digit numbers) with step-by-step carry propagation.

**Curriculum training** (4 stages, each replacing more CoT text with latent thoughts):

| Stage | Description | Val Loss | Val Accuracy (teacher-forced) |
|-------|------------|----------|-------------------------------|
| 0 | Full text CoT | 0.21 | N/A (text mode) |
| 1 | Last step latent | 0.50 | **69.4%** |
| 2 | Last 2 steps latent | 1.83 | 18.6% |
| 3 | All steps latent | 3.64 | 2.4% |

**Key finding**: The model successfully learns to perform the final reasoning step in latent space (Stage 1, 69% accuracy), but struggles as more steps become latent. This is consistent with the Coconut paper's findings — latent reasoning is harder to learn and requires more training.

**Generation accuracy** (Stage 1, autoregressive): 24.8% — lower than teacher-forced due to BPE tokenization issues with number generation.

## Activation Oracle Results

### Linear Probes (Baseline)

Linear probes trained on Stage 1 hidden states at layer 6 (50% depth):

| Probe | Accuracy |
|-------|----------|
| **First token prediction** | **100%** |
| **Step identity classification** | **100%** |
| **Answer digit prediction** (exact match) | **41.9%** |

**Key finding**: The latent hidden states encode rich, linearly readable information about the reasoning step. The 100% accuracy on first token and step identity proves that the model's latent thoughts carry specific, structured information about what computation is being performed.

### Activation Oracle (AO-style)

The AO (GPT-2-small + LoRA, 4.7M trainable params) with norm-matched injection after layer 1:

| Metric | Value |
|--------|-------|
| CoT exact match | 0.0% |
| CoT token F1 | 26.4% |
| CoT BLEU | 12.2% |
| Answer exact match | 0.0% |
| Context BLEU | 5.8% |
| Random baseline exact match | 0.0% |

**Key finding**: The AO partially recovers reasoning content (26% token F1 vs 0% random baseline), but suffers from mode collapse — producing repetitive outputs. The injection mechanism alone doesn't provide enough signal for GPT-2-small to learn meaningful activation interpretation.

## Analysis: Why Linear Probes Succeed But AO Struggles

1. **Model capacity**: The AO paper uses 8B-70B parameter models. Our GPT-2-small (124M) oracle with only LoRA (4.7M trainable) lacks the capacity to learn the complex mapping from injected activations to natural language descriptions.

2. **Injection mechanism**: Norm-matched addition after layer 1 is a relatively weak signal. The original paper benefits from larger hidden dimensions and more layers to process the injected information.

3. **Training scale**: We trained on 50K examples with 3 epochs. The AO paper uses ~1M examples across diverse tasks.

4. **Linear probes work because**: The information IS there in the hidden states — linearly separable, in fact. The challenge is specifically in the *generation* pipeline (injection → multi-layer processing → text generation).

## Conclusions

### Is extending AOs to Coconut latent reasoning viable?

**Partially yes, with caveats**:

1. **The information is extractable**: Linear probes achieve 100% accuracy on identifying reasoning steps from latent hidden states, proving the information exists and is structured.

2. **AO-style injection needs more scale**: The natural-language Q&A approach requires larger oracle models and more diverse training data. GPT-2-small is too small for this paradigm.

3. **The Coconut model itself needs improvement**: With only 2-3% accuracy at the all-latent stage, the latent representations aren't fully developed. More training (especially with the Coconut paper's full curriculum) would produce richer latent states.

4. **Promising direction**: If scaled to larger models (e.g., Llama-7B as both Coconut and Oracle), this approach could enable interpreting what a model is "thinking" during continuous reasoning — a capability not possible with standard interpretability tools.

## Self-Oracle: Using the Coconut Model as its Own AO

The separate-model AO collapsed because GPT-2-small lacked capacity to learn both arithmetic and activation interpretation from scratch. We hypothesized that using the **Coconut model itself** as the oracle — since it already understands arithmetic and its own internal representations — would work better.

### Approach
- Start from Stage 1 Coconut checkpoint (69% teacher-forced accuracy)
- Full fine-tune (all 124M params, no LoRA) on mixed objective:
  - 70% AO tasks (CoT recovery, answer prediction, context prediction)
  - 30% original text-CoT data (to prevent catastrophic forgetting)
- Norm-matched injection after layer 1 (same mechanism)
- 5 epochs, lr=5e-6

### Results

| Metric | Self-Oracle | Separate AO | Linear Probes |
|--------|-------------|-------------|---------------|
| **CoT token F1** | **32.5%** | 26.4% | N/A |
| CoT exact match | 0% | 0% | 100%* |
| Answer exact match | 0% | 0% | 41.9% |
| Random baseline | 0% | 0% | N/A |
| Text loss (forgetting) | 0.205 (stable) | N/A | N/A |

\* Probes measure first-token accuracy from a single linear layer, not full text generation.

### Key improvements over separate AO
1. **No mode collapse**: The self-oracle produces varied, arithmetic-flavored outputs that differ based on the input activation (the separate AO collapsed to a single repeated output).
2. **Higher token F1**: 32.5% vs 26.4% — a 23% relative improvement.
3. **No catastrophic forgetting**: Text loss stayed stable at 0.205 across all 5 epochs, confirming the mixed training objective works.
4. **AO loss still improving**: Val AO loss decreased every epoch (2.03 → 1.98), suggesting more training would continue to help.

### Analysis
The self-oracle generates text like `"8+1=9 write 9  carry 1 write 1  9993..."` — the beginning contains real arithmetic operations before degenerating. This is qualitatively different from the separate AO's total mode collapse. The model is clearly using the injected activations to condition its output, but GPT-2-small still lacks the capacity to precisely decode the activation into the exact correct CoT step.

### What would get this to work fully
1. **Model scale**: GPT-2-XL (1.5B) or Qwen-1.5B would likely cross the threshold. The AO paper's smallest successful model is 8B, but with the self-oracle advantage + narrow domain, 1.5B should suffice.
2. **More training**: AO loss was still decreasing at epoch 5 — training for 20+ epochs could help.
3. **Better stopping**: The model doesn't know when to stop generating. Adding EOS tokens to training targets would fix the "too much output" problem.

## Reproduction

```bash
pip install -r requirements.txt
bash scripts/run_all.sh  # ~3 hours on RTX 4090 (base experiment)
python scripts/07_train_self_oracle.py  # ~1.5 hours additional
python scripts/08_eval_self_oracle.py   # ~20 minutes
```

## Hardware

- GPU: NVIDIA RTX 4090 (24GB)
- Total runtime: ~5 hours
