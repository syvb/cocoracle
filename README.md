# Cocoracle

**Can we interpret what a model is "thinking" during latent reasoning?**

This project combines two ideas:
- [**Coconut**](https://arxiv.org/abs/2412.06769) (Chain of Continuous Thought): LLMs that reason in latent space instead of generating text chain-of-thought tokens
- [**Activation Oracles**](https://arxiv.org/abs/2512.15674) (AOs): Models trained to answer natural-language questions about another model's internal activations

We train Coconut models on GPT-2 that perform arithmetic with latent reasoning steps, then build a "self-oracle" — the same model fine-tuned to answer questions about its own hidden states — to interpret what the latent thoughts encode.

## Key Results

### Coconut model: latent reasoning works at scale

GPT-2-large (774M) learns to reason in latent space far better than GPT-2-small (124M):

| Stage | Description | GPT-2-small | GPT-2-large |
|-------|------------|-------------|-------------|
| 0 | Full text CoT | trained | trained |
| 1 | Last step latent | 69% | **99%** |
| 2 | Last 2 steps latent | 19% | **71%** |
| 3 | All steps latent | 2.4% | **45%** |

### Self-oracle: reading latent thoughts

The self-oracle approach — fine-tuning the Coconut model to interpret its own activations — achieves the first non-zero exact match on recovering chain-of-thought text from latent hidden states:

| Configuration | CoT Exact Match | CoT Token F1 | AO Val Loss |
|--------------|----------------|--------------|-------------|
| Separate AO (GPT-2-small + LoRA) | 0% | 26.4% | 2.92 |
| Self-oracle, GPT-2-small, stage 1 | 0% | 32.5% | 1.98 |
| Self-oracle, GPT-2-large, stage 1 | 0% | 25.6% | 1.10 |
| **Self-oracle, GPT-2-large, all-latent** | **6.9%** | **34.2%** | **0.55** |

The all-latent GPT-2-large self-oracle correctly recovers exact CoT strings like `"2+8=10 write 0 carry 1"` and `"4+4=8 write 8"` from latent hidden states alone 6.9% of the time, with 34% token-level F1. The random baseline is 0%, confirming the model reads real signal from the injected activations.

### Linear probes confirm the information exists

Linear probes on the latent thought hidden states (GPT-2-small, layer 6) achieve:
- **100%** accuracy classifying which arithmetic step the model is on
- **100%** accuracy predicting the first token of the reasoning step
- **42%** exact match on predicting the full answer

The information is there and linearly separable. The challenge is in the generation pipeline.

## How It Works

### 1. Coconut Model

GPT-2 fine-tuned with a multi-stage curriculum on multi-digit addition:

```
Problem: 347 + 285 =
CoT: 7+5=12 write 2 carry 1 | 4+8+1=13 write 3 carry 1 | 3+2+1=6 write 6
Answer: 632
```

Stage 0 trains with full text CoT. Stages 1-3 progressively replace CoT steps with latent continuous thought vectors — hidden states fed back as inputs instead of being decoded to text. At the "all-latent" stage, the model sees only the problem, performs all reasoning internally through a sequence of hidden state vectors, then produces the answer.

### 2. Activation Collection

From the trained Coconut model, we extract hidden states at each latent thought position from the 50% depth layer (layer 6 for GPT-2-small, layer 18 for GPT-2-large). Each hidden state is paired with the ground-truth CoT text it replaced.

### 3. Self-Oracle

The Coconut model itself is fine-tuned to answer questions about its own activations:

```
Input:  "Layer 18: <act> What is the intermediate calculation at this reasoning step?"
        (with Coconut's layer-18 hidden state injected at <act> via norm-matched addition)
Target: "7+5=12 write 2 carry 1<|endoftext|>"
```

Key design choices:
- **Self-interpretation**: Using the same model (not a separate oracle) since it already understands arithmetic and its own internal representations
- **Layer-matched injection**: Injecting at layer N-1 where activations were extracted from layer N, keeping the signal in the same representational space
- **2x injection scaling**: Doubling the norm-matched signal strength to make the activation more visible to subsequent layers
- **Mixed training**: 70% AO tasks + 30% original text CoT to prevent catastrophic forgetting
- **EOS tokens**: Training targets end with `<|endoftext|>` so the model learns when to stop

### What made the breakthrough

The jump from 0% to 6.9% exact match came from three factors combining:

1. **GPT-2-large (774M)** produces a much better Coconut model (45% all-latent accuracy vs 2.4%), creating richer latent representations
2. **All-latent mode** gives the AO multiple thought-step activations per problem instead of just one, providing more training signal
3. **Self-oracle approach** leverages the model's existing knowledge of arithmetic and its own internal representations

## Project Structure

```
src/
  data_gen.py          # Synthetic arithmetic dataset with carry-propagation CoT
  coconut_model.py     # GPT-2-small with continuous thought support
  activation_oracle.py # Separate-model AO with LoRA (baseline)
  self_oracle.py       # Self-oracle: Coconut model interprets its own activations

scripts/
  01_generate_data.py          # Generate 100K addition problems with CoT
  02_train_coconut.py          # 4-stage curriculum (GPT-2-small)
  03_collect_activations.py    # Extract hidden states from latent thought positions
  04_train_oracle.py           # Train separate AO (baseline)
  05_train_probes.py           # Train linear probe baselines
  06_evaluate.py               # Full evaluation suite
  07_train_self_oracle.py      # Train self-oracle (GPT-2-small)
  08_eval_self_oracle.py       # Evaluate self-oracle
  09_gpt2large_experiment.py   # GPT-2-large, stage 1 only
  10_gpt2large_alllatent.py    # GPT-2-large, all stages through all-latent (best results)
  run_all.sh                   # End-to-end pipeline (scripts 01-06)
```

## Reproduction

```bash
pip install -r requirements.txt

# GPT-2-small base experiment (~3 hours on RTX 4090)
bash scripts/run_all.sh

# GPT-2-small self-oracle (~1.5 hours)
python scripts/07_train_self_oracle.py
python scripts/08_eval_self_oracle.py

# GPT-2-large stage 1 (~8 hours)
python scripts/09_gpt2large_experiment.py

# GPT-2-large all-latent — best results (~16 hours)
python scripts/10_gpt2large_alllatent.py
```

Requires a GPU with >= 16GB VRAM. Tested on NVIDIA RTX 4090 (24GB).

## Models

Pre-trained checkpoints are available on HuggingFace: [syvb/cocoracle](https://huggingface.co/syvb/cocoracle)

- `stage3_alllatent.pt` — GPT-2-large Coconut model, all-latent (45% accuracy)
- `self_oracle_alllatent.pt` — GPT-2-large self-oracle (6.9% CoT exact match)

## What Would Improve This Further

1. **More training**: AO loss was still decreasing at epoch 5 (0.55 and dropping). Training for 20+ epochs would likely push exact match higher.
2. **Larger models**: The AO paper succeeds at 8B+ scale. A Qwen-8B or Llama-8B Coconut model with self-oracle training should produce much better results.
3. **Better Coconut training**: Our all-latent model only achieves 45% accuracy. The original Coconut paper uses more data and longer training. A 90%+ all-latent model would produce much richer latent states.
4. **Multi-layer injection**: Currently we inject at a single layer. Injecting the same activation at multiple layers could strengthen the signal.
5. **Diverse training tasks**: The AO paper uses classification, context prediction, and system prompt QA. We only use arithmetic. More diverse tasks could improve generalization.

## References

- Hao et al., "Training Large Language Models to Reason in a Continuous Latent Space" (2024). [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
- Karvonen et al., "Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers" (2026). [arXiv:2512.15674](https://arxiv.org/abs/2512.15674)
