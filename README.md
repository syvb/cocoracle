# Cocoracle

**Can we interpret what a model is "thinking" during latent reasoning?**

This project combines two ideas:
- [**Coconut**](https://arxiv.org/abs/2412.06769) (Chain of Continuous Thought): LLMs that reason in latent space instead of generating text chain-of-thought tokens
- [**Activation Oracles**](https://arxiv.org/abs/2512.15674) (AOs): Models trained to answer natural-language questions about another model's internal activations

We train a Coconut model (GPT-2, 124M params) that does arithmetic with latent reasoning steps, then try to build an oracle that can look at those hidden reasoning states and tell us what the model is computing.

## Key Results

### The information is there

Linear probes on the latent thought hidden states achieve:
- **100%** accuracy classifying which arithmetic step the model is on
- **100%** accuracy predicting the first token of the reasoning step
- **42%** exact match on predicting the full answer

This proves the Coconut model's latent states encode rich, structured information about the computation being performed.

### Natural-language interpretation is hard at small scale

We tried three approaches to produce natural-language descriptions of latent states:

| Approach | CoT Token F1 | AO Val Loss | Notes |
|----------|-------------|-------------|-------|
| Separate AO (GPT-2 + LoRA) | 26.4% | 2.92 | Mode collapse: same output for every input |
| Self-oracle v1 (same model, layer 1) | 32.5% | 1.98 | Varied outputs, no collapse, no forgetting |
| **Self-oracle v2 (layer 5, 2x signal)** | **22.3%** | **1.10** | Short CoT-formatted outputs, 17.7% correct token sets |

The self-oracle v2 has the lowest loss and generates structured arithmetic text conditioned on the input activation, but exact match remains ~0% at GPT-2-small scale. The model gets the right *type* of step (carry vs addition) and often the right *tokens* (17.7% token-set match for carry steps), but can't reliably decode specific digits.

### Why the gap?

The AO paper's smallest successful model is 8B parameters. At 124M, GPT-2-small can learn to *read* the injection signal (proven by the low loss and structured outputs), but doesn't have enough capacity to precisely *decode* a 768-dimensional activation vector into the exact correct CoT text through 6 layers of processing.

## How It Works

### 1. Coconut Model

GPT-2-small fine-tuned with a 4-stage curriculum on multi-digit addition:

```
Problem: 347 + 285 =
CoT: 7+5=12 write 2 carry 1 | 4+8+1=13 write 3 carry 1 | 3+2+1=6 write 6
Answer: 632
```

Stage 0 trains with full text CoT, then stages 1-3 progressively replace CoT steps with latent continuous thought vectors (hidden states fed back as inputs instead of decoded to text).

**Important caveat:** Our Coconut model is not very good. Stage 1 (only the last step latent) achieves 69% teacher-forced accuracy, which is reasonable. But the model degrades sharply as more steps become latent: Stage 2 drops to 19%, and Stage 3 (full replacement, all steps latent) manages only 2.4%. This means the all-latent model barely works, and even Stage 1 is far from reliable. A proper Coconut implementation would need significantly more training, larger models, and the full curriculum from the original paper. Our AO results are therefore probing activations from a weak reasoner — a stronger Coconut model would likely produce richer, more interpretable latent states.

### 2. Activation Collection

From the trained Coconut model, we extract hidden states at each latent thought position from layers 3, 6, and 9 (25/50/75% depth). Each hidden state is paired with the ground-truth CoT text it replaced.

### 3. Self-Oracle

The Coconut model itself is fine-tuned to also answer questions about its own activations:

```
Input:  "Layer 6: <act> What is the intermediate calculation at this reasoning step?"
        (with Coconut's layer-6 hidden state injected at <act> via norm-matched addition)
Target: "7+5=12 write 2 carry 1"
```

Training mixes 70% AO tasks with 30% original CoT data to prevent catastrophic forgetting. The injection happens at layer 5 (one before the extraction layer) with 2x signal scaling.

## Project Structure

```
src/
  data_gen.py          # Synthetic arithmetic dataset with carry-propagation CoT
  coconut_model.py     # GPT-2 with continuous thought (latent reasoning) support
  activation_oracle.py # Separate-model AO with LoRA (baseline)
  self_oracle.py       # Self-oracle: Coconut model interprets its own activations

scripts/
  01_generate_data.py       # Generate 100K addition problems with CoT
  02_train_coconut.py       # 4-stage curriculum training
  03_collect_activations.py # Extract hidden states from latent thought positions
  04_train_oracle.py        # Train separate AO (baseline)
  05_train_probes.py        # Train linear probe baselines
  06_evaluate.py            # Full evaluation suite
  07_train_self_oracle.py   # Train self-oracle (best approach)
  08_eval_self_oracle.py    # Evaluate self-oracle
  run_all.sh                # End-to-end pipeline (scripts 01-06)
```

## Reproduction

```bash
pip install -r requirements.txt

# Base experiment: Coconut + separate AO + probes (~3 hours on RTX 4090)
bash scripts/run_all.sh

# Self-oracle experiment (~1.5 hours additional)
python scripts/07_train_self_oracle.py
python scripts/08_eval_self_oracle.py
```

Requires a GPU with >= 16GB VRAM. Tested on NVIDIA RTX 4090 (24GB).

## What Would Make This Work

The experiment demonstrates the concept is viable but needs more scale:

1. **GPT-2-medium/large (355M-774M)**: Drop-in replacement, no code changes needed. 3-6x more capacity should cross the threshold for precise digit decoding.
2. **Longer training**: AO loss was still decreasing at epoch 5. 20-50 epochs would help.
3. **Larger Coconut model**: Our all-latent stage only gets 2.4% accuracy. A properly trained Coconut model (more data, larger model) would produce richer latent representations.
4. **At 8B+ scale**: The AO paper shows this works. A Qwen-8B or Llama-8B Coconut model with self-oracle training should produce human-readable descriptions of latent reasoning steps.

## References

- Hao et al., "Training Large Language Models to Reason in a Continuous Latent Space" (2024). [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
- Karvonen et al., "Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers" (2026). [arXiv:2512.15674](https://arxiv.org/abs/2512.15674)
