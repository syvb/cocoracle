---
license: mit
tags:
  - coconut
  - activation-oracle
  - interpretability
  - chain-of-thought
  - latent-reasoning
  - gpt2
datasets:
  - synthetic
language:
  - en
base_model:
  - openai-community/gpt2-large
pipeline_tag: text-generation
---

# Cocoracle: Activation Oracles for Coconut Latent Reasoning

Checkpoints from the [Cocoracle](https://github.com/syvb/cocoracle) experiment -- interpreting what a model "thinks" during latent reasoning.

Combines [Coconut](https://arxiv.org/abs/2412.06769) (Chain of Continuous Thought) with [Activation Oracles](https://arxiv.org/abs/2512.15674) to train models that answer natural-language questions about their own latent chain-of-thought hidden states.

## Models

### GPT-2-large Coconut (all-latent)

**`stage3_alllatent.pt`** -- GPT-2-large (774M) fine-tuned with the Coconut curriculum to perform multi-digit addition using entirely latent reasoning.

- **Task**: Multi-digit addition (2-4 digits) with carry propagation
- **Accuracy**: 45.4% (teacher-forced) on all-latent reasoning
- **Architecture**: GPT-2-large + 4 special tokens (`<bot>`, `<sep>`, `<eot>`, `<act>`)

### GPT-2-large Self-Oracle (all-latent)

**`self_oracle_alllatent.pt`** -- The Coconut model further fine-tuned to interpret its own latent reasoning activations via norm-matched injection at layer 17.

- **CoT exact match**: 6.9%
- **CoT token F1**: 34.2%
- **Random baseline**: 0%

## Usage

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<bot>", "<sep>", "<eot>", "<act>"]
})

model = GPT2LMHeadModel.from_pretrained("gpt2-large")
model.resize_token_embeddings(len(tokenizer))
state = torch.load("stage3_alllatent.pt", map_location="cpu")
model.load_state_dict(state)
```

See the [GitHub repo](https://github.com/syvb/cocoracle) for full code and an interactive demo (`scripts/interactive.py`).

## Results

| Configuration | CoT Exact Match | CoT Token F1 | AO Val Loss |
|--------------|----------------|--------------|-------------|
| Separate AO (GPT-2-small + LoRA) | 0% | 26.4% | 2.92 |
| Self-oracle, GPT-2-small | 0% | 32.5% | 1.98 |
| Self-oracle, GPT-2-large, stage 1 | 0% | 25.6% | 1.10 |
| **Self-oracle, GPT-2-large, all-latent** | **6.9%** | **34.2%** | **0.55** |

## References

- Hao et al., [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
- Karvonen et al., [arXiv:2512.15674](https://arxiv.org/abs/2512.15674)
