"""Synthetic arithmetic dataset with explicit carry-propagation chain-of-thought."""

import random
import torch
from transformers import GPT2Tokenizer


# Special token strings
BOT = "<bot>"
SEP = "<sep>"
EOT = "<eot>"
ACT = "<act>"
SPECIAL_TOKENS = [BOT, SEP, EOT, ACT]


def make_tokenizer():
    """Load GPT-2 tokenizer and add special tokens."""
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tok


def generate_addition_problem(num_digits):
    """Generate a single addition problem with CoT steps.

    Returns:
        problem_str: e.g. "347 + 285 ="
        cot_steps: list of step strings, e.g. ["7+5=12 write 2 carry 1", ...]
        answer_str: e.g. "632"
    """
    lo = 10 ** (num_digits - 1)
    hi = 10**num_digits - 1
    a = random.randint(lo, hi)
    b = random.randint(lo, hi)
    result = a + b

    digits_a = [int(d) for d in str(a)][::-1]  # least significant first
    digits_b = [int(d) for d in str(b)][::-1]

    carry = 0
    cot_steps = []
    for i in range(max(len(digits_a), len(digits_b))):
        da = digits_a[i] if i < len(digits_a) else 0
        db = digits_b[i] if i < len(digits_b) else 0
        total = da + db + carry
        digit = total % 10
        new_carry = total // 10

        if carry > 0:
            step = f"{da}+{db}+{carry}={total} write {digit}"
        else:
            step = f"{da}+{db}={total} write {digit}"

        if new_carry > 0:
            step += f" carry {new_carry}"
        carry = new_carry
        cot_steps.append(step)

    # Handle final carry
    if carry > 0:
        cot_steps.append(f"carry {carry} write {carry}")

    problem_str = f"{a} + {b} ="
    answer_str = str(result)
    return problem_str, cot_steps, answer_str


def format_full_example(problem_str, cot_steps, answer_str):
    """Format a complete training example with text CoT."""
    cot_text = f" {SEP} ".join(cot_steps)
    return f"{problem_str} {BOT} {cot_text} {EOT} {answer_str}"


def generate_dataset(n, digit_dist=None, seed=42):
    """Generate n addition problems.

    Args:
        n: number of problems
        digit_dist: dict mapping num_digits -> probability (default: 30% 2-digit, 40% 3-digit, 30% 4-digit)

    Returns:
        list of (problem_str, cot_steps, answer_str) tuples
    """
    if digit_dist is None:
        digit_dist = {2: 0.3, 3: 0.4, 4: 0.3}

    rng = random.Random(seed)
    digits_choices = list(digit_dist.keys())
    digits_weights = list(digit_dist.values())

    data = []
    for _ in range(n):
        nd = rng.choices(digits_choices, weights=digits_weights, k=1)[0]
        # Use the rng for reproducibility
        old_state = random.getstate()
        random.setstate(rng.getstate())
        problem_str, cot_steps, answer_str = generate_addition_problem(nd)
        rng.setstate(random.getstate())
        random.setstate(old_state)
        data.append((problem_str, cot_steps, answer_str))
    return data


def tokenize_dataset(data, tokenizer, max_len=256):
    """Tokenize dataset into tensors.

    Returns dict with:
        - input_ids: (N, max_len) padded token IDs
        - labels: (N, max_len) token IDs for loss (-100 for padding)
        - lengths: (N,) actual lengths
        - problems: list of problem strings
        - cot_steps: list of lists of CoT step strings
        - answers: list of answer strings
    """
    all_ids = []
    all_labels = []
    all_lengths = []
    problems = []
    cot_steps_list = []
    answers = []

    for problem_str, cot_steps, answer_str in data:
        full_text = format_full_example(problem_str, cot_steps, answer_str)
        ids = tokenizer.encode(full_text, add_special_tokens=False)

        if len(ids) > max_len:
            ids = ids[:max_len]

        length = len(ids)
        # Pad
        padded = ids + [tokenizer.pad_token_id] * (max_len - length)
        labels = ids + [-100] * (max_len - length)

        all_ids.append(padded)
        all_labels.append(labels)
        all_lengths.append(length)
        problems.append(problem_str)
        cot_steps_list.append(cot_steps)
        answers.append(answer_str)

    return {
        "input_ids": torch.tensor(all_ids, dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "lengths": torch.tensor(all_lengths, dtype=torch.long),
        "problems": problems,
        "cot_steps": cot_steps_list,
        "answers": answers,
    }


if __name__ == "__main__":
    # Quick test
    tok = make_tokenizer()
    data = generate_dataset(5, seed=0)
    for p, c, a in data:
        print(f"Problem: {p}")
        print(f"CoT: {c}")
        print(f"Answer: {a}")
        print(f"Full: {format_full_example(p, c, a)}")
        print()
