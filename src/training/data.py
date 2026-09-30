"""Small-slice data loading for the TCL diagnostic run.

Pulls a slice of jasontae/temporal-delta (HF) and formats each row into a
chat-style training example: user asks the question, assistant answers and
appends the gold hedge token. Falls back to the local
data/samples/*.jsonl stress-test files if the HF pull is unavailable, per
the task spec ("doesn't need to be the full dataset, just enough real
examples to prove the gradient path works").
"""
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from .hedge_tokens import HEDGE_TOKENS, HEDGE_TO_CONFIDENCE

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"
LOCAL_TRAIN = Path(__file__).resolve().parents[2] / "data" / "prep" / "temporal_delta" / "temporal_delta_train.jsonl"


@dataclass
class RawExample:
    question: str
    answer: str
    hedge_token: str
    volatility: str  # "fast" | "slow" | "immutable"


def load_hf_slice(n_per_volatility: int = 40, seed: int = 0) -> List[RawExample]:
    """Pull a small, volatility-balanced slice of jasontae/temporal-delta.

    Balanced (not just first-N) because the raw train split is ~60% fast /
    ~40% slow / <0.1% immutable -- an unbalanced small slice would make it
    hard to tell whether L_over/L_under are moving because of the fix or
    because the batch just happens to be mostly one volatility class.
    """
    if LOCAL_TRAIN.exists():
        # Same rows, same order as the HF split (FETCH_ALL.sh step 2 pins it).
        ds = [json.loads(l) for l in open(LOCAL_TRAIN)]
    else:
        from datasets import load_dataset

        ds = load_dataset("jasontae/temporal-delta", split="train")
    by_vol = {"fast": [], "slow": [], "immutable": []}
    for row in ds:
        vol = row["volatility"]
        if vol in by_vol and row["answer"] and row["hedge"] in HEDGE_TOKENS:
            by_vol[vol].append(row)

    rng = random.Random(seed)
    examples: List[RawExample] = []
    for vol, rows in by_vol.items():
        rng.shuffle(rows)
        for row in rows[:n_per_volatility]:
            examples.append(
                RawExample(
                    question=row["question"],
                    answer=row["answer"],
                    hedge_token=row["hedge"],
                    volatility=vol,
                )
            )
    rng.shuffle(examples)
    return examples


def load_local_samples() -> List[RawExample]:
    """Fallback: build examples from data/samples/*.jsonl in this repo.

    stress_stable_facts_sample.jsonl already has question/gold_answer/
    volatility/expected_hedge in the right shape. The mixed-paragraph file
    is claim-based (not directly Q&A) so it's skipped for this fallback --
    the stable-facts file alone is enough for a differentiability check.
    """
    examples: List[RawExample] = []
    path = SAMPLES_DIR / "stress_stable_facts_sample.jsonl"
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            examples.append(
                RawExample(
                    question=row["question"],
                    answer=row["gold_answer"],
                    hedge_token=row["expected_hedge"],
                    volatility=row["volatility"],
                )
            )
    return examples


def load_slice(n_per_volatility: int = 40, seed: int = 0, prefer_hf: bool = True,
               strict: bool = False) -> List[RawExample]:
    """`strict=True` raises instead of falling back to the 49 local samples.

    The fallback exists for the 0.5B gradient diagnostic. A real training run
    that silently fell back would train on the wrong data and still finish.
    """
    if prefer_hf:
        try:
            examples = load_hf_slice(n_per_volatility=n_per_volatility, seed=seed)
            if examples:
                return examples
        except Exception as exc:  # pragma: no cover - network/env dependent
            if strict:
                raise RuntimeError(f"temporal-delta train split unavailable ({exc!r}); "
                                   "run docs/orientation/FETCH_ALL.sh step 2") from exc
            print(f"[data] HF pull failed ({exc!r}), falling back to local samples")
    if strict:
        raise RuntimeError("temporal-delta train split unavailable; run docs/orientation/FETCH_ALL.sh step 2")
    return load_local_samples()


class TCLDataset(torch.utils.data.Dataset):
    """Tokenizes RawExamples into (input_ids, labels, hedge_position, c_gold,
    volatile_mask) tuples ready for the training loop.

    Format: "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer} {hedge}<|im_end|>"
    CE loss is computed only over the assistant turn (labels = -100 elsewhere,
    standard instruction-tuning masking). hedge_position is the index of the
    LAST assistant-content token before the hedge token -- next-token logits
    at that position are what predicts the hedge token, per causal-LM
    convention (logits[i] predicts token[i+1]).
    """

    def __init__(self, examples: List[RawExample], tokenizer, max_len: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tok = self.tokenizer

        prompt_ids = list(tok.apply_chat_template(
            [{"role": "user", "content": ex.question}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        ))
        # answer text followed by a space then the hedge token, so the hedge
        # token is its own final piece of the assistant turn and its
        # predicting position is unambiguous.
        answer_ids = tok(ex.answer + " ", add_special_tokens=False)["input_ids"]
        hedge_ids = tok(ex.hedge_token, add_special_tokens=False)["input_ids"]
        assert len(hedge_ids) == 1, (
            f"hedge token {ex.hedge_token!r} tokenized to {len(hedge_ids)} ids, expected 1 "
            "-- did add_hedge_tokens() run on this tokenizer?"
        )
        eos_ids = [tok.eos_token_id]

        input_ids = prompt_ids + answer_ids + hedge_ids + eos_ids
        labels = (
            [-100] * len(prompt_ids)
            + answer_ids
            + hedge_ids
            + eos_ids
        )
        # position whose next-token logits predict the hedge token
        hedge_position = len(prompt_ids) + len(answer_ids) - 1

        input_ids = input_ids[: self.max_len]
        labels = labels[: self.max_len]
        hedge_position = min(hedge_position, len(input_ids) - 2)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "hedge_position": hedge_position,
            "c_gold": HEDGE_TO_CONFIDENCE[ex.hedge_token],
            "volatile": 1.0 if ex.volatility in ("fast", "slow") else 0.0,
        }


def collate(batch, pad_token_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    hedge_positions = torch.zeros(len(batch), dtype=torch.long)
    c_gold = torch.zeros(len(batch), dtype=torch.float)
    volatile_mask = torch.zeros(len(batch), dtype=torch.float)

    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        attention_mask[i, :n] = 1
        hedge_positions[i] = b["hedge_position"]
        c_gold[i] = b["c_gold"]
        volatile_mask[i] = b["volatile"]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "hedge_positions": hedge_positions,
        "c_gold": c_gold,
        "volatile_mask": volatile_mask,
    }
