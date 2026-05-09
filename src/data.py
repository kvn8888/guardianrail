from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer


@dataclass(frozen=True)
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    texts: list[str]


def load_texts(dataset: str, dataset_config: str | None, split: str, text_column: str) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the optional dataset dependency with `pip install datasets`.") from exc

    ds = load_dataset(dataset, dataset_config, split=split)
    texts: list[str] = []
    for item in ds:
        text = item.get(text_column)
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


def tokenize_texts(
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        yield TokenBatch(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            texts=batch_texts,
        )
