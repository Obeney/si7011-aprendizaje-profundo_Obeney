"""Tokenización y carga de datos."""

from __future__ import annotations

from datasets import load_dataset

from .utils import HF_DATASET, MAX_LENGTH


def load_tweet_eval():
    return load_dataset(*HF_DATASET)


def make_tokenized_dataset(raw, tokenizer, max_length: int = MAX_LENGTH):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return raw.map(tokenize, batched=True, remove_columns=["text"])
