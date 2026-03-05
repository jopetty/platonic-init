from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


TEXT_FIELD = "text"


def load_text_dataset(data_path: str) -> Dataset:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Synthetic data file not found: {path}")
    ds = load_dataset("text", data_files=str(path), split="train")
    if TEXT_FIELD not in ds.column_names:
        raise ValueError(f"Expected '{TEXT_FIELD}' column in dataset")
    return ds


def build_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_for_clm(ds: Dataset, tokenizer, block_size: int) -> Dataset:
    def tokenize_batch(batch):
        return tokenizer(batch[TEXT_FIELD], truncation=False)

    tokenized = ds.map(tokenize_batch, batched=True, remove_columns=ds.column_names)

    def group_texts(batch):
        concatenated = []
        for ids in batch["input_ids"]:
            concatenated.extend(ids)
        total = len(concatenated)
        usable = (total // block_size) * block_size
        if usable == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids = [
            concatenated[i : i + block_size] for i in range(0, usable, block_size)
        ]
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * block_size for _ in input_ids],
            "labels": [chunk[:] for chunk in input_ids],
        }

    grouped = tokenized.map(group_texts, batched=True)
    grouped = grouped.filter(lambda example: len(example["input_ids"]) > 0)
    return grouped
