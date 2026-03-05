from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from .config import InitEvalDataConfig


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


def _limit_dataset(ds: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None:
        return ds
    if len(ds) <= max_samples:
        return ds
    return ds.shuffle(seed=seed).select(range(max_samples))


def load_init_eval_datasets(
    cfg: InitEvalDataConfig,
    default_local_path: str,
    eval_ratio: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    if cfg.source == "local_text":
        data_path = cfg.local_data_path or default_local_path
        ds = load_text_dataset(data_path)
        split = ds.train_test_split(test_size=eval_ratio, seed=seed)
        train_ds = split["train"]
        eval_ds = split["test"]
    elif cfg.source == "hf":
        train_ds = load_dataset(
            cfg.dataset_name,
            cfg.dataset_config_name,
            split=cfg.train_split,
        )
        if cfg.eval_split:
            eval_ds = load_dataset(
                cfg.dataset_name,
                cfg.dataset_config_name,
                split=cfg.eval_split,
            )
        else:
            split = train_ds.train_test_split(test_size=eval_ratio, seed=seed)
            train_ds = split["train"]
            eval_ds = split["test"]
    else:
        raise ValueError(f"Unsupported init_eval_data.source: {cfg.source}")

    if cfg.text_field != "text":
        if cfg.text_field not in train_ds.column_names:
            raise ValueError(f"Text field '{cfg.text_field}' not found in train dataset")
        train_ds = train_ds.rename_column(cfg.text_field, "text")
        if cfg.text_field in eval_ds.column_names:
            eval_ds = eval_ds.rename_column(cfg.text_field, "text")
    if "text" not in train_ds.column_names or "text" not in eval_ds.column_names:
        raise ValueError("Both train/eval datasets must include a 'text' column")

    train_ds = _limit_dataset(train_ds, cfg.max_train_samples, seed=seed)
    eval_ds = _limit_dataset(eval_ds, cfg.max_eval_samples, seed=seed + 1)
    return train_ds, eval_ds
