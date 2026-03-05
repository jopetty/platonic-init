from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from .config import InitEvalDataConfig


TEXT_FIELD = "text"

_PAD_TOKEN = "<pad>"
_UNK_TOKEN = "<unk>"
_BOS_TOKEN = "<bos>"
_EOS_TOKEN = "<eos>"


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


class CharTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab: dict[str, int], **kwargs):
        self.vocab = dict(vocab)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        super().__init__(
            pad_token=_PAD_TOKEN,
            unk_token=_UNK_TOKEN,
            bos_token=_BOS_TOKEN,
            eos_token=_EOS_TOKEN,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def _tokenize(self, text: str) -> list[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def build_inputs_with_special_tokens(self, token_ids_0: list[int], token_ids_1: list[int] | None = None):
        if token_ids_1 is not None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        if token_ids_1 is not None:
            return [0] * (len(token_ids_0) + len(token_ids_1) + 3)
        return [0] * (len(token_ids_0) + 2)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        name = "char_vocab.json" if filename_prefix is None else f"{filename_prefix}-char_vocab.json"
        out_path = path / name
        out_path.write_text(json.dumps(self.vocab, indent=2), encoding="utf-8")
        return (str(out_path),)


def _iter_text_chars(text_path: str | Path) -> Iterable[str]:
    with Path(text_path).open("r", encoding="utf-8") as f:
        for line in f:
            for ch in line.rstrip("\n"):
                yield ch


def build_char_tokenizer_from_text(text_path: str | Path) -> CharTokenizer:
    vocab = {
        _PAD_TOKEN: 0,
        _UNK_TOKEN: 1,
        _BOS_TOKEN: 2,
        _EOS_TOKEN: 3,
    }
    for ch in sorted(set(_iter_text_chars(text_path))):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return CharTokenizer(vocab=vocab)


def load_saved_tokenizer(path: str | Path):
    p = Path(path)
    vocab_path = p / "char_vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        return CharTokenizer(vocab=vocab)
    return build_tokenizer(str(p))


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
