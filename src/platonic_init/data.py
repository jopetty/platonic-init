"""Dataset and tokenizer helpers used by both training workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from datasets import Dataset, IterableDataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizer

from .config import InitEvalDataConfig

TEXT_FIELD = "text"

_PAD_TOKEN = "<pad>"
_UNK_TOKEN = "<unk>"
_BOS_TOKEN = "<bos>"
_EOS_TOKEN = "<eos>"
_REFERENCE_PAD_TOKEN = "<|padding|>"


def load_text_dataset(data_path: str) -> Dataset:
    """Load a local plain-text corpus into a Hugging Face dataset."""

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Synthetic data file not found: {path}")
    ds = load_dataset("text", data_files=str(path), split="train")
    if TEXT_FIELD not in ds.column_names:
        raise ValueError(f"Expected '{TEXT_FIELD}' column in dataset")
    return ds


def build_tokenizer(model_name_or_path: str):
    """Load a pretrained tokenizer and ensure it has a pad token."""

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": _REFERENCE_PAD_TOKEN})
    return tokenizer


class CharTokenizer(PreTrainedTokenizer):
    """Minimal character-level tokenizer for synthetic text experiments."""

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab: dict[str, int], **kwargs):
        """Create a tokenizer from an explicit character vocabulary."""

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
        """Return vocabulary size."""

        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        """Return the tokenizer vocabulary."""

        return dict(self.vocab)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize by splitting the input into characters."""

        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Map a token to its integer id with unknown fallback."""

        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """Map an integer id back to its token string."""

        return self.ids_to_tokens.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ):
        """Wrap one or two token sequences with BOS/EOS markers."""

        if token_ids_1 is not None:
            return (
                [self.bos_token_id]
                + token_ids_0
                + [self.eos_token_id]
                + token_ids_1
                + [self.eos_token_id]
            )
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """Return all-zero token type ids for compatibility with HF APIs."""

        if token_ids_1 is not None:
            return [0] * (len(token_ids_0) + len(token_ids_1) + 3)
        return [0] * (len(token_ids_0) + 2)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        """Persist the character vocabulary in the same layout HF expects."""

        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        name = (
            "char_vocab.json"
            if filename_prefix is None
            else f"{filename_prefix}-char_vocab.json"
        )
        out_path = path / name
        out_path.write_text(json.dumps(self.vocab, indent=2), encoding="utf-8")
        return (str(out_path),)


def _iter_text_chars(text_path: str | Path) -> Iterable[str]:
    """Yield all non-newline characters from a text file."""

    with Path(text_path).open("r", encoding="utf-8") as f:
        for line in f:
            for ch in line.rstrip("\n"):
                yield ch


def build_char_tokenizer_from_text(text_path: str | Path) -> CharTokenizer:
    """Build a character tokenizer directly from a corpus file."""

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
    """Load either a saved char tokenizer or a standard pretrained tokenizer."""

    p = Path(path)
    vocab_path = p / "char_vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        return CharTokenizer(vocab=vocab)
    return build_tokenizer(str(p))


def tokenizer_cache_key(tokenizer) -> str:
    """Create a stable cache key for a tokenizer configuration."""

    if isinstance(tokenizer, CharTokenizer):
        payload = json.dumps(tokenizer.get_vocab(), sort_keys=True)
    else:
        payload = json.dumps(
            {
                "name_or_path": getattr(
                    tokenizer, "name_or_path", type(tokenizer).__name__
                ),
                "vocab_size": len(tokenizer),
                "bos": tokenizer.bos_token_id,
                "eos": tokenizer.eos_token_id,
                "pad": tokenizer.pad_token_id,
            },
            sort_keys=True,
        )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def dataset_cache_key(*parts: object) -> str:
    """Create a stable cache key for tokenized dataset artifacts."""

    payload = json.dumps([str(part) for part in parts], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def tokenize_for_clm(
    ds: Dataset | IterableDataset, tokenizer, block_size: int
) -> Dataset | IterableDataset:
    """Tokenize and group a dataset into fixed-size causal-LM blocks."""

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

    grouped = tokenized.map(
        group_texts, batched=True, remove_columns=tokenized.column_names
    )
    grouped = grouped.filter(lambda example: len(example["input_ids"]) > 0)
    column_names = getattr(grouped, "column_names", None)
    extra_columns = [
        name
        for name in column_names or []
        if name not in {"input_ids", "attention_mask", "labels"}
    ]
    if extra_columns:
        grouped = grouped.remove_columns(extra_columns)
    return grouped


def load_or_create_tokenized_dataset(
    ds: Dataset | IterableDataset,
    tokenizer,
    *,
    block_size: int,
    cache_dir: str | Path,
    cache_key: str,
) -> Dataset | IterableDataset:
    """Reuse a cached tokenized dataset when possible, otherwise build it."""

    if isinstance(ds, IterableDataset):
        return tokenize_for_clm(ds, tokenizer, block_size=block_size)

    cache_path = Path(cache_dir) / cache_key
    if cache_path.exists():
        return load_from_disk(str(cache_path))
    tokenized = tokenize_for_clm(ds, tokenizer, block_size=block_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(cache_path))
    return tokenized


def _limit_dataset(ds: Dataset, max_samples: int | None, seed: int) -> Dataset:
    """Optionally subsample a dataset for faster experiments."""

    if max_samples is None:
        return ds
    if len(ds) <= max_samples:
        return ds
    return ds.shuffle(seed=seed).select(range(max_samples))


def _hf_split_with_limit(split: str, max_samples: int | None) -> str:
    """Request only the needed prefix of a HF split when a hard sample cap is known."""

    if max_samples is None:
        return split
    return f"{split}[:{max_samples}]"


def load_init_eval_datasets(
    cfg: InitEvalDataConfig,
    default_local_path: str,
    eval_ratio: float,
    seed: int,
) -> tuple[Dataset | IterableDataset, Dataset]:
    """Load train/eval datasets for downstream initialization evaluation."""

    if cfg.source == "local_text":
        data_path = cfg.local_data_path or default_local_path
        ds = load_text_dataset(data_path)
        split = ds.train_test_split(test_size=eval_ratio, seed=seed)
        train_ds = split["train"]
        eval_ds = split["test"]
    elif cfg.source == "hf":
        stream_train = cfg.max_train_samples is None
        train_ds = load_dataset(
            cfg.dataset_name,
            cfg.dataset_config_name,
            split=_hf_split_with_limit(cfg.train_split, cfg.max_train_samples),
            streaming=stream_train,
        )
        if cfg.eval_split:
            eval_ds = load_dataset(
                cfg.dataset_name,
                cfg.dataset_config_name,
                split=_hf_split_with_limit(cfg.eval_split, cfg.max_eval_samples),
            )
        else:
            split = train_ds.train_test_split(test_size=eval_ratio, seed=seed)
            train_ds = split["train"]
            eval_ds = split["test"]
    else:
        raise ValueError(f"Unsupported init_eval_data.source: {cfg.source}")

    if cfg.text_field != "text":
        if cfg.text_field not in train_ds.column_names:
            raise ValueError(
                f"Text field '{cfg.text_field}' not found in train dataset"
            )
        train_ds = train_ds.rename_column(cfg.text_field, "text")
        if cfg.text_field in eval_ds.column_names:
            eval_ds = eval_ds.rename_column(cfg.text_field, "text")
    if "text" not in train_ds.column_names or "text" not in eval_ds.column_names:
        raise ValueError("Both train/eval datasets must include a 'text' column")

    if not isinstance(train_ds, IterableDataset):
        train_ds = _limit_dataset(train_ds, cfg.max_train_samples, seed=seed)
    eval_ds = _limit_dataset(eval_ds, cfg.max_eval_samples, seed=seed + 1)
    return train_ds, eval_ds
