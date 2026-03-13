#!/usr/bin/env python
"""Warm raw and tokenized init-eval datasets for a config on CPU."""

from __future__ import annotations

import argparse

from platonic_init.config import load_config
from platonic_init.data import (
    build_tokenizer,
    dataset_cache_key,
    load_init_eval_datasets,
    load_or_create_tokenized_dataset,
    tokenizer_cache_key,
)
from platonic_init.support import dataset_cache_root, load_project_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Fallback eval split ratio for local datasets")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic dataset subsampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_project_env()
    cfg = load_config(args.config)

    train_ds, eval_ds = load_init_eval_datasets(
        cfg=cfg.init_eval_data,
        default_local_path=cfg.data_path,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    tokenizer = build_tokenizer(cfg.training.model_name_or_path)
    cache_root = dataset_cache_root(cfg)

    load_or_create_tokenized_dataset(
        train_ds,
        tokenizer,
        block_size=cfg.training.block_size,
        cache_dir=cache_root,
        cache_key=dataset_cache_key(
            "init-eval-train",
            cfg.init_eval_data.source,
            cfg.init_eval_data.dataset_name,
            cfg.init_eval_data.dataset_config_name,
            cfg.init_eval_data.train_split,
            cfg.init_eval_data.text_field,
            cfg.init_eval_data.local_data_path or cfg.data_path,
            cfg.init_eval_data.max_train_samples,
            args.eval_ratio,
            args.seed,
            cfg.training.block_size,
            tokenizer_cache_key(tokenizer),
        ),
    )
    load_or_create_tokenized_dataset(
        eval_ds,
        tokenizer,
        block_size=cfg.training.block_size,
        cache_dir=cache_root,
        cache_key=dataset_cache_key(
            "init-eval-eval",
            cfg.init_eval_data.source,
            cfg.init_eval_data.dataset_name,
            cfg.init_eval_data.dataset_config_name,
            cfg.init_eval_data.eval_split,
            cfg.init_eval_data.text_field,
            cfg.init_eval_data.local_data_path or cfg.data_path,
            cfg.init_eval_data.max_eval_samples,
            args.eval_ratio,
            args.seed,
            cfg.training.block_size,
            tokenizer_cache_key(tokenizer),
        ),
    )

    print(f"Prepared init-eval datasets under: {cache_root}")


if __name__ == "__main__":
    main()
