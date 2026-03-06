from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    model_name_or_path: str = "gpt2"
    block_size: int = 128
    # If null, pre-pretraining trains for one full pass over the dataset.
    max_steps: int | None = None
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    save_steps: int = 100
    logging_steps: int = 10
    pretrain_packing: bool = True
    prepretrain_char_tokenizer: bool = True
    bf16: bool = False
    fp16: bool = False
    report_to: list[str] = field(default_factory=list)
    run_name: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None


@dataclass
class SweepConfig:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    output_root: str = "runs"
    experiment_name: str = "baseline"


@dataclass
class AnalysisConfig:
    top_k_components: int = 8
    max_params_per_tensor: int | None = None
    dtype: str = "float32"


@dataclass
class AnalyticFitConfig:
    basis_type: str = "chebyshev"
    poly_degree: int = 5
    exp_scales: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    chebyshev_degree: int = 12
    fourier_degree: int = 8
    rbf_num_centers: int = 8
    rbf_sigma: float = 0.08


@dataclass
class RebasinConfig:
    enabled: bool = True
    max_iter: int = 50
    seed: int = 0


@dataclass
class InitEvalDataConfig:
    source: str = "hf"  # one of: "hf", "local_text"
    dataset_name: str = "wikitext"
    dataset_config_name: str | None = "wikitext-2-raw-v1"
    train_split: str = "train"
    eval_split: str = "validation"
    text_field: str = "text"
    local_data_path: str | None = None
    max_train_samples: int | None = 10000
    max_eval_samples: int | None = 2000


@dataclass
class ExperimentConfig:
    data_path: str = "data/synthetic.txt"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    analytic_fit: AnalyticFitConfig = field(default_factory=AnalyticFitConfig)
    rebasin: RebasinConfig = field(default_factory=RebasinConfig)
    init_eval_data: InitEvalDataConfig = field(default_factory=InitEvalDataConfig)


def _merge_dataclass(dc_obj: Any, values: dict[str, Any]) -> Any:
    for k, v in values.items():
        current = getattr(dc_obj, k)
        if hasattr(current, "__dataclass_fields__") and isinstance(v, dict):
            _merge_dataclass(current, v)
        else:
            setattr(dc_obj, k, v)
    return dc_obj


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    cfg = ExperimentConfig()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _merge_dataclass(cfg, raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f)
