"""Experiment configuration dataclasses and YAML loading helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    model_name_or_path: str = "gpt2"
    block_size: int = 2048
    # If null, pre-pretraining trains for one full pass over the dataset.
    max_steps: int | None = None
    reference_effective_batch_size: int | None = 8
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    warmup_steps: int | None = 500
    warmup_ratio: float = 0.03
    min_lr_rate: float = 0.1
    optimizer_type: str = "adamw"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    muon_learning_rate: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_nesterov: bool = True
    save_steps: int = 100
    logging_steps: int = 10
    pretrain_packing: bool = True
    prepretrain_char_tokenizer: bool = False
    bf16: bool = True
    fp16: bool = False
    prefer_flash_attention_2: bool = True
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
class AnalyticFitBlockConfig(AnalyticFitConfig):
    name: str = "chebyshev"

    def to_fit_config(self) -> AnalyticFitConfig:
        return AnalyticFitConfig(
            basis_type=self.basis_type,
            poly_degree=self.poly_degree,
            exp_scales=list(self.exp_scales),
            chebyshev_degree=self.chebyshev_degree,
            fourier_degree=self.fourier_degree,
            rbf_num_centers=self.rbf_num_centers,
            rbf_sigma=self.rbf_sigma,
        )


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
class PrepretrainStageConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)


@dataclass
class FitInitializationsStageConfig:
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    rebasin: RebasinConfig = field(default_factory=RebasinConfig)
    reference_init_seed: int | None = None
    fit_blocks: list[AnalyticFitBlockConfig] = field(default_factory=list)


@dataclass
class PretrainEvalStageConfig:
    init_eval_data: InitEvalDataConfig = field(default_factory=InitEvalDataConfig)
    train_steps: int = 10000
    reference_effective_batch_size: int | None = 8
    eval_every: int = 100
    logging_steps: int = 10


@dataclass
class StagesConfig:
    prepretrain: PrepretrainStageConfig = field(default_factory=PrepretrainStageConfig)
    fit_initializations: FitInitializationsStageConfig = field(
        default_factory=FitInitializationsStageConfig
    )
    pretrain_eval: PretrainEvalStageConfig = field(
        default_factory=PretrainEvalStageConfig
    )


@dataclass
class ExperimentConfig:
    data_path: str = "data/synthetic.txt"
    stages: StagesConfig = field(default_factory=StagesConfig)

    @property
    def training(self) -> TrainingConfig:
        return self.stages.prepretrain.training

    @property
    def sweep(self) -> SweepConfig:
        return self.stages.prepretrain.sweep

    @property
    def analysis(self) -> AnalysisConfig:
        return self.stages.fit_initializations.analysis

    @property
    def rebasin(self) -> RebasinConfig:
        return self.stages.fit_initializations.rebasin

    @property
    def init_eval_data(self) -> InitEvalDataConfig:
        return self.stages.pretrain_eval.init_eval_data

    @property
    def fit_blocks(self) -> list[AnalyticFitBlockConfig]:
        return self.stages.fit_initializations.fit_blocks


def _merge_dataclass(dc_obj: Any, values: dict[str, Any]) -> Any:
    """Recursively overlay plain dictionaries onto nested dataclass instances."""

    for k, v in values.items():
        current = getattr(dc_obj, k)
        if hasattr(current, "__dataclass_fields__") and isinstance(v, dict):
            _merge_dataclass(current, v)
        else:
            setattr(dc_obj, k, v)
    return dc_obj


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from YAML and normalize stage-specific blocks."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "stages" not in raw:
        raise ValueError(f"Config must define a top-level 'stages' mapping: {path}")
    cfg = _merge_dataclass(ExperimentConfig(), raw)
    return _normalize_config(cfg)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Serialize an experiment config back to YAML."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f)


def _normalize_fit_blocks(
    items: list[Any], *, prefix: str = "fit"
) -> list[AnalyticFitBlockConfig]:
    """Normalize fit-block entries into `AnalyticFitBlockConfig` objects."""

    normalized_blocks: list[AnalyticFitBlockConfig] = []
    for i, block in enumerate(items):
        if isinstance(block, AnalyticFitBlockConfig):
            normalized_blocks.append(block)
            continue
        if isinstance(block, dict):
            merged = _merge_dataclass(AnalyticFitBlockConfig(), block)
            if not merged.name:
                merged.name = f"{prefix}_{i}"
            normalized_blocks.append(merged)
            continue
        raise TypeError(f"Invalid fit_blocks entry at index {i}: {type(block)!r}")
    return normalized_blocks


def _normalize_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Apply post-load config normalization rules."""

    cfg.training.optimizer_type = str(cfg.training.optimizer_type).lower()
    if cfg.training.optimizer_type not in {"adamw", "muon"}:
        raise ValueError(
            "stages.prepretrain.training.optimizer_type must be one of: "
            "'adamw', 'muon'"
        )
    cfg.stages.fit_initializations.fit_blocks = _normalize_fit_blocks(
        cfg.stages.fit_initializations.fit_blocks
    )
    return cfg
