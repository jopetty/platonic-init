# Platonic Initialization

This repository implements an experimental pipeline for testing whether the *training trajectory* of synthetic pre-pretraining is necessary, or whether a shared low-dimensional weight-space structure can serve as a direct initializer.

The motivating setup follows your pre-pretraining framing in [arXiv:2502.19249](https://arxiv.org/abs/2502.19249): train on a fixed synthetic text distribution, then transfer to broader tasks.

## Final Thinking (Hypothesis + Strategy)

Core hypothesis:
- If synthetic pre-pretraining is simple relative to model capacity, independent runs should converge to a narrow, shared subspace.
- That shared subspace can be approximated by a very low-dimensional analytic representation.
- Initializing from this representation (without running synthetic pre-pretraining each time) should outperform random initialization on early optimization and sample-efficiency metrics.

Operationalized approach:
1. Train many identical models (same architecture/data/hparams, different random seeds).
2. Analyze checkpoints tensor-by-tensor to extract a shared core: mean weight vector + top principal directions.
3. Fit those principal directions with low-dimensional analytic bases (polynomial, exponential, or combined).
4. Build a closed-form initializer from the fitted representation.
5. Compare `random` vs `weight_transfer` vs basis-derived analytic initializations under the same downstream optimization budget.

## Experimental Plan

### 1) Controlled Seed Sweep (Fixed Shape)
- Fix one model architecture and one synthetic corpus (`data/synthetic.txt`).
- Train `N` seed replicas (`N >= 8` recommended).
- Keep optimizer schedule and token budget fixed.

Outputs:
- `runs/<experiment>/<model>/seed_<k>/` checkpoints

### 2) Shared Core Weight-Space Estimation
- For each tensor name shared across checkpoints, flatten and stack across seeds.
- Compute:
  - Tensor mean (`mu_t`) as a candidate deterministic initializer.
  - Top-`k` principal axes (`U_t`) for low-dimensional variation.
  - Explained variance ratios to quantify how low-rank the synthetic convergence manifold is.

Outputs:
- `artifacts/weight_subspace.pt`
- `artifacts/weight_subspace_summary.json`

### 3) Analytic Compression
- Fit each principal axis with analytic bases over parameter index:
  - `chebyshev` (recommended default)
  - `fourier`
  - `rbf`
  - `poly_exp` (legacy baseline)
- Keep coefficients only, not dense vectors, yielding a compact initialization function.
- Track relative reconstruction error per tensor/component.

Outputs:
- `artifacts/experiments/<experiment>/analysis/basis_sweep/analytic_subspace_<fit>.pt`
- `artifacts/experiments/<experiment>/analysis/basis_sweep/analytic_fit_report_<fit>.json`

### 4) Initialization Evaluation
- Compare variants with equal training budget:
  - `random`
  - basis-derived analytic inits (`chebyshev`, `fourier`, `rbf`, `poly_exp`)
  - `weight_transfer` from a pre-pretrained seed
- Evaluate early loss/perplexity and convergence speed on a standard LM dataset
  configured under `init_eval_data` (defaults to WikiText-2 in this repo).

Output:
- `artifacts/experiments/<experiment>/pretraining/init_eval.json`

### 5) Model Size and Shape Accounting
Because tensor alignment requires matching shapes, run experiments in **shape-homogeneous cohorts**:
- Cohort A: small model shape
- Cohort B: medium model shape
- Cohort C: larger model shape

For each cohort, run stages 1-4 independently, then compare:
- Subspace dimensionality needed for fixed explained variance.
- Analytic-fit error at equal basis size.
- Transfer gains from platonic initialization at equal compute.

## Code Layout

- `src/platonic_init/pipeline.py`: single CLI entrypoint plus stage orchestration and job selection
- `src/platonic_init/training.py`: pre-pretraining, downstream init-eval runs, and shared model/trainer runtime helpers
- `src/platonic_init/initialization.py`: checkpoint loading, tensorwise PCA, analytic basis fitting, and platonic init reconstruction
- `src/platonic_init/data.py`: dataset and tokenizer helpers
- `src/platonic_init/config.py`: experiment config dataclasses and YAML loading
- `src/platonic_init/rebasin.py`: permutation alignment logic for cross-seed checkpoint analysis
- `src/platonic_init/support.py`: environment loading and artifact-path helpers
- `configs/experiment.yaml`: main experiment config
- `scripts/*.sh`: convenience wrappers

## Quickstart (uv)

1. Install deps:
```bash
uv sync
```

For notebook support in the project venv:
```bash
uv sync --extra notebook
uv run python -m ipykernel install --user --name platonic-init --display-name "Python (platonic-init)"
uv run jupyter lab
```

Weights & Biases logging is configured via `stages.prepretrain.training.report_to` in config files.
Before runs with W&B enabled, authenticate once:
```bash
uv run wandb login
```

All CLI entrypoints automatically load `.env` from the repository root (if present),
so API keys can be provided there for W&B / Hugging Face / other SDKs.
Example `.env`:
```bash
WANDB_API_KEY=...
WANDB_ENTITY=...
HF_TOKEN=...
HUGGINGFACE_HUB_TOKEN=...
```

2. Put your fixed synthetic corpus at:
```text
data/synthetic.txt
```

Optional: generate a Dyck dataset with power-law depth sampling:
```bash
uv run python scripts/generate_dyck.py --n-samples 5000 --max-depth 10 --alpha 1.5 --output data/dyck_d10_5k.txt
```

For paper-style formal-language corpora, use the generalized generator:
```bash
uv run python scripts/generate_formal_language.py --language shuffle_dyck --k 64 --n-samples 20000 --max-depth 10 --output data/shuffle_dyck_k64_d10_20k.txt
uv run python scripts/generate_formal_language.py --language dyck --k 64 --n-samples 20000 --max-depth 10 --output data/dyck_k64_d10_20k.txt
uv run python scripts/generate_formal_language.py --language ww --ww-alphabet-size 64 --n-samples 20000 --output data/ww_k64_20k.txt
```

Demo config for a fast 2-seed Dyck run:
```bash
./scripts/run_pipeline.sh configs/experiment_dyck_d10_20k_demo.yaml
```
This demo uses `sshleifer/tiny-gpt2` to keep runtime low.

3. Run full pipeline:
```bash
./scripts/run_pipeline.sh configs/experiment.yaml
```

Or run modular stages with the unified pipeline:
```bash
uv run python -m platonic_init.pipeline --config configs/experiment.yaml --stages prepretrain
uv run python -m platonic_init.pipeline --config configs/experiment.yaml --stages fit_initializations
uv run python -m platonic_init.pipeline --config configs/experiment.yaml --stages pretrain
```

Resume after pre-pretraining:
```bash
uv run python -m platonic_init.pipeline --config configs/experiment.yaml --stages fit_initializations pretrain
```

Validate stage prerequisites without running:
```bash
uv run python -m platonic_init.pipeline --config configs/experiment.yaml --stages pretrain --doctor
```

For the tiny-GPT2 paper-replication proxy on C4 using direct checkpoint transfer:
```bash
uv run python scripts/generate_formal_language.py --language shuffle_dyck --k 64 --n-samples 20000 --max-depth 10 --output data/shuffle_dyck_k64_d10_20k.txt
uv run python -m platonic_init.pipeline --config configs/gpt2_tiny_c4_ppt_reproduction.yaml --stages prepretrain
uv run python -m platonic_init.pipeline --config configs/gpt2_tiny_c4_ppt_reproduction.yaml --stages pretrain --skip-fits
```
This transfer-only path now loads the selected pre-pretrained checkpoint directly for `weight_transfer`; it does not require analytic-fit artifacts.

For Torch cluster runs, use the size-specific submit wrappers instead of setting
`CONFIG_PATH` by hand:
```bash
scripts/submit_gpt2_tiny.sh prepretrain
scripts/submit_gpt2_tiny.sh pretrain
scripts/submit_gpt2_tiny.sh pretrain-fits

scripts/submit_gpt2_medium.sh prepretrain
scripts/submit_gpt2_medium.sh pretrain
scripts/submit_gpt2_medium.sh pretrain-fits
```
Available wrappers:
- `scripts/submit_gpt2_tiny.sh`
- `scripts/submit_gpt2.sh`
- `scripts/submit_gpt2_medium.sh`
- `scripts/submit_gpt2_large.sh`
- `scripts/submit_gpt2_xl.sh`

Each wrapper accepts the stage name first, followed by optional extra `sbatch`
args. Example:
```bash
scripts/submit_gpt2_medium.sh pretrain-fits --export=FIT_NAMES=chebyshev_d24
```

Configs are stage-scoped only. The canonical shape is:
```yaml
stages:
  prepretrain: ...
  fit_initializations: ...
  pretrain_eval: ...
```

## Notes / Current Constraints

- Current subspace extraction assumes identical tensor names/shapes across checkpoints.
- Cross-architecture sharing (different shapes) is intentionally handled as separate cohorts rather than forced alignment.
- `analysis.max_params_per_tensor` should remain `null` if you want to reconstruct full tensors for initialization.
- Recommended analytic family sweep order:
  1. `chebyshev` (stable orthogonal baseline for smooth global structure)
  2. `fourier` (good for low-frequency / periodic-ish structure)
  3. `rbf` (good for localized structure)
  4. `poly_exp` (legacy baseline from earlier experiments)
- `fit_blocks[*].basis_type` currently supports:
  - `chebyshev`: orthogonal polynomial basis on normalized index coordinates.
  - `fourier`: truncated sin/cos basis for smooth periodic-ish structure.
  - `rbf`: Gaussian radial basis functions for localized structure.
  - `poly`, `exp`, `poly_exp`: legacy options for backward compatibility.

## Suggested Next Extensions

- Add permutation/orthogonal alignment before PCA for stronger cross-seed matching.
- Add token-budget matched downstream benchmarks beyond the synthetic corpus.
- Add multiple analytic basis families (e.g., rational/Chebyshev/Fourier) with AIC/BIC-based model selection.
