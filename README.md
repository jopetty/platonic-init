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
5. Compare `random` vs `platonic_mean` vs `platonic_sampled` initializations under the same downstream optimization budget.

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
  - `poly`
  - `exp`
  - `poly_exp`
- Keep coefficients only, not dense vectors, yielding a compact initialization function.
- Track relative reconstruction error per tensor/component.

Outputs:
- `artifacts/analytic_subspace.pt`
- `artifacts/analytic_fit_report.json`

### 4) Initialization Evaluation
- Compare three variants with equal training budget:
  - `random`
  - `platonic_mean` (deterministic)
  - `platonic_sampled` (mean + sampled latent in learned subspace)
- Evaluate early loss/perplexity and convergence speed.

Output:
- `artifacts/init_eval.json`

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

- `src/platonic_init/train.py`: multi-seed pre-pretraining with TRL `SFTTrainer`
- `src/platonic_init/analyze.py`: tensorwise shared-subspace extraction
- `src/platonic_init/analytic.py`: analytic basis fitting for PCA components
- `src/platonic_init/init_fn.py`: closed-form initializer construction/application
- `src/platonic_init/eval_init.py`: random vs platonic init comparison
- `src/platonic_init/pipeline.py`: end-to-end orchestration
- `configs/experiment.yaml`: main experiment config
- `scripts/*.sh`: convenience wrappers

## Quickstart (uv)

1. Install deps:
```bash
uv sync
```

2. Put your fixed synthetic corpus at:
```text
data/synthetic.txt
```

Optional: generate a Dyck dataset with power-law depth sampling:
```bash
uv run python scripts/generate_dyck.py --n-samples 5000 --max-depth 10 --alpha 1.5 --output data/dyck_d10_5k.txt
```

3. Run full pipeline:
```bash
./scripts/run_pipeline.sh configs/experiment.yaml
```

Or run stages manually:
```bash
./scripts/train_sweep.sh configs/experiment.yaml
./scripts/analyze_subspace.sh configs/experiment.yaml
./scripts/fit_analytic.sh configs/experiment.yaml
uv run python -m platonic_init.eval_init --config configs/experiment.yaml
```

## Notes / Current Constraints

- Current subspace extraction assumes identical tensor names/shapes across checkpoints.
- Cross-architecture sharing (different shapes) is intentionally handled as separate cohorts rather than forced alignment.
- `analysis.max_params_per_tensor` should remain `null` if you want to reconstruct full tensors for initialization.

## Suggested Next Extensions

- Add permutation/orthogonal alignment before PCA for stronger cross-seed matching.
- Add token-budget matched downstream benchmarks beyond the synthetic corpus.
- Add multiple analytic basis families (e.g., rational/Chebyshev/Fourier) with AIC/BIC-based model selection.
