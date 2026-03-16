from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from datasets import Dataset

from platonic_init.config import AnalyticFitBlockConfig, ExperimentConfig, load_config
from platonic_init.data import (
    CharTokenizer,
    dataset_cache_key,
    load_or_create_tokenized_dataset,
    tokenizer_cache_key,
)
from platonic_init.pipeline import (
    _doctor_checks,
    _merge_results_by_label,
    _stage_plan,
    build_pretrain_jobs,
)
from platonic_init.support import basis_sweep_dir, prepretraining_seed_dir


class PipelineStageTests(unittest.TestCase):
    def test_stage_plan_defaults(self) -> None:
        run_prepretrain, run_fit_initializations, run_pretrain = _stage_plan(
            ["prepretrain", "fit_initializations", "pretrain"]
        )
        self.assertTrue(run_prepretrain)
        self.assertTrue(run_fit_initializations)
        self.assertTrue(run_pretrain)

    def test_stage_plan_subset(self) -> None:
        run_prepretrain, run_fit_initializations, run_pretrain = _stage_plan(
            ["fit_initializations", "pretrain"]
        )
        self.assertFalse(run_prepretrain)
        self.assertTrue(run_fit_initializations)
        self.assertTrue(run_pretrain)


class PipelineDoctorTests(unittest.TestCase):
    def _args(self, **kwargs) -> argparse.Namespace:
        defaults = dict(
            skip_transfer=False,
            skip_random=False,
            skip_fits=False,
            init_mode="sampled",
            transfer_seed=0,
            fit_names=["chebyshev", "fourier"],
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def _config(self) -> ExperimentConfig:
        cfg = ExperimentConfig()
        cfg.stages.fit_initializations.fit_blocks = [
            AnalyticFitBlockConfig(name="chebyshev", basis_type="chebyshev"),
            AnalyticFitBlockConfig(name="fourier", basis_type="fourier"),
        ]
        return cfg

    def test_doctor_flags_missing_transfer_checkpoint(self) -> None:
        cfg = self._config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.sweep.output_root = tmp
            cfg.sweep.experiment_name = "exp_missing_transfer"
            args = self._args()
            issues = _doctor_checks(
                cfg, args, run_fit_initializations=False, run_pretrain=True
            )
        self.assertTrue(any("Missing transfer checkpoint" in issue for issue in issues))

    def test_doctor_flags_missing_basis_subspace(self) -> None:
        cfg = self._config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.sweep.output_root = tmp
            cfg.sweep.experiment_name = "exp_missing_basis"
            # Satisfy transfer checkpoint requirement.
            prepretraining_seed_dir(cfg, 0).mkdir(parents=True, exist_ok=True)
            args = self._args()
            issues = _doctor_checks(
                cfg, args, run_fit_initializations=False, run_pretrain=True
            )
        self.assertTrue(any("Missing analytic subspace" in issue for issue in issues))

    def test_doctor_skips_transfer_requirements_when_transfer_disabled(self) -> None:
        cfg = self._config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.sweep.output_root = tmp
            cfg.sweep.experiment_name = "exp_skip_transfer"
            args = self._args(skip_transfer=True)
            issues = _doctor_checks(
                cfg, args, run_fit_initializations=False, run_pretrain=True
            )
        self.assertTrue(any("Missing analytic subspace" in issue for issue in issues))
        self.assertFalse(
            any("Missing transfer checkpoint" in issue for issue in issues)
        )

    def test_doctor_allows_transfer_only_without_fit_artifacts(self) -> None:
        cfg = self._config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.sweep.output_root = tmp
            cfg.sweep.experiment_name = "exp_transfer_only"
            prepretraining_seed_dir(cfg, 0).mkdir(parents=True, exist_ok=True)
            args = self._args(skip_fits=True)
            issues = _doctor_checks(
                cfg, args, run_fit_initializations=False, run_pretrain=True
            )
        self.assertEqual(issues, [])

    def test_doctor_ok_for_pretrain_only_when_inputs_exist(self) -> None:
        cfg = self._config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.sweep.output_root = tmp
            cfg.sweep.experiment_name = "exp_ok"
            prepretraining_seed_dir(cfg, 0).mkdir(parents=True, exist_ok=True)
            bs_dir = basis_sweep_dir(cfg)
            bs_dir.mkdir(parents=True, exist_ok=True)
            for basis in ("chebyshev", "fourier"):
                (bs_dir / f"analytic_subspace_{basis}.pt").touch()
            (bs_dir / "merged_rebasin_state.pt").touch()
            args = self._args(fit_names=["chebyshev", "fourier"])
            issues = _doctor_checks(
                cfg, args, run_fit_initializations=False, run_pretrain=True
            )
        self.assertEqual(issues, [])


class ConfigTests(unittest.TestCase):
    def test_load_config_requires_stage_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.yaml"
            path.write_text(
                (
                    "data_path: data/synthetic.txt\n"
                    "training:\n"
                    "  model_name_or_path: gpt2\n"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "top-level 'stages'"):
                load_config(path)


class DatasetCacheTests(unittest.TestCase):
    def test_tokenized_dataset_cache_round_trip(self) -> None:
        tokenizer = CharTokenizer(
            vocab={"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "a": 4, "b": 5}
        )
        ds = Dataset.from_dict({"text": ["ab", "ba", "ab"]})
        with tempfile.TemporaryDirectory() as tmp:
            key = dataset_cache_key("round-trip", tokenizer_cache_key(tokenizer), 4)
            first = load_or_create_tokenized_dataset(
                ds, tokenizer, block_size=4, cache_dir=tmp, cache_key=key
            )
            second = load_or_create_tokenized_dataset(
                ds, tokenizer, block_size=4, cache_dir=tmp, cache_key=key
            )
            self.assertEqual(len(first), len(second))
            self.assertTrue((Path(tmp) / key).exists())


class PretrainJobTests(unittest.TestCase):
    def _config(self) -> ExperimentConfig:
        cfg = ExperimentConfig()
        cfg.sweep.experiment_name = "job_exp"
        cfg.stages.fit_initializations.fit_blocks = [
            AnalyticFitBlockConfig(name="chebyshev", basis_type="chebyshev"),
            AnalyticFitBlockConfig(name="fourier", basis_type="fourier"),
        ]
        return cfg

    def _args(self, **kwargs) -> argparse.Namespace:
        defaults = dict(
            skip_random=False,
            skip_fits=False,
            skip_transfer=False,
            init_mode="sampled",
            transfer_seed=0,
            fit_names=None,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_build_pretrain_jobs_selects_requested_variants(self) -> None:
        cfg = self._config()
        jobs = build_pretrain_jobs(
            cfg,
            self._args(fit_names=["fourier"]),
            basis_subspaces={"fourier": {"tensor": {}}},
            transfer_model_path="runs/prepretraining/job_exp/seed_0",
            transfer_state_dict=None,
        )
        self.assertEqual(
            [job.label for job in jobs], ["random", "fourier", "weight_transfer"]
        )

    def test_build_pretrain_jobs_allows_transferless_run(self) -> None:
        cfg = self._config()
        jobs = build_pretrain_jobs(
            cfg,
            self._args(skip_transfer=True, fit_names=["chebyshev"], skip_random=True),
            basis_subspaces={"chebyshev": {"tensor": {}}},
            transfer_model_path=None,
            transfer_state_dict=None,
        )
        self.assertEqual([job.label for job in jobs], ["chebyshev"])


class PipelineResultMergeTests(unittest.TestCase):
    def test_merge_results_by_label_replaces_existing_and_preserves_others(
        self,
    ) -> None:
        existing = [
            {"label": "random", "final_eval_loss": 9.0},
            {"label": "chebyshev", "final_eval_loss": 8.0},
            {"label": "weight_transfer", "final_eval_loss": 7.0},
        ]
        updated = [
            {"label": "weight_transfer", "final_eval_loss": 6.5},
        ]
        merged = _merge_results_by_label(existing, updated)
        self.assertEqual(
            [x["label"] for x in merged], ["random", "chebyshev", "weight_transfer"]
        )
        self.assertEqual(merged[-1]["final_eval_loss"], 6.5)


if __name__ == "__main__":
    unittest.main()
