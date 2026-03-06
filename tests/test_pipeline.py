from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from platonic_init.config import AnalyticFitBlockConfig, ExperimentConfig
from platonic_init.pipeline import _doctor_checks, _stage_plan
from platonic_init.paths import basis_sweep_dir, prepretraining_seed_dir


class PipelineStageTests(unittest.TestCase):
    def test_stage_plan_defaults(self) -> None:
        run_prepretrain, run_fit_initializations, run_pretrain = _stage_plan(
            ["prepretrain", "fit_initializations", "pretrain"]
        )
        self.assertTrue(run_prepretrain)
        self.assertTrue(run_fit_initializations)
        self.assertTrue(run_pretrain)

    def test_stage_plan_subset(self) -> None:
        run_prepretrain, run_fit_initializations, run_pretrain = _stage_plan(["fit_initializations", "pretrain"])
        self.assertFalse(run_prepretrain)
        self.assertTrue(run_fit_initializations)
        self.assertTrue(run_pretrain)


class PipelineDoctorTests(unittest.TestCase):
    def _args(self, **kwargs) -> argparse.Namespace:
        defaults = dict(
            skip_transfer=False,
            transfer_seed=0,
            fit_names=["chebyshev", "fourier"],
            basis=None,
            basis_dir=None,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_doctor_flags_missing_transfer_checkpoint(self) -> None:
        cfg = ExperimentConfig()
        cfg.analytic_fit_blocks = [
            AnalyticFitBlockConfig(name="chebyshev", basis_type="chebyshev"),
            AnalyticFitBlockConfig(name="fourier", basis_type="fourier"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cfg.sweep.output_root = tmp
            cfg.sweep.experiment_name = "exp_missing_transfer"
            args = self._args()
            issues = _doctor_checks(cfg, args, run_fit_initializations=False, run_pretrain=True)
        self.assertTrue(any("Missing transfer checkpoint" in issue for issue in issues))

    def test_doctor_flags_missing_basis_subspace(self) -> None:
        cfg = ExperimentConfig()
        cfg.analytic_fit_blocks = [
            AnalyticFitBlockConfig(name="chebyshev", basis_type="chebyshev"),
            AnalyticFitBlockConfig(name="fourier", basis_type="fourier"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cfg.sweep.output_root = tmp
            cfg.sweep.experiment_name = "exp_missing_basis"
            # Satisfy transfer checkpoint requirement.
            prepretraining_seed_dir(cfg, 0).mkdir(parents=True, exist_ok=True)
            args = self._args()
            issues = _doctor_checks(cfg, args, run_fit_initializations=False, run_pretrain=True)
        self.assertTrue(any("Missing analytic subspace" in issue for issue in issues))

    def test_doctor_ok_for_pretrain_only_when_inputs_exist(self) -> None:
        cfg = ExperimentConfig()
        cfg.analytic_fit_blocks = [
            AnalyticFitBlockConfig(name="chebyshev", basis_type="chebyshev"),
            AnalyticFitBlockConfig(name="fourier", basis_type="fourier"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cfg.sweep.output_root = tmp
            cfg.sweep.experiment_name = "exp_ok"
            prepretraining_seed_dir(cfg, 0).mkdir(parents=True, exist_ok=True)
            bs_dir = basis_sweep_dir(cfg)
            bs_dir.mkdir(parents=True, exist_ok=True)
            for basis in ("chebyshev", "fourier"):
                (bs_dir / f"analytic_subspace_{basis}.pt").touch()
            args = self._args(fit_names=["chebyshev", "fourier"])
            issues = _doctor_checks(cfg, args, run_fit_initializations=False, run_pretrain=True)
        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
