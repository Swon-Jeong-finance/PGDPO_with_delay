"""Integrity and statistical tests for Stage-I seed aggregation."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scipy.stats import t as student_t

from pgdpo_delay.core import artifacts
from pgdpo_delay.reporting import stage1_aggregate as aggregate_mod
from pgdpo_delay.reporting.stage1_aggregate import (
    Stage1AggregationError,
    aggregate_stage1_runs,
)


class Stage1AggregateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "p1-main-u"
        self.root.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _seed(
        self,
        seed,
        *,
        fingerprint="run-fp",
        config=None,
        method="stage1_lstm_dpo",
        status="COMPLETE",
        metrics=None,
        status_fingerprint=None,
        problem_config_hash=None,
        error=None,
    ):
        directory = self.root / f"seed{seed}"
        directory.mkdir()
        config = {"lr": 1e-3, "batch": 64} if config is None else config
        config = dict(config)
        if problem_config_hash is not None:
            config["problem_config_hash"] = problem_config_hash
        extra = {"run_fingerprint": fingerprint}
        if problem_config_hash is not None:
            extra["problem_config_hash"] = problem_config_hash
        artifacts.write_manifest(
            directory,
            problem="p1",
            method=method,
            config=config,
            seeds={"train": seed},
            device=f"cuda:{seed % 3}",
            solver="torch-stage1",
            extra=extra,
        )
        status_payload = {
            "schema": 1,
            "status": status,
            "problem": "p1",
            "method": method,
            "seed": seed,
            "run_fingerprint": status_fingerprint or fingerprint,
        }
        if problem_config_hash is not None:
            status_payload["problem_config_hash"] = problem_config_hash
        if error is not None:
            status_payload["error"] = error
        artifacts.atomic_write_json(directory / "status.json", status_payload)
        if metrics is None:
            metrics = {
                "control_nrmse": float(seed),
                "best_iter": float(20 + seed),
                "clip_frac": 0.1 * seed,
                "total_runtime_seconds": float(9 + seed),
            }
        if metrics is not False:
            artifacts.atomic_write_json(directory / "metrics.json", metrics)
        return directory

    def test_writes_per_seed_and_student_t_summary(self):
        for seed in (1, 2, 3):
            self._seed(seed)

        result = aggregate_stage1_runs(self.root, expected_seeds=[1, 2, 3])

        self.assertEqual(result.payload["counts"]["complete"], 3)
        self.assertEqual(result.payload["identity"]["run_fingerprint"], "run-fp")
        self.assertTrue(result.per_seed_csv.is_file())
        self.assertTrue(result.summary_csv.is_file())
        self.assertTrue(result.summary_json.is_file())

        with open(result.per_seed_csv, newline="", encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
        self.assertEqual([int(row["seed"]) for row in rows], [1, 2, 3])
        self.assertEqual({row["status"] for row in rows}, {"COMPLETE"})
        # Health/runtime fields are first-class metrics, not discarded metadata.
        for name in ("best_iter", "clip_frac", "total_runtime_seconds"):
            self.assertIn(name, rows[0])

        stats = result.payload["metrics"]["control_nrmse"]
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(stats["sd"], 1.0)
        self.assertAlmostEqual(stats["se"], 1.0 / math.sqrt(3.0))
        radius = float(student_t.ppf(0.975, 2)) / math.sqrt(3.0)
        self.assertAlmostEqual(stats["ci95_low"], 2.0 - radius)
        self.assertAlmostEqual(stats["ci95_high"], 2.0 + radius)

    def test_single_seed_has_undefined_variance_and_ci(self):
        self._seed(7)
        result = aggregate_stage1_runs(self.root, expected_seeds=[7])
        stats = result.payload["metrics"]["control_nrmse"]
        self.assertEqual(stats["n"], 1)
        self.assertIsNone(stats["sd"])
        self.assertIsNone(stats["se"])
        self.assertIsNone(stats["ci95_low"])
        self.assertIsNone(stats["ci95_high"])
        parsed = json.loads(result.summary_json.read_text(encoding="utf-8"))
        self.assertIsNone(parsed["metrics"]["control_nrmse"]["sd"])

    def test_shared_evaluation_diagnostic_suppresses_fake_seed_ci(self):
        for seed in (1, 2, 3):
            self._seed(seed, metrics={
                "control_nrmse": float(seed),
                "J_exact": 1.234,
            })
        result = aggregate_stage1_runs(
            self.root,
            expected_seeds=[1, 2, 3],
            metric_roles={"J_exact": "shared_evaluation_diagnostic"},
            shared_evaluation_bank=True,
        )
        shared = result.payload["metrics"]["J_exact"]
        self.assertEqual(shared["n"], 3)
        self.assertEqual(shared["mean"], 1.234)
        self.assertIsNone(shared["sd"])
        self.assertIsNone(shared["ci95_low"])
        self.assertFalse(shared["seed_uncertainty_applicable"])
        self.assertIn("conditional on the shared evaluation bank",
                      result.payload["uncertainty"]["training_seed_axis"])

    def test_shared_evaluation_diagnostic_must_match_across_seeds(self):
        self._seed(1, metrics={"control_nrmse": 1.0, "J_exact": 1.234})
        self._seed(2, metrics={"control_nrmse": 2.0, "J_exact": 1.235})
        with self.assertRaisesRegex(
                Stage1AggregationError, "differs across training seeds"):
            aggregate_stage1_runs(
                self.root,
                expected_seeds=[1, 2],
                metric_roles={"J_exact": "shared_evaluation_diagnostic"},
                shared_evaluation_bank=True,
            )

    def test_incomplete_seed_rejected_by_default(self):
        self._seed(1)
        self._seed(2, status="FAILED", metrics=False, error="CUDA OOM")

        with self.assertRaisesRegex(Stage1AggregationError, "incomplete"):
            aggregate_stage1_runs(self.root, expected_seeds=[1, 2])
        self.assertFalse((self.root / "summary.json").exists())

    def test_allow_incomplete_retains_failure_metadata_and_row(self):
        self._seed(1)
        self._seed(2, status="FAILED", metrics=False, error="CUDA OOM")

        result = aggregate_stage1_runs(
            self.root, expected_seeds=[1, 2, 3], allow_incomplete=True)

        self.assertEqual(result.payload["counts"]["complete"], 1)
        self.assertEqual(result.payload["counts"]["failed"], 1)
        self.assertEqual(result.payload["counts"]["missing"], 1)
        self.assertEqual(result.payload["metrics"]["control_nrmse"]["n"], 1)
        failures = {item["seed"]: item for item in result.payload["failures"]}
        self.assertEqual(failures[2]["error"], "CUDA OOM")
        self.assertIn("seed directory is missing", failures[3]["problems"])
        with open(result.per_seed_csv, newline="", encoding="utf-8") as fp:
            rows = {int(row["seed"]): row for row in csv.DictReader(fp)}
        self.assertEqual(rows[2]["status"], "FAILED")
        self.assertEqual(rows[3]["status"], "MISSING")
        self.assertEqual(rows[2]["control_nrmse"], "")

    def test_scheduler_summary_classifies_unpublished_failed_seed(self):
        self._seed(1)
        artifacts.atomic_write_json(
            self.root / "run_summary.json",
            {
                "schema": 1,
                "run_fingerprint": "run-fp",
                "status": "FAILED",
                "seeds": [
                    {"seed": 1, "status": "COMPLETE", "path": "seed1"},
                    {
                        "seed": 2,
                        "status": "FAILED",
                        "path": "failed/seed2-attempt",
                        "returncode": 7,
                        "elapsed_seconds": 1.25,
                        "error": "child process exited with code 7",
                    },
                ],
            },
        )

        result = aggregate_stage1_runs(self.root, allow_incomplete=True)

        self.assertEqual(result.payload["seeds"]["expected"], [1, 2])
        self.assertEqual(result.payload["counts"]["failed"], 1)
        self.assertEqual(result.payload["counts"]["missing"], 0)
        failure = result.payload["failures"][0]
        self.assertEqual(failure["seed"], 2)
        self.assertEqual(failure["returncode"], 7.0)
        self.assertEqual(failure["elapsed_seconds"], 1.25)
        self.assertEqual(failure["path"], "failed/seed2-attempt")

    def test_scheduler_summary_preserves_cancelled_seed_status(self):
        self._seed(1)
        artifacts.atomic_write_json(
            self.root / "run_summary.json",
            {
                "schema": 1,
                "run_fingerprint": "run-fp",
                "status": "FAILED",
                "seeds": [
                    {"seed": 1, "status": "COMPLETE", "path": "seed1"},
                    {
                        "seed": 2,
                        "status": "CANCELLED",
                        "path": None,
                        "error": "not started because scheduler was cancelled",
                    },
                ],
            },
        )
        result = aggregate_stage1_runs(self.root, allow_incomplete=True)
        self.assertEqual(result.payload["counts"]["cancelled"], 1)
        self.assertEqual(result.payload["counts"]["missing"], 0)
        with open(result.per_seed_csv, newline="", encoding="utf-8") as fp:
            rows = {int(row["seed"]): row for row in csv.DictReader(fp)}
        self.assertEqual(rows[2]["status"], "CANCELLED")

    def test_scheduler_fingerprint_mismatch_is_rejected(self):
        self._seed(1)
        artifacts.atomic_write_json(
            self.root / "run_summary.json",
            {
                "schema": 1,
                "run_fingerprint": "different-fp",
                "status": "COMPLETE",
                "seeds": [
                    {"seed": 1, "status": "COMPLETE", "path": "seed1"},
                ],
            },
        )
        with self.assertRaisesRegex(Stage1AggregationError, "disagrees"):
            aggregate_stage1_runs(self.root)

    def test_rejects_mixed_fingerprint_even_if_runs_are_complete(self):
        self._seed(1, fingerprint="fp-a")
        self._seed(2, fingerprint="fp-b")
        with self.assertRaisesRegex(Stage1AggregationError, "inconsistent run_fingerprint"):
            aggregate_stage1_runs(self.root)

    def test_manifest_status_fingerprint_conflict_is_never_summarized(self):
        self._seed(1, fingerprint="fp-a", status_fingerprint="fp-b")
        with self.assertRaisesRegex(Stage1AggregationError, "within seed artifacts"):
            aggregate_stage1_runs(self.root, allow_incomplete=True)

    def test_rejects_mixed_config_and_method(self):
        self._seed(1)
        self._seed(2, config={"lr": 2e-3, "batch": 64})
        with self.assertRaisesRegex(Stage1AggregationError, "inconsistent config_hash"):
            aggregate_stage1_runs(self.root)

        # Rebuild in a fresh root for the independent method check.
        self._tmp.cleanup()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "p1-main-u"
        self.root.mkdir()
        self._seed(1)
        self._seed(2, method="stage1_pgdpo")
        with self.assertRaisesRegex(Stage1AggregationError, "inconsistent method"):
            aggregate_stage1_runs(self.root)

    def test_scientific_problem_hash_is_carried_and_must_match_across_seeds(self):
        self._seed(1, problem_config_hash="scientific-A")
        self._seed(2, problem_config_hash="scientific-A")
        result = aggregate_stage1_runs(self.root, expected_seeds=[1, 2])
        self.assertEqual(
            result.payload["identity"]["problem_config_hash"],
            "scientific-A",
        )
        with open(result.per_seed_csv, newline="", encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
        self.assertEqual(
            {row["problem_config_hash"] for row in rows}, {"scientific-A"})

        self._tmp.cleanup()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "p1-main-u"
        self.root.mkdir()
        self._seed(1, problem_config_hash="scientific-A")
        self._seed(2, problem_config_hash="scientific-B")
        with self.assertRaisesRegex(
                Stage1AggregationError, "inconsistent problem_config_hash"):
            aggregate_stage1_runs(self.root, expected_seeds=[1, 2])

    def test_rejects_config_tampering_after_scheduler_publication(self):
        directory = self._seed(1)
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        config["lr"] = 9e-3
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(Stage1AggregationError, "config_hash"):
            aggregate_stage1_runs(self.root)

    def test_rejects_inconsistent_metric_schema(self):
        self._seed(1)
        self._seed(2, metrics={"control_nrmse": 2.0, "best_iter": 22.0})
        with self.assertRaisesRegex(Stage1AggregationError, "metric keys"):
            aggregate_stage1_runs(self.root)

    def test_csv_replace_failure_preserves_existing_file(self):
        target = self.root / "per_seed.csv"
        target.write_bytes(b"old-content\n")
        before = target.read_bytes()
        with mock.patch.object(
            aggregate_mod.os, "replace", side_effect=OSError("forced replace failure")
        ):
            with self.assertRaisesRegex(OSError, "forced replace failure"):
                aggregate_mod._atomic_write_csv(target, ["a"], [{"a": 1}])
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse(list(self.root.glob(".per_seed.csv.tmp-*")))


if __name__ == "__main__":
    unittest.main()
