"""Stage-I one-seed worker protocol and artifact contract."""
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from pgdpo_delay.core import artifacts
from pgdpo_delay.core.runner import _validate_complete_directory
from pgdpo_delay.core.stage1_run import (
    WORKER_API_VERSION,
    canonical_run_spec,
    load_stage1_protocol,
    metric_roles_for_spec,
    required_metrics_for_spec,
    run_fingerprint,
)


def test_production_protocol_is_phase_b_and_torch_free():
    protocol = load_stage1_protocol("p1_u")
    assert protocol["training"] == {
        "iters": 3000,
        "batch": 1024,
        "lr": 5e-5,
        "hidden": 256,
        "num_layers": 2,
        "clip_grad": 1.0,
        "log_every": 100,
        "val_every": 100,
        "val_batch": 1024,
    }
    assert protocol["evaluation"] == {
        "Np": 50000,
        "seed": 123,
        "batch_size": 4096,
    }


def test_canonical_run_spec_and_fingerprint_are_stable():
    first = canonical_run_spec("p1", "p1_u")
    second = canonical_run_spec("p1", "p1_u")
    assert first == second
    assert first["worker_api_version"] == WORKER_API_VERSION
    assert first["chart"] == "identity"
    assert first["problem_config"]["name"] == "main_u"
    assert len(first["problem_config_raw_hash"]) == 64
    assert first["problem_config_raw_hash"] == \
        first["problem_config"]["raw_hash"]
    assert len(first["problem_config_hash"]) == 16
    assert first["problem_config_hash"] == \
        first["problem_config"]["scientific_hash"]
    assert first["problem_config"]["scientific"]["schema"] == 2
    assert first["initial_law"]["api"] == "p1.make_hist-v1"
    assert first["input_schema"]["feat_dim"] == 3
    assert first["input_schema"]["feature_schema"] == \
        "state_global_time_relative_lag_v2"
    assert first["training"]["num_layers"] == 2
    assert len(first["source_identity"]["source_tree_sha256"]) == 64
    assert "core/stage1.py" in first["source_identity"]["files"]
    assert "core/structured.py" in first["source_identity"]["files"]
    assert "control_nrmse" in required_metrics_for_spec(first)
    assert metric_roles_for_spec(first)["J_exact"] == \
        "shared_evaluation_diagnostic"
    assert metric_roles_for_spec(first)["dJ_se"] == \
        "within_policy_paired_mc_se"
    assert len(run_fingerprint(first)) == 24
    assert run_fingerprint(first) == run_fingerprint(second)
    # The fingerprint is a scientific protocol identity, not a GPU/seed ID.
    assert "seed" not in first["training"]
    assert "device" not in first


def test_protocol_problem_mismatch_is_refused():
    with pytest.raises(ValueError, match="not requested problem"):
        canonical_run_spec("p2", "p1_u")


def test_canonical_protocol_shadowing_is_refused(tmp_path, monkeypatch):
    canonical = files("pgdpo_delay.configs").joinpath(
        "stage1", "p1_u.yaml").read_text()
    directory = tmp_path / "configs" / "stage1"
    directory.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    (directory / "p1_u.yaml").write_text(canonical)
    assert load_stage1_protocol("p1_u")["training"]["iters"] == 3000
    (directory / "p1_u.yaml").write_text(
        canonical.replace("iters: 3000", "iters: 2999"))
    with pytest.raises(RuntimeError, match="shadows canonical"):
        load_stage1_protocol("p1_u")


def test_single_seed_smoke_writes_complete_contract(tmp_path):
    torch = pytest.importorskip("torch")
    from pgdpo_delay.core.stage1_models import load_checkpoint
    from pgdpo_delay.core.stage1_run import run_single_seed

    spec = canonical_run_spec("p1", "p1_u_smoke")
    fingerprint = run_fingerprint(spec)
    outdir = tmp_path / "seed7"
    outdir.mkdir()
    (outdir / "run.log").write_text("")  # scheduler-owned capture is allowed
    result = run_single_seed(
        problem="p1", protocol_name="p1_u_smoke", seed=7, device="cpu",
        outdir=outdir, expected_run_fingerprint=fingerprint)
    assert result["status"] == "COMPLETE"

    expected = {
        "run.log", "status.json", "stage1_state.pt", "stage1_spec.json",
        "training_trace.npz", "metrics.json", "manifest.json", "config.json",
    }
    assert {path.name for path in outdir.iterdir()} == expected
    status = json.loads((outdir / "status.json").read_text())
    manifest = json.loads((outdir / "manifest.json").read_text())
    metrics = json.loads((outdir / "metrics.json").read_text())
    checkpoint_spec = json.loads((outdir / "stage1_spec.json").read_text())
    assert status["status"] == "COMPLETE" and status["seed"] == 7
    assert status["run_fingerprint"] == fingerprint
    assert status["problem_config_hash"] == spec["problem_config_hash"]
    assert manifest["seeds"]["train"] == 7
    assert manifest["extra"]["run_fingerprint"] == fingerprint
    assert manifest["extra"]["problem_config_hash"] == \
        spec["problem_config_hash"]
    assert manifest["extra"]["chart"] == "identity"
    assert checkpoint_spec["run_fingerprint"] == fingerprint
    assert checkpoint_spec["problem_config"] == "main_u"
    assert checkpoint_spec["chart"] == "identity"
    assert checkpoint_spec["source_tree_sha256"] == \
        spec["source_identity"]["source_tree_sha256"]
    assert checkpoint_spec["initial_law"] == spec["initial_law"]
    assert checkpoint_spec["input_schema"] == spec["input_schema"]
    assert checkpoint_spec["problem_config_hash"] == \
        spec["problem_config_hash"]
    assert checkpoint_spec["model_schema"] == "buffer_scan_lstm_mlp_v2"
    assert checkpoint_spec["num_layers"] == 2
    assert checkpoint_spec["config_hash"] == manifest["config_hash"]
    assert all(isinstance(value, (int, float)) for value in metrics.values())
    assert {"control_nrmse", "dJ_paired", "dJ_se", "best_iter",
            "clip_frac", "total_runtime_seconds", "mc_anchor_gap_se"} \
        <= metrics.keys()
    with np.load(outdir / "training_trace.npz", allow_pickle=False) as trace:
        assert trace["training_loss"].shape == (1,)
        assert trace["validation_loss"].shape == (1,)
    policy, loaded_spec = load_checkpoint(
        outdir,
        expected={
            "problem": "p1",
            "problem_config": "main_u",
            "chart": "identity",
            "run_fingerprint": fingerprint,
        },
    )
    assert policy.training is False
    assert loaded_spec["seed"] == 7
    with pytest.raises(ValueError, match="checkpoint binding mismatch"):
        load_checkpoint(outdir, expected={"chart": "sigmoid"})

    required = tuple(Path(name) for name in (
        "manifest.json", "config.json", "metrics.json", "status.json",
        "stage1_state.pt", "stage1_spec.json", "training_trace.npz",
    ))
    _validate_complete_directory(outdir, fingerprint, required, 7)

    # Even coordinated config + manifest-hash edits cannot retain the stale
    # run fingerprint/checkpoint binding and be accepted by resume.
    config_path = outdir / "config.json"
    config = json.loads(config_path.read_text())
    config["training"]["batch"] += 1
    config_path.write_text(json.dumps(config))
    manifest["config_hash"] = artifacts.config_hash(config)
    (outdir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="run_fingerprint mismatch"):
        _validate_complete_directory(outdir, fingerprint, required, 7)


def test_worker_fails_instead_of_publishing_nonfinite_required_metric(
        tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import pgdpo_delay.core.stage1_run as worker

    spec = worker.canonical_run_spec("p1", "p1_u_smoke")
    fingerprint = worker.run_fingerprint(spec)
    outdir = tmp_path / "seed8"

    real_evaluate = worker._evaluate

    def nonfinite_evaluate(*args, **kwargs):
        metrics = real_evaluate(*args, **kwargs)
        metrics["control_nrmse"] = float("nan")
        return metrics

    monkeypatch.setattr(worker, "_evaluate", nonfinite_evaluate)
    with pytest.raises(FloatingPointError, match="control_nrmse"):
        worker.run_single_seed(
            problem="p1", protocol_name="p1_u_smoke", seed=8,
            device="cpu", outdir=outdir,
            expected_run_fingerprint=fingerprint)

    status = json.loads((outdir / "status.json").read_text())
    assert status["status"] == "FAILED"
    assert "FloatingPointError" in status["error"]
    assert not (outdir / "manifest.json").exists()
    assert not (outdir / "metrics.json").exists()


def test_worker_refuses_fingerprint_mismatch_before_writing(tmp_path):
    pytest.importorskip("torch")
    from pgdpo_delay.core.stage1_run import run_single_seed

    outdir = tmp_path / "wrong"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_single_seed(
            problem="p1", protocol_name="p1_u_smoke", seed=1, device="cpu",
            outdir=outdir, expected_run_fingerprint="wrong")
    assert not outdir.exists()
