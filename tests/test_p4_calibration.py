"""Focused P4 calibration/reference-publication regressions."""

import json

import numpy as np


def test_p4_frozen_config_and_response_diagnostics():
    from pgdpo_delay.problems.p4.config import load_config
    from pgdpo_delay.problems.p4.oracle import riccati
    from pgdpo_delay.problems.p4 import calibrate

    cfg = load_config("main")
    assert cfg["calibration_status"] == "frozen"
    assert cfg["init"]["alpha0_law"] == "deterministic"
    oracle = riccati(cfg)
    result = calibrate.rollout_diagnostics(
        cfg, oracle, Np=512, seed=47
    )
    assert result["history_pair_count"] == 2430
    assert result["history_response_ratio"] > 0.05
    assert result["history_du_impact_correlation"] < -0.8
    assert 0.05 < result["signal_response_ratio"] < 0.5
    assert result["signal_gain_min"] > 0.0
    assert result["p_alignment_action_rmse"] > 0.0
    assert 0.003 < result["p_alignment_action_nrmse"] < 0.02
    assert np.isfinite(list(result.values())).all()
    paths = calibrate.diagnostic_paths(cfg, oracle, Np=4, seed=53)
    assert paths["p_cur"].shape == (cfg["N"], 4)
    assert paths["p_nxt"].shape == (cfg["N"], 4)
    assert paths["q_QQ"].shape == (cfg["N"], 4)
    assert paths["u_rec_pnxt"].shape == paths["u"].shape
    assert paths["u_rec_pcur"].shape == paths["u"].shape
    np.testing.assert_array_equal(paths["u_rec"], paths["u_rec_pnxt"])
    np.testing.assert_allclose(
        paths["p_alignment_action_gap"],
        paths["u_rec_pcur"] - paths["u"],
        rtol=0.0, atol=0.0,
    )
    np.testing.assert_allclose(
        paths["p_alignment_action_gap"],
        (paths["p_cur"] - paths["p_nxt"])
        / paths["recovery_curvature"][:, None],
        rtol=1e-13, atol=1e-13,
    )
    expected_rmse = np.sqrt(np.mean(
        paths["p_alignment_action_gap"] ** 2
    ))
    expected_nrmse = np.sqrt(
        np.sum(paths["p_alignment_action_gap"] ** 2)
        / np.sum(paths["u"] ** 2)
    )
    np.testing.assert_allclose(
        float(paths["diagnostic_bank_p_alignment_action_rmse"]),
        expected_rmse,
        rtol=1e-15, atol=1e-15,
    )
    np.testing.assert_allclose(
        float(paths["diagnostic_bank_p_alignment_action_nrmse"]),
        expected_nrmse,
        rtol=1e-15, atol=1e-15,
    )
    assert float(paths["diagnostic_bank_p_alignment_action_nrmse"]) > 0.0
    assert paths["recovery_action_max_abs_error"] < 1e-10


def test_p4_fast_verify_publishes_metadata_only(tmp_path):
    from pgdpo_delay.cli import _verify_one
    from pgdpo_delay.core.artifacts import resolve_current_bundle

    _verify_one("p4", full=False, config="main", output_root=tmp_path)
    bundle = resolve_current_bundle(tmp_path / "p4", "fast")
    assert bundle is not None
    assert (bundle / "manifest.json").is_file()
    assert (bundle / "config.json").is_file()
    assert not list(bundle.glob("p4_*"))
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["api_versions"]["p4"].startswith("p4-signed-lq-v1")
    assert manifest["extra"]["p4"]["calibration_status"] == "frozen"
    provenance = manifest["extra"]["p4"]["source_provenance"]
    assert len(provenance["source_tree_sha256"]) == 64
    assert "problems/p4/oracle.py" in provenance["file_sha256"]


def test_p4_canonical_config_cannot_be_shadowed(tmp_path, monkeypatch):
    from importlib.resources import files
    import pytest

    from pgdpo_delay.problems.p4.config import load_config

    canonical = files("pgdpo_delay.configs").joinpath(
        "p4", "main.yaml"
    ).read_text()
    local = tmp_path / "configs" / "p4"
    local.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    (local / "main.yaml").write_text(canonical.replace("gamma: 0.75", "gamma: 0.5"))
    with pytest.raises(RuntimeError):
        load_config("main")
