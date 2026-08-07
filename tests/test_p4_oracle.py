import numpy as np

from pgdpo_delay.problems.p4.config import load_config
from pgdpo_delay.problems.p4.oracle import (
    curvature_certificate,
    detached_curvature,
    exact_recovery_inputs,
    riccati,
    self_checks,
)


def test_p4_oracle_keeps_value_and_detached_curvatures_distinct():
    cfg = load_config("main")
    oracle = riccati(cfg)
    detached = detached_curvature(cfg)
    cert = curvature_certificate(cfg, oracle, detached)

    assert oracle["Pval"].shape == (
        cfg["N"] + 1, cfg["state_dim"], cfg["state_dim"]
    )
    assert oracle["F"].shape == (cfg["N"], cfg["state_dim"])
    assert oracle["Lambda"].shape == (cfg["N"],)
    assert detached["Gol"].shape == oracle["Pval"].shape
    assert detached["Pi"].shape == (cfg["N"] + 1,)
    assert cert["min_Lambda"] > 0.0
    assert cert["min_recovery_curvature"] > 0.0
    # This is the key semantic guard: Pi is not read from Pval_QQ.
    assert cert["max_abs_Pval_minus_Gol"] > 1e-6


def test_p4_q_form_and_recovery_are_exact_on_same_grid():
    cfg = load_config("main")
    oracle = riccati(cfg)
    detached = detached_curvature(cfg)
    rng = np.random.default_rng(18)
    p_gap_seen = False
    for _ in range(20):
        k = int(rng.integers(cfg["N"]))
        z = rng.normal(size=cfg["state_dim"])
        target = exact_recovery_inputs(k, z, cfg, oracle, detached)
        assert abs(target["q_foc_residual"]) < 1e-11
        assert abs(target["u_rec"] - target["u"]) < 1e-11
        assert abs(target["u_rec_pnxt"] - target["u"]) < 1e-11
        assert target["u_rec"] == target["u_rec_pnxt"]
        expected_gap = (
            (target["p_cur"] - target["p_nxt"])
            / target["recovery_curvature"]
        )
        np.testing.assert_allclose(
            target["u_rec_pcur"] - target["u"], expected_gap,
            rtol=1e-13, atol=1e-13,
        )
        np.testing.assert_allclose(
            target["p_alignment_action_gap"], expected_gap,
            rtol=1e-13, atol=1e-13,
        )
        assert abs(target["recovered_residual"]) < 1e-11
        p_gap_seen |= not np.isclose(target["p_cur"], target["p_nxt"])
    assert p_gap_seen


def test_p4_oracle_full_algebra_checks():
    metrics = self_checks(load_config("main"), samples=30, seed=7)
    assert metrics["dense_max_abs_error"] < 1e-12
    assert metrics["bellman_max_abs_error"] < 1e-10
    assert metrics["q_foc_max_abs_error"] < 1e-10
    assert metrics["recovery_max_abs_action_error"] < 1e-10
