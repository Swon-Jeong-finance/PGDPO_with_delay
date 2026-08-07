"""P1-local Stage-II recovery and KKT contract tests."""
import numpy as np
import pytest

from pgdpo_delay.core.stage2 import RawRecoveryInputs
from pgdpo_delay.problems.p1.config import load_config
from pgdpo_delay.problems.p1 import evaluate
from pgdpo_delay.problems.p1 import stage2


def _p_for_gradient(cfg, u, desired_gradient):
    p = cfg["params"]
    return (desired_gradient - p["R"] * u) / p["b"]


def test_recovery_uses_p_cur_and_keeps_p_nxt_diagnostic_separate():
    cfg = load_config("main_u")
    values = stage2.P1RecoveryInputs(
        p_cur=0.7,
        zeta=-0.2,
        Pi=0.4,
        sigma_bar=0.3,
        p_nxt_diagnostic=10_000.0,
    )
    result = stage2.recover_from_inputs(cfg, values)
    p = cfg["params"]
    expected = -(
        p["b"] * values.p_cur
        + p["gu"] * values.zeta
        + p["gu"] * values.Pi * values.sigma_bar
    ) / (p["R"] + p["gu"] ** 2 * values.Pi)
    assert result.action == pytest.approx(expected)
    assert result.action == result.unconstrained_action
    assert result.clipped is False
    assert stage2.P1_STAGE2_GRADIENT == "p_cur"
    assert stage2.p_alignment_diagnostic(p_cur=0.7, p_nxt=0.9) == pytest.approx(0.2)

    # A keyword named p_nxt cannot silently enter the decoder.
    with pytest.raises(TypeError):
        stage2.recover(
            cfg,
            p_nxt=values.p_nxt_diagnostic,
            zeta=values.zeta,
            Pi=values.Pi,
            sigma_bar=values.sigma_bar,
        )


def test_p1u_is_unconstrained_and_p1c_is_exact_scalar_box_solve():
    cfg_u = load_config("main_u")
    cfg_c = load_config("main")
    common = dict(p_cur=np.array([-10.0, 0.0, 10.0]), zeta=0.0, Pi=0.0, sigma_bar=0.0)
    result_u = stage2.recover(cfg_u, **common)
    result_c = stage2.recover(cfg_c, **common)
    raw = -cfg_u["params"]["b"] * common["p_cur"] / cfg_u["params"]["R"]
    assert np.array_equal(result_u.action, raw)
    assert not np.any(result_u.clipped)
    assert np.array_equal(result_c.action, np.clip(raw, *cfg_c["bounds"]))
    assert np.array_equal(result_c.clipped, result_c.action != raw)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_recovery_denominator_rejects_nonfinite_values(bad):
    cfg = load_config("main_u")
    with pytest.raises(FloatingPointError, match="denominator"):
        stage2.recovery_denominator(cfg, bad)


def test_recovery_denominator_fails_hard_at_noncoercive_boundary():
    cfg = load_config("main_u")
    p = cfg["params"]
    zero_denom_pi = -p["R"] / p["gu"] ** 2
    with pytest.raises(FloatingPointError, match="non-coercive"):
        stage2.recover(
            cfg, p_cur=0.0, zeta=0.0, Pi=zero_denom_pi, sigma_bar=0.0
        )
    with pytest.raises(FloatingPointError, match="non-coercive"):
        stage2.recover(
            cfg, p_cur=0.0, zeta=0.0, Pi=zero_denom_pi - 1.0, sigma_bar=0.0
        )


def test_cost_min_box_kkt_signs_and_unconstrained_residual():
    cfg = load_config("main")
    lo, hi = cfg["bounds"]
    base = dict(zeta=0.0, Pi=0.0, sigma_bar=0.0)

    # lower optimum needs g >= 0; a negative g points into the feasible set.
    assert stage2.kkt_residual(
        cfg, u=lo, p_cur=_p_for_gradient(cfg, lo, 2.0), **base
    ) == pytest.approx(0.0)
    assert stage2.kkt_residual(
        cfg, u=lo, p_cur=_p_for_gradient(cfg, lo, -2.0), **base
    ) == pytest.approx(2.0)

    # upper optimum needs g <= 0; a positive g points into the feasible set.
    assert stage2.kkt_residual(
        cfg, u=hi, p_cur=_p_for_gradient(cfg, hi, -3.0), **base
    ) == pytest.approx(0.0)
    assert stage2.kkt_residual(
        cfg, u=hi, p_cur=_p_for_gradient(cfg, hi, 3.0), **base
    ) == pytest.approx(3.0)

    assert stage2.kkt_residual(
        cfg, u=0.0, p_cur=_p_for_gradient(cfg, 0.0, -1.25), **base
    ) == pytest.approx(1.25)

    cfg_u = load_config("main_u")
    assert stage2.kkt_residual(
        cfg_u, u=0.0, p_cur=_p_for_gradient(cfg_u, 0.0, -1.25), **base
    ) == pytest.approx(1.25)


def test_recovered_action_has_zero_kkt_residual_in_both_variants():
    for variant in ("main_u", "main"):
        cfg = load_config(variant)
        inputs = dict(p_cur=1.8, zeta=-0.4, Pi=0.3, sigma_bar=0.2)
        result = stage2.recover(cfg, **inputs)
        residual = stage2.kkt_residual(cfg, u=result.action, **inputs)
        assert residual < 1e-12


def test_sigma_bar_uses_same_anchor_and_identity_audit_has_no_hidden_bounds():
    cfg = load_config("main_u")
    assert stage2.sigma_bar_from_anchor(cfg, sigma_ref=0.8, u_ref=0.5) == pytest.approx(
        0.8 - cfg["params"]["gu"] * 0.5
    )
    blocks = stage2.identity_audit_projectors()
    assert set(blocks) == {"p", "q_anc", "Pi"}
    assert all(projector.mode == "identity-audit" for projector in blocks.values())
    assert len({id(projector) for projector in blocks.values()}) == 3
    values = {
        "p": np.array([1.0]),
        "q_anc": np.array([2.0]),
        "Pi": np.array([[3.0]]),
    }
    for name, value in values.items():
        assert blocks[name](value) is value


def test_p1u_rollout_and_estimator_accept_no_bounds():
    cfg = load_config("main_u")
    policy = lambda k, Z: np.zeros(len(np.atleast_2d(Z)))
    paired = evaluate.rollout_paired(cfg, policy, policy, Np=4, seed=9)
    assert paired["delta_A_minus_B"] == pytest.approx(0.0)
    assert np.isfinite([paired["J_A"], paired["J_B"], paired["se"]]).all()

    z = np.zeros(cfg["H"] + 1)
    inputs = evaluate.estimator_inputs(
        cfg, policy, k=cfg["N"] - 2, z=z, M=8, Mout=4, Min=2, seed=10
    )
    assert np.isfinite(list(inputs.values())).all()


def test_active_set_diagnostics_remain_explicitly_box_only():
    cfg = load_config("main_u")
    policy = lambda k, Z: np.zeros(len(np.atleast_2d(Z)))
    with pytest.raises(ValueError, match="box-only"):
        evaluate.kkt_residual(
            cfg, dict(u=0.0, p=0.0, zeta=0.0, Pi=0.0, sigma_bar=0.0)
        )
    with pytest.raises(ValueError, match="box-only"):
        evaluate.active_set_stats(cfg, policy, Np=2, seed=0)
    with pytest.raises(ValueError, match="box-only"):
        evaluate.regime_disagreement(
            cfg, policy, policy, [(0, np.zeros(cfg["H"] + 1))]
        )


def test_torch_recovery_preserves_tensor_and_gradient_when_available():
    torch = pytest.importorskip("torch")
    cfg = load_config("main_u")
    p_cur = torch.tensor([0.3, -0.2], requires_grad=True)
    result = stage2.recover(
        cfg,
        p_cur=p_cur,
        zeta=torch.zeros_like(p_cur),
        Pi=torch.full_like(p_cur, 0.4),
        sigma_bar=torch.full_like(p_cur, 0.1),
    )
    assert isinstance(result.action, torch.Tensor)
    result.action.sum().backward()
    assert p_cur.grad is not None and torch.isfinite(p_cur.grad).all()


@pytest.mark.parametrize("variant", ["main_u", "main"])
def test_p1_adapter_executes_the_common_stage2_order(variant):
    cfg = load_config(variant)
    anchor = stage2.P1RecoveryAnchor(
        u_ref=np.array([0.2, -0.1]),
        state=np.zeros((2, cfg["H"] + 1)),
        time_index=7,
        anchor_id="same-history-and-control",
    )
    raw = RawRecoveryInputs(
        p=np.array([0.12, -0.08]),
        zeta=np.array([0.03, -0.02]),
        Pi=np.array([0.4, 0.35]),
        sigma_ref=np.array([0.25, -0.15]),
        anchor=anchor,
        pi_layout="scalar",
    )
    result = stage2.execute_p1_stage2(
        cfg, raw, stage2.identity_audit_projectors()
    )

    sigma_bar = stage2.sigma_bar_from_anchor(
        cfg, raw.sigma_ref, anchor.u_ref
    )
    expected = stage2.recover(
        cfg,
        p_cur=raw.p,
        zeta=raw.zeta,
        Pi=raw.Pi,
        sigma_bar=sigma_bar,
    )
    np.testing.assert_allclose(result.action, expected.action)
    np.testing.assert_allclose(
        result.inputs.q_anc, raw.zeta + raw.Pi * raw.sigma_ref
    )
    np.testing.assert_allclose(result.inputs.zeta, raw.zeta)
    assert result.inputs.anchor is anchor
    assert result.exact_solve
    assert result.residual.maximum < 1e-12
    assert result.recovery_health["coordinate"] == "p_cur"
    assert result.recovery_health["denominator_min"] > 0.0


def test_p1_common_stage2_requires_the_named_same_control_anchor():
    cfg = load_config("main_u")
    raw = RawRecoveryInputs(0.0, 0.0, 0.0, 0.0, anchor={"u_ref": 0.0})
    with pytest.raises(TypeError, match="P1RecoveryAnchor"):
        stage2.execute_p1_stage2(
            cfg, raw, stage2.identity_audit_projectors()
        )


def test_p1_common_stage2_rejects_implicit_identity_projection():
    cfg = load_config("main_u")
    raw = RawRecoveryInputs(
        0.0,
        0.0,
        0.0,
        0.0,
        anchor=stage2.P1RecoveryAnchor(u_ref=0.0),
    )
    with pytest.raises(ValueError, match="identity_audit_projectors"):
        stage2.execute_p1_stage2(cfg, raw, None)
