import numpy as np
import pytest

from pgdpo_delay.core.estimators import (
    AnchoredArray,
    BranchBudgets,
    EstimatorContractError,
    NestedAntitheticSamples,
    OLBPTTSamples,
    anchored_nested_antithetic_regression,
    assemble_raw_recovery_inputs,
    reduce_ol_bptt,
)
from pgdpo_delay.core.stage2 import ProjectionBlocks, prepare_inputs


def _nested_fixture(*, batch=(), M_out=24, M_in=3, n=3, d=2, seed=4):
    rng = np.random.default_rng(seed)
    D = rng.normal(size=batch + (M_out, d))
    q = rng.normal(size=batch + (n, d))
    response = np.einsum("...md,...nd->...mn", D, q)
    common = rng.normal(size=batch + (M_out, M_in, n))
    plus = common + response[..., :, None, :]
    minus = common - response[..., :, None, :]
    Pi = rng.normal(size=batch + (n, n))
    sigma = rng.normal(size=batch + (n, d))
    budgets = BranchBudgets(
        M=11,
        M_out=M_out,
        M_in=M_in,
        branch_batch_size=5,
    )
    samples = NestedAntitheticSamples("anchor-A", D, plus, minus)
    return budgets, samples, Pi, sigma, q


def test_branch_budgets_separate_scientific_counts_from_chunking():
    a = BranchBudgets(M=8192, M_out=512, M_in=8, branch_batch_size=1024)
    b = BranchBudgets(M=8192, M_out=512, M_in=8, branch_batch_size=3000)

    assert a.direct_continuations == b.direct_continuations == 8192
    assert a.nested_continuations == b.nested_continuations == 8192
    assert a.total_continuations == b.total_continuations == 16384
    assert list(a.chunks(2500)) == [slice(0, 1024), slice(1024, 2048), slice(2048, 2500)]
    assert list(b.chunks(2500)) == [slice(0, 2500)]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(M=0, M_out=2, M_in=1, branch_batch_size=1),
        dict(M=2, M_out=True, M_in=1, branch_batch_size=1),
        dict(M=2, M_out=2, M_in=1, branch_batch_size=0),
    ],
)
def test_branch_budgets_reject_nonpositive_or_boolean_counts(kwargs):
    with pytest.raises(EstimatorContractError, match="positive integer"):
        BranchBudgets(**kwargs)


def test_reduce_ol_bptt_preserves_raw_matrix_and_batch_shapes():
    budgets = BranchBudgets(M=4, M_out=2, M_in=1, branch_batch_size=3)
    p_samples = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    Pi_samples = np.arange(2 * 4 * 3 * 3, dtype=float).reshape(2, 4, 3, 3)
    result = reduce_ol_bptt(
        OLBPTTSamples("same-state-and-action", p_samples, Pi_samples), budgets
    )

    assert result.anchor_id == "same-state-and-action"
    assert result.p.shape == (2, 3)
    assert result.Pi.shape == (2, 3, 3)
    np.testing.assert_allclose(result.p, p_samples.mean(axis=-2))
    np.testing.assert_allclose(result.Pi, Pi_samples.mean(axis=-3))
    # Harvesting must not silently symmetrise Pi; that happens before projection.
    assert not np.allclose(result.Pi, np.swapaxes(result.Pi, -1, -2))
    assert np.all(np.isfinite(result.p_mc_se))
    assert np.all(np.isfinite(result.Pi_mc_se))


def test_reduce_ol_bptt_enforces_M_and_matrix_dimensions():
    budgets = BranchBudgets(M=4, M_out=2, M_in=1, branch_batch_size=2)
    with pytest.raises(EstimatorContractError, match="budget M"):
        reduce_ol_bptt(
            OLBPTTSamples("a", np.zeros((3, 2)), np.zeros((3, 2, 2))), budgets
        )
    with pytest.raises(EstimatorContractError, match="trailing dimensions"):
        reduce_ol_bptt(
            OLBPTTSamples("a", np.zeros((4, 2)), np.zeros((4, 2, 3))), budgets
        )


def test_zero_ridge_reconstructs_ols_q_for_matrix_noise_and_batched_anchors():
    budgets, samples, Pi, sigma, q_true = _nested_fixture(batch=(2,))
    result = anchored_nested_antithetic_regression(
        samples,
        AnchoredArray("anchor-A", Pi),
        AnchoredArray("anchor-A", sigma),
        budgets,
        ridge=0.0,
    )

    curvature_shift = Pi @ sigma
    assert result.zeta.shape == (2, 3, 2)
    assert result.q_ols.shape == (2, 3, 2)
    np.testing.assert_allclose(result.q_ols, q_true, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(
        result.zeta + curvature_shift,
        result.q_ols,
        atol=2e-14,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        result.zeta, q_true - curvature_shift, atol=2e-14, rtol=2e-14
    )

    diag = result.diagnostics
    assert diag.outer_samples == budgets.M_out
    assert diag.inner_samples == budgets.M_in
    assert diag.state_dim == 3
    assert diag.noise_dim == 2
    for value in (
        diag.gram_rank,
        diag.system_condition,
        diag.gram_min_eigenvalue,
        diag.gram_max_eigenvalue,
        diag.antithetic_response_rms,
        diag.ols_residual_rms,
        diag.anchored_identity_max_abs,
    ):
        assert np.all(np.isfinite(value))
    assert np.max(diag.anchored_identity_max_abs) < 2e-14


def test_nonzero_ridge_regularises_zeta_not_q():
    budgets, samples, Pi, sigma, _ = _nested_fixture(batch=(), seed=9)
    ridge = 1.7
    result = anchored_nested_antithetic_regression(
        samples,
        AnchoredArray("anchor-A", Pi),
        AnchoredArray("anchor-A", sigma),
        budgets,
        ridge=ridge,
    )

    D = samples.brownian_offsets
    response = 0.5 * (
        samples.adjoint_plus.mean(axis=-2) - samples.adjoint_minus.mean(axis=-2)
    )
    gram = D.T @ D / budgets.M_out
    cross = D.T @ response / budgets.M_out
    expected_zeta_t = np.linalg.solve(
        gram + ridge * np.eye(D.shape[-1]),
        cross - gram @ (Pi @ sigma).T,
    )
    np.testing.assert_allclose(result.zeta, expected_zeta_t.T)
    assert result.diagnostics.ridge == ridge
    assert np.isfinite(result.diagnostics.system_condition)


def test_float32_nonzero_ridge_preserves_common_stage2_dtype_contract():
    """Regularising zeta must not promote one block of a float32 raw tuple."""
    budgets, samples, Pi, sigma, _ = _nested_fixture(batch=(), seed=29)
    samples32 = NestedAntitheticSamples(
        samples.anchor_id,
        samples.brownian_offsets.astype(np.float32),
        samples.adjoint_plus.astype(np.float32),
        samples.adjoint_minus.astype(np.float32),
    )
    Pi32 = Pi.astype(np.float32)
    sigma32 = sigma.astype(np.float32)
    nested = anchored_nested_antithetic_regression(
        samples32,
        AnchoredArray("anchor-A", Pi32),
        AnchoredArray("anchor-A", sigma32),
        budgets,
        ridge=0.35,
    )
    direct = reduce_ol_bptt(
        OLBPTTSamples(
            "anchor-A",
            np.ones((budgets.M, Pi32.shape[-1]), dtype=np.float32),
            np.repeat(Pi32[None, ...], budgets.M, axis=0),
        ),
        budgets,
    )
    raw = assemble_raw_recovery_inputs(
        direct,
        nested,
        AnchoredArray("anchor-A", sigma32),
        anchor=None,
        pi_layout="matrix",
    )

    assert nested.zeta.dtype == np.dtype(np.float32)
    assert nested.q_ols.dtype == np.dtype(np.float32)
    assert {raw.p.dtype, raw.zeta.dtype, raw.Pi.dtype, raw.sigma_ref.dtype} == {
        np.dtype(np.float32)
    }

    projected = prepare_inputs(raw, ProjectionBlocks.identity())
    assert projected.zeta.dtype == np.dtype(np.float32)
    assert projected.q_anc.dtype == np.dtype(np.float32)
    assert projected.diagnostics.coordinate_identity_max < 2e-6


def test_nonzero_ridge_is_invariant_to_repeating_the_outer_bank():
    """Ridge uses averaged moments, so duplicating data cannot change it."""
    budgets, samples, Pi, sigma, _ = _nested_fixture(batch=(), seed=19)
    base = anchored_nested_antithetic_regression(
        samples,
        AnchoredArray("anchor-A", Pi),
        AnchoredArray("anchor-A", sigma),
        budgets,
        ridge=0.35,
    )
    repeated_budgets = BranchBudgets(
        M=budgets.M,
        M_out=2 * budgets.M_out,
        M_in=budgets.M_in,
        branch_batch_size=budgets.branch_batch_size,
    )
    repeated = NestedAntitheticSamples(
        "anchor-A",
        np.repeat(samples.brownian_offsets, 2, axis=-2),
        np.repeat(samples.adjoint_plus, 2, axis=-3),
        np.repeat(samples.adjoint_minus, 2, axis=-3),
    )
    duplicate = anchored_nested_antithetic_regression(
        repeated,
        AnchoredArray("anchor-A", Pi),
        AnchoredArray("anchor-A", sigma),
        repeated_budgets,
        ridge=0.35,
    )
    np.testing.assert_allclose(duplicate.zeta, base.zeta, atol=2e-14, rtol=2e-14)


def test_nested_regression_rejects_mixed_anchor_ids():
    budgets, samples, Pi, sigma, _ = _nested_fixture()
    with pytest.raises(EstimatorContractError, match="same anchor_id"):
        anchored_nested_antithetic_regression(
            samples,
            AnchoredArray("different-anchor", Pi),
            AnchoredArray("anchor-A", sigma),
            budgets,
        )


def test_estimator_to_stage2_bridge_preserves_one_anchor_and_raw_coordinates():
    budgets, samples, Pi, sigma, _ = _nested_fixture(batch=(), seed=23)
    nested = anchored_nested_antithetic_regression(
        samples,
        AnchoredArray("anchor-A", Pi),
        AnchoredArray("anchor-A", sigma),
        budgets,
    )
    p_samples = np.ones((budgets.M, Pi.shape[-1]))
    pi_samples = np.repeat(Pi[None, ...], budgets.M, axis=0)
    direct = reduce_ol_bptt(
        OLBPTTSamples("anchor-A", p_samples, pi_samples), budgets
    )
    marker = object()
    raw = assemble_raw_recovery_inputs(
        direct,
        nested,
        AnchoredArray("anchor-A", sigma),
        anchor=marker,
        pi_layout="matrix",
    )
    assert raw.anchor is marker
    assert raw.pi_layout == "matrix"
    np.testing.assert_allclose(raw.p, direct.p)
    np.testing.assert_allclose(raw.zeta, nested.zeta)
    np.testing.assert_allclose(raw.Pi, direct.Pi)
    np.testing.assert_allclose(raw.sigma_ref, sigma)

    with pytest.raises(EstimatorContractError, match="same anchor_id"):
        assemble_raw_recovery_inputs(
            direct,
            nested,
            AnchoredArray("different", sigma),
            anchor=marker,
        )


def test_nested_regression_enforces_inner_budget_and_matrix_shapes():
    budgets, samples, Pi, sigma, _ = _nested_fixture()
    wrong_inner = NestedAntitheticSamples(
        samples.anchor_id,
        samples.brownian_offsets,
        samples.adjoint_plus[..., :-1, :],
        samples.adjoint_minus[..., :-1, :],
    )
    with pytest.raises(EstimatorContractError, match="budget M_in"):
        anchored_nested_antithetic_regression(
            wrong_inner,
            AnchoredArray("anchor-A", Pi),
            AnchoredArray("anchor-A", sigma),
            budgets,
        )

    with pytest.raises(EstimatorContractError, match="sigma_ref must have shape"):
        anchored_nested_antithetic_regression(
            samples,
            AnchoredArray("anchor-A", Pi),
            AnchoredArray("anchor-A", sigma[:, :1]),
            budgets,
        )


def test_nested_regression_rejects_rank_deficient_or_nonfinite_design():
    budgets, samples, Pi, sigma, _ = _nested_fixture(d=2)
    repeated = np.repeat(samples.brownian_offsets[:, :1], 2, axis=-1)
    rank_deficient = NestedAntitheticSamples(
        "anchor-A", repeated, samples.adjoint_plus, samples.adjoint_minus
    )
    with pytest.raises(EstimatorContractError, match="rank deficient"):
        anchored_nested_antithetic_regression(
            rank_deficient,
            AnchoredArray("anchor-A", Pi),
            AnchoredArray("anchor-A", sigma),
            budgets,
        )

    bad_offsets = samples.brownian_offsets.copy()
    bad_offsets[0, 0] = np.nan
    nonfinite = NestedAntitheticSamples(
        "anchor-A", bad_offsets, samples.adjoint_plus, samples.adjoint_minus
    )
    with pytest.raises(EstimatorContractError, match="finite"):
        anchored_nested_antithetic_regression(
            nonfinite,
            AnchoredArray("anchor-A", Pi),
            AnchoredArray("anchor-A", sigma),
            budgets,
        )
