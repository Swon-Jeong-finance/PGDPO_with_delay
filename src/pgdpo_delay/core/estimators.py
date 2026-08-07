"""Backend-neutral statistical primitives for the common Stage-II harvest.

This module deliberately stops at *raw* estimator outputs.  A problem/backend
specific harvester may use NumPy, torch autograd, or another differentiable
simulator to produce the branch samples defined below, but it must obey two
contracts:

* future policy values are fixed-control values for differentiation (the
  policy is re-evaluated on each branch state and then detached); physical and
  delay-state re-entry remain differentiable; and
* the nested antithetic regression returns raw ``zeta``.  Reconstruction of
  ``q_anc = zeta + Pi @ sigma_ref`` belongs to :mod:`pgdpo_delay.core.stage2`
  and must happen there, before projection.

Shape convention (batch prefixes are denoted by ``...``)::

    p samples          (..., M, n)
    Pi samples         (..., M, n, n)
    Brownian offsets   (..., M_out, d)
    nested adjoints    (..., M_out, M_in, n)
    p                  (..., n)
    Pi                 (..., n, n)
    sigma_ref, zeta    (..., n, d)

``branch_batch_size`` is only a compute/chunking parameter.  It never changes
the scientific continuation counts ``M`` and ``2*M_out*M_in``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


Array = np.ndarray


class EstimatorContractError(ValueError):
    """Raised when branch samples violate the common Stage-II contract."""


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise EstimatorContractError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise EstimatorContractError(f"{name} must be a positive integer")
    return value


def _anchor_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EstimatorContractError("anchor_id must be a non-empty string")
    return value


def _finite_array(value: object, name: str) -> Array:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise EstimatorContractError(f"{name} must be numeric")
    if not np.all(np.isfinite(array)):
        raise EstimatorContractError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class BranchBudgets:
    """Scientific branch counts plus an independent compute chunk size.

    ``M`` is the direct continuation count for the OL-BPTT estimates of
    ``p`` and ``Pi``.  The nested estimator consumes an antithetic pair for
    every outer/inner branch and therefore has exactly
    ``2 * M_out * M_in`` continuations.  ``branch_batch_size`` only partitions
    either workload for execution.
    """

    M: int
    M_out: int
    M_in: int
    branch_batch_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "M", _positive_int(self.M, "M"))
        object.__setattr__(self, "M_out", _positive_int(self.M_out, "M_out"))
        object.__setattr__(self, "M_in", _positive_int(self.M_in, "M_in"))
        object.__setattr__(
            self,
            "branch_batch_size",
            _positive_int(self.branch_batch_size, "branch_batch_size"),
        )

    @property
    def direct_continuations(self) -> int:
        return self.M

    @property
    def nested_continuations(self) -> int:
        return 2 * self.M_out * self.M_in

    @property
    def total_continuations(self) -> int:
        return self.direct_continuations + self.nested_continuations

    def chunks(self, count: int) -> Iterator[slice]:
        """Yield execution slices without altering ``count``.

        This helper is intentionally agnostic about which scientific bank is
        being processed.  Callers pass either ``M`` or the nested continuation
        count explicitly, which keeps chunking out of estimator mathematics.
        """

        count = _positive_int(count, "count")
        for start in range(0, count, self.branch_batch_size):
            yield slice(start, min(start + self.branch_batch_size, count))


@dataclass(frozen=True)
class OLBPTTSamples:
    """Fixed-control OL-BPTT branch outputs at one named anchor bank."""

    anchor_id: str
    p: Array
    Pi: Array


@dataclass(frozen=True)
class RawDerivativeEstimate:
    """Monte-Carlo means of the raw direct derivative branches."""

    anchor_id: str
    p: Array
    Pi: Array
    p_mc_se: Array
    Pi_mc_se: Array
    samples: int


@dataclass(frozen=True)
class AnchoredArray:
    """An array carrying the identity of the state/control anchor it uses."""

    anchor_id: str
    value: Array


@dataclass(frozen=True)
class NestedAntitheticSamples:
    """Nested CRN adjoints from plus/minus branches of the same anchor.

    ``adjoint_plus`` and ``adjoint_minus`` retain the inner-replication axis.
    For every ``(outer, inner)`` index the two entries must have been generated
    with the same future-noise path.  That CRN pairing is a harvester
    responsibility; retaining aligned array axes makes the pairing explicit
    and auditable here.
    """

    anchor_id: str
    brownian_offsets: Array
    adjoint_plus: Array
    adjoint_minus: Array


@dataclass(frozen=True)
class RegressionDiagnostics:
    """Numerical diagnostics for the anchored nested regression."""

    outer_samples: int
    inner_samples: int
    state_dim: int
    noise_dim: int
    ridge: float
    gram_rank: Array
    system_condition: Array
    gram_min_eigenvalue: Array
    gram_max_eigenvalue: Array
    antithetic_response_rms: Array
    ols_residual_rms: Array
    anchored_identity_max_abs: Array


@dataclass(frozen=True)
class NestedZetaEstimate:
    """Raw nested-regression output.

    ``q_ols`` is retained only as an estimator audit.  There is deliberately no
    projected or recovered ``q_anc`` field: the solver layer reconstructs that
    coordinate from ``zeta`` and ``Pi @ sigma_ref`` in its mandated order.
    """

    anchor_id: str
    zeta: Array
    q_ols: Array
    diagnostics: RegressionDiagnostics


def assemble_raw_recovery_inputs(
    derivatives: RawDerivativeEstimate,
    nested: NestedZetaEstimate,
    sigma_ref: AnchoredArray,
    *,
    anchor: object,
    pi_layout: str = "auto",
):
    """Bridge validated estimator outputs into the shared Stage-II core.

    This is intentionally the only convenience bridge: it refuses to combine
    direct derivatives, nested regression, and diffusion values from different
    state/control anchors.  Projection and ``q_anc`` reconstruction still
    happen later inside :func:`core.stage2.prepare_inputs`.
    """

    if not isinstance(derivatives, RawDerivativeEstimate):
        raise TypeError("derivatives must be RawDerivativeEstimate")
    if not isinstance(nested, NestedZetaEstimate):
        raise TypeError("nested must be NestedZetaEstimate")
    if not isinstance(sigma_ref, AnchoredArray):
        raise TypeError("sigma_ref must be AnchoredArray")
    ids = (
        _anchor_id(derivatives.anchor_id),
        _anchor_id(nested.anchor_id),
        _anchor_id(sigma_ref.anchor_id),
    )
    if len(set(ids)) != 1:
        raise EstimatorContractError(
            "direct, nested, and sigma_ref estimates must use the same "
            f"anchor_id; received {ids}"
        )
    from .stage2 import RawRecoveryInputs

    return RawRecoveryInputs(
        p=derivatives.p,
        zeta=nested.zeta,
        Pi=derivatives.Pi,
        sigma_ref=_finite_array(sigma_ref.value, "sigma_ref"),
        anchor=anchor,
        pi_layout=pi_layout,
    )


def reduce_ol_bptt(
    samples: OLBPTTSamples,
    budgets: BranchBudgets,
) -> RawDerivativeEstimate:
    """Average direct fixed-control OL-BPTT samples.

    The raw ``Pi`` samples are not symmetrised here.  Symmetrisation is part of
    the Stage-II pre-projection contract, so doing it at harvest time would
    obscure pipeline-order mistakes.
    """

    anchor_id = _anchor_id(samples.anchor_id)
    p = _finite_array(samples.p, "p samples")
    Pi = _finite_array(samples.Pi, "Pi samples")
    if p.ndim < 2:
        raise EstimatorContractError("p samples must have shape (..., M, n)")
    if Pi.ndim != p.ndim + 1:
        raise EstimatorContractError(
            "Pi samples must have shape (..., M, n, n) matching p samples"
        )
    if p.shape[:-2] != Pi.shape[:-3]:
        raise EstimatorContractError("p and Pi batch prefixes must match")
    if p.shape[-2] != Pi.shape[-3]:
        raise EstimatorContractError("p and Pi must use the same M samples")
    n = p.shape[-1]
    if Pi.shape[-2:] != (n, n):
        raise EstimatorContractError("Pi trailing dimensions must be (n, n)")
    if p.shape[-2] != budgets.M:
        raise EstimatorContractError(
            f"direct sample count {p.shape[-2]} does not match budget M={budgets.M}"
        )

    # ddof=0 is defined even for M=1 and records the Monte-Carlo standard
    # deviation of the mean, not an inferential confidence interval.
    scale = np.sqrt(float(budgets.M))
    return RawDerivativeEstimate(
        anchor_id=anchor_id,
        p=np.mean(p, axis=-2),
        Pi=np.mean(Pi, axis=-3),
        p_mc_se=np.std(p, axis=-2, ddof=0) / scale,
        Pi_mc_se=np.std(Pi, axis=-3, ddof=0) / scale,
        samples=budgets.M,
    )


def _validate_nested_shapes(
    samples: NestedAntitheticSamples,
    Pi: Array,
    sigma_ref: Array,
    budgets: BranchBudgets,
) -> tuple[Array, Array, Array, tuple[int, ...], int, int]:
    offsets = _finite_array(samples.brownian_offsets, "brownian_offsets")
    plus = _finite_array(samples.adjoint_plus, "adjoint_plus")
    minus = _finite_array(samples.adjoint_minus, "adjoint_minus")
    if offsets.ndim < 2:
        raise EstimatorContractError(
            "brownian_offsets must have shape (..., M_out, d)"
        )
    if plus.ndim != offsets.ndim + 1 or minus.shape != plus.shape:
        raise EstimatorContractError(
            "nested adjoints must both have shape (..., M_out, M_in, n)"
        )
    prefix = offsets.shape[:-2]
    if plus.shape[:-3] != prefix:
        raise EstimatorContractError(
            "Brownian offsets and nested adjoints must have matching batch prefixes"
        )
    if offsets.shape[-2] != plus.shape[-3]:
        raise EstimatorContractError(
            "Brownian offsets and nested adjoints must share M_out"
        )
    if offsets.shape[-2] != budgets.M_out:
        raise EstimatorContractError(
            f"outer sample count {offsets.shape[-2]} does not match "
            f"budget M_out={budgets.M_out}"
        )
    if plus.shape[-2] != budgets.M_in:
        raise EstimatorContractError(
            f"inner sample count {plus.shape[-2]} does not match "
            f"budget M_in={budgets.M_in}"
        )

    n, d = plus.shape[-1], offsets.shape[-1]
    expected_Pi = prefix + (n, n)
    expected_sigma = prefix + (n, d)
    if Pi.shape != expected_Pi:
        raise EstimatorContractError(
            f"Pi must have shape {expected_Pi}; received {Pi.shape}"
        )
    if sigma_ref.shape != expected_sigma:
        raise EstimatorContractError(
            f"sigma_ref must have shape {expected_sigma}; received {sigma_ref.shape}"
        )
    return offsets, plus, minus, prefix, n, d


def anchored_nested_antithetic_regression(
    samples: NestedAntitheticSamples,
    Pi: AnchoredArray,
    sigma_ref: AnchoredArray,
    budgets: BranchBudgets,
    *,
    ridge: float = 0.0,
) -> NestedZetaEstimate:
    """Estimate raw ``zeta`` by anchored nested antithetic CRN regression.

    Let ``D`` contain the outer Brownian offsets and let ``Y`` be half the
    plus/minus difference after averaging the common-random-number inner
    continuations.  With ``K = Pi @ sigma_ref``, the estimator regresses
    ``Y - D @ K.T`` on ``D`` without an intercept.  A nonzero ``ridge``
    regularises *zeta*, not ``q``.  Consequently, at ``ridge == 0`` the
    algebraic reconstruction ``zeta + K`` equals the ordinary least-squares
    estimate ``q_ols`` (up to floating-point roundoff).

    All three inputs carry ``anchor_id`` because mixing a derivative estimate,
    diffusion, and nested bank from different state/control anchors is a
    silent but result-changing error.
    """

    sample_anchor = _anchor_id(samples.anchor_id)
    pi_anchor = _anchor_id(Pi.anchor_id)
    sigma_anchor = _anchor_id(sigma_ref.anchor_id)
    if not (sample_anchor == pi_anchor == sigma_anchor):
        raise EstimatorContractError(
            "nested samples, Pi, and sigma_ref must use the same anchor_id; "
            f"received {sample_anchor!r}, {pi_anchor!r}, {sigma_anchor!r}"
        )
    if isinstance(ridge, bool) or not np.isscalar(ridge):
        raise EstimatorContractError("ridge must be a finite nonnegative scalar")
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise EstimatorContractError("ridge must be a finite nonnegative scalar")

    Pi_value = _finite_array(Pi.value, "Pi")
    sigma_value = _finite_array(sigma_ref.value, "sigma_ref")
    D, plus, minus, prefix, n, d = _validate_nested_shapes(
        samples, Pi_value, sigma_value, budgets
    )

    # Average CRN inner continuations first, retain the antithetic outer pairs.
    response = 0.5 * (np.mean(plus, axis=-2) - np.mean(minus, axis=-2))
    # Appendix A.8 defines both moments as *outer-sample averages*,
    # G_N = N_out^{-1} sum d_i d_i^T and
    # S_N = N_out^{-1} sum d_i y_i^T.  The normalization cancels when
    # ridge == 0, but it is essential when kappa_Z > 0: using raw sums would
    # silently divide the declared ridge strength by M_out.
    outer_scale = float(budgets.M_out)
    gram = np.einsum("...mi,...mj->...ij", D, D) / outer_scale
    cross = np.einsum("...mi,...mn->...in", D, response) / outer_scale

    ranks = np.linalg.matrix_rank(gram)
    if np.any(ranks < d):
        raise EstimatorContractError(
            "Brownian offset Gram matrix is rank deficient; independent noise "
            "directions are required for the unregularized q_ols audit"
        )

    # Shapes here are (..., d, n).  The public q/zeta convention is (..., n, d).
    q_ols_t = np.linalg.solve(gram, cross)
    curvature_shift = np.matmul(Pi_value, sigma_value)
    curvature_shift_t = np.swapaxes(curvature_shift, -1, -2)

    if ridge == 0.0:
        # This is the exact algebraic form of the residual regression.  Besides
        # saving a redundant solve, it makes the kappa=0 audit identity visible
        # at machine precision without moving q reconstruction into this layer.
        zeta_t = q_ols_t - curvature_shift_t
        system = gram
    else:
        # ``ridge`` is a scalar tuning parameter, not a request to change the
        # estimator precision.  Keeping both regularisation operands in the
        # Gram dtype prevents an otherwise isolated float32 -> float64
        # promotion of zeta, which would make the assembled Stage-II raw tuple
        # violate its common-dtype contract.
        eye = np.eye(d, dtype=gram.dtype)
        ridge_t = np.asarray(ridge, dtype=gram.dtype)
        system = gram + ridge_t * eye
        residual_cross = cross - np.matmul(gram, curvature_shift_t)
        zeta_t = np.linalg.solve(system, residual_cross)
        zeta_t = np.asarray(zeta_t, dtype=cross.dtype)

    q_ols = np.swapaxes(q_ols_t, -1, -2)
    zeta = np.swapaxes(zeta_t, -1, -2)
    q_reconstructed = zeta + curvature_shift

    fitted_ols = np.matmul(D, q_ols_t)
    residual_ols = response - fitted_ols
    eigvals = np.linalg.eigvalsh(gram)
    condition = np.linalg.cond(system)
    response_rms = np.sqrt(np.mean(response * response, axis=(-2, -1)))
    residual_rms = np.sqrt(np.mean(residual_ols * residual_ols, axis=(-2, -1)))
    identity_max = np.max(np.abs(q_reconstructed - q_ols), axis=(-2, -1))

    diagnostic_values = (
        condition,
        eigvals[..., 0],
        eigvals[..., -1],
        response_rms,
        residual_rms,
        identity_max,
    )
    if not all(np.all(np.isfinite(value)) for value in diagnostic_values):
        raise EstimatorContractError(
            "nested regression produced non-finite conditioning or diagnostics"
        )

    diagnostics = RegressionDiagnostics(
        outer_samples=budgets.M_out,
        inner_samples=budgets.M_in,
        state_dim=n,
        noise_dim=d,
        ridge=ridge,
        gram_rank=np.asarray(ranks),
        system_condition=np.asarray(condition),
        gram_min_eigenvalue=np.asarray(eigvals[..., 0]),
        gram_max_eigenvalue=np.asarray(eigvals[..., -1]),
        antithetic_response_rms=np.asarray(response_rms),
        ols_residual_rms=np.asarray(residual_rms),
        anchored_identity_max_abs=np.asarray(identity_max),
    )
    return NestedZetaEstimate(
        anchor_id=sample_anchor,
        zeta=zeta,
        q_ols=q_ols,
        diagnostics=diagnostics,
    )


__all__ = [
    "AnchoredArray",
    "BranchBudgets",
    "EstimatorContractError",
    "NestedAntitheticSamples",
    "NestedZetaEstimate",
    "OLBPTTSamples",
    "RawDerivativeEstimate",
    "RegressionDiagnostics",
    "anchored_nested_antithetic_regression",
    "assemble_raw_recovery_inputs",
    "reduce_ol_bptt",
]
