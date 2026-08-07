"""P1-local mathematics for the common Stage-II recovery pipeline.

This module intentionally does *not* harvest OL-BPTT/Monte-Carlo inputs and
does not choose numerical projection sets.  The common core owns harvest ->
``q_anc`` reconstruction -> independent block projection -> ``zeta``
re-coordination.  P1 owns only the scalar controlled-diffusion recovery and
its cost-minimisation KKT convention.

The deployed Path-A decoder consumes ``p_cur``.  ``p_nxt`` is retained only
as an exact same-grid Euler-FOC diagnostic; giving it a different field name
and keyword-only recovery API makes an accidental swap visible in code.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.stage2 import (
    BoxRecoverySet,
    LocalRecoveryOutput,
    RawRecoveryInputs,
    UnconstrainedRecoverySet,
    execute_stage2,
)
from .evaluate import active_tol


P1_STAGE2_GRADIENT = "p_cur"
IDENTITY_AUDIT_MODE = "identity-audit"


@dataclass(frozen=True)
class P1RecoveryInputs:
    """Projected P1 inputs at one state/anchor.

    ``p_nxt_diagnostic`` is deliberately not accepted by the recovery
    functions.  It may travel beside the Stage-II tuple for finite-grid and
    exact-oracle reporting, but it is never substituted for ``p_cur``.
    """

    p_cur: Any
    zeta: Any
    Pi: Any
    sigma_bar: Any
    p_nxt_diagnostic: Any | None = None


@dataclass(frozen=True)
class P1RecoveryResult:
    """Recovered action plus numerical diagnostics needed in artifacts."""

    action: Any
    unconstrained_action: Any
    denominator: Any
    clipped: Any


@dataclass(frozen=True)
class P1RecoveryAnchor:
    """Unprojected state/control anchor shared by every Stage-II block.

    ``u_ref`` is the frozen Stage-I action at the observed history.  The
    optional state/time fields are provenance for the future harvester and
    are intentionally carried through the common core without projection.
    """

    u_ref: Any
    state: Any = None
    time_index: int | None = None
    anchor_id: str | None = None


@dataclass(frozen=True)
class P1RawHarvest:
    """Scalar P1 view of a common directional Torch harvest."""

    raw: RawRecoveryInputs
    common: Any


@dataclass(frozen=True)
class IdentityAuditProjector:
    """Explicit no-op projector for audit/smoke runs only.

    The manuscript/configuration has not frozen P1 numerical projection
    bounds.  Calling this object records that the run intentionally performed
    no numerical-set modification instead of silently inventing thresholds.
    The common Stage-II core still symmetrises ``Pi`` before this block.
    """

    block: str
    mode: str = IDENTITY_AUDIT_MODE

    def __call__(self, value):
        return value


def identity_audit_projectors():
    """Return independent named no-op blocks for ``p``, ``q_anc`` and ``Pi``."""

    return {
        "p": IdentityAuditProjector("p"),
        "q_anc": IdentityAuditProjector("q_anc"),
        "Pi": IdentityAuditProjector("Pi"),
    }


def _torch_module(*values):
    """Load torch lazily only when an input is already a torch tensor."""

    if any(type(value).__module__.split(".", 1)[0] == "torch" for value in values):
        import torch

        return torch
    return None


def _has_bad(value, predicate):
    torch = _torch_module(value)
    if torch is not None:
        return bool(predicate(torch, value).any().detach().cpu().item())
    return bool(np.any(predicate(np, np.asarray(value))))


def _require_finite(name, value):
    if _has_bad(value, lambda xp, x: ~xp.isfinite(x)):
        raise FloatingPointError(f"non-finite P1 Stage-II {name}")


def _scalarise_numpy(value):
    """Keep scalar public calls ergonomic without converting torch tensors."""

    if _torch_module(value) is None:
        array = np.asarray(value)
        if array.ndim == 0:
            return array.item()
    return value


def recovery_denominator(cfg, Pi, *, denom_tol=1e-10):
    """Return ``R + gamma_u^2 Pi`` and fail hard if it is not coercive."""

    if not np.isfinite(denom_tol) or denom_tol < 0:
        raise ValueError("denom_tol must be finite and nonnegative")
    params = cfg["params"]
    denominator = params["R"] + params["gu"] ** 2 * Pi
    _require_finite("recovery denominator", denominator)
    if _has_bad(denominator, lambda xp, x: x <= denom_tol):
        raise FloatingPointError(
            "P1 Stage-II recovery denominator is non-coercive: "
            f"requires R + gamma_u^2 Pi > {denom_tol:g}"
        )
    return _scalarise_numpy(denominator)


def sigma_bar_from_anchor(cfg, sigma_ref, u_ref):
    """Control-free diffusion part at the same Stage-II reference anchor."""

    sigma_bar = sigma_ref - cfg["params"]["gu"] * u_ref
    _require_finite("sigma_bar", sigma_bar)
    return _scalarise_numpy(sigma_bar)


def recover(
    cfg,
    *,
    p_cur,
    zeta,
    Pi,
    sigma_bar,
    denom_tol=1e-10,
):
    """Recover the P1 action using the manuscript ``p_cur`` Path-A decoder.

    P1-U applies the unconstrained minimiser unchanged.  P1-C clips that same
    minimiser to its configured box; clipping is the exact scalar box solve,
    not a projection-set choice for ``p/q/Pi``.
    """

    params = cfg["params"]
    denominator = recovery_denominator(cfg, Pi, denom_tol=denom_tol)
    numerator = (
        params["b"] * p_cur
        + params["gu"] * zeta
        + params["gu"] * Pi * sigma_bar
    )
    _require_finite("recovery numerator", numerator)
    unconstrained = -numerator / denominator
    _require_finite("recovered action", unconstrained)

    bounds = cfg["bounds"]
    torch = _torch_module(unconstrained)
    if bounds is None:
        action = unconstrained
        clipped = torch.zeros_like(action, dtype=torch.bool) if torch is not None else np.zeros_like(action, dtype=bool)
    else:
        lo, hi = bounds
        if torch is not None:
            action = torch.clamp(unconstrained, min=lo, max=hi)
            clipped = action != unconstrained
        else:
            action = np.clip(unconstrained, lo, hi)
            clipped = action != unconstrained

    return P1RecoveryResult(
        action=_scalarise_numpy(action),
        unconstrained_action=_scalarise_numpy(unconstrained),
        denominator=_scalarise_numpy(denominator),
        clipped=_scalarise_numpy(clipped),
    )


def recover_from_inputs(cfg, inputs: P1RecoveryInputs, *, denom_tol=1e-10):
    """Typed convenience wrapper; ``p_nxt_diagnostic`` remains unused."""

    return recover(
        cfg,
        p_cur=inputs.p_cur,
        zeta=inputs.zeta,
        Pi=inputs.Pi,
        sigma_bar=inputs.sigma_bar,
        denom_tol=denom_tol,
    )


def hamiltonian_gradient(cfg, *, u, p_cur, zeta, Pi, sigma_bar):
    """Exact derivative of the scalar P1 generalized Hamiltonian (cost min)."""

    params = cfg["params"]
    gradient = (
        params["R"] * u
        + params["b"] * p_cur
        + params["gu"]
        * (zeta + Pi * (sigma_bar + params["gu"] * u))
    )
    _require_finite("Hamiltonian gradient", gradient)
    return _scalarise_numpy(gradient)


def kkt_residual(
    cfg,
    *,
    u,
    p_cur,
    zeta,
    Pi,
    sigma_bar,
    tol=None,
):
    """Distance-to-normal-cone residual in the cost-minimisation convention.

    For P1-U this is ``|g|``.  For P1-C it is ``max(0,-g)`` at the lower
    bound, ``max(0,g)`` at the upper bound, and ``|g|`` in the interior.
    These signs correspond to minimising over ``[lo, hi]``.
    """

    gradient = hamiltonian_gradient(
        cfg, u=u, p_cur=p_cur, zeta=zeta, Pi=Pi, sigma_bar=sigma_bar
    )
    bounds = cfg["bounds"]
    torch = _torch_module(u, gradient)
    if bounds is None:
        residual = torch.abs(gradient) if torch is not None else np.abs(gradient)
        return _scalarise_numpy(residual)

    lo, hi = bounds
    if tol is None:
        if torch is not None and getattr(u, "dtype", None) is not None:
            dtype = np.float32 if str(u.dtype) in ("torch.float16", "torch.float32", "torch.bfloat16") else np.float64
        else:
            dtype = np.asarray(u).dtype if np.issubdtype(np.asarray(u).dtype, np.floating) else np.float64
        tol = active_tol(lo, hi, dtype=dtype)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")

    if torch is not None:
        residual = torch.where(
            u <= lo + tol,
            torch.clamp_min(-gradient, 0.0),
            torch.where(
                u >= hi - tol,
                torch.clamp_min(gradient, 0.0),
                torch.abs(gradient),
            ),
        )
    else:
        u_array = np.asarray(u)
        gradient_array = np.asarray(gradient)
        residual = np.where(
            u_array <= lo + tol,
            np.maximum(0.0, -gradient_array),
            np.where(
                u_array >= hi - tol,
                np.maximum(0.0, gradient_array),
                np.abs(gradient_array),
            ),
        )
    return _scalarise_numpy(residual)


def p_alignment_diagnostic(*, p_cur, p_nxt):
    """Named finite-grid diagnostic; never an estimator-error substitute."""

    difference = p_nxt - p_cur
    _require_finite("p_nxt - p_cur diagnostic", difference)
    return _scalarise_numpy(difference)


def _minimum_as_float(value):
    torch = _torch_module(value)
    if torch is not None:
        return float(value.detach().amin().cpu().item())
    return float(np.min(np.asarray(value)))


def execute_p1_stage2(
    cfg,
    raw: RawRecoveryInputs,
    projection_blocks,
    *,
    denom_tol=1e-10,
    feasibility_tolerance=1e-8,
):
    """Connect P1 mathematics to the shared Stage-II ordering core.

    This function starts from already harvested raw inputs.  It does not
    choose projection radii and it does not run OL-BPTT.  In particular, an
    identity-audit call must pass :func:`identity_audit_projectors`
    explicitly.  The frozen ``u_ref`` comes from ``raw.anchor`` so the
    diffusion coordinate cannot accidentally be reconstructed at a different
    policy/state anchor.
    """

    if not isinstance(raw, RawRecoveryInputs):
        raise TypeError("raw must be core.stage2.RawRecoveryInputs")
    if not isinstance(raw.anchor, P1RecoveryAnchor):
        raise TypeError(
            "P1 raw.anchor must be P1RecoveryAnchor carrying the same u_ref"
        )
    if projection_blocks is None:
        raise ValueError(
            "P1 Stage-II requires explicit numerical projectors or "
            "identity_audit_projectors(); projection_blocks=None is forbidden"
        )

    def sigma_bar(projected):
        return sigma_bar_from_anchor(
            cfg, projected.sigma_ref, projected.anchor.u_ref
        )

    def health(projected, context):
        denominator = recovery_denominator(
            cfg, projected.Pi, denom_tol=denom_tol
        )
        return {
            "ok": True,
            "coordinate": P1_STAGE2_GRADIENT,
            "denominator_min": _minimum_as_float(denominator),
        }

    def solve(projected, context):
        result = recover(
            cfg,
            p_cur=projected.p,
            zeta=projected.zeta,
            Pi=projected.Pi,
            sigma_bar=sigma_bar(projected),
            denom_tol=denom_tol,
        )
        return LocalRecoveryOutput(
            action=result.action,
            diagnostics={
                "unconstrained_action": result.unconstrained_action,
                "denominator": result.denominator,
                "clipped": result.clipped,
                "gradient_coordinate": P1_STAGE2_GRADIENT,
            },
            exact=True,
        )

    def gradient(projected, action, context):
        return hamiltonian_gradient(
            cfg,
            u=action,
            p_cur=projected.p,
            zeta=projected.zeta,
            Pi=projected.Pi,
            sigma_bar=sigma_bar(projected),
        )

    if cfg["bounds"] is None:
        feasible_set = UnconstrainedRecoverySet(event_ndim=0)
    else:
        lo, hi = cfg["bounds"]
        feasible_set = BoxRecoverySet(
            lo, hi, active_tolerance=active_tol(lo, hi)
        )
    return execute_stage2(
        raw,
        projection_blocks,
        local_recovery=solve,
        objective_gradient=gradient,
        recovery_set=feasible_set,
        sense="minimize",
        recovery_health=health,
        feasibility_tolerance=feasibility_tolerance,
    )


def harvest_p1_raw_torch(
    adapter,
    policy,
    *,
    state,
    time_index,
    budgets,
    seed,
    anchor_id,
    ridge=0.0,
):
    """Harvest P1's current-coordinate raw tuple from a frozen actor.

    The common harvester uses the full matrix convention even for the single
    P1 injection/noise coordinate.  This wrapper verifies those singleton
    axes and exposes scalars to the P1 local solver, preventing accidental
    broadcasting of a ``(1,1)`` action through a path batch.
    """

    from ...core.stage2_torch import harvest_fixed_control_torch

    common = harvest_fixed_control_torch(
        adapter,
        policy,
        state=state,
        time_index=time_index,
        budgets=budgets,
        seed=seed,
        anchor_id=anchor_id,
        anchor=None,
        injection=None,  # canonical current coordinate e_0
        ridge=ridge,
    )
    p = np.asarray(common.raw.p)
    zeta = np.asarray(common.raw.zeta)
    Pi = np.asarray(common.raw.Pi)
    sigma_ref = np.asarray(common.raw.sigma_ref)
    u_ref = np.asarray(common.u_ref)
    if p.shape != (1,) or zeta.shape != (1, 1) or Pi.shape != (1, 1) \
            or sigma_ref.shape != (1, 1) or u_ref.size != 1:
        raise ValueError(
            "P1 Stage-II harvest requires one current and one noise coordinate"
        )
    if type(state).__module__.split(".", 1)[0] == "torch":
        state_record = state.detach().cpu().numpy()
    else:
        state_record = np.asarray(state).copy()
    anchor = P1RecoveryAnchor(
        u_ref=float(u_ref.reshape(-1)[0]),
        state=state_record,
        time_index=int(time_index),
        anchor_id=anchor_id,
    )
    raw = RawRecoveryInputs(
        p=float(p[0]),
        zeta=float(zeta[0, 0]),
        Pi=float(Pi[0, 0]),
        sigma_ref=float(sigma_ref[0, 0]),
        anchor=anchor,
        pi_layout="scalar",
    )
    return P1RawHarvest(raw=raw, common=common)
