"""Exact same-grid oracle for optional Problem 4 (signed execution).

The augmented state is

``Z_k = (Q_k, Q_{k-1}, ..., Q_{k-H}, alpha_k)``

and the shared Euler transition is

``Z_{k+1} = A Z_k + B u_k + D_Q u_k dW_k^Q + S_alpha dW_k^alpha``.

The two Brownian columns are independent.  With the *pre-trade* impact row
``r_I`` from :mod:`p4.dynamics`, the one-step cost is the generalized-LQ
quadratic

``h [phi/2 Q_k^2 + eta/2 u_k^2 + u_k (r_I Z_k - alpha_k)]``.

Accordingly, the value Hessian ``Pval`` obeys a generalized stochastic
Riccati recursion containing both the state-control cross weight and the
control-driven fill-noise curvature.  The additive signal noise changes the
value constant ``c`` but not the feedback.

Two second-order objects are intentionally kept separate:

``Pval``
    Hessian of the *re-optimised* value, including the Riccati Schur term.

``Gol`` / ``Pi``
    Detached fixed-control OL-BPTT Hessian and its current-inventory block.
    The control is held fixed when differentiating.  Hence the bilinear
    impact term has zero state Hessian, the state Jacobian is simply ``A``,
    and there is no Schur term.  In particular ``Pi`` must never be obtained
    by reading the ``(Q,Q)`` entry of ``Pval``.

For the same-grid Euler first-order condition, ``p_nxt`` denotes the
conditional next-step inventory gradient.  It is distinct from the
manuscript-level current gradient ``p_cur``.  The q-form and recovered form
are algebraically identical when

``zeta = q_QQ - Pi * (-sigma_Q u)``.

The module implements only the signed, independent-noise main instance.  It
does not silently add clipping, NMPC, or correlated-noise terms.
"""

from __future__ import annotations

import numpy as np

from .dynamics import impact, impact_row, linear_matrices, running_cost


ORACLE_API_VERSION = "p4-signed-lq-v1-pcur-pnext"
P_ALIGNMENT_SCHEMA_VERSION = 1


def _stage_coefficients(cfg):
    """Return ``(Qstage, Nstage, Rstage)`` in the convention

    ``ell(Z,u) = 1/2 Z'Qstage Z + Z'Nstage u + 1/2 Rstage u^2``.
    """
    p = cfg["params"]
    h, H, n = cfg["h"], cfg["H"], cfg["state_dim"]
    if cfg.get("bounds") is not None or cfg.get("control_kind") != "signed":
        raise ValueError("the exact P4 Riccati oracle requires signed controls")
    if float(cfg.get("noise", {}).get("correlation", np.nan)) != 0.0:
        raise ValueError("the P4 oracle is derived for independent noises")

    e_q = np.zeros(n, dtype=float)
    e_q[0] = 1.0
    Qstage = h * p["phi"] * np.outer(e_q, e_q)

    # r_I Z - alpha.  impact_row contains inventory-buffer coefficients only.
    ell = np.zeros(n, dtype=float)
    row = np.asarray(impact_row(cfg), dtype=float)
    if row.shape != (H + 1,):
        raise ValueError("impact_row(cfg) must have shape (H+1,)")
    ell[: H + 1] = row
    ell[-1] = -1.0
    Nstage = h * ell
    Rstage = h * p["eta"]
    return Qstage, Nstage, Rstage


def _validate_matrices(cfg, A, B, Dq, Salpha):
    n = cfg["state_dim"]
    if A.shape != (n, n):
        raise ValueError(f"A must have shape {(n, n)}")
    for name, value in (("B", B), ("Dq", Dq), ("Salpha", Salpha)):
        if value.shape != (n,):
            raise ValueError(f"{name} must have shape {(n,)}")
    p, h = cfg["params"], cfg["h"]
    # These identities pin the sign/scaling used by Appendix C.4, Eq. (68).
    if not np.isclose(B[0], -h) or not np.isclose(Dq[0], -p["sigma_Q"]):
        raise ValueError("P4 requires B_Q=-h and Dq_Q=-sigma_Q")
    if np.count_nonzero(B[1:]) or np.count_nonzero(Dq[1:]):
        raise ValueError("fill drift/noise may enter only the inventory row")


def riccati(cfg, *, require_positive: bool = True) -> dict:
    """Compute the exact signed-control generalized Riccati oracle.

    The value convention is ``V_k(z)=1/2 z'Pval[k]z+c[k]`` and the optimal
    action is ``u_k^*=F[k]@z``.  ``Lambda`` is the full one-step Bellman
    control curvature, including the outer factor ``h``.  Thus
    ``Lambda/h`` is the curvature in running-cost units.

    A nonpositive ``Lambda`` means that the signed same-grid problem is not
    retained by the calibration gate; no clipped or regularised fallback is
    substituted.
    """
    A, B, Dq, Salpha = (np.asarray(x, dtype=float)
                         for x in linear_matrices(cfg))
    _validate_matrices(cfg, A, B, Dq, Salpha)
    Qstage, Nstage, Rstage = _stage_coefficients(cfg)
    N, n, h = cfg["N"], cfg["state_dim"], cfg["h"]

    Pval = np.zeros((N + 1, n, n), dtype=float)
    c = np.zeros(N + 1, dtype=float)
    F = np.zeros((N, n), dtype=float)
    Lambda = np.zeros(N, dtype=float)

    e_q = np.zeros(n, dtype=float)
    e_q[0] = 1.0
    Pval[N] = cfg["params"]["kappa"] * np.outer(e_q, e_q)

    for k in range(N - 1, -1, -1):
        Pn = Pval[k + 1]
        lam = Rstage + B @ Pn @ B + h * (Dq @ Pn @ Dq)
        if not np.isfinite(lam):
            raise FloatingPointError(f"non-finite P4 Riccati Lambda at k={k}")
        if require_positive and lam <= 0.0:
            raise ValueError(
                f"P4 signed Riccati loses Bellman convexity at k={k}: "
                f"Lambda={lam:.16g}"
            )

        # K is the state-control cross column in the joint one-step Hessian.
        K = Nstage + A.T @ Pn @ B
        base = Qstage + A.T @ Pn @ A
        Pk = base - np.outer(K, K) / lam
        Pval[k] = 0.5 * (Pk + Pk.T)
        F[k] = -K / lam
        Lambda[k] = lam

        # W_Q and W_alpha are independent.  The W_Q term is proportional to
        # u and is already in Lambda; only additive signal noise enters c.
        c[k] = c[k + 1] + 0.5 * h * (Salpha @ Pn @ Salpha)

    return dict(Pval=Pval, c=c, F=F, Lambda=Lambda,
                Lambda_over_h=Lambda / h,
                api_version=ORACLE_API_VERSION)


def dense_riccati(cfg, *, require_positive: bool = True) -> dict:
    """Independent dense joint-Hessian form of :func:`riccati`.

    This verification implementation explicitly assembles the ``(n+1)`` by
    ``(n+1)`` Bellman Hessian in ``(Z,u)`` and takes its scalar Schur
    complement.  Keeping this separate catches cross-weight or fill-noise
    scaling mistakes in the production formula.
    """
    A, B, Dq, Salpha = (np.asarray(x, dtype=float)
                         for x in linear_matrices(cfg))
    _validate_matrices(cfg, A, B, Dq, Salpha)
    Qstage, Nstage, Rstage = _stage_coefficients(cfg)
    N, n, h = cfg["N"], cfg["state_dim"], cfg["h"]

    Pval = np.zeros((N + 1, n, n), dtype=float)
    c = np.zeros(N + 1, dtype=float)
    F = np.zeros((N, n), dtype=float)
    Lambda = np.zeros(N, dtype=float)
    e_q = np.zeros(n, dtype=float)
    e_q[0] = 1.0
    Pval[N] = cfg["params"]["kappa"] * np.outer(e_q, e_q)

    for k in range(N - 1, -1, -1):
        Pn = Pval[k + 1]
        joint = np.zeros((n + 1, n + 1), dtype=float)
        joint[:n, :n] = Qstage + A.T @ Pn @ A
        joint[:n, n] = Nstage + A.T @ Pn @ B
        joint[n, :n] = joint[:n, n]
        joint[n, n] = Rstage + B @ Pn @ B + h * Dq @ Pn @ Dq
        lam = joint[n, n]
        if require_positive and lam <= 0.0:
            raise ValueError(
                f"P4 dense Riccati loses Bellman convexity at k={k}: "
                f"Lambda={lam:.16g}"
            )
        Pk = joint[:n, :n] - np.outer(joint[:n, n], joint[n, :n]) / lam
        Pval[k] = 0.5 * (Pk + Pk.T)
        F[k] = -joint[:n, n] / lam
        Lambda[k] = lam
        c[k] = c[k + 1] + 0.5 * h * Salpha @ Pn @ Salpha

    return dict(Pval=Pval, c=c, F=F, Lambda=Lambda,
                Lambda_over_h=Lambda / h,
                api_version=ORACLE_API_VERSION)


def detached_curvature(cfg, *, require_positive: bool = True) -> dict:
    """Return the detached fixed-control Hessian ``Gol`` and ``Pi=Gol_QQ``.

    ``Gol`` is a Lyapunov recursion, not a Riccati recursion.  The impact
    cross term is linear in the state with control detached, and the fill
    diffusion is state independent, so neither contributes a state Hessian.
    """
    A, _B, _Dq, _Salpha = (np.asarray(x, dtype=float)
                            for x in linear_matrices(cfg))
    Qstage, _Nstage, _Rstage = _stage_coefficients(cfg)
    N, n = cfg["N"], cfg["state_dim"]
    Gol = np.zeros((N + 1, n, n), dtype=float)
    e_q = np.zeros(n, dtype=float)
    e_q[0] = 1.0
    Gol[N] = cfg["params"]["kappa"] * np.outer(e_q, e_q)
    for k in range(N - 1, -1, -1):
        Gk = Qstage + A.T @ Gol[k + 1] @ A
        Gol[k] = 0.5 * (Gk + Gk.T)

    Pi = Gol[:, 0, 0].copy()
    sigma_q = cfg["params"]["sigma_Q"]
    recovery_curvature = cfg["params"]["eta"] + sigma_q**2 * Pi
    if require_positive and np.min(recovery_curvature[:N]) <= 0.0:
        k = int(np.argmin(recovery_curvature[:N]))
        raise ValueError(
            f"P4 recovery curvature is nonpositive at k={k}: "
            f"eta+sigma_Q^2 Pi={recovery_curvature[k]:.16g}"
        )
    return dict(Gol=Gol, Pi=Pi,
                recovery_curvature=recovery_curvature,
                api_version=ORACLE_API_VERSION)


def curvature_certificate(cfg, oracle=None, detached=None) -> dict:
    """Report and enforce the two distinct P4 curvature gates."""
    if oracle is None:
        oracle = riccati(cfg)
    if detached is None:
        detached = detached_curvature(cfg)
    lam = np.asarray(oracle["Lambda"], dtype=float)
    rec = np.asarray(detached["recovery_curvature"], dtype=float)[: cfg["N"]]
    if np.min(lam) <= 0.0:
        raise ValueError("P4 Bellman Lambda must stay strictly positive")
    if np.min(rec) <= 0.0:
        raise ValueError("P4 eta+sigma_Q^2 Pi must stay strictly positive")
    return dict(
        min_Lambda=float(np.min(lam)),
        min_Lambda_over_h=float(np.min(lam) / cfg["h"]),
        min_recovery_curvature=float(np.min(rec)),
        max_abs_Pval_minus_Gol=float(
            np.max(np.abs(oracle["Pval"] - detached["Gol"]))
        ),
    )


def quadratic_value(z, P, c):
    """Evaluate ``1/2 z'Pz+c`` for a single augmented state."""
    z = np.asarray(z, dtype=float)
    return float(0.5 * z @ P @ z + c)


def bellman_rhs(cfg, oracle, k: int, z, u: float) -> float:
    """Exact one-step Bellman RHS using four-point noise quadrature.

    Independent Rademacher increments ``+/-sqrt(h)`` reproduce all moments
    needed by this quadratic problem exactly.  Gaussian increments remain the
    convention for stochastic rollouts; this helper is algebraic only.
    """
    if not 0 <= k < cfg["N"]:
        raise IndexError("k must index a P4 control stage")
    z = np.asarray(z, dtype=float)
    if z.shape != (cfg["state_dim"],):
        raise ValueError("z has the wrong shape")
    A, B, Dq, Salpha = (np.asarray(x, dtype=float)
                         for x in linear_matrices(cfg))
    m = A @ z + B * float(u)
    root_h = np.sqrt(cfg["h"])
    continuation = 0.0
    for sign_q in (-1.0, 1.0):
        for sign_alpha in (-1.0, 1.0):
            zn = (m + Dq * float(u) * sign_q * root_h
                  + Salpha * sign_alpha * root_h)
            continuation += quadratic_value(
                zn, oracle["Pval"][k + 1], oracle["c"][k + 1]
            )
    continuation *= 0.25
    return float(running_cost(cfg, z, float(u)) + continuation)


def exact_recovery_inputs(k: int, z, cfg, oracle, detached) -> dict:
    """Return exact q-form and recovered-form inputs at one state.

    ``p_cur`` is the current value gradient.  ``p_nxt`` is the conditional
    next-step gradient used by the exact Euler q-form FOC.  They are returned
    separately so a finite-grid alignment difference cannot be mistaken for
    an estimator error.
    """
    if not 0 <= k < cfg["N"]:
        raise IndexError("k must index a P4 control stage")
    z = np.asarray(z, dtype=float)
    if z.shape != (cfg["state_dim"],):
        raise ValueError("z has the wrong shape")
    A, B, Dq, Salpha = (np.asarray(x, dtype=float)
                         for x in linear_matrices(cfg))
    p = cfg["params"]

    u = float(oracle["F"][k] @ z)
    mean_next = A @ z + B * u
    p_cur_vec = oracle["Pval"][k] @ z
    p_nxt_vec = oracle["Pval"][k + 1] @ mean_next

    # q^{.,Q}=E_k[p_{k+1} dW_Q]/h and similarly for the signal column.
    q_fill_vec = oracle["Pval"][k + 1] @ (Dq * u)
    q_signal_vec = oracle["Pval"][k + 1] @ Salpha
    p_cur = float(p_cur_vec[0])
    p_nxt = float(p_nxt_vec[0])
    q_QQ = float(q_fill_vec[0])
    Pi = float(detached["Pi"][k])
    sigma_star_Q = float(Dq[0] * u)  # -sigma_Q*u
    zeta_QQ = q_QQ - Pi * sigma_star_Q
    I = float(impact(cfg, z))
    alpha = float(z[-1])

    recovery_curvature = p["eta"] + p["sigma_Q"] ** 2 * Pi
    if recovery_curvature <= 0.0:
        raise ValueError("nonpositive P4 recovery curvature")
    u_rec_pnxt = (
        alpha + p_nxt + p["sigma_Q"] * zeta_QQ - I
    ) / recovery_curvature
    # This is intentionally NOT an exact Euler-FOC recovery.  It inserts the
    # manuscript Stage-II current costate coordinate while holding the exact
    # q/Pi/zeta inputs fixed.  Its gap from u is the finite-grid alignment
    # floor and must not be counted as estimator or learned-policy error.
    u_rec_pcur = (
        alpha + p_cur + p["sigma_Q"] * zeta_QQ - I
    ) / recovery_curvature
    q_foc_residual = (
        p["eta"] * u + I - alpha - p_nxt - p["sigma_Q"] * q_QQ
    )
    recovered_residual = (
        recovery_curvature * u
        - (alpha + p_nxt + p["sigma_Q"] * zeta_QQ - I)
    )

    return dict(
        u=u,
        # ``u_rec`` is retained as a backwards-compatible alias for the exact
        # next-step-coordinate recovery.
        u_rec=float(u_rec_pnxt),
        u_rec_pnxt=float(u_rec_pnxt),
        u_rec_pcur=float(u_rec_pcur),
        p_alignment_action_gap=float(u_rec_pcur - u),
        p_alignment_schema_version=P_ALIGNMENT_SCHEMA_VERSION,
        p_cur=p_cur,
        p_nxt=p_nxt,
        p_cur_vec=p_cur_vec,
        p_nxt_vec=p_nxt_vec,
        q_QQ=q_QQ,
        q_fill_vec=q_fill_vec,
        q_signal_vec=q_signal_vec,
        Pi=Pi,
        zeta_QQ=float(zeta_QQ),
        sigma_star_Q=sigma_star_Q,
        impact=I,
        alpha=alpha,
        recovery_curvature=float(recovery_curvature),
        q_foc_residual=float(q_foc_residual),
        recovered_residual=float(recovered_residual),
    )


def self_checks(cfg, *, samples: int = 200, seed: int = 0) -> dict:
    """Run dense, Bellman, scalar-minimizer, FOC, and recovery checks."""
    from scipy.optimize import minimize_scalar

    if samples < 1:
        raise ValueError("samples must be positive")
    oracle = riccati(cfg)
    dense = dense_riccati(cfg)
    detached = detached_curvature(cfg)
    cert = curvature_certificate(cfg, oracle, detached)

    dense_error = max(
        float(np.max(np.abs(oracle[key] - dense[key])))
        for key in ("Pval", "c", "F", "Lambda")
    )
    rng = np.random.default_rng(seed)
    bellman_error = 0.0
    scalar_action_error = 0.0
    foc_error = 0.0
    recovery_error = 0.0
    anchored_error = 0.0
    p_alignment_num = 0.0
    p_alignment_den = 0.0

    for _ in range(samples):
        k = int(rng.integers(0, cfg["N"]))
        z = rng.normal(0.0, 1.0, cfg["state_dim"])
        u = float(oracle["F"][k] @ z)
        rhs = bellman_rhs(cfg, oracle, k, z, u)
        lhs = quadratic_value(z, oracle["Pval"][k], oracle["c"][k])
        bellman_error = max(bellman_error, abs(rhs - lhs))

        width = 5.0 + abs(u)
        result = minimize_scalar(
            lambda candidate: bellman_rhs(cfg, oracle, k, z, candidate),
            bounds=(u - width, u + width),
            method="bounded",
            options={"xatol": 1e-12},
        )
        scalar_action_error = max(scalar_action_error, abs(float(result.x) - u))

        target = exact_recovery_inputs(k, z, cfg, oracle, detached)
        foc_error = max(foc_error, abs(target["q_foc_residual"]))
        recovery_error = max(recovery_error, abs(target["u_rec"] - u))
        anchored_error = max(anchored_error, abs(target["recovered_residual"]))
        p_alignment_num += (target["p_nxt"] - target["p_cur"]) ** 2
        p_alignment_den += target["p_cur"] ** 2

    metrics = dict(
        dense_max_abs_error=dense_error,
        bellman_max_abs_error=float(bellman_error),
        scalar_minimizer_max_action_error=float(scalar_action_error),
        q_foc_max_abs_error=float(foc_error),
        recovery_max_abs_action_error=float(recovery_error),
        recovered_identity_max_abs_error=float(anchored_error),
        p_nxt_vs_p_cur_nrmse=float(
            np.sqrt(p_alignment_num / max(p_alignment_den, np.finfo(float).tiny))
        ),
        **cert,
    )
    if dense_error >= 1e-12:
        raise AssertionError(f"P4 dense Riccati mismatch: {dense_error}")
    if bellman_error >= 1e-10:
        raise AssertionError(f"P4 Bellman residual too large: {bellman_error}")
    if scalar_action_error >= 2e-6:
        raise AssertionError(
            f"P4 scalar minimizer action mismatch: {scalar_action_error}"
        )
    if max(foc_error, recovery_error, anchored_error) >= 1e-10:
        raise AssertionError("P4 FOC/recovery identity check failed")
    return metrics


def run_checks(cfg=None) -> dict:
    """CLI-friendly exact-oracle check using the canonical P4 pilot config."""
    if cfg is None:
        from .config import load_config

        cfg = load_config("main")
    metrics = self_checks(cfg)
    for key, value in metrics.items():
        print(f"{key:40s} = {value:.12g}")
    return metrics


if __name__ == "__main__":
    run_checks()
