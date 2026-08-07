"""Calibration and certification diagnostics for the P4 exact reference.

This module deliberately stays on the non-deep reference track.  It checks
the causal realised-fill convention, evaluates the exact Riccati policy on a
fixed Gaussian holdout bank, and measures whether the chosen P4 calibration
has a visible signal and delay-history response.  It does not implement an
NMPC or learned competitor.
"""

from __future__ import annotations

import numpy as np

from . import dynamics


def dynamics_contract_checks(cfg, *, Np: int = 32, seed: int = 41) -> dict:
    """Numerically pin Eq. (78), realised fills, and pre-trade timing."""
    bank = dynamics.make_bank(cfg, Np=Np, seed=seed)
    policy = lambda k, Z: 0.7 + 0.02 * k + 0.15 * Z[:, -1]
    out = dynamics.simulate_from_bank(
        cfg, policy, bank, return_paths=True
    )["paths"]

    p, H, h = cfg["params"], cfg["H"], cfg["h"]
    G = p["gamma"] * np.exp(-p["rho_G"] * h * np.arange(1, H + 1))
    convolution_error = 0.0
    for k in range(cfg["N"]):
        direct = np.zeros(Np)
        for j in range(1, min(H, k) + 1):
            direct += G[j - 1] * out["realized_fill"][k - j]
        convolution_error = max(
            convolution_error,
            float(np.max(np.abs(out["impact"][k] - direct))),
        )

    fill_identity_error = float(np.max(np.abs(
        out["realized_fill"]
        - (out["Z"][:-1, :, 0] - out["Z"][1:, :, 0])
    )))
    row = dynamics.impact_row(cfg)
    initial_impact_error = float(np.max(np.abs(out["impact"][0])))

    # The impact API takes the pre-decision state only.  Evaluating it beside
    # two different current controls makes the no-current-action convention
    # explicit in the certificate without inventing a continuous-time weight.
    z = out["Z"][min(3, cfg["N"] - 1)]
    _u_a = np.full(Np, -2.0)
    _u_b = np.full(Np, +3.0)
    impact_a = dynamics.impact(cfg, z)
    impact_b = dynamics.impact(cfg, z)
    current_action_gap = float(np.max(np.abs(impact_a - impact_b)))

    return dict(
        impact_row_sum=float(row.sum()),
        initial_impact_max_abs=initial_impact_error,
        impact_convolution_max_abs_error=convolution_error,
        realized_fill_identity_max_abs_error=fill_identity_error,
        current_action_impact_max_abs_gap=current_action_gap,
    )


def _oracle_rollout(cfg, oracle, bank) -> tuple[dict, np.ndarray]:
    """Streaming exact-policy rollout with feasibility/alignment diagnostics."""
    from .oracle import P_ALIGNMENT_SCHEMA_VERSION, detached_curvature

    Np = int(bank["Np"])
    Z = np.asarray(bank["Z0"], dtype=float).copy()
    z0 = Z.copy()
    cost = np.zeros(Np)
    min_Q = Z[:, 0].copy()
    sell_volume = np.zeros(Np)
    buy_volume = np.zeros(Np)
    controls = np.empty((cfg["N"], Np), dtype=float)
    A, B, _Dq, _Salpha = dynamics.linear_matrices(cfg)
    detached = detached_curvature(cfg)
    alignment_squared_error = 0.0
    oracle_control_squared = 0.0

    for k in range(cfg["N"]):
        u = Z @ oracle["F"][k]
        controls[k] = u

        # p_cur is the manuscript Stage-II coordinate; p_nxt is the exact
        # same-grid Euler-FOC coordinate.  Exact q/Pi/zeta recovery implies
        # u_rec_pcur-u=(p_cur-p_nxt)/(eta+sigma_Q^2 Pi), so no additional
        # stochastic branch simulation is needed to measure this floor.
        p_cur = Z @ oracle["Pval"][k, :, 0]
        p_nxt_column = oracle["Pval"][k + 1, :, 0]
        p_nxt = (Z @ (A.T @ p_nxt_column)
                 + u * float(B @ p_nxt_column))
        action_gap = (
            (p_cur - p_nxt) / detached["recovery_curvature"][k]
        )
        alignment_squared_error += float(action_gap @ action_gap)
        oracle_control_squared += float(u @ u)

        cost += dynamics.running_cost(cfg, Z, u)
        sell_volume += cfg["h"] * np.maximum(u, 0.0)
        buy_volume += cfg["h"] * np.maximum(-u, 0.0)
        Z = dynamics.step(
            cfg, Z, u, bank["dW_Q"][:, k], bank["dW_alpha"][:, k]
        )
        min_Q = np.minimum(min_Q, Z[:, 0])

    cost += 0.5 * cfg["params"]["kappa"] * Z[:, 0] ** 2
    value0 = (
        0.5 * np.einsum("bi,ij,bj->b", z0, oracle["Pval"][0], z0)
        + oracle["c"][0]
    )
    return dict(
        cost=cost,
        value0=value0,
        Q_T=Z[:, 0].copy(),
        min_Q=min_Q,
        sell_volume=sell_volume,
        buy_volume=buy_volume,
        controls=controls,
        u_rms=float(np.sqrt(np.mean(controls ** 2))),
        p_alignment_action_rmse=float(np.sqrt(
            alignment_squared_error / (Np * cfg["N"])
        )),
        p_alignment_action_nrmse=float(np.sqrt(
            alignment_squared_error
            / max(oracle_control_squared, np.finfo(float).tiny)
        )),
        p_alignment_schema_version=int(P_ALIGNMENT_SCHEMA_VERSION),
    ), Z


def history_response(cfg, oracle, *, u_rms: float) -> dict:
    """Evaluate a deterministic bank of reachable zero-net-fill histories.

    Every pair has the same current inventory, signal, and oldest inventory.
    It differs only by a recent/far pair of equal and opposite realised fills,
    so the diagnostic cannot be explained by a different current state or net
    historical liquidation.
    """
    H, n, q0 = cfg["H"], cfg["state_dim"], cfg["params"]["q0"]
    if H < 20 or cfg["N"] < 100:
        raise ValueError(
            "canonical P4 history bank requires H>=20 and N>=100"
        )
    stages = (20, 40, 60, 80, cfg["N"] - 1)
    q_levels = (0.25 * q0, 0.50 * q0, 1.00 * q0)
    alpha_scale = (
        cfg["params"]["sigma_alpha"]
        / np.sqrt(2.0 * cfg["params"]["kappa_alpha"])
    )
    alpha_levels = (-alpha_scale, 0.0, alpha_scale)
    near_lags = (1, 3, 5)
    far_lags = (10, 15, 20)
    amplitudes = (0.02 * q0, 0.05 * q0, 0.10 * q0)

    delta_u = []
    delta_I = []
    for k in stages:
        for q in q_levels:
            for alpha in alpha_levels:
                for near in near_lags:
                    for far in far_lags:
                        for amplitude in amplitudes:
                            for orientation in (-1.0, 1.0):
                                # ell_j = Q_{k-j}-Q_{k-j+1}.  The two nonzero
                                # pulses sum to zero; B receives the negative.
                                fills = np.zeros(H + 1)
                                fills[near] = orientation * amplitude / 2.0
                                fills[far] = -orientation * amplitude / 2.0
                                ZA = np.empty(n)
                                ZA[0] = q
                                ZA[1:H + 1] = q + np.cumsum(fills[1:])
                                ZA[-1] = alpha
                                ZB = ZA.copy()
                                ZB[1:H + 1] = q - np.cumsum(fills[1:])
                                du = oracle["F"][k] @ (ZA - ZB)
                                dI = (dynamics.impact(cfg, ZA)
                                      - dynamics.impact(cfg, ZB))
                                delta_u.append(float(du))
                                delta_I.append(float(dI))

    delta_u = np.asarray(delta_u)
    delta_I = np.asarray(delta_I)
    corr = float(np.corrcoef(delta_u, delta_I)[0, 1])
    du_rms = float(np.sqrt(np.mean(delta_u ** 2)))
    dI_rms = float(np.sqrt(np.mean(delta_I ** 2)))
    scale = max(float(u_rms), 1e-12 * q0 / cfg["T"])
    return dict(
        history_pair_count=int(delta_u.size),
        history_delta_u_rms=du_rms,
        history_delta_impact_rms=dI_rms,
        history_response_ratio=float(du_rms / scale),
        history_du_impact_correlation=corr,
    )


def signal_response(cfg, oracle, *, u_rms: float) -> dict:
    """Normalize the policy's signal gain by a stationary OU scale."""
    p = cfg["params"]
    alpha_scale = p["sigma_alpha"] / np.sqrt(2.0 * p["kappa_alpha"])
    gain = np.asarray(oracle["F"][:, -1], dtype=float)
    scale = max(float(u_rms), 1e-12 * p["q0"] / cfg["T"])
    return dict(
        signal_probe_scale=float(alpha_scale),
        signal_gain_min=float(gain.min()),
        signal_gain_max=float(gain.max()),
        signal_response_ratio=float(alpha_scale * np.sqrt(np.mean(gain ** 2))
                                    / scale),
    )


def rollout_diagnostics(cfg, oracle, *, Np: int, seed: int) -> dict:
    """Production feasibility and value check on one fixed holdout bank."""
    if Np < 2:
        raise ValueError("P4 rollout diagnostics require Np>=2")
    bank = dynamics.make_bank(cfg, Np=Np, seed=seed)
    rollout, _ = _oracle_rollout(cfg, oracle, bank)
    q0, T = cfg["params"]["q0"], cfg["T"]

    residual = rollout["cost"] - rollout["value0"]
    se = float(residual.std(ddof=1) / np.sqrt(Np))
    negative_excursion = np.maximum(-rollout["min_Q"], 0.0) / q0
    abs_terminal = np.abs(rollout["Q_T"]) / q0
    buy = rollout["buy_volume"] / q0
    sell = rollout["sell_volume"] / q0
    total = buy + sell
    roundtrip = 2.0 * np.minimum(buy, sell) / np.maximum(total, 1e-15)
    net_sell = sell - buy

    constant = lambda k, Z: np.full(Z.shape[0], q0 / T)
    const_cost = dynamics.simulate_from_bank(cfg, constant, bank)["cost"]
    improvement = const_cost - rollout["cost"]
    improvement_se = float(improvement.std(ddof=1) / np.sqrt(Np))

    controls = rollout["controls"]
    result = dict(
        Np=int(Np),
        seed=int(seed),
        oracle_objective_mean=float(rollout["cost"].mean()),
        exact_initial_value_mean=float(rollout["value0"].mean()),
        mc_minus_value_mean=float(residual.mean()),
        mc_minus_value_se=se,
        terminal_abs_mean_ratio=float(abs_terminal.mean()),
        terminal_abs_p95_ratio=float(np.quantile(abs_terminal, 0.95)),
        overshoot_mean_ratio=float(negative_excursion.mean()),
        overshoot_p95_ratio=float(np.quantile(negative_excursion, 0.95)),
        overshoot_p99_ratio=float(np.quantile(negative_excursion, 0.99)),
        material_overshoot_probability=float(np.mean(negative_excursion > 0.05)),
        buy_volume_mean_ratio=float(buy.mean()),
        buy_volume_p95_ratio=float(np.quantile(buy, 0.95)),
        intended_roundtrip_mean=float(roundtrip.mean()),
        intended_roundtrip_p95=float(np.quantile(roundtrip, 0.95)),
        nonpositive_net_sell_probability=float(np.mean(net_sell <= 0.0)),
        u_rms=rollout["u_rms"],
        u_abs_p999=float(np.quantile(np.abs(controls), 0.999)),
        p_alignment_action_rmse=rollout["p_alignment_action_rmse"],
        p_alignment_action_nrmse=rollout["p_alignment_action_nrmse"],
        p_alignment_schema_version=rollout[
            "p_alignment_schema_version"
        ],
        constant_objective_mean=float(const_cost.mean()),
        constant_minus_oracle_mean=float(improvement.mean()),
        constant_minus_oracle_se=improvement_se,
        constant_relative_improvement=float(
            improvement.mean() / max(abs(float(const_cost.mean())), 1e-15)
        ),
    )
    result.update(history_response(cfg, oracle, u_rms=rollout["u_rms"]))
    result.update(signal_response(cfg, oracle, u_rms=rollout["u_rms"]))
    return result


def diagnostic_paths(cfg, oracle, *, Np: int = 16, seed: int = 53) -> dict:
    """Return a small audit bank with exact recovery targets.

    Both ``p_cur`` and the Euler-FOC ``p_nxt`` are stored so downstream code
    cannot silently substitute one for the other.  The target coordinates are
    the same ``(q_QQ, Pi, zeta_QQ)`` convention used by
    :func:`p4.oracle.exact_recovery_inputs`.
    """
    from .oracle import P_ALIGNMENT_SCHEMA_VERSION, detached_curvature

    bank = dynamics.make_bank(cfg, Np=Np, seed=seed)
    policy = lambda k, Z: Z @ oracle["F"][k]
    out = dynamics.simulate_from_bank(
        cfg, policy, bank, return_paths=True
    )["paths"]
    Z, u = out["Z"], out["u"]
    step_cost = np.stack([
        dynamics.running_cost(cfg, Z[k], u[k]) for k in range(cfg["N"])
    ])
    cumulative = np.vstack((np.zeros((1, Np)), np.cumsum(step_cost, axis=0)))
    terminal = 0.5 * cfg["params"]["kappa"] * Z[-1, :, 0] ** 2
    cumulative[-1] += terminal
    A, B, Dq, Salpha = dynamics.linear_matrices(cfg)
    detached = detached_curvature(cfg)
    p_cur = np.empty_like(u)
    p_nxt = np.empty_like(u)
    q_QQ = np.empty_like(u)
    q_Qalpha = np.empty_like(u)
    zeta_QQ = np.empty_like(u)
    u_rec_pnxt = np.empty_like(u)
    u_rec_pcur = np.empty_like(u)
    for k in range(cfg["N"]):
        mean_next = Z[k] @ A.T + u[k, :, None] * B
        Pk = oracle["Pval"][k]
        Pn = oracle["Pval"][k + 1]
        p_cur[k] = Z[k] @ Pk[:, 0]
        p_nxt[k] = mean_next @ Pn[:, 0]
        q_QQ[k] = u[k] * float((Pn @ Dq)[0])
        q_Qalpha[k] = float((Pn @ Salpha)[0])
        sigma_star = -cfg["params"]["sigma_Q"] * u[k]
        zeta_QQ[k] = q_QQ[k] - detached["Pi"][k] * sigma_star
        common = (Z[k, :, -1]
                  + cfg["params"]["sigma_Q"] * zeta_QQ[k]
                  - out["impact"][k])
        u_rec_pnxt[k] = (
            common + p_nxt[k]
        ) / detached["recovery_curvature"][k]
        u_rec_pcur[k] = (
            common + p_cur[k]
        ) / detached["recovery_curvature"][k]
    alignment_gap = u_rec_pcur - u
    alignment_squared_error = float(np.sum(alignment_gap ** 2))
    oracle_control_squared = float(np.sum(u ** 2))
    return dict(
        Q=Z[:, :, 0],
        alpha=Z[:, :, -1],
        u=u,
        impact=out["impact"],
        realized_fill=out["realized_fill"],
        dW_Q=bank["dW_Q"].T,
        dW_alpha=bank["dW_alpha"].T,
        cumulative_cost=cumulative,
        p_cur=p_cur,
        p_nxt=p_nxt,
        q_QQ=q_QQ,
        q_Qalpha=q_Qalpha,
        Pi=detached["Pi"][:cfg["N"]],
        zeta_QQ=zeta_QQ,
        recovery_curvature=detached["recovery_curvature"][:cfg["N"]],
        # ``u_rec`` remains the legacy alias for exact p_nxt recovery.
        u_rec=u_rec_pnxt,
        u_rec_pnxt=u_rec_pnxt,
        u_rec_pcur=u_rec_pcur,
        p_alignment_action_gap=alignment_gap,
        # These two scalars describe only this small saved audit bank.  The
        # publication scalar uses the much larger rollout bank and is stored
        # under the unqualified p_alignment_action_* names in certification.
        diagnostic_bank_p_alignment_action_rmse=np.asarray(float(np.sqrt(
            alignment_squared_error / alignment_gap.size
        ))),
        diagnostic_bank_p_alignment_action_nrmse=np.asarray(float(np.sqrt(
            alignment_squared_error
            / max(oracle_control_squared, np.finfo(float).tiny)
        ))),
        p_alignment_schema_version=np.asarray(
            P_ALIGNMENT_SCHEMA_VERSION, dtype=np.int64
        ),
        recovery_action_max_abs_error=np.asarray(
            float(np.max(np.abs(u_rec_pnxt - u)))
        ),
        seed=np.asarray(seed),
    )
