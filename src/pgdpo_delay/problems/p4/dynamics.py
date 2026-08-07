"""Shared discrete dynamics and objective for optional Problem 4.

State ordering is

``Z_k = (Q_k, Q_{k-1}, ..., Q_{k-H}, alpha_k)``.

Every method must use the pre-trade realized-fill impact in Appendix C.4,
Eq. (78).  In particular, ``I_k`` is a function of ``Z_k`` only: the current
decision ``u_k`` is excluded.  Controls are signed and are never clipped.
"""

import numpy as np


def impact_row(cfg) -> np.ndarray:
    """Return the length-``H+1`` state row for causal pre-trade impact.

    With ``G_j = gamma exp(-rho_G j h)``, Eq. (78) and
    ``Delta L_m = Q_m-Q_{m+1}`` give

    ``(-G_1, G_1-G_2, ..., G_{H-1}-G_H, G_H)``.

    Hence a constant prehistory has zero initial impact and the row contains
    no current-control coefficient.
    """
    p = cfg["params"]
    H, h = cfg["H"], cfg["h"]
    ages = np.arange(1, H + 1, dtype=float) * h
    G = p["gamma"] * np.exp(-p["rho_G"] * ages)
    row = np.empty(H + 1, dtype=float)
    row[0] = -G[0]
    if H > 1:
        row[1:H] = G[:-1] - G[1:]
    row[H] = G[-1]
    return row


def impact(cfg, Z) -> np.ndarray:
    """Evaluate ``I_k`` from one state or a batch of states."""
    Z = np.asarray(Z, dtype=float)
    if Z.shape[-1] != cfg["state_dim"]:
        raise ValueError(
            f"last state dimension must be {cfg['state_dim']}, got {Z.shape[-1]}"
        )
    return Z[..., : cfg["H"] + 1] @ impact_row(cfg)


def linear_matrices(cfg):
    """Return ``(A, B, Dq, Salpha)`` for the shared Euler state equation.

    The shapes are ``A: (n,n)`` and ``B,Dq,Salpha: (n,)``, where
    ``n=H+2``.  The update is

    ``Z_next = A Z + B u + Dq u dW_Q + Salpha dW_alpha``.
    """
    p = cfg["params"]
    H, h, n = cfg["H"], cfg["h"], cfg["state_dim"]
    A = np.zeros((n, n), dtype=float)
    A[0, 0] = 1.0
    A[np.arange(1, H + 1), np.arange(H)] = 1.0
    A[-1, -1] = 1.0 - p["kappa_alpha"] * h

    B = np.zeros(n, dtype=float)
    B[0] = -h
    Dq = np.zeros(n, dtype=float)
    Dq[0] = -p["sigma_Q"]
    Salpha = np.zeros(n, dtype=float)
    Salpha[-1] = p["sigma_alpha"]
    return A, B, Dq, Salpha


def initial_state(cfg, Np: int, rng: np.random.Generator) -> np.ndarray:
    """Sample the explicit initial law with constant ``q0`` prehistory."""
    if int(Np) != Np or Np < 1:
        raise ValueError("Np must be a positive integer")
    Np = int(Np)
    Z = np.empty((Np, cfg["state_dim"]), dtype=float)
    Z[:, : cfg["H"] + 1] = cfg["init"]["q0"]
    if cfg["init"]["alpha0_law"] == "deterministic":
        Z[:, -1] = cfg["init"]["alpha0_mean"]
    else:
        Z[:, -1] = rng.normal(
            cfg["init"]["alpha0_mean"], cfg["init"]["alpha0_std"], Np
        )
    return Z


def make_bank(cfg, Np: int, seed: int) -> dict:
    """Create an immutable-by-convention common-random-number rollout bank."""
    rng = np.random.default_rng(seed)
    Z0 = initial_state(cfg, Np, rng)
    shape = (int(Np), cfg["N"])
    dW_Q = rng.normal(0.0, np.sqrt(cfg["h"]), shape)
    # A separate draw from the same generator implements independent channels.
    dW_alpha = rng.normal(0.0, np.sqrt(cfg["h"]), shape)
    return dict(Z0=Z0, dW_Q=dW_Q, dW_alpha=dW_alpha,
                Np=int(Np), seed=int(seed))


def _validate_bank(cfg, bank) -> int:
    Np = int(bank["Np"])
    if np.asarray(bank["Z0"]).shape != (Np, cfg["state_dim"]):
        raise ValueError("bank.Z0 has the wrong shape")
    expected = (Np, cfg["N"])
    if np.asarray(bank["dW_Q"]).shape != expected:
        raise ValueError("bank.dW_Q has the wrong shape")
    if np.asarray(bank["dW_alpha"]).shape != expected:
        raise ValueError("bank.dW_alpha has the wrong shape")
    return Np


def _policy_values(policy, k: int, Z: np.ndarray) -> np.ndarray:
    u = np.asarray(policy(k, Z), dtype=float)
    if u.ndim == 0:
        u = np.full(Z.shape[0], float(u))
    if u.shape != (Z.shape[0],):
        raise ValueError(
            f"policy must return shape ({Z.shape[0]},), got {u.shape}"
        )
    if not np.all(np.isfinite(u)):
        raise FloatingPointError("policy returned a non-finite signed control")
    return u


def step(cfg, Z, u, dW_Q, dW_alpha) -> np.ndarray:
    """One vectorized Euler step for single states or equally shaped batches."""
    Z = np.asarray(Z, dtype=float)
    u = np.asarray(u, dtype=float)
    dW_Q = np.asarray(dW_Q, dtype=float)
    dW_alpha = np.asarray(dW_alpha, dtype=float)
    if Z.shape[-1] != cfg["state_dim"]:
        raise ValueError("Z has the wrong final dimension")
    A, B, Dq, Salpha = linear_matrices(cfg)
    return (
        Z @ A.T
        + np.expand_dims(u, -1) * B
        + np.expand_dims(u * dW_Q, -1) * Dq
        + np.expand_dims(dW_alpha, -1) * Salpha
    )


def running_cost(cfg, Z, u) -> np.ndarray:
    """One-step cost, including the outer factor ``h``."""
    Z = np.asarray(Z, dtype=float)
    u = np.asarray(u, dtype=float)
    p = cfg["params"]
    Q, alpha = Z[..., 0], Z[..., -1]
    I = impact(cfg, Z)
    return cfg["h"] * (
        0.5 * p["phi"] * Q**2
        + 0.5 * p["eta"] * u**2
        + u * I
        - alpha * u
    )


def simulate_from_bank(cfg, policy, bank, return_paths: bool = False) -> dict:
    """Roll out one signed policy on a fixed evaluation bank."""
    Np = _validate_bank(cfg, bank)
    Z = np.asarray(bank["Z0"], dtype=float).copy()
    cost = np.zeros(Np, dtype=float)
    if return_paths:
        paths = dict(Z=[Z.copy()], u=[], impact=[], realized_fill=[])

    for k in range(cfg["N"]):
        u = _policy_values(policy, k, Z)
        I = impact(cfg, Z)
        cost += running_cost(cfg, Z, u)
        dW_Q = np.asarray(bank["dW_Q"])[:, k]
        dW_alpha = np.asarray(bank["dW_alpha"])[:, k]
        dL = cfg["h"] * u + cfg["params"]["sigma_Q"] * u * dW_Q
        Z = step(cfg, Z, u, dW_Q, dW_alpha)
        if return_paths:
            paths["u"].append(u.copy())
            paths["impact"].append(I.copy())
            paths["realized_fill"].append(dL.copy())
            paths["Z"].append(Z.copy())

    cost += 0.5 * cfg["params"]["kappa"] * Z[:, 0] ** 2
    out = dict(cost=cost, Q_T=Z[:, 0].copy(), alpha_T=Z[:, -1].copy(),
               Z_T=Z.copy())
    if return_paths:
        out["paths"] = {key: np.asarray(value) for key, value in paths.items()}
    return out


def simulate(cfg, policy, Np: int, seed: int, return_paths: bool = False) -> dict:
    """Roll out ``policy(k,Z)->u``; reuse ``seed`` for CRN comparisons."""
    bank = make_bank(cfg, Np, seed)
    return simulate_from_bank(cfg, policy, bank, return_paths=return_paths)


def simulate_paired(cfg, polA, polB, Np: int, seed: int) -> dict:
    """Common-noise paired objective comparison for two signed policies."""
    bank = make_bank(cfg, Np, seed)
    cost_A = simulate_from_bank(cfg, polA, bank)["cost"]
    cost_B = simulate_from_bank(cfg, polB, bank)["cost"]
    delta = cost_A - cost_B
    se = (float(delta.std(ddof=1) / np.sqrt(int(Np)))
          if int(Np) > 1 else 0.0)
    return dict(
        J_A=float(cost_A.mean()),
        J_B=float(cost_B.mean()),
        delta_A_minus_B=float(delta.mean()),
        se=se,
    )
