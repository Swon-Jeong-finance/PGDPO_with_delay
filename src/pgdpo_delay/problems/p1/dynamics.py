"""P1 dynamics utilities: history generators and oracle-feedback rollouts.
Frozen numbers live ONLY in configs/p1/*.yaml (see .config).
"""
import numpy as np
from .oracle import build_dense

def grid(T, delta, h):
    N, H = round(T/h), round(delta/h)
    assert np.isclose(N*h, T) and np.isclose(H*h, delta)
    tt = np.linspace(-delta, 0, H+1)[::-1]      # tt[0]=0 (current) .. -delta
    return N, H, tt

def make_hist(rng, n, tt, delta):
    """Single continuous initial history from three templates."""
    a = rng.uniform(-1.2, 1.2)
    j = rng.integers(3); ph = rng.uniform(0, 2*np.pi)
    if j == 0: return a*np.ones(n)
    if j == 1: return a*(1 + tt/delta)
    return a*np.cos(2*np.pi*tt/delta + ph)

def make_hist_pair(rng, n, tt, delta, npair):
    """Continuous equal-current pairs: x(theta) = x0 + amp*psi(theta) with
    psi(0)=0, so both members share the current value BY CONSTRUCTION."""
    tau = -tt/delta                               # 0 at current, 1 at oldest
    bases = np.stack([tau, np.sin(np.pi*tau), 1.0 - np.cos(2*np.pi*tau)])
    x0 = rng.uniform(-1.0, 1.0, npair)
    ia = rng.integers(0, 3, npair)
    ib = (ia + 1 + rng.integers(0, 2, npair)) % 3  # different basis
    aa = rng.uniform(-1.2, 1.2, npair); ab = rng.uniform(-1.2, 1.2, npair)
    Za = x0[:, None] + aa[:, None]*bases[ia]
    Zb = x0[:, None] + ab[:, None]*bases[ib]
    return Za, Zb

def feedback_path(orc, params, H, h, z0, k0, kend, rng):
    A, B, C, D, Sg = build_dense(params, H, h)
    Z = z0.copy()
    for j in range(k0, kend):
        u = orc["F"][j] @ Z + orc["f"][j]
        Z = A @ Z + B*u + (C @ Z + D*u + Sg)*rng.normal(0, np.sqrt(h))
    return Z

def rollout(orc, params, H, h, N, Np, rng, clip=None, Z0=None, snap_every=None):
    """Vectorised oracle-feedback rollout. Snapshots (if requested) store
    (k, Z_k) BEFORE the update (time-aligned with exact_recovery_inputs)."""
    A, B, C, D, Sg = build_dense(params, H, h)
    n = H + 1
    Z = (np.stack([make_hist(rng, n, np.linspace(-h*H, 0, n)[::-1], h*H)
                   for _ in range(Np)]) if Z0 is None else Z0.copy())
    U = np.zeros((Np, N)); X = np.zeros((Np, N+1)); X[:, 0] = Z[:, 0]
    snaps = []
    dWs = rng.normal(0, np.sqrt(h), (Np, N))
    for k in range(N):
        if snap_every and k % snap_every == 0:
            snaps.append((k, Z.copy()))
        u = Z @ orc["F"][k] + orc["f"][k]
        if clip is not None: u = np.clip(u, clip[0], clip[1])
        U[:, k] = u
        Z = Z @ A.T + np.outer(u, B) + (Z @ C.T + np.outer(u, D) + Sg)*dWs[:, k, None]
        X[:, k+1] = Z[:, 0]
    return U, X, snaps, dWs
