"""P3-R forward simulator on the (I, M) Markov lift, SHARED by every method
(oracle rollouts, Stage I/II, baselines) so no scheme mismatch enters any
comparison (design-handoff sec.7.9).

Scheme convention (V1; wording per review 2026-08-07 sec.3.5):
full-truncation Euler with an EMPIRICAL nonnegativity audit -- drift/
diffusion coefficients, running cost, and the M-update all use the truncated
state Ib = max(I, 0); the stored I itself is NOT floored (canonical full
truncation), so pathwise positivity is NOT guaranteed mathematically.
Excursions below 0 are O(dt) transients (sigma ~ Ib vanishes at 0, drift at
I = 0 is beta*M >= 0) and the negative-hit fraction is a reported gate
(measured 0 under the V1 production calibration).
M stays >= 0 because dM = rho (Ib - M) dt with rho*dt <= 1 (validated in
config). Cost half-convention: (c_I/2) I^2 + (R/2) u^2, terminal (c_T/2) I^2.
"""
import numpy as np


def step(params, dt, I, M, u, dW):
    """One shared Euler step. I, M, u, dW are (Np,) arrays; returns (I', M')."""
    Ib = np.maximum(I, 0.0)
    drift = (params["beta"]*(1.0 - Ib/params["Npop"])*M
             - params["gamma"]*Ib - params["b"]*u*Ib)
    sig = params["sigma0"]*(1.0 - params["eta_sigma"]*u)*Ib
    In = I + drift*dt + sig*dW
    Mn = M + params["rho"]*(Ib - M)*dt
    return In, Mn


def simulate(cfg, policy, Np, seed, return_paths=False):
    """Rollout under policy(k, I, M) -> u in [0,1]; returns per-path cost.
    Common-random-number pairing: pass the same seed to both methods."""
    p, dt, N = cfg["params"], cfg["dt"], cfg["N"]
    rng = np.random.default_rng(seed)
    lo_i, hi_i = cfg["init"]["I0"]; lo_m, hi_m = cfg["init"]["M0"]
    I = rng.uniform(lo_i, hi_i, Np); M = rng.uniform(lo_m, hi_m, Np)
    dW = rng.normal(0.0, np.sqrt(dt), (Np, N))
    cost = np.zeros(Np)
    paths = dict(I=[I.copy()], M=[M.copy()], u=[]) if return_paths else None
    for k in range(N):
        u = np.clip(policy(k, I, M), *cfg["bounds"])
        Ib = np.maximum(I, 0.0)
        cost += dt*(0.5*p["c_I"]*Ib**2 + 0.5*p["R"]*u**2)
        I, M = step(p, dt, I, M, u, dW[:, k])
        if return_paths:
            paths["u"].append(u.copy()); paths["I"].append(I.copy())
            paths["M"].append(M.copy())
    cost += 0.5*p["c_T"]*np.maximum(I, 0.0)**2
    out = dict(cost=cost, I_T=I, M_T=M)
    if return_paths:
        out["paths"] = {k: np.array(v) for k, v in paths.items()}
    return out


def simulate_paired(cfg, polA, polB, Np, seed):
    """CRN-paired objective comparison (same initial states and Brownian bank)."""
    p, dt, N = cfg["params"], cfg["dt"], cfg["N"]
    rng = np.random.default_rng(seed)
    lo_i, hi_i = cfg["init"]["I0"]; lo_m, hi_m = cfg["init"]["M0"]
    I0 = rng.uniform(lo_i, hi_i, Np); M0 = rng.uniform(lo_m, hi_m, Np)
    dW = rng.normal(0.0, np.sqrt(dt), (Np, N))
    costs = []
    for pol in (polA, polB):
        I, M = I0.copy(), M0.copy(); cost = np.zeros(Np)
        for k in range(N):
            u = np.clip(pol(k, I, M), *cfg["bounds"])
            cost += dt*(0.5*p["c_I"]*np.maximum(I, 0.0)**2 + 0.5*p["R"]*u**2)
            I, M = step(p, dt, I, M, u, dW[:, k])
        cost += 0.5*p["c_T"]*np.maximum(I, 0.0)**2
        costs.append(cost)
    d = costs[0] - costs[1]
    return dict(J_A=float(costs[0].mean()), J_B=float(costs[1].mean()),
                delta_A_minus_B=float(d.mean()),
                se=float(d.std(ddof=1)/np.sqrt(Np)))


# ---------------------------------------------------------------------------
# P3-D distributed-incubation variant: truncated-Gamma kernel + I-buffer.
# ---------------------------------------------------------------------------

def kernel_weights(cfg):
    """Normalized trapezoidal weights of the truncated Gamma kernel on the
    tap grid a_j = j*dt, j = 0..H (endpoint halving, sum 1 -- the SAME
    discretisation convention as the P2 distributed kernel). K(0) = 0 for
    m_K > 1, so w_0 vanishes and M is strictly past-driven."""
    d = cfg["dist"]; dt = cfg["dt"]; H = d["H"]
    a = np.arange(H+1)*dt
    K = a**(d["m_K"]-1.0)*np.exp(-a/d["theta"])
    w = K.copy(); w[0] *= 0.5; w[-1] *= 0.5
    s = w.sum()
    if s <= 0: raise ValueError("degenerate kernel weights")
    return w/s


def init_history(cfg, Np, rng):
    """Initial I-history buffer B[:, j] = I(-j*dt), j = 0..H: linear ramp from
    I0 (at t=0) back to Ipast (at t=-delta). Convention V1 (documented for
    review): the ramp family gives history diversity without adding kernel
    machinery to the initial law."""
    H = cfg["dist"]["H"]
    I0 = rng.uniform(*cfg["init"]["I0"], Np)
    Ip = rng.uniform(*cfg["init"]["Ipast"], Np)
    frac = np.arange(H+1)/H
    return I0[:, None] + (Ip - I0)[:, None]*frac[None, :]


def step_dist(params, dt, B, w, u, dW):
    """One shared buffered Euler step (full-truncation coefficients as in
    `step`). B: (Np, H+1) with B[:,0] = I_k; returns the shifted buffer."""
    Ib = np.maximum(B[:, 0], 0.0)
    M = np.maximum(B, 0.0) @ w
    drift = (params["beta"]*(1.0 - Ib/params["Npop"])*M
             - params["gamma"]*Ib - params["b"]*u*Ib)
    sig = params["sigma0"]*(1.0 - params["eta_sigma"]*u)*Ib
    In = B[:, 0] + drift*dt + sig*dW
    Bn = np.empty_like(B); Bn[:, 0] = In; Bn[:, 1:] = B[:, :-1]
    return Bn


def simulate_dist(cfg, policy, Np, seed, return_paths=False):
    """Single-policy P3-D rollout on a fixed stochastic holdout bank."""
    p, dt, N = cfg["params"], cfg["dt"], cfg["N"]
    w = kernel_weights(cfg)
    rng = np.random.default_rng(seed)
    B = init_history(cfg, Np, rng)
    dW = rng.normal(0.0, np.sqrt(dt), (Np, N))
    cost = np.zeros(Np)
    paths = dict(B=[B.copy()], u=[]) if return_paths else None
    for k in range(N):
        u = np.clip(policy(k, B), *cfg["bounds"])
        cost += dt*(0.5*p["c_I"]*np.maximum(B[:, 0], 0.0)**2
                    + 0.5*p["R"]*u**2)
        B = step_dist(p, dt, B, w, u, dW[:, k])
        if return_paths:
            paths["u"].append(u.copy()); paths["B"].append(B.copy())
    cost += 0.5*p["c_T"]*np.maximum(B[:, 0], 0.0)**2
    out = dict(cost=cost, B_T=B)
    if return_paths:
        out["paths"] = {k: np.array(v) for k, v in paths.items()}
    return out


def simulate_dist_paired(cfg, polA, polB, Np, seed):
    """CRN-paired comparison for the distributed variant. Policies are
    pol(k, B) -> u with B the (Np, H+1) I-buffer."""
    p, dt, N = cfg["params"], cfg["dt"], cfg["N"]
    w = kernel_weights(cfg)
    rng = np.random.default_rng(seed)
    B0 = init_history(cfg, Np, rng)
    dW = rng.normal(0.0, np.sqrt(dt), (Np, N))
    costs = []
    for pol in (polA, polB):
        B = B0.copy(); cost = np.zeros(Np)
        for k in range(N):
            u = np.clip(pol(k, B), *cfg["bounds"])
            cost += dt*(0.5*p["c_I"]*np.maximum(B[:, 0], 0.0)**2
                        + 0.5*p["R"]*u**2)
            B = step_dist(p, dt, B, w, u, dW[:, k])
        cost += 0.5*p["c_T"]*np.maximum(B[:, 0], 0.0)**2
        costs.append(cost)
    d = costs[0] - costs[1]
    return dict(J_A=float(costs[0].mean()), J_B=float(costs[1].mean()),
                delta_A_minus_B=float(d.mean()),
                se=float(d.std(ddof=1)/np.sqrt(Np)))
