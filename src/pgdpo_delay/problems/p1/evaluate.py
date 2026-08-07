"""P1-C evaluation layer (review sec.5, sec.9): policy-agnostic metrics for the
H=16 main variant (and reusable for the small audit).

Policies are callables  pol(k, Z) -> u  (Z: (Np, H+1) buffers). Provided:
  rollout_paired   : common-random-number paired dJ (headline 5.1)
  estimator_inputs : INDEPENDENT-bank frozen-policy estimator for (p, zeta, Pi)
                     at a state, with optional no-anticipation tangents
                     (the delayed re-entry entries A[0,H], C[0,H] are removed
                     from the TANGENT propagation only; rollouts unchanged)
  kkt_residual     : box KKT residual at the deployed action (headline 5.2)
  active_set_stats : occupancy / transitions / first switching time (5.3)
  regime_disagreement: paired-history or paired-policy regime labels (5.3-5.4)
Hamiltonian gain is intentionally NOT a headline metric (review sec.6).
"""
import numpy as np
from .oracle import build_dense
from .dynamics import make_hist

def rollout_paired(cfg, polA, polB, Np, seed):
    p, h, N, H = cfg["params"], cfg["h"], cfg["N"], cfg["H"]
    A, B, C, D, Sg = build_dense(p, H, h)
    rng = np.random.default_rng(seed)
    Z0 = np.stack([make_hist(rng, H+1, cfg["tt"], cfg["delta"]) for _ in range(Np)])
    dW = rng.normal(0, np.sqrt(h), (Np, N))
    out = []
    for pol in (polA, polB):
        Z = Z0.copy(); cost = np.zeros(Np)
        for k in range(N):
            u = np.clip(pol(k, Z), *cfg["bounds"])
            cost += h*(0.5*p["Q"]*(Z[:, 0]-cfg["xref"][k])**2 + 0.5*p["R"]*u*u)
            Z = Z @ A.T + np.outer(u, B) + (Z @ C.T + np.outer(u, D) + Sg)*dW[:, k, None]
        cost += 0.5*p["QT"]*(Z[:, 0]-cfg["xtar"])**2
        out.append(cost)
    d = out[0] - out[1]
    return d.mean(), d.std(ddof=1)/np.sqrt(Np)

def _tangent_mats(cfg, no_anticipation):
    A, B, C, D, Sg = build_dense(cfg["params"], cfg["H"], cfg["h"])
    At, Ct = A.copy(), C.copy()
    if no_anticipation:
        At[0, cfg["H"]] = 0.0; Ct[0, cfg["H"]] = 0.0
    return A, B, C, D, Sg, At, Ct

def estimator_inputs(cfg, pol, k, z, M, Mout, Min, seed, no_anticipation=False):
    """Frozen-policy conditional estimators at (k, z): manuscript p^_k (lam0),
    Pi^_k, nested zeta^ anchored at the deployed action. Independent bank."""
    p, h, N, H = cfg["params"], cfg["h"], cfg["N"], cfg["H"]
    Q, QT = p["Q"], p["QT"]
    A, B, C, D, Sg, At, Ct = _tangent_mats(cfg, no_anticipation)
    rng = np.random.default_rng(seed); n = H+1
    Z = np.tile(z, (M, 1))
    lam0 = np.zeros(M); PiT = np.zeros(M)
    Vk = np.zeros((M, n)); Vk[:, 0] = 1.0
    for j in range(k, N):
        u = np.clip(pol(j, Z), *cfg["bounds"])
        lam0 += h*Q*(Z[:, 0]-cfg["xref"][j])*Vk[:, 0]
        PiT += h*Q*Vk[:, 0]**2
        dW = rng.normal(0, np.sqrt(h), M)
        cv = Vk @ Ct[0]
        Vk = Vk @ At.T; Vk[:, 0] += dW*cv
        Z = Z @ A.T + np.outer(u, B) + (Z @ C.T + np.outer(u, D) + Sg)*dW[:, None]
    lam0 += QT*(Z[:, 0]-cfg["xtar"])*Vk[:, 0]
    PiT += QT*Vk[:, 0]**2
    p_hat, Pi_hat = lam0.mean(), PiT.mean()
    # nested antithetic CRN zeta, anchored at sigma(z, deployed action)
    u0 = float(np.clip(pol(k, z[None, :]), *cfg["bounds"])[0])
    m = A @ z + B*u0; sv = C @ z + D*u0 + Sg; sig_star = sv[0]
    d = rng.normal(0, np.sqrt(h), Mout)
    Zb = np.concatenate([np.repeat(m[None, :] + d[:, None]*sv[None, :], Min, 0),
                         np.repeat(m[None, :] - d[:, None]*sv[None, :], Min, 0)], 0)
    V = np.zeros((Zb.shape[0], n)); V[:, 0] = 1.0
    lam = np.zeros(Zb.shape[0])
    inner = rng.normal(0, np.sqrt(h), (Mout*Min, N-(k+1)))
    noise = np.concatenate([inner, inner], 0)
    for jj, j in enumerate(range(k+1, N)):
        lam += h*Q*(Zb[:, 0]-cfg["xref"][j])*V[:, 0]
        dWj = noise[:, jj]
        cv = V @ Ct[0]
        V = V @ At.T; V[:, 0] += dWj*cv
        u = np.clip(pol(j, Zb), *cfg["bounds"])
        Zb = Zb @ A.T + np.outer(u, B) + (Zb @ C.T + np.outer(u, D) + Sg)*dWj[:, None]
    lam += QT*(Zb[:, 0]-cfg["xtar"])*V[:, 0]
    lam = lam.reshape(2, Mout, Min).mean(axis=2)
    y = 0.5*(lam[0] - lam[1])
    ze = (d @ (y - d*(Pi_hat*sig_star)))/(d @ d)
    return dict(u=u0, p=p_hat, Pi=Pi_hat, zeta=ze, sigma_star=sig_star,
                sigma_bar=sig_star - p["gu"]*u0)

def kkt_residual(cfg, inp, tol=1e-9):
    """Box KKT residual of the estimated generalized Hamiltonian at the
    deployed action (minimisation convention)."""
    p = cfg["params"]; lo, hi = cfg["bounds"]; u = inp["u"]
    g = p["R"]*u + p["b"]*inp["p"] + p["gu"]*(inp["zeta"]
        + inp["Pi"]*(inp["sigma_bar"] + p["gu"]*u))
    if u <= lo + tol:  return max(0.0, -g)
    if u >= hi - tol:  return max(0.0, g)
    return abs(g)

def active_set_stats(cfg, pol, Np, seed):
    p, h, N, H = cfg["params"], cfg["h"], cfg["N"], cfg["H"]
    A, B, C, D, Sg = build_dense(p, H, h)
    lo, hi = cfg["bounds"]
    rng = np.random.default_rng(seed)
    Z = np.stack([make_hist(rng, H+1, cfg["tt"], cfg["delta"]) for _ in range(Np)])
    occ = np.zeros(3); trans = np.zeros(Np); first = np.full(Np, np.nan); prev = None
    for k in range(N):
        u = np.clip(pol(k, Z), lo, hi)
        reg = np.where(u <= lo+1e-9, -1, np.where(u >= hi-1e-9, 1, 0))
        occ += [(reg == -1).mean(), (reg == 0).mean(), (reg == 1).mean()]
        if prev is not None:
            ch = reg != prev
            trans += ch
            first = np.where(np.isnan(first) & ch, k*h, first)
        prev = reg
        dW = rng.normal(0, np.sqrt(h), Np)
        Z = Z @ A.T + np.outer(u, B) + (Z @ C.T + np.outer(u, D) + Sg)*dW[:, None]
    return dict(occ=occ/N, transitions=trans.mean(),
                first_switch=np.nanmean(first), switched_frac=np.mean(~np.isnan(first)))

def regime_disagreement(cfg, polA, polB, states):
    lo, hi = cfg["bounds"]; dis = 0
    lab = lambda u: -1 if u <= lo+1e-9 else (1 if u >= hi-1e-9 else 0)
    for (k, z) in states:
        ua = float(np.clip(polA(k, z[None, :]), lo, hi)[0])
        ub = float(np.clip(polB(k, z[None, :]), lo, hi)[0])
        dis += lab(ua) != lab(ub)
    return dis/len(states)
