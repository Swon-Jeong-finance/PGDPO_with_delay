"""P1-4 calibration harness (post-review): gate metrics under the exact P1-U
oracle. Fixes applied per review 2026-08-06:
  (3.1) snapshots store Z_k before the update (time-aligned);
  (3.2) paired histories are continuous with psi(0)=0 (equal current by
        construction, no coordinate overwrite);
  (3.3) G2 is labelled delay visibility / history dependence; the drift metric
        is the delayed-FORWARD-drift share. Anticipation itself is measured by
        the separate full vs no-anticipation ablation, not here;
  (3.4) past-coefficient L1 ratio renamed; actual rollout past action-share
        added.
Tail audit (sec.7): 100k-rollout finiteness and 99% quantiles for V3.
G3 remains a clipped-unconstrained PROXY; exact P1-C DP re-audits occupancy.
"""
import json
import numpy as np
from .oracle import (ORACLE_API_VERSION, build_dense, riccati,
                     detached_curvature, exact_recovery_inputs)
from .config import load_config
from .dynamics import make_hist, make_hist_pair, rollout
assert ORACLE_API_VERSION == "p1-v3-pcur-pnext"

_CFG = load_config("main")
V3 = _CFG["params"]; P1C_BOUNDS = _CFG["bounds"]
T, delta, h = _CFG["T"], _CFG["delta"], _CFG["h"]
N, H, tt = _CFG["N"], _CFG["H"], _CFG["tt"]; n = H+1
xtar = _CFG["xtar"]; xref = _CFG["xref"]

def init_bank(rng, Np):
    return np.stack([make_hist(rng, n, tt, delta) for _ in range(Np)])

def gates(params, tag, rng):
    b, gu, R = params["b"], params["gu"], params["R"]
    orc = riccati(params, H, h, N, xref, xtar)
    G = detached_curvature(params, H, h, N)
    U, X, snaps, _ = rollout(orc, params, H, h, N, 2000, rng,
                             Z0=init_bank(rng, 2000), snap_every=8)
    u_rms = np.sqrt((U**2).mean()); u_pool = U.ravel()

    # G2 delay visibility / history dependence
    coefL1 = np.mean([np.abs(orc["F"][k][1:]).sum()/max(abs(orc["F"][k][0]), 1e-12)
                      for k in range(N)])
    Za, Zb = make_hist_pair(rng, n, tt, delta, 400)      # continuous, psi(0)=0
    du0 = np.abs((Za - Zb) @ orc["F"][0])
    Ua, _, _, _ = rollout(orc, params, H, h, N, 400, np.random.default_rng(9), Z0=Za)
    Ub, _, _, _ = rollout(orc, params, H, h, N, 400, np.random.default_rng(9), Z0=Zb)
    du_path = np.abs(Ua - Ub).mean()
    drift_sh = np.abs(params["ad"]*X[:, :N-H]).mean() / (
        np.abs(params["a"]*X[:, H:N]).mean() + np.abs(params["ad"]*X[:, :N-H]).mean()
        + np.abs(b*U[:, H:]).mean())

    # G4 channel shares + actual past action-share on time-aligned snapshots
    sq, sz, sig, pshare = [], [], [], []
    for (k, Zs) in snaps:
        Fk, fk = orc["F"][k], orc["f"][k]
        for z in Zs[::100]:
            t = exact_recovery_inputs(k, z, params, H, h, orc, G, xref)
            sq.append(abs(gu*t["q"])/(abs(b*t["p_nxt"]) + abs(gu*t["q"]) + 1e-12))
            sz.append(abs(gu*t["zeta"])/(abs(b*t["p_cur"]) + abs(gu*t["zeta"])
                      + abs(gu*t["Pi"]*t["sigma_bar"]) + 1e-12))
            sig.append(abs(t["sigma_star"]))
            c0, cp, cf = abs(Fk[0]*z[0]), abs(Fk[1:] @ z[1:]), abs(fk)
            pshare.append(cp/(c0 + cp + cf + 1e-12))
    Pi_min = min(G[k][0, 0] for k in range(N+1))

    # G3 box proxy (design-stage only)
    lo, hi = np.quantile(u_pool, [0.2, 0.8])
    Uc, _, _, _ = rollout(orc, params, H, h, N, 2000, np.random.default_rng(11),
                          clip=(lo, hi), Z0=init_bank(np.random.default_rng(12), 2000))
    at_lo = (Uc <= lo + 1e-9).mean(); at_hi = (Uc >= hi - 1e-9).mean()

    print(f"[{tag}] u_rms={u_rms:.3f}  mean|X|={np.abs(X).mean():.3f}  "
          f"mean|sigma*|={np.mean(sig):.3f}")
    print(f"  G2 delay visibility : past-coeff L1 ratio={coefL1:.3f}  "
          f"past action-share={np.mean(pshare):.3f}")
    print(f"      history pairs   : du(k=0)/u_rms={du0.mean()/u_rms:.2%}  "
          f"CRN path avg/u_rms={du_path/u_rms:.2%}  delayed-forward-drift share={drift_sh:.2f}")
    print(f"  G4 ctrl-diffusion   : FOC q-share={np.mean(sq):.2%}  "
          f"PathA zeta-share={np.mean(sz):.2%}  min(R+gu^2 Pi)={R+gu*gu*Pi_min:.3f}")
    print(f"  G3 box PROXY        : [{lo:+.3f},{hi:+.3f}]  "
          f"active lo/hi={at_lo:.1%}/{at_hi:.1%}  total={(at_lo+at_hi):.1%}")
    return dict(tag=tag, u_rms=u_rms, coefL1=coefL1, pshare=float(np.mean(pshare)),
                du0=float(du0.mean()/u_rms), du_path=float(du_path/u_rms),
                drift=float(drift_sh), qsh=float(np.mean(sq)), zsh=float(np.mean(sz)),
                lo=float(lo), hi=float(hi), act=float(at_lo+at_hi))

def tail_audit(params, Np=100_000):
    """Sec.7: multiplicative-noise tail audit for the adopted V3."""
    Q, R, QT = params["Q"], params["R"], params["QT"]
    A, B, C, D, Sg = build_dense(params, H, h)
    rng = np.random.default_rng(21)
    orc = riccati(params, H, h, N, xref, xtar)
    Z = init_bank(rng, Np)
    mX = np.abs(Z[:, 0]); mU = np.zeros(Np); cost = np.zeros(Np)
    for k in range(N):
        u = Z @ orc["F"][k] + orc["f"][k]
        cost += h*(0.5*Q*(Z[:, 0]-xref[k])**2 + 0.5*R*u*u)
        mU = np.maximum(mU, np.abs(u))
        dW = rng.normal(0, np.sqrt(h), Np)
        Z = Z @ A.T + np.outer(u, B) + (Z @ C.T + np.outer(u, D) + Sg)*dW[:, None]
        mX = np.maximum(mX, np.abs(Z[:, 0]))
    cost += 0.5*QT*(Z[:, 0]-xtar)**2
    assert np.isfinite(cost).all() and np.isfinite(mX).all()
    q = lambda v: np.quantile(v, 0.99)
    print(f"  tail audit (Np={Np}): all finite; 99%q max|X|={q(mX):.2f}  "
          f"99%q max|u|={q(mU):.2f}  99%q cost={q(cost):.2f}")

if __name__ == "__main__":
    base = dict(a=-0.3, ad=0.6, b=1.0, s0=0.2, cx=0.1, cy=0.15, gu=0.4,
                Q=1.0, R=0.1, QT=2.0)
    out = []
    for tag, upd in [("base", {}), ("V1 gu=0.7", dict(gu=0.7)),
                     ("V2 gu=0.7 cy=0.30", dict(gu=0.7, cy=0.30)),
                     ("V3 (adopted)", dict(gu=0.7, cy=0.30, ad=0.9))]:
        p = dict(base); p.update(upd)
        out.append(gates(p, tag, np.random.default_rng(3)))
    tail_audit(V3)
    print(f"  P1-C provisional: H=3 (h=delta/3), fixed bounds {P1C_BOUNDS} "
          f"(exact DP re-audit pending)")
    with open("p1_calibrate_results.json", "w") as fp:
        json.dump(dict(api=ORACLE_API_VERSION, h=h, N=N, H=H, gates=out,
                       p1c_bounds=P1C_BOUNDS), fp, indent=1)
    print("saved: p1_calibrate_results.json")
