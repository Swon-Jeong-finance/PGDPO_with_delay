"""P2 scaling calibration: gate metrics across the design-doc sweep grid.

Sweeps (h = 0.02 fixed, T = 1, delta = H h, ring Laplacian, coefficients are
fixed functions of L so mode spectra live in the d-independent range [0,4]):
  dimension sweep  d in {5,20,50,100} at H = 25
  memory sweep     H in {10,25,50}   at d = 20
  extreme corner   (d,H) = (100,50)
  Brownian-rank sensitivity r in {2,4,8} at (20,25); MAIN proposes r = 4 fixed.
Kernel shape held fixed across the memory sweep via rho = 2.5/delta
(design proposal; simulator must match kernel_weights()).

Gates: oracle runtime ~ O(d N H^2); min Lam_i/h > 0 and Pi_i >= 0 uniformly;
memory visibility (kernel-tap action share, A_M drift share); controlled-
diffusion visibility (FOC q-share, zeta share); per-node u_rms and J/d
roughly d-stable (clean scaling message).
"""
import time
import numpy as np
from p2_oracle import P2_API_VERSION, kernel_weights, p2_mode_oracle, \
                      exact_recovery_inputs_p2, mode_rows
assert P2_API_VERSION == "p2-v1-trapezoid-modes"

T, h = 1.0, 0.02
N = round(T/h)

def ring_spec(d, r, rng, variant="A"):
    lam = 2 - 2*np.cos(2*np.pi*np.arange(d)/d)
    fl = lambda a0, a1: a0 + a1*lam
    c0 = [0.15, 0.10, 0.08, 0.12][:r] + [0.1]*max(0, r-4)
    c1 = [0.05, -0.03, 0.02, 0.00][:r] + [0.0]*max(0, r-4)
    m0 = [0.10, 0.00, 0.05, 0.03][:r] + [0.0]*max(0, r-4)
    m1 = [0.00, 0.05, 0.00, 0.02][:r] + [0.0]*max(0, r-4)
    d0 = [0.30, 0.15, 0.20, 0.10][:r] + [0.1]*max(0, r-4)
    d1 = [0.10, 0.00, 0.05, 0.00][:r] + [0.0]*max(0, r-4)
    av = dict(A=(-0.5, -0.3), AM=(0.4, 0.1)) if variant == "A" \
         else dict(A=(-0.5, -0.15), AM=(0.6, 0.15))
    return dict(a=fl(*av["A"]), aM=fl(*av["AM"]), b=fl(1.0, 0.2),
                c=[fl(c0[l], c1[l]) for l in range(r)],
                cM=[fl(m0[l], m1[l]) for l in range(r)],
                dd=[fl(d0[l], d1[l]) for l in range(r)],
                q=fl(1.0, 0.2), r=fl(0.1, 0.02), qT=fl(2.0, 0.0),
                sig=[rng.normal(0, 0.2, d) for l in range(r)], nbm=r)

def mode_hist(rng, d, n1, tt, delta):
    amp = rng.uniform(-1.0, 1.0, d)
    kind = rng.integers(0, 3, d); ph = rng.uniform(0, 2*np.pi, d)
    Z = np.empty((d, n1))
    Z[kind == 0] = amp[kind == 0, None]
    Z[kind == 1] = amp[kind == 1, None]*(1 + tt/delta)[None, :]
    Z[kind == 2] = amp[kind == 2, None]*np.cos(2*np.pi*tt[None, :]/delta + ph[kind == 2, None])
    return Z

def calibrate(d, H, r, rng, variant="A"):
    delta = H*h; n1 = H+1
    rho = 2.5/delta
    w = kernel_weights(rho, delta, h, H)
    spec = ring_spec(d, r, rng, variant)
    t0 = time.perf_counter()
    orc = p2_mode_oracle(spec, w, H, h, N)
    t_orc = time.perf_counter() - t0
    lam_min = min(min(o["Lam"])/h for o in orc)
    Pi_min = min(o["G"][k][0, 0] for o in orc for k in range(N+1))

    # mode-space closed-loop rollout with common Brownians
    Np = 200
    tt = np.linspace(-delta, 0, n1)[::-1]
    rows = np.stack([mode_rows(spec, i, w, h, n1)[0] for i in range(d)])
    crows = np.stack([np.stack(mode_rows(spec, i, w, h, n1)[1]) for i in range(d)])  # (d,r,n1)
    Z = np.stack([mode_hist(rng, d, n1, tt, delta) for _ in range(Np)])              # (Np,d,n1)
    F = np.stack([np.stack(o["F"]) for o in orc])                                    # (d,N,n1)
    f_ = np.stack([np.array(o["f"]) for o in orc])
    cost = np.zeros(Np); u2 = 0.0; memdr = curdr = ctldr = 0.0
    snaps = []
    for k in range(N):
        u = np.einsum('pdn,dn->pd', Z, F[:, k]) + f_[None, :, k]
        cost += 0.5*h*((Z[:, :, 0]**2*spec["q"][None]).sum(1) + (u*u*spec["r"][None]).sum(1))
        u2 += (u*u).mean()
        mem = np.einsum('pdn,n->pd', Z, np.r_[w[0], w[1:]])*spec["aM"][None]
        memdr += np.abs(mem).mean(); curdr += np.abs(spec["a"][None]*Z[:, :, 0]).mean()
        ctldr += np.abs(spec["b"][None]*u).mean()
        if k in (N//5, N//2, 4*N//5): snaps.append((k, Z.copy()))
        dW = rng.normal(0, np.sqrt(h), (Np, r))
        ddm = np.stack(spec["dd"], axis=1)                  # (d, r)
        sgm = np.stack(spec["sig"], axis=1)                 # (d, r)
        load = np.einsum('pdn,drn->pdr', Z, crows) \
               + ddm[None]*u[:, :, None] + sgm[None]
        X1 = Z[:, :, 0] + h*(spec["a"][None]*Z[:, :, 0]
              + spec["aM"][None]*np.einsum('pdn,n->pd', Z, w)
              + spec["b"][None]*u) + np.einsum('pdr,pr->pd', load, dW)
        Z = np.concatenate([X1[:, :, None], Z[:, :, :-1]], axis=2)
    cost += 0.5*(Z[:, :, 0]**2*spec["qT"][None]).sum(1)
    u_rms = np.sqrt(u2/N)          # (u*u).mean() is already per-mode: per-node RMS by Parseval

    # channel shares on snapshot states (subsampled modes/paths)
    sq, sz, mshare = [], [], []
    modes = np.linspace(0, d-1, min(d, 12)).astype(int)
    for (k, Zs) in snaps:
        for p_ in range(0, Np, 50):
            for i in modes:
                t = exact_recovery_inputs_p2(i, k, Zs[p_, i], spec, orc[i], w, H, h)
                bq = abs(spec["b"][i]*t["p_nxt"])
                qq = abs(sum(spec["dd"][l][i]*t["q"][l] for l in range(r)))
                sq.append(qq/(bq + qq + 1e-12))
                zz = abs(sum(spec["dd"][l][i]*t["zeta"][l] for l in range(r)))
                pz = abs(sum(spec["dd"][l][i]*t["Pi"]*t["sig_bar"][l] for l in range(r)))
                sz.append(zz/(abs(spec["b"][i]*t["p_cur"]) + zz + pz + 1e-12))
                Fk = orc[i]["F"][k]
                c0 = abs(Fk[0]*Zs[p_, i, 0]); cp = abs(Fk[1:] @ Zs[p_, i, 1:])
                mshare.append(cp/(c0 + cp + abs(orc[i]["f"][k]) + 1e-12))
    assert np.isfinite(cost).all()
    print(f"d={d:3d} H={H:2d} r={r}: orc {t_orc*1e3:7.1f}ms  minLam/h={lam_min:.3f} "
          f"minPi={Pi_min:.3f}  u_rms/node={u_rms:.3f}  J/d={cost.mean()/d:.3f}  "
          f"memAct={np.mean(mshare):.2f}  memDrift={memdr/(memdr+curdr+ctldr):.2f}  "
          f"FOCq={np.mean(sq):.2%}  zeta={np.mean(sz):.2%}")

print(f"[{P2_API_VERSION}] h={h} N={N}, kernel rho*delta=2.5 (shape-fixed proposal)")
print("--- dimension sweep (H=25, r=4, variant B proposed main) ---")
for d in (5, 20, 50, 100): calibrate(d, 25, 4, np.random.default_rng(5), "B")
print("--- memory sweep (d=20, r=4) ---")
for H in (10, 25, 50): calibrate(20, H, 4, np.random.default_rng(5), "B")
print("--- extreme corner ---")
calibrate(100, 50, 4, np.random.default_rng(5), "B")
print("--- Brownian-rank sensitivity (d=20, H=25) ---")
for r in (2, 4, 8): calibrate(20, 25, r, np.random.default_rng(5), "B")
