"""P2 scaling calibration (post-review): gate metrics across the sweep grid.

Review patches applied:
  H1 diagonalization guard lives in p2_oracle.mode_spec (ring spectra here are
     exact by construction); kernel guards in kernel_weights.
  H2 production oracle preallocates (no insert(0)); Lam finiteness checked.
  H3 J/d is the EXACT oracle value 0.5 z'P0 z + s0'z + c0 averaged over the
     initial bank (rollout J_mc reported separately as simulator sanity).
  H4 the r-sweep is reported as a diffusion-channel-count/INTENSITY
     sensitivity; r = 4 is the main design convention, not a sweep outcome.
Recommended: separate RNG streams (model/history/noise); runtime = median of
repeats after warm-up; full-vector orthonormal-invariant channel shares;
corner tail audit; main() guard; CSV/JSON artifacts.
Kernel: rho*delta = 2.5 shape-fixed across the memory sweep (appendix
robustness: one fixed-rho line). Ring case is network-coupled (sparse
circulant) in observed coordinates, not dense coupling.
"""
import csv, json, time
import numpy as np
from .oracle import P2_API_VERSION, kernel_weights, p2_mode_oracle, \
                      exact_recovery_inputs_p2, mode_rows
from .config import load_p2_config
assert P2_API_VERSION == "p2-v2-guards"

CFG = load_p2_config("scaling")
T, h = CFG["grid"]["T"], CFG["grid"]["dt"]
N = round(T/h)
RHO_DELTA = CFG["kernel"]["rho_delta"]
R_MAIN = int(CFG["r_main"])
COEF = CFG["coefficients"]

def _pad(base, r, fill):
    return list(base)[:r] + [fill]*max(0, r - len(base))

def ring_spec(d, r, rng_model, variant=None):
    """Coefficient spectra from configs/p2/scaling.yaml ONLY (single source)."""
    variant = variant or CFG["variant"]
    lam = 2 - 2*np.cos(2*np.pi*np.arange(d)/d)
    fl = lambda a0, a1: a0 + a1*lam
    c0 = _pad(COEF["C0"], r, 0.1);  c1 = _pad(COEF["C1"], r, 0.0)
    m0 = _pad(COEF["CM0"], r, 0.0); m1 = _pad(COEF["CM1"], r, 0.0)
    d0 = _pad(COEF["D0"], r, 0.1);  d1 = _pad(COEF["D1"], r, 0.0)
    return dict(a=fl(*COEF["A"][variant]), aM=fl(*COEF["AM"][variant]),
                b=fl(*COEF["Bc"]),
                c=[fl(c0[l], c1[l]) for l in range(r)],
                cM=[fl(m0[l], m1[l]) for l in range(r)],
                dd=[fl(d0[l], d1[l]) for l in range(r)],
                q=fl(*COEF["Q"]), r=fl(*COEF["R"]), qT=fl(*COEF["QT"]),
                sig=[rng_model.normal(0, COEF["sigma0_scale"], d) for l in range(r)],
                nbm=r)

def mode_hist(rng, d, n1, tt, delta):
    amp = rng.uniform(-1.0, 1.0, d)
    kind = rng.integers(0, 3, d); ph = rng.uniform(0, 2*np.pi, d)
    Z = np.empty((d, n1))
    Z[kind == 0] = amp[kind == 0, None]
    Z[kind == 1] = amp[kind == 1, None]*(1 + tt/delta)[None, :]
    Z[kind == 2] = amp[kind == 2, None]*np.cos(2*np.pi*tt[None, :]/delta + ph[kind == 2, None])
    return Z

def calibrate(d, H, r, seeds=None, variant=None, tail=False, reps=None, outdir="."):
    seeds = seeds or (CFG["seeds"]["model"], CFG["seeds"]["hist"], CFG["seeds"]["noise"])
    reps = reps or int(CFG["budgets"]["reps"])
    delta = H*h; n1 = H+1
    w = kernel_weights(RHO_DELTA/delta, delta, h, H)
    rng_model = np.random.default_rng(seeds[0])
    rng_hist  = np.random.default_rng(seeds[1])
    rng_noise = np.random.default_rng(seeds[2])
    spec = ring_spec(d, r, rng_model, variant)
    ts = []
    for rep in range(reps + 1):                     # first call = warm-up
        t0 = time.perf_counter()
        orc = p2_mode_oracle(spec, w, H, h, N)
        if rep > 0: ts.append(time.perf_counter() - t0)
    t_med = float(np.median(ts))
    lam_min = min(min(o["Lam"])/h for o in orc)
    Pi_min = min(o["G"][k][0, 0] for o in orc for k in range(N+1))

    Np = int(CFG["budgets"]["Np"])
    tt = np.linspace(-delta, 0, n1)[::-1]
    Z0 = np.stack([mode_hist(rng_hist, d, n1, tt, delta) for _ in range(Np)])
    # H3: exact per-node objective from the value oracle at k = 0
    Jx = np.zeros(Np)
    for i, oi in enumerate(orc):
        zi = Z0[:, i, :]
        Jx += 0.5*np.einsum("pn,nm,pm->p", zi, oi["Pval"][0], zi) + zi @ oi["s"][0] + oi["c"][0]
    J_exact = Jx.mean()/d; J_se = Jx.std(ddof=1)/np.sqrt(Np)/d

    rows_ = np.stack([mode_rows(spec, i, w, h, n1)[0] for i in range(d)])
    crows = np.stack([np.stack(mode_rows(spec, i, w, h, n1)[1]) for i in range(d)])
    F = np.stack([np.stack(o["F"]) for o in orc]); f_ = np.stack([np.array(o["f"]) for o in orc])
    ddm = np.stack(spec["dd"], axis=1); sgm = np.stack(spec["sig"], axis=1)
    Z = Z0.copy(); cost = np.zeros(Np); u2 = 0.0
    memdr = curdr = ctldr = 0.0
    mx = np.abs(np.linalg.norm(Z[:, :, 0], axis=1))/np.sqrt(d); mu_ = np.zeros(Np)
    snaps = []
    for k in range(N):
        u = np.einsum('pdn,dn->pd', Z, F[:, k]) + f_[None, :, k]
        cost += 0.5*h*((Z[:, :, 0]**2*spec["q"][None]).sum(1) + (u*u*spec["r"][None]).sum(1))
        u2 += (u*u).mean()
        mem = np.einsum('pdn,n->pd', Z, w)*spec["aM"][None]
        memdr += np.abs(mem).mean(); curdr += np.abs(spec["a"][None]*Z[:, :, 0]).mean()
        ctldr += np.abs(spec["b"][None]*u).mean()
        mu_ = np.maximum(mu_, np.linalg.norm(u, axis=1)/np.sqrt(d))
        if k in (N//5, N//2, 4*N//5): snaps.append((k, Z.copy()))
        dW = rng_noise.normal(0, np.sqrt(h), (Np, r))
        load = np.einsum('pdn,drn->pdr', Z, crows) + ddm[None]*u[:, :, None] + sgm[None]
        X1 = Z[:, :, 0] + h*(spec["a"][None]*Z[:, :, 0]
              + spec["aM"][None]*np.einsum('pdn,n->pd', Z, w) + spec["b"][None]*u) \
              + np.einsum('pdr,pr->pd', load, dW)
        Z = np.concatenate([X1[:, :, None], Z[:, :, :-1]], axis=2)
        mx = np.maximum(mx, np.linalg.norm(Z[:, :, 0], axis=1)/np.sqrt(d))
    cost += 0.5*(Z[:, :, 0]**2*spec["qT"][None]).sum(1)
    assert np.isfinite(cost).all() and np.isfinite(mx).all()
    u_rms = np.sqrt(u2/N)
    J_mc = cost.mean()/d; J_mc_se = cost.std(ddof=1)/np.sqrt(Np)/d

    # full-vector orthonormal-invariant channel shares (rec. 6.2)
    sq, sz, ms = [], [], []
    for (k, Zs) in snaps:
        for p_ in range(0, Np, 66):
            tms = [exact_recovery_inputs_p2(i, k, Zs[p_, i], spec, orc[i], w, H, h)
                   for i in range(d)]
            vp = np.array([spec["b"][i]*tms[i]["p_nxt"] for i in range(d)])
            vq = np.array([sum(spec["dd"][l][i]*tms[i]["q"][l] for l in range(r)) for i in range(d)])
            sq.append(np.linalg.norm(vq)/(np.linalg.norm(vp) + np.linalg.norm(vq) + 1e-12))
            vpc = np.array([spec["b"][i]*tms[i]["p_cur"] for i in range(d)])
            vz = np.array([sum(spec["dd"][l][i]*tms[i]["zeta"][l] for l in range(r)) for i in range(d)])
            vps = np.array([sum(spec["dd"][l][i]*tms[i]["Pi"]*tms[i]["sig_bar"][l]
                                for l in range(r)) for i in range(d)])
            sz.append(np.linalg.norm(vz)/(np.linalg.norm(vpc) + np.linalg.norm(vz)
                                          + np.linalg.norm(vps) + 1e-12))
            F0 = np.array([orc[i]["F"][k][0]*Zs[p_, i, 0] for i in range(d)])
            Fp = np.array([orc[i]["F"][k][1:] @ Zs[p_, i, 1:] for i in range(d)])
            fo = np.array([orc[i]["f"][k] for i in range(d)])
            ms.append(np.linalg.norm(Fp)/(np.linalg.norm(F0) + np.linalg.norm(Fp)
                                          + np.linalg.norm(fo) + 1e-12))
    row = dict(d=d, H=H, r=r, orc_ms=t_med*1e3, lam_min=lam_min, Pi_min=Pi_min,
               u_rms=u_rms, J_exact=J_exact, J_exact_se=J_se, J_mc=J_mc, J_mc_se=J_mc_se,
               memAct=float(np.mean(ms)), memDrift=memdr/(memdr+curdr+ctldr),
               FOCq=float(np.mean(sq)), zeta=float(np.mean(sz)))
    print(f"d={d:3d} H={H:2d} r={r}: orc {t_med*1e3:7.1f}ms(med/{len(ts)})  "
          f"minLam/h={lam_min:.3f} minPi={Pi_min:.3f}  u_rms/node={u_rms:.3f}  "
          f"J/d(exact)={J_exact:.4f}±{J_se:.4f}  [J/d MC={J_mc:.3f}±{J_mc_se:.3f}]  "
          f"memAct={row['memAct']:.2f}  memDrift={row['memDrift']:.2f}  "
          f"FOCq={row['FOCq']:.2%}  zeta={row['zeta']:.2%}")
    if tail:
        q99 = lambda v: float(np.quantile(v, 0.99))
        print(f"   corner tail: 99%q max||X||/√d={q99(mx):.2f}  99%q max||u||/√d={q99(mu_):.2f}  all finite")
        row.update(tail_X99=q99(mx), tail_u99=q99(mu_))
    return row

def main(outdir="."):
    from pathlib import Path
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    sw = CFG["sweeps"]
    print(f"[{P2_API_VERSION}] h={h} N={N}, kernel rho*delta={RHO_DELTA} shape-fixed "
          f"(appendix: one fixed-rho robustness line); ring = network-coupled")
    out = []
    print(f"--- dimension sweep (H={sw['dimension']['H']}, r={R_MAIN} main convention) ---")
    for d in sw["dimension"]["d"]: out.append(calibrate(d, sw["dimension"]["H"], R_MAIN))
    print(f"--- memory sweep (d={sw['memory']['d']}, r={R_MAIN}) ---")
    for H in sw["memory"]["H"]: out.append(calibrate(sw["memory"]["d"], H, R_MAIN))
    print("--- extreme corner ---")
    out.append(calibrate(sw["corner"]["d"], sw["corner"]["H"], R_MAIN, tail=True))
    print(f"--- diffusion-channel-count/intensity sensitivity (NOT rank-only) ---")
    for r in sw["channels"]["r"]: out.append(calibrate(sw["channels"]["d"], sw["channels"]["H"], r))
    with open(outdir/"p2_calibration_results.csv", "w", newline="") as fp:
        wcsv = csv.DictWriter(fp, fieldnames=sorted({k for rw in out for k in rw}))
        wcsv.writeheader(); wcsv.writerows(out)
    with open(outdir/"p2_calibration_config.json", "w") as fp:
        json.dump(dict(api=P2_API_VERSION, h=h, T=T, N=N, variant=CFG["variant"],
                       r_main=R_MAIN, kernel=f"trapezoidal rho*delta={RHO_DELTA}",
                       Np=int(CFG["budgets"]["Np"]), seeds=CFG["seeds"]), fp, indent=1)
    print("saved: p2_calibration_results.csv, p2_calibration_config.json")

if __name__ == "__main__":
    main()
