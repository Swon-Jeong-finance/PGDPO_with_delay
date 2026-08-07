"""Finite-h alignment pilot on the ADOPTED V3 calibration (exact targets only).

For h in {0.1, 0.0667, 0.05, 0.025, 0.0125, 0.00625} at fixed (T, delta),
stratified oracle-visited states:
  (i)  nRMSE(p_{k+1|k}, p_k)               one-step input alignment
  (ii) RMSE(u_exact_PathA, u*)/mean|u*|    finite-h decoder floor
Unconstrained P1-U proxy; the box-constrained P1-C floor near active-set
switches must be re-measured on the exact tensor-grid DP.
"""
import csv
import numpy as np
from .oracle import ORACLE_API_VERSION, riccati, detached_curvature, exact_recovery_inputs
from .config import load_config
from .dynamics import grid, make_hist, feedback_path
assert ORACLE_API_VERSION == "p1-v3-pcur-pnext"

def run(hs=(0.1, 1/15, 0.05, 0.025, 0.0125, 0.00625), save=True):
    V3 = load_config("main")["params"]
    T, delta = 1.0, 0.2
    b, gu, R = V3["b"], V3["gu"], V3["R"]
    rows = []
    print(f"{'h':>8} {'H':>4} {'nRMSE(p_nxt,p_cur)':>20} {'PathA floor RMSE':>18} {'relative':>10}")
    for h in hs:
        rng = np.random.default_rng(7)
        N, H, tt = grid(T, delta, h); n = H+1
        xref = 0.5*np.sin(2*np.pi*np.arange(N)*h/T); xtar = 0.3
        orc = riccati(V3, H, h, N, xref, xtar)
        G = detached_curvature(V3, H, h, N)
        strata = [(0, 1), (max(H-1, 0), H+1), (N//2-1, N//2+1),
                  (max(N-H-1, 0), N-H+1), (N-1, N-1)]
        gaps = []
        for lo, hi in strata:
            for _ in range(8):
                z0 = make_hist(rng, n, tt, delta)
                k = int(rng.integers(lo, hi+1))
                z = feedback_path(orc, V3, H, h, z0, 0, k, rng)
                t = exact_recovery_inputs(k, z, V3, H, h, orc, G, xref)
                u_pcur = -(b*t["p_cur"] + gu*t["zeta"] + gu*t["Pi"]*t["sigma_bar"])/(R + gu*gu*t["Pi"])
                gaps.append((t["p_nxt"] - t["p_cur"], t["p_cur"], u_pcur - t["u"], t["u"]))
        a = np.array(gaps)
        nr_p = np.sqrt(np.mean(a[:,0]**2))/np.sqrt(np.mean(a[:,1]**2))
        rm_u = np.sqrt(np.mean(a[:,2]**2)); mu = np.mean(np.abs(a[:,3]))
        print(f"{h:8.4f} {H:4d} {nr_p:20.4%} {rm_u:18.3e} {rm_u/mu:10.3%}")
        rows.append(dict(h=h, H=H, nrmse_p=nr_p, floor_rmse=rm_u, floor_rel=rm_u/mu))
    if save:
        with open("p1_h_refine_v3.csv", "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        print("saved: p1_h_refine_v3.csv")
    return rows

if __name__ == "__main__":
    run()
    