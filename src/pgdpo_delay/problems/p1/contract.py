"""P1-3: estimator contract check (Theorem 5.1 / A.8 empirical contract).

Panel A  (manuscript input contract, Stage-II Path A):
    p^_k        current detached gradient (lam0)      vs  p_cur
    q^_k        raw covariance AND nested OLS         vs  q
    Pi^_k       branch Hessian mean                   vs  Pi
    zeta^_k     nested residual regression            vs  zeta
Panel B  (exact same-grid Euler-FOC diagnostic; NOT a Stage-II input):
    p^_{k+1|k} = lam1.mean()  vs  p_nxt;  estimated FOC residual;
    finite-h alignment of the p_cur-decoder action.
Panel C  (estimator-design ablation, matched q-estimator continuation
         count; full-pipeline simulator/BPTT cost is reported separately):
    q_raw (single-level covariance)  vs  q_ols_antithetic (nested CRN OLS);
    zeta_direct (q_raw - Pi^ sigma_ref)  vs  zeta_nested (residual regression).
    With kappa_Z = 0 the samplewise identity  q_anc == q_ols  holds exactly
    (anchor cancels); gains over q_raw are nested-antithetic-CRN variance
    reduction, NOT evidence that the anchor itself is required.

Budgets: nested uses Mout outer ANTITHETIC PAIRS x Min inner continuations,
i.e. 2*Mout*Min signed inner continuations total; equal-budget raw uses
M = 2*Mout*Min. SNR = |zeta^|/SE is an auxiliary Monte-Carlo resolution
diagnostic only (OLS SE excludes anchor uncertainty and CRN structure).
Prefix budget sweep is a paired fixed-bank sanity, not a rate estimate.
Runs on the ADOPTED P1-U calibration: V3 parameters, h = 0.0125 (H = 16).
"""
import json
import numpy as np
from .oracle import (ORACLE_API_VERSION, build_dense, riccati,
                     detached_curvature, exact_recovery_inputs)
from .config import load_config
from .dynamics import make_hist, feedback_path

assert ORACLE_API_VERSION == "p1-v3-pcur-pnext", ORACLE_API_VERSION

def branch_stats(params, H, h, N, orc, xref, xtar, k, z, M, rng):
    """M Gaussian continuations from (k, z), frozen feedback values (detached).
    Returns lam0 (current gradient, tangent at k), lam1 (tangent at k+1),
    dW_k, and the pathwise detached Hessian PiT (tangent at k)."""
    A, B, C, D, Sg = build_dense(params, H, h)
    Q, QT = params["Q"], params["QT"]
    n = H + 1
    Z = np.tile(z, (M, 1))
    lam0 = np.zeros(M); lam1 = np.zeros(M); PiT = np.zeros(M)
    Vk = np.zeros((M, n)); Vk[:, 0] = 1.0
    V1 = np.zeros((M, n))
    dW0 = None
    for j in range(k, N):
        u = Z @ orc["F"][j] + orc["f"][j]
        dW = rng.normal(0, np.sqrt(h), M)
        lam0 += h*Q*(Z[:, 0] - xref[j])*Vk[:, 0]
        if j > k:
            lam1 += h*Q*(Z[:, 0] - xref[j])*V1[:, 0]
        PiT += h*Q*Vk[:, 0]**2
        cvK = Vk @ C[0]; cv1 = V1 @ C[0]
        Vk = Vk @ A.T; Vk[:, 0] += dW*cvK
        V1 = V1 @ A.T; V1[:, 0] += dW*cv1
        if j == k:
            dW0 = dW.copy()
            V1[:, 0] = 1.0                       # injection at k+1
        Z = Z @ A.T + np.outer(u, B) + (Z @ C.T + np.outer(u, D) + Sg)*dW[:, None]
    lam0 += QT*(Z[:, 0] - xtar)*Vk[:, 0]
    lam1 += QT*(Z[:, 0] - xtar)*V1[:, 0]
    PiT += QT*Vk[:, 0]**2
    return lam0, lam1, dW0, PiT

def zeta_nested(params, H, h, N, orc, xref, xtar, k, z, Pi_hat, Mout, Min, rng):
    """Anchored nested antithetic CRN regression, kappa_Z = 0. Returns
    (zeta_hat, q_ols_antithetic, ols_se, sigma_star). Samplewise identity:
    zeta_hat + Pi_hat*sigma_star == q_ols_antithetic exactly."""
    A, B, C, D, Sg = build_dense(params, H, h)
    Q, QT = params["Q"], params["QT"]
    n = H + 1
    u0 = orc["F"][k] @ z + orc["f"][k]
    m = A @ z + B*u0
    sig_vec = C @ z + D*u0 + Sg; sig_star = sig_vec[0]
    d = rng.normal(0, np.sqrt(h), Mout)
    Zp = m[None, :] + d[:, None]*sig_vec[None, :]
    Zm = m[None, :] - d[:, None]*sig_vec[None, :]
    Z = np.concatenate([np.repeat(Zp, Min, 0), np.repeat(Zm, Min, 0)], 0)
    Mtot = Z.shape[0]
    V = np.zeros((Mtot, n)); V[:, 0] = 1.0
    lam = np.zeros(Mtot)
    inner = rng.normal(0, np.sqrt(h), (Mout*Min, N - (k+1)))
    noise = np.concatenate([inner, inner], 0)     # CRN across +/- legs
    for jj, j in enumerate(range(k+1, N)):
        lam += h*Q*(Z[:, 0] - xref[j])*V[:, 0]
        dW = noise[:, jj]
        cv = V @ C[0]
        V = V @ A.T; V[:, 0] += dW*cv
        u = Z @ orc["F"][j] + orc["f"][j]
        Z = Z @ A.T + np.outer(u, B) + (Z @ C.T + np.outer(u, D) + Sg)*dW[:, None]
    lam += QT*(Z[:, 0] - xtar)*V[:, 0]
    lam = lam.reshape(2, Mout, Min).mean(axis=2)
    y = 0.5*(lam[0] - lam[1])
    q_ols = (d @ y)/(d @ d)
    r = y - d*(Pi_hat*sig_star)
    zeta = (d @ r)/(d @ d)
    se = np.sqrt(((r - zeta*d)**2).sum()/((Mout - 1)*(d @ d)))
    return zeta, q_ols, se, sig_star

# ---------------------------------------------------------------- contract check
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    cfg = load_config("main")               # adopted P1-U calibration (YAML)
    params = cfg["params"]
    T, delta, h = cfg["T"], cfg["delta"], cfg["h"]
    N, H, tt = cfg["N"], cfg["H"], cfg["tt"]; n = H+1
    xref, xtar = cfg["xref"], cfg["xtar"]
    b, gu, R = params["b"], params["gu"], params["R"]
    orc = riccati(params, H, h, N, xref, xtar)
    G = detached_curvature(params, H, h, N)
    assert min(orc["Lam"]) > 0 and min(G[k][0, 0] for k in range(N+1)) >= -1e-12

    # stratified diagnostic states: early / first re-entry / mid / cutoff / terminal
    strata = [(0, 1), (max(H-1, 0), H+1), (N//2-1, N//2+1),
              (max(N-H-1, 0), N-H+1), (N-1, N-1)]
    states = []
    for lo, hi in strata:
        for _ in range(8):
            z0 = make_hist(rng, n, tt, delta)
            k = int(rng.integers(lo, hi+1))                        # inclusive
            states.append((k, feedback_path(orc, params, H, h, z0, 0, k, rng)))

    Mout, Min = 512, 8
    M_eq = 2*Mout*Min          # equal-budget raw:   8192 continuations
    assert Mout > 1
    rows = []
    sweep = {m: [] for m in (512, 2048, 8192)}
    for (k, z) in states:
        t = exact_recovery_inputs(k, z, params, H, h, orc, G, xref)
        lam0, lam1, dW0, PiT = branch_stats(params, H, h, N, orc, xref, xtar,
                                            k, z, M_eq, rng)
        p_hat, p_nxt_hat, Pi_hat = lam0.mean(), lam1.mean(), PiT.mean()
        q_raw = (lam1*dW0).mean()/h
        ze_nst, q_ols, ze_se, sig_star = zeta_nested(params, H, h, N, orc, xref,
                                                     xtar, k, z, Pi_hat, Mout, Min, rng)
        q_anc = ze_nst + Pi_hat*sig_star
        assert abs(q_anc - q_ols) < 1e-12       # samplewise identity (kappa_Z=0)
        ze_dir = q_raw - Pi_hat*sig_star
        u_rec = -(b*p_nxt_hat + gu*ze_nst + gu*Pi_hat*t["sigma_bar"])/(R + gu*gu*Pi_hat)
        foc_est = R*t["u"] + b*p_nxt_hat + gu*q_ols
        u_pcur = -(b*t["p_cur"] + gu*t["zeta"] + gu*t["Pi"]*t["sigma_bar"])/(R + gu*gu*t["Pi"])
        u_phat = -(b*p_hat + gu*ze_nst + gu*Pi_hat*t["sigma_bar"])/(R + gu*gu*Pi_hat)
        rows.append(dict(k=k, p_hat=p_hat, p_cur=t["p_cur"], p_nxt_hat=p_nxt_hat,
                         p_nxt=t["p_nxt"], q_raw=q_raw, q_ols=q_ols, q=t["q"],
                         Pi_hat=Pi_hat, Pi=t["Pi"], ze_nst=ze_nst, ze_dir=ze_dir,
                         zeta=t["zeta"], snr=abs(ze_nst)/ze_se, u_rec=u_rec,
                         u=t["u"], foc=foc_est, u_pcur=u_pcur, u_phat=u_phat))
        for m in sweep:
            sweep[m].append((lam0[:m].mean() - t["p_cur"], PiT[:m].mean() - t["Pi"]))
    g = lambda key: np.array([r[key] for r in rows])
    nr = lambda e, t_: np.sqrt(np.mean((g(e)-g(t_))**2))/np.sqrt(np.mean(g(t_)**2))

    print(f"[{ORACLE_API_VERSION}] Ns={len(rows)} stratified states, "
          f"nested = {Mout} pairs x {Min} inner x 2 legs = {M_eq} continuations, "
          f"raw M = {M_eq} (matched q-estimator continuation count), kappa_Z = 0")
    print("--- Panel A: manuscript input contract (Stage-II Path A) ---")
    print(f"nRMSE  p^_k (lam0)        = {nr('p_hat','p_cur'):.4%}")
    print(f"nRMSE  q^raw              = {nr('q_raw','q'):.4%}   (single-level covariance)")
    print(f"nRMSE  q^ols_antithetic   = {nr('q_ols','q'):.4%}   (== q^anc samplewise)")
    print(f"nRMSE  Pi^                = {nr('Pi_hat','Pi'):.4%}")
    print(f"nRMSE  zeta^_nested       = {nr('ze_nst','zeta'):.4%}   mean|zeta| = {np.mean(np.abs(g('zeta'))):.4f}")
    snr = g('snr'); fin = snr[np.isfinite(snr) & (snr < 1e6)]
    ndeg = len(snr) - len(fin)
    print(f"auxiliary MC-resolution SNR (not a t-stat): median = {np.median(fin):.1f} "
          f"over {len(fin)} states ({ndeg} terminal-boundary states exact, SE=0, excluded)")
    print("--- Panel B: exact same-grid Euler-FOC diagnostic ---")
    print(f"nRMSE  p^_(k+1|k) (lam1)  = {nr('p_nxt_hat','p_nxt'):.4%}")
    print(f"finite-h alignment nRMSE(p_nxt, p_cur)      = {nr('p_nxt','p_cur'):.4%}")
    print(f"estimated FOC residual RMS                  = {np.sqrt(np.mean(g('foc')**2)):.3e}")
    print(f"recovered action RMSE (p_nxt decoder)       = {np.sqrt(np.mean((g('u_rec')-g('u'))**2)):.3e}")
    rm = lambda a, b_: np.sqrt(np.mean((g(a)-g(b_))**2))
    print("--- Path-A action decomposition (Theorem 5.3 layer split) ---")
    print(f"estimator layer   RMSE(u_hat_PathA, u_exact_PathA) = {rm('u_phat','u_pcur'):.3e}")
    print(f"finite-h layer    RMSE(u_exact_PathA, u*)          = {rm('u_pcur','u'):.3e}")
    print(f"combined pilot    RMSE(u_hat_PathA, u*)            = {rm('u_phat','u'):.3e}"
          f"   (mean|u*| = {np.mean(np.abs(g('u'))):.3f})")
    print("--- Panel C: estimator ablation (matched q-estimator continuation count) ---")
    print(f"nRMSE  zeta^_direct       = {nr('ze_dir','zeta'):.4%}   (q_raw - Pi^ sigma_ref)")
    print(f"nRMSE  zeta^_nested       = {nr('ze_nst','zeta'):.4%}   (residual regression)")
    print("--- fixed-bank prefix budget sanity (paired; not a rate estimate) ---")
    for m in sorted(sweep):
        e = np.array(sweep[m])
        print(f"  M={m:5d}: p {np.sqrt((e[:,0]**2).mean()):.3e}   Pi {np.sqrt((e[:,1]**2).mean()):.3e}")

    with open("p1_contract_config.json", "w") as fp:
        json.dump(dict(api=ORACLE_API_VERSION, params=params, T=T, delta=delta,
                       h=h, N=N, H=H, Mout=Mout, Min=Min, M_equal_budget=M_eq,
                       seed=7, Ns=len(rows)), fp, indent=1)
    np.savez("p1_contract_states.npz",
             k=np.array([r["k"] for r in rows]),
             Z=np.stack([z for _, z in states]),
             **{key: g(key) for key in rows[0] if key != "k"})
    print("saved: p1_contract_config.json, p1_contract_states.npz")
