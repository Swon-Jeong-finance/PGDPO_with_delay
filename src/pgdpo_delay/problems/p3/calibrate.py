"""P3-R calibration gates (design-handoff sec.11): measured under the shared
simulator with the HJB reference policy. Gate roster:
  G1 pointwise exact-min vs brute force        (machine; oracle self-check)
  G2 MC objective vs V(0, I0, M0)              (|dJ| <= 3 SE + disc tol)
  G3 control saturation fraction in [0.2, 0.7] (either bound counts)
  G4 renewal-memory visibility                 (same I, different M -> du)
  G5 curvature positivity A2 > 0 on the visited region
  G6 two-level grid consistency                (coarse vs fine, report)
Active-set labels use the dtype-aware tolerance (evaluate contract)."""
import numpy as np
from . import dynamics, oracle
from ..p1.evaluate import active_tol


def rollout_diagnostics(cfg, hjb, Np=4000, seed=7):
    pol = oracle.hjb_policy(hjb)
    p, dt, N = cfg["params"], cfg["dt"], cfg["N"]
    lo, hi = cfg["bounds"]; atol = active_tol(lo, hi)
    rng = np.random.default_rng(seed)
    I = rng.uniform(*cfg["init"]["I0"], Np); M = rng.uniform(*cfg["init"]["M0"], Np)
    dW = rng.normal(0.0, np.sqrt(dt), (Np, N))
    I0, M0 = I.copy(), M.copy()
    cost = np.zeros(Np); occ = np.zeros(3)
    Imax_seen = 0.0; neg_frac = []
    for k in range(N):
        u = np.clip(pol(k, I, M), lo, hi)
        occ += [(u <= lo+atol).mean(), ((u > lo+atol) & (u < hi-atol)).mean(),
                (u >= hi-atol).mean()]
        cost += dt*(0.5*p["c_I"]*np.maximum(I, 0.0)**2 + 0.5*p["R"]*u**2)
        I, M = dynamics.step(p, dt, I, M, u, dW[:, k])
        Imax_seen = max(Imax_seen, float(I.max()))
        neg_frac.append(float((I < 0).mean()))
    cost += 0.5*p["c_T"]*np.maximum(I, 0.0)**2
    J_mc, se = float(cost.mean()), float(cost.std(ddof=1)/np.sqrt(Np))
    V0 = oracle.value_at(hjb, I0, M0)
    D = cost - V0                       # paired residual (review sec.3.6):
    # the test object is J_i - V0(I0_i, M0_i), so the gate SE is the paired
    # residual SE (much tighter than the unpaired cost SE).
    return dict(J_mc=J_mc, se=se, V0_mean=float(V0.mean()),
                paired_mean=float(D.mean()),
                paired_se=float(D.std(ddof=1)/np.sqrt(Np)),
                occ=occ/N, sat=float(occ[0]/N + occ[2]/N),
                Imax_seen=Imax_seen, neg_frac_mean=float(np.mean(neg_frac)))


def memory_visibility(cfg, hjb, n=64, seed=11):
    """Same current I, different memory M: RMS policy gap at t=0 relative to
    rollout u_rms. This is THE renewal-delay channel the paper claims."""
    rng = np.random.default_rng(seed)
    I = rng.uniform(*cfg["init"]["I0"], n)
    Ma = rng.uniform(*cfg["init"]["M0"], n)
    Mb = rng.uniform(*cfg["init"]["M0"], n)
    pol = oracle.hjb_policy(hjb)
    ua, ub = pol(0, I, Ma), pol(0, I, Mb)
    du = np.sqrt(np.mean((ua - ub)**2))
    dm = np.sqrt(np.mean((Ma - Mb)**2))
    return dict(du_rms=float(du), dM_rms=float(dm),
                gain=float(du/max(dm, 1e-12)))


def curvature_positivity(cfg, hjb):
    """A2 = R + eta^2 s0^2 I^2 V_II on the solve grid at t=0: min and the
    fraction of nodes with A2 <= 0 (endpoint branch usage)."""
    p = cfg["params"]; Ig = hjb["Ig"]; dI = Ig[1]-Ig[0]
    V_I, V_II = oracle._derivatives(hjb["V0"], dI)
    A2 = p["R"] + (p["eta_sigma"]*p["sigma0"]*Ig[:, None])**2*V_II
    return dict(A2_min=float(A2.min()), frac_nonpos=float((A2 <= 0).mean()))


def brute_force_check(cfg, n=2000, seed=3):
    """G1: exact quadratic minimiser vs dense grid + endpoints, random tuples
    including negative-curvature draws."""
    p = cfg["params"]; rng = np.random.default_rng(seed)
    I = rng.uniform(0.0, cfg["hjb"]["I_max"], n)
    V_I = rng.normal(0.0, 3.0, n); V_II = rng.normal(0.0, 30.0, n)
    u_ex = oracle.pointwise_min_u(p, I, V_I, V_II)
    ug = np.linspace(0.0, 1.0, 20001)
    s2I2 = (p["sigma0"]*I)**2
    A2 = p["R"] + (p["eta_sigma"]**2)*s2I2*V_II
    num = p["b"]*I*V_I + p["eta_sigma"]*s2I2*V_II
    f = 0.5*A2[:, None]*ug[None, :]**2 - num[:, None]*ug[None, :]
    u_bf = ug[f.argmin(axis=1)]
    fx = lambda u: 0.5*A2*u*u - num*u
    return float(np.max(np.abs(fx(u_ex) - fx(u_bf))))


def diffusion_channel_share(cfg, hjb, I_lo=0.02, I_hi=1.0):
    """Canonical controlled-diffusion channel share (review sec.4.2): on the
    t = 0 grid restricted to the visited band I in [I_lo, I_hi],
        share = |eta s0^2 I^2 V_II| / (|b I V_I| + |eta s0^2 I^2 V_II| + eps).
    Reported as mean/median/quantiles so the manuscript number is a stored
    artifact, not an ad hoc pilot printout."""
    p = cfg["params"]; Ig = hjb["Ig"]; dI = Ig[1]-Ig[0]
    V_I, V_II = oracle._derivatives(hjb["V0"], dI)
    II = Ig[:, None]
    num_dr = np.abs(p["b"]*II*V_I)
    num_df = np.abs(p["eta_sigma"]*(p["sigma0"]*II)**2*V_II)
    share = num_df/(num_dr + num_df + 1e-15)
    band = (Ig >= I_lo) & (Ig <= I_hi)
    sb = share[band]
    return dict(mean=float(sb.mean()), median=float(np.median(sb)),
                q10=float(np.quantile(sb, 0.1)),
                q90=float(np.quantile(sb, 0.9)))


def _readout_k(hjb, k, I, M):
    from scipy.ndimage import map_coordinates
    Ig, Mg = hjb["Ig"], hjb["Mg"]
    ci = np.clip((np.atleast_1d(I) - Ig[0])/(Ig[1]-Ig[0]), 0, len(Ig)-1)
    cm = np.clip((np.atleast_1d(M) - Mg[0])/(Mg[1]-Mg[0]), 0, len(Mg)-1)
    return map_coordinates(hjb["Vs"][k], np.stack([ci, cm]),
                           order=1, mode="nearest")


def bellman_residual(cfg, hjb, Np=4000, seed=7):
    """Per-time-step MC Bellman residual along closed-loop rollouts under
    the stored HJB policy (re-review 2026-08-07 sec.5): with the
    time-indexed value snapshots V_k,
        res_k = E[ V_k(X_k) - h ell_k - V_{k+1}(X_{k+1}) ],
    estimated on the evaluation initial law. The per-k profile localises
    the time-discretisation error in (simulation) time; the sum telescopes
    pathwise to V_0 minus realized cost (the negative of the cost-minus-value
    paired residual reported by ``rollout_diagnostics``).  This provides an
    independent cross-check through the forward simulator and bilinear
    readouts only, never the backward recursion code path."""
    if hjb.get("Vs") is None:
        raise ValueError("bellman_residual needs solve_hjb(store_value=True)")
    p, dt, N = cfg["params"], cfg["dt"], cfg["N"]
    rng = np.random.default_rng(seed)
    I = rng.uniform(*cfg["init"]["I0"], Np)
    M = rng.uniform(*cfg["init"]["M0"], Np)
    pol = oracle.hjb_policy(hjb)
    res_mean = np.zeros(N); res_se = np.zeros(N)
    cum = np.zeros(Np)
    for k in range(N):
        u = pol(k, I, M)
        Ib = np.maximum(I, 0.0)
        run = 0.5*p["c_I"]*Ib**2 + 0.5*p["R"]*u**2
        Vk = _readout_k(hjb, k, I, M)
        In, Mn = dynamics.step(p, dt, I, M, u, rng.normal(0, np.sqrt(dt), Np))
        if k == N - 1:
            # Match the realized simulator cost exactly.  Interpolating the
            # terminal table here creates a spurious endpoint mismatch and
            # prevents the pathwise residual from telescoping.
            Vn = 0.5*p["c_T"]*np.maximum(In, 0.0)**2
        else:
            Vn = _readout_k(hjb, k+1, In, Mn)
        D = Vk - dt*run - Vn
        res_mean[k] = D.mean(); res_se[k] = D.std(ddof=1)/np.sqrt(Np)
        cum += D
        I, M = In, Mn
    return dict(mean=res_mean, se=res_se,
                max_abs_mean=float(np.max(np.abs(res_mean))),
                total=float(cum.mean()),
                total_se=float(cum.std(ddof=1)/np.sqrt(Np)))


def policy_optimality_residual(cfg, hjb, ks):
    """Independent recertification of the STORED artifact (re-review
    sec.5): at each selected time k, recompute the branch-aware discrete
    minimisation from the stored float32 value snapshot and compare the
    stored policy table's upwind objective against the recomputed minimum
    and a dense feasible action grid. Certifies that the frozen tables are
    self-consistent without re-running the backward recursion."""
    p = cfg["params"]
    Ig = hjb["Ig"]; dI = Ig[1]-Ig[0]
    II = Ig[:, None]; MM = hjb["Mg"][None, :]
    base = p["beta"]*(1.0 - II/p["Npop"])*MM - p["gamma"]*II
    ug = np.linspace(0.0, 1.0, 4001)
    rows = []
    for k in ks:
        V = hjb["Vs"][k].astype(np.float64)
        fwd, bwd, DII = oracle._updiffs(V, dI)
        f_st = oracle.upwind_control_objective(p, II, base, fwd, bwd, DII,
                                               hjb["pol"][k].astype(np.float64))
        u_re = oracle.discrete_min_u(p, II, base, fwd, bwd, DII)
        f_re = oracle.upwind_control_objective(p, II, base, fwd, bwd, DII, u_re)
        gap = f_st - f_re
        gap_d_max = -np.inf                 # chunked dense check (memory)
        for i0 in range(0, len(Ig), 4):
            sl = slice(i0, i0+4)
            f_dense = oracle.upwind_control_objective(
                p, II[sl, :, None], base[sl, :, None], fwd[sl, :, None],
                bwd[sl, :, None], DII[sl, :, None],
                ug[None, None, :]).min(axis=-1)
            gap_d_max = max(gap_d_max, float((f_re[sl] - f_dense).max()))
        rows.append(dict(k=k, stored_vs_remin_max=float(gap.max()),
                         stored_vs_remin_rms=float(np.sqrt((gap**2).mean())),
                         remin_vs_dense_max=gap_d_max))
    return rows
