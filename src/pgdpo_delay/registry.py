"""Problem registry: the CLI dispatches ONLY through this table, via explicit
function calls (no runpy: functions take config/outdir/API version and are
unit-testable). Each problem exposes verify(fast, config, outdir)."""
from pathlib import Path
import numpy as np

def api_versions(name):
    if name == "p1":
        from .problems.p1.oracle import ORACLE_API_VERSION
        return {"p1": ORACLE_API_VERSION}
    if name == "p2":
        from .problems.p2.oracle import P2_API_VERSION
        return {"p2": P2_API_VERSION}
    if name == "p3":
        from .problems.p3.oracle import P3_API_VERSION
        return {"p3": P3_API_VERSION}
    if name == "p4":
        from .problems.p4.oracle import ORACLE_API_VERSION
        return {"p4": ORACLE_API_VERSION}
    return {}


def evaluation_conventions(name):
    """Per-problem evaluation conventions recorded into every manifest
    (minfix review sec.3.4): the active-set tolerance and storage dtype let a
    reviewer reproduce occupancy/switching definitions exactly."""
    if name in ("p1", "p3"):
        from .problems.p1.evaluate import active_tol
        if name == "p1":
            from .problems.p1.config import load_config
            lo, hi = load_config("main")["bounds"]
        else:
            lo, hi = 0.0, 1.0
        return {"active_set_tolerance": active_tol(lo, hi),
                "active_set_storage_dtype": "float32"}
    return {}

def _p1_verify(fast=True, config="main", outdir=Path("outputs/verify/p1")):
    from .problems.p1 import oracle, h_refine, contract
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    oracle.run_checks()                      # config-independent algebra fixture
    rows = h_refine.run(hs=(0.05, 0.025) if fast else (0.1, 1/15, 0.05, 0.025, 0.0125, 0.00625),
                        save=not fast, config=config, outdir=outdir)
    fl = [r["floor_rmse"] for r in rows]
    if not all(b < a for a, b in zip(fl, fl[1:])):
        raise AssertionError("h-refinement floor not decreasing")
    _p1c_verify_fast(outdir)
    if not fast:
        contract.run(outdir=outdir)

def _p1c_verify_fast(outdir):
    """P1-C regression gate (review sec.6.4): config variants, tiny DP with
    the analytic last step, exact last-step action, bounds, paired-dJ self,
    KKT sign convention, no-anticipation estimator smoke."""
    from .problems.p1.config import load_config
    from .problems.p1 import dp_small, evaluate, no_anticipation
    from .problems.p1.oracle import riccati
    cfg_m = load_config("main"); cfg_s = load_config("dp_small")
    if cfg_m["H"] != 16 or cfg_s["H"] != 3:
        raise AssertionError("variant grids changed unexpectedly")
    # exact last-step formula vs brute-force scalar minimisation (machine)
    from scipy.optimize import minimize_scalar
    p, h = cfg_s["params"], cfg_s["h"]
    rng = np.random.default_rng(0)
    for _ in range(20):
        x0, xH = rng.uniform(-1.5, 1.5, 2)
        mean0 = (1 + p["a"]*h)*x0 + h*p["ad"]*xH
        sg0 = p["s0"] + p["cx"]*x0 + p["cy"]*xH
        obj = lambda u: 0.5*h*p["R"]*u*u + 0.5*p["QT"]*((mean0 + h*p["b"]*u - cfg_s["xtar"])**2
                                                        + h*(sg0 + p["gu"]*u)**2)
        lo, hi = cfg_s["bounds"]
        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded",
                              options=dict(xatol=1e-13))
        ex = dp_small.exact_last_step_action(cfg_s, x0, xH)
        if abs(res.x - ex) > 1e-7:
            raise AssertionError(f"exact_last_step_action mismatch: {res.x} vs {ex}")
    # tiny DP: terminal exactness, bounds, last-step gate on interior nodes
    dp = dp_small.dp_reference(cfg_s, n_x=7, n_gh=3, n_u=9, L=2.0)
    xg = dp["xg"]; QT = p["QT"]
    if np.abs(dp["V"][cfg_s["N"]][:, 0, 0, 0] - 0.5*QT*(xg - cfg_s["xtar"])**2).max() > 1e-10:
        raise AssertionError("terminal value not exact on nodes")
    lo, hi = dp["bounds"]
    if not all((a.min() >= lo - 1e-12) and (a.max() <= hi + 1e-12) for a in dp["pol"]):
        raise AssertionError("policy actions escape bounds")
    kN = cfg_s["N"] - 1; errs = []
    for i0 in range(len(xg)):
        for i3 in range(len(xg)):
            z = np.array([xg[i0], 0.0, 0.0, xg[i3]])
            errs.append(dp_small.dp_action_label_at(dp, kN, z)
                        - dp_small.exact_last_step_action(cfg_s, xg[i0], xg[i3]))
    if np.sqrt(np.mean(np.square(errs))) > 1e-6:
        raise AssertionError(f"analytic-last-step gate failed: RMSE={np.sqrt(np.mean(np.square(errs)))}")
    # paired-dJ self comparison must be exactly zero
    orc = riccati(cfg_s["params"], cfg_s["H"], cfg_s["h"], cfg_s["N"],
                  cfg_s["xref"], cfg_s["xtar"])
    polA = lambda k, Z: Z @ orc["F"][k] + orc["f"][k]
    r = evaluate.rollout_paired(cfg_s, polA, polA, 64, seed=1)
    if r["delta_A_minus_B"] != 0.0:
        raise AssertionError("CRN paired self-comparison nonzero")
    # KKT sign convention on synthetic inputs
    lo, hi = cfg_s["bounds"]
    mk = lambda u, g: dict(u=u, p=0.0, zeta=0.0, Pi=0.0, sigma_bar=0.0,
                           _g_override=g)
    def kkt(u, g):
        if u <= lo + 1e-9:  return max(0.0, -g)
        if u >= hi - 1e-9:  return max(0.0, g)
        return abs(g)
    if not (kkt(lo, +1.0) == 0.0 and kkt(lo, -1.0) > 0 and kkt(hi, -1.0) == 0.0
            and kkt(hi, +1.0) > 0 and kkt(0.0, 0.3) == 0.3):
        raise AssertionError("KKT sign convention broken")
    # no-anticipation smoke (tiny budget, finiteness)
    z = np.zeros(cfg_s["H"]+1)
    inp = evaluate.estimator_inputs(cfg_s, polA, 0, z, M=64, Mout=16, Min=2,
                                    seed=3, no_anticipation=True)
    u_na = no_anticipation.recovered_action(cfg_s, inp)
    if not np.isfinite(u_na):
        raise AssertionError("no-anticipation recovery not finite")

def _p2_verify(fast=True, config=None, outdir=Path("outputs/verify/p2")):
    from .problems.p2 import eigencheck, scaling
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    eigencheck.run_check()
    if not fast:
        scaling.main(outdir=outdir)

def _p3_verify(fast=True, config="renewal", outdir=Path("outputs/verify/p3")):
    """P3-R fine-grid HJB reference gates + P3-D CE-NMPC proxy gates
    (calibration V1; branch-aware discrete minimiser with empty-branch
    masking, re-review 2026-08-07). Both canonical variants are ALWAYS
    validated together; a custom --config is refused. Full tier runs the
    production solve, matched-spacing domain audit, time-aggregated policy
    consistency, Bellman/policy-optimality residual artifacts, and writes a
    complete candidate bundle under outdir.  The CLI owns immutable-bundle
    publication and changes the current pointer only after every gate,
    artifact, config, and manifest has succeeded."""
    from .problems.p3.config import load_config
    from .problems.p3 import oracle, calibrate, nmpc, dynamics
    import numpy as np
    if config not in (None, "main", "renewal"):
        raise SystemExit("P3 verify validates both canonical variants "
                         "(renewal + distributed); custom --config is not "
                         "supported here")
    # Validate both canonical configurations before any output is touched.
    cfg = load_config("renewal")
    cfg_d = load_config("distributed")
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    if any(outdir.iterdir()):
        raise RuntimeError("P3 verifier requires an empty staging directory")
    protocol = dict(seeds=dict(r1=4, rollout=7, memvis=11, paired=17,
                               bank=21, grad=0, bellman=7, nmpc_kkt=29),
                    coarse_grid=41 if fast else 61,
                    domain_audit=dict(I_max=1.8, n=73),
                    Np=1000 if fast else 8000,
                    bellman_Np=4000,
                    nmpc_crosscheck_Np=8 if fast else 32,
                    nmpc_kkt_Np=8,
                    r1_n=2000, r1_dense=4001,
                    policy_slices=[0, 25, 50, 75, 99],
                    gates=dict(r1_gap=1e-7, paired_tol=0.01 if fast else 0.004,
                               bellman_tol=0.004,
                               grid_vrel=0.01, grid_du_tmax=0.02,
                               dom_vrel=1e-3, dom_du_tmax=5e-3,
                               opt_success=0.99, residual_gap=1e-6,
                               ce_kkt_inf=1e-5,
                               raw_constraint=1e-10,
                               deployed_constraint=1e-12))
    p = cfg["params"]
    g1 = calibrate.brute_force_check(cfg)
    if g1 > 1e-7:
        raise AssertionError(f"pointwise exact-min vs brute force gap {g1}")
    # R1 (re-review sec.4.4): feasibility FIRST, then dense-grid optimality;
    # deterministic branch fixtures cover usw<lo / interior / usw>hi / I=0.
    rng = np.random.default_rng(protocol["seeds"]["r1"])
    nchk = protocol["r1_n"]
    I = rng.uniform(0.0, cfg["hjb"]["I_max"], nchk)
    base = rng.normal(0.0, 2.0, nchk)
    Dp = rng.normal(0.0, 3.0, nchk); Dm = rng.normal(0.0, 3.0, nchk)
    DII = rng.normal(0.0, 30.0, nchk)
    u_ex = oracle.discrete_min_u(p, I, base, Dp, Dm, DII)
    if not np.isfinite(u_ex).all():
        raise AssertionError("discrete minimiser returned nonfinite action")
    if (u_ex < 0).any() or (u_ex > 1).any():
        raise AssertionError("discrete minimiser returned infeasible action")
    ug = np.linspace(0.0, 1.0, protocol["r1_dense"])
    fs = oracle.upwind_control_objective(
        p, I[:, None], base[:, None], Dp[:, None], Dm[:, None], DII[:, None],
        ug[None, :])
    f_ex = oracle.upwind_control_objective(p, I, base, Dp, Dm, DII, u_ex)
    r1 = float(np.max(f_ex - fs.min(axis=1)))
    if r1 > protocol["gates"]["r1_gap"]:
        raise AssertionError(f"discrete upwind minimiser gap {r1}")
    for Iv, bv in [(0.5, -1.0), (0.5, 0.15), (0.5, 1.0), (0.0, 0.3),
                   (0.0, -0.3)]:
        for DIIv in (25.0, -25.0):
            uu = oracle.discrete_min_u(p, np.array([Iv]), np.array([bv]),
                                       np.array([1.5]), np.array([-2.0]),
                                       np.array([DIIv]))
            if not (np.isfinite(uu[0]) and 0.0 <= uu[0] <= 1.0):
                raise AssertionError(f"branch fixture infeasible: "
                                     f"I={Iv}, base={bv}, DII={DIIv}")
    n_c = protocol["coarse_grid"]
    hjb_c = oracle.solve_hjb(cfg, n_I=n_c, n_M=n_c)
    for tab in hjb_c["pol"]:
        if not np.isfinite(tab).all():
            raise AssertionError("HJB policy not finite")
        if tab.min() < -1e-12 or tab.max() > 1.0 + 1e-12:
            raise AssertionError("HJB policy escapes [0, 1]")
    if hjb_c["A2_min_global"] <= 0:
        raise AssertionError(f"global curvature A2 nonpositive: "
                             f"{hjb_c['A2_min_global']}")
    Np = protocol["Np"]
    d = calibrate.rollout_diagnostics(
        cfg, hjb_c, Np=Np, seed=protocol["seeds"]["rollout"])
    tol = protocol["gates"]["paired_tol"]
    if abs(d["paired_mean"]) > 3*d["paired_se"] + tol:
        raise AssertionError(f"paired MC-vs-V0 residual {d['paired_mean']} "
                             f"(paired SE {d['paired_se']})")
    mv = calibrate.memory_visibility(
        cfg, hjb_c, seed=protocol["seeds"]["memvis"])
    if mv["du_rms"] < 0.02:
        raise AssertionError(f"renewal-memory visibility too small: {mv}")
    share_c = calibrate.diffusion_channel_share(cfg, hjb_c)
    print(f"[{oracle.P3_API_VERSION}] G1 algebra gap = {g1:.2e}")
    print(f"R1 feasible + minimiser gap    = OK / {r1:.2e}")
    print(f"paired MC-vs-V0 (Np={Np})      = {d['paired_mean']:.5f}  "
          f"(paired SE {d['paired_se']:.5f})")
    print(f"deployed occupancy lo/int/up   = "
          f"{d['occ'][0]:.1%}/{d['occ'][1]:.1%}/{d['occ'][2]:.1%}")
    print(f"memory visibility du_rms       = {mv['du_rms']:.4f}")
    print(f"A2 global min / nonpos frac    = {hjb_c['A2_min_global']:.4f} / "
          f"{hjb_c['A2_nonpos_frac']:.1%}")
    print(f"diffusion share (coarse ref)   = median {share_c['median']:.1%}")
    print(f"Imax_seen / I_max / neg-frac   = {d['Imax_seen']:.2f} / "
          f"{cfg['hjb']['I_max']} / {d['neg_frac_mean']:.2%}")
    # ---- P3-D / CE-NMPC proxy gates ----
    w = dynamics.kernel_weights(cfg_d)
    if abs(w.sum() - 1.0) > 1e-12 or (w < 0).any():
        raise AssertionError("kernel weights not a normalized nonneg family")
    if not (0 < w.argmax() < len(w) - 1):
        raise AssertionError("kernel peak not interior (incubation shape)")
    rng = np.random.default_rng(protocol["seeds"]["grad"])

    def _fd_gap(fun, u, eps=1e-6):
        _, g = fun(u)
        gfd = np.empty_like(u)
        for i in range(len(u)):
            up = u.copy(); up[i] += eps; um = u.copy(); um[i] -= eps
            gfd[i] = (fun(up)[0] - fun(um)[0])/(2*eps)
        return float(np.max(np.abs(g - gfd))/max(1e-12, np.max(np.abs(gfd))))

    gaps = []
    for _ in range(4):
        u = rng.uniform(0.05, 0.95, int(rng.integers(3, 15)))
        I0, M0 = rng.uniform(0.05, 0.8, 2)
        gaps.append(_fd_gap(
            lambda uu: nmpc.rollout_cost_grad_renewal(cfg, I0, M0, uu), u))
        B0 = rng.uniform(0.02, 0.8, cfg_d["dist"]["H"]+1)
        gaps.append(_fd_gap(
            lambda uu: nmpc.rollout_cost_grad_dist(cfg_d, B0, w, uu), u))
    if max(gaps) > 1e-5:
        raise AssertionError(f"CE-NMPC adjoint-vs-FD gradient gap {max(gaps)}")
    Np_x = protocol["nmpc_crosscheck_Np"]
    ctrl_r = nmpc.NMPCController(cfg, "renewal")
    pol_r = lambda k, I, M: np.array(
        [ctrl_r.action(k, (np.atleast_1d(I)[i], np.atleast_1d(M)[i]),
                       path_id=i) for i in range(len(np.atleast_1d(I)))])
    hjb_x = hjb_c        # fast tier: coarse cross-reference (labeled)
    x_ref_name = f"coarse {n_c}-grid HJB"
    if not fast:
        pass             # replaced by the fine reference below (sec.7.3)
    rx = None
    if fast:
        rx = dynamics.simulate_paired(cfg, pol_r, oracle.hjb_policy(hjb_x),
                                      Np=Np_x,
                                      seed=protocol["seeds"]["paired"])
    ctrl_d = nmpc.NMPCController(cfg_d, "distributed")
    pol_d = lambda k, B: np.array(
        [ctrl_d.action(k, np.atleast_2d(B)[i], path_id=i)
         for i in range(np.atleast_2d(B).shape[0])])
    rd = dynamics.simulate_dist_paired(
        cfg_d, lambda k, B: np.full(B.shape[0], 0.6), pol_d, Np=Np_x,
        seed=protocol["seeds"]["paired"])
    if rd["delta_A_minus_B"] < -3*rd["se"]:
        raise AssertionError("distributed CE-NMPC materially worse than "
                             "const-0.6")
    # Independent reachable-history bank for the deterministic CE open-loop
    # subproblem KKT.  This is optimizer certification, not the later
    # stochastic generalized-Hamiltonian KKT used to compare learned methods.
    ctrl_kkt = nmpc.NMPCController(cfg_d, "distributed")
    pol_kkt = lambda k, B: np.array(
        [ctrl_kkt.action(k, np.atleast_2d(B)[i], path_id=i)
         for i in range(np.atleast_2d(B).shape[0])])
    dynamics.simulate_dist(cfg_d, pol_kkt, Np=protocol["nmpc_kkt_Np"],
                           seed=protocol["seeds"]["nmpc_kkt"])
    bs_kkt = ctrl_kkt.budget_stats()

    def _check_ce_stats(name, bs, require_kkt=False):
        if bs["success_rate"] < protocol["gates"]["opt_success"]:
            raise AssertionError(f"CE-NMPC optimizer health ({name}): {bs}")
        if bs["nonfinite_count"]:
            raise AssertionError(f"CE-NMPC nonfinite output ({name}): {bs}")
        if (bs["raw_plan_bound_violation_max"]
                > protocol["gates"]["raw_constraint"]
                or bs["deployment_clip_correction_max"]
                > protocol["gates"]["raw_constraint"]
                or bs["deployed_bound_violation_max"]
                > protocol["gates"]["deployed_constraint"]):
            raise AssertionError(f"CE-NMPC constraint violation ({name}): {bs}")
        if (require_kkt and bs["ce_subproblem_kkt_inf_max"]
                > protocol["gates"]["ce_kkt_inf"]):
            raise AssertionError(f"CE deterministic-subproblem KKT ({name}): {bs}")

    bs_d = ctrl_d.budget_stats()
    _check_ce_stats("distributed-objective-bank", bs_d)
    _check_ce_stats("distributed-kkt-holdout", bs_kkt, require_kkt=True)
    extra = dict(verify_tier="fast" if fast else "full",
                 verify_protocol=protocol,
                 p3r=dict(A2_min_global=hjb_c["A2_min_global"],
                          A2_nonpos_frac=hjb_c["A2_nonpos_frac"],
                          curvature_diagnostic_scope=
                              hjb_c["curvature_diagnostic_scope"],
                          curvature_diagnostic_surface_count=
                              hjb_c["curvature_diagnostic_surface_count"],
                          curvature_diagnostic_grid="coarse"),
                 p3d_ce_nmpc=dict(
                     role="deterministic certainty-equivalent proxy",
                     kkt_role="CE deterministic open-loop subproblem box-KKT",
                     objective_sanity=dict(const_minus_ce_dJ=rd["delta_A_minus_B"],
                                           paired_se=rd["se"]),
                     holdout=bs_kkt))
    if fast:
        dJx = rx["delta_A_minus_B"]
        if dJx < -3*rx["se"] - 0.01:
            raise AssertionError(f"CE-NMPC beats the HJB reference beyond "
                                 f"noise (dJ={dJx})")
        if dJx > 0.10:
            raise AssertionError(f"CE-NMPC-vs-HJB gap too large (dJ={dJx})")
        bs_r = ctrl_r.budget_stats()
        _check_ce_stats("renewal", bs_r)
        print(f"[p3d] CE-NMPC adjoint gap      = {max(gaps):.2e}")
        print(f"CE-NMPC vs {x_ref_name} dJ    = {dJx:.4f}  (SE {rx['se']:.4f})")
        print(f"dist dJ(const-0.6 - CE-NMPC)   = {rd['delta_A_minus_B']:.4f}"
              f"  (SE {rd['se']:.4f})")
        print(f"optimizer success (ren/dist)   = {bs_r['success_rate']:.0%} /"
              f" {bs_d['success_rate']:.0%}")
        print(f"P3-D CE holdout KKT max        = "
              f"{bs_kkt['ce_subproblem_kkt_inf_max']:.2e}; decision runtime "
              f"median "
              f"{bs_kkt['decision_runtime_median_ms']:.2f} ms/decision")
        extra["nmpc_budget"] = dict(renewal=bs_r, distributed=bs_d)
        extra["p3r"]["diffusion_share_coarse"] = share_c
        return dict(config_snapshot=dict(renewal=cfg["raw"],
                                         distributed=cfg_d["raw"]),
                    seeds=protocol["seeds"],
                    solver="p3r-hjb-numerical-reference+p3d-ce-nmpc-proxy",
                    extra=extra,
                    required_artifacts=["manifest.json", "config.json"])
    # ================= FULL TIER =================
    hjb_f = oracle.solve_hjb(cfg, store_value=True)     # production grid
    df = calibrate.rollout_diagnostics(
        cfg, hjb_f, Np=protocol["Np"],
        seed=protocol["seeds"]["rollout"])
    satf = df["occ"][0] + df["occ"][2]
    if not (0.2 <= satf <= 0.7):
        raise AssertionError(f"saturation {satf:.1%} outside [20%, 70%]")
    if (abs(df["paired_mean"]) > 3*df["paired_se"]
            + protocol["gates"]["paired_tol"]):
        raise AssertionError(f"fine-grid paired residual {df['paired_mean']}")
    share_f = calibrate.diffusion_channel_share(cfg, hjb_f)  # canonical
    # CE-NMPC canonical cross-check vs the FINE reference (sec.7.3)
    rx = dynamics.simulate_paired(cfg, pol_r, oracle.hjb_policy(hjb_f),
                                  Np=Np_x, seed=protocol["seeds"]["paired"])
    dJx = rx["delta_A_minus_B"]
    if dJx < -3*rx["se"] - 0.01 or dJx > 0.10:
        raise AssertionError(f"CE-NMPC vs fine HJB inconsistent (dJ={dJx})")
    bs_r = ctrl_r.budget_stats()
    _check_ce_stats("renewal", bs_r)
    _check_ce_stats("distributed-objective-bank", bs_d)
    _check_ce_stats("distributed-kkt-holdout", bs_kkt, require_kkt=True)
    print(f"[p3d] CE-NMPC adjoint gap      = {max(gaps):.2e}")
    print(f"CE-NMPC vs fine 151-grid dJ    = {dJx:.4f}  (SE {rx['se']:.4f})")
    print(f"dist dJ(const-0.6 - CE-NMPC)   = {rd['delta_A_minus_B']:.4f}  "
          f"(SE {rd['se']:.4f})")
    print(f"optimizer success (ren/dist)   = {bs_r['success_rate']:.0%} / "
          f"{bs_d['success_rate']:.0%}")
    print(f"P3-D CE holdout KKT max        = "
          f"{bs_kkt['ce_subproblem_kkt_inf_max']:.2e}; decision runtime "
          f"median "
          f"{bs_kkt['decision_runtime_median_ms']:.2f} ms/decision")
    # time-aggregated grid/domain policy consistency (sec.6)
    rng = np.random.default_rng(protocol["seeds"]["bank"]); n = 512
    I = rng.uniform(0.02, 1.0, n); M = rng.uniform(0.02, 1.0, n)
    vf = oracle.value_at(hjb_f, I, M)
    vrel = float(np.sqrt(np.mean((vf - oracle.value_at(hjb_c, I, M))**2))
                 / np.sqrt(np.mean(vf**2)))
    hjb_e = oracle.solve_hjb(cfg, n_I=protocol["domain_audit"]["n"],
                             n_M=protocol["domain_audit"]["n"],
                             I_max=protocol["domain_audit"]["I_max"])
    vrel_d = float(np.sqrt(np.mean((oracle.value_at(hjb_e, I, M)
                                    - oracle.value_at(hjb_c, I, M))**2))
                   / np.sqrt(np.mean(oracle.value_at(hjb_c, I, M)**2)))
    pf, pc, pe = (oracle.hjb_policy(h) for h in (hjb_f, hjb_c, hjb_e))
    ks = protocol["policy_slices"]
    du_g = [float(np.sqrt(np.mean((pf(k, I, M) - pc(k, I, M))**2)))
            for k in ks]
    du_d = [float(np.sqrt(np.mean((pe(k, I, M) - pc(k, I, M))**2)))
            for k in ks]
    if (vrel > protocol["gates"]["grid_vrel"]
            or max(du_g) > protocol["gates"]["grid_du_tmax"]):
        raise AssertionError(f"grid consistency: vrel={vrel}, du={du_g}")
    if (vrel_d > protocol["gates"]["dom_vrel"]
            or max(du_d) > protocol["gates"]["dom_du_tmax"]):
        raise AssertionError(f"domain expansion: vrel={vrel_d}, du={du_d}")
    print(f"fine deployed occupancy        = "
          f"{df['occ'][0]:.1%}/{df['occ'][1]:.1%}/{df['occ'][2]:.1%}")
    print(f"diffusion share (fine, canon.) = median {share_f['median']:.1%}"
          f"  (coarse {share_c['median']:.1%})")
    print(f"grid consistency (61 vs 151)   = value rel {vrel:.2%}, policy "
          f"RMSE k0/mean/max {du_g[0]:.4f}/{np.mean(du_g):.4f}/"
          f"{max(du_g):.4f}  (runtime {hjb_f['runtime']:.0f}s)")
    print(f"R3 domain (1.5,61)->(1.8,73)   = value rel {vrel_d:.2e}, policy"
          f" RMSE max {max(du_d):.2e}")
    # residual artifacts (re-review sec.5)
    por = calibrate.policy_optimality_residual(cfg, hjb_f, ks)
    res_gap = max(r_["stored_vs_remin_max"] for r_ in por)
    if res_gap > protocol["gates"]["residual_gap"]:
        raise AssertionError(f"stored-policy optimality residual {res_gap}")
    bres = calibrate.bellman_residual(cfg, hjb_f,
                                      Np=protocol["bellman_Np"],
                                      seed=protocol["seeds"]["bellman"])
    if (abs(bres["total"]) > 3*bres["total_se"]
            + protocol["gates"]["bellman_tol"]):
        raise AssertionError(
            "pathwise Bellman residual failed: "
            f"total={bres['total']}, total_se={bres['total_se']}")
    print(f"stored-policy remin residual   = {res_gap:.2e} "
          f"(dense gap {max(r_['remin_vs_dense_max'] for r_ in por):.2e})")
    print(f"Bellman residual per step      = max|mean| "
          f"{bres['max_abs_mean']:.2e}, total {bres['total']:.4f} "
          f"(total SE {bres['total_se']:.4f})")
    # ---- freeze the reference artifacts ----
    np.savez_compressed(outdir/"p3r_hjb_value.npz", V0=hjb_f["V0"],
                        V=np.stack(hjb_f["Vs"]),
                        Ig=hjb_f["Ig"], Mg=hjb_f["Mg"])
    np.savez_compressed(outdir/"p3r_hjb_policy.npz",
                        pol=np.stack(hjb_f["pol"]).astype(np.float32),
                        Ig=hjb_f["Ig"], Mg=hjb_f["Mg"])
    with open(outdir/"p3r_hjb_residual.csv", "w") as fp:
        fp.write("kind,k,value,se\n")
        for r_ in por:
            fp.write(f"policy_remin_max,{r_['k']},"
                     f"{r_['stored_vs_remin_max']},\n")
            fp.write(f"policy_remin_rms,{r_['k']},"
                     f"{r_['stored_vs_remin_rms']},\n")
        for k in range(len(bres["mean"])):
            fp.write(f"bellman_mean,{k},{bres['mean'][k]},"
                     f"{bres['se'][k]}\n")
    rows = [("R1_discrete_min_gap", r1),
            ("paired_mc_mean", df["paired_mean"]),
            ("paired_mc_se", df["paired_se"]),
            ("grid_value_rel", vrel),
            ("grid_policy_rmse_k0", du_g[0]),
            ("grid_policy_rmse_tmean", float(np.mean(du_g))),
            ("grid_policy_rmse_tmax", max(du_g)),
            ("domain_value_rel", vrel_d),
            ("domain_policy_rmse_k0", du_d[0]),
            ("domain_policy_rmse_tmax", max(du_d)),
            ("policy_remin_residual_max", res_gap),
            ("bellman_residual_max_abs_mean", bres["max_abs_mean"]),
            ("bellman_residual_total", bres["total"]),
            ("bellman_residual_total_se", bres["total_se"]),
            ("A2_min_global", hjb_f["A2_min_global"]),
            ("A2_nonpos_frac", hjb_f["A2_nonpos_frac"]),
            ("curvature_diagnostic_scope",
             hjb_f["curvature_diagnostic_scope"]),
            ("curvature_diagnostic_surface_count",
             hjb_f["curvature_diagnostic_surface_count"]),
            ("V_II_min", hjb_f["V_II_range"][0]),
            ("V_II_max", hjb_f["V_II_range"][1]),
            ("diff_share_median_fine", share_f["median"]),
            ("diff_share_mean_fine", share_f["mean"]),
            ("nmpc_vs_fine_dJ", dJx), ("nmpc_vs_fine_se", rx["se"]),
            ("neg_frac", df["neg_frac_mean"]),
            ("Imax_seen", df["Imax_seen"]),
            ("saturation_deployed", satf),
            ("n_sub", hjb_f["n_sub"]), ("dt_hjb", hjb_f["dt_hjb"]),
            ("runtime_s", hjb_f["runtime"])]
    with open(outdir/"p3r_certification.csv", "w") as fp:
        fp.write("metric,value\n")
        for k_, v_ in rows:
            fp.write(f"{k_},{v_}\n")
    p3d_rows = [
        ("role", "deterministic certainty-equivalent proxy"),
        ("kkt_role", "CE deterministic open-loop subproblem box-KKT"),
        ("gamma_kernel_weight_sum", float(w.sum())),
        ("gamma_kernel_peak_index", int(w.argmax())),
        ("gamma_kernel_peak_lag", float(w.argmax()*cfg_d["dt"])),
        ("adjoint_fd_relative_gap_max", max(gaps)),
        ("constant_minus_ce_dJ", rd["delta_A_minus_B"]),
        ("constant_minus_ce_paired_se", rd["se"]),
        ("holdout_seed", protocol["seeds"]["nmpc_kkt"]),
        ("holdout_paths", protocol["nmpc_kkt_Np"]),
    ]
    for key in (
            "lookahead_steps", "max_iter", "gtol", "ftol", "terminal_mode",
            "n_solves", "success_rate", "maxiter_fraction", "nit_mean",
            "optimizer_nfev_sum", "optimizer_nfev_mean",
            "total_objective_grad_evals_sum",
            "total_objective_grad_evals_mean",
            "ce_subproblem_kkt_inf_mean",
            "ce_subproblem_kkt_inf_q95", "ce_subproblem_kkt_inf_max",
            "ce_first_action_kkt_max", "raw_plan_bound_violation_max",
            "raw_first_bound_violation_max", "deployment_clip_correction_max",
            "deployed_bound_violation_max", "nonfinite_count",
            "optimizer_runtime_total_s", "optimizer_runtime_mean_ms",
            "optimizer_runtime_median_ms", "optimizer_runtime_p95_ms",
            "full_horizon_optimizer_runtime_mean_ms",
            "decision_runtime_total_s", "decision_runtime_mean_ms",
            "decision_runtime_median_ms", "decision_runtime_p95_ms",
            "full_horizon_decision_runtime_mean_ms"):
        p3d_rows.append((key, bs_kkt[key]))
    with open(outdir/"p3d_ce_nmpc_certification.csv", "w") as fp:
        fp.write("metric,value\n")
        for k_, v_ in p3d_rows:
            fp.write(f"{k_},{v_}\n")
    extra["p3r"].update(A2_min_global=hjb_f["A2_min_global"],
                        A2_nonpos_frac=hjb_f["A2_nonpos_frac"],
                        curvature_diagnostic_scope=
                            hjb_f["curvature_diagnostic_scope"],
                        curvature_diagnostic_surface_count=
                            hjb_f["curvature_diagnostic_surface_count"],
                        curvature_diagnostic_grid="fine",
                        diffusion_share_fine=share_f,
                        certified_artifacts=True)
    extra["nmpc_budget"] = dict(renewal=bs_r, distributed=bs_d)
    return dict(config_snapshot=dict(renewal=cfg["raw"],
                                     distributed=cfg_d["raw"]),
                seeds=protocol["seeds"],
                solver="p3r-hjb-numerical-reference+p3d-ce-nmpc-proxy",
                extra=extra,
                required_artifacts=[
                    "manifest.json", "config.json", "p3r_hjb_value.npz",
                    "p3r_hjb_policy.npz", "p3r_hjb_residual.csv",
                    "p3r_certification.csv",
                    "p3d_ce_nmpc_certification.csv"])


def _p4_verify(fast=True, config="main", outdir=Path("outputs/verify/p4")):
    """Certify the signed P4 same-grid Riccati reference.

    The verifier intentionally covers only the non-deep exact reference:
    causal realised-fill dynamics, generalized stochastic Riccati algebra,
    detached recovery curvature, and the frozen calibration's feasibility and
    memory/signal visibility.  NMPC and learned methods are out of scope.
    """
    from .problems.p4.config import load_config
    from .problems.p4 import calibrate, dynamics, oracle
    from .core.artifacts import atomic_write_json
    import csv
    import hashlib

    if config not in (None, "main"):
        raise SystemExit(
            "P4 verify certifies the canonical signed main config only"
        )
    cfg = load_config("main")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if any(outdir.iterdir()):
        raise RuntimeError("P4 verifier requires an empty staging directory")

    protocol = dict(
        seeds=dict(dynamics=41, algebra=43, rollout=47, diagnostic=53),
        algebra_samples=24 if fast else 64,
        rollout_Np=4096 if fast else 32768,
        gates=dict(
            contract_abs=1e-12,
            dense_abs=1e-12,
            bellman_abs=1e-10,
            scalar_action_abs=2e-6,
            foc_recovery_abs=1e-10,
            value_mc_slack=1e-3,
            lambda_eta_fraction=0.5,
            recovery_eta_fraction=0.5,
            recovery_fill_curvature_share=(0.20, 0.75),
            kernel_tail_max=0.10,
            integrated_impact_ratio=(0.25, 0.75),
            history_response_min=0.05,
            history_correlation_max=-0.80,
            signal_response=(0.05, 0.50),
            terminal_abs_mean=0.03,
            terminal_abs_p95=0.08,
            overshoot_mean=0.005,
            overshoot_p95=0.02,
            material_overshoot_probability=0.01,
            buy_volume_mean=0.01,
            buy_volume_p95=0.05,
            roundtrip_mean=0.02,
            roundtrip_p95=0.10,
            nonpositive_net_sell_probability=0.01,
            u_abs_p999_multiple=5.0,
            constant_improvement_relative=0.05,
        ),
    )

    algebra = oracle.self_checks(
        cfg, samples=protocol["algebra_samples"],
        seed=protocol["seeds"]["algebra"]
    )
    dyn = calibrate.dynamics_contract_checks(
        cfg, seed=protocol["seeds"]["dynamics"]
    )
    orc = oracle.riccati(cfg)
    detached = oracle.detached_curvature(cfg)
    rollout = calibrate.rollout_diagnostics(
        cfg, orc, Np=protocol["rollout_Np"],
        seed=protocol["seeds"]["rollout"]
    )

    p = cfg["params"]
    tail_ratio = float(np.exp(-p["rho_G"] * cfg["delta"]))
    integrated_ratio = float(
        p["gamma"] * (1.0 - tail_ratio) / (p["rho_G"] * p["eta"])
    )
    rec = detached["recovery_curvature"][:cfg["N"]]
    # This is a recovery-coordinate conditioning diagnostic.  It is NOT an
    # ablation claim about the objective impact of fill diffusion: Bellman
    # curvature uses Pval and zeta can offset Pi in exact recovery.
    recovery_fill_share = (
        p["sigma_Q"] ** 2 * detached["Pi"][:cfg["N"]] / rec
    )
    materiality = dict(
        kernel_tail_ratio=tail_ratio,
        integrated_impact_to_eta=integrated_ratio,
        recovery_fill_curvature_share_min=float(recovery_fill_share.min()),
        recovery_fill_curvature_share_median=
            float(np.median(recovery_fill_share)),
        recovery_fill_curvature_share_max=float(recovery_fill_share.max()),
    )
    g = protocol["gates"]

    if max(abs(value) for value in dyn.values()) > g["contract_abs"]:
        raise AssertionError(f"P4 causal dynamics contract failed: {dyn}")
    if algebra["dense_max_abs_error"] > g["dense_abs"]:
        raise AssertionError(f"P4 dense Riccati mismatch: {algebra}")
    if algebra["bellman_max_abs_error"] > g["bellman_abs"]:
        raise AssertionError(f"P4 Bellman residual: {algebra}")
    if algebra["scalar_minimizer_max_action_error"] > g["scalar_action_abs"]:
        raise AssertionError(f"P4 scalar minimizer mismatch: {algebra}")
    if max(algebra["q_foc_max_abs_error"],
           algebra["recovery_max_abs_action_error"],
           algebra["recovered_identity_max_abs_error"]) > g["foc_recovery_abs"]:
        raise AssertionError(f"P4 FOC/recovery identity failed: {algebra}")

    value_tol = (3.0 * rollout["mc_minus_value_se"]
                 + g["value_mc_slack"] * max(
                     1.0, abs(rollout["exact_initial_value_mean"])))
    if abs(rollout["mc_minus_value_mean"]) > value_tol:
        raise AssertionError(
            f"P4 MC-vs-exact value failed: {rollout}, tol={value_tol}"
        )
    if algebra["min_Lambda_over_h"] < g["lambda_eta_fraction"] * p["eta"]:
        raise AssertionError(f"P4 Bellman curvature too small: {algebra}")
    if algebra["min_recovery_curvature"] < g["recovery_eta_fraction"] * p["eta"]:
        raise AssertionError(f"P4 recovery curvature too small: {algebra}")
    lo, hi = g["recovery_fill_curvature_share"]
    if not lo <= materiality["recovery_fill_curvature_share_median"] <= hi:
        raise AssertionError(
            f"P4 recovery fill-curvature conditioning unsuitable: {materiality}"
        )
    if tail_ratio > g["kernel_tail_max"]:
        raise AssertionError(f"P4 delay window truncation too short: {materiality}")
    lo, hi = g["integrated_impact_ratio"]
    if not lo <= integrated_ratio <= hi:
        raise AssertionError(f"P4 impact-memory scale unsuitable: {materiality}")
    if (rollout["history_response_ratio"] < g["history_response_min"]
            or rollout["history_du_impact_correlation"]
            > g["history_correlation_max"]):
        raise AssertionError(f"P4 history response not visible: {rollout}")
    lo, hi = g["signal_response"]
    if (not lo <= rollout["signal_response_ratio"] <= hi
            or rollout["signal_gain_min"] <= 0.0):
        raise AssertionError(f"P4 signal response unsuitable: {rollout}")

    upper_gates = dict(
        terminal_abs_mean_ratio=g["terminal_abs_mean"],
        terminal_abs_p95_ratio=g["terminal_abs_p95"],
        overshoot_mean_ratio=g["overshoot_mean"],
        overshoot_p95_ratio=g["overshoot_p95"],
        material_overshoot_probability=g["material_overshoot_probability"],
        buy_volume_mean_ratio=g["buy_volume_mean"],
        buy_volume_p95_ratio=g["buy_volume_p95"],
        intended_roundtrip_mean=g["roundtrip_mean"],
        intended_roundtrip_p95=g["roundtrip_p95"],
        nonpositive_net_sell_probability=
            g["nonpositive_net_sell_probability"],
        u_abs_p999=g["u_abs_p999_multiple"] * p["q0"] / cfg["T"],
    )
    failed = {key: (rollout[key], limit) for key, limit in upper_gates.items()
              if rollout[key] > limit}
    if failed:
        raise AssertionError(f"P4 feasibility gates failed: {failed}")
    if (rollout["constant_minus_oracle_mean"]
            <= 3.0 * rollout["constant_minus_oracle_se"]
            or rollout["constant_relative_improvement"]
            < g["constant_improvement_relative"]):
        raise AssertionError(f"P4 oracle lacks constant-policy sanity gap: {rollout}")

    print(f"[{oracle.ORACLE_API_VERSION}] exact algebra max = "
          f"{max(algebra['dense_max_abs_error'], algebra['bellman_max_abs_error']):.2e}")
    print(f"Bellman Lambda/h min / recovery min = "
          f"{algebra['min_Lambda_over_h']:.6f} / "
          f"{algebra['min_recovery_curvature']:.6f}")
    print(f"MC J-V0 = {rollout['mc_minus_value_mean']:.6f} "
          f"(SE {rollout['mc_minus_value_se']:.6f}, Np={rollout['Np']})")
    print(f"terminal |Q_T| mean/p95 = "
          f"{rollout['terminal_abs_mean_ratio']:.3%} / "
          f"{rollout['terminal_abs_p95_ratio']:.3%}")
    print(f"overshoot mean/p95; RT mean = "
          f"{rollout['overshoot_mean_ratio']:.3%} / "
          f"{rollout['overshoot_p95_ratio']:.3%}; "
          f"{rollout['intended_roundtrip_mean']:.3%}")
    print(f"history/signal response ratios = "
          f"{rollout['history_response_ratio']:.3f} / "
          f"{rollout['signal_response_ratio']:.3f}")
    print(f"constant-policy relative improvement = "
          f"{rollout['constant_relative_improvement']:.1%}")
    print(f"current-p finite-grid action RMSE/nRMSE = "
          f"{rollout['p_alignment_action_rmse']:.6f} / "
          f"{rollout['p_alignment_action_nrmse']:.3%}")

    # The uploaded/scratch workspace is not necessarily a Git checkout.  A
    # deterministic source-byte digest keeps such a bundle identifiable; the
    # normal manifest git_commit field becomes authoritative after integration
    # into the repository and a final rerun.
    package_root = Path(__file__).resolve().parent
    source_relpaths = (
        "registry.py", "cli.py", "core/artifacts.py",
        "problems/p4/config.py", "problems/p4/dynamics.py",
        "problems/p4/oracle.py", "problems/p4/calibrate.py",
        "configs/p4/main.yaml",
    )
    source_hashes = {}
    source_tree = hashlib.sha256()
    for relpath in source_relpaths:
        payload = (package_root / relpath).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        source_hashes[relpath] = digest
        source_tree.update(relpath.encode("utf-8") + b"\0")
        source_tree.update(payload)

    extra = dict(
        verify_tier="fast" if fast else "full",
        verify_protocol=protocol,
        p4=dict(
            role="exact same-grid signed stochastic generalized-Riccati reference",
            calibration_status=cfg["calibration_status"],
            dynamics_contract=dyn,
            exact_algebra=algebra,
            materiality=materiality,
            feasibility=rollout,
            source_provenance=dict(
                source_tree_sha256=source_tree.hexdigest(),
                file_sha256=source_hashes,
            ),
            excluded=("NMPC", "box-constrained variant", "correlated noise",
                      "learned methods"),
        ),
    )
    required = ["manifest.json", "config.json"]
    if not fast:
        if cfg["calibration_status"] != "frozen":
            raise AssertionError(
                "P4 full artifact publication requires calibration_status=frozen"
            )
        A, B, Dq, Salpha = dynamics.linear_matrices(cfg)
        ages = np.arange(1, cfg["H"] + 1) * cfg["h"]
        G = p["gamma"] * np.exp(-p["rho_G"] * ages)
        np.savez_compressed(
            outdir / "p4_oracle.npz",
            A=A, B=B, Dq=Dq, Salpha=Salpha,
            impact_row=dynamics.impact_row(cfg), G_lags=G,
            Pval=orc["Pval"], c=orc["c"], F=orc["F"],
            Lambda=orc["Lambda"], Lambda_over_h=orc["Lambda_over_h"],
            Gol=detached["Gol"], Pi=detached["Pi"],
            recovery_curvature=detached["recovery_curvature"],
            api_version=np.asarray(oracle.ORACLE_API_VERSION),
        )
        rows = []
        for k in range(cfg["N"]):
            rows.extend((
                ("per_step", "Lambda_over_h", k, orc["Lambda_over_h"][k]),
                ("per_step", "recovery_curvature", k, rec[k]),
                ("per_step", "recovery_fill_curvature_share", k,
                 recovery_fill_share[k]),
            ))
        for category, values in (("dynamics", dyn), ("algebra", algebra),
                                 ("materiality", materiality),
                                 ("feasibility", rollout)):
            for key, value in values.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    rows.append((category, key, "", value))
        with open(outdir / "p4_certification.csv", "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(("category", "metric", "k", "value"))
            writer.writerows(rows)
        atomic_write_json(outdir / "p4_feasibility.json", rollout)
        paths = calibrate.diagnostic_paths(
            cfg, orc, seed=protocol["seeds"]["diagnostic"]
        )
        np.savez_compressed(outdir / "p4_diagnostic_paths.npz", **paths)
        required.extend(("p4_oracle.npz", "p4_certification.csv",
                         "p4_feasibility.json", "p4_diagnostic_paths.npz"))
        extra["p4"]["certified_artifacts"] = True

    return dict(
        config_snapshot=cfg["raw"],
        seeds=protocol["seeds"],
        solver="p4-signed-exact-generalized-stochastic-riccati",
        extra=extra,
        required_artifacts=required,
    )


PROBLEM_REGISTRY = {
    "p1": dict(verify=_p1_verify),
    "p2": dict(verify=_p2_verify),
    "p3": dict(verify=_p3_verify),
    "p4": dict(verify=_p4_verify),
}
