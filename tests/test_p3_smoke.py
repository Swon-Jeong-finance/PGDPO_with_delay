"""P3-R smoke tests: config validation, exact quadratic minimiser, tiny-grid
HJB consistency with the shared simulator."""
import numpy as np
import pytest
from pgdpo_delay.problems.p3.config import load_config
from pgdpo_delay.problems.p3 import oracle, calibrate, dynamics


def test_p3_config_loads_and_validates():
    cfg = load_config("renewal")
    p = cfg["params"]
    assert 0.0 < p["eta_sigma"] < 1.0 and p["R"] > 0
    assert cfg["bounds"] == (0.0, 1.0)
    assert np.isclose(cfg["N"]*cfg["dt"], cfg["T"])
    assert cfg["hjb"]["I_max"] > p["Npop"]


def test_p3_eta_sigma_range_enforced(tmp_path):
    import yaml
    from importlib.resources import files
    raw = yaml.safe_load(
        files("pgdpo_delay.configs").joinpath("p3", "renewal.yaml").read_text())
    raw["problem"]["eta_sigma"] = 1.2
    f = tmp_path/"bad.yaml"; f.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(str(f))


def test_p3_pointwise_min_exact():
    cfg = load_config("renewal")
    assert calibrate.brute_force_check(cfg, n=500) < 1e-7


def test_p3_negative_curvature_endpoint_branch():
    # A2 <= 0 must return the better ENDPOINT, never an interior vertex
    cfg = load_config("renewal"); p = cfg["params"]
    I = np.array([1.0]); V_II = np.array([-100.0])
    for V_I, expect in ((np.array([5.0]), None), (np.array([-5.0]), None)):
        u = oracle.pointwise_min_u(p, I, V_I, V_II)
        assert u[0] in (0.0, 1.0)
        # exact objective comparison confirms the chosen endpoint is minimal
        s2I2 = (p["sigma0"]*I)**2
        A2 = p["R"] + (p["eta_sigma"]**2)*s2I2*V_II
        num = p["b"]*I*V_I + p["eta_sigma"]*s2I2*V_II
        f = lambda u_: 0.5*A2*u_*u_ - num*u_
        assert f(u)[0] <= min(f(np.zeros(1))[0], f(np.ones(1))[0]) + 1e-12


def test_p3_tiny_hjb_mc_consistency():
    cfg = load_config("renewal")
    hjb = oracle.solve_hjb(cfg, n_I=31, n_M=31)
    d = calibrate.rollout_diagnostics(cfg, hjb, Np=800, seed=5)
    assert abs(d["J_mc"] - d["V0_mean"]) <= 3*d["se"] + 0.02
    assert d["neg_frac_mean"] < 0.01                # positivity scheme sanity


def test_p3_curvature_diagnostics_include_t0_endpoint():
    """The final V(0) surface has no following substep, but it is part of a
    genuinely global stored-curvature diagnostic.  Use c_T=0 so its positive
    running-cost curvature is absent from the terminal surface and therefore
    detects an omitted t=0 endpoint."""
    base = load_config("renewal")
    cfg = dict(base, params=dict(base["params"], c_T=0.0),
               N=1, T=base["dt"])
    hjb = oracle.solve_hjb(cfg, n_I=7, n_M=7, store_value=True)
    assert hjb["n_sub"] == 1
    assert hjb["curvature_diagnostic_scope"] == (
        "all_solver_surfaces_including_t0")
    assert hjb["curvature_diagnostic_surface_count"] == 2

    dI = hjb["Ig"][1] - hjb["Ig"][0]
    _, _, terminal_dii = oracle._updiffs(hjb["Vs"][1].astype(float), dI)
    _, _, t0_dii = oracle._updiffs(hjb["V0"], dI)
    assert terminal_dii.max() == 0.0
    assert t0_dii.max() > 0.0
    np.testing.assert_allclose(
        hjb["V_II_range"],
        (min(float(terminal_dii.min()), float(t0_dii.min())),
         max(float(terminal_dii.max()), float(t0_dii.max()))),
        rtol=0.0, atol=1e-14)


def test_p3_bellman_residual_exact_terminal_and_pathwise_se():
    """The cumulative statistic must use the realized terminal quadratic and
    the SE of the pathwise sum, not an interpolated terminal table or a sum of
    per-step SEs."""
    base = load_config("renewal")
    cfg = dict(base, N=2, T=2*base["dt"])
    Ig = np.linspace(0.0, 1.5, 4); Mg = np.linspace(0.0, 1.5, 4)
    II, MM = Ig[:, None], Mg[None, :]
    V0 = 0.3*II + 0.1*MM
    V1 = 0.2*II - 0.05*MM
    # A deliberately impossible terminal table makes this a strong regression:
    # the Bellman implementation must bypass it at the final transition.
    VN_bad = np.full_like(V0, 1.0e6)
    pol = [np.full_like(V0, 0.25) for _ in range(cfg["N"])]
    hjb = dict(V0=V0, Vs=[V0, V1, VN_bad], pol=pol, Ig=Ig, Mg=Mg)

    Np, seed = 64, 37
    got = calibrate.bellman_residual(cfg, hjb, Np=Np, seed=seed)

    p, dt = cfg["params"], cfg["dt"]
    rng = np.random.default_rng(seed)
    I = rng.uniform(*cfg["init"]["I0"], Np)
    M = rng.uniform(*cfg["init"]["M0"], Np)
    V_initial = calibrate._readout_k(hjb, 0, I, M)
    policy = oracle.hjb_policy(hjb)
    realized_cost = np.zeros(Np)
    for k in range(cfg["N"]):
        u = policy(k, I, M)
        realized_cost += dt*(0.5*p["c_I"]*np.maximum(I, 0.0)**2
                             + 0.5*p["R"]*u**2)
        I, M = dynamics.step(
            p, dt, I, M, u, rng.normal(0.0, np.sqrt(dt), Np))
    terminal_exact = 0.5*p["c_T"]*np.maximum(I, 0.0)**2
    pathwise_total = V_initial - realized_cost - terminal_exact

    np.testing.assert_allclose(got["total"], pathwise_total.mean(),
                               rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        got["total_se"], pathwise_total.std(ddof=1)/np.sqrt(Np),
        rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(got["mean"].sum(), got["total"],
                               rtol=0.0, atol=1e-13)
    assert np.max(np.abs(calibrate._readout_k(hjb, cfg["N"], I, M)
                         - terminal_exact)) > 1.0


def test_p3_simulator_paired_self_zero():
    cfg = load_config("renewal")
    pol = lambda k, I, M: np.full_like(np.atleast_1d(I), 0.5)
    r = dynamics.simulate_paired(cfg, pol, pol, Np=64, seed=2)
    assert r["delta_A_minus_B"] == 0.0              # CRN self-comparison exact


def test_p3d_config_and_kernel():
    cfg = load_config("distributed")
    assert cfg["kind"] == "distributed" and cfg["dist"]["H"] == 25
    w = dynamics.kernel_weights(cfg)
    assert np.isclose(w.sum(), 1.0) and (w >= 0).all()
    assert 0 < w.argmax() < len(w) - 1              # interior incubation peak
    assert w[0] == 0.0                              # m_K > 1 -> strictly past


def test_p3d_shared_params_match_renewal():
    """P3-R and P3-D share the primitive epidemic/cost/control coefficients.
    This is a configuration-correctness check only: their memory dynamics and
    initial-history laws differ, and no direct P3-R/P3-D numerical comparison
    is part of the retained experiment scope."""
    r, d = load_config("renewal"), load_config("distributed")
    for k in ("beta", "gamma", "b", "sigma0", "eta_sigma", "Npop",
              "c_I", "R", "c_T"):
        assert r["params"][k] == d["params"][k]


def test_p3_nmpc_gradients_match_fd():
    from pgdpo_delay.problems.p3 import nmpc
    cfgR, cfgD = load_config("renewal"), load_config("distributed")
    w = dynamics.kernel_weights(cfgD)
    rng = np.random.default_rng(1)
    u = rng.uniform(0.1, 0.9, 8)

    def fd_gap(fun):
        _, g = fun(u)
        gfd = np.empty_like(u)
        for i in range(len(u)):
            up = u.copy(); up[i] += 1e-6; um = u.copy(); um[i] -= 1e-6
            gfd[i] = (fun(up)[0] - fun(um)[0])/2e-6
        return np.max(np.abs(g - gfd))/np.max(np.abs(gfd))

    assert fd_gap(lambda uu: nmpc.rollout_cost_grad_renewal(
        cfgR, 0.3, 0.4, uu)) < 1e-5
    B0 = rng.uniform(0.05, 0.7, cfgD["dist"]["H"]+1)
    assert fd_gap(lambda uu: nmpc.rollout_cost_grad_dist(
        cfgD, B0, w, uu)) < 1e-5


def test_p3_projected_box_kkt_sign_convention():
    from pgdpo_delay.problems.p3 import nmpc
    # Minimisation KKT: g >= 0 at the lower bound and g <= 0 at the upper.
    u = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    g = np.array([2.0, -2.0, 0.3, -2.0, 2.0])
    out = nmpc.projected_box_kkt(u, g)
    np.testing.assert_allclose(out["vector"], [0.0, -1.0, 0.3, 0.0, 1.0])
    assert out["inf"] == 1.0 and out["first"] == 0.0
    assert nmpc.projected_box_kkt(
        np.array([0.0, 0.4, 1.0]), np.array([1.0, 0.0, -1.0]))["inf"] == 0.0


def test_p3d_nmpc_solution_runtime_and_terminal_proxy_contract():
    from pgdpo_delay.problems.p3 import nmpc
    cfg = load_config("distributed")
    assert cfg["nmpc"]["terminal_mode"] == "quadratic_proxy"

    # On a window ending before physical T, the same quadratic is explicitly
    # an MPC terminal surrogate.  Reconstruct that objective independently.
    w = dynamics.kernel_weights(cfg)
    B0 = np.linspace(0.25, 0.45, cfg["dist"]["H"]+1)
    u = np.array([0.2, 0.4, 0.6])
    objective, _ = nmpc.rollout_cost_grad_dist(cfg, B0, w, u)
    assert len(u) < cfg["N"]
    p, dt = cfg["params"], cfg["dt"]
    B = B0[None, :].copy(); expected = 0.0
    for uj in u:
        expected += dt*(0.5*p["c_I"]*max(B[0, 0], 0.0)**2
                        + 0.5*p["R"]*uj**2)
        B = dynamics.step_dist(p, dt, B, w, np.array([uj]), np.zeros(1))
    expected += 0.5*p["c_T"]*max(B[0, 0], 0.0)**2
    np.testing.assert_allclose(objective, expected, rtol=0.0, atol=1e-14)

    ctrl = nmpc.NMPCController(cfg, "distributed", lookahead=5, max_iter=20)
    rec = ctrl.solve(0, np.full(cfg["dist"]["H"]+1, 0.3))
    assert rec.horizon == 5 and rec.success
    assert rec.optimizer_runtime_s >= 0.0
    assert rec.decision_runtime_s >= rec.optimizer_runtime_s
    assert rec.total_objective_grad_evals == rec.optimizer_nfev + 1
    assert np.isfinite(rec.plan_raw).all() and rec.kkt_inf <= 1e-5
    assert rec.action_deployed == np.clip(rec.action_raw, 0.0, 1.0)
    assert rec.raw_plan_bound_violation <= 1e-10
    assert rec.raw_first_bound_violation <= 1e-10
    assert rec.deployment_clip_correction <= 1e-10
    assert rec.deployed_bound_violation <= 1e-12
    stats = ctrl.budget_stats()
    assert stats["n_solves"] == 1 and stats["terminal_mode"] == "quadratic_proxy"
    assert stats["optimizer_runtime_total_s"] == rec.optimizer_runtime_s
    assert stats["decision_runtime_total_s"] == rec.decision_runtime_s
    assert stats["optimizer_nfev_sum"] == rec.optimizer_nfev
    assert (stats["total_objective_grad_evals_sum"]
            == rec.total_objective_grad_evals)
    assert stats["ce_subproblem_kkt_inf_max"] == rec.kkt_inf
    assert stats["raw_plan_bound_violation_max"] == rec.raw_plan_bound_violation


def test_p3d_nmpc_keeps_raw_and_deployed_constraint_metrics(monkeypatch):
    """A safety clip must not erase evidence of an infeasible raw plan."""
    from types import SimpleNamespace
    from pgdpo_delay.problems.p3 import nmpc
    cfg = load_config("distributed")

    def fake_minimize(fun, u0, **kwargs):
        plan = np.array([-0.2, 1.3])
        value, grad = fun(plan)
        return SimpleNamespace(x=plan, fun=value, jac=grad, success=True,
                               status=0, nit=0, nfev=1)

    monkeypatch.setattr(nmpc, "minimize", fake_minimize)
    ctrl = nmpc.NMPCController(cfg, "distributed", lookahead=2, max_iter=2)
    rec = ctrl.solve(0, np.full(cfg["dist"]["H"]+1, 0.3))
    assert rec.action_raw == -0.2 and rec.action_deployed == 0.0
    assert rec.raw_plan_bound_violation == pytest.approx(0.3)
    assert rec.raw_first_bound_violation == pytest.approx(0.2)
    assert rec.deployment_clip_correction == pytest.approx(0.2)
    assert rec.deployed_bound_violation == 0.0
    stats = ctrl.budget_stats()
    assert stats["raw_plan_bound_violation_max"] == pytest.approx(0.3)
    assert stats["deployment_clip_correction_max"] == pytest.approx(0.2)


def test_p3d_nmpc_separates_optimizer_and_full_decision_accounting(monkeypatch):
    """The paper-facing timer and evaluation count include the one final
    objective/gradient recomputation used for KKT and deployment diagnostics."""
    from types import SimpleNamespace
    from pgdpo_delay.problems.p3 import nmpc
    cfg = load_config("distributed")
    clock = iter([10.0, 12.0, 15.0])
    monkeypatch.setattr(nmpc.time, "perf_counter", lambda: next(clock))

    plan = np.array([0.2, 0.4])
    def fake_minimize(fun, u0, **kwargs):
        value, grad = fun(plan)
        return SimpleNamespace(x=plan, fun=value, jac=grad, success=True,
                               status=0, nit=3, nfev=7)

    monkeypatch.setattr(nmpc, "minimize", fake_minimize)
    ctrl = nmpc.NMPCController(cfg, "distributed", lookahead=2, max_iter=4)
    B0 = np.full(cfg["dist"]["H"]+1, 0.3)
    rec = ctrl.solve(0, B0)
    expected_objective, expected_grad = nmpc.rollout_cost_grad_dist(
        cfg, B0, dynamics.kernel_weights(cfg), plan)
    expected_kkt = nmpc.projected_box_kkt(plan, expected_grad)["inf"]

    assert rec.optimizer_runtime_s == 2.0
    assert rec.decision_runtime_s == 5.0
    assert rec.optimizer_nfev == 7
    assert rec.total_objective_grad_evals == 8
    assert rec.objective == expected_objective
    assert rec.kkt_inf == expected_kkt
    stats = ctrl.budget_stats()
    assert stats["optimizer_runtime_total_s"] == 2.0
    assert stats["decision_runtime_total_s"] == 5.0
    assert stats["optimizer_nfev_sum"] == 7
    assert stats["optimizer_nfev_mean"] == 7
    assert stats["total_objective_grad_evals_sum"] == 8
    assert stats["total_objective_grad_evals_mean"] == 8


def test_p3d_paired_self_zero_and_nmpc_smoke():
    from pgdpo_delay.problems.p3 import nmpc
    cfg = load_config("distributed")
    pol = lambda k, B: np.full(B.shape[0], 0.5)
    r = dynamics.simulate_dist_paired(cfg, pol, pol, Np=32, seed=2)
    assert r["delta_A_minus_B"] == 0.0
    # tiny NMPC smoke: finite, in-bounds first actions
    ctrl = nmpc.NMPCController(cfg, "distributed", lookahead=5, max_iter=5)
    B0 = np.full(cfg["dist"]["H"]+1, 0.3)
    a = ctrl.action(0, B0)
    assert np.isfinite(a) and 0.0 <= a <= 1.0


def test_p3_discrete_upwind_minimizer_matches_dense_grid():
    """R1 (review sec.3.2): the branch-aware discrete minimiser must attain
    the dense-grid minimum of the FULL upwind objective, including negative-
    curvature and drift-sign-switch cases."""
    cfg = load_config("renewal"); p = cfg["params"]
    rng = np.random.default_rng(7); n = 500
    I = rng.uniform(0.0, cfg["hjb"]["I_max"], n)
    base = rng.normal(0.0, 2.0, n)
    Dp = rng.normal(0.0, 3.0, n); Dm = rng.normal(0.0, 3.0, n)
    DII = rng.normal(0.0, 30.0, n)          # both curvature signs occur
    u = oracle.discrete_min_u(p, I, base, Dp, Dm, DII)
    assert np.isfinite(u).all()                    # feasibility FIRST
    assert (u >= 0.0).all() and (u <= 1.0).all()   # (re-review sec.4.4)
    ug = np.linspace(0.0, 1.0, 4001)
    fs = oracle.upwind_control_objective(
        p, I[:, None], base[:, None], Dp[:, None], Dm[:, None],
        DII[:, None], ug[None, :])
    fx = oracle.upwind_control_objective(p, I, base, Dp, Dm, DII, u)
    assert float(np.max(fx - fs.min(axis=1))) < 1e-7
    # deterministic branch fixtures: usw<lo / interior / usw>hi / I=0,
    # each at both curvature signs
    for Iv, bv in [(0.5, -1.0), (0.5, 0.15), (0.5, 1.0), (0.0, 0.3),
                   (0.0, -0.3)]:
        for DIIv in (25.0, -25.0):
            uu = oracle.discrete_min_u(p, np.array([Iv]), np.array([bv]),
                                       np.array([1.5]), np.array([-2.0]),
                                       np.array([DIIv]))
            assert np.isfinite(uu[0]) and 0.0 <= uu[0] <= 1.0
            fd = oracle.upwind_control_objective(
                p, np.array([Iv]), np.array([bv]), np.array([1.5]),
                np.array([-2.0]), np.array([DIIv]), ug).min()
            fu = oracle.upwind_control_objective(
                p, np.array([Iv]), np.array([bv]), np.array([1.5]),
                np.array([-2.0]), np.array([DIIv]), uu)[0]
            assert fu <= fd + 1e-12


def test_p3_kind_field_consistency(tmp_path):
    import yaml
    from importlib.resources import files
    raw = yaml.safe_load(
        files("pgdpo_delay.configs").joinpath("p3", "renewal.yaml").read_text())
    assert raw["kind"] == "renewal"
    raw["kind"] = "distributed"             # contradicts inferred variant
    f = tmp_path/"bad_kind.yaml"; f.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(str(f))
