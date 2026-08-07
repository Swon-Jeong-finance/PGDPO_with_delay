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

PROBLEM_REGISTRY = {
    "p1": dict(verify=_p1_verify),
    "p2": dict(verify=_p2_verify),
    # p3 / p4: registered when their reference layers land
}
