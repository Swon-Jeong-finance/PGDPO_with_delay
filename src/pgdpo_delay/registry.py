"""Problem registry: main.py dispatches ONLY through this table.
Each problem exposes verify(fast) now; run/evaluate arrive with the solver
layer (core.stage1/stage2) and must follow the single-Stage-II contract."""
import runpy

def _p1_verify(fast=True):
    from .problems.p1 import oracle, h_refine
    oracle.run_checks()
    rows = h_refine.run(hs=(0.05, 0.025) if fast else (0.1, 1/15, 0.05, 0.025, 0.0125, 0.00625),
                        save=not fast)
    fl = [r["floor_rmse"] for r in rows]
    assert all(b < a for a, b in zip(fl, fl[1:])), "h-refinement floor not decreasing"
    if not fast:
        runpy.run_module("pgdpo_delay.problems.p1.contract", run_name="__main__")

def _p2_verify(fast=True):
    from .problems.p2 import eigencheck
    eigencheck.run_check()
    if not fast:
        runpy.run_module("pgdpo_delay.problems.p2.scaling", run_name="__main__")

PROBLEM_REGISTRY = {
    "p1": dict(verify=_p1_verify),
    "p2": dict(verify=_p2_verify),
    # p3 / p4: registered when their reference layers land
}
