"""Regression tests for the float32 active-set tolerance bug
(review 2026-08-07 sec.4.3): float32-stored bound actions were classified as
interior under the old 1e-9 test, zeroing upper occupancy (0% vs 29.4%) and
corrupting switching statistics. Torch policies emit float32, so this guards
the H=16 headline metrics, not just the DP audit."""
import numpy as np
from pgdpo_delay.problems.p1.config import load_config
from pgdpo_delay.problems.p1 import evaluate


def test_float32_bound_roundtrip_documents_the_bug():
    # the premise: float32(hi) < hi - 1e-9, so the OLD test misclassifies
    lo, hi = load_config("dp_small")["bounds"]
    u32 = float(np.float32(hi))
    assert u32 < hi - 1e-9          # old tolerance fails on this action
    assert u32 >= hi - evaluate.active_tol(lo, hi)   # new tolerance catches it
    l32 = float(np.float32(lo))
    assert l32 <= lo + evaluate.active_tol(lo, hi)


def test_active_tol_does_not_swallow_interior_actions():
    lo, hi = load_config("dp_small")["bounds"]
    atol = evaluate.active_tol(lo, hi)
    assert atol < 1e-5              # tiny vs box width 1.181
    mid = 0.5*(lo + hi)
    assert not (mid <= lo + atol or mid >= hi - atol)


def test_kkt_residual_float32_upper_bound_action():
    cfg = load_config("dp_small")
    lo, hi = cfg["bounds"]
    # minimisation convention (registry sign gate): at the UPPER bound the
    # residual is zero iff g <= 0 (objective still decreasing at the bound).
    # p = -1 makes g = R*u + b*p < 0; under the old 1e-9 tol this float32
    # action fell into the interior branch and returned |g| > 0 instead.
    inp = dict(u=float(np.float32(hi)), p=-1.0, zeta=0.0, Pi=0.0, sigma_bar=0.0)
    assert evaluate.kkt_residual(cfg, inp) == 0.0
    assert evaluate.kkt_residual(cfg, inp, tol=1e-9) > 0.0   # old-tol bug demo
    # interior action keeps |g|
    inp2 = dict(u=0.0, p=1.0, zeta=0.0, Pi=0.0, sigma_bar=0.0)
    assert evaluate.kkt_residual(cfg, inp2) == abs(cfg["params"]["b"])


def test_active_set_stats_float32_constant_upper_policy():
    cfg = load_config("dp_small")
    hi = cfg["bounds"][1]
    pol = lambda k, Z: np.full(len(np.atleast_2d(Z)), np.float32(hi), dtype=np.float32)
    st = evaluate.active_set_stats(cfg, pol, Np=32, seed=0)
    assert st["occ"][2] == 1.0      # upper occupancy, was 0.0 under 1e-9
    assert st["occ"][1] == 0.0


def test_regime_disagreement_float32_vs_float64_bound():
    cfg = load_config("dp_small")
    hi = cfg["bounds"][1]
    pol64 = lambda k, Z: np.full(len(np.atleast_2d(Z)), hi)
    pol32 = lambda k, Z: np.full(len(np.atleast_2d(Z)), np.float32(hi), dtype=np.float32)
    states = [(0, np.zeros(cfg["H"]+1)), (1, np.ones(cfg["H"]+1))]
    assert evaluate.regime_disagreement(cfg, pol64, pol32, states) == 0.0
