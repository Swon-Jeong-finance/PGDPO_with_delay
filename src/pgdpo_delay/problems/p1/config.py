"""P1 configuration. Canonical paper numbers live ONLY in the packaged
resources pgdpo_delay/configs/p1/*.yaml (wheel-safe via importlib.resources).
Resolution order for load_config(name):
  1) explicit filesystem path (contains '/' or endswith .yaml)
  2) user-derived ./configs/p1/<name>.yaml in the current working directory
     (written by `main.py config`; overrides never live inside the package)
  3) packaged canonical resource.
Validation raises explicit exceptions (never bare assert: `python -O` strips
asserts) and there are NO fallback defaults for paper parameters.
"""
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import numpy as np
import yaml

@dataclass(frozen=True)
class P1Problem:
    a: float; a_delay: float; b: float; sigma0: float
    c_x: float; c_y: float; gamma_u: float
    Q: float; R: float; Q_T: float
    def __post_init__(self):
        if self.R <= 0: raise ValueError("R must be positive")
        if self.Q < 0 or self.Q_T < 0: raise ValueError("Q, Q_T must be nonnegative")

def _read(variant: str) -> dict:
    if "/" in variant or variant.endswith(".yaml"):
        path = Path(variant)
        if not path.exists(): raise KeyError(f"no config file {path}")
        return yaml.safe_load(path.read_text())
    local = Path.cwd() / "configs" / "p1" / f"{variant}.yaml"
    if local.exists():
        return yaml.safe_load(local.read_text())
    res = files("pgdpo_delay.configs").joinpath("p1", f"{variant}.yaml")
    if not res.is_file(): raise KeyError(f"unknown P1 config: {variant}")
    return yaml.safe_load(res.read_text())

def load_config(variant: str) -> dict:
    raw = _read(variant)
    pb = P1Problem(**raw["problem"])                 # missing key -> TypeError
    T, delta, taps = raw["grid"]["T"], raw["grid"]["delta"], int(raw["grid"]["taps"])
    if T <= 0 or delta <= 0: raise ValueError("T, delta must be positive")
    if delta > T: raise ValueError("delta must not exceed T")
    if taps < 1: raise ValueError("taps must be >= 1")
    h = delta / taps
    N, H = round(T/h), taps
    if not (np.isclose(N*h, T) and np.isclose(H*h, delta)):
        raise ValueError("grid does not satisfy N*dt=T and H*dt=delta")
    lo, hi = raw["control"]["lower"], raw["control"]["upper"]
    if not lo < hi: raise ValueError("control.lower must be < control.upper")
    bud = raw.get("budgets", {})
    for k, v in bud.items():
        if int(v) <= 0: raise ValueError(f"budgets.{k} must be positive")
    if "dp" in raw:
        dp = raw["dp"]
        for k in ("n_x", "n_gh", "n_u"):
            if int(dp[k]) < 2: raise ValueError(f"dp.{k} must be >= 2")
        if dp["L"] <= 0: raise ValueError("dp.L must be positive")
    tt = np.linspace(-delta, 0, H+1)[::-1]
    amp, xtar = raw["tracking"]["ref_amplitude"], raw["tracking"]["x_target"]
    xref = amp*np.sin(2*np.pi*np.arange(N)*h/T)
    params = dict(a=pb.a, ad=pb.a_delay, b=pb.b, s0=pb.sigma0, cx=pb.c_x,
                  cy=pb.c_y, gu=pb.gamma_u, Q=pb.Q, R=pb.R, QT=pb.Q_T)
    cfg = dict(variant=Path(variant).stem, params=params, T=T, delta=delta, h=h,
               N=N, H=H, tt=tt, xref=xref, xtar=xtar, bounds=(lo, hi),
               budgets=bud, raw=raw)
    if "dp" in raw: cfg["dp"] = raw["dp"]
    return cfg
