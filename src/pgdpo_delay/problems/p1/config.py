"""P1 configuration. Canonical paper numbers live ONLY in the packaged
resources pgdpo_delay/configs/p1/*.yaml (wheel-safe via importlib.resources).
Resolution order for load_config(name) (review 2026-08-07 minfix sec.4.5):
  1) explicit filesystem path (contains '/' or endswith .yaml)
  2) packaged canonical resource; a CWD copy of a canonical name with
     DIFFERING content is refused (identical copy tolerated)
  3) user-derived ./configs/p1/<name>.yaml only when no packaged name exists
     (written by `main.py config`; overrides never live inside the package).
Validation raises explicit exceptions (never bare assert: `python -O` strips
asserts) and there are NO fallback defaults for paper parameters.
"""
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import numpy as np
import yaml

from ...core.artifacts import config_hash


P1_INITIAL_LAW_API = "p1.make_hist-v1"

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
    res = files("pgdpo_delay.configs").joinpath("p1", f"{variant}.yaml")
    if res.is_file():
        # Canonical names load ONLY from the package (review 2026-08-07
        # sec.4.9). A CWD copy with identical content is tolerated; a
        # differing copy silently rewriting paper numbers is refused.
        text = res.read_text()
        if local.exists() and local.read_text() != text:
            raise RuntimeError(
                f"{local} shadows the canonical packaged config with different "
                f"content; canonical names load only from the package. Derive "
                f"under a NEW name instead (main.py config --name <new>).")
        return yaml.safe_load(text)
    if local.exists():                       # user-derived (non-canonical) name
        return yaml.safe_load(local.read_text())
    raise KeyError(f"unknown P1 config: {variant}")

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
    kind = raw["control"].get("kind", "box")     # H1 (Stage-I review):
    # explicit P1-U/P1-C separation; unknown kinds are refused so a silent
    # wrong-chart training run is impossible.
    if kind == "box":
        lo, hi = raw["control"]["lower"], raw["control"]["upper"]
        if not lo < hi:
            raise ValueError("control.lower must be < control.upper")
        bounds = (lo, hi)
    elif kind == "unconstrained":
        if "lower" in raw["control"] or "upper" in raw["control"]:
            raise ValueError("unconstrained control must not carry bounds")
        bounds = None
    else:
        raise ValueError(f"unknown control.kind: {kind}")
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
               N=N, H=H, tt=tt, xref=xref, xtar=xtar, bounds=bounds,
               control_kind=kind, budgets=bud, raw=raw)
    if "dp" in raw: cfg["dp"] = raw["dp"]
    return cfg


def scientific_config_snapshot(cfg: dict) -> dict:
    """Common P1 identity shared by learned methods and references.

    Optimizer/protocol settings have their own method ``config_hash``.  This
    smaller snapshot binds the dynamics, mesh, control set, and evaluation
    initial-history law so a comparison artifact cannot mix different P1
    problems merely because both are labelled ``p1_u`` or ``p1_c``.
    """

    required = {
        "raw", "variant", "T", "delta", "h", "N", "H", "control_kind",
        "bounds",
    }
    missing = sorted(required.difference(cfg))
    if missing:
        raise ValueError(f"P1 config lacks scientific identity fields: {missing}")
    raw = cfg["raw"]
    scientific_sections = ("problem", "grid", "tracking", "control")
    missing_raw = [name for name in scientific_sections if name not in raw]
    if missing_raw:
        raise ValueError(
            "P1 raw config lacks scientific sections: " + ", ".join(missing_raw)
        )
    # Deliberately do not hash numerical-estimator budgets, DP audit settings,
    # optimizers, devices, or training/evaluation protocols here.  Those belong
    # to the method/run identity.  This hash denotes only the controlled problem
    # and its initial-history law, so otherwise identical methods remain
    # comparable when their computational budgets differ.
    return {
        "schema": 2,
        "problem": "p1",
        "problem_parameters": raw["problem"],
        "grid": raw["grid"],
        "tracking": raw["tracking"],
        "control": raw["control"],
        "derived": {
            "variant": cfg["variant"],
            "T": cfg["T"],
            "delta": cfg["delta"],
            "h": cfg["h"],
            "N": cfg["N"],
            "H": cfg["H"],
            "control_kind": cfg["control_kind"],
            "bounds": cfg["bounds"],
        },
        "initial_law_api": P1_INITIAL_LAW_API,
    }


def scientific_config_hash(cfg: dict) -> str:
    """Stable hash used by the P1-wide output contract."""

    return config_hash(scientific_config_snapshot(cfg))
