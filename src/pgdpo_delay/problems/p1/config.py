"""P1 configuration: YAML is the ONLY source of paper numbers (no fallback
defaults). Variants: "main" (taps=16 long-history) / "small" (taps=3 DP audit).
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[4] / "configs" / "p1"

@dataclass(frozen=True)
class P1Problem:
    a: float; a_delay: float; b: float; sigma0: float
    c_x: float; c_y: float; gamma_u: float
    Q: float; R: float; Q_T: float
    def __post_init__(self):
        assert self.R > 0 and self.Q >= 0 and self.Q_T >= 0

def load_config(variant: str) -> dict:
    raw = yaml.safe_load((CONFIG_DIR / f"{variant if variant != 'main' else 'main'}.yaml").read_text()) \
          if variant in ("main", "dp_small") else None
    if raw is None:
        raise KeyError(f"unknown P1 variant: {variant}")
    pb = P1Problem(**raw["problem"])                     # missing key -> TypeError
    T, delta, taps = raw["grid"]["T"], raw["grid"]["delta"], int(raw["grid"]["taps"])
    h = delta / taps
    N, H = round(T/h), taps
    assert np.isclose(N*h, T) and np.isclose(H*h, delta)
    tt = np.linspace(-delta, 0, H+1)[::-1]
    amp, xtar = raw["tracking"]["ref_amplitude"], raw["tracking"]["x_target"]
    xref = amp*np.sin(2*np.pi*np.arange(N)*h/T)
    params = dict(a=pb.a, ad=pb.a_delay, b=pb.b, s0=pb.sigma0, cx=pb.c_x,
                  cy=pb.c_y, gu=pb.gamma_u, Q=pb.Q, R=pb.R, QT=pb.Q_T)
    cfg = dict(variant=variant, params=params, T=T, delta=delta, h=h, N=N, H=H,
               tt=tt, xref=xref, xtar=xtar,
               bounds=(raw["control"]["lower"], raw["control"]["upper"]),
               budgets=raw.get("budgets", {}), raw=raw)
    if "dp" in raw: cfg["dp"] = raw["dp"]
    return cfg
