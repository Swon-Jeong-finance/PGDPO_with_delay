"""P3 configuration. Canonical paper numbers live ONLY in the packaged
resources pgdpo_delay/configs/p3/*.yaml. Same resolution rule as p1/p2
(review 2026-08-07 sec.4.9): canonical names load only from the package and a
differing CWD shadow is refused; user-derived names resolve in CWD configs/.
Explicit exceptions, no fallback defaults for paper parameters."""
from importlib.resources import files
from pathlib import Path
import numpy as np
import yaml


def _read(variant: str) -> dict:
    if "/" in variant or variant.endswith(".yaml"):
        path = Path(variant)
        if not path.exists(): raise KeyError(f"no config file {path}")
        return yaml.safe_load(path.read_text())
    local = Path.cwd() / "configs" / "p3" / f"{variant}.yaml"
    res = files("pgdpo_delay.configs").joinpath("p3", f"{variant}.yaml")
    if res.is_file():
        text = res.read_text()
        if local.exists() and local.read_text() != text:
            raise RuntimeError(
                f"{local} shadows the canonical packaged config with different "
                f"content; canonical names load only from the package. Derive "
                f"under a NEW name instead (main.py config --name <new>).")
        return yaml.safe_load(text)
    if local.exists():
        return yaml.safe_load(local.read_text())
    raise KeyError(f"unknown P3 config: {variant}")


def load_config(variant: str = "renewal") -> dict:
    raw = _read(variant)
    p = raw["problem"]
    base_keys = ("beta", "gamma", "b", "sigma0", "eta_sigma", "Npop",
                 "c_I", "R", "c_T")
    for k in base_keys:
        if k not in p: raise KeyError(f"problem.{k} missing")
    if not (0.0 < p["eta_sigma"] < 1.0):
        raise ValueError("eta_sigma must lie in (0, 1)")
    if p["R"] <= 0: raise ValueError("R must be positive")
    if min(p["beta"], p["gamma"], p["b"], p["sigma0"], p["Npop"]) <= 0:
        raise ValueError("beta, gamma, b, sigma0, Npop must be positive")
    if p["c_I"] < 0 or p["c_T"] < 0: raise ValueError("c_I, c_T must be >= 0")
    T, dt = raw["grid"]["T"], raw["grid"]["dt_sim"]
    if T <= 0 or dt <= 0: raise ValueError("T, dt_sim must be positive")
    N = round(T/dt)
    if not np.isclose(N*dt, T): raise ValueError("dt_sim must divide T")
    lo, hi = raw["control"]["lower"], raw["control"]["upper"]
    if not (lo == 0.0 and hi == 1.0):
        raise ValueError("P3 control box is [0, 1] by design")
    kind = "distributed" if "dist" in raw else "renewal"
    if "kind" in raw and raw["kind"] != kind:
        raise ValueError(f"explicit kind '{raw['kind']}' contradicts the "
                         f"inferred variant '{kind}'")
    cfg = dict(variant=Path(variant).stem, kind=kind, params=dict(p), T=T,
               dt=dt, N=N, bounds=(lo, hi), raw=raw)
    nm = raw.get("nmpc")
    if nm is not None:
        required_nm = ("lookahead_steps", "max_iter", "gtol", "ftol",
                       "terminal_mode")
        for k in required_nm:
            if k not in nm: raise KeyError(f"nmpc.{k} missing")
        if not (1 <= int(nm["lookahead_steps"]) <= N):
            raise ValueError("nmpc.lookahead_steps must lie in [1, N]")
        if int(nm["max_iter"]) < 1:
            raise ValueError("nmpc.max_iter must be >= 1")
        if not (0 < float(nm["gtol"]) < 1 and 0 < float(nm["ftol"]) < 1):
            raise ValueError("nmpc gtol/ftol must lie in (0, 1)")
        if nm["terminal_mode"] != "quadratic_proxy":
            raise ValueError("P3 CE-NMPC terminal_mode must be quadratic_proxy")
        cfg["nmpc"] = dict(lookahead_steps=int(nm["lookahead_steps"]),
                           max_iter=int(nm["max_iter"]),
                           gtol=float(nm["gtol"]), ftol=float(nm["ftol"]),
                           terminal_mode=nm["terminal_mode"])
    if kind == "renewal":
        if "rho" not in p: raise KeyError("problem.rho missing (renewal)")
        if p["rho"] <= 0: raise ValueError("rho must be positive")
        if p["rho"]*dt > 1.0:
            raise ValueError("rho*dt_sim must be <= 1 (M positivity)")
        hj = raw["hjb"]
        if hj["I_max"] <= p["Npop"]:
            raise ValueError("I_max must exceed Npop (stochastic overshoot room)")
        for k in ("n_I", "n_M"):
            if int(hj[k]) < 5: raise ValueError(f"hjb.{k} must be >= 5")
        if not (0 < hj["cfl_safety"] <= 1):
            raise ValueError("cfl_safety in (0, 1]")
        i0, m0 = raw["init"]["I0"], raw["init"]["M0"]
        if not (0 <= i0[0] < i0[1] <= hj["I_max"]
                and 0 <= m0[0] < m0[1] <= hj["I_max"]):
            raise ValueError("init boxes must sit inside [0, I_max]")
        cfg["init"] = dict(I0=tuple(i0), M0=tuple(m0))
        cfg["hjb"] = dict(hj)
    else:
        d = raw["dist"]
        if d["delta"] <= 0: raise ValueError("dist.delta must be positive")
        H = round(d["delta"]/dt)
        if not np.isclose(H*dt, d["delta"]):
            raise ValueError("dt_sim must divide dist.delta")
        if d["m_K"] < 1.0: raise ValueError("dist.m_K must be >= 1")
        if d["theta"] <= 0: raise ValueError("dist.theta must be positive")
        i0, ip = raw["init"]["I0"], raw["init"]["Ipast"]
        if not (0 <= i0[0] < i0[1] and 0 <= ip[0] < ip[1]):
            raise ValueError("init ranges must be nondegenerate and >= 0")
        cfg["dist"] = dict(d, H=H)
        cfg["init"] = dict(I0=tuple(i0), Ipast=tuple(ip))
        if "nmpc" not in cfg:
            raise KeyError("distributed P3 requires an nmpc section")
    return cfg
