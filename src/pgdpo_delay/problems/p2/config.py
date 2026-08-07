"""P2 configuration loader: pgdpo_delay/configs/p2/scaling.yaml is the single
source for the frozen scaling calibration (variant coefficients, r_main,
kernel rule, sweeps, budgets, seeds). Explicit exceptions, no fallbacks."""
from importlib.resources import files
from pathlib import Path
import yaml

def load_p2_config(name: str = "scaling") -> dict:
    if "/" in name or name.endswith(".yaml"):
        path = Path(name)
        if not path.exists(): raise KeyError(f"no config file {path}")
        raw = yaml.safe_load(path.read_text())
    else:
        local = Path.cwd() / "configs" / "p2" / f"{name}.yaml"
        res = files("pgdpo_delay.configs").joinpath("p2", f"{name}.yaml")
        if res.is_file():
            # canonical names: package only; differing CWD shadow refused
            # (review 2026-08-07 sec.4.9, same rule as p1/config.py)
            text = res.read_text()
            if local.exists() and local.read_text() != text:
                raise RuntimeError(
                    f"{local} shadows the canonical packaged config with "
                    f"different content; derive under a NEW name instead.")
            raw = yaml.safe_load(text)
        elif local.exists():
            raw = yaml.safe_load(local.read_text())
        else:
            raise KeyError(f"unknown P2 config: {name}")
    if raw["grid"]["T"] <= 0 or raw["grid"]["dt"] <= 0:
        raise ValueError("grid.T and grid.dt must be positive")
    if int(raw["r_main"]) < 1: raise ValueError("r_main must be >= 1")
    if raw["kernel"]["rho_delta"] <= 0: raise ValueError("kernel.rho_delta must be positive")
    return raw
