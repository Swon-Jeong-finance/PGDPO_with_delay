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
        if local.exists():
            raw = yaml.safe_load(local.read_text())
        else:
            res = files("pgdpo_delay.configs").joinpath("p2", f"{name}.yaml")
            if not res.is_file(): raise KeyError(f"unknown P2 config: {name}")
            raw = yaml.safe_load(res.read_text())
    if raw["grid"]["T"] <= 0 or raw["grid"]["dt"] <= 0:
        raise ValueError("grid.T and grid.dt must be positive")
    if int(raw["r_main"]) < 1: raise ValueError("r_main must be >= 1")
    if raw["kernel"]["rho_delta"] <= 0: raise ValueError("kernel.rho_delta must be positive")
    return raw
