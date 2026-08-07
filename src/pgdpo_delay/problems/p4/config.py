"""Configuration contract for optional Problem 4 (optimal execution).

Canonical paper/pilot numbers live only in the packaged resource
``pgdpo_delay/configs/p4/main.yaml``.  As for Problems 1--3, an explicit
filesystem path may be loaded for a derived calibration, while a differing
CWD file is not allowed to shadow a packaged canonical name.

The main instance is deliberately narrow: signed controls, independent fill
and signal noises, pre-trade impact over *realized* fills, and no bounds.  The
checks below make those modelling choices executable rather than comments.
"""

from importlib.resources import files
from pathlib import Path

import numpy as np
import yaml


def _read(variant: str) -> dict:
    if "/" in variant or variant.endswith(".yaml"):
        path = Path(variant)
        if not path.exists():
            raise KeyError(f"no config file {path}")
        return yaml.safe_load(path.read_text())

    local = Path.cwd() / "configs" / "p4" / f"{variant}.yaml"
    resource = files("pgdpo_delay.configs").joinpath("p4", f"{variant}.yaml")
    if resource.is_file():
        text = resource.read_text()
        if local.exists() and local.read_text() != text:
            raise RuntimeError(
                f"{local} shadows the canonical packaged config with "
                "different content; canonical names load only from the "
                "package. Derive under a NEW name instead."
            )
        return yaml.safe_load(text)
    if local.exists():
        return yaml.safe_load(local.read_text())
    raise KeyError(f"unknown P4 config: {variant}")


def load_config(variant: str = "main") -> dict:
    """Load and validate the signed-trading, independent-noise P4 contract."""
    raw = _read(variant)

    required_problem = (
        "q0",
        "sigma_Q",
        "kappa_alpha",
        "sigma_alpha",
        "gamma",
        "rho_G",
        "phi",
        "eta",
        "kappa",
    )
    problem = raw["problem"]
    for key in required_problem:
        if key not in problem:
            raise KeyError(f"problem.{key} missing")
        if not np.isfinite(problem[key]):
            raise ValueError(f"problem.{key} must be finite")

    strictly_positive = required_problem
    if any(float(problem[key]) <= 0.0 for key in strictly_positive):
        raise ValueError(
            "q0, sigma_Q, kappa_alpha, sigma_alpha, gamma, rho_G, phi, "
            "eta, and kappa must all be positive"
        )

    grid = raw["grid"]
    T = float(grid["T"])
    h = float(grid["dt"])
    delta = float(grid["delta"])
    if min(T, h, delta) <= 0.0:
        raise ValueError("grid.T, grid.dt, and grid.delta must be positive")
    if delta > T:
        raise ValueError("grid.delta must not exceed grid.T")
    N = round(T / h)
    H = round(delta / h)
    if not (np.isclose(N * h, T) and np.isclose(H * h, delta)):
        raise ValueError("grid.dt must divide both grid.T and grid.delta")
    if N < 1 or H < 1:
        raise ValueError("the time and memory grids must each have at least one step")

    control = raw["control"]
    if control.get("kind") != "signed":
        raise ValueError("P4 main control.kind must be 'signed'")
    if "lower" in control or "upper" in control:
        raise ValueError("signed P4 main control must not carry box bounds")

    model = raw["model"]
    if model.get("state_discretization") != "euler":
        raise ValueError("P4 same-grid reference requires Euler state steps")
    if model.get("impact_kernel") != "exponential":
        raise ValueError("P4 main requires the exponential impact kernel")
    if model.get("impact_timing") != "pre_trade":
        raise ValueError("P4 requires pre-trade impact (the current u is excluded)")
    if model.get("impact_source") != "realized_fills":
        raise ValueError("P4 impact must be computed from realized fills")

    noise = raw["noise"]
    if float(noise.get("correlation", np.nan)) != 0.0:
        raise ValueError("P4 main requires independent W_Q and W_alpha")
    channels = tuple(noise.get("channels", ()))
    if channels != ("W_Q", "W_alpha"):
        raise ValueError("noise.channels must be exactly [W_Q, W_alpha]")

    init = raw["init"]
    if float(init["q0"]) != float(problem["q0"]):
        raise ValueError("init.q0 and problem.q0 must agree")
    alpha0 = init["alpha0"]
    alpha_law = alpha0.get("law")
    if alpha_law == "gaussian":
        alpha_mean = float(alpha0["mean"])
        alpha_std = float(alpha0["std"])
        if not np.isfinite(alpha_mean) or not np.isfinite(alpha_std):
            raise ValueError("Gaussian alpha0 mean/std must be finite")
        if alpha_std <= 0.0:
            raise ValueError("Gaussian alpha0 std must be positive")
    elif alpha_law == "deterministic":
        alpha_mean = float(alpha0["value"])
        alpha_std = 0.0
        if not np.isfinite(alpha_mean):
            raise ValueError("deterministic alpha0 value must be finite")
    else:
        raise ValueError(
            "init.alpha0.law must explicitly be 'gaussian' or 'deterministic'"
        )

    status = raw.get("calibration_status")
    if status not in ("provisional_pilot", "frozen"):
        raise ValueError(
            "calibration_status must explicitly be provisional_pilot or frozen"
        )

    params = {key: float(problem[key]) for key in required_problem}
    return dict(
        variant=Path(variant).stem,
        calibration_status=status,
        params=params,
        T=T,
        h=h,
        dt=h,
        delta=delta,
        N=N,
        H=H,
        state_dim=H + 2,
        bounds=None,
        control_kind="signed",
        init=dict(q0=params["q0"], alpha0_mean=alpha_mean,
                  alpha0_std=alpha_std, alpha0_law=alpha_law),
        noise=dict(correlation=0.0, channels=channels),
        raw=raw,
    )
