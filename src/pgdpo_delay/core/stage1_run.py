"""One-seed Stage-I worker and torch-free protocol/fingerprint API.

This module is the process boundary used by the multi-GPU scheduler.  It
intentionally runs exactly one training seed and never chooses a GPU or a
seed itself.  Orchestration can therefore launch isolated subprocesses and
recycle the first free GPU slot without sharing CUDA state between jobs.

The protocol/fingerprint functions do not import torch.  Torch and the
problem adapter are imported only after the worker has validated its target
directory and written ``status.json`` with ``RUNNING``.  A successful worker
writes ``COMPLETE`` to that file *last*, so a directory with checkpoints but
no COMPLETE status is never a publishable result.

Direct invocation (normally called by the scheduler)::

    python -m pgdpo_delay.core.stage1_run \
      --problem p1 --protocol p1_u --seed 1 --device cuda:0 \
      --outdir /path/to/fresh/staging --run-fingerprint <fingerprint>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import time
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .artifacts import atomic_write_json, config_hash, write_manifest


WORKER_API_VERSION = 4
PROTOCOL_SCHEMA = 2
STATUS_SCHEMA = 1
METRICS_SCHEMA_VERSION = 1
_METHOD = "stage1_lstm_dpo"
_DTYPES = {"float32", "float64"}
_TRAIN_KEYS = {
    "iters", "batch", "lr", "hidden", "num_layers", "clip_grad", "log_every",
    "val_every", "val_batch",
}
_EVAL_KEYS = {"Np", "seed", "batch_size"}
_P1U_REQUIRED_METRICS = (
    "control_nrmse",
    "dJ_paired",
    "dJ_se",
    "J_policy",
    "J_oracle_mc",
    "J_exact",
    "mc_anchor_gap",
    "mc_anchor_gap_se",
    "initial_train_loss",
    "final_train_loss",
    "best_iter",
    "clip_frac",
    "train_runtime_seconds",
    "evaluation_runtime_seconds",
    "total_runtime_seconds",
)


def _jsonable(value: Any) -> Any:
    """Return a canonical JSON-compatible copy (also handles numpy values)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _source_identity(problem: str) -> dict:
    """Hash the exact code surface that can change a Stage-I result.

    A source hash is mandatory because uploaded ZIP snapshots have no Git
    metadata, while a Git commit alone would miss uncommitted changes.  P1's
    list is intentionally narrow: an unrelated future P2 edit must not
    invalidate already-completed P1 seeds.
    """
    package_root = Path(__file__).resolve().parents[1]
    common = (
        "cli.py",
        "core/artifacts.py",
        "core/runner.py",
        "core/stage1.py",
        "core/stage1_models.py",
        "core/stage1_run.py",
        "core/structured.py",
        "reporting/stage1_aggregate.py",
    )
    if problem == "p1":
        problem_files = (
            "problems/p1/config.py",
            "problems/p1/dynamics.py",
            "problems/p1/oracle.py",
            "problems/p1/stage1_torch.py",
        )
    else:
        raise KeyError(problem)
    file_hashes = {}
    combined = hashlib.sha256()
    for relative in (*common, *problem_files):
        path = package_root / relative
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        file_hashes[relative] = digest
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(data)
        combined.update(b"\0")
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    return {
        "git_commit": commit,
        "source_tree_sha256": combined.hexdigest(),
        "files": file_hashes,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "pyyaml": _package_version("pyyaml"),
            "torch": _package_version("torch"),
            "pgdpo_delay": _package_version("pgdpo-delay"),
        },
    }


def _read_protocol(name_or_path: str) -> tuple[dict, str]:
    """Resolve an explicit path, packaged protocol, or user-derived protocol.

    Canonical packaged names cannot be shadowed by a differing CWD file.  The
    rule mirrors the problem-config loaders and prevents a stale experiment
    YAML from silently retaining a canonical protocol name.
    """
    name_or_path = str(name_or_path)
    if "/" in name_or_path or "\\" in name_or_path \
            or name_or_path.endswith((".yaml", ".yml")):
        path = Path(name_or_path)
        if not path.is_file():
            raise KeyError(f"no Stage-I protocol file {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return payload, path.stem

    local = Path.cwd() / "configs" / "stage1" / f"{name_or_path}.yaml"
    res = files("pgdpo_delay.configs").joinpath(
        "stage1", f"{name_or_path}.yaml")
    if res.is_file():
        text = res.read_text(encoding="utf-8")
        if local.exists() and local.read_text(encoding="utf-8") != text:
            raise RuntimeError(
                f"{local} shadows canonical Stage-I protocol "
                f"{name_or_path!r} with different content; derive under a "
                "new name or pass an explicit path.")
        return yaml.safe_load(text), name_or_path
    if local.is_file():
        return yaml.safe_load(local.read_text(encoding="utf-8")), name_or_path
    raise KeyError(f"unknown Stage-I protocol: {name_or_path}")


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if out != value or out < (0 if allow_zero else 1):
        bound = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{field} must be a {bound} integer")
    return out


def _validate_protocol(protocol: dict, resolved_name: str) -> dict:
    if not isinstance(protocol, dict):
        raise ValueError("Stage-I protocol YAML must contain a mapping")
    required = {
        "schema", "name", "problem", "problem_config", "method", "chart",
        "dtype", "training", "evaluation",
    }
    missing = sorted(required - protocol.keys())
    extra = sorted(protocol.keys() - required)
    if missing or extra:
        raise ValueError(f"invalid Stage-I protocol keys: missing={missing}, "
                         f"unknown={extra}")
    if protocol["schema"] != PROTOCOL_SCHEMA:
        raise ValueError(f"unsupported Stage-I protocol schema "
                         f"{protocol['schema']!r}")
    for key in ("name", "problem", "problem_config", "method", "chart",
                "dtype"):
        if not isinstance(protocol[key], str) or not protocol[key]:
            raise ValueError(f"protocol.{key} must be a nonempty string")
    # For canonical/user names, the file name is part of the identity.  An
    # explicit path may use any stem only when the declared name matches it.
    if protocol["name"] != resolved_name:
        raise ValueError(
            f"protocol name {protocol['name']!r} does not match file/name "
            f"{resolved_name!r}")
    if protocol["method"] != _METHOD:
        raise ValueError(f"unsupported Stage-I method: {protocol['method']}")
    if protocol["dtype"] not in _DTYPES:
        raise ValueError(f"unsupported dtype: {protocol['dtype']}")

    train = protocol["training"]
    evaluation = protocol["evaluation"]
    if not isinstance(train, dict) or set(train) != _TRAIN_KEYS:
        raise ValueError("protocol.training must contain exactly "
                         f"{sorted(_TRAIN_KEYS)}")
    if not isinstance(evaluation, dict) or set(evaluation) != _EVAL_KEYS:
        raise ValueError("protocol.evaluation must contain exactly "
                         f"{sorted(_EVAL_KEYS)}")
    for key in (
        "iters", "batch", "hidden", "num_layers", "log_every", "val_batch",
    ):
        train[key] = _positive_int(train[key], f"training.{key}")
    train["val_every"] = _positive_int(
        train["val_every"], "training.val_every", allow_zero=True)
    for key in ("lr", "clip_grad"):
        try:
            train[key] = float(train[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"training.{key} must be numeric") from exc
        if not np.isfinite(train[key]) or train[key] <= 0:
            raise ValueError(f"training.{key} must be positive and finite")
    evaluation["Np"] = _positive_int(evaluation["Np"], "evaluation.Np")
    if evaluation["Np"] < 2:
        raise ValueError(
            "evaluation.Np must be at least 2 for paired Monte Carlo SE")
    evaluation["seed"] = _positive_int(
        evaluation["seed"], "evaluation.seed", allow_zero=True)
    evaluation["batch_size"] = _positive_int(
        evaluation["batch_size"], "evaluation.batch_size")
    return _jsonable(protocol)


def load_stage1_protocol(name_or_path: str) -> dict:
    """Load and strictly validate a Stage-I protocol without importing torch."""
    payload, resolved_name = _read_protocol(name_or_path)
    return _validate_protocol(payload, resolved_name)


def _problem_snapshot(problem: str, config_name: str) -> tuple[dict, dict]:
    """Return (derived config, JSON problem snapshot), torch-free."""
    if problem == "p1":
        from ..problems.p1.config import (
            load_config,
            scientific_config_hash,
            scientific_config_snapshot,
        )
        cfg = load_config(config_name)
    else:
        raise KeyError(
            f"Stage-I worker has no registered problem adapter for {problem!r}")
    raw = _jsonable(cfg["raw"])
    # Store the source YAML in full.  The compact derived block catches loader
    # semantics (grid/control changes) while avoiding duplicated large arrays.
    derived = {
        "variant": cfg["variant"],
        "T": cfg["T"],
        "delta": cfg["delta"],
        "h": cfg["h"],
        "N": cfg["N"],
        "H": cfg["H"],
        "control_kind": cfg["control_kind"],
        "bounds": cfg["bounds"],
    }
    snapshot = {
        "name": config_name,
        "raw": raw,
        "raw_hash": _sha256(raw),
        "derived": _jsonable(derived),
        "scientific": _jsonable(scientific_config_snapshot(cfg)),
        "scientific_hash": scientific_config_hash(cfg),
    }
    return cfg, snapshot


def canonical_run_spec(problem: str, protocol_name: str) -> dict:
    """Build the seed-independent scientific run identity.

    The scheduler and worker both call this function.  A training seed and a
    physical CUDA slot are intentionally excluded: seeds are replicate IDs,
    while a GPU slot may change when a failed job is retried.
    """
    protocol = load_stage1_protocol(protocol_name)
    if protocol["problem"] != problem:
        raise ValueError(
            f"protocol {protocol['name']!r} is for {protocol['problem']!r}, "
            f"not requested problem {problem!r}")
    cfg, problem_snapshot = _problem_snapshot(
        problem, protocol["problem_config"])
    if problem == "p1":
        actual_chart = "identity" if cfg["control_kind"] == "unconstrained" \
            else protocol["chart"]
        if protocol["chart"] != actual_chart:
            raise ValueError(
                f"protocol chart {protocol['chart']!r} disagrees with P1 "
                f"control kind {cfg['control_kind']!r} (expected "
                f"{actual_chart!r})")
        adapter_class = \
            "pgdpo_delay.problems.p1.stage1_torch.P1Stage1Adapter"
    else:  # guarded by _problem_snapshot; keeps the identity explicit.
        raise KeyError(problem)
    protocol_hash = _sha256(protocol)
    if problem == "p1":
        initial_law = {
            "api": "p1.make_hist-v1",
            "shared_by_training_and_evaluation": True,
            "template_selector": "DiscreteUniform{constant,ramp,cosine}",
            "amplitude": "Uniform[-1.2,1.2]",
            "phase": "Uniform[0,2*pi]",
            "ramp": "a*(1+theta/delta)",
            "cosine": "a*cos(2*pi*theta/delta+phase)",
        }
        input_schema = {
            "api": "p1.stage1_features-v2",
            "feature_schema": "state_global_time_relative_lag_v2",
            "feat_dim": 3,
            "sequence_order": "oldest_to_newest",
            "token_features": [
                "state_value",
                "global_current_time_kh_over_T",
                "relative_lag_minus1_to_0",
            ],
            "state_scaling": "none",
        }
    return {
        "schema": 1,
        "worker_api_version": WORKER_API_VERSION,
        "problem": problem,
        "method": protocol["method"],
        "protocol_name": protocol["name"],
        "protocol_hash": protocol_hash,
        "problem_config": problem_snapshot,
        "problem_config_raw_hash": problem_snapshot["raw_hash"],
        "problem_config_hash": problem_snapshot["scientific_hash"],
        "chart": protocol["chart"],
        "dtype": protocol["dtype"],
        "adapter_class": adapter_class,
        "initial_law": initial_law,
        "input_schema": input_schema,
        "source_identity": _source_identity(problem),
        "training": protocol["training"],
        "evaluation": protocol["evaluation"],
    }


def run_fingerprint(spec: dict) -> str:
    """Stable, seed-independent fingerprint of :func:`canonical_run_spec`."""
    if not isinstance(spec, dict):
        raise TypeError("run spec must be a mapping")
    return _sha256(spec)[:24]


def required_metrics_for_spec(spec: dict) -> tuple[str, ...]:
    """Exact finite metric contract consumed by Stage-I aggregation."""
    if spec.get("problem") == "p1" and \
            spec.get("problem_config", {}).get("derived", {}).get(
                "control_kind") == "unconstrained":
        required = list(_P1U_REQUIRED_METRICS)
        if spec.get("training", {}).get("val_every", 0):
            required.append("best_validation_loss")
        return tuple(required)
    raise KeyError("no metric contract registered for this Stage-I spec")


def metric_roles_for_spec(spec: dict) -> dict[str, str]:
    """Classify metrics so aggregation does not invent an uncertainty axis."""
    required = required_metrics_for_spec(spec)
    roles = {name: "training_seed_metric" for name in required}
    for name in (
        "J_exact", "J_oracle_mc", "mc_anchor_gap", "mc_anchor_gap_se",
    ):
        roles[name] = "shared_evaluation_diagnostic"
    roles["dJ_se"] = "within_policy_paired_mc_se"
    for name in (
        "initial_train_loss", "final_train_loss", "best_validation_loss",
        "best_iter", "clip_frac", "train_runtime_seconds",
        "evaluation_runtime_seconds", "total_runtime_seconds",
        "peak_gpu_memory_mb",
    ):
        roles[name] = "health_or_runtime"
    return roles


def _status_payload(*, status: str, problem: str, seed: int,
                    fingerprint: str, problem_config_hash: str,
                    runtime_seconds: float = 0.0,
                    error: str | None = None) -> dict:
    payload = {
        "schema": STATUS_SCHEMA,
        "status": status,
        "problem": problem,
        "method": _METHOD,
        "seed": int(seed),
        "run_fingerprint": fingerprint,
        "problem_config_hash": problem_config_hash,
        "runtime_seconds": float(runtime_seconds),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if error is not None:
        payload["error"] = error
    return payload


def _device_metadata(torch, requested, resolved) -> dict:
    meta = {
        "requested": str(requested),
        "resolved": str(resolved),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "numpy_version": str(np.__version__),
        "scipy_version": _package_version("scipy"),
        "cuda_runtime_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"),
    }
    if resolved.type == "cuda":
        index = resolved.index
        if index is None:
            index = int(torch.cuda.current_device())
        prop = torch.cuda.get_device_properties(index)
        meta.update({
            "cuda_index": int(index),
            "cuda_name": str(prop.name),
            "cuda_capability": list(torch.cuda.get_device_capability(index)),
            "cuda_total_memory_bytes": int(prop.total_memory),
        })
    return meta


def _synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_problem(problem: str, spec: dict, device):
    """Lazy torch/problem construction for the registered worker adapter."""
    import torch
    dtype = torch.float32 if spec["dtype"] == "float32" else torch.float64
    if problem == "p1":
        from ..problems.p1.config import load_config
        from ..problems.p1.stage1_torch import P1Stage1Adapter
        cfg = load_config(spec["problem_config"]["name"])
        # P1Stage1Adapter accepts sigmoid/clip as the box-chart selector; its
        # unconstrained branch always promotes this to identity.
        selector = "sigmoid" if spec["chart"] == "identity" \
            else spec["chart"]
        adapter = P1Stage1Adapter(
            cfg, device=device, dtype=dtype, chart=selector)
        if adapter.chart_kind != spec["chart"]:
            raise RuntimeError(
                f"adapter resolved chart {adapter.chart_kind!r}; expected "
                f"{spec['chart']!r}")
        expected_input = spec["input_schema"]
        if adapter.feat_dim != expected_input["feat_dim"] or \
                adapter.feature_schema != expected_input["feature_schema"] or \
                adapter.sequence_order != expected_input["sequence_order"]:
            raise RuntimeError(
                "P1 adapter input schema disagrees with frozen run spec")
        return cfg, adapter
    raise KeyError(problem)


def _evaluate(problem: str, cfg: dict, adapter, policy, evaluation: dict) \
        -> dict:
    if problem == "p1":
        from ..problems.p1.stage1_torch import p1u_pilot_metrics
        return p1u_pilot_metrics(
            cfg, adapter, policy, Np=evaluation["Np"],
            seed=evaluation["seed"],
            policy_batch_size=evaluation["batch_size"])
    raise KeyError(problem)


def run_single_seed(*, problem: str, protocol_name: str, seed: int,
                    device: str, outdir: str | Path,
                    expected_run_fingerprint: str) -> dict:
    """Execute and persist exactly one Stage-I replicate.

    ``outdir`` may be absent or an existing empty directory.  The sole
    exception is a regular ``run.log`` opened by the scheduler for subprocess
    stdout/stderr capture.  Any other entry is refused; retries/publication
    are scheduler responsibilities.
    """
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    spec = canonical_run_spec(problem, protocol_name)
    fingerprint = run_fingerprint(spec)
    if expected_run_fingerprint != fingerprint:
        raise ValueError(
            f"run fingerprint mismatch: scheduler supplied "
            f"{expected_run_fingerprint!r}, worker computed {fingerprint!r}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    initial_entries = list(outdir.iterdir())
    allowed_log = outdir / "run.log"
    unexpected = [path for path in initial_entries
                  if path != allowed_log or not path.is_file()
                  or path.is_symlink()]
    if unexpected:
        raise FileExistsError(
            f"Stage-I worker target contains unexpected entries: "
            f"{[path.name for path in unexpected]}")
    started = time.monotonic()
    atomic_write_json(outdir / "status.json", _status_payload(
        status="RUNNING", problem=problem, seed=seed,
        fingerprint=fingerprint,
        problem_config_hash=spec["problem_config_hash"]))

    try:
        import torch
        from .stage1 import train_stage1
        from .stage1_models import save_checkpoint

        resolved_device = torch.device(device)
        if resolved_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device {device!r} requested but CUDA is unavailable")
            if resolved_device.index is not None \
                    and resolved_device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device {device!r} is outside available range "
                    f"0..{torch.cuda.device_count()-1}")
            # Pin the process-wide default before train_stage1 calls
            # torch.manual_seed or constructs any CUDA object.  Every seed
            # process then owns exactly the GPU slot assigned by the parent.
            torch.cuda.set_device(resolved_device)
            torch.cuda.reset_peak_memory_stats(resolved_device)
        elif resolved_device.type != "cpu":
            raise ValueError("Stage-I worker supports only cpu or cuda devices")

        cfg, adapter = _build_problem(problem, spec, resolved_device)
        train_hp = dict(spec["training"])
        _synchronize(torch, resolved_device)
        train_started = time.monotonic()
        result = train_stage1(
            adapter, seed=seed, device=resolved_device, **train_hp)
        _synchronize(torch, resolved_device)
        train_seconds = time.monotonic() - train_started

        eval_started = time.monotonic()
        oracle_metrics = _evaluate(
            problem, cfg, adapter, result["policy"], spec["evaluation"])
        _synchronize(torch, resolved_device)
        evaluation_seconds = time.monotonic() - eval_started

        checkpoint_extra = {
            "checkpoint_schema": 3,
            "problem": problem,
            "method": spec["method"],
            "problem_config": spec["problem_config"]["name"],
            "problem_config_raw_hash": spec["problem_config_raw_hash"],
            "problem_config_hash": spec["problem_config_hash"],
            "protocol": spec["protocol_name"],
            "protocol_hash": spec["protocol_hash"],
            "chart": spec["chart"],
            "run_fingerprint": fingerprint,
            "config_hash": config_hash(spec),
            "worker_api_version": WORKER_API_VERSION,
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "source_tree_sha256": spec["source_identity"][
                "source_tree_sha256"],
            "initial_law": spec["initial_law"],
            "input_schema": spec["input_schema"],
            "seed": seed,
            "seeds": result["seeds"],
        }
        save_checkpoint(result["policy"], outdir, extra=checkpoint_extra)

        val_trace = np.asarray(result["val_trace"], dtype=np.float64)
        if val_trace.size:
            val_iterations = val_trace[:, 0].astype(np.int64)
            val_losses = val_trace[:, 1]
            best_validation_loss = float(np.min(val_losses))
        else:
            val_iterations = np.empty(0, dtype=np.int64)
            val_losses = np.empty(0, dtype=np.float64)
            best_validation_loss = float("nan")
        np.savez_compressed(
            outdir / "training_trace.npz",
            iteration=np.arange(1, len(result["losses"]) + 1,
                                dtype=np.int64),
            training_loss=np.asarray(result["losses"], dtype=np.float64),
            grad_norm=np.asarray(result["grad_norms"], dtype=np.float64),
            validation_iteration=val_iterations,
            validation_loss=val_losses,
        )

        total_seconds = time.monotonic() - started
        metrics = {str(k): float(v) for k, v in oracle_metrics.items()}
        metrics.update({
            "initial_train_loss": float(result["losses"][0]),
            "final_train_loss": float(result["losses"][-1]),
            "best_iter": float(result["best_iter"]),
            "clip_frac": float(result["clip_frac"]),
            "train_runtime_seconds": float(train_seconds),
            "evaluation_runtime_seconds": float(evaluation_seconds),
            "total_runtime_seconds": float(total_seconds),
        })
        if val_trace.size:
            metrics["best_validation_loss"] = best_validation_loss
        if resolved_device.type == "cuda":
            metrics["peak_gpu_memory_mb"] = float(
                torch.cuda.max_memory_allocated(resolved_device) / 2**20)
        required_metrics = required_metrics_for_spec(spec)
        missing_metrics = sorted(set(required_metrics) - set(metrics))
        if missing_metrics:
            raise RuntimeError(
                f"worker omitted required Stage-I metrics: {missing_metrics}")
        nonfinite_metrics = sorted(
            key for key, value in metrics.items() if not np.isfinite(value))
        if nonfinite_metrics:
            raise FloatingPointError(
                "nonfinite Stage-I metrics: " + ", ".join(nonfinite_metrics))
        atomic_write_json(outdir / "metrics.json", metrics)

        device_meta = _device_metadata(torch, device, resolved_device)
        extra = {
            "status": "COMPLETE",
            "run_fingerprint": fingerprint,
            "worker_api_version": WORKER_API_VERSION,
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "required_metrics": list(required_metrics),
            "protocol": spec["protocol_name"],
            "protocol_hash": spec["protocol_hash"],
            "problem_config": spec["problem_config"]["name"],
            "problem_config_raw_hash": spec["problem_config_raw_hash"],
            "problem_config_hash": spec["problem_config_hash"],
            "chart": spec["chart"],
            "initial_law": spec["initial_law"],
            "trainer_hp": result["hp"],
            "source_identity": spec["source_identity"],
            "best_iter": int(result["best_iter"]),
            "clip_frac": float(result["clip_frac"]),
            "runtime": {
                "train_seconds": float(train_seconds),
                "evaluation_seconds": float(evaluation_seconds),
                "total_seconds": float(total_seconds),
            },
            "device_metadata": device_meta,
        }
        if resolved_device.type == "cuda":
            extra["device_metadata"]["peak_memory_bytes"] = int(
                torch.cuda.max_memory_allocated(resolved_device))
        all_seeds = {"train": seed, **result["seeds"],
                     "evaluation": int(spec["evaluation"]["seed"])}
        manifest = write_manifest(
            outdir, problem=problem, method=spec["method"], config=spec,
            seeds=all_seeds, device=str(resolved_device),
            api_versions={"stage1_worker": WORKER_API_VERSION},
            solver="pytorch-bptt", extra=extra)
        if manifest["config_hash"] != config_hash(spec):
            raise RuntimeError("manifest config hash mismatch")

        # Publication marker: this must remain the final filesystem write.
        total_seconds = time.monotonic() - started
        atomic_write_json(outdir / "status.json", _status_payload(
            status="COMPLETE", problem=problem, seed=seed,
            fingerprint=fingerprint,
            problem_config_hash=spec["problem_config_hash"],
            runtime_seconds=total_seconds))
        return {"metrics": metrics, "manifest": manifest,
                "status": "COMPLETE", "outdir": str(outdir)}
    except BaseException as exc:
        atomic_write_json(outdir / "status.json", _status_payload(
            status="FAILED", problem=problem, seed=seed,
            fingerprint=fingerprint,
            problem_config_hash=spec["problem_config_hash"],
            runtime_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}"))
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="one-seed Stage-I worker")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--device", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--run-fingerprint", required=True)
    args = ap.parse_args(argv)
    run_single_seed(
        problem=args.problem, protocol_name=args.protocol, seed=args.seed,
        device=args.device, outdir=args.outdir,
        expected_run_fingerprint=args.run_fingerprint)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess use
    raise SystemExit(main())
