"""Seed-parallel run orchestration.

The process scheduler in this module is deliberately solver agnostic.  It
assigns at most one child process to each supplied device label, publishes a
seed directory only after validating the child's completion artifacts, and
keeps failed attempts for diagnosis.  It neither imports torch nor uses
threads; callers provide the complete child-process command.

The older :func:`run_seeds` helper is retained for callers that still execute
an in-process ``run_fn`` sequentially.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
from numbers import Real
from pathlib import Path
import signal
import subprocess
import time
import uuid
from typing import Callable, Iterable, Mapping, Sequence

from . import artifacts


_BASE_REQUIRED_FILES = (
    "manifest.json",
    "config.json",
    "metrics.json",
    "status.json",
)
_SUMMARY_SCHEMA = 1


class _SchedulerSignalExit(SystemExit):
    """Turn default termination signals into a cleanup-capable exit."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(128 + signum)


def run_seeds(problem_name, run_fn, config, seeds, outroot):
    """Run the legacy in-process seed loop sequentially."""
    results = []
    for s in seeds:
        outdir = f"{outroot}/{problem_name}/seed{s}"
        artifacts.write_manifest(outdir, problem=problem_name, method="run",
                                 config=config, seeds=dict(train=s))
        results.append(run_fn(config, s, outdir))
    return results


@dataclass
class _ActiveProcess:
    seed: int
    device: str
    stage_dir: Path
    started_monotonic: float
    process: subprocess.Popen
    log_file: object
    log_offset: int = 0
    log_pending: bytes = b""


def _emit_live_log(child: _ActiveProcess, *, final: bool = False) -> None:
    """Mirror newly appended child-log lines to the parent terminal.

    The worker keeps writing directly to its private ``run.log``; the
    scheduler tails that regular file instead of inserting a PIPE/thread into
    the multi-GPU process lifecycle.  Complete lines are prefixed only on the
    parent terminal.  The immutable per-seed log remains byte-for-byte equal
    to the child output.
    """
    log_path = child.stage_dir / "run.log"
    try:
        with log_path.open("rb") as source:
            source.seek(child.log_offset)
            chunk = source.read()
    except OSError:
        chunk = b""
    child.log_offset += len(chunk)
    payload = child.log_pending + chunk
    pieces = payload.split(b"\n")
    if final:
        complete = pieces
        child.log_pending = b""
    else:
        complete = pieces[:-1]
        child.log_pending = pieces[-1]
    prefix = f"[seed={child.seed} device={child.device}]"
    for index, raw in enumerate(complete):
        if raw.endswith(b"\r"):
            raw = raw[:-1]
        # A terminal newline creates a trailing empty split item; it is not a
        # child log line and must not generate a prefixed blank line.
        if not raw and final and index == len(complete) - 1:
            continue
        print(f"{prefix} {raw.decode('utf-8', errors='replace')}",
              flush=True)


def _safe_required_file(name: str) -> Path:
    rel = Path(name)
    if (not isinstance(name, str) or not name or rel.is_absolute()
            or not rel.parts
            or any(part in ("", ".", "..") for part in rel.parts)):
        raise ValueError(f"unsafe required artifact path: {name!r}")
    return rel


def _validate_inputs(
        seeds: Iterable[int], devices: Iterable[str],
        run_fingerprint: str, required_files: Iterable[str],
) -> tuple[list[int], list[str], tuple[Path, ...]]:
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("at least one seed is required")
    if any(isinstance(seed, bool) or not isinstance(seed, int)
           for seed in seed_list):
        raise TypeError("seeds must be integers")
    if any(seed < 0 for seed in seed_list):
        raise ValueError("seeds must be nonnegative")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("seeds must be unique")

    device_list = list(devices)
    if not device_list:
        raise ValueError("at least one device label is required")
    if any(not isinstance(device, str) or not device.strip()
           for device in device_list):
        raise ValueError("device labels must be non-empty strings")
    if len(set(device_list)) != len(device_list):
        raise ValueError("device labels must be unique")
    if not isinstance(run_fingerprint, str) or not run_fingerprint:
        raise ValueError("run_fingerprint must be a non-empty string")

    required = list(_BASE_REQUIRED_FILES)
    required.extend(required_files)
    required_paths = tuple(dict.fromkeys(
        _safe_required_file(name) for name in required))
    return seed_list, device_list, required_paths


def _load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path.name}")
    return payload


def _object_field(payload: Mapping, key: str, *, artifact: str) -> Mapping:
    """Return one required nested JSON object with a controlled error."""
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{artifact}.{key} must contain a JSON object")
    return value


def _validate_complete_directory(
        directory: Path, run_fingerprint: str,
        required_files: Sequence[Path], expected_seed: int,
) -> None:
    """Validate one unpublished stage or an existing final seed directory."""
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"seed artifact path is not a regular directory: {directory}")
    missing = []
    for rel in required_files:
        candidate = directory / rel
        if not candidate.is_file() or candidate.is_symlink():
            missing.append(rel.as_posix())
    if missing:
        raise FileNotFoundError(
            "incomplete seed artifacts; missing regular files: "
            + ", ".join(missing))

    # Parse and cross-check the scientific identity. File existence alone is
    # insufficient for resume: a stale/tampered checkpoint must never be
    # marked SKIPPED merely because status.json carries the requested hash.
    manifest = _load_json(directory / "manifest.json")
    config = _load_json(directory / "config.json")
    metrics = _load_json(directory / "metrics.json")
    status = _load_json(directory / "status.json")
    if status.get("status") != "COMPLETE":
        raise ValueError(
            "status.json does not declare status='COMPLETE'")
    actual_fingerprint = status.get("run_fingerprint")
    if actual_fingerprint != run_fingerprint:
        raise ValueError(
            "status.json run_fingerprint mismatch: "
            f"expected {run_fingerprint!r}, found {actual_fingerprint!r}")
    if status.get("seed") != expected_seed:
        raise ValueError(
            f"status.json seed mismatch: expected {expected_seed}, "
            f"found {status.get('seed')!r}")

    problem = manifest.get("problem")
    method = manifest.get("method")
    if not isinstance(problem, str) or not problem:
        raise ValueError("manifest.json lacks a nonempty problem")
    if not isinstance(method, str) or not method:
        raise ValueError("manifest.json lacks a nonempty method")
    if status.get("problem") != problem or status.get("method") != method:
        raise ValueError("status.json problem/method disagrees with manifest")
    seeds = manifest.get("seeds")
    if not isinstance(seeds, Mapping) or seeds.get("train") != expected_seed:
        raise ValueError(
            "manifest.json training seed disagrees with seed directory")
    extra = manifest.get("extra")
    if not isinstance(extra, Mapping) or \
            extra.get("run_fingerprint") != run_fingerprint:
        raise ValueError(
            "manifest.json run_fingerprint missing or mismatched")
    if manifest.get("config_hash") != artifacts.config_hash(config):
        raise ValueError("manifest config_hash does not match config.json")
    if config.get("problem") != problem or config.get("method") != method:
        raise ValueError("config.json problem/method disagrees with manifest")
    problem_config_hash = config.get("problem_config_hash")
    if problem_config_hash is not None:
        if not isinstance(problem_config_hash, str) or not problem_config_hash:
            raise ValueError("config.json problem_config_hash must be nonempty")
        if status.get("problem_config_hash") != problem_config_hash:
            raise ValueError(
                "status.json problem_config_hash disagrees with config.json")
        if extra.get("problem_config_hash") != problem_config_hash:
            raise ValueError(
                "manifest problem_config_hash disagrees with config.json")

    required_metrics = extra.get("required_metrics", [])
    if not isinstance(required_metrics, list) or any(
            not isinstance(name, str) or not name
            for name in required_metrics):
        raise ValueError("manifest required_metrics must be a string list")
    missing_metrics = sorted(set(required_metrics) - set(metrics))
    if missing_metrics:
        raise ValueError(
            f"metrics.json lacks required metrics: {missing_metrics}")
    for name in required_metrics:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, Real) or \
                not math.isfinite(float(value)):
            raise ValueError(
                f"required metric {name!r} is not finite numeric data")

    checkpoint_path = directory / "stage1_spec.json"
    if checkpoint_path.exists():
        if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
            raise ValueError("stage1_spec.json is not a regular file")
        checkpoint = _load_json(checkpoint_path)
        problem_config = _object_field(
            config, "problem_config", artifact="config.json")
        source_identity = _object_field(
            config, "source_identity", artifact="config.json")
        computed_fingerprint = hashlib.sha256(json.dumps(
            config, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")).hexdigest()[:24]
        if computed_fingerprint != run_fingerprint:
            raise ValueError(
                "config.json Stage-I run_fingerprint mismatch: "
                f"expected {run_fingerprint!r}, recomputed "
                f"{computed_fingerprint!r}")
        computed_config_hash = artifacts.config_hash(config)
        expected_binding = {
            "problem": problem,
            "method": method,
            "problem_config": problem_config.get("name"),
            "problem_config_raw_hash": config.get(
                "problem_config_raw_hash"),
            "problem_config_hash": problem_config_hash,
            "protocol": config.get("protocol_name"),
            "protocol_hash": config.get("protocol_hash"),
            "chart": config.get("chart"),
            "run_fingerprint": run_fingerprint,
            "config_hash": computed_config_hash,
            "worker_api_version": config.get("worker_api_version"),
            "source_tree_sha256": source_identity.get("source_tree_sha256"),
            "initial_law": config.get("initial_law"),
            "input_schema": config.get("input_schema"),
            "seed": expected_seed,
        }
        mismatches = [
            f"{key}: {checkpoint.get(key)!r} != {wanted!r}"
            for key, wanted in expected_binding.items()
            if checkpoint.get(key) != wanted
        ]
        if mismatches:
            raise ValueError(
                "stage1_spec.json binding mismatch: " + "; ".join(mismatches))


def _failure_destination(failed_root: Path, seed: int) -> Path:
    failed_root.mkdir(parents=True, exist_ok=True)
    return failed_root / f"seed{seed}-{uuid.uuid4().hex}"


def _retain_failure(
        stage_dir: Path, failed_root: Path, seed: int, payload: Mapping,
) -> Path:
    """Annotate and atomically retain one private staging directory."""
    if not stage_dir.is_dir():
        stage_dir.mkdir(parents=True, exist_ok=False)
    artifacts.atomic_write_json(stage_dir / "scheduler_failure.json", payload)
    destination = _failure_destination(failed_root, seed)
    os.replace(stage_dir, destination)
    return destination


def _relative_to_root(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _new_stage(staging_root: Path, seed: int) -> Path:
    staging_root.mkdir(parents=True, exist_ok=True)
    stage = staging_root / f"seed{seed}-{uuid.uuid4().hex}"
    stage.mkdir()
    return stage


def _install_termination_handlers() -> dict[int, object]:
    """Make default SIGTERM/SIGHUP paths unwind through scheduler cleanup.

    Signal handlers can only be changed from Python's main thread.  A library
    caller with a custom/ignored handler keeps ownership of that policy; only
    default handlers are temporarily replaced.  SIGINT already becomes
    ``KeyboardInterrupt`` under Python's default handler.
    """
    previous = {}

    def handle(signum, _frame):
        raise _SchedulerSignalExit(signum)

    for name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            old = signal.getsignal(signum)
            if old != signal.SIG_DFL:
                continue
            signal.signal(signum, handle)
        except (OSError, ValueError):
            continue
        previous[signum] = old
    return previous


def _restore_termination_handlers(previous: Mapping[int, object]) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (OSError, ValueError):
            pass


def _stop_processes(
        children: Sequence[_ActiveProcess], timeout: float,
) -> None:
    """Terminate all live children concurrently, then kill grace overruns."""
    live = []
    for child in children:
        if child.process.poll() is None:
            try:
                child.process.terminate()
            except OSError:
                pass
            live.append(child)

    # Use one shared deadline: N devices must not turn the grace interval into
    # N times the requested timeout.
    deadline = time.monotonic() + timeout
    for child in live:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            child.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                child.process.kill()
            except OSError:
                pass
    for child in live:
        try:
            child.process.wait()
        except (ChildProcessError, OSError):
            pass


def _summary_payload(
        *, records: Sequence[dict], seed_list: Sequence[int],
        device_list: Sequence[str], run_fingerprint: str,
        started_wall: float, force_failed: bool = False,
) -> dict:
    finished_wall = time.time()
    ordered = sorted(records, key=lambda item: seed_list.index(item["seed"]))
    counts = {
        status.lower(): sum(record["status"] == status for record in ordered)
        for status in ("COMPLETE", "FAILED", "SKIPPED", "CANCELLED")
    }
    return {
        "schema": _SUMMARY_SCHEMA,
        "status": "FAILED" if force_failed or counts["failed"]
        or counts["cancelled"] else "COMPLETE",
        "run_fingerprint": run_fingerprint,
        "requested_seeds": list(seed_list),
        "devices": list(device_list),
        "started_unix": started_wall,
        "finished_unix": finished_wall,
        "elapsed_seconds": finished_wall - started_wall,
        "counts": counts,
        "seeds": ordered,
    }


def _preflight_final_directories(
        root: Path, seeds: Sequence[int], resume: bool,
        run_fingerprint: str, required_files: Sequence[Path],
) -> tuple[list[int], list[dict]]:
    pending = []
    records = []
    errors = []
    for seed in seeds:
        final_dir = root / f"seed{seed}"
        if not final_dir.exists() and not final_dir.is_symlink():
            pending.append(seed)
            continue
        if not resume:
            errors.append(
                f"seed {seed}: target already exists ({final_dir})")
            continue
        try:
            _validate_complete_directory(
                final_dir, run_fingerprint, required_files, seed)
        except (OSError, ValueError) as exc:
            errors.append(
                f"seed {seed}: existing target is not a matching COMPLETE "
                f"run ({exc})")
            continue
        records.append({
            "seed": seed,
            "status": "SKIPPED",
            "device": None,
            "returncode": None,
            "elapsed_seconds": 0.0,
            "path": _relative_to_root(final_dir, root),
            "error": None,
        })
    if errors:
        raise FileExistsError(
            "refusing to overwrite or reuse seed directories:\n"
            + "\n".join(errors))
    return pending, records


def run_seed_processes(
        command_builder: Callable[[int, str, Path], Sequence[str]],
        seeds: Iterable[int],
        devices: Iterable[str],
        outroot,
        run_fingerprint: str,
        *,
        resume: bool = False,
        required_files: Iterable[str] = (),
        poll_interval: float = 0.05,
        cancel_timeout: float = 1.0,
        env: Mapping[str, str] | None = None,
        live_output: bool = False,
) -> dict:
    """Run independent seed commands on a dynamically refilled device pool.

    ``command_builder(seed, device, staging_dir)`` must return an argv
    sequence.  The child writes all artifacts into ``staging_dir`` and must
    finish with four JSON files: ``manifest.json``, ``config.json``,
    ``metrics.json``, and ``status.json``.  The latter must contain
    ``{"status": "COMPLETE", "run_fingerprint": ...}``.

    One child is active per device label.  Whenever a child exits, its slot is
    finalized and immediately assigned the next pending seed.  A valid stage
    is atomically renamed to ``outroot/seed<seed>``.  Nonzero exits, launch
    errors, and artifact-validation errors are retained under
    ``outroot/failed`` and represented in ``outroot/run_summary.json``.

    With ``resume=True``, only an existing, valid COMPLETE directory carrying
    the same fingerprint is skipped.  Every other existing target is refused;
    no seed directory is ever overwritten.  ``live_output=True`` mirrors
    complete lines from each private staging log to the parent terminal with
    seed/device prefixes while preserving the original per-seed log.
    """
    if not callable(command_builder):
        raise TypeError("command_builder must be callable")
    if isinstance(poll_interval, bool) or poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    if isinstance(cancel_timeout, bool) or cancel_timeout < 0:
        raise ValueError("cancel_timeout must be non-negative")
    if not isinstance(live_output, bool):
        raise TypeError("live_output must be a bool")
    seed_list, device_list, required_paths = _validate_inputs(
        seeds, devices, run_fingerprint, required_files)

    root = Path(outroot).resolve()
    root.mkdir(parents=True, exist_ok=True)
    pending_list, records = _preflight_final_directories(
        root, seed_list, resume, run_fingerprint, required_paths)
    pending = deque(pending_list)
    available = deque(device_list)
    active: dict[str, _ActiveProcess] = {}
    staging_root = root / ".staging"
    failed_root = root / "failed"
    started_wall = time.time()
    previous_signal_handlers = _install_termination_handlers()

    def record_failure(
            seed: int, device: str, stage_dir: Path, error: str,
            *, returncode: int | None, elapsed: float,
    ) -> None:
        failure_payload = {
            "seed": seed,
            "device": device,
            "run_fingerprint": run_fingerprint,
            "returncode": returncode,
            "error": error,
        }
        retained = _retain_failure(
            stage_dir, failed_root, seed, failure_payload)
        records.append({
            "seed": seed,
            "status": "FAILED",
            "device": device,
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "path": _relative_to_root(retained, root),
            "error": error,
        })

    try:
        while pending or active:
            # Fill every currently free slot.  Builder/launch failures consume
            # the seed but release the slot immediately for the next seed.
            while pending and available:
                seed = pending.popleft()
                device = available.popleft()
                stage_dir = _new_stage(staging_root, seed)
                started = time.monotonic()
                log_path = stage_dir / "run.log"
                log_file = None
                process = None
                try:
                    argv = command_builder(seed, device, stage_dir)
                    if (isinstance(argv, (str, bytes))
                            or not isinstance(argv, Sequence)
                            or not argv
                            or any(not isinstance(arg, str) or not arg
                                   for arg in argv)):
                        raise TypeError(
                            "command_builder must return a non-empty argv "
                            "sequence of non-empty strings")
                    log_file = log_path.open("xb")
                    process = subprocess.Popen(
                        list(argv),
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        shell=False,
                        env=None if env is None else dict(env),
                    )
                    # Register immediately after Popen: all later cancellation
                    # paths can now find and reap this child.
                    active[device] = _ActiveProcess(
                        seed=seed,
                        device=device,
                        stage_dir=stage_dir,
                        started_monotonic=started,
                        process=process,
                        log_file=log_file,
                    )
                except Exception as exc:
                    if log_file is not None:
                        log_file.close()
                    record_failure(
                        seed, device, stage_dir,
                        f"launch failed: {type(exc).__name__}: {exc}",
                        returncode=None,
                        elapsed=time.monotonic() - started,
                    )
                    available.append(device)
                    continue
                except BaseException as exc:
                    # Cover the narrow window after Popen but before insertion
                    # into ``active``; otherwise the outer handler owns it.
                    if process is not None and device not in active:
                        child = _ActiveProcess(
                            seed=seed,
                            device=device,
                            stage_dir=stage_dir,
                            started_monotonic=started,
                            process=process,
                            log_file=log_file,
                        )
                        _stop_processes([child], cancel_timeout)
                        if live_output:
                            _emit_live_log(child, final=True)
                        if log_file is not None:
                            log_file.close()
                        record_failure(
                            seed, device, stage_dir,
                            f"scheduler cancelled during launch: "
                            f"{type(exc).__name__}: {exc}",
                            returncode=process.poll(),
                            elapsed=time.monotonic() - started,
                        )
                    elif process is None:
                        if log_file is not None:
                            log_file.close()
                        record_failure(
                            seed, device, stage_dir,
                            f"scheduler cancelled during command build: "
                            f"{type(exc).__name__}: {exc}",
                            returncode=None,
                            elapsed=time.monotonic() - started,
                        )
                    raise

            if live_output:
                for child in active.values():
                    _emit_live_log(child)

            completed_devices = [
                device for device, child in active.items()
                if child.process.poll() is not None
            ]
            if completed_devices:
                for device in completed_devices:
                    # Keep the child registered until publication and its
                    # summary record are both complete.  Cancellation can then
                    # recover even if it lands during finalization.
                    child = active[device]
                    returncode = child.process.returncode
                    if live_output:
                        _emit_live_log(child, final=True)
                    child.log_file.close()
                    elapsed = time.monotonic() - child.started_monotonic
                    if returncode != 0:
                        record_failure(
                            child.seed, device, child.stage_dir,
                            f"child process exited with code {returncode}",
                            returncode=returncode,
                            elapsed=elapsed,
                        )
                        active.pop(device)
                        available.append(device)
                        continue
                    try:
                        _validate_complete_directory(
                            child.stage_dir, run_fingerprint, required_paths,
                            child.seed)
                        final_dir = root / f"seed{child.seed}"
                        # Preflight guaranteed this did not exist.  Re-check
                        # here to refuse races rather than overwrite it.
                        if final_dir.exists() or final_dir.is_symlink():
                            raise FileExistsError(
                                f"target appeared during run: {final_dir}")
                        os.replace(child.stage_dir, final_dir)
                    except (OSError, ValueError) as exc:
                        record_failure(
                            child.seed, device, child.stage_dir,
                            f"publication validation failed: "
                            f"{type(exc).__name__}: {exc}",
                            returncode=returncode,
                            elapsed=elapsed,
                        )
                    else:
                        records.append({
                            "seed": child.seed,
                            "status": "COMPLETE",
                            "device": device,
                            "returncode": returncode,
                            "elapsed_seconds": elapsed,
                            "path": _relative_to_root(final_dir, root),
                            "error": None,
                        })
                    active.pop(device)
                    available.append(device)
                # Re-enter immediately so newly free slots receive pending
                # seeds without an artificial polling delay.
                continue

            if active:
                time.sleep(poll_interval)

        summary = _summary_payload(
            records=records,
            seed_list=seed_list,
            device_list=device_list,
            run_fingerprint=run_fingerprint,
            started_wall=started_wall,
        )
        artifacts.atomic_write_json(root / "run_summary.json", summary)
        return summary
    except BaseException as exc:
        if isinstance(exc, _SchedulerSignalExit):
            reason = f"received signal {exc.signum}"
        else:
            reason = f"{type(exc).__name__}: {exc}"

        # Reap every child before touching its log/staging files.  A child that
        # has already exited is still waited on and recorded below.
        children = list(active.values())
        _stop_processes(children, cancel_timeout)
        recorded_seeds = {record["seed"] for record in records}
        for child in children:
            if live_output:
                _emit_live_log(child, final=True)
            if not child.log_file.closed:
                child.log_file.close()
            if child.seed in recorded_seeds:
                continue
            elapsed = time.monotonic() - child.started_monotonic
            final_dir = root / f"seed{child.seed}"
            if not child.stage_dir.exists() and final_dir.is_dir():
                # Cancellation may land immediately after the atomic rename.
                # Recover the already-published success instead of fabricating
                # an empty failed stage alongside it.
                try:
                    _validate_complete_directory(
                        final_dir, run_fingerprint, required_paths,
                        child.seed)
                except (OSError, ValueError):
                    pass
                else:
                    records.append({
                        "seed": child.seed,
                        "status": "COMPLETE",
                        "device": child.device,
                        "returncode": child.process.poll(),
                        "elapsed_seconds": elapsed,
                        "path": _relative_to_root(final_dir, root),
                        "error": None,
                    })
                    recorded_seeds.add(child.seed)
                    continue
            try:
                record_failure(
                    child.seed, child.device, child.stage_dir,
                    f"scheduler cancelled: {reason}",
                    returncode=child.process.poll(),
                    elapsed=elapsed,
                )
            except Exception as cleanup_exc:
                # Child reaping has already succeeded.  Preserve a truthful
                # summary even if filesystem retention itself failed.
                records.append({
                    "seed": child.seed,
                    "status": "FAILED",
                    "device": child.device,
                    "returncode": child.process.poll(),
                    "elapsed_seconds": elapsed,
                    "path": None,
                    "error": f"scheduler cancelled: {reason}; artifact "
                             f"retention failed: {cleanup_exc}",
                })
            recorded_seeds.add(child.seed)
        active.clear()

        # Seeds never launched are explicit in the interrupted run ledger.
        recorded_seeds = {record["seed"] for record in records}
        for seed in seed_list:
            if seed not in recorded_seeds:
                records.append({
                    "seed": seed,
                    "status": "CANCELLED",
                    "device": None,
                    "returncode": None,
                    "elapsed_seconds": 0.0,
                    "path": None,
                    "error": f"not started because scheduler was cancelled: "
                             f"{reason}",
                })
        summary = _summary_payload(
            records=records,
            seed_list=seed_list,
            device_list=device_list,
            run_fingerprint=run_fingerprint,
            started_wall=started_wall,
            force_failed=True,
        )
        # This write is best-effort only for catastrophic filesystem errors;
        # the original cancellation must remain the exception seen by caller.
        try:
            artifacts.atomic_write_json(root / "run_summary.json", summary)
        except Exception:
            pass
        raise
    finally:
        _restore_termination_handlers(previous_signal_handlers)
