"""Strict aggregation for independent Stage-I seed runs.

The Stage-I scheduler writes one directory per seed::

    run_root/seed<seed>/{manifest.json,status.json,metrics.json,...}

This module is deliberately downstream-only: it never imports a model or
re-runs training.  A successful aggregation verifies that all completed seed
runs belong to the same problem, method, configuration, and run fingerprint,
then publishes three small, atomic artifacts:

``per_seed.csv``
    One wide row per expected/discovered seed.  Failed or missing seeds are
    retained with blank metric cells when ``allow_incomplete=True``.
``summary.csv``
    One row per numeric metric with the sample standard deviation, standard
    error, and two-sided 95% Student-t confidence interval.
``summary.json``
    The same statistics plus run identity, counts, and optional failure
    metadata for machine consumption.

The canonical successful status is exactly ``"COMPLETE"``.  In particular,
the existence of a checkpoint or ``metrics.json`` alone never makes a run
eligible for aggregation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from pgdpo_delay.core.artifacts import atomic_write_json, config_hash


_COMPLETE = "COMPLETE"
_SEED_DIR = re.compile(r"^seed(-?\d+)$")
_BASE_COLUMNS = (
    "seed",
    "status",
    "problem",
    "method",
    "problem_config_hash",
    "config_hash",
    "run_fingerprint",
    "device",
    "error",
)


class Stage1AggregationError(ValueError):
    """Raised when seed artifacts cannot form one trustworthy experiment."""


@dataclass(frozen=True)
class Stage1AggregationResult:
    """Paths and in-memory payload returned by :func:`aggregate_stage1_runs`."""

    per_seed_csv: Path
    summary_csv: Path
    summary_json: Path
    payload: Mapping[str, Any]


@dataclass
class _SeedRecord:
    seed: int | None
    path: Path | None
    status: str
    manifest: dict[str, Any] | None
    config: dict[str, Any] | None
    status_payload: dict[str, Any] | None
    metrics: dict[str, int | float]
    problem_config_hash: str | None
    problems: list[str]
    error: str | None

    @property
    def complete(self) -> bool:
        return self.status == _COMPLETE and not self.problems


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a CSV through a same-directory temporary file and ``replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    created = False
    try:
        with open(tmp, "x", encoding="utf-8", newline="") as fp:
            created = True
            writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        if created:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def _fsync_dir(path: Path) -> None:
    """Best-effort durability for the directory entry after ``os.replace``."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _read_json_record(path: Path, label: str, problems: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        problems.append(f"missing {label}")
        return None
    if not path.is_file() or path.is_symlink():
        problems.append(f"{label} is not a regular non-symlink file")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"invalid {label}: {exc}")
        return None
    if not isinstance(payload, dict):
        problems.append(f"{label} must contain a JSON object")
        return None
    return payload


def _integer_seed(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _resolve_seed(
    directory: Path,
    manifest: Mapping[str, Any] | None,
    status: Mapping[str, Any] | None,
    problems: list[str],
) -> int | None:
    candidates: list[tuple[str, int]] = []
    match = _SEED_DIR.fullmatch(directory.name)
    if match:
        candidates.append(("directory", int(match.group(1))))
    if manifest is not None:
        seed = _integer_seed(manifest.get("seed"))
        if seed is None and isinstance(manifest.get("seeds"), Mapping):
            seed = _integer_seed(manifest["seeds"].get("train"))
        if seed is not None:
            candidates.append(("manifest", seed))
    if status is not None:
        seed = _integer_seed(status.get("seed"))
        if seed is not None:
            candidates.append(("status", seed))
    if not candidates:
        problems.append("seed is absent from directory, manifest, and status")
        return None
    distinct = {seed for _, seed in candidates}
    if len(distinct) != 1:
        detail = ", ".join(f"{source}={seed}" for source, seed in candidates)
        problems.append(f"conflicting seed identities ({detail})")
        return None
    return distinct.pop()


def _run_fingerprint(
    manifest: Mapping[str, Any] | None,
    status: Mapping[str, Any] | None,
    problems: list[str],
) -> str | None:
    values: list[tuple[str, Any]] = []
    if manifest is not None:
        extra = manifest.get("extra")
        if isinstance(extra, Mapping) and extra.get("run_fingerprint") is not None:
            values.append(("manifest.extra", extra.get("run_fingerprint")))
        if manifest.get("run_fingerprint") is not None:
            values.append(("manifest", manifest.get("run_fingerprint")))
    if status is not None and status.get("run_fingerprint") is not None:
        values.append(("status", status.get("run_fingerprint")))
    strings = [(source, value) for source, value in values
               if isinstance(value, str) and value]
    if len(strings) != len(values):
        problems.append("run_fingerprint must be a non-empty string")
    distinct = {value for _, value in strings}
    if len(distinct) > 1:
        detail = ", ".join(f"{source}={value!r}" for source, value in strings)
        problems.append(f"conflicting run_fingerprint values ({detail})")
        return None
    if not distinct:
        problems.append("missing run_fingerprint")
        return None
    return distinct.pop()


def _numeric_metrics(
    payload: Mapping[str, Any], problems: list[str]
) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            problems.append("metrics.json keys must be non-empty strings")
            continue
        if key in _BASE_COLUMNS:
            problems.append(f"metric name {key!r} is reserved")
            continue
        if isinstance(value, (dict, list, tuple)):
            problems.append(f"metrics.json must be flat; {key!r} is nested")
            continue
        # JSON booleans are integers in Python, but are not experimental
        # measurements and must not silently enter means or confidence bands.
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        number = float(value)
        if not math.isfinite(number):
            problems.append(f"metric {key!r} is not finite")
            continue
        # Preserve integral health fields such as ``best_iter`` in the
        # per-seed artifact while all summary arithmetic still uses float64.
        metrics[key] = value if isinstance(value, int) else number
    if not metrics:
        problems.append("metrics.json contains no finite numeric metrics")
    return metrics


def _identity(record: _SeedRecord) -> dict[str, str | None]:
    manifest = record.manifest or {}
    problem = manifest.get("problem")
    method = manifest.get("method")
    config_hash = manifest.get("config_hash")
    fingerprint = _run_fingerprint(record.manifest, record.status_payload, [])
    return {
        "problem": problem if isinstance(problem, str) and problem else None,
        "method": method if isinstance(method, str) and method else None,
        "problem_config_hash": record.problem_config_hash,
        "config_hash": config_hash if isinstance(config_hash, str) and config_hash else None,
        "run_fingerprint": fingerprint,
    }


def _load_seed_record(directory: Path) -> _SeedRecord:
    problems: list[str] = []
    manifest = _read_json_record(directory / "manifest.json", "manifest.json", problems)
    config = _read_json_record(directory / "config.json", "config.json", problems)
    status_payload = _read_json_record(directory / "status.json", "status.json", problems)
    metrics_payload = _read_json_record(directory / "metrics.json", "metrics.json", problems)
    seed = _resolve_seed(directory, manifest, status_payload, problems)

    status_value: Any = status_payload.get("status") if status_payload else None
    if not isinstance(status_value, str) or not status_value:
        problems.append("status.json lacks a non-empty status")
        status = "INVALID"
    else:
        status = status_value
        if status != _COMPLETE:
            problems.append(f"run status is {status!r}, not {_COMPLETE!r}")

    fingerprint_problems: list[str] = []
    _run_fingerprint(manifest, status_payload, fingerprint_problems)
    problems.extend(fingerprint_problems)

    if manifest is not None:
        for field in ("problem", "method", "config_hash"):
            value = manifest.get(field)
            if not isinstance(value, str) or not value:
                problems.append(f"manifest.json lacks non-empty {field}")
        if status_payload is not None:
            for field in ("problem", "method"):
                manifest_value = manifest.get(field)
                status_value = status_payload.get(field)
                if (status_value is not None
                        and manifest_value is not None
                        and status_value != manifest_value):
                    problems.append(
                        f"conflicting {field} within seed artifacts "
                        f"(manifest={manifest_value!r}, status={status_value!r})")
        if config is not None and manifest.get("config_hash") != \
                config_hash(config):
            problems.append(
                "manifest config_hash does not match config.json")

    hash_candidates: list[tuple[str, Any]] = []
    if config is not None and "problem_config_hash" in config:
        hash_candidates.append(("config", config.get("problem_config_hash")))
    if manifest is not None and isinstance(manifest.get("extra"), Mapping) \
            and "problem_config_hash" in manifest["extra"]:
        hash_candidates.append(
            ("manifest.extra", manifest["extra"].get("problem_config_hash")))
    if status_payload is not None and "problem_config_hash" in status_payload:
        hash_candidates.append(
            ("status", status_payload.get("problem_config_hash")))
    if config is not None and "problem_config_hash" in config:
        present_sources = {source for source, _ in hash_candidates}
        missing_sources = {
            "config", "manifest.extra", "status"
        } - present_sources
        if missing_sources:
            problems.append(
                "problem_config_hash missing from official Stage-I artifacts: "
                + ", ".join(sorted(missing_sources))
            )
    valid_hashes = []
    for source, value in hash_candidates:
        if not isinstance(value, str) or not value:
            problems.append(f"{source} problem_config_hash must be non-empty")
        else:
            valid_hashes.append((source, value))
    distinct_hashes = {value for _, value in valid_hashes}
    if len(distinct_hashes) > 1:
        detail = ", ".join(
            f"{source}={value!r}" for source, value in valid_hashes)
        problems.append(
            f"conflicting problem_config_hash values ({detail})")
        problem_config_hash = None
    else:
        problem_config_hash = next(iter(distinct_hashes), None)

    metrics = _numeric_metrics(metrics_payload, problems) if metrics_payload is not None else {}
    error = status_payload.get("error") if status_payload else None
    if error is not None and not isinstance(error, str):
        error = str(error)
    return _SeedRecord(
        seed=seed,
        path=directory,
        status=status,
        manifest=manifest,
        config=config,
        status_payload=status_payload,
        metrics=metrics,
        problem_config_hash=problem_config_hash,
        problems=problems,
        error=error,
    )


def _missing_record(
    seed: int,
    *,
    scheduler_record: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> _SeedRecord:
    scheduler_record = dict(scheduler_record or {})
    scheduler_status = scheduler_record.get("status")
    error = scheduler_record.get("error")
    if error is not None and not isinstance(error, str):
        error = str(error)
    path: Path | None = None
    stored_path = scheduler_record.get("path")
    if root is not None and isinstance(stored_path, str) and stored_path:
        candidate = Path(stored_path)
        if not candidate.is_absolute() and ".." not in candidate.parts:
            path = root / candidate

    if scheduler_status in ("FAILED", "CANCELLED"):
        status = scheduler_status
        problems = [
            f"scheduler reported a {scheduler_status.lower()} seed without "
            "a published COMPLETE directory"
        ]
    elif scheduler_status in ("COMPLETE", "SKIPPED"):
        status = "MISSING"
        problems = [
            f"scheduler reports {scheduler_status!r}, but the published seed directory is missing"
        ]
    else:
        status = "MISSING"
        problems = ["seed directory is missing"]
    return _SeedRecord(
        seed=seed,
        path=path,
        status=status,
        manifest=None,
        config=None,
        status_payload=scheduler_record or None,
        metrics={},
        problem_config_hash=None,
        problems=problems,
        error=error,
    )


def _load_scheduler_summary(root: Path) -> tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]:
    """Load optional scheduler metadata used only to explain absent seeds."""
    path = root / "run_summary.json"
    if not path.exists():
        return None, {}
    if not path.is_file() or path.is_symlink():
        raise Stage1AggregationError(
            "run_summary.json is not a regular non-symlink file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage1AggregationError(f"invalid run_summary.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise Stage1AggregationError("run_summary.json must contain a JSON object")
    raw_records = payload.get("seeds")
    if not isinstance(raw_records, list):
        raise Stage1AggregationError("run_summary.json.seeds must be a list")
    records: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise Stage1AggregationError(
                f"run_summary.json.seeds[{index}] must be an object")
        seed = _integer_seed(record.get("seed"))
        if seed is None:
            raise Stage1AggregationError(
                f"run_summary.json.seeds[{index}] has invalid seed")
        if seed in records:
            raise Stage1AggregationError(
                f"run_summary.json contains duplicate seed {seed}")
        records[seed] = record
    fingerprint = payload.get("run_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise Stage1AggregationError(
            "run_summary.json lacks a non-empty run_fingerprint")
    return payload, records


def _validate_common_identity(records: Sequence[_SeedRecord]) -> dict[str, str | None]:
    identity: dict[str, str | None] = {}
    for key in (
        "problem", "method", "problem_config_hash", "config_hash",
        "run_fingerprint",
    ):
        values: dict[str, list[int | None]] = {}
        for record in records:
            value = _identity(record).get(key)
            if value is not None:
                values.setdefault(value, []).append(record.seed)
        if len(values) > 1:
            detail = "; ".join(
                f"{value!r}: seeds {seeds}" for value, seeds in sorted(values.items()))
            raise Stage1AggregationError(f"inconsistent {key} across seed runs: {detail}")
        identity[key] = next(iter(values), None)
    return identity


def _validate_metric_schema(
    records: Sequence[_SeedRecord], required_metrics: Iterable[str] | None
) -> list[str]:
    complete = [record for record in records if record.complete]
    if not complete:
        return []
    schemas = {tuple(sorted(record.metrics)) for record in complete}
    if len(schemas) != 1:
        detail = "; ".join(
            f"seed {record.seed}: {sorted(record.metrics)}" for record in complete)
        raise Stage1AggregationError(
            "inconsistent numeric metric keys across completed seeds: " + detail)
    metric_names = list(next(iter(schemas)))
    if required_metrics is not None:
        requested = list(dict.fromkeys(required_metrics))
        invalid = [name for name in requested if not isinstance(name, str) or not name]
        if invalid:
            raise Stage1AggregationError(
                f"required_metrics contains invalid names: {invalid}")
        missing = sorted(set(requested) - set(metric_names))
        if missing:
            raise Stage1AggregationError(
                f"completed seeds lack required numeric metrics: {missing}")
    return metric_names


def _metric_summary(
    values: Sequence[float], *, role: str = "training_seed_metric"
) -> dict[str, float | int | str | bool | None]:
    array = np.asarray(values, dtype=np.float64)
    n = int(array.size)
    mean = float(np.mean(array))
    if role == "shared_evaluation_diagnostic":
        # Every learned-policy seed reuses the same reference/evaluation bank,
        # so repeated identical oracle anchors are not independent samples.
        # Suppress a misleading zero-width training-seed confidence interval.
        return {
            "n": n,
            "mean": mean,
            "sd": None,
            "se": None,
            "ci95_low": None,
            "ci95_high": None,
            "role": role,
            "seed_uncertainty_applicable": False,
        }
    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "sd": None,
            "se": None,
            "ci95_low": None,
            "ci95_high": None,
            "role": role,
            "seed_uncertainty_applicable": True,
        }
    sd = float(np.std(array, ddof=1))
    se = sd / math.sqrt(n)
    critical = float(student_t.ppf(0.975, df=n - 1))
    radius = critical * se
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
        "role": role,
        "seed_uncertainty_applicable": True,
    }


def _failure_payload(record: _SeedRecord, root: Path) -> dict[str, Any]:
    if record.path is None:
        relative_path = None
    else:
        try:
            relative_path = record.path.relative_to(root).as_posix()
        except ValueError:
            relative_path = str(record.path)
    status = record.status_payload or {}
    failure: dict[str, Any] = {
        "seed": record.seed,
        "status": record.status,
        "path": relative_path,
        "problems": list(record.problems),
    }
    if record.error:
        failure["error"] = record.error
    for key in ("returncode", "runtime_seconds", "elapsed_seconds"):
        value = status.get(key)
        if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)):
            failure[key] = float(value)
    return failure


def _per_seed_rows(
    records: Sequence[_SeedRecord],
    identity: Mapping[str, str | None],
    metric_names: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (item.seed is None, item.seed or 0)):
        manifest = record.manifest or {}
        row: dict[str, Any] = {
            "seed": "" if record.seed is None else record.seed,
            "status": record.status,
            "problem": manifest.get("problem") or identity.get("problem") or "",
            "method": manifest.get("method") or identity.get("method") or "",
            "problem_config_hash": record.problem_config_hash
            or identity.get("problem_config_hash") or "",
            "config_hash": manifest.get("config_hash") or identity.get("config_hash") or "",
            "run_fingerprint": _identity(record).get("run_fingerprint")
            or identity.get("run_fingerprint") or "",
            "device": manifest.get("device", ""),
            "error": record.error or "; ".join(record.problems),
        }
        for name in metric_names:
            row[name] = record.metrics.get(name, "") if record.complete else ""
        rows.append(row)
    return rows


def aggregate_stage1_runs(
    run_root: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    expected_seeds: Iterable[int] | None = None,
    allow_incomplete: bool = False,
    include_failure_metadata: bool = True,
    required_metrics: Iterable[str] | None = None,
    metric_roles: Mapping[str, str] | None = None,
    shared_evaluation_bank: bool = False,
) -> Stage1AggregationResult:
    """Validate and aggregate one Stage-I multi-seed experiment.

    Parameters
    ----------
    run_root:
        Directory containing immediate ``seed<integer>`` subdirectories.
    output_dir:
        Destination for the three aggregate artifacts.  Defaults to
        ``run_root``.
    expected_seeds:
        Optional declared seed roster. Missing seeds are integrity failures.
    allow_incomplete:
        If false (default), any failed, running, malformed, or missing seed
        aborts without publishing aggregate artifacts. If true, those seeds
        remain in ``per_seed.csv`` and failure counts/metadata while statistics
        use completed seeds only.
    include_failure_metadata:
        Include detailed failure records in ``summary.json``. Counts are
        always included.
    required_metrics:
        Optional numeric metric names that every completed run must contain.
    metric_roles:
        Optional metric-to-role mapping. ``shared_evaluation_diagnostic``
        suppresses a spurious across-seed confidence interval for values
        repeated from one common evaluation bank.
    shared_evaluation_bank:
        Whether every training seed was scored on the same evaluation bank.
        This changes the interpretation, not the arithmetic, of seed-level
        uncertainty.

    Returns
    -------
    Stage1AggregationResult
        Artifact paths plus the exact ``summary.json`` payload.
    """
    root = Path(run_root)
    if not root.is_dir():
        raise Stage1AggregationError(f"run_root is not a directory: {root}")

    scheduler_summary, scheduler_records = _load_scheduler_summary(root)

    expected: list[int] | None = None
    if expected_seeds is not None:
        raw_expected = list(expected_seeds)
        expected = []
        for value in raw_expected:
            seed = _integer_seed(value)
            if seed is None:
                raise Stage1AggregationError(f"invalid expected seed: {value!r}")
            expected.append(seed)
        if len(set(expected)) != len(expected):
            raise Stage1AggregationError(f"expected_seeds contains duplicates: {expected}")
        expected.sort()
    elif scheduler_records:
        expected = sorted(scheduler_records)

    directories = sorted(
        path for path in root.iterdir()
        if path.is_dir() and _SEED_DIR.fullmatch(path.name))
    if not directories and not expected:
        raise Stage1AggregationError(f"no seed directories found under {root}")
    records = [_load_seed_record(path) for path in directories]

    by_seed: dict[int, _SeedRecord] = {}
    for record in records:
        if record.seed is None:
            continue
        if record.seed in by_seed:
            raise Stage1AggregationError(
                f"duplicate artifacts for seed {record.seed}: "
                f"{by_seed[record.seed].path} and {record.path}")
        by_seed[record.seed] = record

    if expected is not None:
        unexpected = sorted(set(by_seed) - set(expected))
        if unexpected:
            raise Stage1AggregationError(
                f"discovered seeds not present in expected_seeds: {unexpected}")
        scheduler_unexpected = sorted(set(scheduler_records) - set(expected))
        if scheduler_unexpected:
            raise Stage1AggregationError(
                "scheduler summary contains seeds not present in expected_seeds: "
                f"{scheduler_unexpected}")
        records.extend(
            _missing_record(
                seed,
                scheduler_record=scheduler_records.get(seed),
                root=root,
            )
            for seed in expected if seed not in by_seed
        )

    for seed, record in by_seed.items():
        scheduled = scheduler_records.get(seed)
        if scheduled is None:
            continue
        scheduled_status = scheduled.get("status")
        if scheduled_status not in ("COMPLETE", "SKIPPED"):
            raise Stage1AggregationError(
                f"scheduler/seed-directory status conflict for seed {seed}: "
                f"scheduler={scheduled_status!r}, directory={record.status!r}")

    identity = _validate_common_identity(records)
    if scheduler_summary is not None:
        scheduler_fingerprint = scheduler_summary["run_fingerprint"]
        if (identity["run_fingerprint"] is not None
                and identity["run_fingerprint"] != scheduler_fingerprint):
            raise Stage1AggregationError(
                "run_summary.json run_fingerprint disagrees with seed runs: "
                f"{scheduler_fingerprint!r} != {identity['run_fingerprint']!r}")
        if identity["run_fingerprint"] is None:
            identity["run_fingerprint"] = scheduler_fingerprint
    hard_conflicts = [
        (record.seed, problem)
        for record in records
        for problem in record.problems
        if problem.startswith("conflicting ")
    ]
    if hard_conflicts:
        detail = "; ".join(
            f"seed {seed}: {problem}" for seed, problem in hard_conflicts)
        raise Stage1AggregationError(
            "inconsistent identities within seed artifacts: " + detail)
    incomplete = [record for record in records if not record.complete]
    if incomplete and not allow_incomplete:
        detail = "; ".join(
            f"seed {record.seed}: {', '.join(record.problems)}"
            for record in sorted(incomplete, key=lambda item: (item.seed is None, item.seed or 0)))
        raise Stage1AggregationError("incomplete Stage-I seed runs: " + detail)

    metric_names = _validate_metric_schema(records, required_metrics)
    completed = [record for record in records if record.complete]
    roles = dict(metric_roles or {})
    invalid_roles = {
        name: role for name, role in roles.items()
        if (not isinstance(name, str) or not name
            or not isinstance(role, str) or not role)
    }
    if invalid_roles:
        raise Stage1AggregationError(
            f"metric_roles contains invalid entries: {invalid_roles}")
    unknown_roles = sorted(set(roles) - set(metric_names))
    if unknown_roles:
        # A CUDA-only metric such as peak memory may be declared but absent in
        # a CPU smoke.  Only reject unknown roles not explicitly optional.
        unknown_required = sorted(
            set(unknown_roles) & set(required_metrics or ()))
        if unknown_required:
            raise Stage1AggregationError(
                f"metric_roles names are absent from completed runs: "
                f"{unknown_required}")
    for name in metric_names:
        if roles.get(name) != "shared_evaluation_diagnostic":
            continue
        values = np.asarray(
            [record.metrics[name] for record in completed], dtype=np.float64)
        if values.size > 1 and not np.allclose(
                values, values[0], rtol=1e-12, atol=1e-12):
            raise Stage1AggregationError(
                f"shared evaluation diagnostic {name!r} differs across "
                "training seeds")
    summaries = {
        name: _metric_summary(
            [record.metrics[name] for record in completed],
            role=roles.get(name, "training_seed_metric"))
        for name in metric_names
    }

    failure_records = [_failure_payload(record, root) for record in incomplete]
    counts = {
        "expected": len(expected) if expected is not None else len(records),
        "discovered": len(directories),
        "complete": len(completed),
        "incomplete": len(incomplete),
        "failed": sum(record.status == "FAILED" for record in incomplete),
        "cancelled": sum(record.status == "CANCELLED" for record in incomplete),
        "missing": sum(record.status == "MISSING" for record in incomplete),
    }
    payload: dict[str, Any] = {
        "schema": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "counts": counts,
        "seeds": {
            "expected": expected,
            "complete": sorted(record.seed for record in completed if record.seed is not None),
            "incomplete": sorted(record.seed for record in incomplete if record.seed is not None),
        },
        "metrics": summaries,
        "uncertainty": {
            "training_seed_axis": (
                "sample SD/SE and Student-t 95% CI across independent "
                "training seeds, conditional on the shared evaluation bank"
                if shared_evaluation_bank else
                "sample SD/SE and Student-t 95% CI across completed seed runs"
            ),
            "within_policy_paired_mc_se_metric": "dJ_se",
            "shared_evaluation_bank": bool(shared_evaluation_bank),
            "shared_diagnostic_seed_ci": "suppressed",
            "primary_training_seed_dispersion": "sample_sd",
        },
    }
    if include_failure_metadata:
        payload["failures"] = failure_records

    destination = Path(output_dir) if output_dir is not None else root
    per_seed_path = destination / "per_seed.csv"
    summary_csv_path = destination / "summary.csv"
    summary_json_path = destination / "summary.json"

    per_seed_rows = _per_seed_rows(records, identity, metric_names)
    summary_rows = [dict(metric=name, **summaries[name]) for name in metric_names]
    _atomic_write_csv(per_seed_path, [*_BASE_COLUMNS, *metric_names], per_seed_rows)
    _atomic_write_csv(
        summary_csv_path,
        [
            "metric", "role", "seed_uncertainty_applicable", "n", "mean",
            "sd", "se", "ci95_low", "ci95_high",
        ],
        summary_rows,
    )
    atomic_write_json(summary_json_path, payload)
    return Stage1AggregationResult(
        per_seed_csv=per_seed_path,
        summary_csv=summary_csv_path,
        summary_json=summary_json_path,
        payload=payload,
    )


# Short alias convenient for a CLI or scheduler callback.
aggregate = aggregate_stage1_runs


__all__ = [
    "Stage1AggregationError",
    "Stage1AggregationResult",
    "aggregate",
    "aggregate_stage1_runs",
]
