"""CLI implementation (root main.py is a thin wrapper).

Reference verification remains registry-driven.  Stage-I learned-policy
runs use the common subprocess scheduler so each training seed owns an
isolated CUDA process and the first free device receives the next seed.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
from .registry import PROBLEM_REGISTRY, api_versions, evaluation_conventions
from .core import artifacts


_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CUDA_DEVICE = re.compile(r"^cuda:[0-9]+$")
_STAGE1_REQUIRED_ARTIFACTS = (
    "stage1_state.pt",
    "stage1_spec.json",
    "training_trace.npz",
)


def _csv_seeds(value):
    if value is None or not value.strip():
        raise SystemExit("Stage-I run requires --seeds, e.g. --seeds 1,2,3")
    try:
        seeds = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise SystemExit("--seeds must be comma-separated nonnegative integers") \
            from exc
    if any(seed < 0 for seed in seeds):
        raise SystemExit("--seeds must be nonnegative")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds contains duplicates")
    return seeds


def _csv_devices(value):
    if value is None or not value.strip():
        raise SystemExit("Stage-I run requires at least one --devices entry")
    raw_devices = [item.strip() for item in value.split(",")]
    if any(not item for item in raw_devices):
        raise SystemExit("--devices contains an empty entry")
    invalid = [item for item in raw_devices
               if item != "cpu" and not _CUDA_DEVICE.fullmatch(item)]
    if invalid:
        raise SystemExit(
            "--devices entries must be cpu or explicitly indexed cuda:N; "
            "invalid: "
            f"{invalid}")
    # Canonicalize aliases such as cuda:00 before checking uniqueness; two
    # subprocesses must never be assigned to the same physical slot merely
    # because its index was formatted differently.
    devices = []
    for item in raw_devices:
        if item.startswith("cuda:"):
            item = f"cuda:{int(item.split(':', 1)[1])}"
        devices.append(item)
    if len(set(devices)) != len(devices):
        raise SystemExit("--devices contains duplicates")
    kinds = {"cpu" if item == "cpu" else "cuda" for item in devices}
    if len(kinds) > 1:
        raise SystemExit(
            "--devices may not mix CPU and CUDA slots in one run")
    return devices


def _default_stage1_protocol(problem, problem_config):
    if problem == "p1" and problem_config in (None, "main_u"):
        return "p1_u"
    raise SystemExit(
        "no default Stage-I protocol for this problem/config; pass "
        "--protocol explicitly")


def _stage1_run_root(out_root, problem, protocol_name, run_name):
    name = run_name or Path(protocol_name).stem
    if not _SAFE_RUN_NAME.fullmatch(name):
        raise SystemExit(
            "--run-name must contain only letters, digits, '.', '_' or '-' "
            "and may not start with punctuation")
    return Path(out_root) / problem / "stage1" / name


def _ensure_run_spec(run_root, spec, fingerprint):
    """Freeze the seed-independent identity without overwriting a mismatch."""
    path = run_root / "run_spec.json"
    payload = {"schema": 1, "run_fingerprint": fingerprint, "spec": spec}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid existing run specification {path}: {exc}") \
                from exc
        if existing != payload:
            raise SystemExit(
                f"{path} belongs to a different Stage-I protocol; choose a "
                "new --run-name rather than mixing runs")
        return path
    run_root.mkdir(parents=True, exist_ok=True)
    artifacts.atomic_write_json(path, payload)
    return path


def _run_stage1(a):
    from .core.runner import run_seed_processes
    from .core.stage1_run import (
        canonical_run_spec,
        metric_roles_for_spec,
        required_metrics_for_spec,
        run_fingerprint,
    )
    from .reporting.stage1_aggregate import aggregate_stage1_runs

    if a.problem is None:
        raise SystemExit("Stage-I run requires --problem")
    if str(a.stage) != "1":
        raise SystemExit("only Stage I is implemented; Stage II remains pending")
    seeds = _csv_seeds(a.seeds)
    devices = _csv_devices(a.devices)
    protocol_name = a.protocol or _default_stage1_protocol(
        a.problem, a.config)
    spec = canonical_run_spec(a.problem, protocol_name)
    expected_problem_config = spec["problem_config"]["name"]
    if a.config is not None and a.config != expected_problem_config:
        raise SystemExit(
            f"--config {a.config!r} disagrees with protocol "
            f"{protocol_name!r}, which freezes {expected_problem_config!r}")
    fingerprint = run_fingerprint(spec)
    run_root = _stage1_run_root(
        a.out_root, a.problem, protocol_name, a.run_name)

    plan = {
        "problem": a.problem,
        "stage": 1,
        "protocol": protocol_name,
        "problem_config": expected_problem_config,
        "run_fingerprint": fingerprint,
        "seeds": seeds,
        "devices": devices,
        "run_root": str(run_root.resolve()),
        "resume": bool(a.resume),
    }
    print(json.dumps(plan, indent=2))
    if a.dry_run:
        print("Stage-I dry run: no subprocesses or artifacts were created.")
        return plan

    _ensure_run_spec(run_root, spec, fingerprint)

    def command_builder(seed, device, staging_dir):
        print(f"[stage1 scheduler] seed {seed} -> {device}", flush=True)
        return [
            sys.executable,
            "-m",
            "pgdpo_delay.core.stage1_run",
            "--problem", a.problem,
            "--protocol", protocol_name,
            "--seed", str(seed),
            "--device", device,
            "--outdir", str(staging_dir),
            "--run-fingerprint", fingerprint,
        ]

    # The root main.py bootstrap modifies sys.path but not the environment
    # inherited by a new interpreter.  Add the package's source parent so the
    # worker works both from an editable install and directly from this tree.
    child_env = os.environ.copy()
    source_parent = str(Path(__file__).resolve().parents[1])
    old_pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = source_parent if not old_pythonpath else \
        os.pathsep.join((source_parent, old_pythonpath))
    child_env["PYTHONUNBUFFERED"] = "1"

    scheduler_summary = run_seed_processes(
        command_builder,
        seeds=seeds,
        devices=devices,
        outroot=run_root,
        run_fingerprint=fingerprint,
        resume=a.resume,
        required_files=_STAGE1_REQUIRED_ARTIFACTS,
        poll_interval=a.poll_interval,
        env=child_env,
        live_output=True,
    )
    complete = scheduler_summary["status"] == "COMPLETE"
    aggregate = aggregate_stage1_runs(
        run_root,
        expected_seeds=seeds,
        allow_incomplete=not complete,
        required_metrics=required_metrics_for_spec(spec),
        metric_roles=metric_roles_for_spec(spec),
        shared_evaluation_bank=True,
    )
    print(f"Stage-I per-seed results: {aggregate.per_seed_csv}")
    print(f"Stage-I seed summary:     {aggregate.summary_csv}")
    print(f"Stage-I machine summary:  {aggregate.summary_json}")
    if not complete:
        raise SystemExit(
            "one or more Stage-I seeds failed or were cancelled; inspect "
            f"{run_root / 'run_summary.json'} and {run_root / 'failed'}")
    print("Stage-I run: ALL SEEDS COMPLETE")
    return scheduler_summary


def _verify_one(name, *, full, config, output_root=Path("outputs/verify")):
    """Run one verifier and write its manifest.

    P3 and P4 own publication-grade multi-file reference artifacts and run
    inside immutable bundle transactions.  P1/P2 keep their existing
    lightweight output convention.
    """
    root = Path(output_root)/name
    transactional = name in ("p3", "p4")
    tx = artifacts.begin_bundle(root, "full" if full else "fast") \
        if transactional else None
    outdir = tx.stage_dir if tx is not None else root
    try:
        res = PROBLEM_REGISTRY[name]["verify"](
            fast=not full, config=config, outdir=outdir) or {}
        extra = dict(evaluation_conventions=evaluation_conventions(name),
                     **res.get("extra", {}))
        if tx is not None:
            extra["artifact_bundle"] = dict(
                schema=1, tier=tx.tier, bundle_id=tx.bundle_id)
        artifacts.write_manifest(
            outdir, problem=name, method="verify",
            config=res.get("config_snapshot",
                           dict(tier="full" if full else "fast",
                                config=config)),
            seeds=res.get("seeds"),
            api_versions=api_versions(name),
            solver=res.get("solver", "exact-reference"),
            extra=extra)
        if tx is not None:
            required = res.get("required_artifacts")
            if required is None:
                raise RuntimeError(
                    f"{name.upper()} verifier omitted its artifact contract"
                )
            if not full:
                forbidden = (("p3r_*", "p3d_*") if name == "p3"
                             else ("p4_*",))
            else:
                forbidden = ()
            tx.publish(required, forbidden_globs=forbidden)
        return res
    except BaseException:
        if tx is not None:
            tx.abort()
        raise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", choices=["run", "evaluate", "report", "verify", "config"])
    ap.add_argument("--problem", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--stage", default="1")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--devices", default="cuda:0")
    ap.add_argument("--out-root", default="outputs/runs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll-interval", type=float, default=0.25)
    ap.add_argument("--base", default="main")
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.verb == "config":
        return derive_config(a)
    if a.verb == "verify":
        names = list(PROBLEM_REGISTRY) if a.all else [a.problem]
        if not all(n in PROBLEM_REGISTRY for n in names):
            raise SystemExit(f"unknown problem in {names}")
        for n in names:
            print(f"===== verify {n} ({'full' if a.full else 'fast'}) =====")
            _verify_one(n, full=a.full, config=a.config or "main")
        print("verify: ALL PASS")
    elif a.verb == "run":
        if a.set:
            raise SystemExit("run takes no --set overrides: save a config first "
                             "(python main.py config --problem ... --set ... --name NAME) "
                             "and run with --config NAME.")
        return _run_stage1(a)
    else:
        raise SystemExit(f"'{a.verb}' arrives with the solver/reporting layers.")

def derive_config(a):
    """Derive ./configs/<problem>/<name>.yaml (user space, CWD) from a base
    config (packaged canonical or another user file). Unknown keys rejected."""
    import yaml
    from importlib.resources import files
    if not (a.problem and a.name):
        raise SystemExit("config needs --problem and --name")
    # base resolution mirrors the loaders (review 2026-08-07 minfix sec.4.2/4.3):
    # canonical bases come ONLY from the package; a differing CWD copy of a
    # canonical name is refused so stale numbers cannot leak into derived files.
    local = Path("configs")/a.problem/f"{a.base}.yaml"
    res = files("pgdpo_delay.configs").joinpath(a.problem, f"{a.base}.yaml")
    if res.is_file():
        text = res.read_text()
        if local.exists() and local.read_text() != text:
            raise SystemExit(
                f"{local} shadows canonical base {a.problem}/{a.base}.yaml "
                f"with different content; remove it or choose a non-canonical "
                f"base.")
        cfg = yaml.safe_load(text)
    elif local.exists():
        cfg = yaml.safe_load(local.read_text())
    else:
        raise SystemExit(f"unknown base config: {a.base}")
    for kv in a.set:
        key, val = kv.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"unknown key '{key}' in {a.base}.yaml")
        node[parts[-1]] = yaml.safe_load(val)
    if files("pgdpo_delay.configs").joinpath(a.problem, f"{a.name}.yaml").is_file():
        raise SystemExit(f"--name {a.name} would shadow a canonical packaged "
                         f"config; pick a different name (canonical yaml lives "
                         f"only inside the package).")
    out = Path("configs")/a.problem/f"{a.name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists (use --force to overwrite)")
    out.write_text(f"# derived from {a.base}.yaml; overrides: {a.set or 'none'}\n"
                   + yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out}")
