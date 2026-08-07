"""Run manifest and artifact conventions. EVERY run/verify writes a manifest:
config hash, git commit, problem/method, seeds (train/eval/Brownian bank),
device, API versions, exact/inexact solver flag."""
import hashlib, json, subprocess, time
from pathlib import Path

def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def write_manifest(outdir, *, problem, method, config, seeds=None, device="cpu",
                   api_versions=None, solver="exact", extra=None):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    man = dict(problem=problem, method=method, config_hash=config_hash(config),
               git_commit=git_commit(), seeds=seeds or {}, device=device,
               api_versions=api_versions or {}, solver=solver,
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), extra=extra or {})
    with open(outdir/"manifest.json", "w") as fp:
        json.dump(man, fp, indent=1)
    with open(outdir/"config.json", "w") as fp:
        json.dump(config, fp, indent=1, default=str)
    return man
