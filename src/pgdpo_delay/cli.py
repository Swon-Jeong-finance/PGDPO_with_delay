"""CLI implementation (root main.py is a thin wrapper; also exposed as the
`pgdpo-delay` console script). No math, no problem-specific code."""
import argparse
from pathlib import Path
from .registry import PROBLEM_REGISTRY, api_versions
from .core import artifacts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", choices=["run", "evaluate", "report", "verify", "config"])
    ap.add_argument("--problem", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--base", default="main")
    ap.add_argument("--config", default="main")
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
            outdir = Path("outputs/verify")/n
            print(f"===== verify {n} ({'full' if a.full else 'fast'}) =====")
            PROBLEM_REGISTRY[n]["verify"](fast=not a.full, config=a.config, outdir=outdir)
            artifacts.write_manifest(outdir, problem=n, method="verify",
                                     config=dict(tier="full" if a.full else "fast",
                                                 config=a.config),
                                     api_versions=api_versions(n),
                                     solver="exact-reference")
        print("verify: ALL PASS")
    elif a.verb == "run":
        if a.set:
            raise SystemExit("run takes no --set overrides: save a config first "
                             "(python main.py config --problem ... --set ... --name NAME) "
                             "and run with --config NAME.")
        raise SystemExit(f"run --problem {a.problem} --config {a.config}: solver layer "
                         f"pending (single Stage II lives in core/stage2.py).")
    else:
        raise SystemExit(f"'{a.verb}' arrives with the solver/reporting layers.")

def derive_config(a):
    """Derive ./configs/<problem>/<name>.yaml (user space, CWD) from a base
    config (packaged canonical or another user file). Unknown keys rejected."""
    import yaml
    from importlib.resources import files
    if not (a.problem and a.name):
        raise SystemExit("config needs --problem and --name")
    local = Path("configs")/a.problem/f"{a.base}.yaml"
    if local.exists():
        cfg = yaml.safe_load(local.read_text())
    else:
        res = files("pgdpo_delay.configs").joinpath(a.problem, f"{a.base}.yaml")
        if not res.is_file(): raise SystemExit(f"unknown base config: {a.base}")
        cfg = yaml.safe_load(res.read_text())
    for kv in a.set:
        key, val = kv.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"unknown key '{key}' in {a.base}.yaml")
        node[parts[-1]] = yaml.safe_load(val)
    out = Path("configs")/a.problem/f"{a.name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists (use --force to overwrite)")
    out.write_text(f"# derived from {a.base}.yaml; overrides: {a.set or 'none'}\n"
                   + yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out}")
