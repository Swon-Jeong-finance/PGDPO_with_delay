#!/usr/bin/env python3
"""pgdpo-delay orchestrator (thin CLI; no math, no problem-specific code).

  python main.py verify --problem p1|p2 | --all   [--full]
  python main.py run      --problem p1 --seeds 1,2,3     (solver layer pending)
  python main.py evaluate --problem p1                    (solver layer pending)
  python main.py report   --all                           (reporting layer pending)
"""
import argparse, sys
from pathlib import Path
try:
    import pgdpo_delay                                   # pip install -e .
except ImportError:                                      # bootstrap convenience only
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    import pgdpo_delay
from pgdpo_delay.registry import PROBLEM_REGISTRY
from pgdpo_delay.core import artifacts

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", choices=["run", "evaluate", "report", "verify"])
    ap.add_argument("--problem", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seeds", default=None)
    a = ap.parse_args()
    if a.verb == "verify":
        names = list(PROBLEM_REGISTRY) if a.all else [a.problem]
        assert all(n in PROBLEM_REGISTRY for n in names), f"unknown problem in {names}"
        for n in names:
            print(f"===== verify {n} ({'full' if a.full else 'fast'}) =====")
            PROBLEM_REGISTRY[n]["verify"](fast=not a.full)
            artifacts.write_manifest(f"outputs/verify/{n}", problem=n, method="verify",
                                     config=dict(tier="full" if a.full else "fast"),
                                     solver="exact-reference")
        print("verify: ALL PASS")
    else:
        raise SystemExit(f"'{a.verb}' arrives with the solver/reporting layers "
                         f"(single Stage II lives in core/stage2.py).")

if __name__ == "__main__":
    cli()
