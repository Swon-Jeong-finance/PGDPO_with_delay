#!/usr/bin/env python3
"""Thin wrapper around pgdpo_delay.cli:main (also installed as `pgdpo-delay`).

  python main.py verify   --problem p1|p2 | --all   [--full]
  python main.py config   --problem p1 --base main --name main_gu05 --set problem.gamma_u=0.5
  python main.py run      --problem p1 --seeds 1,2,3 --config main_gu05
"""
import sys
from pathlib import Path
try:
    from pgdpo_delay.cli import main                 # pip install -e .
except ImportError:                                  # bootstrap convenience only
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from pgdpo_delay.cli import main

if __name__ == "__main__":
    main()
