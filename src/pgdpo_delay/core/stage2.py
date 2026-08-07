"""Stage II: SINGLE implementation contract (solver layer, torch port pending).

The pipeline ORDER is owned here and is identical for every problem
(manuscript Stage-II convention, corrected 2026-08-07 review):

    1. OL-BPTT/MC harvest of RAW inputs  (p^, Pi^, zeta^)
       -- zeta^ comes FIRST, from the anchored nested antithetic CRN
          regression at sigma_ref; it is a harvest output, not a
          post-projection step.
    2. q^_anc = zeta^ + Pi^ sigma_ref          (reconstruction BEFORE projection)
    3. (p^N, q^anc,N, Pi^N) = P(p^, q^_anc, sym(Pi^))   (blockwise projection)
    4. zeta^N = q^anc,N - Pi^N sigma_ref       (ALGEBRAIC re-coordinatisation;
                                                never re-estimated after P)
    5. problem-local recovery solve            (clip / QP / bounded nonlinear)
    6. r_num logging                            (KKT residual of step 5).

Problems inject ONLY mathematical parts via the registry interface:
    simulate(...), running_cost(...), terminal_cost(...),
    local_recovery(...), reference(...)
and MUST NOT reimplement the loop: divergence between problems would break
the paper's "identical Stage II" claim. The denominator/positivity of the
projected Pi^N is checked here (guard, not in problem code).
"""
from dataclasses import dataclass
from typing import Any

@dataclass
class RawRecoveryInputs:
    p: Any; zeta: Any; Pi: Any; sigma_ref: Any

@dataclass
class ProjectedRecoveryInputs:
    p: Any; q_anc: Any; Pi: Any; zeta: Any

def prepare_inputs(raw: RawRecoveryInputs, projection_blocks):   # pragma: no cover
    """Steps 2-4 of the pipeline (torch implementation pending). Kept here so
    the order is code, not prose: q_anc BEFORE projection, zeta AFTER it by
    algebra only."""
    raise NotImplementedError("Stage II torch implementation is the next solver-layer task")

def run_stage2(problem, config, seed):        # pragma: no cover - solver pending
    raise NotImplementedError("Stage II torch implementation is the next solver-layer task")
