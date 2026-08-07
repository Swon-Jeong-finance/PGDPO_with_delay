"""Stage II: SINGLE implementation contract (solver layer, torch port pending).

The pipeline ORDER is owned here and is identical for every problem:

    harvest (p^, Pi^)  ->  q^_anc reconstruction (BEFORE any projection)
    ->  blockwise projection P  ->  zeta^ nested reconstruction
    ->  problem-local recovery solve  ->  r_num logging.

Problems inject ONLY mathematical parts via the registry interface:
    simulate(...), running_cost(...), terminal_cost(...),
    local_recovery(...)   # clip / QP / nonlinear bounded solve
    reference(...)
Problem classes MUST NOT reimplement the loop; divergence between problems
would silently break the paper's "identical Stage II" claim.
"""
def run_stage2(problem, config, seed):        # pragma: no cover - solver pending
    raise NotImplementedError("Stage II torch implementation is the next solver-layer task")
