"""Seed-parallel run orchestration (skeleton).
Discipline: run(config, seed) is pure w.r.t. disk artifacts; every run writes
a manifest (core.artifacts) and raw npz/csv; aggregation and figures NEVER
retrain. RNG streams are separated (model / history / noise) and paired-dJ
evaluations share Brownian banks across policies within a seed."""
from . import artifacts

def run_seeds(problem_name, run_fn, config, seeds, outroot):
    results = []
    for s in seeds:
        outdir = f"{outroot}/{problem_name}/seed{s}"
        artifacts.write_manifest(outdir, problem=problem_name, method="run",
                                 config=config, seeds=dict(train=s))
        results.append(run_fn(config, s, outdir))
    return results
