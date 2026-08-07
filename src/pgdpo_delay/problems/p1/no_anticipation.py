"""P1-C no-DELAYED-RE-ENTRY TANGENT ablation ("no-anticipation adjoint
ablation"; review 2026-08-07 sec.12 naming). Physical dynamics and the frozen
policy values are untouched -- ONLY the tangent re-entry derivatives are cut.
The ablated variant
computes recovery inputs with the delayed re-entry channel REMOVED from the
tangent propagation (A[0,H] = C[0,H] = 0 in tangents only; rollouts and the
frozen policy are identical, same noise budget). Recovery then clips the
generalized-Hamiltonian minimiser with the ablated (p, zeta, Pi).
"""
import numpy as np
from .evaluate import estimator_inputs

def recovered_action(cfg, inp, denom_tol=1e-10):
    p = cfg["params"]
    denom = p["R"] + p["gu"]**2*inp["Pi"]
    if not np.isfinite(denom) or denom <= denom_tol:
        raise FloatingPointError(f"recovery denominator degenerate: {denom}")
    u = -(p["b"]*inp["p"] + p["gu"]*inp["zeta"]
          + p["gu"]*inp["Pi"]*inp["sigma_bar"])/denom
    return float(np.clip(u, *cfg["bounds"]))

def compare_inputs(cfg, pol, states, M, Mout, Min, seed):
    rows = []
    for i, (k, z) in enumerate(states):
        full = estimator_inputs(cfg, pol, k, z, M, Mout, Min, seed+i, False)
        na   = estimator_inputs(cfg, pol, k, z, M, Mout, Min, seed+i, True)
        rows.append(dict(k=k, p_full=full["p"], p_na=na["p"], Pi_full=full["Pi"],
                         Pi_na=na["Pi"], z_full=full["zeta"], z_na=na["zeta"],
                         u_full=recovered_action(cfg, full),
                         u_na=recovered_action(cfg, na)))
    return rows
