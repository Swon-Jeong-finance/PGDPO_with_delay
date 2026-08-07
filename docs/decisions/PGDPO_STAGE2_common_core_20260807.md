# Stage-II common core implementation boundary (2026-08-07)

## Frozen order

The single implementation in `core/stage2.py` executes

1. accept raw `p_cur`, `zeta`, `Pi`, and `sigma_ref` from one state/control
   anchor;
2. form `q_anc = zeta + Pi @ sigma_ref` before any projection;
3. project `p`, `q_anc`, and `sym(Pi)` with three independent callbacks;
4. reconstruct `zeta_N = q_anc_N - Pi_N @ sigma_ref` algebraically;
5. call the problem-local recovery solver;
6. report feasibility and the projected-objective normal-cone residual.

State/history, `u_ref`, and `sigma_ref` are carried through unchanged. There
is no `zeta` projector and a joint tuple projector is rejected. The objective
sense is mandatory because the manuscript is Hamiltonian maximization while
Appendix-C benchmark code uses cost minimization.

## Implemented components

- NumPy/Torch-preserving scalar and matrix recovery inputs
- independent block-projection diagnostics
- box and unconstrained feasibility/normal-cone geometry
- raw OL-BPTT sample reduction and anchored nested antithetic CRN regression
- Torch fixed-control OL-BPTT branch harvesting: actor values are recomputed
  at each branch state and detached, while physical/delay re-entry derivatives
  remain live; complete RNG banks are generated before memory chunking
- scientific branch budgets separated from GPU chunk size
- P1 `p_cur` recovery, with `p_nxt` retained only as a finite-grid diagnostic
- P1-U unconstrained and P1-C exact scalar-box recovery
- explicit P1 identity-audit projectors while numerical projection sets remain
  unfrozen
- P1 Stage-I/Stage-II/reference JSON+CSV output contract

The nested regression uses averaged outer moments
`G_N = mean(d d^T)` and `S_N = mean(d y^T)`. Thus a declared nonzero ridge is
independent of `M_out`; when the ridge is zero,
`zeta + Pi @ sigma_ref` equals the unregularized OLS `q` samplewise.

## Deliberately still pending

This patch does not claim a runnable P1 Stage-II experiment. The following
must be connected after the Stage-I checkpoint is available:

- checkpoint loader/state-bank loop that invokes the harvester along deployed
  histories
- frozen numerical projection sets (or an explicitly reported identity-audit
  experiment); no radii were invented here
- independent holdout KKT bank and paired Stage-I/Stage-II rollout evaluation
- Stage-II checkpoint worker, multi-GPU scheduler protocol, CLI, and immutable
  per-seed artifacts

The current P1-U Stage-I paired rollout size (`Np=50000`) is not a Stage-II
branch count.
`M`, `M_out`, and `M_in` are statistical budgets; `branch_batch_size` is only
a memory/chunking setting.
