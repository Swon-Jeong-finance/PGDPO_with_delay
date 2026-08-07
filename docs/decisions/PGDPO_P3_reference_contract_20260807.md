# PGDPO P3 reference/benchmark code contract (2026-08-07)

## 1. Frozen scope

This code track finishes the non-deep-learning part of P3 before Stage I/II.

- **P3-R:** a controlled-diffusion 2D HJB **numerical reference** on the
  renewal lift `(I,M)`. It is not called an exact oracle.
- **P3-D:** one truncated-Gamma **CE-NMPC proxy benchmark** on the full history
  buffer. It is not a stochastic oracle/reference.
- The renewal CE-NMPC versus HJB calculation is an internal proxy sanity check,
  not a direct P3-R/P3-D experiment.

The following are outside the retained experiment scope: direct P3-R/P3-D
comparison, initial-law matching experiment, no-memory/frozen-memory
ablations, a separate matched-kernel experiment, local perturbation, a full
lookahead/iteration budget frontier, and a scenario-count sweep. Scenario-NMPC
is only a contingency if the frozen CE-NMPC is later demonstrably inadequate.

## 2. Mathematical and numerical contract

P3-R uses the state lift

```text
dI = [beta (1-I/Npop) M - gamma I - b u I] dt
     + sigma0 (1-eta_sigma u) I dW,
dM = rho (I-M) dt,                         u in [0,1].
```

The HJB control minimization includes the controlled-diffusion term
`0.5*sigma0^2*(1-eta_sigma*u)^2*I^2*V_II`. The discrete solver minimizes the
full branch-aware upwind Hamiltonian, including the drift-sign switch and both
box endpoints. The shared stochastic evaluator uses full-truncation Euler.

The Bellman audit uses

```text
D_k = V_k(X_k) - dt*ell(X_k,u_k) - V_{k+1}(X_{k+1}).
```

At the final transition, `V_N(X_N)` is evaluated as the realized exact
quadratic `0.5*c_T*max(I_N,0)^2`; it is not interpolated from the terminal
table. The headline total and its SE are computed from the pathwise cumulative
sum `sum_k D_k`, so the statistic telescopes pathwise.

P3-D uses normalized trapezoidal weights for the truncated-Gamma kernel on
`j=0,...,H`, with the exact kernel configuration checked in tests. CE-NMPC
sets future noise to zero in its deterministic prediction problem. Its frozen
YAML knobs are lookahead 25, maximum 40 L-BFGS-B iterations, `gtol=1e-8`, and
`ftol=1e-12`. Before a prediction window reaches physical `T`, the same
quadratic endpoint shape is explicitly an MPC terminal surrogate
(`terminal_mode: quadratic_proxy`).

The optimizer diagnostic is named precisely
**CE deterministic open-loop subproblem box-KKT**:

```text
G(u) = u - projection_[0,1](u - grad Phi_CE(u)),
r_CE-KKT = ||G(u)||_infinity.
```

This is not the stochastic generalized-Hamiltonian KKT used later for learned
methods, and it does not establish global optimality of the CE subproblem.
Raw optimizer bounds, deployment clipping, deployed bounds, optimizer health,
and runtime are recorded separately. `optimizer_runtime_*` stops when
L-BFGS-B returns, whereas the paper-facing `decision_runtime_*` also includes
the final objective/gradient evaluation, projected KKT, clipping, warm-start
update, and telemetry. Accordingly, `optimizer_nfev` is complemented by
`total_objective_grad_evals = optimizer_nfev + 1`.

The global curvature diagnostic covers every pre-update HJB solver surface
and the final stored `t=0` surface. Its scope and surface count are persisted
with `A2_min_global` and the `V_II` range; this diagnostic does not alter the
HJB recursion, policy, or value.

## 3. Artifact lifecycle

P3 verify output is transactionally published:

```text
outputs/verify/p3/
  bundles/full/<bundle-id>/...
  bundles/fast/<bundle-id>/...
  current-full.json
  current-fast.json
```

A run builds in a same-filesystem private staging directory. Only after every
gate and required file succeeds is the complete directory moved to a new
immutable bundle and the small tier pointer atomically replaced. Failed full
runs and fast runs cannot overwrite the current full reference.

The current full contract requires:

- `manifest.json`, `config.json`
- `p3r_hjb_value.npz`, `p3r_hjb_policy.npz`
- `p3r_hjb_residual.csv`, `p3r_certification.csv`
- `p3d_ce_nmpc_certification.csv`

## 4. Frozen verification result

Command:

```bash
PYTHONPATH=src python main.py verify --problem p3 --full
```

Result: PASS under API `p3-oracle-v3-reference-contract`.

| Gate/metric | Frozen value |
|---|---:|
| P3-R grid value relative difference | 0.00111791 |
| P3-R grid policy RMSE, max over audited times | 0.00518714 |
| P3-R domain policy RMSE, max | 8.03756e-06 |
| P3-R stored-policy re-minimization residual | 1.66315e-08 |
| P3-R Bellman cumulative residual | -8.19432e-05 |
| P3-R Bellman cumulative SE | 0.00218448 |
| P3-D CE adjoint-vs-finite-difference relative gap | 2.50569e-08 |
| P3-D CE holdout solves / success | 800 / 100% |
| P3-D CE holdout maximum projected KKT | 1.53990e-07 |
| P3-D raw/deployed maximum bound violation | 0 / 0 |

The test suite result in the non-Torch environment was `40 passed, 2 skipped`;
the skipped tests are Stage-I/Torch tests outside this patch.

## 5. Provenance and next boundary

The parent uploaded snapshot SHA-256 is
`03d921a2d6e7ba4d13909ce04f77c19289f586135f2951650731ae2d8e6e17a3`.
It was an archive rather than a Git checkout, so its generated manifest records
the Git commit honestly as `unknown`; the delivered snapshot hash provides the
archive-level provenance.

Appendix C remains a manuscript draft, not a source of truth that silently
overrides checked code. Any later equation change must be reconciled against
the shared dynamics, HJB operator, CE objective/adjoint, and the regression
tests above. Stage I/II and learned-method comparisons are intentionally not
modified here. When those experiments begin, the retained P3-D main outputs
are common-noise paired objective with CI, stochastic method KKT, constraint
satisfaction, and runtime against this fixed CE-NMPC proxy.
