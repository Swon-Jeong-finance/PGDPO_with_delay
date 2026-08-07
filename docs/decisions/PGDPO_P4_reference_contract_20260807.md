# P4 signed-reference contract (2026-08-07)

This is the code-side source of truth for the non-deep P4 reference.  It does
not make the current manuscript draft canonical, and it does not include
NMPC, a constrained variant, or learned-method experiments.

## Frozen discrete problem

The state is `Z_k=(Q_k,Q_{k-1},...,Q_{k-H},alpha_k)`.  The shared Euler step is

```
Z_{k+1} = A Z_k + B u_k + D_Q u_k dW^Q_k + S_alpha dW^alpha_k,
B_Q=-h,  (D_Q)_Q=-sigma_Q,
```

with independent increments of variance `h`.  Pre-trade realised-fill impact
uses no current action:

```
I_k = ell Z_k,
ell = (-G_1, G_1-G_2, ..., G_{H-1}-G_H, G_H),
G_j = gamma exp(-rho_G jh).
```

There is no extra factor of `h` in `ell`: realised fills already equal
`Q_j-Q_{j+1}`.  The cost is exactly

```
sum_k h [phi Q_k^2/2 + eta u_k^2/2 + u_k I_k - alpha_k u_k]
+ kappa Q_N^2/2.
```

The packaged `configs/p4/main.yaml` is the sole numerical source.  V1 fixes
`T=1`, `h=.01`, `delta=.2`, `q0=1`, `sigma_Q=.1`, `kappa_alpha=2`,
`sigma_alpha=.15`, `gamma=.75`, `rho_G=15`, `phi=eta=.1`, `kappa=10`, and
deterministic `alpha0=0`.

## Two distinct curvatures

For `Nstage=h(ell,-1)`, the exact Bellman recursion uses

```
Lambda = h eta + B' Pnext B + h D_Q' Pnext D_Q,
K      = Nstage + A' Pnext B,
Pval   = h phi e_Q e_Q' + A' Pnext A - K K'/Lambda,
F      = -K/Lambda.
```

Independent additive signal noise contributes
`c_k-c_{k+1}=h S_alpha'Pnext S_alpha/2`.  The detached fixed-control curvature
is instead

```
Gol_k = h phi e_Q e_Q' + A' Gol_{k+1} A,
Pi_k  = (Gol_k)_{QQ}.
```

`Pval` must not be used as `Pi`.  The two separately certified denominators
are `Lambda/h` and `eta+sigma_Q^2 Pi`.

The exact Euler FOC uses `p_nxt`, not `p_cur`:

```
eta u + I - alpha - p_nxt - sigma_Q q_QQ = 0,
zeta_QQ = q_QQ - Pi(-sigma_Q u),
u = (alpha+p_nxt+sigma_Q zeta_QQ-I)/(eta+sigma_Q^2 Pi).
```

Both gradients are stored under distinct names in the diagnostic artifact.
The artifact also stores explicit `u_rec_pnxt` and `u_rec_pcur` arrays.  The
former recovers the same-grid oracle exactly; the latter holds exact
`(q,Pi,zeta)` fixed and substitutes the manuscript current costate.  Thus

```
u_rec_pcur-u = (p_cur-p_nxt)/(eta+sigma_Q^2 Pi).
```

Its large holdout-bank RMSE/nRMSE is certified in `p4_certification.csv`, the
manifest, and `p4_feasibility.json` as a finite-grid alignment floor, not an
estimator, PGDPO, or oracle error.  The small diagnostic NPZ stores pathwise
arrays and labels its own scalar summary with a `diagnostic_bank_` prefix.  The
alignment is diagnostic only and is not a reference pass/fail gate.

## Certification and scope

`python main.py verify --problem p4 --full` publishes an immutable bundle only
after causal-impact, dense/Bellman/FOC/recovery, curvature, MC-value,
terminal-inventory, overshoot, intended-round-trip, kernel materiality, and
signal/history-response gates pass.  The bundle contains:

- `p4_oracle.npz`
- `p4_certification.csv`
- `p4_feasibility.json`
- `p4_diagnostic_paths.npz`
- `config.json` and `manifest.json`

The exact reference is frozen.  P4's eventual paper inclusion remains gated
on later learned-method results; no P4 NMPC is required for that reference.
