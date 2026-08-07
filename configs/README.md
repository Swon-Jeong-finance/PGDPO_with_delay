# configs/ — user-derived space ONLY

Canonical (paper-number) YAML lives exclusively inside the package:
`src/pgdpo_delay/configs/<problem>/*.yaml`. Do NOT copy canonical files here:
a differing copy of a canonical name is refused by the loaders
(review 2026-08-07 sec.4.9), and `main.py config` refuses `--name` values
that collide with canonical names.

Derived configs are written here by:
    python main.py config --problem p1 --base main --name <new> --set k=v

Canonical Stage-I experiment protocols live in
`src/pgdpo_delay/configs/stage1/*.yaml`.  The production P1-U protocol is
`p1_u`; `p1_u_smoke` checks execution and artifacts only and is never a paper
result.  `p1_u` uses protocol schema 2 and freezes `batch=1024`, `lr=5e-5`,
`hidden=256`, `num_layers=2`, `val_batch=1024`, evaluation `Np=50000`, and
policy-forward `batch_size=4096`; edit the canonical YAML only when creating a
new, intentionally fingerprint-incompatible protocol.  Stage-I CLI mirrors
seed logs live with terminal-only seed/device prefixes while preserving each
seed's original `run.log`.  See
`docs/decisions/PGDPO_STAGE1_MULTIGPU_RUNNER_20260807.md` and
`docs/decisions/PGDPO_P1_STAGE1_feature_model_v2_20260807.md`.
