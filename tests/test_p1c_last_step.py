import numpy as np

def test_exact_last_step_matches_tiny_dp():
    from pgdpo_delay.problems.p1.config import load_config
    from pgdpo_delay.problems.p1 import dp_small
    cfg = load_config("dp_small")
    dp = dp_small.dp_reference(cfg, n_x=7, n_gh=3, n_u=9, L=2.0)
    kN = cfg["N"] - 1
    errs = []
    for x0 in dp["xg"]:
        for xH in dp["xg"]:
            z = np.array([x0, 0.0, 0.0, xH])
            errs.append(dp_small.dp_action_label_at(dp, kN, z)
                        - dp_small.exact_last_step_action(cfg, x0, xH))
    assert np.sqrt(np.mean(np.square(errs))) < 1e-6
