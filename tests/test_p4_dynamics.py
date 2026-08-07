"""Focused tests for the P4 signed causal-simulator contract."""

import numpy as np

from pgdpo_delay.problems.p4.config import load_config
from pgdpo_delay.problems.p4 import dynamics


def test_p4_config_and_linear_matrix_contract():
    cfg = load_config("main")
    assert cfg["control_kind"] == "signed"
    assert cfg["bounds"] is None
    assert cfg["noise"]["correlation"] == 0.0
    assert cfg["state_dim"] == cfg["H"] + 2

    A, B, Dq, Salpha = dynamics.linear_matrices(cfg)
    n = cfg["state_dim"]
    assert A.shape == (n, n)
    assert all(v.shape == (n,) for v in (B, Dq, Salpha))
    assert B[0] == -cfg["h"]
    assert Dq[0] == -cfg["params"]["sigma_Q"]
    assert Salpha[-1] == cfg["params"]["sigma_alpha"]


def test_p4_eq78_row_matches_direct_realized_fill_sum():
    cfg = load_config("main")
    bank = dynamics.make_bank(cfg, Np=7, seed=41)
    policy = lambda k, Z: 0.7 + 0.03 * k + 0.2 * Z[:, -1]
    out = dynamics.simulate_from_bank(cfg, policy, bank, return_paths=True)
    paths = out["paths"]

    # The prescribed constant q0 prehistory makes I_0 exactly zero.
    np.testing.assert_allclose(paths["impact"][0], np.zeros(7),
                               atol=1.0e-15, rtol=0.0)
    assert abs(dynamics.impact_row(cfg).sum()) < 1.0e-15

    ages = np.arange(1, cfg["H"] + 1) * cfg["h"]
    G = cfg["params"]["gamma"] * np.exp(-cfg["params"]["rho_G"] * ages)
    for k in range(cfg["N"]):
        direct = np.zeros(7)
        for j in range(1, min(cfg["H"], k) + 1):
            direct += G[j - 1] * paths["realized_fill"][k - j]
        np.testing.assert_allclose(paths["impact"][k], direct,
                                   atol=2.0e-15, rtol=0.0)

    # Realized fills and inventory use exactly the same stochastic increment.
    np.testing.assert_allclose(
        paths["Z"][1:, :, 0],
        paths["Z"][:-1, :, 0] - paths["realized_fill"],
        atol=2.0e-15,
        rtol=0.0,
    )


def test_p4_signed_control_is_not_clipped_and_self_pair_is_zero():
    cfg = load_config("main")
    signed = lambda k, Z: np.full(Z.shape[0], -2.75 if k == 0 else 3.25)
    out = dynamics.simulate(cfg, signed, Np=5, seed=9, return_paths=True)
    assert np.all(out["paths"]["u"][0] == -2.75)
    assert np.all(out["paths"]["u"][1:] == 3.25)

    paired = dynamics.simulate_paired(cfg, signed, signed, Np=16, seed=91)
    assert paired["delta_A_minus_B"] == 0.0
    assert paired["se"] == 0.0


def test_p4_common_noise_bank_has_independent_reproducible_channels():
    cfg = load_config("main")
    a = dynamics.make_bank(cfg, Np=4096, seed=117)
    b = dynamics.make_bank(cfg, Np=4096, seed=117)
    assert np.array_equal(a["Z0"], b["Z0"])
    assert np.array_equal(a["dW_Q"], b["dW_Q"])
    assert np.array_equal(a["dW_alpha"], b["dW_alpha"])
    assert not np.array_equal(a["dW_Q"], a["dW_alpha"])
    correlation = np.corrcoef(a["dW_Q"].ravel(),
                              a["dW_alpha"].ravel())[0, 1]
    assert abs(correlation) < 0.01
