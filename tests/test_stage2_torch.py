"""Fixed-control Torch OL-BPTT harvesting contracts."""

import numpy as np
import pytest

from pgdpo_delay.core.estimators import BranchBudgets
from pgdpo_delay.core.stage2_torch import harvest_fixed_control_torch


torch = pytest.importorskip("torch")


class _ScalarAdapter:
    noise_dim = 1

    def __init__(self):
        self.device = "cpu"
        self.dtype = torch.float64
        self.cfg = {
            "N": 1,
            "h": 0.2,
            "a": 0.7,
            "b": 0.8,
            "sigma": 0.4,
            "Q": 1.3,
            "R": 0.5,
            "QT": 2.0,
        }

    def grid(self, cfg):
        return cfg["N"], cfg["h"]

    def features(self, cfg, state, k):
        return state

    def chart(self, cfg, raw):
        return raw.squeeze(-1)

    def conditional_mean(self, cfg, state, action):
        return cfg["a"] * state + cfg["b"] * action[:, None]

    def diffusion_matrix(self, cfg, state, action):
        return torch.full(
            (state.shape[0], 1, 1),
            cfg["sigma"],
            dtype=state.dtype,
            device=state.device,
        )

    def step(self, cfg, state, action, dW):
        mean = self.conditional_mean(cfg, state, action)
        return mean + self.diffusion_matrix(cfg, state, action).squeeze(-1) * dW

    def running_cost(self, cfg, state, action, k):
        return 0.5 * cfg["Q"] * state[:, 0] ** 2 \
            + 0.5 * cfg["R"] * action ** 2

    def terminal_cost(self, cfg, state):
        return 0.5 * cfg["QT"] * state[:, 0] ** 2


class _LinearActor(torch.nn.Module):
    def __init__(self, coefficient):
        super().__init__()
        self.coefficient = coefficient

    def forward(self, features):
        return self.coefficient * features


def _analytic_harvest(branch_batch_size):
    adapter = _ScalarAdapter()
    policy = _LinearActor(1.4).eval()
    budgets = BranchBudgets(
        M=2048,
        M_out=32,
        M_in=3,
        branch_batch_size=branch_batch_size,
    )
    return adapter, harvest_fixed_control_torch(
        adapter,
        policy,
        state=np.array([0.6]),
        time_index=0,
        budgets=budgets,
        seed=17,
        anchor_id="one-step-analytic",
        anchor={"u_ref_source": "frozen-actor"},
        ridge=0.0,
    )


def test_fixed_control_derivatives_detach_actor_but_keep_physical_flow():
    adapter, result = _analytic_harvest(branch_batch_size=137)
    cfg = adapter.cfg
    x = 0.6
    u_ref = 1.4 * x
    expected_p = (
        cfg["h"] * cfg["Q"] * x
        + cfg["QT"] * (cfg["a"] * x + cfg["b"] * u_ref) * cfg["a"]
    )
    expected_pi = cfg["h"] * cfg["Q"] + cfg["QT"] * cfg["a"] ** 2
    expected_q = cfg["QT"] * cfg["sigma"]

    # p alone has direct-branch MC noise; curvature and nested q are exact in
    # this one-step quadratic fixture.
    assert result.direct.p[0] == pytest.approx(expected_p, abs=0.02)
    assert result.direct.Pi[0, 0] == pytest.approx(expected_pi, abs=2e-12)
    assert result.nested.q_ols[0, 0] == pytest.approx(expected_q, abs=2e-12)
    assert result.raw.zeta[0, 0] == pytest.approx(
        expected_q - expected_pi * cfg["sigma"], abs=2e-12
    )

    # If the actor derivative had leaked into OL-BPTT, this much larger
    # closed-loop Hessian would have appeared instead.
    leaked = (
        cfg["h"] * (cfg["Q"] + cfg["R"] * 1.4 ** 2)
        + cfg["QT"] * (cfg["a"] + cfg["b"] * 1.4) ** 2
    )
    assert abs(result.direct.Pi[0, 0] - leaked) > 1.0


def test_branch_chunk_size_does_not_change_the_generated_science_bank():
    _, small = _analytic_harvest(branch_batch_size=29)
    _, large = _analytic_harvest(branch_batch_size=4096)
    np.testing.assert_allclose(small.raw.p, large.raw.p, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(small.raw.Pi, large.raw.Pi, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(small.raw.zeta, large.raw.zeta, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        small.raw.sigma_ref, large.raw.sigma_ref, atol=0.0, rtol=0.0
    )


def test_p1_torch_harvest_connects_to_common_recovery():
    from pgdpo_delay.problems.p1.config import load_config
    from pgdpo_delay.problems.p1.stage1_torch import P1Stage1Adapter
    from pgdpo_delay.problems.p1.stage2 import (
        execute_p1_stage2,
        harvest_p1_raw_torch,
        identity_audit_projectors,
    )

    class ZeroActor(torch.nn.Module):
        def forward(self, features):
            return torch.zeros(
                (features.shape[0], 1),
                dtype=features.dtype,
                device=features.device,
            )

    cfg = load_config("main_u")
    adapter = P1Stage1Adapter(cfg, device="cpu", dtype=torch.float64)
    harvested = harvest_p1_raw_torch(
        adapter,
        ZeroActor().eval(),
        state=np.zeros(cfg["H"] + 1),
        time_index=cfg["N"] - 1,
        budgets=BranchBudgets(M=8, M_out=4, M_in=2, branch_batch_size=3),
        seed=31,
        anchor_id="p1-last-step-smoke",
    )
    raw = harvested.raw
    assert raw.pi_layout == "scalar"
    assert raw.anchor.u_ref == pytest.approx(0.0)
    assert np.isfinite([raw.p, raw.zeta, raw.Pi, raw.sigma_ref]).all()

    recovered = execute_p1_stage2(
        cfg, raw, identity_audit_projectors()
    )
    assert np.isfinite(recovered.action)
    assert recovered.residual.maximum < 1e-11
    assert recovered.recovery_health["denominator_min"] > 0.0
