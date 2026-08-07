"""Torch fixed-control OL-BPTT harvester for the common Stage-II core.

The frozen actor is re-evaluated on every branch state, but its returned
action is detached before the simulator/cost graph is differentiated.  The
physical state graph, including every delay-buffer shift and re-entry path,
remains differentiable.  This is policy evaluation at fixed future controls,
not closed-loop actor differentiation.

The harvester operates at one observed history/time anchor.  Statistical
branch counts come from :class:`core.estimators.BranchBudgets`; chunks only
control graph memory and do not change generated noise banks or estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .estimators import (
    AnchoredArray,
    BranchBudgets,
    NestedAntitheticSamples,
    NestedZetaEstimate,
    OLBPTTSamples,
    RawDerivativeEstimate,
    anchored_nested_antithetic_regression,
    assemble_raw_recovery_inputs,
    reduce_ol_bptt,
)
from .stage2 import RawRecoveryInputs


class TorchHarvestContractError(ValueError):
    """Raised when a frozen-policy harvester hook violates its contract."""


@dataclass(frozen=True)
class TorchFixedControlHarvest:
    """Raw Stage-II tuple and estimator diagnostics for one anchor."""

    raw: RawRecoveryInputs
    direct: RawDerivativeEstimate
    nested: NestedZetaEstimate
    u_ref: np.ndarray
    sigma_ref: np.ndarray
    injection: np.ndarray
    anchor_id: str
    time_index: int
    seed: int
    budgets: BranchBudgets


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Stage-II harvesting requires the solver extra: "
            "pip install -e '.[solver]'"
        ) from exc
    return torch


def _finite_tensor(name: str, value) -> None:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TorchHarvestContractError(f"{name} must be a torch.Tensor")
    if value.is_complex():
        raise TorchHarvestContractError(f"{name} must be real")
    if not bool(torch.isfinite(value).all().detach().cpu().item()):
        raise TorchHarvestContractError(f"{name} contains non-finite values")


def _state_tensor(adapter, state):
    torch = _torch()
    device = getattr(adapter, "device", "cpu")
    dtype = getattr(adapter, "dtype", None) or torch.float32
    if isinstance(state, torch.Tensor):
        out = state.detach().to(device=device, dtype=dtype)
    else:
        out = torch.as_tensor(state, device=device, dtype=dtype)
    if out.ndim != 1:
        raise TorchHarvestContractError(
            f"one anchor state must have shape (n,), received {tuple(out.shape)}"
        )
    _finite_tensor("anchor state", out)
    return out


def _injection_tensor(state, injection):
    torch = _torch()
    n = state.shape[0]
    if injection is None:
        out = torch.zeros((n, 1), device=state.device, dtype=state.dtype)
        out[0, 0] = 1.0
    elif isinstance(injection, torch.Tensor):
        out = injection.detach().to(device=state.device, dtype=state.dtype)
    else:
        out = torch.as_tensor(
            injection, device=state.device, dtype=state.dtype
        )
    if out.ndim != 2 or out.shape[0] != n or out.shape[1] < 1:
        raise TorchHarvestContractError(
            f"injection must have shape ({n}, r) with r>=1"
        )
    _finite_tensor("current injection", out)
    return out


def _frozen_action(adapter, policy, cfg, state, k):
    torch = _torch()
    # Re-evaluate the actor value on this branch state, but never retain its
    # state or parameter derivative in the OL-BPTT graph.
    with torch.no_grad():
        raw = policy(adapter.features(cfg, state, k))
        action = adapter.chart(cfg, raw)
    action = action.detach()
    _finite_tensor("frozen actor action", action)
    if action.shape[0] != state.shape[0]:
        raise TorchHarvestContractError(
            "frozen actor action must retain the branch batch axis"
        )
    return action


def _rollout_cost(adapter, policy, cfg, state, start, noise, h, N):
    torch = _torch()
    if noise.ndim != 3 or noise.shape[0] != state.shape[0] or \
            noise.shape[1] != N - start:
        raise TorchHarvestContractError(
            "future noise must have shape (branches, N-start, noise_dim)"
        )
    cost = torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)
    for offset, k in enumerate(range(start, N)):
        action = _frozen_action(adapter, policy, cfg, state, k)
        running = adapter.running_cost(cfg, state, action, k)
        if running.shape != cost.shape:
            raise TorchHarvestContractError(
                "adapter.running_cost must return one scalar per branch"
            )
        cost = cost + h * running
        state = adapter.step(cfg, state, action, noise[:, offset])
        _finite_tensor("simulated branch state", state)
    terminal = adapter.terminal_cost(cfg, state)
    if terminal.shape != cost.shape:
        raise TorchHarvestContractError(
            "adapter.terminal_cost must return one scalar per branch"
        )
    return cost + terminal


def _directional_derivatives(
    adapter,
    policy,
    cfg,
    base_state,
    start,
    noise,
    injection,
    h,
    N,
    *,
    second_order,
):
    torch = _torch()
    branches = base_state.shape[0]
    r = injection.shape[1]
    eps = torch.zeros(
        (branches, r),
        device=base_state.device,
        dtype=base_state.dtype,
        requires_grad=True,
    )
    perturbed = base_state + eps @ injection.transpose(0, 1)
    cost = _rollout_cost(
        adapter, policy, cfg, perturbed, start, noise, h, N
    )
    first = torch.autograd.grad(
        cost.sum(), eps, create_graph=second_order, allow_unused=False
    )[0]
    _finite_tensor("OL-BPTT first derivative", first)
    if not second_order:
        return first.detach(), None

    if not first.requires_grad:
        second = torch.zeros(
            (branches, r, r), device=eps.device, dtype=eps.dtype
        )
    else:
        rows = []
        for j in range(r):
            component = first[:, j]
            if component.requires_grad:
                row = torch.autograd.grad(
                    component.sum(),
                    eps,
                    retain_graph=j + 1 < r,
                    allow_unused=True,
                )[0]
            else:  # pragma: no cover - mixed linear/quadratic direction
                row = None
            rows.append(torch.zeros_like(eps) if row is None else row)
        second = torch.stack(rows, dim=-2)
    _finite_tensor("OL-BPTT second derivative", second)
    return first.detach(), second.detach()


def _adapter_geometry(adapter, cfg, state, action):
    torch = _torch()
    mean_hook = getattr(adapter, "conditional_mean", None)
    diffusion_hook = getattr(adapter, "diffusion_matrix", None)
    if not callable(mean_hook) or not callable(diffusion_hook):
        raise TorchHarvestContractError(
            "Stage-II adapter requires conditional_mean and diffusion_matrix"
        )
    mean = mean_hook(cfg, state, action)
    diffusion = diffusion_hook(cfg, state, action)
    _finite_tensor("conditional mean", mean)
    _finite_tensor("diffusion matrix", diffusion)
    if mean.shape != state.shape:
        raise TorchHarvestContractError(
            "conditional_mean must have the state shape"
        )
    if diffusion.ndim != 3 or diffusion.shape[:2] != state.shape:
        raise TorchHarvestContractError(
            "diffusion_matrix must have shape (branches, state_dim, noise_dim)"
        )
    noise_dim = int(getattr(adapter, "noise_dim", diffusion.shape[-1]))
    if diffusion.shape[-1] != noise_dim:
        raise TorchHarvestContractError(
            "diffusion_matrix noise axis disagrees with adapter.noise_dim"
        )
    return mean, diffusion


def harvest_fixed_control_torch(
    adapter,
    policy,
    *,
    state,
    time_index: int,
    budgets: BranchBudgets,
    seed: int,
    anchor_id: str,
    anchor: Any = None,
    injection=None,
    ridge: float = 0.0,
) -> TorchFixedControlHarvest:
    """Harvest raw recovery inputs at one frozen history/control anchor.

    The function generates complete direct and nested noise banks before
    chunking.  Re-running with the same seed and scientific budgets therefore
    yields the same samples regardless of ``branch_batch_size``.
    """

    torch = _torch()
    if not isinstance(budgets, BranchBudgets):
        raise TypeError("budgets must be BranchBudgets")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise TorchHarvestContractError("seed must be a nonnegative integer")
    if not isinstance(anchor_id, str) or not anchor_id.strip():
        raise TorchHarvestContractError("anchor_id must be a non-empty string")
    if getattr(policy, "training", False):
        raise TorchHarvestContractError(
            "Stage-II requires a frozen policy in eval() mode"
        )
    cfg = getattr(adapter, "cfg", None)
    if cfg is None:
        raise TorchHarvestContractError("adapter must own cfg")
    N, h = adapter.grid(cfg)
    if isinstance(time_index, bool) or not isinstance(time_index, int) or not \
            0 <= time_index < N:
        raise TorchHarvestContractError(
            f"time_index must be an integer in [0,{N})"
        )
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise TorchHarvestContractError("adapter grid step must be positive")

    state0 = _state_tensor(adapter, state)
    direction = _injection_tensor(state0, injection)
    noise_dim = int(getattr(adapter, "noise_dim", 0))
    if noise_dim < 1:
        raise TorchHarvestContractError("adapter.noise_dim must be positive")
    generator = torch.Generator(device=state0.device)
    generator.manual_seed(int(seed))
    sqrt_h = float(np.sqrt(h))

    # Generate first, chunk second: compute memory cannot alter science/RNG.
    direct_noise = torch.randn(
        budgets.M,
        N - time_index,
        noise_dim,
        device=state0.device,
        dtype=state0.dtype,
        generator=generator,
    ) * sqrt_h
    p_parts, pi_parts = [], []
    for chunk in budgets.chunks(budgets.M):
        count = chunk.stop - chunk.start
        base = state0.unsqueeze(0).expand(count, -1)
        first, second = _directional_derivatives(
            adapter,
            policy,
            cfg,
            base,
            time_index,
            direct_noise[chunk],
            direction,
            h,
            N,
            second_order=True,
        )
        p_parts.append(first.cpu())
        pi_parts.append(second.cpu())
    p_samples = torch.cat(p_parts, dim=0).numpy()
    pi_samples = torch.cat(pi_parts, dim=0).numpy()
    direct = reduce_ol_bptt(
        OLBPTTSamples(anchor_id, p_samples, pi_samples), budgets
    )

    anchor_state = state0.unsqueeze(0)
    u_ref_t = _frozen_action(adapter, policy, cfg, anchor_state, time_index)
    mean, diffusion = _adapter_geometry(
        adapter, cfg, anchor_state, u_ref_t
    )
    sigma_ref_t = direction.transpose(0, 1) @ diffusion[0]

    outer = torch.randn(
        budgets.M_out,
        noise_dim,
        device=state0.device,
        dtype=state0.dtype,
        generator=generator,
    ) * sqrt_h
    increments = torch.einsum("nd,md->mn", diffusion[0], outer)
    plus_outer = mean[0].unsqueeze(0) + increments
    minus_outer = mean[0].unsqueeze(0) - increments
    plus = plus_outer[:, None, :].expand(-1, budgets.M_in, -1).reshape(
        budgets.M_out * budgets.M_in, -1
    )
    minus = minus_outer[:, None, :].expand(-1, budgets.M_in, -1).reshape(
        budgets.M_out * budgets.M_in, -1
    )
    inner = torch.randn(
        budgets.M_out,
        budgets.M_in,
        N - time_index - 1,
        noise_dim,
        device=state0.device,
        dtype=state0.dtype,
        generator=generator,
    ) * sqrt_h
    inner_flat = inner.reshape(
        budgets.M_out * budgets.M_in, N - time_index - 1, noise_dim
    )
    branch_states = torch.cat((plus, minus), dim=0)
    branch_noise = torch.cat((inner_flat, inner_flat), dim=0)
    first_parts = []
    for chunk in budgets.chunks(budgets.nested_continuations):
        first, _ = _directional_derivatives(
            adapter,
            policy,
            cfg,
            branch_states[chunk],
            time_index + 1,
            branch_noise[chunk],
            direction,
            h,
            N,
            second_order=False,
        )
        first_parts.append(first.cpu())
    nested_gradients = torch.cat(first_parts, dim=0).numpy().reshape(
        2, budgets.M_out, budgets.M_in, direction.shape[1]
    )
    sigma_ref = sigma_ref_t.detach().cpu().numpy()
    nested_samples = NestedAntitheticSamples(
        anchor_id=anchor_id,
        brownian_offsets=outer.detach().cpu().numpy(),
        adjoint_plus=nested_gradients[0],
        adjoint_minus=nested_gradients[1],
    )
    nested = anchored_nested_antithetic_regression(
        nested_samples,
        AnchoredArray(anchor_id, direct.Pi),
        AnchoredArray(anchor_id, sigma_ref),
        budgets,
        ridge=ridge,
    )
    raw = assemble_raw_recovery_inputs(
        direct,
        nested,
        AnchoredArray(anchor_id, sigma_ref),
        anchor=anchor,
        pi_layout="matrix",
    )
    return TorchFixedControlHarvest(
        raw=raw,
        direct=direct,
        nested=nested,
        u_ref=np.asarray(u_ref_t.detach().cpu().numpy()),
        sigma_ref=sigma_ref,
        injection=direction.detach().cpu().numpy(),
        anchor_id=anchor_id,
        time_index=time_index,
        seed=int(seed),
        budgets=budgets,
    )


__all__ = [
    "TorchFixedControlHarvest",
    "TorchHarvestContractError",
    "harvest_fixed_control_torch",
]
