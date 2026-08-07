"""P1 Stage-I torch adapter (H1: explicit P1-U/P1-C separation).

The chart is selected by cfg["control_kind"], set by the canonical YAML:
  * "unconstrained" (main_u.yaml, P1-U): identity head. The exact Riccati
    oracle's actions leave the P1-C box ~39% of the time, so P1-U must never
    train through a box chart.
  * "box" (main.yaml / dp_small.yaml, P1-C): smooth sigmoid chart by default
    (main LSTM-DPO baseline), or chart="clip" for the activation-capable DPO
    ablation (H3: clamp(raw, lo, hi) CAN attain the bounds exactly, so the
    0/100/0 occupancy of the sigmoid baseline is not confounded with chart
    bias when both are reported).
Unknown kinds raise immediately -- a silent wrong-chart run is impossible.

Ownership (H5): the adapter owns cfg/device/dtype; the trainer reads them.
wrap_numpy infers the device from the policy parameters.

The torch simulator is a direct port of the SAME dense matrices used by the
verified numpy reference (oracle.build_dense); path equality is a tested
contract. Buffer layout: Z (B, H+1) newest-first, scan input reversed to
oldest -> newest.  Every scan token contains the state value, the normalized
current decision time, and its normalized relative lag in the delay window:

    [X_{k-H+j}, k h / T, (j-H) / H],  j = 0, ..., H.

Thus the global decision time is shared by the whole observed window while
the lag channel explicitly locates each tap from oldest (-1) to current (0).
"""
import numpy as np
from .oracle import build_dense, riccati
from .dynamics import make_hist

_CHARTS = ("sigmoid", "clip")


class P1Stage1Adapter:
    feat_dim = 3          # (buffer tap, global decision time, relative lag)
    action_dim = 1
    noise_dim = 1
    head_bias = 0.0
    feature_schema = "state_global_time_relative_lag_v2"
    sequence_order = "oldest_to_newest"
    history_law = "make_hist templates (const/ramp/cosine, amp U[-1.2,1.2])"

    def __init__(self, cfg, device="cpu", dtype=None, chart="sigmoid"):
        import torch
        self.cfg = cfg
        self.device = device
        self.dtype = dtype or torch.float32
        self.kind = cfg["control_kind"]
        if self.kind not in ("box", "unconstrained"):
            raise ValueError(f"unknown control kind: {self.kind}")
        if chart not in _CHARTS:
            raise ValueError(f"unknown chart: {chart} (use one of {_CHARTS})")
        self.chart_kind = chart if self.kind == "box" else "identity"
        p, H, h = cfg["params"], cfg["H"], cfg["h"]
        A, B, C, D, Sg = build_dense(p, H, h)
        tf = dict(dtype=self.dtype, device=device)
        self.A = torch.tensor(A, **tf); self.B = torch.tensor(B, **tf)
        self.C = torch.tensor(C, **tf); self.D = torch.tensor(D, **tf)
        self.Sg = torch.tensor(Sg, **tf)
        self.xref = torch.tensor(cfg["xref"], **tf)

    def grid(self, cfg):
        return cfg["N"], cfg["h"]

    def init_state(self, cfg, B, np_rng, device):
        import torch
        Z = np.stack([make_hist(np_rng, cfg["H"]+1, cfg["tt"], cfg["delta"])
                      for _ in range(B)])
        return torch.tensor(Z, dtype=self.dtype, device=device)

    def features(self, cfg, Z, k):
        import torch
        state = Z.flip(1).unsqueeze(-1)                # oldest -> newest
        t = torch.full_like(state, k*cfg["h"]/cfg["T"])
        lag = torch.linspace(-1.0, 0.0, state.shape[1],
                             dtype=state.dtype, device=state.device)
        lag = lag.view(1, -1, 1).expand(state.shape[0], -1, -1)
        return torch.cat([state, t, lag], dim=-1)      # (B, H+1, 3)

    def chart(self, cfg, raw):
        import torch
        raw = raw.squeeze(-1)
        if self.chart_kind == "identity":             # P1-U
            return raw
        lo, hi = cfg["bounds"]
        if self.chart_kind == "sigmoid":              # P1-C main baseline
            return lo + (hi - lo)*torch.sigmoid(raw)
        return torch.clamp(raw, lo, hi)               # P1-C ablation (H3)

    def step(self, cfg, Z, u, dW):
        drift = self.conditional_mean(cfg, Z, u)
        vol = self.diffusion_matrix(cfg, Z, u).squeeze(-1)
        return drift + vol*dW                          # dW: (B, 1) broadcast

    def conditional_mean(self, cfg, Z, u):
        """Euler conditional mean required by the Stage-II branch harvester."""
        return Z @ self.A.T + u[:, None]*self.B

    def diffusion_matrix(self, cfg, Z, u):
        """State-by-noise Euler diffusion matrix, shape ``(B,H+1,1)``."""
        vol = Z @ self.C.T + u[:, None]*self.D + self.Sg
        return vol.unsqueeze(-1)

    def running_cost(self, cfg, Z, u, k):
        p = cfg["params"]
        return 0.5*p["Q"]*(Z[:, 0] - self.xref[k])**2 + 0.5*p["R"]*u**2

    def terminal_cost(self, cfg, Z):
        p = cfg["params"]
        return 0.5*p["QT"]*(Z[:, 0] - cfg["xtar"])**2

    def wrap_numpy(self, cfg, policy, device=None, batch_size=None):
        """evaluate-layer pol(k, Zb): numpy in/out; device inferred from the
        policy parameters (H5) unless overridden.  ``batch_size`` chunks only
        the neural forward pass; it does not change the evaluation bank or
        any Monte-Carlo statistic."""
        import torch
        device = device or next(policy.parameters()).device
        if batch_size is not None:
            if isinstance(batch_size, bool) or int(batch_size) != batch_size \
                    or int(batch_size) <= 0:
                raise ValueError("policy batch_size must be a positive integer")
            batch_size = int(batch_size)

        def pol(k, Zb):
            array = np.atleast_2d(Zb)
            step = len(array) if batch_size is None else batch_size
            outputs = []
            with torch.no_grad():
                for start in range(0, len(array), step):
                    Zt = torch.tensor(array[start:start + step],
                                      dtype=self.dtype, device=device)
                    u = self.chart(cfg, policy(self.features(cfg, Zt, k)))
                    outputs.append(u.cpu().numpy())
            return np.concatenate(outputs, axis=0)
        return pol


def p1u_pilot_metrics(cfg, adapter, policy, Np=2000, seed=123,
                      policy_batch_size=None):
    """Canonical P1-U Stage-I pilot metrics against the EXACT Riccati oracle
    (Stage-I review H2 / Phase B). CRN-paired: the learned policy and the
    oracle feedback are rolled on the SAME initial histories and Brownian
    bank, so dJ_paired has the tight paired SE; the value oracle supplies an
    absolute anchor (J_exact) whose gap to the oracle-rollout MC mean is a
    discretisation/MC consistency check, not a per-path residual (per-path
    value residuals carry the full Brownian variance and are useless as a
    gate). Control nRMSE is measured along the LEARNED policy's rollout.
    Only valid for control_kind == 'unconstrained'."""
    if cfg["control_kind"] != "unconstrained":
        raise ValueError("p1u_pilot_metrics is for the P1-U variant only")
    p, h, N, H = cfg["params"], cfg["h"], cfg["N"], cfg["H"]
    orc = riccati(p, H, h, N, cfg["xref"], cfg["xtar"])
    A, B, C, D, Sg = build_dense(p, H, h)
    rng = np.random.default_rng(seed)
    Z0 = np.stack([make_hist(rng, H+1, cfg["tt"], cfg["delta"])
                   for _ in range(Np)])
    dW = rng.normal(0, np.sqrt(h), (Np, N))
    Jx = (0.5*np.einsum("pn,nm,pm->p", Z0, orc["Pval"][0], Z0)
          + Z0 @ orc["s"][0] + orc["c"][0])
    pol = adapter.wrap_numpy(cfg, policy, batch_size=policy_batch_size)
    costs = []; se_u = ss_u = 0.0
    for which in ("net", "oracle"):
        Z = Z0.copy(); cost = np.zeros(Np)
        for k in range(N):
            if which == "net":
                u = pol(k, Z)
                u_star = Z @ orc["F"][k] + orc["f"][k]
                se_u += float(np.sum((u - u_star)**2))
                ss_u += float(np.sum(u_star**2))
            else:
                u = Z @ orc["F"][k] + orc["f"][k]
            cost += h*(0.5*p["Q"]*(Z[:, 0]-cfg["xref"][k])**2
                       + 0.5*p["R"]*u**2)
            Z = Z @ A.T + np.outer(u, B) \
                + (Z @ C.T + np.outer(u, D) + Sg)*dW[:, k, None]
        cost += 0.5*p["QT"]*(Z[:, 0]-cfg["xtar"])**2
        costs.append(cost)
    d = costs[0] - costs[1]
    anchor_residual = costs[1] - Jx
    return dict(control_nrmse=float(np.sqrt(se_u/max(ss_u, 1e-300))),
                dJ_paired=float(d.mean()),
                dJ_se=float(d.std(ddof=1)/np.sqrt(Np)),
                J_policy=float(costs[0].mean()),
                J_oracle_mc=float(costs[1].mean()),
                J_exact=float(Jx.mean()),
                mc_anchor_gap=float(anchor_residual.mean()),
                mc_anchor_gap_se=float(
                    anchor_residual.std(ddof=1)/np.sqrt(Np)))
