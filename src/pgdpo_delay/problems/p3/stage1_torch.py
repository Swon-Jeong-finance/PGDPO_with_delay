"""P3 Stage-I torch adapters: renewal (I, M) lift and distributed I-buffer,
both direct ports of the shared numpy simulator (dynamics.step /
dynamics.step_dist) with the same full-truncation convention -- torch-vs-
numpy path equality is a testable contract.

Chart: u in [0, 1] via sigmoid (head_bias 0 -> u starts at 0.5, matching the
old-notebook practice of initialising near a sensible action). Costs use the
half-convention with the truncated state, exactly as the numpy layer.
"""
import numpy as np
from . import dynamics


class _P3Base:
    action_dim = 1
    noise_dim = 1
    head_bias = 0.0

    def grid(self, cfg):
        return cfg["N"], cfg["dt"]

    def chart(self, cfg, raw):
        import torch
        return torch.sigmoid(raw.squeeze(-1))         # [0, 1] box

    def _rate(self, cfg, Ib, u):
        p = cfg["params"]
        return 0.5*p["c_I"]*Ib**2 + 0.5*p["R"]*u**2

    def _terminal(self, cfg, Ib):
        return 0.5*cfg["params"]["c_T"]*Ib**2


class P3RenewalStage1Adapter(_P3Base):
    """State (B, 2) = (I, M): the exact 2D Markov lift (no buffer needed);
    the LSTM scan degenerates to a length-1 window with features (I, M, t)."""
    feat_dim = 3

    history_law = "I0, M0 independent uniforms (renewal init law)"

    def __init__(self, cfg, device="cpu", dtype=None):
        import torch
        self.cfg = cfg
        self.device = device
        self.dtype = dtype or torch.float32

    def init_state(self, cfg, B, np_rng, device):
        import torch
        I = np_rng.uniform(*cfg["init"]["I0"], B)
        M = np_rng.uniform(*cfg["init"]["M0"], B)
        return torch.tensor(np.stack([I, M], 1), dtype=self.dtype,
                            device=device)

    def features(self, cfg, S, k):
        import torch
        t = torch.full_like(S[:, :1], k*cfg["dt"]/cfg["T"])
        return torch.cat([S, t], dim=-1).unsqueeze(1)  # (B, 1, 3)

    def step(self, cfg, S, u, dW):
        import torch
        p, dt = cfg["params"], cfg["dt"]
        I, M = S[:, 0], S[:, 1]
        Ib = torch.clamp(I, min=0.0)
        drift = (p["beta"]*(1.0 - Ib/p["Npop"])*M - p["gamma"]*Ib
                 - p["b"]*u*Ib)
        sig = p["sigma0"]*(1.0 - p["eta_sigma"]*u)*Ib
        In = I + drift*dt + sig*dW.squeeze(-1)
        Mn = M + p["rho"]*(Ib - M)*dt
        return torch.stack([In, Mn], dim=1)

    def running_cost(self, cfg, S, u, k):
        import torch
        return self._rate(cfg, torch.clamp(S[:, 0], min=0.0), u)

    def terminal_cost(self, cfg, S):
        import torch
        return self._terminal(cfg, torch.clamp(S[:, 0], min=0.0))

    def wrap_numpy(self, cfg, policy, device=None):
        import torch
        device = device or next(policy.parameters()).device

        def pol(k, I, M):
            S = torch.tensor(np.stack([np.atleast_1d(I), np.atleast_1d(M)], 1),
                             dtype=self.dtype, device=device)
            with torch.no_grad():
                u = self.chart(cfg, policy(self.features(cfg, S, k)))
            return u.cpu().numpy()
        return pol


class P3DistStage1Adapter(_P3Base):
    """State (B, H+1) = I-buffer, newest-first (dynamics.step_dist layout);
    scan input reverses to oldest -> newest with features (I-tap, t)."""
    feat_dim = 2

    history_law = "linear I-ramp (I0, Ipast independent uniforms)"

    def __init__(self, cfg, device="cpu", dtype=None):
        import torch
        self.cfg = cfg
        self.device = device
        self.dtype = dtype or torch.float32
        self.w = torch.tensor(dynamics.kernel_weights(cfg),
                              dtype=self.dtype, device=device)

    def init_state(self, cfg, B, np_rng, device):
        import torch
        return torch.tensor(dynamics.init_history(cfg, B, np_rng),
                            dtype=self.dtype, device=device)

    def features(self, cfg, Bf, k):
        import torch
        feats = Bf.flip(1).unsqueeze(-1)
        t = torch.full_like(feats[:, :, :1], k*cfg["dt"]/cfg["T"])
        return torch.cat([feats, t], dim=-1)           # (B, H+1, 2)

    def step(self, cfg, Bf, u, dW):
        import torch
        p, dt = cfg["params"], cfg["dt"]
        Ib = torch.clamp(Bf[:, 0], min=0.0)
        M = torch.clamp(Bf, min=0.0) @ self.w
        drift = (p["beta"]*(1.0 - Ib/p["Npop"])*M - p["gamma"]*Ib
                 - p["b"]*u*Ib)
        sig = p["sigma0"]*(1.0 - p["eta_sigma"]*u)*Ib
        In = Bf[:, 0] + drift*dt + sig*dW.squeeze(-1)
        return torch.cat([In.unsqueeze(1), Bf[:, :-1]], dim=1)

    def running_cost(self, cfg, Bf, u, k):
        import torch
        return self._rate(cfg, torch.clamp(Bf[:, 0], min=0.0), u)

    def terminal_cost(self, cfg, Bf):
        import torch
        return self._terminal(cfg, torch.clamp(Bf[:, 0], min=0.0))

    def wrap_numpy(self, cfg, policy, device=None):
        import torch
        device = device or next(policy.parameters()).device

        def pol(k, Bnp):
            Bt = torch.tensor(np.atleast_2d(Bnp), dtype=self.dtype,
                              device=device)
            with torch.no_grad():
                u = self.chart(cfg, policy(self.features(cfg, Bt, k)))
            return u.cpu().numpy()
        return pol
