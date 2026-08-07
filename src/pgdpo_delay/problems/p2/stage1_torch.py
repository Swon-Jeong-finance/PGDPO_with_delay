"""P2 Stage-I torch adapter: NODE-coordinate simulator for the distributed-
delay network LQ, injected into the common buffer-scan trainer.

The policy trains in PHYSICAL (node) coordinates -- the network is not told
the ring eigenbasis. All coefficient matrices are polynomials a0 + a1*L in
the ring Laplacian and are assembled directly; only sigma0 is generated in
mode coordinates by the frozen calibration (scaling.ring_spec), so it is
transported by V = eigh(L) eigenvectors. Equivalence with the verified
per-mode numpy recursion (scaling.calibrate rollout) through the SAME V is a
testable contract.

State layout: node buffer Z (B, d, H+1), newest-first along the last axis
(the same tap convention as the mode simulator). LSTM scan input: taps
oldest -> newest, feature = the d-vector of that tap plus normalized time
(feat_dim = d + 1). Control is unconstrained (chart = identity), m = d.
"""
import numpy as np
from .oracle import kernel_weights
from . import scaling


def ring_laplacian(d):
    L = 2.0*np.eye(d)
    idx = np.arange(d)
    L[idx, (idx+1) % d] = -1.0
    L[idx, (idx-1) % d] = -1.0
    return L


def p2_stage1_config(d, H, r=None, variant=None):
    """Frozen-calibration node-coordinate model (single source: p2 yaml via
    scaling.ring_spec). Returns the cfg dict consumed by the trainer and the
    adapter."""
    r = r or scaling.R_MAIN
    h, N = scaling.h, scaling.N
    delta = H*h
    w = kernel_weights(scaling.RHO_DELTA/delta, delta, h, H)
    rng_model = np.random.default_rng(scaling.CFG["seeds"]["model"])
    spec = scaling.ring_spec(d, r, rng_model, variant)
    L = ring_laplacian(d)
    lam, V = np.linalg.eigh(L)
    poly = lambda v: (V*np.asarray(v)) @ V.T        # V diag(v) V^T
    mats = dict(A=poly(spec["a"]), AM=poly(spec["aM"]), B=poly(spec["b"]),
                Q=poly(spec["q"]), R=poly(spec["r"]), QT=poly(spec["qT"]),
                Cs=[poly(spec["c"][l]) for l in range(r)],
                CMs=[poly(spec["cM"][l]) for l in range(r)],
                Ds=[poly(spec["dd"][l]) for l in range(r)],
                sig0=[V @ spec["sig"][l] for l in range(r)])
    tt = np.linspace(-delta, 0, H+1)[::-1]
    return dict(variant="scaling", d=d, H=H, r=r, N=N, h=h, T=scaling.T,
                delta=delta, w=w, V=V, spec=spec, mats=mats, tt=tt)


class P2Stage1Adapter:
    action_dim = None      # set per instance (= d)
    noise_dim = None       # set per instance (= r)
    head_bias = 0.0

    history_law = "per-mode templates (mode_hist), transported by V"

    def __init__(self, cfg, device="cpu", dtype=None):
        import torch
        self.cfg = cfg
        self.dtype = dtype or torch.float32
        f32 = dict(dtype=self.dtype, device=device)
        m = cfg["mats"]
        self.A = torch.tensor(m["A"], **f32); self.AM = torch.tensor(m["AM"], **f32)
        self.B = torch.tensor(m["B"], **f32)
        self.Q = torch.tensor(m["Q"], **f32); self.R = torch.tensor(m["R"], **f32)
        self.QT = torch.tensor(m["QT"], **f32)
        self.Cs = torch.stack([torch.tensor(C, **f32) for C in m["Cs"]])
        self.CMs = torch.stack([torch.tensor(C, **f32) for C in m["CMs"]])
        self.Ds = torch.stack([torch.tensor(D, **f32) for D in m["Ds"]])
        self.sig0 = torch.stack([torch.tensor(s, **f32) for s in m["sig0"]])
        self.w = torch.tensor(cfg["w"], **f32)
        self.feat_dim = cfg["d"] + 1
        self.action_dim = cfg["d"]
        self.noise_dim = cfg["r"]
        self.device = device

    def grid(self, cfg):
        return cfg["N"], cfg["h"]

    def init_state(self, cfg, B, np_rng, device):
        """Same initial-history law as the calibration: per-mode templates
        transported to node coordinates by V."""
        import torch
        Zm = np.stack([scaling.mode_hist(np_rng, cfg["d"], cfg["H"]+1,
                                         cfg["tt"], cfg["delta"])
                       for _ in range(B)])            # (B, d, H+1) modes
        Zn = np.einsum("ij,pjn->pin", cfg["V"], Zm)   # node coords
        return torch.tensor(Zn, dtype=self.dtype, device=device)

    def features(self, cfg, Z, k):
        import torch
        import torch
        feats = Z.flip(2).permute(0, 2, 1)            # (B, H+1, d) old->new
        t = torch.full_like(feats[:, :, :1], k*cfg["h"]/cfg["T"])
        return torch.cat([feats, t], dim=-1)          # (B, H+1, d+1)

    def chart(self, cfg, raw):
        return raw                                    # unconstrained (m = d)

    def step(self, cfg, Z, u, dW):
        import torch
        h = cfg["h"]
        X0 = Z[:, :, 0]
        M = Z @ self.w                                # (B, d, H+1) @ (H+1)
        drift = X0 + h*(X0 @ self.A.T + M @ self.AM.T + u @ self.B.T)
        load = (torch.einsum("rij,pj->pri", self.Cs, X0)
                + torch.einsum("rij,pj->pri", self.CMs, M)
                + torch.einsum("rij,pj->pri", self.Ds, u)
                + self.sig0[None])
        X1 = drift + torch.einsum("pri,pr->pi", load, dW)
        return torch.cat([X1.unsqueeze(-1), Z[:, :, :-1]], dim=2)

    def running_cost(self, cfg, Z, u, k):
        import torch
        X0 = Z[:, :, 0]
        return 0.5*((X0 @ self.Q)*X0).sum(-1) + 0.5*((u @ self.R)*u).sum(-1)

    def terminal_cost(self, cfg, Z):
        import torch
        X0 = Z[:, :, 0]
        return 0.5*((X0 @ self.QT)*X0).sum(-1)

    def wrap_numpy(self, cfg, policy, device=None):
        import torch
        device = device or next(policy.parameters()).device

        def pol(k, Znode):                            # (B, d, H+1) numpy
            Zt = torch.tensor(np.asarray(Znode), dtype=self.dtype,
                              device=device)
            if Zt.dim() == 2:
                Zt = Zt.unsqueeze(0)
            with torch.no_grad():
                u = self.chart(cfg, policy(self.features(cfg, Zt, k)))
            return u.cpu().numpy()
        return pol


def oracle_node_policy(cfg, orc):
    """Mode oracle transported to node coordinates for paired-dJ validation:
    u_node = V [F_i z_i + f_i],  z = V^T Z_node."""
    V = cfg["V"]

    def pol(k, Znode):
        Zm = np.einsum("ji,pjn->pin", V, np.asarray(Znode))
        um = np.stack([Zm[:, i] @ orc[i]["F"][k] + orc[i]["f"][k]
                       for i in range(cfg["d"])], axis=1)
        return um @ V.T
    return pol
