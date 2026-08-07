"""Stage-I adapter contract tests for P2 (node-coordinate network LQ) and P3
(renewal lift / distributed buffer). The torch-vs-numpy path-equality tests
are the durable regression guards: each torch simulator must reproduce the
verified numpy reference on identical (state0, u, dW) paths at float32
precision. Skipped without torch (optional [solver] dependency)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pgdpo_delay.core.stage1 import train_stage1
from pgdpo_delay.problems.p2.stage1_torch import (p2_stage1_config,
                                                  P2Stage1Adapter)
from pgdpo_delay.problems.p2 import scaling
from pgdpo_delay.problems.p2.oracle import mode_rows
from pgdpo_delay.problems.p3.config import load_config as p3_load
from pgdpo_delay.problems.p3 import dynamics as p3d
from pgdpo_delay.problems.p3.stage1_torch import (P3RenewalStage1Adapter,
                                                  P3DistStage1Adapter)


def test_p2_torch_node_matches_numpy_mode():
    """Torch node-coordinate step must equal the verified per-mode numpy
    recursion transported through the SAME eigenbasis V, on identical
    (Z0, u, dW) paths."""
    cfg = p2_stage1_config(d=4, H=3, r=2)
    ad = P2Stage1Adapter(cfg)
    sp, w, V, h = cfg["spec"], cfg["w"], cfg["V"], cfg["h"]
    d, r, N, Np = 4, 2, 20, 4
    rng = np.random.default_rng(0)
    Zm = np.stack([scaling.mode_hist(rng, d, cfg["H"]+1, cfg["tt"],
                                     cfg["delta"]) for _ in range(Np)])
    useq = rng.normal(0, 0.5, (N, Np, d))
    dW = rng.normal(0, np.sqrt(h), (N, Np, r))
    crows = np.stack([np.stack(mode_rows(sp, i, w, h, cfg["H"]+1)[1])
                      for i in range(d)])
    ddm = np.stack(sp["dd"], axis=1); sgm = np.stack(sp["sig"], axis=1)
    Z = Zm.copy(); cn = np.zeros(Np)
    for k in range(N):
        u = useq[k]
        cn += 0.5*h*((Z[:, :, 0]**2*sp["q"][None]).sum(1)
                     + (u*u*sp["r"][None]).sum(1))
        load = (np.einsum('pdn,drn->pdr', Z, crows)
                + ddm[None]*u[:, :, None] + sgm[None])
        X1 = Z[:, :, 0] + h*(sp["a"][None]*Z[:, :, 0]
                             + sp["aM"][None]*np.einsum('pdn,n->pd', Z, w)
                             + sp["b"][None]*u) \
            + np.einsum('pdr,pr->pd', load, dW[k])
        Z = np.concatenate([X1[:, :, None], Z[:, :, :-1]], axis=2)
    cn += 0.5*(Z[:, :, 0]**2*sp["qT"][None]).sum(1)
    Zt = torch.tensor(np.einsum("ij,pjn->pin", V, Zm), dtype=torch.float32)
    ct = torch.zeros(Np)
    for k in range(N):
        un = torch.tensor(useq[k] @ V.T, dtype=torch.float32)
        ct += h*ad.running_cost(cfg, Zt, un, k)
        Zt = ad.step(cfg, Zt, un, torch.tensor(dW[k], dtype=torch.float32))
    ct += ad.terminal_cost(cfg, Zt)
    assert float(np.max(np.abs(ct.numpy()-cn))/np.max(np.abs(cn))) < 1e-5
    Zback = np.einsum("ji,pjn->pin", V, Zt.numpy())
    assert float(np.max(np.abs(Zback - Z))/np.max(np.abs(Z))) < 1e-5


def _p3_path_gap(cfg, ad, buffered):
    Np = 8; rng = np.random.default_rng(1)
    useq = rng.uniform(0, 1, (cfg["N"], Np))
    dW = rng.normal(0, np.sqrt(cfg["dt"]), (cfg["N"], Np))
    p = cfg["params"]
    if not buffered:
        I = rng.uniform(0.05, 0.6, Np); M = rng.uniform(0.05, 0.6, Np)
        In, Mn = I.copy(), M.copy(); cn = np.zeros(Np)
        for k in range(cfg["N"]):
            cn += cfg["dt"]*(0.5*p["c_I"]*np.maximum(In, 0)**2
                             + 0.5*p["R"]*useq[k]**2)
            In, Mn = p3d.step(p, cfg["dt"], In, Mn, useq[k], dW[k])
        cn += 0.5*p["c_T"]*np.maximum(In, 0)**2
        S = torch.tensor(np.stack([I, M], 1), dtype=torch.float32)
        ct = torch.zeros(Np)
        for k in range(cfg["N"]):
            u = torch.tensor(useq[k], dtype=torch.float32)
            ct += cfg["dt"]*ad.running_cost(cfg, S, u, k)
            S = ad.step(cfg, S, u,
                        torch.tensor(dW[k][:, None], dtype=torch.float32))
        ct += ad.terminal_cost(cfg, S)
    else:
        w = p3d.kernel_weights(cfg)
        B0 = p3d.init_history(cfg, Np, np.random.default_rng(2))
        Bn = B0.copy(); cn = np.zeros(Np)
        for k in range(cfg["N"]):
            cn += cfg["dt"]*(0.5*p["c_I"]*np.maximum(Bn[:, 0], 0)**2
                             + 0.5*p["R"]*useq[k]**2)
            Bn = p3d.step_dist(p, cfg["dt"], Bn, w, useq[k], dW[k])
        cn += 0.5*p["c_T"]*np.maximum(Bn[:, 0], 0)**2
        Bt = torch.tensor(B0, dtype=torch.float32); ct = torch.zeros(Np)
        for k in range(cfg["N"]):
            u = torch.tensor(useq[k], dtype=torch.float32)
            ct += cfg["dt"]*ad.running_cost(cfg, Bt, u, k)
            Bt = ad.step(cfg, Bt, u,
                         torch.tensor(dW[k][:, None], dtype=torch.float32))
        ct += ad.terminal_cost(cfg, Bt)
    return float(np.max(np.abs(ct.numpy()-cn))/np.max(np.abs(cn)))


def test_p3r_torch_matches_numpy():
    cfg = p3_load("renewal")
    assert _p3_path_gap(cfg, P3RenewalStage1Adapter(cfg), False) < 1e-5


def test_p3d_torch_matches_numpy():
    cfg = p3_load("distributed")
    assert _p3_path_gap(cfg, P3DistStage1Adapter(cfg), True) < 1e-5


def test_p3_chart_and_wrapped_contracts():
    cfgR = p3_load("renewal"); adR = P3RenewalStage1Adapter(cfgR)
    raw = torch.linspace(-50, 50, 51).unsqueeze(-1)
    u = adR.chart(cfgR, raw)
    assert float(u.min()) >= 0.0 and float(u.max()) <= 1.0   # sigmoid box
    out = train_stage1(adR, cfgR, seed=1, iters=2, batch=16, log_every=100)
    pol = adR.wrap_numpy(cfgR, out["policy"])
    r = p3d.simulate_paired(cfgR, pol, pol, Np=16, seed=3)
    assert r["delta_A_minus_B"] == 0.0                       # CRN self-zero
    cfgD = p3_load("distributed"); adD = P3DistStage1Adapter(cfgD)
    outD = train_stage1(adD, cfgD, seed=1, iters=2, batch=8, log_every=100)
    polD = adD.wrap_numpy(cfgD, outD["policy"])
    uD = polD(0, np.full((5, cfgD["dist"]["H"]+1), 0.3))
    assert uD.shape == (5,) and np.isfinite(uD).all()
    assert (uD >= 0).all() and (uD <= 1).all()


def test_p2_wrapped_contract():
    cfg = p2_stage1_config(d=4, H=3, r=2)
    ad = P2Stage1Adapter(cfg)
    out = train_stage1(ad, cfg, seed=1, iters=2, batch=8, log_every=100)
    pol = ad.wrap_numpy(cfg, out["policy"])
    u = pol(0, np.zeros((3, 4, 4)))
    assert u.shape == (3, 4) and np.isfinite(u).all()
