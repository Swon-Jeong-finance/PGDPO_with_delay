"""Stage-I (buffer-scan LSTM-DPO) contract tests for P1, expanded per the
Stage-I review sec.7 roster. Skipped without torch ([solver] extra)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pgdpo_delay.problems.p1.config import load_config
from pgdpo_delay.problems.p1 import oracle, dynamics, evaluate
from pgdpo_delay.problems.p1.stage1_torch import P1Stage1Adapter
from pgdpo_delay.core.stage1 import train_stage1
from pgdpo_delay.core import stage1_models


def test_torch_simulator_matches_numpy_reference():
    """Full-path cost equality vs the verified dense recursion."""
    cfg = load_config("dp_small")
    ad = P1Stage1Adapter(cfg)
    p, H, h, N = cfg["params"], cfg["H"], cfg["h"], cfg["N"]
    A, B, C, D, Sg = oracle.build_dense(p, H, h)
    rng = np.random.default_rng(0); Np = 16
    Z0 = np.stack([dynamics.make_hist(rng, H+1, cfg["tt"], cfg["delta"])
                   for _ in range(Np)])
    useq = rng.uniform(*cfg["bounds"], (Np, N))
    dW = rng.normal(0, np.sqrt(h), (Np, N))
    Zn = Z0.copy(); cn = np.zeros(Np)
    for k in range(N):
        cn += h*(0.5*p["Q"]*(Zn[:, 0]-cfg["xref"][k])**2
                 + 0.5*p["R"]*useq[:, k]**2)
        Zn = Zn @ A.T + np.outer(useq[:, k], B) \
            + (Zn @ C.T + np.outer(useq[:, k], D) + Sg)*dW[:, k, None]
    cn += 0.5*p["QT"]*(Zn[:, 0]-cfg["xtar"])**2
    Zt = torch.tensor(Z0, dtype=torch.float32); ct = torch.zeros(Np)
    for k in range(N):
        u = torch.tensor(useq[:, k], dtype=torch.float32)
        ct += h*ad.running_cost(cfg, Zt, u, k)
        Zt = ad.step(cfg, Zt, u, torch.tensor(dW[:, k:k+1],
                                              dtype=torch.float32))
    ct += ad.terminal_cost(cfg, Zt)
    assert float(np.max(np.abs(ct.numpy()-cn))/np.max(np.abs(cn))) < 1e-5


def test_torch_step_matches_scalar_formula():
    """sec.7.2-1: independent of build_dense -- one step against the raw
    scalar recursion plus the buffer shift."""
    cfg = load_config("dp_small"); p = cfg["params"]; h = cfg["h"]
    ad = P1Stage1Adapter(cfg)
    rng = np.random.default_rng(3)
    Z = rng.normal(0, 1, (7, cfg["H"]+1))
    u = rng.uniform(*cfg["bounds"], 7); dW = rng.normal(0, np.sqrt(h), 7)
    Zt = ad.step(cfg, torch.tensor(Z, dtype=torch.float32),
                 torch.tensor(u, dtype=torch.float32),
                 torch.tensor(dW[:, None], dtype=torch.float32)).numpy()
    x, xH = Z[:, 0], Z[:, -1]
    xn = x + h*(p["a"]*x + p["ad"]*xH + p["b"]*u) \
        + (p["s0"] + p["cx"]*x + p["cy"]*xH + p["gu"]*u)*dW
    assert np.max(np.abs(Zt[:, 0] - xn)) < 1e-5
    assert np.max(np.abs(Zt[:, 1:] - Z[:, :-1])) < 1e-6   # shift


def test_features_orientation_asymmetric_history():
    """sec.7.2-2: distinct taps -- the scan must be oldest -> newest, the
    LAST token must be the CURRENT state Z[:, 0], global decision time is
    shared across the window, and relative lag runs from -1 to 0."""
    cfg = load_config("dp_small")
    ad = P1Stage1Adapter(cfg)
    Z = torch.arange(cfg["H"]+1, dtype=torch.float32).unsqueeze(0)  # 0=newest
    k = 2
    f = ad.features(cfg, Z, k=k)
    assert f.shape == (1, cfg["H"]+1, 3)
    assert float(f[0, -1, 0]) == float(Z[0, 0])     # last token = current
    assert float(f[0, 0, 0]) == float(Z[0, -1])     # first token = oldest
    assert torch.equal(
        f[0, :, 1],
        torch.full((cfg["H"]+1,), k*cfg["h"]/cfg["T"]),
    )
    assert torch.equal(
        f[0, :, 2], torch.linspace(-1.0, 0.0, cfg["H"]+1),
    )
    assert ad.feat_dim == 3
    assert ad.feature_schema == "state_global_time_relative_lag_v2"
    assert ad.sequence_order == "oldest_to_newest"


def test_stateless_contract_interleaved_calls():
    """sec.7.2-3: identical (k, Z) must give identical output regardless of
    interleaved calls (no hidden carry)."""
    cfg = load_config("dp_small")
    ad = P1Stage1Adapter(cfg)
    out = train_stage1(ad, seed=1, iters=2, batch=8, log_every=100)
    pol = ad.wrap_numpy(cfg, out["policy"])
    Z = np.linspace(-1, 1, cfg["H"]+1)[None, :]
    u1 = pol(3, Z)
    pol(0, np.zeros_like(Z)); pol(7, np.ones_like(Z))   # interleave
    u2 = pol(3, Z)
    assert np.array_equal(u1, u2)


def test_numpy_policy_evaluation_chunking_preserves_outputs():
    from pgdpo_delay.core.stage1_models import BufferScanPolicy

    cfg = load_config("main_u")
    ad = P1Stage1Adapter(cfg)
    torch.manual_seed(17)
    policy = BufferScanPolicy(
        ad.feat_dim, ad.action_dim, hidden=8, num_layers=2)
    rng = np.random.default_rng(17)
    Z = rng.normal(size=(11, cfg["H"] + 1))
    full = ad.wrap_numpy(cfg, policy)(7, Z)
    chunked = ad.wrap_numpy(cfg, policy, batch_size=4)(7, Z)
    np.testing.assert_allclose(full, chunked, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="positive integer"):
        ad.wrap_numpy(cfg, policy, batch_size=0)


def test_p1u_chart_identity_and_kind_guards():
    """sec.7.2-4 + H1: unconstrained mode is identity; clip chart attains the
    bounds (H3); unknown chart/kind raise."""
    cfg_u = load_config("main_u")
    assert cfg_u["control_kind"] == "unconstrained" and cfg_u["bounds"] is None
    ad_u = P1Stage1Adapter(cfg_u)
    raw = torch.linspace(-5, 5, 11).unsqueeze(-1)
    assert torch.equal(ad_u.chart(cfg_u, raw), raw.squeeze(-1))
    cfg_c = load_config("dp_small")
    ad_clip = P1Stage1Adapter(cfg_c, chart="clip")
    u = ad_clip.chart(cfg_c, 100*raw)
    lo, hi = cfg_c["bounds"]
    assert float(u.min()) == pytest.approx(lo) and \
        float(u.max()) == pytest.approx(hi)         # bounds attainable
    with pytest.raises(ValueError):
        P1Stage1Adapter(cfg_c, chart="nope")
    bad = dict(cfg_c); bad["control_kind"] = "mystery"
    with pytest.raises(ValueError):
        P1Stage1Adapter(bad)


def test_checkpoint_roundtrip():
    """sec.7.2-5 + H4: named class, save/load, identical outputs."""
    cfg = load_config("dp_small")
    ad = P1Stage1Adapter(cfg)
    out = train_stage1(ad, seed=4, iters=2, batch=8, log_every=100)
    policy = out["policy"]
    assert isinstance(policy, stage1_models.BufferScanPolicy)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        stage1_models.save_checkpoint(policy, td,
                                      extra=dict(seeds=out["seeds"]))
        re, spec = stage1_models.load_checkpoint(td)
    feats = ad.features(cfg, torch.zeros(3, cfg["H"]+1), 0)
    with torch.no_grad():
        assert torch.equal(policy(feats), re(feats))
    assert spec["feat_dim"] == 3


def test_config_mismatch_rejected():
    """sec.7.2-6 + H5: a cfg differing from the adapter's own is refused."""
    ad = P1Stage1Adapter(load_config("dp_small"))
    with pytest.raises(ValueError):
        train_stage1(ad, load_config("main"), seed=0, iters=1, batch=4)


def test_main_h16_short_smoke():
    """sec.7.2-7: the H=16 main grid runs (previous tests were all H=3)."""
    cfg = load_config("main")
    ad = P1Stage1Adapter(cfg)
    out = train_stage1(ad, seed=5, iters=2, batch=8, log_every=100)
    assert np.isfinite(out["losses"]).all()


def test_training_log_reports_validation_and_best_marker(capsys):
    cfg = load_config("dp_small")
    ad = P1Stage1Adapter(cfg)
    train_stage1(
        ad, seed=9, iters=2, batch=4, hidden=8, num_layers=2,
        log_every=100, val_every=1, val_batch=4,
    )
    lines = capsys.readouterr().out.splitlines()
    first = next(line for line in lines if "iter     1" in line)
    second = next(line for line in lines if "iter     2" in line)
    assert "J_train =" in first and "J_val =" in first
    assert first.rstrip().endswith("*")
    assert "J_train =" in second and "J_val =" in second
    assert any("selected best validation J" in line for line in lines)


def test_gradient_path_includes_diffusion_channel():
    """sec.7.2-8: dX_{k+1}/du = h*b + gamma_u*dW exactly (the controlled-
    diffusion channel must be in the autograd path)."""
    cfg = load_config("dp_small"); p = cfg["params"]; h = cfg["h"]
    ad = P1Stage1Adapter(cfg)
    Z = torch.zeros(1, cfg["H"]+1)
    u = torch.tensor([0.3], requires_grad=True)
    dW = torch.tensor([[0.17]])
    X1 = ad.step(cfg, Z, u, dW)[:, 0]
    g, = torch.autograd.grad(X1.sum(), u)
    assert float(g) == pytest.approx(h*p["b"] + p["gu"]*0.17, rel=1e-6)


def test_training_overfit_deterministic_fast():
    """sec.7.3 fast tier: fixed history/noise bank re-used every iteration is
    driven down deterministically (no stochastic-optimization luck)."""
    cfg = load_config("dp_small")
    ad = P1Stage1Adapter(cfg)
    from pgdpo_delay.core.stage1_models import BufferScanPolicy
    torch.manual_seed(0)
    policy = BufferScanPolicy(ad.feat_dim, ad.action_dim, hidden=32)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    rng = np.random.default_rng(0)
    Z0 = ad.init_state(cfg, 32, rng, "cpu")
    dW = torch.randn(32, cfg["N"], 1)*float(np.sqrt(cfg["h"]))
    losses = []
    for _ in range(25):
        opt.zero_grad()
        Z = Z0.clone(); cost = torch.zeros(32)
        for k in range(cfg["N"]):
            u = ad.chart(cfg, policy(ad.features(cfg, Z, k)))
            cost = cost + cfg["h"]*ad.running_cost(cfg, Z, u, k)
            Z = ad.step(cfg, Z, u, dW[:, k])
        loss = (cost + ad.terminal_cost(cfg, Z)).mean()
        loss.backward(); opt.step(); losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] - 0.05


@pytest.mark.slow
def test_training_reduces_objective_smoke():
    """Stochastic 300-iteration smoke (slow tier)."""
    cfg = load_config("dp_small")
    ad = P1Stage1Adapter(cfg)
    out = train_stage1(ad, seed=2, iters=300, batch=256, lr=1e-3,
                       log_every=1000)
    assert np.isfinite(out["losses"]).all()
    assert np.mean(out["losses"][-20:]) < np.mean(out["losses"][:20]) - 0.1
