"""Stage I: common buffer-scan LSTM-DPO trainer (SINGLE implementation).

Convention (user-confirmed 2026-08-07): the policy is STATELESS in the
history -- pol(k, Z) -- and the LSTM scans the explicit buffer window
(length H+1, oldest -> newest) at EVERY step; no hidden state is carried
across time steps, so the same (k, Z) always yields the same action and
Stage-II branch states close on the physical buffer alone. Buffer-scan is
NOT memoryless: the physical history is re-encoded on every call instead of
being carried as an external recurrent state.

Ownership contract (Stage-I review 2026-08-07, H5): the ADAPTER owns cfg,
device, and dtype; the trainer reads them from the adapter. Passing a cfg to
train_stage1 is allowed only as a consistency check -- a differing config
hash raises. wrap_numpy infers the device from the policy parameters.

RNG contract (H6): a single user seed is split via numpy SeedSequence into
independent streams -- seed_model (parameter init), seed_history (initial
states), seed_train_noise (Brownian bank, dedicated torch.Generator on the
training device), seed_validation (fixed validation bank) -- so architecture
or chart ablations do not perturb the training noise. All four are returned
for the manifest.

Problems inject ONLY mathematical parts via an adapter object:
    feat_dim, action_dim, noise_dim, head_bias, cfg, device, dtype
    grid(cfg) -> (N, h);  init_state(cfg, B, np_rng, device)
    features(cfg, state, k);  chart(cfg, raw);  step(cfg, state, u, dW)
    running_cost(cfg, state, u, k)  [rate; trainer applies h]
    terminal_cost(cfg, state);  wrap_numpy(cfg, policy)
and MUST NOT reimplement the loop (same rule as stage2).

torch is an optional dependency (`pip install pgdpo-delay[solver]`); the
named model class and checkpoint API live in core/stage1_models.py.
"""
import hashlib
import numpy as np
from .artifacts import config_hash


def BufferScanPolicy(feat_dim, action_dim, hidden=64, num_layers=2,
                     head_bias=0.0):
    """Lazy factory kept for backward compatibility; returns the NAMED
    nn.Module from core.stage1_models (H4)."""
    from .stage1_models import BufferScanPolicy as _P
    return _P(feat_dim, action_dim, hidden=hidden, num_layers=num_layers,
              head_bias=head_bias)


def _spawn_seeds(seed):
    ss = np.random.SeedSequence(seed)
    kids = ss.spawn(4)
    ints = [int(k.generate_state(1)[0]) for k in kids]
    return dict(seed_model=ints[0], seed_history=ints[1],
                seed_train_noise=ints[2], seed_validation=ints[3])


def train_stage1(adapter, cfg=None, *, seed, iters, batch, lr=3e-4,
                 hidden=64, num_layers=2, device=None, clip_grad=1.0,
                 log_every=100, val_every=0, val_batch=512):
    """Common LSTM-DPO warm-up: unroll -> closed-loop BPTT -> Adam.

    val_every > 0 enables the independent validation bank (fixed histories +
    fixed Brownian bank from seed_validation) with best-checkpoint selection
    (review sec.6.1); the returned policy is then the best-validation model,
    not the last iterate. Returns dict(policy, losses, val_trace, best_iter,
    seeds, hp, grad_norms, clip_frac)."""
    import torch
    from .stage1_models import BufferScanPolicy as _Policy
    acfg = getattr(adapter, "cfg", None)
    if cfg is None:
        if acfg is None:
            raise ValueError("adapter does not own a cfg and none was passed")
        cfg = acfg
    elif acfg is not None and config_hash(_hashable(cfg)) != \
            config_hash(_hashable(acfg)):
        raise ValueError("cfg passed to train_stage1 differs from the "
                         "adapter's cfg (H5 mismatch guard)")
    device = device or getattr(adapter, "device", "cpu")
    dtype = getattr(adapter, "dtype", None) or torch.float32
    seeds = _spawn_seeds(seed)
    torch.manual_seed(seeds["seed_model"])
    policy = _Policy(adapter.feat_dim, adapter.action_dim, hidden=hidden,
                     num_layers=num_layers,
                     head_bias=getattr(adapter, "head_bias", 0.0))
    policy = policy.to(device=device, dtype=dtype)
    gen = torch.Generator(device=device)
    gen.manual_seed(seeds["seed_train_noise"])
    rng_hist = np.random.default_rng(seeds["seed_history"])
    N, h = adapter.grid(cfg)
    sqh = float(np.sqrt(h))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    def _rollout_cost(state, dW):
        cost = torch.zeros(state.shape[0], device=device, dtype=dtype)
        for k in range(N):
            raw = policy(adapter.features(cfg, state, k))
            u = adapter.chart(cfg, raw)
            cost = cost + h*adapter.running_cost(cfg, state, u, k)
            state = adapter.step(cfg, state, u, dW[:, k])
        return cost + adapter.terminal_cost(cfg, state)

    val_state0 = val_dW = None
    if val_every:
        rng_val = np.random.default_rng(seeds["seed_validation"])
        val_state0 = adapter.init_state(cfg, val_batch, rng_val, device)
        gv = torch.Generator(device=device)
        gv.manual_seed(seeds["seed_validation"])
        val_dW = torch.randn(val_batch, N, adapter.noise_dim, device=device,
                             dtype=dtype, generator=gv)*sqh
    losses, val_trace, grad_norms = [], [], []
    clip_hits = 0
    best = dict(J=np.inf, it=0, state=None)
    for it in range(1, iters + 1):
        opt.zero_grad(set_to_none=True)
        state = adapter.init_state(cfg, batch, rng_hist, device)
        dW = torch.randn(batch, N, adapter.noise_dim, device=device,
                         dtype=dtype, generator=gen)*sqh
        loss = _rollout_cost(state, dW).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite training loss at iter {it}")
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                                  clip_grad if clip_grad
                                                  else float("inf")))
        if not np.isfinite(gn):
            raise FloatingPointError(f"nonfinite gradient norm at iter {it}")
        grad_norms.append(gn)
        if clip_grad is not None and gn > clip_grad:
            clip_hits += 1
        opt.step()
        losses.append(float(loss.item()))
        did_validate = False
        new_best = False
        Jv = None
        if val_every and (it % val_every == 0 or it == iters):
            with torch.no_grad():
                Jv = float(_rollout_cost(val_state0.clone(), val_dW)
                           .mean().item())
            val_trace.append((it, Jv))
            did_validate = True
            if Jv < best["J"]:
                new_best = True
                best = dict(J=Jv, it=it,
                            state={k: v.detach().clone()
                                   for k, v in policy.state_dict().items()})
        if it == 1 or it % log_every == 0 or it == iters or did_validate:
            message = (f"[stage1] iter {it:5d}  "
                       f"J_train = {losses[-1]:.6f}")
            if did_validate:
                message += f"  J_val = {Jv:.6f}"
            if new_best:
                message += "  *"
            print(message, flush=True)
    if val_every and best["state"] is not None:
        policy.load_state_dict(best["state"])
        print(f"[stage1] selected best validation J = {best['J']:.6f} "
              f"at iter {best['it']}", flush=True)
    policy.eval()
    hp = dict(iters=iters, batch=batch, lr=lr, hidden=hidden,
              num_layers=num_layers,
              clip_grad=clip_grad, val_every=val_every, val_batch=val_batch,
              dtype=str(dtype), device=str(device),
              history_law=getattr(adapter, "history_law",
                                  "adapter-defined"))
    return dict(policy=policy, losses=losses, val_trace=val_trace,
                best_iter=best["it"] if val_every else iters, seeds=seeds,
                hp=hp, grad_norms=grad_norms,
                clip_frac=clip_hits/max(1, iters))


def _hashable(cfg):
    """Canonical compact representation for trainer/adapter cfg matching.

    Dynamics arrays and raw YAML are scientific identity, not cache noise.
    Arrays are therefore represented by shape, dtype and a full-byte SHA256
    instead of being dropped (the previous behavior could miss a P2
    eigenbasis or variant mismatch).
    """
    if isinstance(cfg, dict):
        return {str(key): _hashable(value) for key, value in cfg.items()}
    if isinstance(cfg, np.ndarray):
        array = np.ascontiguousarray(cfg)
        return {
            "__ndarray__": hashlib.sha256(array.tobytes()).hexdigest(),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    if isinstance(cfg, np.generic):
        return cfg.item()
    if isinstance(cfg, (list, tuple)):
        return [_hashable(value) for value in cfg]
    return cfg
