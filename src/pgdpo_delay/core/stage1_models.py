"""Stage-I torch model classes and the checkpoint contract (H4, Stage-I
review 2026-08-07). This module REQUIRES torch at import time; torch-free
orchestration lives in core/stage1.py, which imports this lazily.

The policy class is a NAMED top-level nn.Module (stable import path
pgdpo_delay.core.stage1_models.BufferScanPolicy), so full-model
serialization, isinstance checks, and state_dict reconstruction all work --
the previous local-class factory could not be pickled.

Checkpoint artifact contract:
    <dir>/stage1_state.pt   -- state_dict
    <dir>/stage1_spec.json  -- architecture, dtype, Torch version and run binding
Reload with `load_checkpoint(dir, expected={...})`; a production caller can
require exact problem/config/chart/fingerprint matching before weights are
read. Outputs must match bitwise on CPU (regression-tested).
"""
import json
from pathlib import Path
import torch
import torch.nn as nn


MODEL_SCHEMA = "buffer_scan_lstm_mlp_v2"
INITIALIZATION_SCHEMA = "xavier_ih_orthogonal_hh_fb1_mlp_v1"


class BufferScanPolicy(nn.Module):
    """Buffer-scan LSTM policy: scans the explicit history window
    (oldest -> newest) at every call; stateless in time -- pol(k, Z)."""

    def __init__(self, feat_dim, action_dim, hidden=256, num_layers=2,
                 head_bias=0.0):
        super().__init__()
        feat_dim = int(feat_dim)
        action_dim = int(action_dim)
        hidden = int(hidden)
        num_layers = int(num_layers)
        head_bias = float(head_bias)
        if feat_dim <= 0 or action_dim <= 0 or hidden <= 0:
            raise ValueError("feat_dim, action_dim and hidden must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        # ``model_schema`` is the architecture boundary used by checkpoint
        # reconstruction.  In particular, v1's single ``head`` tensor must
        # never be silently interpreted as this two-layer MLP head.
        self.spec = {
            "model_schema": MODEL_SCHEMA,
            "feat_dim": feat_dim,
            "action_dim": action_dim,
            "hidden": hidden,
            "num_layers": num_layers,
            "head_hidden": hidden,
            "head_activation": "tanh",
            "head_bias": head_bias,
            "initialization_schema": INITIALIZATION_SCHEMA,
        }
        self.lstm = nn.LSTM(
            feat_dim, hidden, num_layers=num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden, hidden)
        self.act = nn.Tanh()
        self.linear2 = nn.Linear(hidden, action_dim)

        # PyTorch adds bias_ih and bias_hh.  Setting both forget-gate slices
        # to 0.5 therefore gives the intended effective forget bias 1.0.
        # Work under no_grad instead of mutating ``.data`` so initialization
        # remains explicit and autograd-safe.
        with torch.no_grad():
            for name, parameter in self.lstm.named_parameters():
                if name.startswith("weight_ih"):
                    nn.init.xavier_uniform_(parameter)
                elif name.startswith("weight_hh"):
                    for gate in range(4):
                        nn.init.orthogonal_(
                            parameter[gate*hidden:(gate+1)*hidden])
                elif name.startswith("bias"):
                    nn.init.zeros_(parameter)
                    parameter[hidden:2*hidden].fill_(0.5)

            nn.init.xavier_uniform_(
                self.linear1.weight,
                gain=nn.init.calculate_gain("tanh"),
            )
            nn.init.zeros_(self.linear1.bias)
            nn.init.xavier_uniform_(self.linear2.weight, gain=0.1)
            nn.init.constant_(self.linear2.bias, head_bias)

    def forward(self, feats):                    # (B, L, F) -> (B, m)
        out, _ = self.lstm(feats)
        return self.linear2(self.act(self.linear1(out[:, -1, :])))


def save_checkpoint(policy, outdir, extra=None):
    """Write stage1_state.pt + stage1_spec.json under outdir."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    if not isinstance(policy, BufferScanPolicy):
        raise TypeError("policy must be a BufferScanPolicy")
    extra = dict(extra or {})
    conflicts = sorted(
        key for key in policy.spec
        if key in extra and extra[key] != policy.spec[key]
    )
    if conflicts:
        raise ValueError(
            "checkpoint metadata conflicts with policy architecture: "
            + ", ".join(conflicts))
    torch.save(policy.state_dict(), outdir/"stage1_state.pt")
    spec = dict(policy.spec, dtype=str(next(policy.parameters()).dtype),
                torch_version=torch.__version__, **extra)
    (outdir/"stage1_spec.json").write_text(json.dumps(spec, indent=1))
    return spec


def validate_checkpoint_binding(spec, expected):
    """Reject loading a checkpoint under the wrong scientific adapter.

    ``expected`` is deliberately explicit (for example ``problem``,
    ``problem_config``, ``chart`` and ``run_fingerprint``).  Older smoke
    v2 checkpoints remain loadable without an expectation, while a production
    caller can never silently reinterpret P1-U weights as P1-C or bind a
    checkpoint from another protocol.  Legacy v1 checkpoints are explicitly
    rejected by :func:`load_checkpoint` because their head is incompatible.
    """
    if not isinstance(spec, dict):
        raise TypeError("checkpoint spec must be a mapping")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("expected checkpoint binding must be a nonempty mapping")
    mismatches = []
    for key, wanted in expected.items():
        if key not in spec:
            mismatches.append(f"{key}: missing (expected {wanted!r})")
        elif spec[key] != wanted:
            mismatches.append(
                f"{key}: found {spec[key]!r}, expected {wanted!r}")
    if mismatches:
        raise ValueError("checkpoint binding mismatch: " + "; ".join(mismatches))
    return True


def load_checkpoint(outdir, device="cpu", expected=None):
    """Rebuild a policy after optionally validating its scientific binding."""
    outdir = Path(outdir)
    spec = json.loads((outdir/"stage1_spec.json").read_text())
    if expected is not None:
        validate_checkpoint_binding(spec, expected)
    schema = spec.get("model_schema")
    if schema is None:
        raise ValueError(
            "legacy Stage-I checkpoint has no model_schema and uses an "
            "architecture incompatible with BufferScanPolicy v2; retrain "
            "or load it with the archived v1 code")
    if schema != MODEL_SCHEMA:
        raise ValueError(
            f"unsupported Stage-I model_schema {schema!r}; expected "
            f"{MODEL_SCHEMA!r}")
    required = {
        "feat_dim", "action_dim", "hidden", "num_layers", "head_hidden",
        "head_activation", "head_bias", "initialization_schema",
    }
    missing = sorted(required - spec.keys())
    if missing:
        raise ValueError(
            "incomplete BufferScanPolicy v2 checkpoint spec: missing "
            + ", ".join(missing))
    if spec["head_hidden"] != spec["hidden"] or \
            spec["head_activation"] != "tanh" or \
            spec["initialization_schema"] != INITIALIZATION_SCHEMA:
        raise ValueError(
            "checkpoint architecture disagrees with BufferScanPolicy v2 "
            "fixed MLP head")
    policy = BufferScanPolicy(spec["feat_dim"], spec["action_dim"],
                              hidden=spec["hidden"],
                              num_layers=spec["num_layers"],
                              head_bias=spec["head_bias"])
    if spec.get("dtype") == "torch.float64":
        policy = policy.double()
    try:
        policy.load_state_dict(torch.load(
            outdir/"stage1_state.pt", map_location=device,
            weights_only=True))
    except TypeError:  # torch < 2.0 has no weights_only keyword.
        policy.load_state_dict(torch.load(
            outdir/"stage1_state.pt", map_location=device))
    except RuntimeError as exc:
        raise ValueError(
            "checkpoint tensors are incompatible with declared "
            f"model_schema {MODEL_SCHEMA!r}") from exc
    policy.to(device).eval()
    return policy, spec
