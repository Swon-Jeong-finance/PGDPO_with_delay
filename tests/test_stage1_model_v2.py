"""Focused contract tests for the buffer-scan LSTM+MLP v2 policy."""
import json

import pytest


torch = pytest.importorskip("torch")

from pgdpo_delay.core.stage1_models import (
    INITIALIZATION_SCHEMA,
    MODEL_SCHEMA,
    BufferScanPolicy,
    load_checkpoint,
    save_checkpoint,
)


def test_v2_architecture_and_spec_are_explicit():
    policy = BufferScanPolicy(
        feat_dim=3, action_dim=2, hidden=12, num_layers=2,
        head_bias=0.25,
    )

    assert policy.lstm.num_layers == 2
    assert policy.linear1.in_features == 12
    assert policy.linear1.out_features == 12
    assert policy.linear2.in_features == 12
    assert policy.linear2.out_features == 2
    assert policy.spec == {
        "model_schema": MODEL_SCHEMA,
        "feat_dim": 3,
        "action_dim": 2,
        "hidden": 12,
        "num_layers": 2,
        "head_hidden": 12,
        "head_activation": "tanh",
        "head_bias": 0.25,
        "initialization_schema": INITIALIZATION_SCHEMA,
    }

    output = policy(torch.randn(5, 17, 3))
    assert output.shape == (5, 2)


def test_v2_initialization_has_gatewise_orthogonal_recurrence_and_bias():
    hidden = 8
    policy = BufferScanPolicy(3, 1, hidden=hidden, num_layers=2,
                              head_bias=-0.2)
    identity = torch.eye(hidden)

    for layer in range(2):
        recurrent = getattr(policy.lstm, f"weight_hh_l{layer}").detach()
        for gate in range(4):
            block = recurrent[gate*hidden:(gate+1)*hidden]
            assert torch.allclose(block @ block.T, identity, atol=2e-6,
                                  rtol=2e-6)
        for prefix in ("bias_ih", "bias_hh"):
            bias = getattr(policy.lstm, f"{prefix}_l{layer}").detach()
            assert torch.count_nonzero(bias[:hidden]) == 0
            assert torch.all(bias[hidden:2*hidden] == 0.5)
            assert torch.count_nonzero(bias[2*hidden:]) == 0

    assert torch.count_nonzero(policy.linear1.bias) == 0
    assert torch.all(policy.linear2.bias == -0.2)


def test_v2_checkpoint_roundtrip_reconstructs_num_layers(tmp_path):
    torch.manual_seed(9)
    policy = BufferScanPolicy(3, 1, hidden=10, num_layers=3,
                              head_bias=0.1).double().eval()
    features = torch.randn(4, 6, 3, dtype=torch.float64)
    with torch.no_grad():
        expected = policy(features)

    spec = save_checkpoint(policy, tmp_path, extra={"checkpoint_schema": 3})
    restored, loaded_spec = load_checkpoint(tmp_path)
    with torch.no_grad():
        actual = restored(features)

    assert torch.equal(actual, expected)
    assert restored.lstm.num_layers == 3
    assert spec == loaded_spec
    assert loaded_spec["model_schema"] == MODEL_SCHEMA


def test_legacy_or_misdeclared_checkpoint_fails_clearly(tmp_path):
    legacy = {
        "feat_dim": 2,
        "action_dim": 1,
        "hidden": 8,
        "head_bias": 0.0,
    }
    (tmp_path / "stage1_spec.json").write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="legacy.*incompatible"):
        load_checkpoint(tmp_path)

    legacy["model_schema"] = "buffer_scan_lstm_v1"
    (tmp_path / "stage1_spec.json").write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="unsupported.*model_schema"):
        load_checkpoint(tmp_path)


def test_checkpoint_extra_cannot_relabel_architecture(tmp_path):
    policy = BufferScanPolicy(3, 1, hidden=8, num_layers=2)
    with pytest.raises(ValueError, match="conflicts.*policy architecture"):
        save_checkpoint(policy, tmp_path, extra={"num_layers": 7})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feat_dim": 0, "action_dim": 1},
        {"feat_dim": 1, "action_dim": 0},
        {"feat_dim": 1, "action_dim": 1, "hidden": 0},
        {"feat_dim": 1, "action_dim": 1, "num_layers": 0},
    ],
)
def test_invalid_architecture_dimensions_are_rejected(kwargs):
    with pytest.raises(ValueError, match="positive"):
        BufferScanPolicy(**kwargs)
