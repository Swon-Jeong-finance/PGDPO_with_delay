"""Focused schema/publication tests for the P1-wide result contract."""

from __future__ import annotations

import csv
import json
from copy import deepcopy

import pytest

from pgdpo_delay.core import artifacts
from pgdpo_delay.reporting.p1_output_contract import (
    P1OutputContractError,
    ROLE_SHARED,
    SCOPE_SHARED,
    build_p1_output_contract,
    from_stage1_aggregate,
    publish_p1_output_contract,
    read_p1_output_contract,
)
from pgdpo_delay.reporting.stage1_aggregate import aggregate_stage1_runs


PROBLEM_CONFIG_HASH = "p1-scientific-config-v3"


def _paired_evaluation(*, stage2=False, holdout=False, audit=False):
    evaluation = {
        "paired_rollout": {
            "Np": 20_000,
            "seed": 123,
            "bank_id": "p1-main-u-paired-v1",
            "common_random_numbers": True,
        }
    }
    if stage2:
        evaluation["stage2_recovery_bank"] = {
            "states": 40,
            "seed": 601,
            "bank_id": "p1-stage2-recovery-v1",
            "M": 8192,
            "M_out": 512,
            "M_in": 8,
            "branch_batch_size": 2048,
        }
    if stage2 or holdout:
        evaluation["holdout_kkt_bank"] = {
            "states": 40,
            "seed": 701,
            "bank_id": "p1-stage2-holdout-v1",
            "M": 8192,
            "M_out": 512,
            "M_in": 8,
            "branch_batch_size": 2048,
            "independent_of_recovery": True,
        }
    if audit:
        evaluation["estimator_audit_bank"] = {
            "states": 40,
            "seed": 7,
            "bank_id": "p1-exact-policy-estimator-audit-v1",
            "M": 8192,
            "M_out": 512,
            "M_in": 8,
            "branch_batch_size": 2048,
            "policy": "exact_oracle_affine_feedback",
        }
    return evaluation


def _exact_reference():
    return {
        "method": "exact_riccati",
        "role": "exact_oracle",
        "api_version": "p1-v3-pcur-pnext",
        "problem_config_hash": PROBLEM_CONFIG_HASH,
        "config_hash": "oracle-config-1234",
    }


def _p1u_metrics(seed):
    J_policy = 1.10 + seed / 100.0
    return {
        "J_policy": J_policy,
        "control_nrmse": 0.05 + seed / 1000.0,
        "dJ_paired": J_policy - 1.002,
        "dJ_se": 0.01,
        "best_iter": 100 + seed,
        "total_runtime_seconds": 12.0 + seed,
    }


def _p1u_result(seed, *, role="stage1", method="stage1_lstm_dpo"):
    metrics = _p1u_metrics(seed)
    if role == "stage2":
        metrics.update(
            {
                "solver_r_num_rms": 1e-8,
                "holdout_kkt_rms": 0.02,
                "projection_activation_fraction": 0.10,
                "projection_displacement_mean": 0.003,
                "projection_displacement_max": 0.02,
                "feasibility_violation_rate": 0.0,
                "max_feasibility_violation": 0.0,
                "recovery_denominator_min": 0.4,
                "stage2_runtime_seconds": 15.0,
            }
        )
    result = {
        "method": method,
        "method_role": role,
        "seed": seed,
        "problem_config_hash": PROBLEM_CONFIG_HASH,
        "config_hash": f"config-{role}",
        "run_fingerprint": f"fingerprint-{role}",
        "metrics": metrics,
    }
    if role == "stage2":
        result["projection"] = {
            "mode": "identity-audit",
            "api_version": "p1-stage2-projection-v1",
            "config_hash": "identity-audit-projectors-v1",
        }
    return result


def _shared_u():
    return {
        "J_exact": 1.0,
        "J_oracle_mc": 1.002,
        "mc_anchor_gap": 0.002,
        "mc_anchor_gap_se": 0.004,
    }


def _p1c_result(*, dJ_paired=-0.04):
    return {
        "method": "stage1_lstm_dpo",
        "method_role": "stage1",
        "seed": 3,
        "problem_config_hash": PROBLEM_CONFIG_HASH,
        "config_hash": "p1c-stage1-config",
        "run_fingerprint": "p1c-fingerprint",
        "metrics": {
            "J_policy": 0.91,
            "J_baseline": 0.95,
            "dJ_paired": dJ_paired,
            "dJ_se": 0.01,
            "constraint_violation_rate": 0.0,
            "max_constraint_violation": 0.0,
            "holdout_kkt_rms": 0.03,
        },
    }


def test_p1u_stage1_stage2_and_oracle_share_one_contract(tmp_path):
    payload = build_p1_output_contract(
        variant="p1_u",
        problem_config_hash=PROBLEM_CONFIG_HASH,
        evaluation=_paired_evaluation(stage2=True),
        reference=_exact_reference(),
        seed_results=[
            _p1u_result(1),
            _p1u_result(1, role="stage2", method="pgdpo"),
        ],
        shared_diagnostics=_shared_u(),
        metadata={"purpose": "P1 end-to-end comparison"},
    )

    published = publish_p1_output_contract(tmp_path, payload)
    loaded = read_p1_output_contract(tmp_path)

    assert loaded.bundle_dir == published.bundle_dir
    assert loaded.payload == published.payload
    assert loaded.payload["schema"] == 2
    assert loaded.payload["evaluation"]["paired_rollout"]["Np"] == 20_000
    assert loaded.payload["evaluation"]["stage2_recovery_bank"]["states"] == 40
    assert loaded.payload["evaluation"]["holdout_kkt_bank"]["states"] == 40
    stage2_result = next(
        result for result in loaded.payload["seed_results"]
        if result["method_role"] == "stage2"
    )
    assert stage2_result["projection"] == {
        "mode": "identity-audit",
        "api_version": "p1-stage2-projection-v1",
        "config_hash": "identity-audit-projectors-v1",
    }
    assert loaded.payload["metric_schema"]["J_exact"] == {
        "scope": SCOPE_SHARED,
        "role": ROLE_SHARED,
        "custom": False,
    }

    with open(published.csv_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    exact_rows = [row for row in rows if row["scope"] == SCOPE_SHARED]
    assert len(exact_rows) == len(_shared_u())
    assert {row["seed"] for row in exact_rows} == {""}
    assert {row["method"] for row in exact_rows} == {"exact_riccati"}
    # Shared exact anchors occur once, not once for Stage I and Stage II.
    assert [row["metric"] for row in exact_rows].count("J_exact") == 1


def test_p1c_accepts_paired_constraint_kkt_and_active_metrics():
    metrics = {
        "J_policy": 0.91,
        "J_baseline": 0.95,
        "dJ_paired": -0.04,
        "dJ_se": 0.01,
        "constraint_violation_rate": 0.0,
        "max_constraint_violation": 0.0,
        "holdout_kkt_rms": 0.03,
        "active_lower_fraction": 0.20,
        "active_interior_fraction": 0.50,
        "active_upper_fraction": 0.30,
        "switch_count_mean": 2.3,
        "switched_fraction": 0.60,
    }
    payload = build_p1_output_contract(
        variant="p1_c",
        problem_config_hash=PROBLEM_CONFIG_HASH,
        evaluation=_paired_evaluation(holdout=True),
        reference={
            "method": "stage1_lstm_dpo",
            "role": "learned_baseline",
            "api_version": "stage1-worker-v3",
            "problem_config_hash": PROBLEM_CONFIG_HASH,
            "config_hash": "p1c-stage1-config",
        },
        seed_results=[
            {
                "method": "stage1_lstm_dpo",
                "method_role": "stage1",
                "seed": 3,
                "problem_config_hash": PROBLEM_CONFIG_HASH,
                "config_hash": "p1c-stage1-config",
                "run_fingerprint": "p1c-fingerprint",
                "metrics": metrics,
            }
        ],
        shared_diagnostics={},
    )
    assert payload["variant"] == "p1_c"
    assert payload["seed_results"][0]["metrics"]["holdout_kkt_rms"] == 0.03

    bad = dict(metrics)
    bad["active_upper_fraction"] = 0.35
    with pytest.raises(P1OutputContractError, match="sum to"):
        build_p1_output_contract(
            variant="p1_c",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=_paired_evaluation(holdout=True),
            reference={
                "method": "stage1_lstm_dpo",
                "role": "learned_baseline",
                "api_version": "stage1-worker-v3",
                "problem_config_hash": PROBLEM_CONFIG_HASH,
                "config_hash": "p1c-stage1-config",
            },
            seed_results=[
                {
                    "method": "stage1_lstm_dpo",
                    "method_role": "stage1",
                    "seed": 3,
                    "problem_config_hash": PROBLEM_CONFIG_HASH,
                    "config_hash": "p1c-stage1-config",
                    "run_fingerprint": "p1c-fingerprint",
                    "metrics": bad,
                }
            ],
            shared_diagnostics={},
        )

    with pytest.raises(P1OutputContractError, match="no global exact oracle"):
        build_p1_output_contract(
            variant="p1_c",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=_paired_evaluation(holdout=True),
            reference=_exact_reference(),
            seed_results=[
                {
                    "method": "stage1_lstm_dpo",
                    "method_role": "stage1",
                    "seed": 3,
                    "problem_config_hash": PROBLEM_CONFIG_HASH,
                    "config_hash": "p1c-stage1-config",
                    "run_fingerprint": "p1c-fingerprint",
                    "metrics": metrics,
                }
            ],
            shared_diagnostics={},
        )


def test_learned_adjoint_nrmse_is_not_an_allowed_seed_metric():
    result = _p1u_result(1)
    result["metrics"]["p_nrmse"] = 0.1
    with pytest.raises(P1OutputContractError, match="unknown metric 'p_nrmse'"):
        build_p1_output_contract(
            variant="p1_u",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=_paired_evaluation(),
            reference=_exact_reference(),
            seed_results=[result],
            shared_diagnostics=_shared_u(),
        )


def test_comparison_rejects_a_mixed_problem_configuration():
    mixed = _p1u_result(1)
    mixed["problem_config_hash"] = "different-dynamics"
    with pytest.raises(P1OutputContractError, match="problem_config_hash"):
        build_p1_output_contract(
            variant="p1_u",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=_paired_evaluation(),
            reference=_exact_reference(),
            seed_results=[mixed],
            shared_diagnostics=_shared_u(),
        )


def test_exact_policy_estimator_audit_requires_separate_named_bank():
    shared = {**_shared_u(), "audit_p_cur_nrmse": 0.005}
    with pytest.raises(P1OutputContractError, match="estimator_audit_bank"):
        build_p1_output_contract(
            variant="p1_u",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=_paired_evaluation(),
            reference=_exact_reference(),
            seed_results=[_p1u_result(1)],
            shared_diagnostics=shared,
        )
    payload = build_p1_output_contract(
        variant="p1_u",
        problem_config_hash=PROBLEM_CONFIG_HASH,
        evaluation=_paired_evaluation(audit=True),
        reference=_exact_reference(),
        seed_results=[_p1u_result(1)],
        shared_diagnostics=shared,
    )
    assert payload["evaluation"]["estimator_audit_bank"]["policy"] == \
        "exact_oracle_affine_feedback"


def test_stage2_cannot_inherit_20k_rollout_as_branch_bank():
    with pytest.raises(P1OutputContractError, match="stage2_recovery_bank"):
        build_p1_output_contract(
            variant="p1_u",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=_paired_evaluation(),
            reference=_exact_reference(),
            seed_results=[_p1u_result(1, role="stage2", method="pgdpo")],
            shared_diagnostics=_shared_u(),
        )


@pytest.mark.parametrize(
    "missing_bank", ["stage2_recovery_bank", "holdout_kkt_bank"]
)
def test_stage2_requires_both_named_branch_banks(missing_bank):
    evaluation = _paired_evaluation(stage2=True)
    evaluation.pop(missing_bank)
    with pytest.raises(P1OutputContractError, match=missing_bank):
        build_p1_output_contract(
            variant="p1_u",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=evaluation,
            reference=_exact_reference(),
            seed_results=[_p1u_result(1, role="stage2", method="pgdpo")],
            shared_diagnostics=_shared_u(),
        )


def test_complete_comparison_requires_common_random_numbers():
    evaluation = _paired_evaluation()
    evaluation["paired_rollout"]["common_random_numbers"] = False
    with pytest.raises(P1OutputContractError, match="common_random_numbers=true"):
        build_p1_output_contract(
            variant="p1_u",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=evaluation,
            reference=_exact_reference(),
            seed_results=[_p1u_result(1)],
            shared_diagnostics=_shared_u(),
        )


def test_p1u_rejects_inconsistent_paired_objective_and_anchor_gap():
    wrong_difference = _p1u_result(1)
    wrong_difference["metrics"]["dJ_paired"] *= -1.0
    with pytest.raises(P1OutputContractError, match="dJ_paired.*arithmetic"):
        build_p1_output_contract(
            variant="p1_u",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=_paired_evaluation(),
            reference=_exact_reference(),
            seed_results=[wrong_difference],
            shared_diagnostics=_shared_u(),
        )

    wrong_anchor = _shared_u()
    wrong_anchor["mc_anchor_gap"] = -wrong_anchor["mc_anchor_gap"]
    with pytest.raises(P1OutputContractError, match="mc_anchor_gap.*arithmetic"):
        build_p1_output_contract(
            variant="p1_u",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=_paired_evaluation(),
            reference=_exact_reference(),
            seed_results=[_p1u_result(1)],
            shared_diagnostics=wrong_anchor,
        )


def test_p1c_rejects_inconsistent_paired_objective():
    with pytest.raises(P1OutputContractError, match="dJ_paired.*arithmetic"):
        build_p1_output_contract(
            variant="p1_c",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=_paired_evaluation(holdout=True),
            reference={
                "method": "stage1_lstm_dpo",
                "role": "learned_baseline",
                "api_version": "stage1-worker-v3",
                "problem_config_hash": PROBLEM_CONFIG_HASH,
                "config_hash": "p1c-stage1-config",
            },
            seed_results=[_p1c_result(dJ_paired=0.04)],
            shared_diagnostics={},
        )


def test_holdout_kkt_metric_requires_its_named_bank():
    with pytest.raises(P1OutputContractError, match="holdout_kkt_bank"):
        build_p1_output_contract(
            variant="p1_c",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=_paired_evaluation(),
            reference={
                "method": "stage1_lstm_dpo",
                "role": "learned_baseline",
                "api_version": "stage1-worker-v3",
                "problem_config_hash": PROBLEM_CONFIG_HASH,
                "config_hash": "p1c-stage1-config",
            },
            seed_results=[_p1c_result()],
            shared_diagnostics={},
        )


@pytest.mark.parametrize("shared_field", ["bank_id", "seed"])
def test_stage2_recovery_and_holdout_banks_must_be_independent(shared_field):
    evaluation = deepcopy(_paired_evaluation(stage2=True))
    evaluation["holdout_kkt_bank"][shared_field] = \
        evaluation["stage2_recovery_bank"][shared_field]
    expected = "bank_id" if shared_field == "bank_id" else "different seeds"
    with pytest.raises(P1OutputContractError, match=expected):
        build_p1_output_contract(
            variant="p1_u",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=evaluation,
            reference=_exact_reference(),
            seed_results=[_p1u_result(1, role="stage2", method="pgdpo")],
            shared_diagnostics=_shared_u(),
        )


def test_holdout_bank_requires_explicit_independence_declaration():
    evaluation = _paired_evaluation(holdout=True)
    evaluation["holdout_kkt_bank"]["independent_of_recovery"] = False
    with pytest.raises(P1OutputContractError, match="must be true"):
        build_p1_output_contract(
            variant="p1_c",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=evaluation,
            reference={
                "method": "stage1_lstm_dpo",
                "role": "learned_baseline",
                "api_version": "stage1-worker-v3",
                "problem_config_hash": PROBLEM_CONFIG_HASH,
                "config_hash": "p1c-stage1-config",
            },
            seed_results=[_p1c_result()],
            shared_diagnostics={},
        )


@pytest.mark.parametrize("missing", ["mode", "api_version", "config_hash"])
def test_stage2_projection_provenance_is_complete(missing):
    result = _p1u_result(1, role="stage2", method="pgdpo")
    result["projection"].pop(missing)
    with pytest.raises(P1OutputContractError, match="projection fields"):
        build_p1_output_contract(
            variant="p1_u",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=_paired_evaluation(stage2=True),
            reference=_exact_reference(),
            seed_results=[result],
            shared_diagnostics=_shared_u(),
        )


def test_stage2_projection_provenance_rejects_missing_or_unknown_mode():
    missing = _p1u_result(1, role="stage2", method="pgdpo")
    missing.pop("projection")
    with pytest.raises(P1OutputContractError, match="projection provenance"):
        build_p1_output_contract(
            variant="p1_u",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=_paired_evaluation(stage2=True),
            reference=_exact_reference(),
            seed_results=[missing],
            shared_diagnostics=_shared_u(),
        )

    unknown = _p1u_result(1, role="stage2", method="pgdpo")
    unknown["projection"]["mode"] = "implicit-identity"
    with pytest.raises(P1OutputContractError, match="projection.mode"):
        build_p1_output_contract(
            variant="p1_u",
            problem_config_hash=PROBLEM_CONFIG_HASH,
            evaluation=_paired_evaluation(stage2=True),
            reference=_exact_reference(),
            seed_results=[unknown],
            shared_diagnostics=_shared_u(),
        )


def _write_stage1_seed(root, seed, metrics):
    directory = root / f"seed{seed}"
    directory.mkdir(parents=True)
    config = {
        "protocol": "p1_u",
        "evaluation": {"Np": 20_000, "seed": 123},
        "problem_config_hash": PROBLEM_CONFIG_HASH,
    }
    artifacts.write_manifest(
        directory,
        problem="p1",
        method="stage1_lstm_dpo",
        config=config,
        seeds={"train": seed},
        device=f"cuda:{seed - 1}",
        solver="torch-stage1",
        extra={
            "run_fingerprint": "p1-stage1-fingerprint",
            "problem_config_hash": PROBLEM_CONFIG_HASH,
        },
    )
    artifacts.atomic_write_json(
        directory / "status.json",
        {
            "schema": 1,
            "status": "COMPLETE",
            "problem": "p1",
            "method": "stage1_lstm_dpo",
            "seed": seed,
            "run_fingerprint": "p1-stage1-fingerprint",
            "problem_config_hash": PROBLEM_CONFIG_HASH,
        },
    )
    artifacts.atomic_write_json(directory / "metrics.json", metrics)


def test_stage1_aggregate_adapter_separates_repeated_exact_diagnostics(tmp_path):
    root = tmp_path / "stage1"
    for seed in (1, 2):
        _write_stage1_seed(
            root,
            seed,
            {
                **_p1u_metrics(seed),
                **_shared_u(),
            },
        )
    roles = {
        name: "training_seed_metric" for name in _p1u_metrics(1)
    }
    roles["dJ_se"] = "within_policy_paired_mc_se"
    roles["best_iter"] = "health_or_runtime"
    roles["total_runtime_seconds"] = "health_or_runtime"
    roles.update({name: "shared_evaluation_diagnostic" for name in _shared_u()})
    aggregate = aggregate_stage1_runs(
        root,
        expected_seeds=[1, 2],
        metric_roles=roles,
        shared_evaluation_bank=True,
    )
    payload = from_stage1_aggregate(
        aggregate,
        evaluation=_paired_evaluation(),
        reference=_exact_reference(),
    )
    assert payload["shared_diagnostics"] == _shared_u()
    assert len(payload["seed_results"]) == 2
    assert "J_exact" not in payload["seed_results"][0]["metrics"]
    assert payload["seed_results"][0]["metrics"]["dJ_se"] == 0.01

    with pytest.raises(P1OutputContractError, match="cannot re-stamp"):
        from_stage1_aggregate(
            aggregate,
            problem_config_hash="arbitrary-other-problem",
            evaluation=_paired_evaluation(),
            reference=_exact_reference(),
        )


def test_failed_second_publish_preserves_current_complete_bundle(
        tmp_path, monkeypatch):
    payload = build_p1_output_contract(
        variant="p1_u",
        problem_config_hash=PROBLEM_CONFIG_HASH,
        evaluation=_paired_evaluation(),
        reference=_exact_reference(),
        seed_results=[_p1u_result(1)],
        shared_diagnostics=_shared_u(),
    )
    first = publish_p1_output_contract(tmp_path, payload)
    pointer_before = first.pointer_path.read_bytes()

    import pgdpo_delay.reporting.p1_output_contract as contract_module

    def fail_csv(*args, **kwargs):
        raise OSError("forced CSV write failure")

    monkeypatch.setattr(contract_module, "_atomic_write_csv", fail_csv)
    with pytest.raises(OSError, match="forced CSV write failure"):
        publish_p1_output_contract(tmp_path, payload)

    assert first.pointer_path.read_bytes() == pointer_before
    assert read_p1_output_contract(tmp_path).payload == first.payload
    assert not list((tmp_path / ".staging").iterdir())


def test_reader_rejects_csv_tampering(tmp_path):
    payload = build_p1_output_contract(
        variant="p1_u",
        problem_config_hash=PROBLEM_CONFIG_HASH,
        evaluation=_paired_evaluation(),
        reference=_exact_reference(),
        seed_results=[_p1u_result(1)],
        shared_diagnostics=_shared_u(),
    )
    published = publish_p1_output_contract(tmp_path, payload)
    rows = published.csv_path.read_text(encoding="utf-8")
    published.csv_path.write_text(
        rows.replace(",0.01\n", ",0.02\n", 1), encoding="utf-8"
    )
    with pytest.raises(P1OutputContractError, match="CSV disagrees"):
        read_p1_output_contract(tmp_path)
