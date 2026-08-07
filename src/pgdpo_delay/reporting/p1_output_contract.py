"""Versioned P1 comparison artifacts shared by Stage I and Stage II.

The learned-method seed is an uncertainty axis; an exact Riccati value or a
fixed evaluation-bank diagnostic is not.  This module makes that distinction
structural instead of relying on a plotting script to remember it::

    seed_results[]       one record per (method, training seed)
    shared_diagnostics   one copy of fixed-bank/reference quantities

The same contract supports the two P1 roles:

``p1_u``
    Exact same-grid Riccati comparison: control error and paired objective
    regret. Effective-input nRMSE is allowed only in a separately identified
    audit whose frozen policy is the exact affine oracle, never by comparing
    a suboptimal learned-policy adjoint directly with optimal adjoints.
``p1_c``
    No-global-oracle comparison: common-noise paired objective, independent
    holdout KKT, constraint satisfaction, and optional active-set statistics.

Publication uses :mod:`pgdpo_delay.core.artifacts` immutable bundle
transactions.  JSON and tidy CSV are built in one private staging directory;
only after both validate is a single current-pointer atomically replaced.
The reader independently validates the JSON schema, scientific content hash,
CSV equality, manifest, and config binding.

This is a reporting boundary only.  It imports neither torch nor a Stage-II
solver and can therefore be used by Stage I, Stage II, exact references, and
post-processing without changing their numerical implementations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

from pgdpo_delay.core.artifacts import (
    begin_bundle,
    config_hash,
    resolve_current_bundle,
    write_manifest,
)
from pgdpo_delay.reporting.stage1_aggregate import (
    Stage1AggregationResult,
    _atomic_write_csv,
)


P1_OUTPUT_SCHEMA_VERSION = 2
DEFAULT_BASENAME = "p1_results"
DEFAULT_TIER = "comparison"

_OBJECTIVE_ABS_TOL = 1e-10
_OBJECTIVE_REL_TOL = 1e-8
_PROJECTION_MODES = frozenset(("identity-audit", "numerical-product-set"))

SCOPE_SEED = "seed_result"
SCOPE_SHARED = "shared_diagnostic"

ROLE_SEED = "training_seed_metric"
ROLE_PAIRED_SE = "within_policy_paired_mc_se"
ROLE_HEALTH = "health_or_runtime"
ROLE_SHARED = "shared_evaluation_diagnostic"

_VARIANTS = frozenset(("p1_u", "p1_c"))
_METHOD_ROLES = frozenset(("stage1", "stage2", "benchmark"))
_REFERENCE_ROLES = frozenset(
    ("exact_oracle", "learned_baseline", "numerical_benchmark")
)
_METRIC_ROLES = frozenset((ROLE_SEED, ROLE_PAIRED_SE, ROLE_HEALTH, ROLE_SHARED))

_CSV_FIELDS = (
    "schema",
    "artifact_id",
    "problem",
    "variant",
    "method",
    "method_role",
    "seed",
    "problem_config_hash",
    "config_hash",
    "run_fingerprint",
    "scope",
    "metric",
    "role",
    "value",
)


class P1OutputContractError(ValueError):
    """Raised when a P1 result artifact violates its declared contract."""


@dataclass(frozen=True)
class MetricDefinition:
    """One known scalar metric and its uncertainty role."""

    scope: str
    role: str
    variants: frozenset[str]
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class P1OutputArtifacts:
    """Paths returned after atomically publishing one comparison bundle."""

    bundle_dir: Path
    json_path: Path
    csv_path: Path
    pointer_path: Path
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class P1OutputReadResult:
    """Validated on-disk comparison payload and its immutable bundle."""

    bundle_dir: Path
    json_path: Path
    csv_path: Path
    payload: Mapping[str, Any]


def _metric(scope: str, role: str, variants=("p1_u", "p1_c"), **bounds):
    return MetricDefinition(scope, role, frozenset(variants), **bounds)


# This catalog is deliberately finite.  A typo cannot silently become a new
# plotted metric.  ``extra_metric_schema`` is the explicit extension point for
# a later protocol/schema revision.
P1_METRIC_CATALOG: Mapping[str, MetricDefinition] = {
    # Common learned-policy comparison.
    "J_policy": _metric(SCOPE_SEED, ROLE_SEED),
    "J_baseline": _metric(SCOPE_SEED, ROLE_SEED, ("p1_c",)),
    "control_nrmse": _metric(SCOPE_SEED, ROLE_SEED, ("p1_u",), lower=0.0),
    "dJ_paired": _metric(SCOPE_SEED, ROLE_SEED),
    "dJ_se": _metric(SCOPE_SEED, ROLE_PAIRED_SE, lower=0.0),
    # Stage-II effective-input/recovery diagnostics against the P1-U oracle.
    # These are intentionally *shared audit* metrics, not learned-policy
    # seed metrics.  A learned warm-up's adjoints evaluate its own suboptimal
    # policy and therefore must never be scored directly against optimal
    # Riccati adjoints.  The names below are legal only with the separately
    # identified exact-oracle-policy estimator audit bank.
    "audit_p_cur_nrmse": _metric(
        SCOPE_SHARED, ROLE_SHARED, ("p1_u",), lower=0.0
    ),
    "audit_p_nxt_nrmse": _metric(
        SCOPE_SHARED, ROLE_SHARED, ("p1_u",), lower=0.0
    ),
    "audit_q_nrmse": _metric(
        SCOPE_SHARED, ROLE_SHARED, ("p1_u",), lower=0.0
    ),
    "audit_Pi_nrmse": _metric(
        SCOPE_SHARED, ROLE_SHARED, ("p1_u",), lower=0.0
    ),
    "audit_zeta_nrmse": _metric(
        SCOPE_SHARED, ROLE_SHARED, ("p1_u",), lower=0.0
    ),
    "finite_h_pcur_pnxt_action_nrmse": _metric(
        SCOPE_SHARED, ROLE_SHARED, ("p1_u",), lower=0.0
    ),
    "recovery_action_rmse": _metric(
        SCOPE_SEED, ROLE_SEED, ("p1_u",), lower=0.0
    ),
    "solver_r_num_rms": _metric(SCOPE_SEED, ROLE_SEED, lower=0.0),
    "solver_r_num_max": _metric(SCOPE_SEED, ROLE_SEED, lower=0.0),
    "holdout_kkt_rms": _metric(SCOPE_SEED, ROLE_SEED, lower=0.0),
    "holdout_kkt_max": _metric(SCOPE_SEED, ROLE_SEED, lower=0.0),
    "dJ_stage2_minus_stage1": _metric(SCOPE_SEED, ROLE_SEED),
    "dJ_stage2_minus_stage1_se": _metric(
        SCOPE_SEED, ROLE_PAIRED_SE, lower=0.0
    ),
    "projection_activation_fraction": _metric(
        SCOPE_SEED, ROLE_SEED, lower=0.0, upper=1.0
    ),
    "projection_displacement_mean": _metric(
        SCOPE_SEED, ROLE_SEED, lower=0.0
    ),
    "projection_displacement_max": _metric(
        SCOPE_SEED, ROLE_SEED, lower=0.0
    ),
    "feasibility_violation_rate": _metric(
        SCOPE_SEED, ROLE_SEED, lower=0.0, upper=1.0
    ),
    "max_feasibility_violation": _metric(
        SCOPE_SEED, ROLE_SEED, lower=0.0
    ),
    "recovery_denominator_min": _metric(SCOPE_SEED, ROLE_SEED),
    # P1-C no-global-oracle main metrics.
    "constraint_violation_rate": _metric(
        SCOPE_SEED, ROLE_SEED, ("p1_c",), lower=0.0, upper=1.0
    ),
    "max_constraint_violation": _metric(
        SCOPE_SEED, ROLE_SEED, ("p1_c",), lower=0.0
    ),
    "active_lower_fraction": _metric(
        SCOPE_SEED, ROLE_SEED, ("p1_c",), lower=0.0, upper=1.0
    ),
    "active_interior_fraction": _metric(
        SCOPE_SEED, ROLE_SEED, ("p1_c",), lower=0.0, upper=1.0
    ),
    "active_upper_fraction": _metric(
        SCOPE_SEED, ROLE_SEED, ("p1_c",), lower=0.0, upper=1.0
    ),
    "switch_count_mean": _metric(
        SCOPE_SEED, ROLE_SEED, ("p1_c",), lower=0.0
    ),
    "switched_fraction": _metric(
        SCOPE_SEED, ROLE_SEED, ("p1_c",), lower=0.0, upper=1.0
    ),
    "first_switch_time_mean": _metric(
        SCOPE_SEED, ROLE_SEED, ("p1_c",), lower=0.0
    ),
    "regime_disagreement": _metric(
        SCOPE_SEED, ROLE_SEED, ("p1_c",), lower=0.0, upper=1.0
    ),
    # Existing Stage-I optimization health/runtime fields.  Keeping their
    # established role makes this contract directly compatible with
    # reporting.stage1_aggregate.
    "initial_train_loss": _metric(SCOPE_SEED, ROLE_HEALTH),
    "final_train_loss": _metric(SCOPE_SEED, ROLE_HEALTH),
    "best_validation_loss": _metric(SCOPE_SEED, ROLE_HEALTH),
    "best_iter": _metric(SCOPE_SEED, ROLE_HEALTH, lower=0.0),
    "clip_frac": _metric(SCOPE_SEED, ROLE_HEALTH, lower=0.0, upper=1.0),
    "train_runtime_seconds": _metric(SCOPE_SEED, ROLE_HEALTH, lower=0.0),
    "evaluation_runtime_seconds": _metric(
        SCOPE_SEED, ROLE_HEALTH, lower=0.0
    ),
    "stage2_runtime_seconds": _metric(SCOPE_SEED, ROLE_HEALTH, lower=0.0),
    "total_runtime_seconds": _metric(SCOPE_SEED, ROLE_HEALTH, lower=0.0),
    "peak_gpu_memory_mb": _metric(SCOPE_SEED, ROLE_HEALTH, lower=0.0),
    "optimizer_success_rate": _metric(
        SCOPE_SEED, ROLE_HEALTH, lower=0.0, upper=1.0
    ),
    # A fixed P1-U evaluation bank/reference contributes these once, not once
    # per learned-policy seed.
    "J_exact": _metric(SCOPE_SHARED, ROLE_SHARED, ("p1_u",)),
    "J_oracle_mc": _metric(SCOPE_SHARED, ROLE_SHARED, ("p1_u",)),
    "mc_anchor_gap": _metric(SCOPE_SHARED, ROLE_SHARED, ("p1_u",)),
    "mc_anchor_gap_se": _metric(
        SCOPE_SHARED, ROLE_SHARED, ("p1_u",), lower=0.0
    ),
}

_REQUIRED_SEED = {
    "p1_u": frozenset(("J_policy", "control_nrmse", "dJ_paired", "dJ_se")),
    "p1_c": frozenset(
        (
            "J_policy",
            "J_baseline",
            "dJ_paired",
            "dJ_se",
            "constraint_violation_rate",
            "max_constraint_violation",
            "holdout_kkt_rms",
        )
    ),
}
_REQUIRED_STAGE2 = frozenset(
    (
        "solver_r_num_rms",
        "holdout_kkt_rms",
        "projection_activation_fraction",
        "projection_displacement_mean",
        "projection_displacement_max",
        "feasibility_violation_rate",
        "max_feasibility_violation",
        "recovery_denominator_min",
        "stage2_runtime_seconds",
    )
)
_REQUIRED_SHARED = {
    "p1_u": frozenset(
        ("J_exact", "J_oracle_mc", "mc_anchor_gap", "mc_anchor_gap_se")
    ),
    "p1_c": frozenset(),
}


def _json_copy(value: Any, label: str) -> Any:
    """Return a JSON-native deep copy while rejecting NaN/Inf and odd keys."""
    try:
        text = json.dumps(value, sort_keys=True, allow_nan=False, default=None)
        return json.loads(text)
    except (TypeError, ValueError) as exc:
        raise P1OutputContractError(f"{label} must be finite JSON data: {exc}") \
            from exc


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise P1OutputContractError(f"{label} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise P1OutputContractError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _finite_scalar(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise P1OutputContractError(f"{label} must be a numeric scalar")
    number = float(value)
    if not math.isfinite(number):
        raise P1OutputContractError(f"{label} must be finite")
    return value if isinstance(value, int) else number


def _normalise_metric_schema_entry(
    name: str,
    raw: Mapping[str, Any],
    *,
    variant: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise P1OutputContractError(
            f"metric_schema[{name!r}] must be an object"
        )
    allowed = {"scope", "role", "custom"}
    extras = set(raw) - allowed
    if extras:
        raise P1OutputContractError(
            f"metric_schema[{name!r}] has unknown fields: {sorted(extras)}"
        )
    scope = raw.get("scope")
    role = raw.get("role")
    custom = raw.get("custom", False)
    if scope not in (SCOPE_SEED, SCOPE_SHARED):
        raise P1OutputContractError(
            f"metric_schema[{name!r}].scope is invalid: {scope!r}"
        )
    if role not in _METRIC_ROLES:
        raise P1OutputContractError(
            f"metric_schema[{name!r}].role is invalid: {role!r}"
        )
    if not isinstance(custom, bool):
        raise P1OutputContractError(
            f"metric_schema[{name!r}].custom must be boolean"
        )
    if scope == SCOPE_SHARED and role != ROLE_SHARED:
        raise P1OutputContractError(
            f"shared metric {name!r} must use role {ROLE_SHARED!r}"
        )
    if scope == SCOPE_SEED and role == ROLE_SHARED:
        raise P1OutputContractError(
            f"seed metric {name!r} cannot use role {ROLE_SHARED!r}"
        )
    known = P1_METRIC_CATALOG.get(name)
    if known is None:
        if not custom:
            raise P1OutputContractError(
                f"unknown metric {name!r}; declare it explicitly with "
                "custom=true"
            )
    else:
        if custom:
            raise P1OutputContractError(
                f"known metric {name!r} must not be redeclared custom"
            )
        if variant not in known.variants:
            raise P1OutputContractError(
                f"metric {name!r} is not valid for variant {variant!r}"
            )
        if (scope, role) != (known.scope, known.role):
            raise P1OutputContractError(
                f"metric {name!r} must use scope/role "
                f"{known.scope!r}/{known.role!r}"
            )
    return {"scope": scope, "role": role, "custom": custom}


def _check_metric_bounds(
    name: str,
    value: int | float,
    *,
    variant: str,
) -> None:
    definition = P1_METRIC_CATALOG.get(name)
    if definition is None or variant not in definition.variants:
        return
    number = float(value)
    if definition.lower is not None and number < definition.lower:
        raise P1OutputContractError(
            f"metric {name!r}={number} is below {definition.lower}"
        )
    if definition.upper is not None and number > definition.upper:
        raise P1OutputContractError(
            f"metric {name!r}={number} exceeds {definition.upper}"
        )


def _normalise_rollout_bank(raw: Any) -> dict[str, Any]:
    """Validate the cheap learned-vs-reference rollout bank.

    This ``Np`` must not be inherited as the number of Stage-II branch states;
    the latter has a separate, explicitly budgeted block below.
    """
    if not isinstance(raw, Mapping):
        raise P1OutputContractError("evaluation.paired_rollout must be an object")
    required = {"Np", "seed", "bank_id", "common_random_numbers"}
    optional = {"initial_law", "metadata"}
    missing = required - set(raw)
    extras = set(raw) - required - optional
    if missing or extras:
        raise P1OutputContractError(
            f"invalid paired_rollout fields: missing={sorted(missing)}, "
            f"unknown={sorted(extras)}"
        )
    result = {
        "Np": _nonnegative_int(
            raw["Np"], "evaluation.paired_rollout.Np", minimum=2
        ),
        "seed": _nonnegative_int(
            raw["seed"], "evaluation.paired_rollout.seed"
        ),
        "bank_id": _nonempty_string(
            raw["bank_id"], "evaluation.paired_rollout.bank_id"
        ),
    }
    if not isinstance(raw["common_random_numbers"], bool):
        raise P1OutputContractError(
            "evaluation.paired_rollout.common_random_numbers must be boolean"
        )
    result["common_random_numbers"] = raw["common_random_numbers"]
    for key in optional:
        if key in raw:
            result[key] = _json_copy(
                raw[key], f"evaluation.paired_rollout.{key}"
            )
    return result


def _normalise_branch_bank(
    raw: Any,
    *,
    label: str,
    require_oracle_policy: bool,
    require_independent_of_recovery: bool = False,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise P1OutputContractError(f"{label} must be an object")
    required = {
        "states", "seed", "bank_id", "M", "M_out", "M_in",
        "branch_batch_size",
    }
    optional = {"policy", "metadata"}
    if require_independent_of_recovery:
        required.add("independent_of_recovery")
    missing = required - set(raw)
    extras = set(raw) - required - optional
    if missing or extras:
        raise P1OutputContractError(
            f"invalid {label} fields: missing={sorted(missing)}, "
            f"unknown={sorted(extras)}"
        )
    result = {
        "states": _nonnegative_int(
            raw["states"], f"{label}.states", minimum=1
        ),
        "seed": _nonnegative_int(raw["seed"], f"{label}.seed"),
        "bank_id": _nonempty_string(raw["bank_id"], f"{label}.bank_id"),
        "M": _nonnegative_int(raw["M"], f"{label}.M", minimum=1),
        "M_out": _nonnegative_int(
            raw["M_out"], f"{label}.M_out", minimum=2
        ),
        "M_in": _nonnegative_int(
            raw["M_in"], f"{label}.M_in", minimum=1
        ),
        "branch_batch_size": _nonnegative_int(
            raw["branch_batch_size"], f"{label}.branch_batch_size", minimum=1
        ),
    }
    total_nested = 2 * result["M_out"] * result["M_in"]
    if result["branch_batch_size"] > max(result["M"], total_nested):
        raise P1OutputContractError(
            f"{label}.branch_batch_size exceeds both statistical budgets"
        )
    if require_independent_of_recovery:
        independent = raw["independent_of_recovery"]
        if independent is not True:
            raise P1OutputContractError(
                f"{label}.independent_of_recovery must be true"
            )
        result["independent_of_recovery"] = True
    if require_oracle_policy:
        if raw.get("policy") != "exact_oracle_affine_feedback":
            raise P1OutputContractError(
                f"{label}.policy must be 'exact_oracle_affine_feedback'"
            )
        result["policy"] = raw["policy"]
    elif "policy" in raw:
        result["policy"] = _nonempty_string(raw["policy"], f"{label}.policy")
    if "metadata" in raw:
        result["metadata"] = _json_copy(raw["metadata"], f"{label}.metadata")
    return result


def _normalise_evaluation(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise P1OutputContractError("evaluation must be an object")
    required = {"paired_rollout"}
    optional = {
        "stage2_recovery_bank", "holdout_kkt_bank",
        "estimator_audit_bank", "metadata",
    }
    missing = required - set(raw)
    extras = set(raw) - required - optional
    if missing or extras:
        raise P1OutputContractError(
            f"invalid evaluation fields: missing={sorted(missing)}, "
            f"unknown={sorted(extras)}"
        )
    result = {"paired_rollout": _normalise_rollout_bank(raw["paired_rollout"])}
    if "stage2_recovery_bank" in raw:
        result["stage2_recovery_bank"] = _normalise_branch_bank(
            raw["stage2_recovery_bank"],
            label="evaluation.stage2_recovery_bank",
            require_oracle_policy=False,
        )
    if "holdout_kkt_bank" in raw:
        result["holdout_kkt_bank"] = _normalise_branch_bank(
            raw["holdout_kkt_bank"],
            label="evaluation.holdout_kkt_bank",
            require_oracle_policy=False,
            require_independent_of_recovery=True,
        )
    if "estimator_audit_bank" in raw:
        result["estimator_audit_bank"] = _normalise_branch_bank(
            raw["estimator_audit_bank"],
            label="evaluation.estimator_audit_bank",
            require_oracle_policy=True,
        )
    if "metadata" in raw:
        result["metadata"] = _json_copy(raw["metadata"], "evaluation.metadata")
    return result


def _normalise_projection_provenance(raw: Any) -> dict[str, str]:
    """Validate the projection implementation used by one Stage-II seed."""

    if not isinstance(raw, Mapping):
        raise P1OutputContractError("seed_result.projection must be an object")
    required = {"mode", "api_version", "config_hash"}
    missing = required - set(raw)
    extras = set(raw) - required
    if missing or extras:
        raise P1OutputContractError(
            "invalid seed_result.projection fields: "
            f"missing={sorted(missing)}, unknown={sorted(extras)}"
        )
    mode = raw["mode"]
    if mode not in _PROJECTION_MODES:
        raise P1OutputContractError(
            f"invalid seed_result.projection.mode: {mode!r}"
        )
    return {
        "mode": mode,
        "api_version": _nonempty_string(
            raw["api_version"], "seed_result.projection.api_version"
        ),
        "config_hash": _nonempty_string(
            raw["config_hash"], "seed_result.projection.config_hash"
        ),
    }


def _normalise_reference(
    raw: Any, *, variant: str, problem_config_hash: str
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise P1OutputContractError("reference must be an object")
    required = {
        "method", "role", "api_version", "problem_config_hash", "config_hash"
    }
    optional = {"metadata"}
    missing = required - set(raw)
    extras = set(raw) - required - optional
    if missing or extras:
        raise P1OutputContractError(
            f"invalid reference fields: missing={sorted(missing)}, "
            f"unknown={sorted(extras)}"
        )
    role = raw["role"]
    if role not in _REFERENCE_ROLES:
        raise P1OutputContractError(f"invalid reference.role: {role!r}")
    if variant == "p1_u" and role != "exact_oracle":
        raise P1OutputContractError(
            "p1_u requires an exact_oracle reference"
        )
    if variant == "p1_c" and role == "exact_oracle":
        raise P1OutputContractError(
            "p1_c main has no global exact oracle; use a learned_baseline or "
            "numerical_benchmark reference"
        )
    result = {
        "method": _nonempty_string(raw["method"], "reference.method"),
        "role": role,
        "api_version": _nonempty_string(
            raw["api_version"], "reference.api_version"
        ),
        "problem_config_hash": _nonempty_string(
            raw["problem_config_hash"], "reference.problem_config_hash"
        ),
        "config_hash": _nonempty_string(
            raw["config_hash"], "reference.config_hash"
        ),
    }
    if result["problem_config_hash"] != problem_config_hash:
        raise P1OutputContractError(
            "reference problem_config_hash disagrees with the comparison"
        )
    if "metadata" in raw:
        result["metadata"] = _json_copy(
            raw["metadata"], "reference.metadata"
        )
    return result


def _normalise_seed_result(
    raw: Any,
    *,
    variant: str,
    metric_schema: Mapping[str, Mapping[str, Any]],
    problem_config_hash: str,
    require_complete: bool,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise P1OutputContractError("each seed_results entry must be an object")
    required = {
        "method", "method_role", "seed", "problem_config_hash", "config_hash",
        "run_fingerprint", "metrics",
    }
    optional = {"metadata", "projection"}
    missing = required - set(raw)
    extras = set(raw) - required - optional
    if missing or extras:
        raise P1OutputContractError(
            f"invalid seed result fields: missing={sorted(missing)}, "
            f"unknown={sorted(extras)}"
        )
    method_role = raw["method_role"]
    if method_role not in _METHOD_ROLES:
        raise P1OutputContractError(
            f"invalid method_role: {method_role!r}"
        )
    metrics_raw = raw["metrics"]
    if not isinstance(metrics_raw, Mapping) or not metrics_raw:
        raise P1OutputContractError("seed result metrics must be a nonempty object")
    metrics: dict[str, int | float] = {}
    for name, value in metrics_raw.items():
        if not isinstance(name, str) or not name:
            raise P1OutputContractError("metric names must be non-empty strings")
        spec = metric_schema.get(name)
        if spec is None:
            raise P1OutputContractError(
                f"seed metric {name!r} is absent from metric_schema"
            )
        if spec["scope"] != SCOPE_SEED:
            raise P1OutputContractError(
                f"shared diagnostic {name!r} cannot be repeated per seed"
            )
        metrics[name] = _finite_scalar(value, f"metrics.{name}")
        _check_metric_bounds(name, metrics[name], variant=variant)
    if require_complete:
        required_metrics = set(_REQUIRED_SEED[variant])
        if method_role == "stage2":
            required_metrics.update(_REQUIRED_STAGE2)
        absent = sorted(required_metrics - set(metrics))
        if absent:
            raise P1OutputContractError(
                f"{variant} seed result lacks required metrics: {absent}"
            )
    result = {
        "method": _nonempty_string(raw["method"], "seed_result.method"),
        "method_role": method_role,
        "seed": _nonnegative_int(raw["seed"], "seed_result.seed"),
        "problem_config_hash": _nonempty_string(
            raw["problem_config_hash"], "seed_result.problem_config_hash"
        ),
        "config_hash": _nonempty_string(
            raw["config_hash"], "seed_result.config_hash"
        ),
        "run_fingerprint": _nonempty_string(
            raw["run_fingerprint"], "seed_result.run_fingerprint"
        ),
        "metrics": dict(sorted(metrics.items())),
    }
    if method_role == "stage2":
        if "projection" not in raw:
            raise P1OutputContractError(
                "Stage-II seed result requires projection provenance"
            )
        result["projection"] = _normalise_projection_provenance(
            raw["projection"]
        )
    elif "projection" in raw:
        raise P1OutputContractError(
            "projection provenance is valid only for Stage-II seed results"
        )
    if result["problem_config_hash"] != problem_config_hash:
        raise P1OutputContractError(
            "seed_result problem_config_hash disagrees with the comparison"
        )
    if "metadata" in raw:
        result["metadata"] = _json_copy(
            raw["metadata"], "seed_result.metadata"
        )
    return result


def _check_active_partition(result: Mapping[str, Any]) -> None:
    names = (
        "active_lower_fraction",
        "active_interior_fraction",
        "active_upper_fraction",
    )
    metrics = result["metrics"]
    present = [name in metrics for name in names]
    if any(present) and not all(present):
        raise P1OutputContractError(
            "active-set occupancy must provide lower/interior/upper together"
        )
    if all(present):
        total = sum(float(metrics[name]) for name in names)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise P1OutputContractError(
                f"active-set occupancy fractions sum to {total}, not 1"
            )


def _require_objective_identity(
    *, label: str, observed: int | float, expected: int | float
) -> None:
    """Reject a reported derived objective that disagrees with its anchors."""

    if not math.isclose(
        float(observed),
        float(expected),
        rel_tol=_OBJECTIVE_REL_TOL,
        abs_tol=_OBJECTIVE_ABS_TOL,
    ):
        raise P1OutputContractError(
            f"{label}={float(observed):.17g} disagrees with the required "
            f"arithmetic value {float(expected):.17g}"
        )


def _check_objective_arithmetic(
    *,
    variant: str,
    seed_results: Sequence[Mapping[str, Any]],
    shared: Mapping[str, int | float],
) -> None:
    """Bind paired differences and the P1-U MC anchor gap to raw means."""

    if variant == "p1_u":
        if {"J_oracle_mc", "J_exact", "mc_anchor_gap"} <= set(shared):
            _require_objective_identity(
                label="shared_diagnostics.mc_anchor_gap",
                observed=shared["mc_anchor_gap"],
                expected=float(shared["J_oracle_mc"])
                - float(shared["J_exact"]),
            )
        for result in seed_results:
            metrics = result["metrics"]
            if {"J_policy", "dJ_paired"} <= set(metrics) and \
                    "J_oracle_mc" in shared:
                _require_objective_identity(
                    label=(
                        f"seed_result[{result['method']!r},"
                        f"{result['seed']}].dJ_paired"
                    ),
                    observed=metrics["dJ_paired"],
                    expected=float(metrics["J_policy"])
                    - float(shared["J_oracle_mc"]),
                )
        return

    for result in seed_results:
        metrics = result["metrics"]
        if {"J_policy", "J_baseline", "dJ_paired"} <= set(metrics):
            _require_objective_identity(
                label=(
                    f"seed_result[{result['method']!r},"
                    f"{result['seed']}].dJ_paired"
                ),
                observed=metrics["dJ_paired"],
                expected=float(metrics["J_policy"])
                - float(metrics["J_baseline"]),
            )


def validate_p1_output_contract(
    payload: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Return a canonical validated copy of a P1 scientific payload.

    Publication-only fields (``artifact_id`` and ``generated_at``) are
    accepted and retained, but are not needed when building a payload in
    memory.  Content-hash validation is performed by the on-disk reader.
    """
    if not isinstance(payload, Mapping):
        raise P1OutputContractError("P1 output payload must be an object")
    required = {
        "schema", "problem", "variant", "problem_config_hash",
        "evaluation", "reference",
        "metric_schema", "seed_results", "shared_diagnostics",
    }
    optional = {"metadata", "artifact_id", "generated_at"}
    missing = required - set(payload)
    extras = set(payload) - required - optional
    if missing or extras:
        raise P1OutputContractError(
            f"invalid top-level fields: missing={sorted(missing)}, "
            f"unknown={sorted(extras)}"
        )
    if payload["schema"] != P1_OUTPUT_SCHEMA_VERSION:
        raise P1OutputContractError(
            f"unsupported P1 output schema: {payload['schema']!r}"
        )
    if payload["problem"] != "p1":
        raise P1OutputContractError("problem must be exactly 'p1'")
    variant = payload["variant"]
    if variant not in _VARIANTS:
        raise P1OutputContractError(f"invalid P1 variant: {variant!r}")
    problem_config_hash = _nonempty_string(
        payload["problem_config_hash"], "problem_config_hash"
    )

    raw_schema = payload["metric_schema"]
    if not isinstance(raw_schema, Mapping) or not raw_schema:
        raise P1OutputContractError("metric_schema must be a nonempty object")
    invalid_schema_names = [
        name for name in raw_schema if not isinstance(name, str) or not name
    ]
    if invalid_schema_names:
        raise P1OutputContractError(
            "metric_schema names must be non-empty strings"
        )
    metric_schema = {
        name: _normalise_metric_schema_entry(
            name, spec, variant=variant
        )
        for name, spec in sorted(raw_schema.items())
    }

    raw_results = payload["seed_results"]
    if not isinstance(raw_results, list):
        raise P1OutputContractError("seed_results must be a list")
    if require_complete and not raw_results:
        raise P1OutputContractError("a complete comparison needs seed_results")
    seed_results = [
        _normalise_seed_result(
            result,
            variant=variant,
            metric_schema=metric_schema,
            problem_config_hash=problem_config_hash,
            require_complete=require_complete,
        )
        for result in raw_results
    ]
    keys = [(item["method"], item["seed"]) for item in seed_results]
    if len(set(keys)) != len(keys):
        raise P1OutputContractError(
            "duplicate (method, seed) entries in seed_results"
        )
    method_roles: dict[str, str] = {}
    for item in seed_results:
        previous = method_roles.setdefault(item["method"], item["method_role"])
        if previous != item["method_role"]:
            raise P1OutputContractError(
                f"method {item['method']!r} has conflicting method roles"
            )
        if variant == "p1_c":
            _check_active_partition(item)

    raw_shared = payload["shared_diagnostics"]
    if not isinstance(raw_shared, Mapping):
        raise P1OutputContractError("shared_diagnostics must be an object")
    shared: dict[str, int | float] = {}
    for name, value in raw_shared.items():
        if not isinstance(name, str) or not name:
            raise P1OutputContractError(
                "shared diagnostic names must be non-empty strings"
            )
        spec = metric_schema.get(name)
        if spec is None:
            raise P1OutputContractError(
                f"shared diagnostic {name!r} is absent from metric_schema"
            )
        if spec["scope"] != SCOPE_SHARED:
            raise P1OutputContractError(
                f"seed metric {name!r} cannot be a shared diagnostic"
            )
        shared[name] = _finite_scalar(value, f"shared_diagnostics.{name}")
        _check_metric_bounds(name, shared[name], variant=variant)

    used_seed = set().union(
        *(set(item["metrics"]) for item in seed_results), set()
    )
    used = used_seed | set(shared)
    stale = sorted(set(metric_schema) - used)
    if stale:
        raise P1OutputContractError(
            f"metric_schema declares unused metrics: {stale}"
        )
    if require_complete:
        missing_shared = sorted(_REQUIRED_SHARED[variant] - set(shared))
        if missing_shared:
            raise P1OutputContractError(
                f"{variant} lacks required shared diagnostics: "
                f"{missing_shared}"
            )

    _check_objective_arithmetic(
        variant=variant, seed_results=seed_results, shared=shared
    )

    evaluation = _normalise_evaluation(payload["evaluation"])
    if require_complete and not evaluation["paired_rollout"][
            "common_random_numbers"]:
        raise P1OutputContractError(
            "a complete P1 comparison requires "
            "evaluation.paired_rollout.common_random_numbers=true"
        )
    has_stage2 = any(
        result["method_role"] == "stage2" for result in seed_results
    )
    if has_stage2:
        missing_stage2_banks = sorted(
            {"stage2_recovery_bank", "holdout_kkt_bank"} - set(evaluation)
        )
        if missing_stage2_banks:
            raise P1OutputContractError(
                "Stage-II seed results require independent recovery and "
                f"holdout banks; missing evaluation fields: "
                f"{missing_stage2_banks}"
            )
    has_holdout_kkt = any(
        {"holdout_kkt_rms", "holdout_kkt_max"} & set(result["metrics"])
        for result in seed_results
    )
    if has_holdout_kkt and "holdout_kkt_bank" not in evaluation:
        raise P1OutputContractError(
            "holdout_kkt metrics require evaluation.holdout_kkt_bank"
        )
    if {"stage2_recovery_bank", "holdout_kkt_bank"} <= set(evaluation):
        recovery_bank = evaluation["stage2_recovery_bank"]
        holdout_bank = evaluation["holdout_kkt_bank"]
        if recovery_bank["bank_id"] == holdout_bank["bank_id"]:
            raise P1OutputContractError(
                "stage2_recovery_bank and holdout_kkt_bank must use "
                "different bank_id values"
            )
        if recovery_bank["seed"] == holdout_bank["seed"]:
            raise P1OutputContractError(
                "stage2_recovery_bank and holdout_kkt_bank must use "
                "different seeds"
            )
    audit_names = {
        "audit_p_cur_nrmse", "audit_p_nxt_nrmse", "audit_q_nrmse",
        "audit_Pi_nrmse", "audit_zeta_nrmse",
        "finite_h_pcur_pnxt_action_nrmse",
    }
    if set(shared) & audit_names and "estimator_audit_bank" not in evaluation:
        raise P1OutputContractError(
            "oracle-policy estimator audit metrics require "
            "evaluation.estimator_audit_bank"
        )

    result: dict[str, Any] = {
        "schema": P1_OUTPUT_SCHEMA_VERSION,
        "problem": "p1",
        "variant": variant,
        "problem_config_hash": problem_config_hash,
        "evaluation": evaluation,
        "reference": _normalise_reference(
            payload["reference"],
            variant=variant,
            problem_config_hash=problem_config_hash,
        ),
        "metric_schema": metric_schema,
        "seed_results": sorted(
            seed_results,
            key=lambda item: (
                item["method_role"], item["method"], item["seed"]
            ),
        ),
        "shared_diagnostics": dict(sorted(shared.items())),
    }
    if "metadata" in payload:
        result["metadata"] = _json_copy(payload["metadata"], "metadata")
    for field in ("artifact_id", "generated_at"):
        if field in payload:
            result[field] = _nonempty_string(payload[field], field)
    return result


def _schema_for_metrics(
    names: set[str],
    *,
    variant: str,
    extra_metric_schema: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    invalid_names = [
        name for name in names if not isinstance(name, str) or not name
    ]
    if invalid_names:
        raise P1OutputContractError(
            "metric names must be non-empty strings"
        )
    extra = dict(extra_metric_schema or {})
    unknown_extra = sorted(set(extra) - names)
    if unknown_extra:
        raise P1OutputContractError(
            f"extra_metric_schema declares unused metrics: {unknown_extra}"
        )
    schema: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        known = P1_METRIC_CATALOG.get(name)
        if known is not None:
            if variant not in known.variants:
                raise P1OutputContractError(
                    f"metric {name!r} is not valid for {variant}"
                )
            schema[name] = {
                "scope": known.scope,
                "role": known.role,
                "custom": False,
            }
            if name in extra:
                raise P1OutputContractError(
                    f"known metric {name!r} must not be overridden"
                )
            continue
        if name not in extra:
            raise P1OutputContractError(
                f"unknown metric {name!r}; add an explicit "
                "extra_metric_schema entry"
            )
        declared = dict(extra[name])
        declared["custom"] = True
        schema[name] = _normalise_metric_schema_entry(
            name, declared, variant=variant
        )
    return schema


def build_p1_output_contract(
    *,
    variant: str,
    problem_config_hash: str,
    evaluation: Mapping[str, Any],
    reference: Mapping[str, Any],
    seed_results: Sequence[Mapping[str, Any]],
    shared_diagnostics: Mapping[str, Real],
    metadata: Mapping[str, Any] | None = None,
    extra_metric_schema: Mapping[str, Mapping[str, Any]] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Build a canonical P1 comparison payload from method seed records."""
    if variant not in _VARIANTS:
        raise P1OutputContractError(f"invalid P1 variant: {variant!r}")
    records = [deepcopy(dict(record)) for record in seed_results]
    shared = dict(shared_diagnostics)
    names = set(shared)
    for result in records:
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise P1OutputContractError(
                "each seed result must contain a metrics object"
            )
        names.update(metrics)
    payload: dict[str, Any] = {
        "schema": P1_OUTPUT_SCHEMA_VERSION,
        "problem": "p1",
        "variant": variant,
        "problem_config_hash": problem_config_hash,
        "evaluation": deepcopy(dict(evaluation)),
        "reference": deepcopy(dict(reference)),
        "metric_schema": _schema_for_metrics(
            names,
            variant=variant,
            extra_metric_schema=extra_metric_schema,
        ),
        "seed_results": records,
        "shared_diagnostics": shared,
    }
    if metadata is not None:
        payload["metadata"] = deepcopy(dict(metadata))
    return validate_p1_output_contract(
        payload, require_complete=require_complete
    )


def _scientific_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop("artifact_id", None)
    result.pop("generated_at", None)
    return result


def _artifact_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _scientific_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return format(float(value), ".17g")


def _csv_rows(payload: Mapping[str, Any]) -> list[dict[str, str | int]]:
    artifact_id = payload["artifact_id"]
    base = {
        "schema": P1_OUTPUT_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "problem": "p1",
        "variant": payload["variant"],
        "problem_config_hash": payload["problem_config_hash"],
    }
    rows: list[dict[str, str | int]] = []
    for result in payload["seed_results"]:
        for name, value in sorted(result["metrics"].items()):
            rows.append(
                {
                    **base,
                    "method": result["method"],
                    "method_role": result["method_role"],
                    "seed": result["seed"],
                    "config_hash": result["config_hash"],
                    "run_fingerprint": result["run_fingerprint"],
                    "scope": SCOPE_SEED,
                    "metric": name,
                    "role": payload["metric_schema"][name]["role"],
                    "value": _format_number(value),
                }
            )
    reference = payload["reference"]
    for name, value in sorted(payload["shared_diagnostics"].items()):
        rows.append(
            {
                **base,
                "method": reference["method"],
                "method_role": "reference",
                "seed": "",
                "config_hash": reference["config_hash"],
                "run_fingerprint": "",
                "scope": SCOPE_SHARED,
                "metric": name,
                "role": payload["metric_schema"][name]["role"],
                "value": _format_number(value),
            }
        )
    return rows


def _contract_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": P1_OUTPUT_SCHEMA_VERSION,
        "problem": "p1",
        "variant": payload["variant"],
        "problem_config_hash": payload["problem_config_hash"],
        "evaluation": payload["evaluation"],
        "reference": payload["reference"],
        "metric_schema": payload["metric_schema"],
    }


def publish_p1_output_contract(
    root: str | Path,
    payload: Mapping[str, Any],
    *,
    tier: str = DEFAULT_TIER,
    basename: str = DEFAULT_BASENAME,
    require_complete: bool = True,
) -> P1OutputArtifacts:
    """Jointly publish validated JSON and CSV under one atomic pointer."""
    if not isinstance(basename, str) or not basename or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in basename
    ):
        raise P1OutputContractError(
            "basename may contain only letters, digits, '_' and '-'"
        )
    canonical = validate_p1_output_contract(
        payload, require_complete=require_complete
    )
    canonical.pop("artifact_id", None)
    canonical.pop("generated_at", None)
    canonical["artifact_id"] = _artifact_id(canonical)
    canonical["generated_at"] = datetime.now(timezone.utc).isoformat()
    json_name = f"{basename}.json"
    csv_name = f"{basename}.csv"
    root = Path(root)
    with begin_bundle(root, tier) as transaction:
        write_manifest(
            transaction.stage_dir,
            problem="p1",
            method="p1_output_contract",
            config=_contract_config(canonical),
            seeds={
                "train": sorted(
                    {result["seed"] for result in canonical["seed_results"]}
                ),
                "evaluation": canonical["evaluation"]["paired_rollout"]["seed"],
            },
            device="reporting-only",
            api_versions={"p1_output_contract": P1_OUTPUT_SCHEMA_VERSION},
            solver="comparison-reporting",
            extra={
                "artifact_id": canonical["artifact_id"],
                "record_count": len(canonical["seed_results"]),
                "reference_role": canonical["reference"]["role"],
            },
        )
        # write_manifest already uses atomic JSON writes.  The shared Stage-I
        # CSV writer provides the same fsync + sibling-temp + replace contract.
        from pgdpo_delay.core.artifacts import atomic_write_json

        atomic_write_json(transaction.stage_dir / json_name, canonical)
        _atomic_write_csv(
            transaction.stage_dir / csv_name,
            _CSV_FIELDS,
            _csv_rows(canonical),
        )
        bundle = transaction.publish(required_files=(json_name, csv_name))
    return P1OutputArtifacts(
        bundle_dir=bundle,
        json_path=bundle / json_name,
        csv_path=bundle / csv_name,
        pointer_path=root / f"current-{tier}.json",
        payload=canonical,
    )


def _regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise P1OutputContractError(
            f"{label} is not a regular non-symlink file: {path}"
        )


def read_p1_output_contract(
    path: str | Path,
    *,
    tier: str = DEFAULT_TIER,
    basename: str = DEFAULT_BASENAME,
    require_complete: bool = True,
) -> P1OutputReadResult:
    """Read either a bundle directory or the root containing its pointer."""
    candidate = Path(path)
    if (candidate / f"{basename}.json").is_file():
        bundle = candidate.resolve()
    else:
        try:
            resolved = resolve_current_bundle(candidate, tier)
        except (RuntimeError, ValueError) as exc:
            raise P1OutputContractError(str(exc)) from exc
        if resolved is None:
            raise P1OutputContractError(
                f"no current P1 output bundle for tier {tier!r} under {candidate}"
            )
        bundle = resolved
    json_path = bundle / f"{basename}.json"
    csv_path = bundle / f"{basename}.csv"
    manifest_path = bundle / "manifest.json"
    config_path = bundle / "config.json"
    for file_path, label in (
        (json_path, "P1 output JSON"),
        (csv_path, "P1 output CSV"),
        (manifest_path, "manifest"),
        (config_path, "config"),
    ):
        _regular_file(file_path, label)
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P1OutputContractError(f"invalid P1 output JSON metadata: {exc}") \
            from exc
    canonical = validate_p1_output_contract(
        raw, require_complete=require_complete
    )
    artifact_id = canonical.get("artifact_id")
    if artifact_id != _artifact_id(canonical):
        raise P1OutputContractError(
            "P1 output artifact_id does not match scientific JSON content"
        )
    expected_config = _contract_config(canonical)
    if config != expected_config:
        raise P1OutputContractError(
            "config.json disagrees with the P1 output contract"
        )
    if not isinstance(manifest, Mapping):
        raise P1OutputContractError("manifest.json must contain an object")
    if manifest.get("problem") != "p1" or \
            manifest.get("method") != "p1_output_contract":
        raise P1OutputContractError("manifest identity is not a P1 output contract")
    if manifest.get("config_hash") != config_hash(config):
        raise P1OutputContractError("manifest config_hash disagrees with config.json")
    extra = manifest.get("extra")
    if not isinstance(extra, Mapping) or extra.get("artifact_id") != artifact_id:
        raise P1OutputContractError(
            "manifest artifact_id disagrees with the P1 output contract"
        )

    try:
        with open(csv_path, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != _CSV_FIELDS:
                raise P1OutputContractError(
                    f"invalid P1 output CSV header: {reader.fieldnames}"
                )
            actual_rows = list(reader)
    except OSError as exc:
        raise P1OutputContractError(f"cannot read P1 output CSV: {exc}") \
            from exc
    expected_rows = [
        {field: str(row[field]) for field in _CSV_FIELDS}
        for row in _csv_rows(canonical)
    ]
    if actual_rows != expected_rows:
        raise P1OutputContractError(
            "P1 output CSV disagrees with the canonical JSON payload"
        )
    return P1OutputReadResult(
        bundle_dir=bundle,
        json_path=json_path,
        csv_path=csv_path,
        payload=canonical,
    )


def from_stage1_aggregate(
    aggregate: Stage1AggregationResult,
    *,
    problem_config_hash: str | None = None,
    evaluation: Mapping[str, Any],
    reference: Mapping[str, Any],
    variant: str = "p1_u",
    method_role: str = "stage1",
    metadata: Mapping[str, Any] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Convert a validated Stage-I aggregate without duplicating anchors.

    ``shared_evaluation_diagnostic`` metrics are copied once from
    ``summary.json``.  All other metric roles remain attached to each seed.
    Thus the existing Stage-I worker/aggregator can feed the P1-wide contract
    without pretending repeated exact-oracle values are independent seeds.
    """
    if not isinstance(aggregate, Stage1AggregationResult):
        raise TypeError("aggregate must be a Stage1AggregationResult")
    payload = aggregate.payload
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise P1OutputContractError("Stage-I aggregate lacks identity metadata")
    aggregate_problem_hash = identity.get("problem_config_hash")
    if not isinstance(aggregate_problem_hash, str) or not aggregate_problem_hash:
        raise P1OutputContractError(
            "Stage-I aggregate lacks its worker-derived problem_config_hash"
        )
    if problem_config_hash is not None and \
            problem_config_hash != aggregate_problem_hash:
        raise P1OutputContractError(
            "caller problem_config_hash cannot re-stamp the Stage-I aggregate"
        )
    problem_config_hash = aggregate_problem_hash
    counts = payload.get("counts")
    if not isinstance(counts, Mapping) or counts.get("incomplete") != 0:
        raise P1OutputContractError(
            "P1 output conversion requires a complete Stage-I aggregate"
        )
    summaries = payload.get("metrics")
    if not isinstance(summaries, Mapping) or not summaries:
        raise P1OutputContractError("Stage-I aggregate has no metric summaries")
    roles: dict[str, str] = {}
    for name, statistics in summaries.items():
        if not isinstance(statistics, Mapping):
            raise P1OutputContractError(
                f"invalid Stage-I summary for metric {name!r}"
            )
        role = statistics.get("role")
        if role not in _METRIC_ROLES:
            raise P1OutputContractError(
                f"invalid Stage-I metric role for {name!r}: {role!r}"
            )
        roles[name] = role

    try:
        with open(aggregate.per_seed_csv, newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise P1OutputContractError(
            f"cannot read Stage-I per_seed.csv: {exc}"
        ) from exc
    if len(rows) != counts.get("complete"):
        raise P1OutputContractError(
            "Stage-I per_seed.csv row count disagrees with summary.json"
        )
    shared_names = {name for name, role in roles.items() if role == ROLE_SHARED}
    seed_names = set(roles) - shared_names
    seed_results = []
    for row in rows:
        if row.get("status") != "COMPLETE":
            raise P1OutputContractError(
                "Stage-I per_seed.csv contains a non-COMPLETE row"
            )
        if row.get("problem_config_hash") != problem_config_hash:
            raise P1OutputContractError(
                "Stage-I per_seed.csv problem_config_hash disagrees with "
                "summary.json"
            )
        metrics = {}
        for name in seed_names:
            text = row.get(name)
            if text in (None, ""):
                raise P1OutputContractError(
                    f"Stage-I seed {row.get('seed')} lacks metric {name!r}"
                )
            try:
                metrics[name] = float(text)
            except ValueError as exc:
                raise P1OutputContractError(
                    f"Stage-I metric {name!r} is not numeric: {text!r}"
                ) from exc
        try:
            seed = int(row["seed"])
        except (KeyError, ValueError) as exc:
            raise P1OutputContractError(
                f"invalid Stage-I seed row: {row.get('seed')!r}"
            ) from exc
        seed_results.append(
            {
                "method": row.get("method"),
                "method_role": method_role,
                "seed": seed,
                "problem_config_hash": problem_config_hash,
                "config_hash": row.get("config_hash"),
                "run_fingerprint": row.get("run_fingerprint"),
                "metrics": metrics,
                "metadata": {"device": row.get("device", "")},
            }
        )
    shared = {}
    for name in shared_names:
        value = summaries[name].get("mean")
        shared[name] = _finite_scalar(
            value, f"Stage-I shared diagnostic {name}"
        )

    extra_schema: dict[str, dict[str, Any]] = {}
    for name, role in roles.items():
        if name in P1_METRIC_CATALOG:
            known = P1_METRIC_CATALOG[name]
            if (known.scope, known.role) != (
                SCOPE_SHARED if role == ROLE_SHARED else SCOPE_SEED,
                role,
            ):
                raise P1OutputContractError(
                    f"Stage-I role for known metric {name!r} disagrees with "
                    "the P1 catalog"
                )
        else:
            extra_schema[name] = {
                "scope": SCOPE_SHARED if role == ROLE_SHARED else SCOPE_SEED,
                "role": role,
            }
    return build_p1_output_contract(
        variant=variant,
        problem_config_hash=problem_config_hash,
        evaluation=evaluation,
        reference=reference,
        seed_results=seed_results,
        shared_diagnostics=shared,
        metadata=metadata,
        extra_metric_schema=extra_schema,
        require_complete=require_complete,
    )


__all__ = [
    "DEFAULT_BASENAME",
    "DEFAULT_TIER",
    "P1_METRIC_CATALOG",
    "P1_OUTPUT_SCHEMA_VERSION",
    "P1OutputArtifacts",
    "P1OutputContractError",
    "P1OutputReadResult",
    "ROLE_HEALTH",
    "ROLE_PAIRED_SE",
    "ROLE_SEED",
    "ROLE_SHARED",
    "SCOPE_SEED",
    "SCOPE_SHARED",
    "build_p1_output_contract",
    "from_stage1_aggregate",
    "publish_p1_output_contract",
    "read_p1_output_contract",
    "validate_p1_output_contract",
]
