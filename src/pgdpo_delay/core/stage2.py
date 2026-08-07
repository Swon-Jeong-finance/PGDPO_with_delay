"""Backend-neutral Stage-II recovery pipeline.

This module owns the order shared by every benchmark::

    raw (p, zeta, Pi, sigma_ref)
      -> q_anc = zeta + Pi @ sigma_ref
      -> project p, q_anc, and sym(Pi) in three independent blocks
      -> zeta_N = q_anc_N - Pi_N @ sigma_ref
      -> problem-local solve over the selected feasible recovery set
      -> normal-cone residual of that same projected objective.

The estimator and the problem Hamiltonian deliberately do not live here.
Callbacks supply those mathematical pieces.  The core neither imports torch
nor converts NumPy/Torch values: dtype, device, and autodiff identity are
preserved through preparation.  Only scalar diagnostics are copied to CPU.

Sign convention
---------------
The manuscript maximises a Hamiltonian and reports
``dist(0, -D H(u) + N_C(u))``.  Cost-based benchmark code minimises and reports
``dist(0, D J(u) + N_C(u))``.  Callers must therefore state ``sense``; there
is intentionally no default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

import numpy as np


ObjectiveSense = Literal["minimize", "maximize"]
Projector = Callable[[Any], Any]
STAGE2_SCHEMA_VERSION = 1


def identity_projection(value: Any) -> Any:
    """Explicit identity block, useful for the projection-audit protocol."""

    return value


@dataclass(frozen=True)
class RawRecoveryInputs:
    """Raw fixed-control estimator output at one or more anchors.

    ``anchor`` is opaque caller context (normally the observed state/history
    and deployed ``u_ref``).  It is carried through unchanged and is never
    projected.  Keeping it together with ``sigma_ref`` prevents accidental
    reconstruction at a different state or action.
    """

    p: Any
    zeta: Any
    Pi: Any
    sigma_ref: Any
    anchor: Any = None
    pi_layout: Literal["auto", "matrix", "scalar"] = "auto"


@dataclass(frozen=True)
class BlockProjectionStats:
    activation_fraction: float
    displacement_mean: float
    displacement_max: float
    displacement_l2: float


@dataclass(frozen=True)
class ProjectionDiagnostics:
    p: BlockProjectionStats
    q_anc: BlockProjectionStats
    Pi: BlockProjectionStats
    pi_symmetrization_max: float
    coordinate_identity_max: float

    @property
    def activated(self) -> bool:
        return any(
            block.activation_fraction > 0.0
            for block in (self.p, self.q_anc, self.Pi)
        )


@dataclass(frozen=True)
class ProjectedRecoveryInputs:
    """Projected tuple used by both the local solve and its residual."""

    p: Any
    q_anc: Any
    Pi: Any
    zeta: Any
    sigma_ref: Any = None
    anchor: Any = None
    diagnostics: ProjectionDiagnostics | None = None
    pi_layout: Literal["matrix", "scalar"] = "scalar"


@dataclass(frozen=True)
class ProjectionBlocks:
    """Independent projectors for the three adjoint blocks.

    There is no zeta projector by design.  A single joint tuple projector is
    not accepted because it would invalidate the product-set projection used
    in the recovery proof.
    """

    p: Projector = identity_projection
    q_anc: Projector = identity_projection
    Pi: Projector = identity_projection

    @classmethod
    def identity(cls) -> "ProjectionBlocks":
        return cls()


@dataclass(frozen=True)
class LocalRecoveryOutput:
    """Output of the problem-local numerical solve."""

    action: Any
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    exact: bool = False


@dataclass(frozen=True)
class NormalConeResidualData:
    """Solver residual for the *projected* objective used in recovery.

    ``objective_gradient`` is the gradient returned by the callback.
    ``stationarity_gradient`` is ``+gradient`` for minimisation and
    ``-gradient`` for maximisation.  ``pointwise`` stores the exact distance
    supplied by the selected recovery set, before aggregation.
    """

    sense: ObjectiveSense
    objective_gradient: Any
    stationarity_gradient: Any
    pointwise: Any
    mean: float
    maximum: float
    rms: float
    role: str = "projected-objective numerical normal-cone residual"


@dataclass(frozen=True)
class Stage2Result:
    inputs: ProjectedRecoveryInputs
    action: Any
    feasibility_violation: Any
    feasibility_max: float
    residual: NormalConeResidualData
    solver_diagnostics: Mapping[str, Any]
    recovery_health: Mapping[str, Any]
    exact_solve: bool
    schema_version: int = STAGE2_SCHEMA_VERSION


class RecoverySet(Protocol):
    """Geometry needed by the common feasibility and residual gates."""

    def violation(self, action: Any) -> Any:
        """Return a non-negative pointwise distance/violation."""

    def normal_cone_residual(
        self, action: Any, stationarity_gradient: Any
    ) -> Any:
        """Return ``dist(0, g + N_C(action))`` pointwise."""


def _is_torch(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "torch"


def _is_numpy(value: Any) -> bool:
    return isinstance(value, (np.ndarray, np.generic))


def _shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, Number):
        return ()
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(
            "Stage-II values must be real scalars, numpy arrays, or "
            "tensor-like objects with shape"
        )
    return tuple(int(v) for v in shape)


def _ndim(value: Any) -> int:
    return len(_shape(value))


def _normalise_container(value: Any) -> Any:
    # Lists have no backend identity to preserve.  NumPy and torch values are
    # returned verbatim.
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


def _to_numpy(value: Any) -> np.ndarray:
    if _is_torch(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _ensure_real_finite(name: str, value: Any) -> None:
    _shape(value)
    if _is_torch(value):
        is_complex = getattr(value, "is_complex", None)
        if callable(is_complex) and is_complex():
            raise TypeError(f"{name} must be real")
        ok = bool(value.isfinite().all().detach().cpu().item())
    else:
        array = np.asarray(value)
        if np.iscomplexobj(array):
            raise TypeError(f"{name} must be real")
        try:
            ok = bool(np.isfinite(array).all())
        except TypeError as exc:
            raise TypeError(f"{name} must be numeric") from exc
    if not ok:
        raise ValueError(f"{name} contains non-finite values")


def _backend(value: Any) -> str:
    if _is_torch(value):
        return "torch"
    if _is_numpy(value):
        return "numpy"
    if isinstance(value, Number):
        return "scalar"
    return type(value).__module__.split(".", 1)[0]


def _check_common_numeric_backend(named_values: Sequence[tuple[str, Any]]) -> None:
    """Reject implicit NumPy/Torch or dtype mixing across recovery blocks."""

    arrays = [(name, value) for name, value in named_values
              if _backend(value) in {"numpy", "torch"}]
    backends = {_backend(value) for _, value in arrays}
    if len(backends) > 1:
        detail = ", ".join(f"{name}={_backend(value)}" for name, value in arrays)
        raise TypeError(f"Stage-II raw blocks mix numeric backends: {detail}")
    if not arrays:
        return
    backend = _backend(arrays[0][1])
    if backend == "torch":
        signatures = {(value.dtype, value.device) for _, value in arrays}
        if len(signatures) > 1:
            raise TypeError("Stage-II torch raw blocks must share dtype/device")
    else:
        dtypes = {np.asarray(value).dtype for _, value in arrays}
        if len(dtypes) > 1:
            raise TypeError("Stage-II NumPy raw blocks must share dtype")


def _check_projection_contract(name: str, before: Any, after: Any) -> None:
    _ensure_real_finite(f"projected {name}", after)
    if _shape(before) != _shape(after):
        raise ValueError(
            f"{name} projector changed shape {_shape(before)} -> {_shape(after)}"
        )
    bb, ba = _backend(before), _backend(after)
    if bb in {"numpy", "torch"} and ba != bb:
        raise TypeError(f"{name} projector changed backend {bb} -> {ba}")
    if bb == "torch":
        if before.dtype != after.dtype or before.device != after.device:
            raise TypeError(f"{name} projector changed torch dtype/device")
    elif bb == "numpy" and np.asarray(before).dtype != np.asarray(after).dtype:
        raise TypeError(f"{name} projector changed numpy dtype")


def _swap_last_two(value: Any) -> Any:
    if _is_torch(value):
        return value.transpose(-1, -2)
    return np.swapaxes(value, -1, -2)


def _has_square_matrix_tail(value: Any) -> bool:
    shape = _shape(value)
    return len(shape) >= 2 and shape[-2] == shape[-1]


def _resolve_pi_matrix_layout(
    raw: RawRecoveryInputs, p: Any, zeta: Any, Pi: Any, sigma_ref: Any
) -> bool:
    if raw.pi_layout not in ("auto", "matrix", "scalar"):
        raise ValueError("pi_layout must be 'auto', 'matrix', or 'scalar'")
    square = _has_square_matrix_tail(Pi)
    if raw.pi_layout == "matrix":
        if not square:
            raise ValueError("pi_layout='matrix' requires Pi shape (...,n,n)")
        return True
    if raw.pi_layout == "scalar":
        return False
    if not square:
        return False

    pshape, pishape = _shape(p), _shape(Pi)
    zshape, sshape = _shape(zeta), _shape(sigma_ref)
    # Standard adjoint layout: p(...,n), Pi(...,n,n).
    if pishape[:-1] == pshape:
        return True
    # Batched scalar P1 values can themselves have a square batch shape.
    # Treat an exact all-block shape match as scalar; callers with the rare
    # matrix/noise case d=n can remove the ambiguity with pi_layout='matrix'.
    if pishape == pshape == zshape == sshape:
        return False
    n = pishape[-1]
    if len(sshape) >= 2 and sshape[-2] == n and zshape == sshape:
        return True
    if sshape and sshape[-1] == n and zshape == sshape:
        return True
    raise ValueError(
        "ambiguous square Pi layout; set RawRecoveryInputs.pi_layout to "
        "'matrix' or 'scalar'"
    )


def _symmetrise(Pi: Any, *, matrix: bool) -> Any:
    if not matrix:
        return Pi
    return (Pi + _swap_last_two(Pi)) * 0.5


def _expand_last(value: Any) -> Any:
    if _is_torch(value):
        return value.unsqueeze(-1)
    return np.expand_dims(value, axis=-1)


def _squeeze_last(value: Any) -> Any:
    if _is_torch(value):
        return value.squeeze(-1)
    return np.squeeze(value, axis=-1)


def _pi_sigma(Pi: Any, sigma_ref: Any, *, matrix: bool) -> Any:
    """Apply Pi to sigma on the state axis, with a scalar special case."""

    pshape, sshape = _shape(Pi), _shape(sigma_ref)
    if not matrix:
        try:
            return Pi * sigma_ref
        except Exception as exc:  # pragma: no cover - backend message varies
            raise ValueError(
                f"scalar-field Pi {pshape} and sigma_ref {sshape} do not "
                "broadcast"
            ) from exc

    n = pshape[-1]
    if not sshape:
        if n != 1:
            raise ValueError(
                f"matrix Pi {pshape} requires sigma_ref with state axis {n}"
            )
        return Pi[..., 0, 0] * sigma_ref
    try:
        # (..., n, d): the manuscript shape, including d=1.
        if len(sshape) >= 2 and sshape[-2] == n:
            return Pi @ sigma_ref
        # (..., n): convenience form with the noise axis squeezed.
        if sshape[-1] == n:
            return _squeeze_last(Pi @ _expand_last(sigma_ref))
    except Exception as exc:  # pragma: no cover - backend message varies
        raise ValueError(
            f"Pi {pshape} and sigma_ref {sshape} have incompatible leading "
            "batch dimensions"
        ) from exc
    raise ValueError(
        f"Pi {pshape} cannot act on sigma_ref {sshape}; expected (...,{n},d)"
    )


def _coerce_projection_blocks(blocks: Any) -> ProjectionBlocks:
    if blocks is None:
        return ProjectionBlocks.identity()
    if isinstance(blocks, ProjectionBlocks):
        out = blocks
    elif isinstance(blocks, Mapping):
        allowed = {"p", "q", "q_anc", "Pi", "pi"}
        unknown = sorted(set(blocks).difference(allowed))
        if unknown:
            raise ValueError(
                "unknown projection blocks (zeta/state/anchor projection is "
                f"forbidden): {unknown}"
            )
        if "q" in blocks and "q_anc" in blocks:
            raise ValueError("specify only one of q or q_anc projection")
        if "Pi" in blocks and "pi" in blocks:
            raise ValueError("specify only one of Pi or pi projection")
        q = blocks.get("q_anc", blocks.get("q"))
        pi = blocks.get("Pi", blocks.get("pi"))
        if blocks.get("p") is None or q is None or pi is None:
            raise ValueError("projection mapping requires p, q_anc (or q), Pi")
        out = ProjectionBlocks(blocks["p"], q, pi)
    elif isinstance(blocks, Sequence) and not isinstance(blocks, (str, bytes)):
        if len(blocks) != 3:
            raise ValueError("projection sequence must contain (p, q_anc, Pi)")
        out = ProjectionBlocks(*blocks)
    else:
        raise TypeError(
            "projection_blocks must be ProjectionBlocks, a three-entry "
            "mapping/sequence, or None for explicit identity audit"
        )
    if not all(callable(v) for v in (out.p, out.q_anc, out.Pi)):
        raise TypeError("every projection block must be callable")
    return out


def _block_stats(before: Any, after: Any) -> BlockProjectionStats:
    b = _to_numpy(before).astype(np.float64, copy=False)
    a = _to_numpy(after).astype(np.float64, copy=False)
    delta = np.abs(a - b)
    if delta.size == 0:
        return BlockProjectionStats(0.0, 0.0, 0.0, 0.0)
    tolerance = 1e-12 + 1e-9 * np.abs(b)
    return BlockProjectionStats(
        activation_fraction=float(np.mean(delta > tolerance)),
        displacement_mean=float(np.mean(delta)),
        displacement_max=float(np.max(delta)),
        displacement_l2=float(np.sqrt(np.sum(delta * delta))),
    )


def _max_abs(value: Any) -> float:
    array = np.abs(_to_numpy(value).astype(np.float64, copy=False))
    return float(np.max(array)) if array.size else 0.0


def prepare_inputs(
    raw: RawRecoveryInputs, projection_blocks: Any
) -> ProjectedRecoveryInputs:
    """Run canonical Stage-II steps 2--4 without changing array backends.

    In particular, ``q_anc`` is built with the raw (not yet symmetrised or
    projected) Pi.  Only then are ``p``, ``q_anc`` and ``sym(Pi)`` projected
    independently.  ``zeta`` is reconstructed algebraically and never sent
    through a fourth projector.
    """

    if not isinstance(raw, RawRecoveryInputs):
        raise TypeError("raw must be RawRecoveryInputs")
    p = _normalise_container(raw.p)
    zeta = _normalise_container(raw.zeta)
    Pi = _normalise_container(raw.Pi)
    sigma_ref = _normalise_container(raw.sigma_ref)
    named_values = (
        ("p", p),
        ("zeta", zeta),
        ("Pi", Pi),
        ("sigma_ref", sigma_ref),
    )
    for name, value in named_values:
        _ensure_real_finite(name, value)
    _check_common_numeric_backend(named_values)

    matrix_pi = _resolve_pi_matrix_layout(raw, p, zeta, Pi, sigma_ref)
    raw_pi_sigma = _pi_sigma(Pi, sigma_ref, matrix=matrix_pi)
    try:
        q_anc = zeta + raw_pi_sigma
    except Exception as exc:  # pragma: no cover - backend message varies
        raise ValueError(
            f"zeta {_shape(zeta)} and Pi@sigma_ref "
            f"{_shape(raw_pi_sigma)} are incompatible"
        ) from exc
    _ensure_real_finite("raw q_anc", q_anc)
    if _shape(q_anc) != _shape(zeta):
        raise ValueError(
            "q_anc broadcasting changed the zeta shape; use explicit leading "
            "batch axes"
        )

    sym_pi = _symmetrise(Pi, matrix=matrix_pi)
    blocks = _coerce_projection_blocks(projection_blocks)
    p_n = blocks.p(p)
    q_n = blocks.q_anc(q_anc)
    pi_n = blocks.Pi(sym_pi)
    for name, before, after in (
        ("p", p, p_n),
        ("q_anc", q_anc, q_n),
        ("Pi", sym_pi, pi_n),
    ):
        _check_projection_contract(name, before, after)

    if matrix_pi:
        asymmetry = _max_abs(pi_n - _swap_last_two(pi_n))
        scale = max(1.0, _max_abs(pi_n))
        if asymmetry > 1e-10 * scale:
            raise ValueError("Pi projector returned a non-symmetric matrix")

    zeta_n = q_n - _pi_sigma(pi_n, sigma_ref, matrix=matrix_pi)
    _ensure_real_finite("projected zeta", zeta_n)
    if _shape(zeta_n) != _shape(q_n):
        raise ValueError("projected zeta reconstruction changed q_anc shape")

    identity_error = (
        zeta_n + _pi_sigma(pi_n, sigma_ref, matrix=matrix_pi) - q_n
    )
    diagnostics = ProjectionDiagnostics(
        p=_block_stats(p, p_n),
        q_anc=_block_stats(q_anc, q_n),
        Pi=_block_stats(sym_pi, pi_n),
        pi_symmetrization_max=_max_abs(Pi - sym_pi),
        coordinate_identity_max=_max_abs(identity_error),
    )
    return ProjectedRecoveryInputs(
        p=p_n,
        q_anc=q_n,
        Pi=pi_n,
        zeta=zeta_n,
        sigma_ref=sigma_ref,
        anchor=raw.anchor,
        diagnostics=diagnostics,
        pi_layout="matrix" if matrix_pi else "scalar",
    )


def _like(value: Any, template: Any) -> Any:
    if _is_torch(template):
        return template.new_tensor(value)
    if _is_numpy(template):
        return np.asarray(value, dtype=np.asarray(template).dtype)
    return value


def _where(condition: Any, yes: Any, no: Any) -> Any:
    if _is_torch(condition):
        # Tensor.where is bound to the value tensor: yes.where(condition, no)
        # is torch.where(condition, yes, no) without importing torch here.
        return yes.where(condition, no)
    return np.where(condition, yes, no)


def _positive(value: Any) -> Any:
    zero = value * 0
    return _where(value > zero, value, zero)


def _event_l2(value: Any, event_ndim: int) -> Any:
    if event_ndim == 0:
        return abs(value)
    if event_ndim != 1:
        raise ValueError("only scalar or vector recovery actions are supported")
    if _is_torch(value):
        return (value * value).sum(dim=-1).sqrt()
    return np.sqrt(np.sum(np.asarray(value) ** 2, axis=-1))


def _event_max(value: Any, event_ndim: int) -> Any:
    if event_ndim == 0:
        return value
    if _is_torch(value):
        return value.amax(dim=-1)
    return np.max(value, axis=-1)


@dataclass(frozen=True)
class BoxRecoverySet:
    """Closed box with an exact outward-normal-cone distance.

    Scalar bounds describe a scalar action (possibly with leading batch
    axes).  Vector bounds describe the final action axis.
    """

    lower: Any
    upper: Any
    active_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        if lo.shape != hi.shape:
            try:
                lo, hi = np.broadcast_arrays(lo, hi)
            except ValueError as exc:
                raise ValueError("box bounds do not broadcast") from exc
        if np.any(lo > hi):
            raise ValueError("box lower bound exceeds upper bound")
        if lo.ndim > 1:
            raise ValueError("box bounds must be scalar or one-dimensional")
        if self.active_tolerance < 0 or not np.isfinite(self.active_tolerance):
            raise ValueError("active_tolerance must be finite and non-negative")

    @property
    def event_ndim(self) -> int:
        return np.broadcast(np.asarray(self.lower), np.asarray(self.upper)).nd

    def violation(self, action: Any) -> Any:
        lo, hi = _like(self.lower, action), _like(self.upper, action)
        component = _positive(lo - action) + _positive(action - hi)
        return _event_max(component, self.event_ndim)

    def normal_cone_residual(
        self, action: Any, stationarity_gradient: Any
    ) -> Any:
        lo, hi = _like(self.lower, action), _like(self.upper, action)
        tol = _like(self.active_tolerance, action)
        zero = stationarity_gradient * 0
        fixed = abs(hi - lo) <= tol
        at_lower = action <= lo + tol
        at_upper = action >= hi - tol
        # For g + N_C: at lower, g>=0 is stationary; at upper, g<=0.
        component = abs(stationarity_gradient)
        component = _where(
            at_lower, _positive(-stationarity_gradient), component
        )
        component = _where(at_upper, _positive(stationarity_gradient), component)
        component = _where(fixed, zero, component)
        return _event_l2(component, self.event_ndim)


@dataclass(frozen=True)
class UnconstrainedRecoverySet:
    """Euclidean action space; ``event_ndim=1`` for vector actions."""

    event_ndim: int = 0

    def violation(self, action: Any) -> Any:
        zero = action * 0
        return _event_l2(zero, self.event_ndim)

    def normal_cone_residual(
        self, action: Any, stationarity_gradient: Any
    ) -> Any:
        return _event_l2(stationarity_gradient, self.event_ndim)


def _nonnegative_summary(name: str, value: Any) -> tuple[float, float, float]:
    _ensure_real_finite(name, value)
    array = _to_numpy(value).astype(np.float64, copy=False)
    if np.any(array < -1e-12):
        raise ValueError(f"{name} must be non-negative")
    array = np.maximum(array, 0.0)
    if array.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(array)),
        float(np.max(array)),
        float(np.sqrt(np.mean(array * array))),
    )


def _validate_sense(sense: str) -> ObjectiveSense:
    if sense not in ("minimize", "maximize"):
        raise ValueError("sense must be explicitly 'minimize' or 'maximize'")
    return sense  # type: ignore[return-value]


def execute_stage2(
    raw: RawRecoveryInputs,
    projection_blocks: Any,
    *,
    local_recovery: Callable[[ProjectedRecoveryInputs, Any], Any],
    objective_gradient: Callable[[ProjectedRecoveryInputs, Any, Any], Any],
    recovery_set: RecoverySet,
    sense: ObjectiveSense,
    context: Any = None,
    recovery_health: Callable[[ProjectedRecoveryInputs, Any], Mapping[str, Any] | None]
    | None = None,
    feasibility_tolerance: float = 1e-8,
) -> Stage2Result:
    """Prepare inputs, solve locally, and certify the returned action.

    The feasibility gate is fail-hard and precedes the normal-cone residual:
    the standard normal cone is empty outside the selected set.  A safety
    clip is never applied here because it would hide a failed local solver.
    ``objective_gradient`` must be evaluated at the returned action using the
    same projected tuple passed to ``local_recovery``.
    """

    sense = _validate_sense(sense)
    if feasibility_tolerance < 0 or not np.isfinite(feasibility_tolerance):
        raise ValueError("feasibility_tolerance must be finite and non-negative")
    if not callable(local_recovery) or not callable(objective_gradient):
        raise TypeError("local_recovery and objective_gradient must be callable")
    if not callable(getattr(recovery_set, "violation", None)) or not callable(
        getattr(recovery_set, "normal_cone_residual", None)
    ):
        raise TypeError("recovery_set does not implement the RecoverySet protocol")

    projected = prepare_inputs(raw, projection_blocks)
    health: Mapping[str, Any] = {}
    if recovery_health is not None:
        health_result = recovery_health(projected, context)
        health = {} if health_result is None else dict(health_result)
        if health.get("ok") is False:
            raise ValueError(
                f"projected recovery health check failed: {health.get('reason', '')}"
            )

    solved = local_recovery(projected, context)
    if isinstance(solved, LocalRecoveryOutput):
        action = solved.action
        solver_diagnostics = dict(solved.diagnostics)
        exact = bool(solved.exact)
    else:
        action = solved
        solver_diagnostics = {}
        exact = False
    _ensure_real_finite("recovered action", action)

    violation = recovery_set.violation(action)
    _, feasibility_max, _ = _nonnegative_summary(
        "feasibility violation", violation
    )
    if feasibility_max > feasibility_tolerance:
        raise ValueError(
            "local recovery returned an infeasible action: "
            f"max violation {feasibility_max:.6g} > "
            f"{feasibility_tolerance:.6g}"
        )

    gradient = objective_gradient(projected, action, context)
    _ensure_real_finite("projected objective gradient", gradient)
    if _shape(gradient) != _shape(action):
        raise ValueError(
            "objective_gradient must have the recovered action shape "
            f"{_shape(action)}, got {_shape(gradient)}"
        )
    stationarity_gradient = gradient if sense == "minimize" else -gradient
    pointwise = recovery_set.normal_cone_residual(
        action, stationarity_gradient
    )
    mean, maximum, rms = _nonnegative_summary(
        "normal-cone residual", pointwise
    )
    residual = NormalConeResidualData(
        sense=sense,
        objective_gradient=gradient,
        stationarity_gradient=stationarity_gradient,
        pointwise=pointwise,
        mean=mean,
        maximum=maximum,
        rms=rms,
    )
    return Stage2Result(
        inputs=projected,
        action=action,
        feasibility_violation=violation,
        feasibility_max=feasibility_max,
        residual=residual,
        solver_diagnostics=solver_diagnostics,
        recovery_health=health,
        exact_solve=exact,
    )


def run_stage2(problem: Any, config: Any, seed: int) -> Stage2Result:
    """Backward-compatible problem entry point using one explicit hook bundle.

    A problem adapter must expose ``stage2_hooks(config, seed)`` returning a
    mapping accepted by :func:`execute_stage2`.  Keeping the bundle explicit
    avoids guessing estimator or state-layout APIs in the generic core.
    """

    hook_factory = getattr(problem, "stage2_hooks", None)
    if not callable(hook_factory):
        raise TypeError(
            "problem must expose stage2_hooks(config, seed); estimator/state "
            "layout belongs to the problem adapter"
        )
    hooks = hook_factory(config, seed)
    if not isinstance(hooks, Mapping):
        raise TypeError("stage2_hooks must return a mapping")
    required = {
        "raw",
        "projection_blocks",
        "local_recovery",
        "objective_gradient",
        "recovery_set",
        "sense",
    }
    missing = sorted(required.difference(hooks))
    if missing:
        raise ValueError(f"stage2_hooks missing required entries: {missing}")
    return execute_stage2(**dict(hooks))
