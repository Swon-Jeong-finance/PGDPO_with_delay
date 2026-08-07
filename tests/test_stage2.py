import numpy as np
import pytest

from pgdpo_delay.core.stage2 import (
    BoxRecoverySet,
    LocalRecoveryOutput,
    ProjectionBlocks,
    RawRecoveryInputs,
    UnconstrainedRecoverySet,
    execute_stage2,
    prepare_inputs,
    run_stage2,
)


def test_prepare_inputs_enforces_canonical_order_and_independent_blocks():
    anchor = object()
    raw = RawRecoveryInputs(
        p=np.array([1.0, 2.0]),
        zeta=np.array([[0.5], [-0.5]]),
        Pi=np.array([[2.0, 4.0], [0.0, 6.0]]),
        sigma_ref=np.array([[1.0], [2.0]]),
        anchor=anchor,
    )
    seen = {}

    def project_p(value):
        seen["p"] = value.copy()
        return value + 1.0

    def project_q(value):
        seen["q"] = value.copy()
        return value * 2.0

    def project_pi(value):
        seen["Pi"] = value.copy()
        return value * 0.5

    out = prepare_inputs(
        raw, ProjectionBlocks(p=project_p, q_anc=project_q, Pi=project_pi)
    )

    # q is reconstructed with RAW Pi before Pi is symmetrised/projected.
    np.testing.assert_allclose(seen["q"], [[10.5], [11.5]])
    np.testing.assert_allclose(seen["Pi"], [[2.0, 2.0], [2.0, 6.0]])
    np.testing.assert_allclose(out.p, [2.0, 3.0])
    np.testing.assert_allclose(out.q_anc, [[21.0], [23.0]])
    np.testing.assert_allclose(out.Pi, [[1.0, 1.0], [1.0, 3.0]])
    np.testing.assert_allclose(out.zeta, [[18.0], [16.0]])
    assert out.sigma_ref is raw.sigma_ref
    assert out.anchor is anchor
    assert out.diagnostics.pi_symmetrization_max == pytest.approx(2.0)
    assert out.diagnostics.coordinate_identity_max < 1e-14
    assert out.diagnostics.activated


def test_prepare_inputs_scalar_special_case_and_identity_audit():
    raw = RawRecoveryInputs(p=2.0, zeta=-1.0, Pi=3.0, sigma_ref=4.0)
    out = prepare_inputs(raw, None)
    assert out.q_anc == pytest.approx(11.0)
    assert out.zeta == pytest.approx(-1.0)
    assert not out.diagnostics.activated


def test_projection_module_is_only_the_common_core_import_surface():
    from pgdpo_delay.core import projection

    assert projection.ProjectionBlocks is ProjectionBlocks


def test_prepare_inputs_rejects_joint_or_malformed_projection():
    raw = RawRecoveryInputs(
        p=np.ones(2), zeta=np.ones((2, 1)), Pi=np.eye(2),
        sigma_ref=np.ones((2, 1)),
    )
    with pytest.raises(TypeError, match="projection_blocks"):
        prepare_inputs(raw, lambda triple: triple)
    with pytest.raises(ValueError, match="forbidden"):
        prepare_inputs(
            raw,
            {"p": lambda x: x, "q": lambda x: x, "Pi": lambda x: x,
             "zeta": lambda x: x},
        )
    with pytest.raises(ValueError, match="changed shape"):
        prepare_inputs(
            raw,
            ProjectionBlocks(
                p=lambda x: x[:1], q_anc=lambda x: x, Pi=lambda x: x
            ),
        )
    with pytest.raises(ValueError, match="non-symmetric"):
        prepare_inputs(
            raw,
            ProjectionBlocks(
                p=lambda x: x,
                q_anc=lambda x: x,
                Pi=lambda x: x + np.array([[0.0, 1.0], [0.0, 0.0]]),
            ),
        )


def test_square_scalar_batch_is_not_misclassified_as_matrix_pi():
    shape = (2, 2)
    raw = RawRecoveryInputs(
        p=np.ones(shape),
        zeta=np.full(shape, 2.0),
        Pi=np.full(shape, 3.0),
        sigma_ref=np.full(shape, 4.0),
    )
    out = prepare_inputs(raw, None)
    np.testing.assert_allclose(out.q_anc, np.full(shape, 14.0))
    np.testing.assert_allclose(out.zeta, np.full(shape, 2.0))
    assert out.pi_layout == "scalar"


def test_prepare_inputs_rejects_implicit_backend_or_dtype_mixing():
    with pytest.raises(TypeError, match="share dtype"):
        prepare_inputs(
            RawRecoveryInputs(
                p=np.ones(2, dtype=np.float32),
                zeta=np.ones(2, dtype=np.float64),
                Pi=np.ones(2, dtype=np.float32),
                sigma_ref=np.ones(2, dtype=np.float32),
                pi_layout="scalar",
            ),
            None,
        )

    torch = pytest.importorskip("torch")
    with pytest.raises(TypeError, match="mix numeric backends"):
        prepare_inputs(
            RawRecoveryInputs(
                p=torch.ones(2),
                zeta=np.ones(2, dtype=np.float32),
                Pi=torch.ones(2),
                sigma_ref=torch.ones(2),
                pi_layout="scalar",
            ),
            None,
        )


@pytest.mark.parametrize(
    "sense,gradient,expected",
    [
        ("minimize", np.array([1.0, 0.3, -1.0]), [0.0, 0.3, 0.0]),
        ("maximize", np.array([-1.0, 0.3, 1.0]), [0.0, 0.3, 0.0]),
    ],
)
def test_execute_stage2_box_normal_cone_sign_convention(
    sense, gradient, expected
):
    raw = RawRecoveryInputs(
        p=0.0, zeta=0.0, Pi=0.0, sigma_ref=0.0
    )
    actions = np.array([0.0, 0.5, 1.0])
    result = execute_stage2(
        raw,
        ProjectionBlocks.identity(),
        local_recovery=lambda projected, context: LocalRecoveryOutput(
            actions, {"iterations": 4}, exact=False
        ),
        objective_gradient=lambda projected, action, context: gradient,
        recovery_set=BoxRecoverySet(0.0, 1.0, active_tolerance=1e-12),
        sense=sense,
    )
    np.testing.assert_allclose(result.residual.pointwise, expected)
    assert result.residual.maximum == pytest.approx(0.3)
    assert result.residual.rms == pytest.approx(np.sqrt(0.03))
    assert result.feasibility_max == 0.0
    assert result.solver_diagnostics == {"iterations": 4}


def test_box_normal_cone_reports_wrong_sign_at_each_boundary():
    recovery_set = BoxRecoverySet(0.0, 1.0, active_tolerance=0.0)
    action = np.array([0.0, 0.5, 1.0])
    # Cost gradient: lower -1 and upper +1 both point into a descent direction.
    out = recovery_set.normal_cone_residual(action, np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_allclose(out, [1.0, 0.0, 1.0])


def test_vector_box_uses_action_axis_for_normal_cone_distance():
    recovery_set = BoxRecoverySet(
        lower=np.array([0.0, -1.0]),
        upper=np.array([1.0, 1.0]),
        active_tolerance=0.0,
    )
    action = np.array([[0.0, 0.0], [1.0, -1.0]])
    gradient = np.array([[-2.0, 3.0], [4.0, -5.0]])
    residual = recovery_set.normal_cone_residual(action, gradient)
    np.testing.assert_allclose(residual, [np.sqrt(13.0), np.sqrt(41.0)])


def test_execute_stage2_fails_infeasible_before_gradient_or_residual():
    called = {"gradient": False}

    def gradient(projected, action, context):
        called["gradient"] = True
        return action

    with pytest.raises(ValueError, match="infeasible action"):
        execute_stage2(
            RawRecoveryInputs(0.0, 0.0, 0.0, 0.0),
            None,
            local_recovery=lambda projected, context: 1.1,
            objective_gradient=gradient,
            recovery_set=BoxRecoverySet(0.0, 1.0),
            sense="minimize",
        )
    assert not called["gradient"]


def test_execute_stage2_health_guard_precedes_local_solve():
    called = {"solve": False}

    def solve(projected, context):
        called["solve"] = True
        return 0.0

    with pytest.raises(ValueError, match="health check failed"):
        execute_stage2(
            RawRecoveryInputs(0.0, 0.0, 0.0, 0.0),
            None,
            local_recovery=solve,
            objective_gradient=lambda projected, action, context: 0.0,
            recovery_set=UnconstrainedRecoverySet(),
            sense="minimize",
            recovery_health=lambda projected, context: {
                "ok": False,
                "reason": "nonpositive curvature",
            },
        )
    assert not called["solve"]


def test_run_stage2_is_thin_explicit_hook_orchestrator():
    class Adapter:
        def stage2_hooks(self, config, seed):
            assert config == {"name": "fixture"}
            assert seed == 7
            return dict(
                raw=RawRecoveryInputs(1.0, 2.0, 3.0, 4.0),
                projection_blocks=None,
                local_recovery=lambda projected, context: 0.25,
                objective_gradient=lambda projected, action, context: 0.0,
                recovery_set=BoxRecoverySet(0.0, 1.0),
                sense="minimize",
            )

    result = run_stage2(Adapter(), {"name": "fixture"}, seed=7)
    assert result.action == pytest.approx(0.25)
    assert result.inputs.q_anc == pytest.approx(14.0)
    with pytest.raises(TypeError, match="stage2_hooks"):
        run_stage2(object(), {}, seed=1)


def test_prepare_inputs_preserves_torch_backend_dtype_and_device():
    torch = pytest.importorskip("torch")
    p = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    zeta = torch.tensor([[0.5], [-0.5]], dtype=torch.float64)
    Pi = torch.tensor([[2.0, 4.0], [0.0, 6.0]], dtype=torch.float64)
    sigma = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    out = prepare_inputs(
        RawRecoveryInputs(p, zeta, Pi, sigma),
        ProjectionBlocks(
            p=lambda x: x + 1.0,
            q_anc=lambda x: x,
            Pi=lambda x: x,
        ),
    )
    assert isinstance(out.p, torch.Tensor)
    assert out.p.dtype == torch.float64
    assert out.p.device == p.device
    assert out.p.requires_grad
    assert isinstance(out.zeta, torch.Tensor)
    torch.testing.assert_close(
        out.zeta, torch.tensor([[4.5], [-2.5]], dtype=torch.float64)
    )


def test_torch_box_normal_cone_uses_backend_where_correctly():
    torch = pytest.importorskip("torch")
    recovery_set = BoxRecoverySet(0.0, 1.0, active_tolerance=0.0)
    action = torch.tensor([0.0, 0.5, 1.0])
    gradient = torch.tensor([-1.0, 0.25, 1.0])
    residual = recovery_set.normal_cone_residual(action, gradient)
    torch.testing.assert_close(residual, torch.tensor([1.0, 0.25, 1.0]))
