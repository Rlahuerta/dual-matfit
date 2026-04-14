import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from unittest.mock import MagicMock
from dualmatfit.fitting.identifiability import (
    analyze_cost_integrator,
    as_2d_jacobian,
    beta_variance_proxy,
)
from dualmatfit.optimization.cost import CostIntegrator, LSQFit


def _xlog_fun(x, param):
    return param[0] * jnp.exp(-param[1] * x) + param[2] + x * param[3]


def test_as_2d_jacobian_accepts_2d_and_single_function_3d_inputs():
    jacobian_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    jacobian_3d = np.array([[[1.0, 2.0], [3.0, 4.0]]])

    np.testing.assert_array_equal(as_2d_jacobian(jacobian_2d), jacobian_2d)
    np.testing.assert_array_equal(as_2d_jacobian(jacobian_3d), jacobian_2d)


def test_analyze_cost_integrator_detects_beta_k1_near_collinearity():
    """Verify that near-collinear alpha/k_1 columns produce high cosine similarity."""
    jacobian = np.array(
        [
            [1.0, 1.0, 0.0],
            [2.0, 2.000001, 0.0],
            [3.0, 3.000001, 1.0],
        ]
    )
    fval = np.array([0.1, 0.2, 0.05])

    cost_fun = MagicMock()
    cost_fun.ncontrol = 3
    cost_fun.nvars = 3
    cost_fun.inp_mat_keys = ["alpha", "k_1", "mu"]

    integrator = MagicMock()
    integrator._cost_function = MagicMock(return_value=float(fval.sum()))
    integrator._residuum_diff = MagicMock(return_value=jacobian[np.newaxis, :, :])
    integrator.inp_mat_keys = ["alpha", "k_1", "mu"]
    integrator.cost_functions = [cost_fun]
    integrator._alpha = 0.001

    report = analyze_cost_integrator(
        integrator,
        np.array([1.0, 2.0, 3.0]),
        fiber_angle_key="alpha",
        stiffness_key="k_1",
    )

    assert report.beta_k1_cosine_similarity > 0.999
    assert report.beta_k1_condition_number > 1.0e5
    assert report.smallest_singular_value >= 0.0


def test_beta_variance_proxy_uses_pinv_for_rank_deficient_normal_matrix():
    singular_normal = np.array([[1.0, 1.0], [1.0, 1.0]])

    variance = beta_variance_proxy(singular_normal, parameter_index=0)

    assert np.isfinite(variance)
    assert variance >= 0.0


def test_analyze_cost_integrator_smoke_with_mock_cost_function():
    cost_fun = MagicMock()
    cost_fun.ncontrol = 3
    cost_fun.nvars = 3
    cost_fun.xi = np.array([1.0, 2.0, 3.0])
    cost_fun.xi_ref = np.array([1.0, 2.0, 3.0])
    cost_fun.xi_bounds = [[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]]
    cost_fun.inp_mat_keys = ["alpha", "k_1", "mu"]
    cost_fun.residuum.return_value = np.array([0.2, -0.1, 0.05])
    cost_fun.residuum_diff.return_value = np.array(
        [
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 1.0],
        ]
    )
    cost_fun.volume.return_value = np.zeros(3)
    cost_fun.volume_diff.return_value = np.zeros((3, 3))

    integrator = CostIntegrator(
        [cost_fun], ftype="cauchy_robust", c=40.0, alpha=0.001, beta=1.0
    )
    report = analyze_cost_integrator(integrator, np.array([1.0, 2.0, 3.0]))

    assert list(report.param_names) == ["alpha", "k_1", "mu"]
    assert report.condition_number_jtj >= 1.0  # inf for perfectly collinear columns
    assert report.omega_tik == 0.001


@pytest.mark.integration
def test_analyze_cost_integrator_with_real_cost_function():
    xi = np.array([2.5, 1.3, 0.5, 0.01], dtype=float)
    mat_params = pd.DataFrame(
        {
            "values": xi,
            "variable": np.ones(xi.shape[0], dtype=bool),
            "lower": 0.01 * np.ones(xi.shape[0], dtype=float),
            "upper": 5.0 * np.ones(xi.shape[0], dtype=float),
        },
        index=["x0", "x1", "x2", "x3"],
    )
    lsq_fun = LSQFit(10, 0.2, mat_params, _xlog_fun, seed=42)
    integrator = CostIntegrator(
        [lsq_fun], ftype="cauchy_robust", c=40.0, alpha=0.001, beta=1.0
    )

    report = analyze_cost_integrator(
        integrator,
        xi,
        fiber_angle_key=lsq_fun.inp_mat_keys[0],
        stiffness_key=lsq_fun.inp_mat_keys[1],
    )

    assert list(report.param_names) == list(lsq_fun.inp_mat_keys)
    assert np.isfinite(report.condition_number_jtj)
    assert report.omega_tik == 0.001
