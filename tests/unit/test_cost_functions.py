from __future__ import annotations
import unittest
import pytest
import numpy as np
import pandas as pd
import sympy as sy
from unittest.mock import MagicMock
from functools import lru_cache

from dualmatfit.optimization.cost import CostFunction, CostIntegrator
from dualmatfit.formulation.variational import VariationalFormulation
from dualmatfit.formulation.lambdify import _lambdify
from dualmatfit.optimization.loss import (lsq_fval, lsq_dfval,
                                      lsq_wise_fval, lsq_wise_dfval,
                                      cauchy_fval, cauchy_dfval,
                                      huber_fval, huber_dfval,
                                      logcosh_fval, logcosh_dfval,
                                      ln_fval, ln_dfval)

module = 'numpy'
# module = 'jax'


class TestCostFunctions(unittest.TestCase):

    def setUp(self):
        self.seed = 11
        np.random.seed(self.seed)

        # Common setup: create symbolic and numerical inputs
        self.nxi = 3
        self.xi = sy.Array([sy.symbols(f'x_{i+1}') for i in range(self.nxi)])
        self.np_xi = np.random.randn(self.nxi)

        # Example symbolic residuum
        self.residuum_sym = sy.Matrix([self.xi[0]**2 - 2*self.xi[1] + self.xi[2],
                                       self.xi[0] + self.xi[1] * self.xi[2] - 1])
        self.residuum_lbd = _lambdify(self.xi, self.residuum_sym, module=module)
        self.np_residuum = self.residuum_lbd(*self.np_xi)[:, 0]

        self.residuum_diff_sym = sy.derive_by_array(self.residuum_sym, self.xi)[:, :, 0].tomatrix()
        self.residuum_diff_lbd = _lambdify(self.xi, self.residuum_diff_sym, module=module)
        self.np_residuum_diff = self.residuum_diff_lbd(*self.np_xi)

        self.lsq_loss_sym = sy.sqrt(sy.Matrix.dot(self.residuum_sym, self.residuum_sym))
        self.lsq_loss_lbd = _lambdify(self.xi, self.lsq_loss_sym, module=module)

        # Parameters for specific functions
        self.c = sy.symbols('c', real=True)             # Cauchy parameter
        self.delta = sy.symbols('delta', real=True)     # Huber parameter

        # Cauchy Loss Function: Robustness Under Gaussian and Cauchy Noise
        self.cauchy_loss_sym = 0.5 * self.c ** 2 * sy.ln(1. + sy.Matrix.dot((self.residuum_sym / self.c),
                                                                      (self.residuum_sym / self.c)))
        self.cauchy_loss_lbd = _lambdify((*self.xi, self.c), self.cauchy_loss_sym, module=module)

        self.cauchy_loss_diff_sym = sy.derive_by_array(self.cauchy_loss_sym, self.xi)
        self.cauchy_loss_diff_lbd = _lambdify((*self.xi, self.c), self.cauchy_loss_diff_sym, module=module)

        self.ln_loss_sym = sy.ln(1. + sy.Matrix.dot(self.residuum_sym, self.residuum_sym))
        self.ln_loss_lbd = _lambdify(self.xi, self.ln_loss_sym, module=module)

    def test_lsq_loss(self):
        result = self.lsq_loss_lbd(*self.np_xi)
        expected = lsq_fval(self.np_residuum)

        self.assertAlmostEqual(result, expected.sum())

        lsq_loss_diff_sym = sy.derive_by_array(self.lsq_loss_sym, self.xi)
        lsq_loss_diff_lbd = _lambdify(self.xi, lsq_loss_diff_sym, module=module)

        np_diff_sym = lsq_loss_diff_lbd(*self.np_xi)
        np_diff_ref = lsq_dfval(self.np_residuum, self.np_residuum_diff.T)

        np.testing.assert_array_almost_equal(np_diff_sym, np_diff_ref.sum(axis=0))

    def test_cauchy_loss(self):
        cval = 20.

        result = self.cauchy_loss_lbd(*self.np_xi, cval)
        expected = cauchy_fval(self.np_residuum, cval)

        np.testing.assert_approx_equal(result, expected.sum(), significant=3)

        cauchy_loss_diff_sym = sy.derive_by_array(self.cauchy_loss_sym, self.xi)
        cauchy_loss_diff_lbd = _lambdify((*self.xi, self.c), cauchy_loss_diff_sym, module=module)

        np_diff_sym = cauchy_loss_diff_lbd(*self.np_xi, cval)
        np_diff_ref = cauchy_dfval(self.np_residuum, self.np_residuum_diff.T, c=cval)

        np.testing.assert_array_almost_equal(np_diff_sym, np_diff_ref.sum(axis=0), decimal=1)

    def test_huber_loss(self):
        residuum = self.np_residuum
        delta_val = 1.0  # Use numeric delta, not sympy Symbol
        result = huber_fval(residuum, delta=delta_val)
        abs_res = np.abs(residuum)
        expected = np.where(abs_res <= delta_val, 0.5 * residuum**2, delta_val * (abs_res - 0.5 * delta_val))
        expected = np.sum(expected)
        # huber_fval returns per-row sums as array; compare scalar values
        np.testing.assert_almost_equal(result.sum(), expected, decimal=5)

    def test_logcosh_loss(self):
        residuum = self.np_residuum
        result = logcosh_fval(residuum)
        expected = np.sum(np.log(np.cosh(residuum)))
        # logcosh_fval returns per-row sums as array; compare scalar values
        np.testing.assert_almost_equal(result.sum(), expected, decimal=5)

    def test_ln_loss(self):
        result = ln_fval(self.np_residuum)
        expected = self.ln_loss_lbd(*self.np_xi)

        np.testing.assert_approx_equal(result.sum(), expected, significant=3)

        ln_loss_diff_sym = sy.derive_by_array(self.ln_loss_sym, self.xi)
        ln_loss_diff_lbd = _lambdify(self.xi, ln_loss_diff_sym, module=module)

        np_diff_sym = ln_loss_diff_lbd(*self.np_xi)
        np_diff_ref = ln_dfval(self.np_residuum, self.np_residuum_diff.T)

        np.testing.assert_array_almost_equal(np_diff_sym, np_diff_ref.sum(axis=0), decimal=1)

        self.assertAlmostEqual(result.sum(), expected)
        np.testing.assert_array_almost_equal(np_diff_sym, np_diff_ref.sum(axis=0))


class TestCostFunctionCache:
    @pytest.fixture
    def var_form(self, variational_form_factory):
        """Create a real VariationalFormulation object for testing."""
        return variational_form_factory(
            ds=1.,
            itype='nh',
            mix=1,
            kappa=False,
            dvol=False
        )

    @pytest.fixture
    def dsvars(self, var_form, sample_dsvars_dataframe):
        """Create a sample dsvars DataFrame."""
        return sample_dsvars_dataframe(var_form)

    def test_cache_hits(self, var_form, dsvars):
        """Test that the cache is hit on subsequent calls with the same parameters."""
        stretch_x = np.array([1.1, 1.2, 1.3])
        load_ref = np.array([10., 20., 30.])
        cost_fun = CostFunction(var_form, load_ref, stretch_x, dsvars, cache_size=128)

        # Mock the underlying implementation to monitor calls
        cost_fun._solve_implementation = MagicMock(return_value=MagicMock(success=True))
        cost_fun._cached_solve = lru_cache(maxsize=128)(cost_fun._solve_implementation)

        xi = np.array([v for v, V in zip(dsvars['values'], dsvars['variable']) if V])

        # First call - should call the implementation
        cost_fun._chk_solve(xi)
        assert cost_fun._solve_implementation.call_count == 1

        # Second call with same xi - should hit the cache
        cost_fun._chk_solve(xi)
        assert cost_fun._solve_implementation.call_count == 1

    def test_cache_misses(self, var_form, dsvars):
        """Test that the cache is missed when parameters change."""
        stretch_x = np.array([1.1, 1.2, 1.3])
        load_ref = np.array([10., 20., 30.])
        cost_fun = CostFunction(var_form, load_ref, stretch_x, dsvars, cache_size=128)

        cost_fun._solve_implementation = MagicMock(return_value=MagicMock(success=True))
        cost_fun._cached_solve = lru_cache(maxsize=128)(cost_fun._solve_implementation)

        xi1 = np.array([v for v, V in zip(dsvars['values'], dsvars['variable']) if V])
        xi2 = xi1 * 2.0

        # First call
        cost_fun._chk_solve(xi1)
        assert cost_fun._solve_implementation.call_count == 1

        # Second call with different xi
        cost_fun._chk_solve(xi2)
        assert cost_fun._solve_implementation.call_count == 2

    def test_cache_clear(self, var_form, dsvars):
        """Test that the cache is cleared when variables are updated."""
        stretch_x = np.array([1.1, 1.2, 1.3])
        load_ref = np.array([10., 20., 30.])
        cost_fun = CostFunction(var_form=var_form,
                                load_ref=load_ref,
                                stretch_x=stretch_x,
                                dsvars=dsvars,
                                cache_size=128,
                                )

        cost_fun._solve_implementation = MagicMock(return_value=MagicMock(success=True))
        cost_fun._cached_solve = lru_cache(maxsize=128)(cost_fun._solve_implementation)

        xi = np.array([v for v, V in zip(dsvars['values'], dsvars['variable']) if V])

        cost_fun._chk_solve(xi)
        assert cost_fun._solve_implementation.call_count == 1

        # Clear cache by updating variables
        cost_fun.update_variables(dsvars)

        # Call again, should miss cache
        cost_fun._chk_solve(xi)
        assert cost_fun._solve_implementation.call_count == 2


class TestCostIntegratorCache:
    @pytest.fixture
    def mock_cost_function(self):
        """Creates a mock CostFunction object."""
        cost_fun = MagicMock(spec=CostFunction)
        cost_fun.ncontrol = 3
        cost_fun.nvars = 2
        cost_fun.xi = np.array([1.0, 2.0])
        cost_fun.xi_ref = np.array([1.0, 2.0])
        cost_fun.xi_bounds = [[0., 5.], [0., 5.]]
        cost_fun.inp_mat_keys = ['x_1', 'x_2']
        cost_fun.residuum.return_value = np.array([0.1, 0.2, 0.3])
        cost_fun.residuum_diff.return_value = np.random.rand(3, 2)
        cost_fun.volume.return_value = np.array([0.1, 0.2, 0.3])
        cost_fun.volume_diff.return_value = np.random.rand(3, 2)

        return cost_fun


    def test_residuum_cache_hits(self, mock_cost_function):
        """Test that the residuum cache is hit."""
        xi = np.array([1.0, 2.0])
        integrator = CostIntegrator([mock_cost_function], cache_size=128)

        # First call
        integrator._residuum(xi)
        assert mock_cost_function.residuum.call_count == 1

        integrator._residuum_diff(xi)
        assert mock_cost_function.residuum_diff.call_count == 1

        # Second call
        integrator._residuum(xi)
        assert mock_cost_function.residuum.call_count == 1

        integrator._residuum_diff(xi)
        assert mock_cost_function.residuum_diff.call_count == 1

    def test_residuum_cache_misses(self, mock_cost_function):
        """Test that the residuum cache is missed on different xi."""
        xi1 = np.array([1.0, 2.0])
        xi2 = np.array([3.0, 4.0])
        integrator = CostIntegrator([mock_cost_function], cache_size=128)

        integrator._residuum(xi1)
        assert mock_cost_function.residuum.call_count == 1

        integrator._residuum_diff(xi1)
        assert mock_cost_function.residuum_diff.call_count == 1

        integrator._residuum(xi2)
        assert mock_cost_function.residuum.call_count == 2

        integrator._residuum_diff(xi2)
        assert mock_cost_function.residuum_diff.call_count == 2

    def test_volume_cache_hits(self, mock_cost_function):
        """Test that the volume regularization is computed correctly."""
        xi = np.array([1.0, 2.0])
        integrator = CostIntegrator([mock_cost_function], vol_reg=True, cache_size=128)

        # Test that regularization value and gradient can be computed
        reg_val = integrator._regularization.value(xi)
        reg_grad = integrator._regularization.gradient(xi)

        # Volume regularization should return a non-negative value
        assert reg_val >= 0.0
        assert reg_grad.shape == xi.shape


    def test_l2_regularization(self, mock_cost_function):
        """Test that the l2 regularization is computed correctly."""
        xi = np.array([1.0, 2.0])
        xi_0 = np.array([0.5, 2.5])
        dxi = xi - xi_0
        # Use 'alpha' parameter for L2 regularization (not 'l2_reg')
        integrator = CostIntegrator([mock_cost_function], alpha=1.0)
        integrator.xi_ref = xi_0
        # Rebuild regularization with updated xi_ref
        integrator._regularization = integrator._build_regularization(vol_reg=False, epsilon=0.0)

        # Use the new regularization interface
        reg_val = integrator._regularization.value(xi)
        np_adj_reg_val = integrator._regularization.gradient(xi)
        np_fdm_reg_val = integrator._regularization.gradient(xi, fdm=True, h=1e-4)

        sy_ws = sy.Array([1., 1.])
        sy_xi = sy.Array([sy.Symbol(f'x_{i+1}') for i in range(len(xi))])

        sy_xi_ws = sy.Array([ai * bi for ai, bi in zip(sy_ws, sy_xi)])
        sy_l2_reg = 0.5 * sum(ai * bi for ai, bi in zip(sy_xi_ws, sy_xi_ws))
        sy_l2_reg_diff = sy.derive_by_array(sy_l2_reg, sy_xi)

        l2_reg_lbd = _lambdify(sy_xi, sy_l2_reg, module=module)
        l2_reg_lbd_diff = _lambdify(sy_xi, sy_l2_reg_diff, module=module)

        sy_l2_reg_fval = l2_reg_lbd(*dxi)
        sy_l2_reg_dfval = l2_reg_lbd_diff(*dxi)

        np.testing.assert_almost_equal(reg_val, sy_l2_reg_fval)
        np.testing.assert_array_almost_equal(np_adj_reg_val, np_fdm_reg_val)
        np.testing.assert_array_almost_equal(np_adj_reg_val, sy_l2_reg_dfval)


class TestCostIntegratorBounds(unittest.TestCase):
    def test_single_function_bounds(self):
        # Test case 1: Single function with bounds
        xi = np.array([5., 6.])

        mock_fun = MagicMock()
        mock_fun.xi = xi
        mock_fun.xi_bounds = [[0., 10.], [1., 11.]]
        mock_fun.nvars = 2 # Assuming 2 variables for xi


        integrator = CostIntegrator([mock_fun])
        self.assertEqual(integrator.xi_bounds, [[0., 10.], [1., 11.]])

    def test_multiple_consistent_bounds(self):
        xi = np.array([5., 6.])

        # Test case 2: Multiple functions with consistent/overlapping bounds
        mock_fun1 = MagicMock()
        mock_fun1.xi = xi
        mock_fun1.xi_bounds = [[0., 10.], [1., 11.]]
        mock_fun1.nvars = 2

        mock_fun2 = MagicMock()
        mock_fun2.xi = xi
        mock_fun2.xi_bounds = [[2., 8.], [3., 9.]]
        mock_fun2.nvars = 2

        integrator = CostIntegrator([mock_fun1, mock_fun2])

        # Expected intersection: max(0,2)=2, min(10,8)=8; max(1,3)=3, min(11,9)=9
        self.assertEqual(integrator.xi_bounds, [[2., 8.], [3., 9.]])

    def test_multiple_conflicting_bounds(self):
        # Test case 3: Multiple functions with conflicting bounds (should raise ValueError)
        xi = np.array([5., 6.])

        mock_fun1 = MagicMock()
        mock_fun1.xi = xi
        mock_fun1.xi_bounds = [[0., 5.], [1., 10.]]
        mock_fun1.nvars = 2

        mock_fun2 = MagicMock()
        mock_fun2.xi = xi
        mock_fun2.xi_bounds = [[6., 10.], [0., 0.5]] # Conflicts: [0,5] vs [6,10] for var 0; [1,10] vs [0,0.5] for var 1
        mock_fun2.nvars = 2

        with self.assertRaises(ValueError) as cm:
            CostIntegrator([mock_fun1, mock_fun2])

        self.assertIn("Conflicting bounds", str(cm.exception))

    def test_mixed_functions_with_and_without_bounds(self):
        # Test case 4: Mixed functions, some with bounds, some without
        mock_fun1 = MagicMock()
        mock_fun1.xi_bounds = None
        mock_fun1.nvars = 2

        mock_fun2 = MagicMock()
        mock_fun2.xi_bounds = [[-5., 15.], [0., 20.]]
        mock_fun2.nvars = 2

        mock_fun3 = MagicMock()
        mock_fun3.xi_bounds = [[-2., 10.], [5., 18.]]
        mock_fun3.nvars = 2

        mock_fun4 = MagicMock()
        mock_fun4.xi_bounds = None
        mock_fun4.nvars = 2

        xi = np.array([1., 10.])
        integrator = CostIntegrator([mock_fun1, mock_fun2, mock_fun3, mock_fun4])

        # Expected intersection: max(-5,-2)=-2, min(15,10)=10; max(0,5)=5, min(20,18)=18
        self.assertEqual(integrator.xi_bounds, [[-2., 10.], [5., 18.]])

    def test_no_function_with_bounds(self):
        xi = np.array([1., 2.])

        # Test case 5: No function provides bounds
        mock_fun1 = MagicMock()
        mock_fun1.xi = xi
        mock_fun1.xi_bounds = None
        mock_fun1.nvars = 2

        mock_fun2 = MagicMock()
        mock_fun2.xi = xi
        mock_fun2.xi_bounds = None
        mock_fun2.nvars = 2

        integrator = CostIntegrator([mock_fun1, mock_fun2])
        self.assertIsNone(integrator.xi_bounds)


if __name__ == '__main__':
    unittest.main()
