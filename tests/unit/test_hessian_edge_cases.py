# -*- coding: utf-8 -*-
"""
Test edge cases for Hessian finite difference implementation.

Following TDD: These tests are written BEFORE fixing the duplicate function issue
to ensure we catch any regressions.
"""

import numpy as np
import pytest
from dualmatfit.solvers.derivative import _fdm, _hessian_fd


class TestHessianEdgeCases:
    """Edge case tests for Hessian computation."""

    def test_hessian_near_bounds_lower(self):
        """Test Hessian computation when parameter is near lower bound."""
        def func(x):
            return x[0]**2 + x[1]**2

        x0 = np.array([0.001, 1.0])  # Very close to lower bound
        bounds = [[1e-6, 1e6], [1e-6, 1e6]]

        # Should not raise an exception
        hessian = _hessian_fd(func, x0, xi_bounds=bounds)

        # Expected Hessian for x^2 + y^2 is 2*I
        expected = np.array([[2.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_almost_equal(hessian, expected, decimal=3)

    def test_hessian_near_bounds_upper(self):
        """Test Hessian computation when parameter is near upper bound."""
        def func(x):
            return x[0]**2 + x[1]**2

        # Use more moderate values to avoid numerical precision issues
        # when parameters differ by many orders of magnitude
        x0 = np.array([10.0, 1.0])  # Near upper bound for reasonable range
        bounds = [[1e-6, 100.0], [1e-6, 100.0]]

        # Should not raise an exception
        hessian = _hessian_fd(func, x0, xi_bounds=bounds)
        expected = np.array([[2.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_almost_equal(hessian, expected, decimal=3)

    def test_hessian_bounds_prevent_central_difference(self):
        """Test that forward/backward difference is used when central diff is blocked."""
        def func(x):
            return x[0]**2 + x[1]**2

        # x0 first param at upper bound - can only use backward difference
        # Using moderate values for numerical stability
        x0 = np.array([10.0, 1.0])
        bounds = [[1e-6, 10.0], [1e-6, 100.0]]  # First param at upper bound

        # Should use backward difference for first param, central for second
        hessian = _hessian_fd(func, x0, xi_bounds=bounds)
        expected = np.array([[2.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_almost_equal(hessian, expected, decimal=2)

    def test_hessian_multivariate_rosenbrock(self):
        """Test Hessian for Rosenbrock function (more complex nonlinear)."""
        def rosenbrock(x):
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

        x0 = np.array([1.0, 1.0])  # At minimum, Hessian should be positive definite
        hessian = _hessian_fd(rosenbrock, x0)

        # Analytical Hessian at (1, 1):
        # d²f/dx² = 2 + 1200*x² - 400*y = 2 + 1200 - 400 = 802
        # d²f/dxdy = -400*x = -400
        # d²f/dy² = 200
        expected = np.array([[802.0, -400.0], [-400.0, 200.0]])
        np.testing.assert_array_almost_equal(hessian, expected, decimal=1)

    def test_hessian_sin_function(self):
        """Test Hessian for sinusoidal function."""
        def sin_func(x):
            return np.sin(x[0]) + np.cos(x[1])

        x0 = np.array([0.5, 0.5])
        hessian = _hessian_fd(sin_func, x0)

        # Hessian of sin(x) + cos(y) at (0.5, 0.5):
        # d²/dx² = -sin(x) = -sin(0.5) ≈ -0.479
        # d²/dy² = -cos(y) = -cos(0.5) ≈ -0.877
        # d²/dxdy = 0
        expected = np.array([
            [-np.sin(0.5), 0.0],
            [0.0, -np.cos(0.5)]
        ])
        np.testing.assert_array_almost_equal(hessian, expected, decimal=3)

    def test_hessian_exp_function(self):
        """Test Hessian for exponential function."""
        def exp_func(x):
            return np.exp(x[0] + x[1])

        x0 = np.array([0.0, 0.0])
        hessian = _hessian_fd(exp_func, x0)

        # Hessian of exp(x+y) at (0, 0) is all 1s (second derivatives of exp)
        expected = np.array([[1.0, 1.0], [1.0, 1.0]])
        np.testing.assert_array_almost_equal(hessian, expected, decimal=3)

    def test_hessian_ill_conditioned_bounds(self):
        """Test Hessian with parameters at exact bound values."""
        def func(x):
            return x[0]**4 + x[1]**2

        x0 = np.array([0.0, 0.0])  # At lower bound
        bounds = [[0.0, 10.0], [0.0, 10.0]]

        # Should handle gracefully with forward/backward differences
        hessian = _hessian_fd(func, x0, xi_bounds=bounds)

        # d²/dx² of x^4 at x=0 is 0
        # d²/dy² of y^2 at y=0 is 2
        expected = np.array([[0.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_almost_equal(hessian, expected, decimal=2)

    def test_hessian_negative_values(self):
        """Test Hessian with negative parameter values."""
        def func(x):
            return x[0]**2 + x[1]**2

        x0 = np.array([-2.0, -3.0])
        hessian = _hessian_fd(func, x0)

        expected = np.array([[2.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_almost_equal(hessian, expected, decimal=3)

    def test_hessian_custom_step_size(self):
        """Test Hessian with custom step sizes."""
        def func(x):
            return x[0]**2 + x[1]**2

        x0 = np.array([1.0, 1.0])

        # Test with larger step size
        hessian_large = _hessian_fd(func, x0, h=1e-2)
        expected = np.array([[2.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_almost_equal(hessian_large, expected, decimal=2)

    def test_hessian_relative_step(self):
        """Test that relative step size is used for large parameters."""
        def func(x):
            return x[0]**2 + x[1]**2

        x0 = np.array([1000.0, 1000.0])  # Large values

        hessian = _hessian_fd(func, x0, rel_step=1e-5)
        expected = np.array([[2.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_almost_equal(hessian, expected, decimal=2)

    def test_hessian_asymmetric_bounds(self):
        """Test Hessian with asymmetric bounds."""
        def func(x):
            return x[0]**2 + x[1]**2

        x0 = np.array([0.0, 0.0])
        bounds = [[-1.0, 1.0], [0.0, 10.0]]  # Asymmetric

        hessian = _hessian_fd(func, x0, xi_bounds=bounds)
        expected = np.array([[2.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_almost_equal(hessian, expected, decimal=2)


class TestHessianVectorValued:
    """Tests for vector-valued functions."""

    def test_hessian_vector_valued_correct_shape(self):
        """Test that vector-valued function returns correct shape."""
        def func(x):
            return np.array([x[0]**2, x[1]**2, x[0] + x[1]])

        x0 = np.array([1.0, 2.0])
        hessians = _hessian_fd(func, x0)

        # Should return shape (n_outputs, n_params, n_params)
        assert hessians.shape == (3, 2, 2)

    def test_hessian_vector_valued_values(self):
        """Test Hessian values for vector-valued function."""
        def func(x):
            return np.array([x[0]**2, x[1]**2])

        x0 = np.array([2.0, 3.0])
        hessians = _hessian_fd(func, x0)

        expected = np.array([
            [[2.0, 0.0], [0.0, 0.0]],   # Hessian of x[0]^2
            [[0.0, 0.0], [0.0, 2.0]]    # Hessian of x[1]^2
        ])
        np.testing.assert_array_almost_equal(hessians, expected, decimal=3)

    def test_hessian_vector_valued_mixed(self):
        """Test Hessian for vector-valued function with mixed terms."""
        def func(x):
            # f[0] = x[0]^2 + x[1]^2 (diagonal Hessian)
            # f[1] = x[0]*x[1] (mixed derivatives)
            return np.array([x[0]**2 + x[1]**2, x[0]*x[1]])

        x0 = np.array([1.0, 2.0])
        hessians = _hessian_fd(func, x0)

        # Hessian of f[0] = x^2 + y^2 is [[2, 0], [0, 2]]
        # Hessian of f[1] = x*y is [[0, 1], [1, 0]]
        expected = np.array([
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 1.0], [1.0, 0.0]]
        ])
        np.testing.assert_array_almost_equal(hessians, expected, decimal=3)


class TestHessianAgainstGradient:
    """Verify Hessian against numerical gradient of gradient."""

    def test_hessian_consistency_with_gradient(self):
        """Test that Hessian diagonal matches gradient derivative."""
        def func(x):
            return x[0]**4 + 2*x[1]**4 + x[0]*x[1]

        x0 = np.array([1.0, 0.5])
        hessian = _hessian_fd(func, x0)

        # Compute gradient at nearby points to verify Hessian diagonal
        grad_plus = _fdm(func, x0 + np.array([1e-5, 0]))
        grad_minus = _fdm(func, x0 - np.array([1e-5, 0]))

        # d²f/dx² ≈ (grad_x_plus - grad_x_minus) / (2*dx)
        hess_xx_from_grad = (grad_plus[0] - grad_minus[0]) / (2e-5)

        np.testing.assert_almost_equal(hessian[0, 0], hess_xx_from_grad, decimal=2)

    def test_cross_derivatives_symmetric(self):
        """Test that cross derivatives are symmetric (Schwarz theorem)."""
        def func(x):
            return x[0]**2 * x[1]**2 + x[0] + x[1]

        x0 = np.array([1.5, 2.5])
        hessian = _hessian_fd(func, x0)

        # Off-diagonal elements should be equal
        np.testing.assert_almost_equal(hessian[0, 1], hessian[1, 0], decimal=3)


class TestHessianErrorHandling:
    """Test error handling and edge cases."""

    def test_hessian_zero_parameter(self):
        """Test Hessian computation with zero parameter value."""
        def func(x):
            return x[0]**2 + x[1]**2

        x0 = np.array([0.0, 1.0])
        hessian = _hessian_fd(func, x0)
        expected = np.array([[2.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_almost_equal(hessian, expected, decimal=2)

    def test_hessian_scalar_result(self):
        """Test Hessian for function returning scalar (not array)."""
        def func(x):
            return float(x[0]**2 + x[1]**2)  # Explicitly return scalar

        x0 = np.array([1.0, 1.0])
        hessian = _hessian_fd(func, x0)

        assert hessian.shape == (2, 2)
        expected = np.array([[2.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_almost_equal(hessian, expected, decimal=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])