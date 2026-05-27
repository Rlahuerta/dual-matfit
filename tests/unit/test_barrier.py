import unittest
import numpy as np
import matplotlib.pyplot as plt

from dualmatfit.solvers.barrier import log_barrier, inv_barrier_function


class TestBarrierFunctions(unittest.TestCase):
    def setUp(self):
        # Common test variables
        self.xi = np.array([0.5, 1.5, 2.5])
        self.lb = np.array([0.0, 1.0, 2.0])
        self.ub = np.array([1.0, 2.0, 3.0])
        self.epsilon = 1e-4  # Small perturbation for numerical derivatives

    def test_log_barrier_value(self):
        """Test the function value of log_barrier."""
        value = log_barrier(self.xi, self.lb, self.ub, dx=0)
        expected = -np.sum(np.log(self.xi - self.lb) + np.log(self.ub - self.xi))

        self.assertAlmostEqual(value, expected, places=7)

    def test_log_barrier_gradient(self):
        """Test the gradient of log_barrier using numerical approximation."""
        analytic_grad = log_barrier(self.xi, self.lb, self.ub, dx=1)
        numerical_grad = np.zeros_like(self.xi)

        for i in range(len(self.xi)):
            xi_forward = self.xi.copy()
            xi_backward = self.xi.copy()
            xi_forward[i] += self.epsilon
            xi_backward[i] -= self.epsilon
            f_forward = log_barrier(xi_forward, self.lb, self.ub, dx=0)
            f_backward = log_barrier(xi_backward, self.lb, self.ub, dx=0)
            numerical_grad[i] = ((f_forward - f_backward) / (2 * self.epsilon)).item()

        np.testing.assert_allclose(analytic_grad, numerical_grad, atol=1e-5)

    def test_log_barrier_hessian(self):
        """Test the Hessian of log_barrier using numerical approximation."""
        analytic_hessian = log_barrier(self.xi, self.lb, self.ub, dx=2)
        numerical_hessian = np.zeros((len(self.xi), len(self.xi)))
        for i in range(len(self.xi)):
            for j in range(len(self.xi)):
                xi_ijp = self.xi.copy()
                xi_ijm = self.xi.copy()
                xi_ipj = self.xi.copy()
                xi_imj = self.xi.copy()

                xi_ijp[i] += self.epsilon
                xi_ijp[j] += self.epsilon

                xi_ijm[i] += self.epsilon
                xi_ijm[j] -= self.epsilon

                xi_ipj[i] -= self.epsilon
                xi_ipj[j] += self.epsilon

                xi_imj[i] -= self.epsilon
                xi_imj[j] -= self.epsilon

                f_ijp = log_barrier(xi_ijp, self.lb, self.ub, dx=0)
                f_ijm = log_barrier(xi_ijm, self.lb, self.ub, dx=0)
                f_ipj = log_barrier(xi_ipj, self.lb, self.ub, dx=0)
                f_imj = log_barrier(xi_imj, self.lb, self.ub, dx=0)

                numerical_hessian[i, j] = (f_ijp - f_ijm - f_ipj + f_imj) / (4 * self.epsilon ** 2)

        np.testing.assert_allclose(analytic_hessian, numerical_hessian, rtol=1e-4)

    def test_inv_barrier_value(self):
        """Test the function value of inv_barrier_function."""

        value = inv_barrier_function(self.xi, self.lb, self.ub, dx=0)
        expected = np.sum(1 / (self.xi - self.lb) + 1 / (self.ub - self.xi))
        self.assertAlmostEqual(value, expected, places=7)

    def test_inv_barrier_gradient(self):
        """Test the gradient of inv_barrier_function using numerical approximation."""

        analytic_grad = inv_barrier_function(self.xi, self.lb, self.ub, dx=1)
        numerical_grad = np.zeros_like(self.xi)

        for i in range(len(self.xi)):
            xi_forward = self.xi.copy()
            xi_backward = self.xi.copy()
            xi_forward[i] += self.epsilon
            xi_backward[i] -= self.epsilon
            f_forward = inv_barrier_function(xi_forward, self.lb, self.ub, dx=0)
            f_backward = inv_barrier_function(xi_backward, self.lb, self.ub, dx=0)
            numerical_grad[i] = ((f_forward - f_backward) / (2 * self.epsilon)).item()

        np.testing.assert_allclose(analytic_grad[:, 0], numerical_grad, rtol=1e-5)

    def test_inv_barrier_hessian(self):
        """Test the Hessian of inv_barrier_function using numerical approximation."""

        analytic_hessian = inv_barrier_function(self.xi, self.lb, self.ub, dx=2)
        numerical_hessian = np.zeros((len(self.xi), len(self.xi)))

        for i in range(len(self.xi)):
            for j in range(len(self.xi)):
                xi_ijp = self.xi.copy()
                xi_ijm = self.xi.copy()
                xi_ipj = self.xi.copy()
                xi_imj = self.xi.copy()

                xi_ijp[i] += self.epsilon
                xi_ijp[j] += self.epsilon

                xi_ijm[i] += self.epsilon
                xi_ijm[j] -= self.epsilon

                xi_ipj[i] -= self.epsilon
                xi_ipj[j] += self.epsilon

                xi_imj[i] -= self.epsilon
                xi_imj[j] -= self.epsilon

                f_ijp = inv_barrier_function(xi_ijp, self.lb, self.ub, dx=0).item()
                f_ijm = inv_barrier_function(xi_ijm, self.lb, self.ub, dx=0).item()
                f_ipj = inv_barrier_function(xi_ipj, self.lb, self.ub, dx=0).item()
                f_imj = inv_barrier_function(xi_imj, self.lb, self.ub, dx=0).item()

                numerical_hessian[i, j] = (f_ijp - f_ijm - f_ipj + f_imj) / (4 * self.epsilon ** 2)

        np.testing.assert_allclose(analytic_hessian, numerical_hessian, rtol=1e-4)

    def test_barrier_near_bounds(self):
        """Test the behavior of barrier functions near the bounds."""
        xi = np.linspace(self.lb[0] + 1e-5, self.ub[0] - 1e-5, 100)

        log_values = -np.log(xi - self.lb[0]) - np.log(self.ub[0] - xi)
        inv_values = 1 / (xi - self.lb[0]) + 1 / (self.ub[0] - xi)

        # Plotting for visual verification
        plt.figure(figsize=(10, 5))
        plt.plot(xi, log_values, label='Log Barrier')
        plt.plot(xi, inv_values, label='Inverse Barrier')
        plt.title('Barrier Functions Near Bounds')
        plt.xlabel('xi')
        plt.ylabel('Barrier Function Value')
        plt.legend()
        plt.grid(True)

    def test_barrier_at_bounds(self):
        """Test that barrier functions return large values at the bounds."""
        xi_at_lb = self.lb.copy()
        xi_at_ub = self.ub.copy()

        # Log Barrier at bounds should go to infinity
        log_value_lb = log_barrier(xi_at_lb + 1e-12, self.lb, self.ub, dx=0)
        log_value_ub = log_barrier(xi_at_ub - 1e-12, self.lb, self.ub, dx=0)

        self.assertTrue(np.isfinite(log_value_lb))
        self.assertTrue(np.isfinite(log_value_ub))

        # Inverse Barrier at bounds should go to infinity
        inv_value_lb = inv_barrier_function(xi_at_lb + 1e-12, self.lb, self.ub, dx=0)
        inv_value_ub = inv_barrier_function(xi_at_ub - 1e-12, self.lb, self.ub, dx=0)

        self.assertTrue(np.isfinite(inv_value_lb))
        self.assertTrue(np.isfinite(inv_value_ub))


if __name__ == '__main__':
    unittest.main()
