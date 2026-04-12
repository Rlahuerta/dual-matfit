import os
import numpy as np
import pandas as pd
import sympy as sy
import unittest

from typing import Union, Tuple, List, Dict, Any, Optional
from scipy import optimize
from dualmatfit.plotting.solution_visuals import PlotSolution2D
from dualmatfit.formulation.variational import VariationalFormulation
from dualmatfit.solvers.extension import ExtensionSolution
from dualmatfit.solvers.solution import Root

current_file_path = os.path.dirname(os.path.abspath(__file__))

# Base directory for solution tests plots
work_path = os.path.join(current_file_path, "tests_plots", "solution")

# Create the base directory if it doesn't exist
os.makedirs(work_path, exist_ok=True)

plot_limits = dict(iso=[-1.5, 1.], ani=[-0.2, 4.2], sum=[-0.2, 3.5], lx=[0.95, 2.])

# module = "numpy"
module = "jax"


def get_detF(results: optimize.OptimizeResult):
    results["detF"] = results.stretch[:, 0] * results.stretch[:, 1] * results.stretch[:, 2]


def get_stretch_check_nrm(results_a: optimize.OptimizeResult, results_b: optimize.OptimizeResult) -> List[float]:

    diff_stretch_y = results_a.stretch[:, 1] - results_b.stretch[:, 1]
    diff_stretch_z = results_a.stretch[:, 2] - results_b.stretch[:, 2]

    nrm_stretch_y = np.sqrt(np.dot(diff_stretch_y, diff_stretch_y))
    nrm_stretch_z = np.sqrt(np.dot(diff_stretch_z, diff_stretch_z))

    return [nrm_stretch_y.item(), nrm_stretch_z.item()]


class TestRoot(unittest.TestCase):

    def setUp(self):
        # Common test variables
        self.fun = lambda x: np.array([x[0]**2 - 4])    # Simple function: x^2 - 4 = 0
        self.jac = lambda x: np.array([[2*x[0]]])       # Its Jacobian

        self.bounds = {'lower': np.array([-10]), 'upper': np.array([10])}
        self.block_array_m1 = [np.array([0])]
        self.block_array_m2 = [np.array([0]), np.array([1])]
        self.block_array_m3 = [np.array([0]), np.array([1]), np.array([2])]

    def test_init(self):
        """Test initialization of the Root class."""

        root_solver = Root(self.fun, jac=self.jac, bounds=self.bounds, mu=0.001, btype='log')
        self.assertEqual(root_solver.mu, 0.001)
        self.assertEqual(root_solver.btype, 'log')
        self.assertTrue(callable(root_solver.fun))
        self.assertTrue(callable(root_solver.jac))
        self.assertTrue(np.array_equal(root_solver.lb, self.bounds['lower']))
        self.assertTrue(np.array_equal(root_solver.ub, self.bounds['upper']))
        self.assertFalse(root_solver.is_mixed) # Check the is_mixed flag

    def test_solve_newton(self):
        """Test solving a simple root-finding problem with Newton's method."""
        x0 = np.array([1.0])
        root_solver = Root(self.fun, jac=self.jac, solver_type='newton')
        result = root_solver.solve(x0)
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.x, [2.0], atol=1e-6)

    def test_solve_least_squares(self):
        """Test solving with least_squares."""
        x0 = np.array([1.0])
        root_solver = Root(self.fun, jac=self.jac, solver_type='least_squares', bounds=self.bounds)
        result = root_solver.solve(x0)
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.x, [2.0], atol=1e-6)

    def test_solve_scipy_root(self):
        """Test solving with scipy.optimize.root."""
        x0 = np.array([1.0])
        root_solver = Root(self.fun, jac=self.jac, solver_type='scipy_root')
        result = root_solver.solve(x0)
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.x, [2.0], atol=1e-6)

    def test_invalid_solver_type(self):
        """Test initialization with an invalid solver type."""

        x0 = np.array([1.0])
        with self.assertRaises(ValueError):
            root_solver = Root(self.fun, self.jac, solver_type='invalid')
            result = root_solver.solve(x0)

    def test_mixed_formulation_m2(self):
        """Test solving a two-field mixed problem (m=2)."""
        def fun(x):
            return np.array([x[0]**2 + x[1] - 2, x[0] - x[1]])

        def jac(x):  # Block Jacobian
            return [[[2*x[0]], [1.0]],
                    [[1.0], [-1.0]]]

        x0 = [np.array([1.0]), np.array([0.0])]  # Initial guess as a list
        root_solver = Root(fun, jac, block_array=self.block_array_m2, solver_type='least_squares') # No need for mtype
        result = root_solver.solve(x0)
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-6)

    def test_mixed_formulation_m3(self):
        """Test solving a three-field mixed problem (m=3)."""
        def fun(x):
            return np.array([x[0]**2 + x[1] - x[2], x[0] - x[1], 2*x[2] + 1])

        def jac(x):  # Block Jacobian
            return [[[2*x[0]], [1.0], [-1.0]],
                    [[1.0], [-1.0], [0.0]],
                    [[0.0], [0.0], [2.0]]]

        x0 = [np.array([-.1]), np.array([-.1]), np.array([-.1])]  # Initial guess as a list
        root_solver = Root(fun, jac, block_array=self.block_array_m3, solver_type='least_squares') # No need for mtype
        result = root_solver.solve(x0)
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.x, [-0.5, -0.611, -0.472], atol=1e-2)

    def test_no_convergence(self):
        """Test the case where the solver doesn't converge."""

        def fun(x):  # A function that's hard to solve near x=0
            return np.array([np.tanh(x[0] * 1000)])

        def jac(x):
            return np.array([[1.0 - np.tanh(x[0] * 1000)**2]]) * 1000

        x0 = np.array([0.1])
        root_solver = Root(fun, jac, max_iter=5)  # Limit iterations
        result = root_solver.solve(x0)
        self.assertFalse(result.success, "Solver should not converge with limited iterations.")

    def test_invalid_x0(self):
        """Test solve method with invalid x0 type."""

        root_solver = Root(self.fun, jac=self.jac)
        with self.assertRaises(TypeError):
            root_solver.solve(x0="invalid")

    def test_high_dimensional_log_barrier(self):
        """Test solving a high-dimensional problem using log barrier."""

        n = 10  # High dimension
        def fun(x):
            # Nonlinear equations: x_i^2 - 100 = 0
            return np.array([x_i ** 2 - 10 for x_i in x])

        def jac(x):
            # Jacobian matrix: 2 * x_i
            np_eye = np.eye(n)
            return 2 * x * np_eye

        x0 = np.full(n, 0.6)  # Initial guess within bounds
        lb = np.full(n, 0.5)  # Lower bounds to exclude negative roots
        ub = np.full(n, 1.5)  # Upper bounds

        bounds = {'lower': lb, 'upper': ub}
        root_solver = Root(fun=fun, jac=jac, bounds=bounds, max_iter=1000, mu=0.1, btype='log')
        result = root_solver.solve(x0)

        self.assertTrue(result.success)

        # Ensure the solution is within bounds
        self.assertTrue(np.all(result.x >= lb))
        self.assertTrue(np.all(result.x <= ub))

        # Expected root is x_i = 1
        np.testing.assert_allclose(result.x, 1.5 * np.ones(n), atol=1e-5)

    def test_solve_bounds_without_barrier(self):
        """Test solving with barrier functions."""
        def fun(x):
            return np.array([x[0] ** 2 - 4])

        def jac(x):
            return np.array([[2 * x[0]]])

        x0 = np.array([3.0])
        bounds = {'lower': np.array([1.0]), 'upper': np.array([5.])}

        root_solver = Root(fun, jac=jac, bounds=bounds, btype=None, solver_type='least_squares')

        result = root_solver.solve(x0=x0)
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.x[0], bounds['lower'][0])
        self.assertLessEqual(result.x[0], bounds['upper'][0])
        np.testing.assert_allclose(result.x, [2.0], atol=1e-6)

    def test_solve_bounds_with_log_barrier(self):
        """Test solving with barrier functions."""
        def fun(x):
            return np.array([x[0] ** 2 - 4])

        def jac(x):
            return np.array([[2 * x[0]]])

        x0 = np.array([3.0])
        bounds = {'lower': np.array([1.0]), 'upper': np.array([5.])}

        root_solver = Root(fun, jac=jac, bounds=bounds, btype="log", max_iter=200)

        result = root_solver.solve(x0=x0)
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.x[0], bounds['lower'][0])
        self.assertLessEqual(result.x[0], bounds['upper'][0])
        np.testing.assert_allclose(result.x, [2.0], atol=1e-6)

    def test_mixed_formulation_scipy_root(self):
        """Test solving a three-field mixed problem with scipy_root."""

        def fun(x):
            return np.array([x[0]**2 + x[1] - x[2], x[0] - x[1], 2*x[2] + 1])

        def jac(x):  # Block Jacobian
            return [[[2*x[0]], [1.0], [-1.0]],
                    [[1.0], [-1.0], [0.0]],
                    [[0.0], [0.0], [2.0]]]

        x0 = [np.array([-1.0]), np.array([-0.4]), np.array([-0.5])]  # Initial guess as a list
        root_solver = Root(fun, jac, block_array=self.block_array_m3, solver_type='scipy_root')
        result = root_solver.solve(x0)

        # self.assertTrue(result.success)
        np.testing.assert_allclose(result.x, [-0.49995689, -0.61111676, -0.47222864], atol=1e-3)

    def test_invalid_bounds(self):
        """Test initialization with invalid bounds."""

        with self.assertRaises(KeyError):
            # Missing 'upper' key
            bounds = {'lower': np.array([-10])}
            Root(self.fun, jac=self.jac, bounds=bounds)

    def test_invalid_btype(self):
        """Test initialization with invalid barrier type."""

        bounds = {'lower': np.array([-10]), 'upper': np.array([10])}
        with self.assertRaises(ValueError):
            root_solver = Root(self.fun, jac=self.jac, bounds=bounds, btype='unknown')

            with self.assertRaises(TypeError):
                root_solver.solve(x0="invalid")


class TestSolutionMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Stretch (mix == 1) - Displacement formulation
        cls.lx, cls.ly, cls.lz = sy.symbols('l_x l_y l_z', real=True)

        cls.ar_def_grad = sy.Array([cls.lx, cls.ly, cls.lz])  # Array Format
        cls.mtx_def_grad = sy.Matrix([[cls.lx, 0, 0], [0, cls.ly, 0], [0, 0, cls.lz]])

        # integration area
        cls.ds = 2.

        # Isotropic Materials Symbols
        cls.mu, cls.lbd = sy.symbols(r'mu \lambda', positive=True)
        cls.k1, cls.k2 = sy.symbols('k_1 k_2', positive=True)
        cls.bulk = 100.
        cls.vol_type = 'simo92'

        cls.num = 501
        cls.np_lx = np.array([1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50])
        cls.np_ly = np.array([0.00, 0.10, 0.15, 0.25, 0.40, 0.55, 0.70, 0.80, 0.90, 0.95, 1.00])

        mat_parameters = np.array([0.0135, cls.bulk, 10., 0.1, 0.4 * np.pi], dtype=float)
        np_mat_params_lwr = 0.0001 * np.ones_like(mat_parameters)
        np_mat_params_upp = np.array([1., 1000., 100., 100., np.pi], dtype=float)

        dsvars_data = {
            'values': mat_parameters,
            'variable': np.ones(mat_parameters.shape[0], dtype=bool),
            'lower': np_mat_params_lwr,
            'upper': np_mat_params_upp
        }

        cls.mat_params = pd.DataFrame(dsvars_data, index=['mu', 'D', 'k_1', 'k_2', 'alpha'])

        cls.module = module
        cls.MAX_STRETCH: float = 5.
        cls.MIN_STRETCH: float = 0.2
        cls.STABILITY_THRESHOLD: float = 1.e-6
        cls.MAX_INC: int = 100

    def _bounds_setup(self, mix: int) -> Tuple[np.ndarray, np.ndarray]:

        np_bounds_lwr = self.MIN_STRETCH * np.ones(2, dtype=float)
        np_bounds_upp = self.MAX_STRETCH * np.ones(2, dtype=float)

        if mix == 2:
            np_bounds_lwr = np.concatenate([np_bounds_lwr, [None]])
            np_bounds_upp = np.concatenate([np_bounds_upp, [None]])

        elif mix == 3:
            np_bounds_lwr = np.concatenate([np_bounds_lwr, [None, None]])
            np_bounds_upp = np.concatenate([np_bounds_upp, [None, None]])

        return np_bounds_lwr, np_bounds_upp

    def _setup_nh_exp(self, mix: int, **ext_kwargs) -> (VariationalFormulation, ExtensionSolution):
        """Test ExtensionSolution with m=mix (displacement formulation)."""
        var_form = VariationalFormulation(ds=self.ds,
                                          itype='nh',
                                          mix=mix,
                                          kappa=False,
                                          dvol=True,
                                          bulk=self.bulk,
                                          iso_split=False,
                                          apply_simplify=False,
                                          )

        ext_sol = ExtensionSolution(var_form, module=self.module, **ext_kwargs)

        return var_form, ext_sol

    def test_extension_mixed_solution(self):
        """Test ExtensionSolution for mixed variational formulations."""

        np_detF_ref = np.ones(self.np_lx.size, dtype=float)
        mat_parameters = self.mat_params['values']

        list_ext_solu_lsq = []
        plot_solu_res_lsq = {}

        list_ext_solu_solver = []
        list_plot_solu_res_solver = []

        print("\n--- neo-hookean material law - mixed variational formulations ---")
        for mix_i in [1, 2, 3]:
            print(f"Try to solve for mixed variational type: {mix_i}")
            var_form_lsq_i, ext_solu_lsq_i = self._setup_nh_exp(mix_i, solver_type='least_squares')
            res_lsq_i = ext_solu_lsq_i.solve(mat_parameters, stretch_x=self.np_lx)

            var_form_hybr_i, ext_solu_hybr_i = self._setup_nh_exp(mix_i, solver_type='scipy_root')
            res_hybr_i = ext_solu_hybr_i.solve(mat_parameters, stretch_x=self.np_lx)

            # Initialization verification
            for ext_solu_ik in [ext_solu_lsq_i, ext_solu_hybr_i]:
                self.assertEqual(ext_solu_ik.var_form.mix, mix_i)
                self.assertEqual(ext_solu_ik.var_form._bulk, self.bulk)

            # Stretch Solution Verification
            nrm2_stretch_yz = get_stretch_check_nrm(res_lsq_i, res_hybr_i)

            self.assertLessEqual(nrm2_stretch_yz[0], 0.1)
            self.assertLessEqual(nrm2_stretch_yz[1], 0.1)

            # Results verification
            for results_ik in [res_lsq_i, res_hybr_i]:
                # self.assertTrue(results_ik.success)
                self.assertEqual(results_ik.stretch.shape[0], self.np_lx.size)

                np.testing.assert_array_almost_equal(results_ik.stretch[:, 0], self.np_lx, decimal=5)
                np.testing.assert_array_almost_equal(np_detF_ref, results_ik.detF, decimal=2)

                # Check that stress in x direction is positive (tension)
                self.assertTrue(np.all(results_ik.stress['full'][1:, 0] >= 0.0))

                # Check that stress in y direction is zero (tension)
                np.testing.assert_array_almost_equal(results_ik.stress['full'][:, 1], np.zeros_like(self.np_lx),
                                                     decimal=6)

                # Check that stress in y direction is zero (tension)
                np.testing.assert_array_almost_equal(results_ik.stress['full'][:, 2], np.zeros_like(self.np_lx),
                                                     decimal=6)

            # Store the Results
            list_ext_solu_lsq.append(ext_solu_lsq_i)
            plot_solu_res_lsq[f"NH-M{mix_i}"] = res_lsq_i

            list_ext_solu_solver.append({f"M{mix_i}-LSQ": ext_solu_lsq_i, f"M{mix_i}-HYBR": ext_solu_hybr_i})
            list_plot_solu_res_solver.append({f"NH-M{mix_i}-LSQ": res_lsq_i, f"NH-M{mix_i}-HYBR": res_hybr_i})

        # Plot the stress results
        ptitle_lsq = f"neo-hookean material law (LSQ Solver) - stress plot"
        ltype = {"NH-M1": "-", "NH-M2": "-.", "NH-M3": "--"}

        plot_solu_test_2d = PlotSolution2D()

        # ##############################################################################################3
        # Plot to compare the mixed solution using least_squares for solver
        plt_fname = f"neo_hookean_{self.vol_type}_plane_stress.png"

        plot_solu_test_2d.components_plot(ptitle_lsq,
                                          plot_solu_res_lsq,
                                          ltype,
                                          f"{work_path}/comps_{plt_fname}",
                                          )

        plot_solu_test_2d.full_plot(ptitle_lsq,
                                    plot_solu_res_lsq,
                                    ltype,
                                    f"{work_path}/total_{plt_fname}",
                                    )

        # ##############################################################################################3
        # Plot to compare the least_squares for solver vs scipy hybrid solver
        line_types = ["-", "-."]

        for i, plot_solu_res_i in enumerate(list_plot_solu_res_solver):
            keys_i  = list(plot_solu_res_i.keys())
            ptitle_i = f"neo-hookean material law - stress plot"
            ltype_i = {key_k: line_types[k]  for k, key_k in enumerate(keys_i)}
            plt_fname_i = f"{keys_i[0][:5]}_{self.vol_type}_plane_stress.png"

            plot_solu_test_2d.components_plot(ptitle_i,
                                              plot_solu_res_i,
                                              ltype_i,
                                              f"{work_path}/solution_{plt_fname_i}",
                                              )


if __name__ == '__main__':
    unittest.main()
