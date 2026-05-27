import unittest
import warnings
import numpy as np
import pandas as pd

from unittest.mock import MagicMock, patch
from scipy.optimize import OptimizeResult

from dualmatfit.formulation.variational import VariationalFormulation
from dualmatfit.formulation.lambdify import LambdifyBuilder
from dualmatfit.solvers.extension import NumericalProblem, ExtensionSolution, _get_initial_guess


class TestExtensionSolution(unittest.TestCase):

    def setUp(self):
        self.itype = 'nh'
        self.mix = 1
        self.kappa = True
        self.dvol = False
        self.bulk = 1000.0
        self.ds = 0.1

        self.vf_args = dict(ds=self.ds, itype=self.itype, mix=self.mix, kappa=self.kappa, dvol=self.dvol, bulk=self.bulk)
        self.var_form = VariationalFormulation(**self.vf_args)

        self.mat_params = np.array([0.5, 1.0, 2.0, 0.0, 0.0])
        mat_params_lwr = 0.0001 * np.ones_like(self.mat_params)
        mat_params_upp = np.array([1., 10., 10., np.pi, (1 / 3.)], dtype=float)

        dsvars_data = {
            'values': self.mat_params,
            'variable': np.ones(self.mat_params.shape[0], dtype=bool),
            'lower': mat_params_lwr,
            'upper': mat_params_upp
        }
        self.dsvars = pd.DataFrame(dsvars_data, index=['mu', 'k_1', 'k_2', 'alpha', 'kappa'])

        self.np_tstretch_x = np.array([1.1, 1.2])

    def test_init(self):
        ext_sol = ExtensionSolution(self.var_form)
        self.assertIsInstance(ext_sol, ExtensionSolution)
        self.assertEqual(ext_sol.nmat_vars, len(self.var_form.mat_vars))
        self.assertEqual(ext_sol.nprm_vars, len(self.var_form.primal_vars))

    def test_init_bounds_mix2(self):
        vf_args = self.vf_args.copy()
        vf_args['mix'] = 2

        var_form_mix2 = VariationalFormulation(**vf_args)
        ext_sol = ExtensionSolution(var_form_mix2)

        self.assertEqual(len(ext_sol.primal_bounds['lower']), 3)
        self.assertEqual(ext_sol.primal_bounds['lower'][-1], -100.)

    def test_init_bounds_mix3(self):
        vf_args = self.vf_args.copy()
        vf_args['mix'] = 3

        var_form_mix3 = VariationalFormulation(**vf_args)
        ext_sol = ExtensionSolution(var_form_mix3)

        self.assertEqual(len(ext_sol.primal_bounds['lower']), 4)
        self.assertLessEqual(ext_sol.primal_bounds['lower'][-2], 0.)

    def test_solve_invalid_mat_params_type(self):
        ext_sol = ExtensionSolution(self.var_form)
        with self.assertRaises(TypeError):
            ext_sol.solve("not_an_array", self.np_tstretch_x)

    def test_solve_invalid_mat_params_shape(self):
        ext_sol = ExtensionSolution(self.var_form)
        with self.assertRaises(TypeError):
            ext_sol.solve(np.array([1.0]), self.np_tstretch_x)

    def test_solve_invalid_stretch_x_dim(self):
        ext_sol = ExtensionSolution(self.var_form)
        sr_mat_params = self.dsvars['values']

        with self.assertRaises(ValueError):
            ext_sol.solve(sr_mat_params, np.array([[1.1], [1.2]]))

    def test_solve_convergence_warning(self):
        mock_solver_class = MagicMock()
        mock_solver_instance = mock_solver_class.return_value
        mock_solver_instance.solve.return_value = OptimizeResult(x=np.array([1., 1.]), success=False, message="Failed", fun=0.0)

        ext_sol = ExtensionSolution(self.var_form, solver=mock_solver_class)
        sr_mat_params = self.dsvars['values']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ext_sol.solve(sr_mat_params, self.np_tstretch_x)
            self.assertTrue(any("Solver did not converge" in str(warn.message) for warn in w))

    def test_solve_nan_warning(self):
        mock_solver_class = MagicMock()
        mock_solver_instance = mock_solver_class.return_value
        mock_solver_instance.solve.return_value = OptimizeResult(x=np.array([np.nan, np.nan]), success=True, fun=0.0, message="NaNs found")

        ext_sol = ExtensionSolution(self.var_form, solver=mock_solver_class)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sr_mat_params = self.dsvars['values']

            ext_sol.solve(sr_mat_params, self.np_tstretch_x)
            self.assertTrue(any("Solver did not converge" in str(warn.message) for warn in w))

    def test_lambdify_builder_init_jax(self):
        builder = LambdifyBuilder(self.var_form, module='jax')
        self.assertEqual(builder.module, 'jax')

    def test_lambdify_builder_constant_hessian(self):
        vf_args = self.vf_args.copy()
        vf_args['mix'] = 1
        vf_args['kappa'] = False
        vf_args['iso_split'] = False

        var_form = VariationalFormulation(**vf_args)
        builder = LambdifyBuilder(var_form, module='jax')
        self.assertTrue(isinstance(builder.block_hessian[0][0](1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), np.ndarray))

    def test_numerical_problem_build_jacobian_invalid_xi_type(self):
        builder = LambdifyBuilder(self.var_form)
        problem = NumericalProblem(builder, self.var_form, np.ones(3), np.ones(5))
        with self.assertRaises(TypeError):
            problem.build_jacobian("invalid")

    def test_numerical_problem_build_hessian_invalid_xi_type(self):
        builder = LambdifyBuilder(self.var_form)
        problem = NumericalProblem(builder, self.var_form, np.ones(3), np.ones(5))
        with self.assertRaises(TypeError):
            problem.build_hessian("invalid")

    def test_numerical_problem_build_jacobian_penal(self):
        vf_args = self.vf_args.copy()
        vf_args['mix'] = 1
        vf_args['dvol'] = True

        var_form = VariationalFormulation(**vf_args)
        builder = LambdifyBuilder(var_form)
        problem = NumericalProblem(builder, var_form, np.ones(3), np.ones(6))
        _, penal_val = problem.build_jacobian(np.ones(2), penal=True)
        self.assertIsNotNone(penal_val)

    def test_numerical_problem_build_hessian_mix2(self):
        vf_args = self.vf_args.copy()
        vf_args['mix'] = 2

        var_form = VariationalFormulation(**vf_args)
        builder = LambdifyBuilder(var_form)
        problem = NumericalProblem(builder, var_form, np.ones(4), np.ones(5))
        hessian = problem.build_hessian(np.ones(3))
        self.assertIsNotNone(hessian)

    def test_numerical_problem_build_jacobian_mix3(self):
        vf_args = self.vf_args.copy()
        vf_args['mix'] = 3

        var_form = VariationalFormulation(**vf_args)
        builder = LambdifyBuilder(var_form)
        problem = NumericalProblem(builder, var_form, np.ones(5), np.ones(5))
        jacobian = problem.build_jacobian(np.ones(4))
        self.assertIsNotNone(jacobian)

    def test_numerical_problem_build_hessian_mix3(self):
        vf_args = self.vf_args.copy()
        vf_args['mix'] = 3

        var_form = VariationalFormulation(**vf_args)
        builder = LambdifyBuilder(var_form)
        problem = NumericalProblem(builder, var_form, np.ones(5), np.ones(5))
        hessian = problem.build_hessian(np.ones(4))
        self.assertIsNotNone(hessian)

    def test_numerical_problem_build_hessian_nan_warning(self):
        vf_args = self.vf_args.copy()
        vf_args['mix'] = 1

        var_form = VariationalFormulation(**vf_args)
        builder = LambdifyBuilder(var_form)
        builder.block_hessian[0][0] = lambda *args: np.array([[np.nan, np.nan], [np.nan, np.nan]])
        problem = NumericalProblem(builder, var_form, np.ones(3), np.ones(5))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            problem.build_hessian(np.ones(2))
            self.assertTrue(any("NaN values encountered in Hessian calculation" in str(warn.message) for warn in w))

    def test_numerical_problem_build_jacobian_value_error(self):
        builder = LambdifyBuilder(self.var_form)
        problem = NumericalProblem(builder, self.var_form, np.ones(3), np.ones(5))
        with self.assertRaises(ValueError):
            problem.build_jacobian(np.ones(10))

    def test_numerical_problem_build_hessian_value_error(self):
        builder = LambdifyBuilder(self.var_form)
        problem = NumericalProblem(builder, self.var_form, np.ones(3), np.ones(5))
        with self.assertRaises(ValueError):
            problem.build_hessian(np.ones(10))


def test_initial_guess():
    """Test the improved initial guess method for all mix formulations."""

    print("Testing Improved Initial Guess for ExtensionSolution")
    print("=" * 60)

    # Test all mix formulations
    for mix_i in [1, 2, 3]:
        print(f"\n--- Mix {mix_i} Formulation ---")
        try:
            # Test no motion case (lx = 1.0)
            guess_no_motion_i = _get_initial_guess(1., mix=mix_i, incompressible=True)
            detF_no_motion_i = 1.0 * guess_no_motion_i[0] * guess_no_motion_i[1]
            print(f"No motion (lx=1.0): {guess_no_motion_i}, j^r: {detF_no_motion_i:.4f}")

            # Test small deformation case (lx = 1.1)
            guess_small_i = _get_initial_guess(1.1, mix=mix_i, incompressible=True)
            detF_small_i = 1.1 * guess_small_i[0] * guess_small_i[1]
            print(f"Small deform (lx=1.1): {guess_small_i}, j^r: {detF_small_i:.4f}")

            # Test larger deformation case (lx = 1.5)
            guess_large_i = _get_initial_guess(1.5, mix=mix_i, incompressible=True)
            detF_large_i = 1.5 * guess_large_i[0] * guess_large_i[1]
            print(f"Large deform (lx=1.5): {guess_large_i}, j^r: {detF_large_i:.4f}")

            # Show variable meanings
            if mix_i == 1:
                print(f"  Variables: [ly, lz]: {guess_no_motion_i}")
            elif mix_i == 2:
                print(f"  Variables: [ly, lz, p]: {guess_small_i}")
            elif mix_i == 3:
                print(f"  Variables: [ly, lz, p, theta]: {guess_small_i}")

            np.testing.assert_allclose(1., detF_no_motion_i, rtol=1e-5)
            np.testing.assert_allclose(1., detF_small_i, rtol=1e-5)
            np.testing.assert_allclose(1., detF_large_i, rtol=1e-5)

        except Exception as e:
            print(f"Error with Mix {mix_i}: {e}")


if __name__ == '__main__':
    unittest.main()
