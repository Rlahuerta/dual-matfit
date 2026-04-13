# import sys
import pytest
import unittest

pytestmark = pytest.mark.integration
import time
import numpy as np
import pandas as pd
import sympy as sy
import jax.numpy as jnp

from typing import Sequence, Tuple, List, Dict, Callable, Union, Optional
from dualmatfit.optimization.cost import LSQFit, CostIntegrator, CostFunction
from dualmatfit.formulation.variational import VariationalFormulation
from dualmatfit.optimization.loss import lsq_fval, lsq_dfval


# module = "numpy"
module = "jax"


def xlog_fun(x, param):
    """
    Computes an exponential logarithmic function with a linear component.

    Parameters:
    ----------
    x : array-like or float
        Input value(s) at which to evaluate the function. This could be a single value or an array of values.

    param : list or array-like
        A list or array of parameters with length 4:
            param[0] : float
                Amplitude of the exponential decay.
            param[1] : float
                Decay rate of the exponential term.
            param[2] : float
                Constant offset.
            param[3] : float
                Slope of the linear term.

    Returns:
    -------
    ndarray or float
    """
    return param[0] * jnp.exp(-param[1] * x) + param[2] + x * param[3]

def sym_xlog_fun():
    params = sy.symbols('param_0 param_1 param_2 param_3')

    # Define x as a sympy Array (or you can use Matrix if you prefer)
    x_axes = sy.symbols('x')  # Assuming x has up to 10 elements; adjust as necessary

    return params[0] * sy.exp(-params[1] * x_axes) + params[2] + x_axes * params[3], params, x_axes


def fdm(fun: Callable, xi: np.ndarray, h: float = 1.e-5) -> np.ndarray:
    """ FDM """

    list_dfun_dxi = []
    fval = fun(xi)

    for i, xi_i in enumerate(xi):
        xi_fwd_i = xi.copy()
        xi_fwd_i[i] += h

        fwd_i = fun(xi_fwd_i)
        list_dfun_dxi.append((fval - fwd_i) / h)

    return np.array(list_dfun_dxi)


class TestMockLSQIntegrator(unittest.TestCase):
    def setUp(self):
        self.xi_ini = np.array([2.5, 1.3, 0.5, 0.01], dtype=float)
        self.nvars = self.xi_ini.shape[0]
        self.ftype = 'sum'
        self.fid = None

        # Initialize random seed for reproducibility
        self.seed = 42

        # Set rho value for KS function
        self.rho = 20.

        dsvars_data = {
            'values': self.xi_ini,
            'variable': np.ones(self.nvars, dtype=bool),
            'lower': 0.01 * np.ones(self.nvars, dtype=float),
            'upper': 5. * np.ones(self.nvars, dtype=float)
        }

        self.mat_params = pd.DataFrame(dsvars_data, index=['x0', 'x1', 'x2', 'x3'])

        # Single cost function integrator
        self.lsq_fun = LSQFit(10, 0.2, self.mat_params, xlog_fun, seed=self.seed)
        self.single_cost_fun_integrator = CostIntegrator(lsq_mat_fun=[self.lsq_fun])

        # Multiple cost functions
        self.list_delta = [0.05, 0.03, -0.15]
        self.list_lsq_fun = [LSQFit(10, dt_i, self.mat_params, xlog_fun, seed=self.seed) for dt_i in self.list_delta]
        self.multi_cost_fun_integrator = CostIntegrator(lsq_mat_fun=self.list_lsq_fun, ftype=self.ftype)

    @staticmethod
    def _fun_ln_sum(residuum: np.ndarray, residuum_diff: np.ndarray = None, weight: np.ndarray = None) -> (float, np.ndarray):

        residuum_in = residuum.copy()

        if len(residuum_in.shape) == 1:
            residuum_in = np.expand_dims(residuum_in, axis=0)

        if weight is None:
            weight = np.ones(residuum_in.shape[0])

        fval = 0.
        list_fvals = []
        for i in range(residuum_in.shape[0]):
            residuum2_i = np.dot(residuum_in[i, :], residuum_in[i, :])
            fval_i = np.log1p(residuum2_i)

            fval += fval_i * weight[i]
            list_fvals.append(fval_i)

        if residuum_diff is not None:
            # Compute total derivative of residuum
            residuum_diff_in = residuum_diff.copy()

            if len(residuum_diff_in.shape) == 2:
                residuum_diff_in = np.expand_dims(residuum_diff_in, axis=0)

            list_dfvals = []
            for i in range(residuum_in.shape[0]):
                denominator_i = 1. + np.dot(residuum_in[i, :], residuum_in[i, :])
                coefficients_i = (2. * residuum_in[i, :]) / denominator_i
                np_dfval_i = np.dot(coefficients_i, residuum_diff_in[i, :, :])
                list_dfvals.append(np_dfval_i * weight[i])

            return fval, np.sum(list_dfvals, axis=0)

        else:
            return fval

    def fun_ln_sum(self, xi: np.ndarray) -> float:

        np_residuum = np.array([fun.residuum(xi) for fun in self.list_lsq_fun])  # Shape: (nfun, ncontrol)

        np_residuum_main = np_residuum[self.multi_cost_fun_integrator._fid, :]
        np_residuum_stab = np.delete(np_residuum.copy(), self.multi_cost_fun_integrator._fid, axis=0)

        fval_main = self._fun_ln_sum(np_residuum_main)
        fval_stab = self._fun_ln_sum(np_residuum_stab)

        return (1. - self.multi_cost_fun_integrator._stab) * fval_main + self.multi_cost_fun_integrator._stab * fval_stab

    def fun_sum(self, xi: np.ndarray) -> float:

        np_residuum = np.array([fun.residuum(xi) for fun in self.list_lsq_fun])  # Shape: (nfun, ncontrol)

        list_fvals = []

        for i in range(np_residuum.shape[0]):
            list_fvals.append(lsq_fval(np_residuum[i, :]))

        return sum(list_fvals)

    def test_call_single_cost_function(self):
        """Test the integrator with a single cost function."""

        xi = np.array([2., 1.2, 0.7, 0.01], dtype=float)

        fval_fun = self.lsq_fun(xi)
        fval_lsq = self.single_cost_fun_integrator(xi)

        self.assertEqual(fval_fun, fval_lsq)

        dfval_fun = self.lsq_fun.derivative(xi)
        dfval_lsq = self.single_cost_fun_integrator.derivative(xi)

        np.testing.assert_array_equal(dfval_fun, dfval_lsq)

    def test_call_multi_cost_function_sum(self):
        """Test the integrator with multiple cost functions using 'sum' ftype."""

        xi = np.array([2.0, 1.2, 0.7, 0.01], dtype=float)
        self.multi_cost_fun_integrator._ftype = 'lsq'
        self.multi_cost_fun_integrator._stab = 1.

        # Compute expected function value and derivative
        np_residuum = np.array([fun.residuum(xi) for fun in self.list_lsq_fun])
        np_dresiduum = np.array([fun._cost_function_diff(fun._xdata, xi) for fun in self.list_lsq_fun])

        list_fvals = []
        list_dfvals = []

        for i in range(np_residuum.shape[0]):
            list_fvals.append(lsq_fval(np_residuum[i, :]))
            list_dfvals.append(lsq_dfval(np_residuum[i, :], np_dresiduum[i, :]))

        expected_fval = sum(list_fvals)
        expected_dfval = np.sum(list_dfvals, axis=0)

        # Compute using LSQIntegrator
        fval = self.multi_cost_fun_integrator(xi)
        dfval = self.multi_cost_fun_integrator.derivative(xi)

        fdm_dfval = fdm(self.fun_sum, xi, h=1.e-5)

        self.assertAlmostEqual(fval, expected_fval, places=6)
        np.testing.assert_array_almost_equal(dfval, expected_dfval, decimal=5)
        np.testing.assert_array_almost_equal(fdm_dfval, expected_dfval, decimal=1)

    def test_multi_cost_function_mse(self):
        """Test the integrator with multiple cost functions using 'ln_sum' ftype."""
        xi = np.array([2.0, 1.2, 0.7, 0.01], dtype=float)
        self.multi_cost_fun_integrator._ftype = 'ln'
        self.multi_cost_fun_integrator._stab = 0.

        # Compute MSE function
        mse_val = self.multi_cost_fun_integrator.mse(xi)
        self.assertGreater(mse_val, 0.)

    def test_input_validation(self):
        """Test that incorrect input shapes raise ValueError."""
        xi = np.array([2.0, 1.2])  # Incorrect shape
        with self.assertRaises(ValueError):
            self.multi_cost_fun_integrator(xi)


class TestLeastSquareFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Stretch (mix == 1) - Displacement formulation
        cls.lx, cls.ly, cls.lz = sy.symbols('l_x l_y l_z', real=True)

        cls.ar_def_grad = sy.Array([cls.lx, cls.ly, cls.lz])  # Array Format
        cls.mtx_def_grad = sy.Matrix([[cls.lx, 0, 0], [0, cls.ly, 0], [0, 0, cls.lz]])

        # integration area
        cls.ds = 2.

        # Isotropic Materials Symbols
        cls.mu, cls.lbd = sy.symbols('mu lambda', positive=True)
        cls.k1, cls.k2 = sy.symbols('k_1 k_2', positive=True)
        cls.bulk = 0.001
        cls.vol_type = 'simo92'

        cls.num = 501
        cls.np_lx = np.array([1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50])
        cls.np_ly = np.array([0.00, 0.10, 0.15, 0.25, 0.40, 0.55, 0.70, 0.80, 0.90, 0.95, 1.00])

        np_x = np.linspace(0., 3., num=cls.np_lx.shape[0])
        cls.np_force_ref = 0.01 * (np_x ** 2 + np.log(0.1 * np_x + 1.))

        np_mat_params = np.array([0.0135, cls.bulk, 10., 0.1, 0.4 * np.pi, 0.5 * (1 / 3.)], dtype=float)
        np_mat_params_lwr = 0.0001 * np.ones_like(np_mat_params)
        np_mat_params_upp = np.array([1., 1000., 100., 100., np.pi, (1 / 3.)], dtype=float)

        dsvars_data = {
            'values': np_mat_params,
            'variable': np.ones(np_mat_params.shape[0], dtype=bool),
            'lower': np_mat_params_lwr,
            'upper': np_mat_params_upp
        }

        cls.mat_params = pd.DataFrame(dsvars_data, index=['mu', 'D', 'k_1', 'k_2', 'alpha', 'kappa'])

        cls.module = module
        cls.MAX_STRETCH: float = 5.
        cls.MIN_STRETCH: float = 0.2
        cls.STABILITY_THRESHOLD: float = 1.e-6
        cls.MAX_INC: int = 100

        var_form = VariationalFormulation(ds=cls.ds, itype='nh', mix=1, iso_split=False, dvol=True, bulk=cls.bulk,
                                          kappa=True)

        cls.least_square_func = CostFunction(
            var_form=var_form,
            load_ref=cls.np_force_ref,
            stretch_x=cls.np_lx,
            dsvars=cls.mat_params,
            ftype='lsq',
            module=module,
            dtype='adjoint'
        )

        cls.inp_mat_keys = cls.least_square_func.inp_mat_keys

    def test_init(self):
        # Test initialization
        self.assertEqual(self.least_square_func.nvars, self.mat_params.shape[0])
        self.assertTrue(np.array_equal(self.least_square_func.xi,
                                       self.mat_params.loc[self.inp_mat_keys, "values"].values))

    def test_update_variables(self):
        # Test updating design variables

        df_ds_vars = self.mat_params.copy()
        df_ds_vars.iloc[1, 0] = 2.

        self.least_square_func.update_variables(df_ds_vars)

        self.assertEqual(self.least_square_func.nvars, df_ds_vars.shape[0])
        self.assertTrue(np.array_equal(self.least_square_func.xi, df_ds_vars.loc[self.inp_mat_keys, "values"].values))

    def test_residuum(self):
        # Test residuum calculation
        xi = self.mat_params.loc[self.inp_mat_keys, "values"].values
        np_residuum = self.least_square_func.residuum(xi)

        self.assertGreater(np.abs(np_residuum.sum()), 0.)

    def test_call(self):
        # Test objective function evaluation
        xi = self.mat_params.loc[self.inp_mat_keys, "values"].values
        fval = self.least_square_func(xi).item()
        self.assertGreater(fval, 0.0001)

    def test_derivative_fdm(self):
        # Test derivative calculation using finite differences
        xi = self.mat_params.loc[self.inp_mat_keys, "values"].values
        derivative = self.least_square_func.derivative(xi)

        self.assertEqual(derivative.shape[0], xi.shape[0])
        self.assertTrue(np.all(derivative != 0.))

    def test_invalid_ftype(self):
        # Test with invalid function type
        self.least_square_func._ftype = 'invalid'
        xi = self.mat_params.loc[self.inp_mat_keys, "values"].values

        with self.assertRaises(NotImplementedError):
            self.least_square_func(xi)

    def test_invalid_dtype(self):
        # Test with invalid derivative type
        self.least_square_func._dtype = 'invalid'
        xi = self.mat_params.loc[self.inp_mat_keys, "values"].values
        with self.assertRaises(NotImplementedError):
            self.least_square_func.derivative(xi)


class TestLSQIntegrator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Stretch (mix == 1) - Displacement formulation
        cls.lx, cls.ly, cls.lz = sy.symbols('l_x l_y l_z', real=True)

        cls.ar_def_grad = sy.Array([cls.lx, cls.ly, cls.lz])  # Array Format
        cls.mtx_def_grad = sy.Matrix([[cls.lx, 0, 0], [0, cls.ly, 0], [0, 0, cls.lz]])

        # integration area
        cls.ds = 2.

        # random seed
        cls.seed = 10

        # Isotropic Materials Symbols
        cls.mu, cls.lbd = sy.symbols('mu lambda', positive=True)
        cls.k1, cls.k2 = sy.symbols('k_1 k_2', positive=True)
        cls.bulk = 0.01
        cls.vol_type = 'simo92'

        cls.num = 501
        cls.np_lx_ref1 = np.array([1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50])
        cls.np_lx_ref2 = np.array([1.00, 1.06, 1.12, 1.18, 1.24, 1.30, 1.36, 1.42, 1.48, 1.54, 1.60 ])

        np_x_ref1 = np.linspace(0., 3., num=cls.np_lx_ref1.shape[0])
        cls.np_force_ref1 = 0.01 * (np_x_ref1 ** 2 + np.log(0.10 * np_x_ref1 + 1.))
        cls.np_force_ref2 = 0.02 * (np_x_ref1 ** 2 + np.log(0.05 * np_x_ref1 + 1.))

        cls.mat_params = np.array([0.0135, cls.bulk, 10., 0.1, 0.4 * np.pi, 0.5 * (1 / 3.)], dtype=float)
        np_mat_params_lwr = 0.0001 * np.ones_like(cls.mat_params)
        np_mat_params_upp = np.array([1., 1000., 100., 100., np.pi, (1 / 3.)], dtype=float)

        dsvars_data = {
            'values': cls.mat_params,
            'variable': np.ones(cls.mat_params.shape[0], dtype=bool),
            'lower': np_mat_params_lwr,
            'upper': np_mat_params_upp
        }

        cls.mat_params = pd.DataFrame(dsvars_data, index=['mu', 'D', 'k_1', 'k_2', 'alpha', 'kappa'])

        cls.module = module
        cls.MAX_STRETCH: float = 5.
        cls.MIN_STRETCH: float = 0.2
        cls.STABILITY_THRESHOLD: float = 1.e-6
        cls.MAX_INC: int = 100

    def _setup_cfun_integration(self,
                                mix: int,
                                ftype: str,
                                c: float = 0.,
                                alpha: float = 0.,
                                dvol: bool = False,
                                kappa: bool = False,
                                vol_reg: bool = False,
                                ) -> Tuple[CostIntegrator, pd.DataFrame]:

        mat_param_keys = dict(mu=True, D=False, k_1=True, k_2=True, alpha=True, kappa=False)

        if dvol:
            mat_param_keys["D"] = True

        if kappa:
            mat_param_keys["kappa"] = True

        sr_mat_param_keys = pd.Series(mat_param_keys)
        mat_params = self.mat_params[sr_mat_param_keys]

        var_form = VariationalFormulation(ds=self.ds,
                                          itype='nh',
                                          mix=mix,
                                          iso_split=False,
                                          dvol=dvol,
                                          bulk=self.bulk,
                                          kappa=kappa,
                                          hv=False,
                                          )

        least_square_func1 = CostFunction(
            var_form=var_form,
            load_ref=self.np_force_ref1,
            stretch_x=self.np_lx_ref1,
            dsvars=mat_params,
            ftype='lsq_sum',
            module=module,
            dtype='adjoint'
        )

        least_square_func2 = CostFunction(
            var_form=var_form,
            load_ref=self.np_force_ref2,
            stretch_x=self.np_lx_ref2,
            dsvars=mat_params,
            ftype='lsq_sum',
            module=module,
            dtype='adjoint',
        )

        lsq_int_fun = CostIntegrator(dsvars=mat_params,
                                     lsq_mat_fun=[least_square_func1, least_square_func2],
                                     ftype=ftype,
                                     c=c,
                                     alpha=alpha,
                                     rescale=None,
                                     vol_reg=vol_reg,
                                     )

        return lsq_int_fun, mat_params

    def test_cauchy_function(self):

        ftype = 'cauchy_robust'
        dvol_flag = True
        # dvol_flag = False

        dict_fvals = dict()
        dict_dfvals = dict()

        for mix_k in [1, 2, 3]:
            list_fvals_k = []
            list_dfvals_k = []

            for i, c_i in enumerate([10, 20, 40, 60, 80]):
                print(f"c: {c_i}, mix: {mix_k}")
                int_cost_fun_ki, pd_xi_ki = self._setup_cfun_integration(mix_k, ftype, c=c_i, alpha=0., dvol=dvol_flag)
                inp_mat_keys_ki = int_cost_fun_ki.inp_mat_keys

                xi = pd_xi_ki.loc[inp_mat_keys_ki, "values"].values

                fval_ki = int_cost_fun_ki(xi)

                # Local derivative verification
                for fun_j in int_cost_fun_ki.cost_function:
                    # res_j = fun_j.residuum(xi)

                    np_adj_res_grad_j = fun_j.residuum_diff(xi)
                    np_fdm_res_grad_j = fun_j.residuum_diff(xi, fdm=True, h=1.e-4)

                    np.testing.assert_almost_equal(np_adj_res_grad_j, np_fdm_res_grad_j, decimal=5)

                np_adj_res_ki = int_cost_fun_ki._residuum_diff(xi)
                np_fdm_res_ki = int_cost_fun_ki._residuum_diff(xi, fdm=True, h=1.e-4)
                np.testing.assert_almost_equal(np_adj_res_ki, np_fdm_res_ki, decimal=5)

                if mix_k == 3:
                    np_adj_res_grad_ki = int_cost_fun_ki._cost_function_diff(xi)
                    np_fdm_res_grad_ki = int_cost_fun_ki._cost_function_diff(xi, fdm=True, h=1.e-4)
                    np.testing.assert_almost_equal(np_adj_res_grad_ki, np_fdm_res_grad_ki, decimal=5)

                    for fun_j in int_cost_fun_ki.cost_function:
                        np_adj_vol_grad_j = fun_j.volume_diff(xi)
                        np_fdm_vol_grad_j = fun_j.volume_diff(xi, fdm=True, h=1.e-4)
                        np.testing.assert_almost_equal(np_adj_vol_grad_j, np_fdm_vol_grad_j, decimal=5)

                        np_adj_res_grad_j = fun_j.residuum_diff(xi)
                        np_fdm_res_grad_j = fun_j.residuum_diff(xi, fdm=True, h=1.e-4)
                        np.testing.assert_almost_equal(np_adj_res_grad_j, np_fdm_res_grad_j, decimal=5)

                np_adj_dfval_ki = int_cost_fun_ki._cost_function_diff(xi)
                np_fdm_dfval_ki = int_cost_fun_ki._cost_function_diff(xi, fdm=True, h=1.e-4)

                np.testing.assert_array_almost_equal(np_adj_dfval_ki, np_fdm_dfval_ki, decimal=3, err_msg=f"c: {c_i}")

                list_fvals_k.append(fval_ki)
                list_dfvals_k.append(np_adj_dfval_ki)

            dict_fvals[mix_k] = list_fvals_k
            dict_dfvals[mix_k] = list_dfvals_k

    def test_regularization(self):

        # L2 Regularization with alpha > 0
        np.random.seed(self.seed)
        int_cost_fun_m1_lsq_sum, pd_xi = self._setup_cfun_integration(mix=1, ftype='lsq_sum', alpha=1.0)
        pd_xi_ord = pd_xi.loc[int_cost_fun_m1_lsq_sum.inp_mat_keys, :]

        dxi = np.diff(pd_xi_ord.loc[:, ['lower', 'upper']].values)[:, 0]
        xi = np.random.rand(dxi.shape[0]) * dxi + pd_xi_ord.loc[:, 'lower'].values
        xi_0 = pd_xi_ord.loc[:, "values"].values

        # Use the new regularization interface
        reg_fval = int_cost_fun_m1_lsq_sum._regularization.value(xi)
        np_adj_dfval = int_cost_fun_m1_lsq_sum._regularization.gradient(xi)
        np_fdm_dfval = int_cost_fun_m1_lsq_sum._regularization.gradient(xi, fdm=True, h=1.e-3)
        np.testing.assert_array_almost_equal(np_adj_dfval, np_fdm_dfval, decimal=4)

        reg_fval_0 = int_cost_fun_m1_lsq_sum._regularization.value(xi_0)
        np_adj_dfval_0 = int_cost_fun_m1_lsq_sum._regularization.gradient(xi_0)
        np_fdm_dfval_0 = int_cost_fun_m1_lsq_sum._regularization.gradient(xi_0, fdm=True, h=1.e-3)
        np.testing.assert_array_almost_equal(np_adj_dfval_0, np_fdm_dfval_0, decimal=4)

        self.assertGreater(reg_fval, reg_fval_0)
        self.assertNotEqual(reg_fval_0, reg_fval)
        self.assertGreater(np.abs(np_adj_dfval).sum(), 0.)

    def test_volume_regularization(self):

        # Volume Regularization - requires vol_reg=True AND epsilon > 0
        int_fun_lsq, pd_xi = self._setup_cfun_integration(mix=3, ftype='lsq_sum', dvol=True, vol_reg=True)
        
        # Need to set epsilon to enable volume regularization
        int_fun_lsq._epsilon = 1.0
        # Rebuild regularization with new epsilon
        int_fun_lsq._regularization = int_fun_lsq._build_regularization(vol_reg=True, epsilon=1.0)
        
        pd_xi_ord = pd_xi.loc[int_fun_lsq.inp_mat_keys, :]
        xi_0 = pd_xi_ord.loc[:, "values"].values

        strain_vol_energy = []

        list_adj_grad = []
        list_fdm_grad = []
        for fun_i in int_fun_lsq.cost_function:
            strain_vol_energy.append(fun_i.volume(xi_0))
            np_adj_grad_i = fun_i.volume_diff(xi_0)
            np_fdm_grad_i = fun_i.volume_diff(xi_0, fdm=True, h=1.e-5)
            np.testing.assert_array_almost_equal(np_adj_grad_i, np_fdm_grad_i, decimal=4)

            list_adj_grad.append(np_adj_grad_i)
            list_fdm_grad.append(np_fdm_grad_i)

        # Use the new regularization interface for volume regularization
        svol_fval = int_fun_lsq._regularization.value(xi_0)

        np_svol_adj_dfval = int_fun_lsq._regularization.gradient(xi_0)
        np_svol_fdm_dfval = int_fun_lsq._regularization.gradient(xi=xi_0, fdm=True, h=1.e-5)

        self.assertGreater(svol_fval, 0.)
        np.testing.assert_array_almost_equal(np_svol_adj_dfval, np_svol_fdm_dfval, decimal=4)

    def test_tikhonov_regularization_gradient(self):
        """Ensure gradient accumulation uses vector derivative (reg_dfval) not scalar reg_fval (regression test)."""
        # Setup integrator with alpha > 0 (no volume regularization to isolate effect)
        alpha = 0.5
        int_fun_lsq, pd_xi = self._setup_cfun_integration(mix=1, ftype='lsq_sum', alpha=alpha, dvol=False)
        xi = pd_xi.loc[int_fun_lsq.inp_mat_keys, "values"].values.copy()

        # Get analytic cost function gradient
        np_adj_grad = int_fun_lsq._cost_function_diff(xi)

        # Finite difference gradient of full cost (including Tikhonov)
        np_fdm_grad = int_fun_lsq._cost_function_diff(xi, fdm=True, h=1.e-5)

        # Compare gradients (loose tolerance due to numerical diff)
        np.testing.assert_allclose(np_fdm_grad, np_adj_grad, rtol=1e-4, atol=1e-5,
                                   err_msg="Tikhonov gradient mismatch; possible regression of previous bug.")

        # Sanity: regularization term increases cost if we move away from xi (shift by small perturbation)
        # Get cost function
        fval = int_fun_lsq._cost_function(xi)

        xi_pert = xi + 1.e-2
        fval_pert = int_fun_lsq._cost_function(xi_pert)
        self.assertNotEqual(fval_pert, fval)

        # Also test with volume regularization active simultaneously
        lsq_int_fun_vol, pd_xi_v = self._setup_cfun_integration(mix=1, ftype='lsq_sum', alpha=alpha, dvol=True)
        pd_xi_v = pd_xi_v.loc[lsq_int_fun_vol.inp_mat_keys, :]
        np_idx_v = np.asarray([ik in pd_xi.index for ik in pd_xi_v.index], dtype=bool)

        fval_v = lsq_int_fun_vol._cost_function(pd_xi_v.loc[:, "values"].values)
        self.assertGreater(fval_v, fval)  # Volume reg adds to cost

        np_adj_grad_v = lsq_int_fun_vol._cost_function_diff(pd_xi_v.loc[:, "values"].values)

        np.testing.assert_array_almost_equal(np_adj_grad, np_adj_grad_v[np_idx_v], decimal=0,
                                   err_msg="Gradient mismatch with both Tikhonov and volume regularization too much!!")


if __name__ == '__main__':
    unittest.main()
