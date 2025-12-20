# -*- coding: utf-8 -*-

import os
# import pytest
import unittest
# import itertools
import numpy as np
import pandas as pd
# import sympy as sy
from typing import Tuple


from dualmatfit.drivers import opt_solvers
# from dualmatfit.solution import Root
from dualmatfit.experimental import InstronData
from dualmatfit.least_square import CostFunction, CostIntegrator
from dualmatfit.material_law import volumetric_strain
from dualmatfit.variational_form import VariationalFormulation, mixed_strain_energy_functional
from dualmatfit.extension_solution import ExtensionSolution
from dualmatfit.plot import PlotSolution2D
from dualmatfit.plotting.experimental_visuals import plot_material_fit


# Base directory for solution tests plots
current_file_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.join(current_file_path, "tests_plots", "derivative")

# Create the base directory if it doesn't exist
os.makedirs(work_path, exist_ok=True)
LIMIT_TIME = 60

da = 0.5
bulk = 0.01

# mu, D, k1, k2, alpha, kappa
np_mat_params = np.array([0.005, bulk, 2., 0.05, 0.4 * np.pi, 0.5 * (1 / 3.)], dtype=float)
np_mat_params_lwr = 0.0001 * np.ones_like(np_mat_params)
np_mat_params_upp = np.array([1., 1000., 100., 100., np.pi, (1 / 3.)], dtype=float)


def _bounds_setup(mix: int, min_stretch, max_stretch) -> Tuple[np.ndarray, np.ndarray]:
    np_bounds_lwr = min_stretch * np.ones(2, dtype=float)
    np_bounds_upp = max_stretch * np.ones(2, dtype=float)

    if mix == 2:
        np_bounds_lwr = np.concatenate([np_bounds_lwr, [None]])
        np_bounds_upp = np.concatenate([np_bounds_upp, [None]])

    elif mix == 3:
        np_bounds_lwr = np.concatenate([np_bounds_lwr, [None, None]])
        np_bounds_upp = np.concatenate([np_bounds_upp, [None, None]])

    return np_bounds_lwr, np_bounds_upp


class TestVariationalFormulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # integration area
        cls.da = da

        # Isotropic Materials Symbols
        cls.bulk = bulk
        cls.vol_type = 'simo92'

        # cls.num = 501
        cls.ncontrol = 21
        cls.np_lx = np.linspace(1., 1.5, num=cls.ncontrol)
        cls.np_ly = np.linspace(1., 0.7, num=cls.ncontrol)

        cls.ds = 0.5
        cls.dp = 1.             # pi * diameter / 2
        cls.lxr = cls.dp / 2.
        cls.np_x = cls.lxr * (cls.np_lx - 1.)

        np_z = np.linspace(0., 3., num=cls.np_lx.shape[0])
        cls.np_force_ref = 0.01 * (np_z ** 2 + np.log(0.1 * np_z + 1.))

        # Create sample pd_data DataFrame
        data = {
            'Time': np.linspace(0., 11., cls.ncontrol),
            'Extension': cls.np_x,                                  # must be displacement
            'Load': cls.np_force_ref,                               # because it is a ring
        }
        cls.df_data = pd.DataFrame(data)
        cls.info_data = {'tcontrol': (0., 11.), 'dp': 1., 'ds': 0.5, 'sample_id': 'TestRat-Ar-A', }

        # Initialize the InstronData instance
        cls.instron_data = InstronData(
            df_data=cls.df_data,
            info_data=cls.info_data,
            ncontrol=cls.ncontrol,
            )

        dsvars_data = {
            'values': np_mat_params,
            'variable': np.ones(np_mat_params.shape[0], dtype=bool),
            'lower': np_mat_params_lwr,
            'upper': np_mat_params_upp
        }

        cls.mat_params = pd.DataFrame(dsvars_data, index=['mu', 'D', 'k_1', 'k_2', 'alpha', 'kappa'])

        # cls.module = 'numpy'
        cls.module = 'jax'

    def setUp(self):

        list_opt_kwargs = []
        for iso_split_j in [True, False]:
            for was_k in [True, False]:
                list_opt_kwargs.append(dict(iso_split=iso_split_j, was=was_k))

        self.inp_extsolu_kwargs = list_opt_kwargs

    def _setup_nh_exp(self, dvol: bool = True, **kwargs) -> Tuple[VariationalFormulation, ExtensionSolution]:
        """Test ExtensionSolution with m=mix (displacement formulation)."""

        var_form = VariationalFormulation(ds=self.da, itype='nh', dvol=dvol, bulk=self.bulk, **kwargs)
        ext_sol = ExtensionSolution(var_form, module=self.module, solver_type='least_squares')

        return var_form, ext_sol

    def _extension_mixed_solution_verification(self, mix):
        """Test ExtensionSolution for mixed variational formulations."""

        list_optimal_mat_params = []
        np_force_ref = self.np_force_ref / 2     # because it is a ring

        for i, opt_kwargs_i in enumerate(self.inp_extsolu_kwargs):
            print(f"Testing Mixed Formulation with Options: {opt_kwargs_i}")
            str_opts_i = f'M{mix}'

            if opt_kwargs_i["iso_split"]:
                str_opts_i = str_opts_i + f"_iso_split"

            if opt_kwargs_i["was"]:
                str_opts_i = str_opts_i + f"_was"

            var_form_i, ext_solu_i = self._setup_nh_exp(mix=mix, **opt_kwargs_i, kappa=True)
            inp_mat_keys = ext_solu_i.lbdf_builder.inp_material_keys

            pd_mat_params_i = self.mat_params.loc[inp_mat_keys, :].copy()
            sr_mat_params_i = pd_mat_params_i["values"]

            res_ini_i = ext_solu_i.solve(sr_mat_params_i, stretch_x=self.np_lx)
            res_ini_i["xforce"] = np_force_ref

            lsq_fun_i = CostFunction(
                var_form=var_form_i,
                load_ref=np_force_ref,
                stretch_x=self.np_lx,
                dsvars=self.mat_params.copy(),
                ftype='cauchy_robust',
                # module='numpy',
                module='jax',
                dtype='adjoint'
            )

            np_mat_params_i = sr_mat_params_i[inp_mat_keys].values.copy()

            # residuum derivative check
            np_res_adj_diff_i = lsq_fun_i.residuum_diff(np_mat_params_i)
            np_res_fdm_diff_i = lsq_fun_i.residuum_diff(np_mat_params_i, fdm=True, h=1.e-5)

            np.testing.assert_array_almost_equal(np_res_adj_diff_i, np_res_fdm_diff_i, decimal=3)

            # volume strain energy derivative check
            np_ese_vol_adj_diff_i = lsq_fun_i.volume_diff(np_mat_params_i)
            np_ese_vol_fdm_diff_i = lsq_fun_i.volume_diff(np_mat_params_i, fdm=True, h=1.e-5)

            np.testing.assert_array_almost_equal(np_ese_vol_adj_diff_i, np_ese_vol_fdm_diff_i, decimal=3) # np.nan for undefined ratios

            ptitle_i = "Reaction Force in X-Axis"

            plot_solu_test_2d = PlotSolution2D()
            plot_solu_test_2d.force_plot(ptitle_i,
                                         res_ini_i,
                                         f"{work_path}/force_nh_{str_opts_i}_{self.vol_type}_ini_ps.png",
                                         )

            lsq_int_fun_i = CostIntegrator(lsq_mat_fun=[lsq_fun_i],
                                           ftype='cauchy_robust',
                                           fid=None,
                                           c=20,
                                           alpha=0.,
                                           )

            ##################################################################
            # Comparison of the residuum derivatives from the integrator and the cost function
            np_res_int_fdm_diff_i = lsq_int_fun_i._residuum_diff(np_mat_params_i, fdm=True, h=1.e-5)
            np.testing.assert_array_almost_equal(np_res_int_fdm_diff_i[0, :, :], np_res_adj_diff_i, decimal=4)

            ##################################################################
            # Regularization derivative check (using new regularization interface)
            np_l2_int_diff_i = lsq_int_fun_i._regularization.gradient(np_mat_params_i)
            np_l2_int_fdm_diff_i = lsq_int_fun_i._regularization.gradient(np_mat_params_i, fdm=True, h=1.e-4)
            np.testing.assert_array_almost_equal(np_l2_int_fdm_diff_i, np_l2_int_diff_i, decimal=2)

            ##################################################################
            # Objective Function Evaluation
            obj_fval_ini_i = lsq_int_fun_i(np_mat_params_i)

            np_obj_adj_dfval_ini_i = lsq_int_fun_i._cost_function_diff(np_mat_params_i)
            np_obj_fdm_dfval_ini_i = lsq_int_fun_i._cost_function_diff(np_mat_params_i, fdm=True, h=1.e-5)
            np.testing.assert_array_almost_equal(np_obj_adj_dfval_ini_i, np_obj_fdm_dfval_ini_i, decimal=2)

            opt_args_i = ('ipopt', lsq_int_fun_i, pd_mat_params_i)
            opt_res_i = opt_solvers(*opt_args_i, miter=50, giter=2, glb=False)
            list_optimal_mat_params.append(dict(x=opt_res_i.x, config=opt_kwargs_i))

            obj_fval_opt_i = lsq_int_fun_i(opt_res_i.x)
            np_obj_dfval_opt_i = lsq_int_fun_i.derivative(opt_res_i.x)

            res_opt_i = ext_solu_i.solve(opt_res_i.series, stretch_x=self.np_lx)
            res_opt_i["xforce"] = np_force_ref

            # ===========================================================================
            # Pos Processing and plots
            plot_material_fit(self.instron_data, res_opt_i, work_path,
                              filename_prefix=f"full_nh_{str_opts_i}_{self.vol_type}_opt_mat.png"
                              )

            plot_solu_test_2d.force_plot(ptitle_i,
                                         res_opt_i,
                                         f"{work_path}/force_nh_{str_opts_i}_{self.vol_type}_opt_ps.png",
                                         )

            self.assertGreater(obj_fval_ini_i, obj_fval_opt_i)

            np_dfval_diff_i = np_obj_adj_dfval_ini_i - np_obj_dfval_opt_i
            self.assertGreater(np.dot(np_dfval_diff_i, np_dfval_diff_i), 0.)

    def test_m1(self):
        self._extension_mixed_solution_verification(1)

    def test_m2(self):
        self._extension_mixed_solution_verification(2)

    def test_m3(self):
        self._extension_mixed_solution_verification(3)

if __name__ == "__main__":
    unittest.main()
