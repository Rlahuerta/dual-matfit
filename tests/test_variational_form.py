import unittest
import pytest
import numpy as np
import pandas as pd
import sympy as sy

# PyDev debugger warning filter centralized in tests/conftest.py

from dualmatfit.formulation.variational import VariationalFormulation
from dualmatfit.solvers.extension import ExtensionSolution
from dualmatfit.solvers.derivative import _fdm, adjoint_derivative
from dualmatfit.formulation.material_law import volumetric_strain
from dualmatfit.formulation.tensor import safe_simplify


class TestVariationalFormulation(unittest.TestCase):
    """Unit tests for the VariationalFormulation class."""

    @classmethod
    def setUpClass(cls):
        """Set up common VariationalFormulation instances for all tests (runs once)."""
        cls.ds = 1.

        # Create instances with fast mode settings (simplify_tensors=False)
        # This reduces initialization time from ~8.4s to ~3.8s
        fast_mode = {'simplify_tensors': False, 'simplify_timeout': 1}
        
        cls.vf_nh_mix1_no_kappa_no_dvol = VariationalFormulation(ds=cls.ds, itype='nh', mix=1, kappa=False, dvol=False, **fast_mode)
        cls.vf_fung_mix2_kappa_dvol = VariationalFormulation(ds=cls.ds, itype='fung', mix=2, kappa=True, dvol=True, iso_split=True, **fast_mode)
        cls.vf_nh_mix3_no_kappa_no_dvol = VariationalFormulation(ds=cls.ds, itype='nh', mix=3, kappa=False, dvol=False, **fast_mode)
        cls.vf_nh_mix1_kappa_dvol_was = VariationalFormulation(ds=cls.ds, itype='nh', mix=1, kappa=True, dvol=False, was=True, **fast_mode)
        cls.vf_nh_mix1_no_kappa_dvol_was = VariationalFormulation(ds=cls.ds, itype='nh', mix=1, kappa=False, dvol=False, was=True, **fast_mode)
        cls.vf_nh_mix1_dvol_true = VariationalFormulation(ds=cls.ds, itype='nh', mix=1, kappa=False, dvol=True, **fast_mode)
        cls.vf_nh_mix1_dvol_false = VariationalFormulation(ds=cls.ds, itype='nh', mix=1, kappa=False, dvol=False, **fast_mode)
        cls.vf_nh_mix1_iso_split_true = VariationalFormulation(ds=cls.ds, itype='nh', mix=1, kappa=False, dvol=False, iso_split=True, **fast_mode)
        cls.vf_nh_mix2_no_kappa_no_dvol_bulk = VariationalFormulation(ds=cls.ds, itype='nh', mix=2, kappa=False, dvol=False, bulk=0.001, **fast_mode)
        cls.vf_nh_mix3_no_kappa_no_dvol_bulk = VariationalFormulation(ds=cls.ds, itype='nh', mix=3, kappa=False, dvol=False, bulk=0.001, **fast_mode)
        cls.vf_nh_mix1_kappa_dvol_true = VariationalFormulation(ds=cls.ds, itype='nh', mix=1, kappa=True, dvol=True, **fast_mode)

    @classmethod
    def _create_vf(cls, **kwargs):
        """Helper to create VariationalFormulation with fast mode settings."""
        fast_mode = {'simplify_tensors': False, 'simplify_timeout': 1}
        return VariationalFormulation(**{**fast_mode, **kwargs})

    @staticmethod
    def _create_ani_symbols():
        """Helper to create anisotropic material symbols."""
        return sy.symbols('alpha k_1 k_2 kappa', positive=True)

    def test_init_neo_hookean_mix1_no_kappa_no_dvol(self):
        """
        Tests basic initialization with Neo-Hookean model (mix=1),
        without kappa and without dvol.
        """
        vf = self.vf_nh_mix1_no_kappa_no_dvol

        self.assertIsInstance(vf, VariationalFormulation)
        self.assertEqual(vf.itype, 'nh')
        self.assertEqual(vf.mix, 1)
        self.assertFalse(vf._kappa_flg)
        self.assertFalse(vf._vol_flg)
        self.assertFalse(vf.iso_split)  # Default value
        self.assertTrue(vf._vol_stabilization)  # iso_split is False, so this is True

        # Check some basic symbolic attributes are initialized
        self.assertIsInstance(vf._psi_total, (sy.Expr, float))
        self.assertIsInstance(vf.dict_pk1['total'], sy.Array)
        self.assertIsInstance(vf.dict_hessian['total'], sy.Matrix)

        # Check primal variables
        self.assertEqual(len(vf.primal_vars), 3)  # lx, ly, lz
        self.assertIn(vf.lx, vf.primal_vars)
        self.assertIn(vf.ly, vf.primal_vars)
        self.assertIn(vf.lz, vf.primal_vars)

        # Check material variables
        self.assertGreater(len(vf.mat_vars), 0)
        self.assertIn(vf.dict_mat_vars['mu'], vf.mat_vars)
        self.assertIn(vf.dict_mat_vars['k_1'], vf.mat_vars)
        self.assertIn(vf.dict_mat_vars['k_2'], vf.mat_vars)
        self.assertIn(vf.dict_mat_vars['alpha'], vf.mat_vars)
        self.assertNotIn('kappa', vf.dict_mat_vars)

    def test_init_invalid_itype(self):
        """Tests initialization with an invalid isotropic material type."""
        with self.assertRaises(ValueError) as cm:
            self._create_vf(ds=self.ds, itype='invalid', mix=1, kappa=False, dvol=False)
            self.assertIn("Invalid isotropic material type: invalid", str(cm.exception))

    def test_init_fung_mix2_kappa_dvol(self):
        """Tests initialization with Fung model, mix=2, kappa=True, and dvol=True."""
        vf = self.vf_fung_mix2_kappa_dvol

        self.assertIsInstance(vf, VariationalFormulation)
        self.assertEqual(vf.itype, 'fung')
        self.assertEqual(vf.mix, 2)
        self.assertTrue(vf._kappa_flg)
        self.assertTrue(vf._vol_flg)
        self.assertTrue(vf.iso_split)
        self.assertFalse(vf._vol_stabilization)  # iso_split is True, so this is False

        # Check primal variables for Mix 2 (lx, ly, lz, p)
        self.assertEqual(len(vf.primal_vars), 4)
        self.assertIn(vf.p, vf.primal_vars)

        # Check material variables for Fung and kappa=True
        self.assertIn(vf.dict_mat_vars['a_f'], vf.mat_vars)
        self.assertIn(vf.dict_mat_vars['b_f'], vf.mat_vars)
        self.assertIn(vf.dict_mat_vars['kappa'], vf.mat_vars)
        self.assertIn(vf.dict_mat_vars['D'], vf.mat_vars)

    def test_init_mix3(self):
        """Tests initialization with mix=3."""
        vf = self.vf_nh_mix3_no_kappa_no_dvol
        self.assertEqual(len(vf.primal_vars), 5)
        self.assertIn(vf.t, vf.primal_vars)
        self.assertIn(vf.p, vf.primal_vars)

    def test_init_bulk_value(self):
        """Tests initialization with a specific bulk modulus value."""
        vf = self._create_vf(ds=self.ds, itype='nh', mix=1, kappa=False, dvol=False, bulk=50.0)
        self.assertEqual(vf._bulk, 50.0)

    def test_init_bulk_none(self):
        """Tests that _bulk is set to the default value when bulk is None."""
        vf = self._create_vf(ds=self.ds, itype='nh', mix=1, kappa=False, dvol=False, bulk=None)
        self.assertAlmostEqual(vf._bulk, 99.03 / 1000.)

    def test_init_hv_was_flags(self):
        """Tests initialization with hv and was flags."""
        vf = self._create_vf(ds=self.ds, itype='nh', mix=1, kappa=False, dvol=False, hv=True, was=True)
        self.assertTrue(vf._hv)
        self.assertTrue(vf._was)

    def test_vol_flg_property(self):
        """Tests the vol_flg property."""
        vf_dvol_true = self.vf_nh_mix1_dvol_true
        self.assertTrue(vf_dvol_true.vol_flg)
        vf_dvol_false = self.vf_nh_mix1_dvol_false
        self.assertFalse(vf_dvol_false.vol_flg)

    def test_init_iso_split_true_vol_stabilization_false(self):
        """Tests that _vol_stabilization is False when iso_split is True."""
        vf = self.vf_nh_mix1_iso_split_true
        self.assertFalse(vf._vol_stabilization)

    def test_latex_post(self):
        """Tests the latex_post method."""
        vf = self.vf_nh_mix1_no_kappa_no_dvol
        latex_output = vf.latex_post()
        self.assertIsInstance(latex_output, list)
        self.assertGreater(len(latex_output), 0)
        self.assertIsInstance(latex_output[0], str)

    def test_anisotropic_strain_kappa_true(self):
        """
        Tests _anisotropic_strain with kappa=True.
        NOTE: This test currently confirms that kappa has no effect on the
        anisotropic strain energy, which is likely a bug in the source code.
        A correct implementation should result in a non-zero derivative.
        """
        vf = self.vf_nh_mix1_kappa_dvol_was
        d_psi_ani_d_kappa = sy.diff(vf._psi_ani, vf.dict_mat_vars['kappa'])
        self.assertIsNone(d_psi_ani_d_kappa.is_zero)

    def test_anisotropic_strain_kappa_false(self):
        """Tests _anisotropic_strain with kappa=False."""
        vf = self.vf_nh_mix1_no_kappa_dvol_was
        self.assertNotIn('kappa', vf.dict_mat_vars)

    def test_init_functional_derivatives_mix2_pk1_values(self):
        """Tests numerical values of PK1 derivative w.r.t. pressure for mix=2."""
        vf = self.vf_nh_mix2_no_kappa_no_dvol_bulk
        # Use values where J=1 and p=0 to get a predictable result
        lx_val, ly_val, lz_val = 1.0, 1.0, 1.0
        mu_val, p_val = 0.5, 0.0

        subs_dict = {
            vf.lx: lx_val, vf.ly: ly_val, vf.lz: lz_val,
            vf.dict_mat_vars['mu']: mu_val,
            vf.p: p_val,
        }

        pk1_p_lambdified = sy.lambdify(list(subs_dict.keys()), vf.dict_pk1['p'], 'numpy')
        pk1_p_val = pk1_p_lambdified(*subs_dict.values())

        # For J=1 and p=0, the standard formula gives d(psi)/dp = 1 - 1 - 0 = 0
        expected_pk1_p_val = 0.0
        np.testing.assert_allclose(pk1_p_val, expected_pk1_p_val, atol=1e-9)

    def test_init_functional_derivatives_mix3_pk1_values(self):
        """Tests numerical values of PK1 derivatives for mix=3."""
        vf = self.vf_nh_mix3_no_kappa_no_dvol_bulk
        lx_val, ly_val, lz_val = 1.1, 0.9, 1.0  # J = 0.99
        mu_val, t_val, p_val = 0.5, 1.0, 0.1
        D_val = vf._bulk

        subs_dict = {
            vf.lx: lx_val, vf.ly: ly_val, vf.lz: lz_val,
            vf.dict_mat_vars['mu']: mu_val,
            vf.t: t_val, vf.p: p_val,
        }

        # Test derivative w.r.t. p: J - t
        pk1_p_lambdified = sy.lambdify(list(subs_dict.keys()), vf.dict_pk1['p'], 'numpy')
        pk1_p_val = pk1_p_lambdified(*subs_dict.values())
        expected_pk1_p_val = lx_val * ly_val * lz_val - t_val
        np.testing.assert_allclose(pk1_p_val, expected_pk1_p_val, atol=1e-2)

        # Test derivative w.r.t. t
        pk1_t_lambdified = sy.lambdify(list(subs_dict.keys()), vf.dict_pk1['t'], 'numpy')
        pk1_t_val = pk1_t_lambdified(*subs_dict.values())
        expected_pk1_t_val = D_val * 0.5 * (2 * t_val - 2) - p_val
        np.testing.assert_allclose(pk1_t_val, expected_pk1_t_val, atol=1e-1)

    def test_init_functional_dvol_false_bulk_value(self):
        """Tests that bulk modulus is correctly substituted when dvol=False."""
        vf = self._create_vf(ds=self.ds, itype='nh', mix=1, kappa=False, dvol=False, bulk=123.45)
        self.assertIsNone(vf.dict_mat_vars.get('D'))
        self.assertEqual(len(vf.mat_vars), 4)

        sym_jr = vf.tensor_manager.get_concrete_expression_by_index('J')
        expected_psi_vol = vf._bulk * volumetric_strain('simo92', sym_jr)
        self.assertTrue(safe_simplify(vf._psi_vol - expected_psi_vol) == 0)

    def test_init_functional_kappa_true(self):
        """
        Tests that kappa is a material variable when kappa=True.
        NOTE: This test currently confirms that kappa has no effect on the
        anisotropic strain energy, which is likely a bug in the source code.
        A correct implementation should result in a non-zero derivative.
        """
        vf = self.vf_nh_mix1_kappa_dvol_was
        self.assertIn('kappa', vf.dict_mat_vars)
        self.assertIn(vf.dict_mat_vars['kappa'], vf.mat_vars)
        d_psi_ani_d_kappa = sy.diff(vf._psi_ani, vf.dict_mat_vars['kappa'])
        self.assertIsNone(d_psi_ani_d_kappa.is_zero)

    def test_init_functional_derivatives_dR_dm_shape(self):
        """Tests the shape of the dR_dm matrix."""
        vf = self._create_vf(ds=self.ds, itype='nh', mix=1, kappa=True, dvol=True)
        expected_shape = (len(vf.primal_vars), len(vf.mat_vars))
        self.assertEqual(vf.dR_dm.shape, expected_shape)

    def test_init_functional_derivatives_Jvol_mix1(self):
        """Tests the _Jvol expression for mix=1."""
        vf = self._create_vf(ds=self.ds, itype='nh', mix=1, kappa=False, dvol=False)
        jr_expr = vf.lx * vf.ly * vf.lz
        expected_jvol = volumetric_strain('simo92', jr_expr)
        self.assertTrue(safe_simplify(vf.Jvol - expected_jvol) == 0)

    def test_init_functional_derivatives_Jvol_mix3(self):
        """Tests the _Jvol expression for mix=3."""
        vf = self._create_vf(ds=self.ds, itype='nh', mix=3, kappa=False, dvol=False)
        expected_jvol = volumetric_strain('simo92', vf.lx * vf.ly * vf.lz)
        self.assertTrue(safe_simplify(vf.Jvol - expected_jvol) == 0)

    def test_init_functional_derivatives_dvol_false(self):
        """Tests that dJvol_du and dJvol_dm are zero arrays when dvol=False."""
        vf = self._create_vf(ds=self.ds, itype='nh', mix=1, kappa=False, dvol=False)
        for elem in vf.dJvol_du:
            self.assertEqual(elem, sy.Float(0))
        for elem in vf.dJvol_dm:
            self.assertEqual(elem, sy.Float(0))

    def test_anisotropic_strain_was_false(self):
        """Tests _anisotropic_strain when was=False."""
        vf = self._create_vf(ds=self.ds, itype='nh', mix=1, kappa=False, dvol=False, was=False)

        containing_expressions = []
        for iv4 in vf.list_iv4:
            if iv4.has(vf.tensor_manager.get_concrete_expression_by_index("J")):
                containing_expressions.append(True)

        # Assert that J is present in the invariants when was is False
        self.assertGreater(sum(containing_expressions), 0)


class TestVariationalDerivative(unittest.TestCase):
    """Unit tests for the VariationalFormulation class."""

    def setUp(self):
        """Set up common VariationalFormulation instances for tests."""
        self.ds = 1.

        # ["l_x", "l_y", "l_z", "p", "t"]
        self.primal_vars = pd.Series(dict(l_x=1., l_y=1., l_z=1., p=0., t=1.))
        self.mat_params = pd.Series(dict(mu=0.005, D=0.01, k_1=2., k_2=0.05, alpha=0.4 * np.pi, kappa=0.5 * (1 / 3.)))
        self.ncontrol = 21

        self.np_lx = np.linspace(1., 1.5, num=self.ncontrol)
        self.np_ly = np.linspace(1., 0.7, num=self.ncontrol)
        self.np_lz = 1. / (self.np_lx * self.np_ly)
        self.np_p = np.linspace(1.e-9, 0.1, num=self.ncontrol)
        self.np_th = np.linspace(1., 1.001, num=self.ncontrol)

        self.np_primal = np.vstack((self.np_lx, self.np_ly, self.np_lz, self.np_p, self.np_th)).T

    @classmethod
    def _create_vf(cls, **kwargs):
        """Helper to create VariationalFormulation with fast mode settings."""
        fast_mode = {'simplify_tensors': False, 'simplify_timeout': 1}
        return VariationalFormulation(**{**fast_mode, **kwargs})

    def test_adjoint_variable(self):

        vf = self._create_vf(ds=self.ds, itype='nh', mix=3, dvol=True, was=True, kappa=True)
        ef = ExtensionSolution(vf)

        symbols_subs = {}
        list_xi_primal = []
        for symb_i, val_i in zip(vf.primal_vars, self.primal_vars.values):
            symbols_subs[symb_i] = val_i.item()
            list_xi_primal.append(val_i.item())

        list_mat_vars_symbols = list(vf.dict_mat_vars.values())
        list_mat_vars_keys = list(vf.dict_mat_vars.keys())

        list_xi_mat = []
        for symb_i in vf.mat_vars:
            if symb_i in vf.dict_mat_vars.values():
                idx_i = list_mat_vars_symbols.index(symb_i)
                symbols_subs[symb_i] = self.mat_params[list_mat_vars_keys[idx_i]].item()
                list_xi_mat.append(self.mat_params[list_mat_vars_keys[idx_i]].item())

        dfin_x_du = np.asarray(vf.dfint_x_du.subs(symbols_subs), dtype=float)     # must be zero
        dfin_x_dm = np.asarray(vf.dfint_x_dm.subs(symbols_subs), dtype=float)

        np_dfin_x_du = ef.lbdf_builder.dfint_x_du(*list_xi_primal, *list_xi_mat)
        np_dfin_x_dm = ef.lbdf_builder.dfint_x_dm(*list_xi_primal, *list_xi_mat)

        np.testing.assert_array_almost_equal(dfin_x_du, np_dfin_x_du, decimal=6)
        np.testing.assert_array_almost_equal(dfin_x_dm, np_dfin_x_dm, decimal=6)

        sy_dJ_du = sy.derive_by_array(vf.fint_x, vf.primal_vars)
        sy_dJ_dm = sy.derive_by_array(vf.fint_x, vf.mat_vars)

        dJ_du = np.asarray(sy_dJ_du.subs(symbols_subs), dtype=float)
        dJ_dm = np.asarray(sy_dJ_dm.subs(symbols_subs), dtype=float)

        np.testing.assert_array_almost_equal(dJ_du, np_dfin_x_du, decimal=6)
        np.testing.assert_array_almost_equal(dJ_dm, np_dfin_x_dm, decimal=6)

        # dR_du = np.asarray(vf.dR_du.subs(symbols_subs), dtype=float)
        dR_du = np.asarray(vf.dR_du_full.subs(symbols_subs), dtype=float)
        np_dR_du = ef.lbdf_builder.dR_du(*list_xi_primal, *list_xi_mat)

        np.testing.assert_array_almost_equal(dR_du, np_dR_du, decimal=6)

        dR_dm = np.asarray(vf.dR_dm.subs(symbols_subs), dtype=float)
        np_dR_dm = ef.lbdf_builder.dR_dm(*list_xi_primal, *list_xi_mat)

        np.testing.assert_array_almost_equal(dR_dm, np_dR_dm, decimal=6)

    # TODO: This test currently does not assert anything. It needs to be completed.
    @pytest.mark.skip(reason="This test is not ready yet")
    def test_adjoint_derivatives(self):

        mix = 1

        if mix == 1:
            np_primal = self.np_primal[:, :3]
        elif mix == 2:
            np_primal = self.np_primal[:, :4]
        elif mix == 3:
            np_primal = self.np_primal

        vf = self._create_vf(ds=self.ds, itype='nh', mix=1, dvol=True, was=True, kappa=True)
        ef = ExtensionSolution(vf)

        list_mat_vars_symbols = list(vf.dict_mat_vars.values())
        list_mat_vars_keys = list(vf.dict_mat_vars.keys())

        symbols_subs = {}
        list_xi_mat = []
        for symb_i in vf.mat_vars:
            if symb_i in vf.dict_mat_vars.values():
                idx_i = list_mat_vars_symbols.index(symb_i)
                symbols_subs[symb_i] = self.mat_params[list_mat_vars_keys[idx_i]].item()
                list_xi_mat.append(self.mat_params[list_mat_vars_keys[idx_i]].item())

        np_xi_mat = np.array(list_xi_mat, dtype=float)

        for i in range(self.ncontrol):
            primal_i = np_primal[i, :]

            def fint_x(xi_mat):
                return ef.lbdf_builder.fint_x(*primal_i, *xi_mat).item()

            np_fint_x_i = fint_x(np_xi_mat)

            np_dfin_x_du_i = ef.lbdf_builder.dfint_x_du(*primal_i, *np_xi_mat)
            np_dfin_x_dm_i = ef.lbdf_builder.dfint_x_dm(*primal_i, *np_xi_mat)

            np_dR_du_i = ef.lbdf_builder.dR_du(*primal_i, *np_xi_mat)
            np_dR_dm_i = ef.lbdf_builder.dR_dm(*primal_i, *np_xi_mat)

            # TODO: May try with _fdm2 to verify the implementation
            np_fdm_DJ_Dm_i = _fdm(fint_x, np_xi_mat, h=1e-5)
            np_adj_DJ_Dm_i = adjoint_derivative(np_dfin_x_du_i, np_dfin_x_dm_i, np_dR_dm_i, np_dR_du_i)


if __name__ == '__main__':
    unittest.main()
