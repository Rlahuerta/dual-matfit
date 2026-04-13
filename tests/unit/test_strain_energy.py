# -*- coding: utf-8 -*-
import os
import gc

import numpy as np
import sympy as sy
import unittest
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from pathlib import Path
# from sympy import latex
from dualmatfit.plotting.plot_helpers import get_y_stretch
from dualmatfit.plotting.analytical_visuals import (plot_test_1d, plot_test_2d, plot_aniso_inv_test_2d,
                                                   plot_aniso_strain_test_2d, plot_aniso_stress_test_2d,
                                                   plot_aniso_stress_split_test_2d)
from dualmatfit.formulation.material_law import (right_cauchy_fun, neo_hookean, fung, volumetric_strain, anisotropic_strain,
                                    anisotropic_invariant, get_fiber_vector, heaviside)
from dualmatfit.formulation.variational import VariationalFormulation
from dualmatfit.formulation.lambdify import _lambdify
from dualmatfit.utils.latex_post import sympy2latex

current_file_path = os.path.dirname(os.path.abspath(__file__))

# Base directory for tests plots
work_path = os.path.join(current_file_path, "tests_plots")

# Create the base directory if it doesn't exist
os.makedirs(work_path, exist_ok=True)
# module = 'numpy'
module = 'jax'


for path_i in ["anisotropic", "hgo", "variational"]:
    # Construct the path for the subdirectory
    work_path_i = os.path.join(work_path, path_i)

    # Create the subdirectory if it doesn't exist
    os.makedirs(work_path_i, exist_ok=True)


class TestVolumetricStrain(unittest.TestCase):
    """
    Unit tests for the volumetric_strain function.
    """

    @classmethod
    def setUpClass(cls):
        # Define class-level variables that are common across tests
        cls.jr_sy = sy.Symbol('jr', real=True, positive=True)
        cls.jr_values = [0.8, 1.0, 1.2]
        cls.vol_types = ['simo92', 'bathe87', 'hencky', 'liu', 'doll8']

        # Expected symbolic functions for each vol_type
        cls.vol_functions_sympy = {
            'simo92': (1 / 4) * ((cls.jr_sy - 1) ** 2 + sy.ln(cls.jr_sy) ** 2),
            'bathe87': (1 / 2) * (cls.jr_sy - 1) ** 2,
            'hencky': (1 / 2) * sy.ln(cls.jr_sy) ** 2,
            'liu': cls.jr_sy * sy.ln(cls.jr_sy) - cls.jr_sy + 1,
            'doll8': ((cls.jr_sy - 1) * sy.ln(cls.jr_sy)) / 2
        }

        # Create numerical functions using lambdify
        cls.vol_functions_numeric = {
            vol_type: _lambdify(cls.jr_sy, expr, module=module)
            for vol_type, expr in cls.vol_functions_sympy.items()
        }

    def test_sympy_values(self):

        # Use class-level variables
        for jr_val in self.jr_values:
            for vol_type in self.vol_types:
                with self.subTest(jr=jr_val, vol_type=vol_type):
                    # Compute the volumetric strain using the function
                    sym_psi_vol = volumetric_strain(vol_type, self.jr_sy)

                    # Substitute jr_sy with jr_val
                    sym_psi_vol_num = sym_psi_vol.subs(self.jr_sy, jr_val).evalf()

                    # Compute expected value using the expected symbolic expression
                    expected_psi_vol = self.vol_functions_sympy[vol_type].subs(self.jr_sy, jr_val).evalf()

                    # Assert that the computed value matches the expected value
                    self.assertAlmostEqual(float(sym_psi_vol_num), float(expected_psi_vol), places=7)

    def test_numeric_values(self):
        # Use class-level variables
        for jr_val in self.jr_values:
            for vol_type in self.vol_types:
                with self.subTest(jr=jr_val, vol_type=vol_type):
                    # Compute the volumetric strain using the function
                    psi_vol_num = volumetric_strain(vol_type, jr_val)

                    # Compute expected value using the lambdified numerical function
                    expected_psi_vol = np.asarray(self.vol_functions_numeric[vol_type](jr_val)).item()

                    # Assert that the computed value matches the expected value
                    np.testing.assert_almost_equal(psi_vol_num, expected_psi_vol, decimal=6)

    def test_invalid_jr_values(self):
        # Test invalid jr values (e.g., zero or negative)
        invalid_jr_values = [0.0, -1.0]

        for jr_val in invalid_jr_values:
            for vol_type in self.vol_types:
                with self.subTest(jr=jr_val, vol_type=vol_type):
                    # Expect ValueError due to invalid input
                    with self.assertRaises(ValueError):
                        psi_vol_num = volumetric_strain(vol_type, jr_val)

    def test_type_errors(self):
        # Test invalid types for jr
        invalid_jr_types = ['string', None, [1, 2, 3]]

        for jr_val in invalid_jr_types:
            for vol_type in self.vol_types:
                with self.subTest(jr=jr_val, vol_type=vol_type):
                    with self.assertRaises(TypeError):
                        psi_vol_num = volumetric_strain(vol_type, jr_val)


class TestIsotropicStrain(unittest.TestCase):
    """
    More information about the anisotropic strain can be found in the following paper:

    Federico, Salvatore, and T. Christian Gasser. "Nonlinear elasticity of biological tissues with statistical fibre
    orientation." Journal of the Royal Society Interface 7.47 (2010): 955-966.

    Menzel, Andreas, Magnus Harrysson, and Matti Ristinmaa. "Towards an orientation-distribution-based multi-scale
    approach for remodelling biological tissues."
    Computer methods in biomechanics and biomedical engineering 11.5 (2008): 505-524.

    """

    @classmethod
    def setUpClass(cls):
        # Stretch (mix == 1) - Displacement formulation
        cls.lx, cls.ly, cls.lz = sy.symbols('l_x l_y l_z', real=True)

        cls.ar_def_grad = sy.Array([cls.lx, cls.ly, cls.lz])  # Array Format
        cls.mtx_def_grad = sy.Matrix([[cls.lx, 0, 0], [0, cls.ly, 0], [0, 0, cls.lz]])

        # Isotropic Materials Symbols
        cls.mu, cls.lbd = sy.symbols('mu lambda', positive=True)
        cls.a_f, cls.b_f = sy.symbols('a_f b_f', positive=True)
        cls.k1, cls.k2 = sy.symbols('k_1 k_2', positive=True)

        # cls.num = 501
        cls.num = 101
        cls.np_lx = (np.linspace(0.5, 2., num=cls.num)).astype(float)
        cls.np_ly = (get_y_stretch(cls.np_lx)).astype(float)
        cls.np_lz = 1. / (cls.np_lx * cls.np_ly)

        # Stress should be non-negative for stretches >= 1 and negative for stretches < 1
        cls.np_idx_xtension = cls.np_lx >= 1.0
        cls.np_idx_xcompression = cls.np_lx < 1.0

        cls.np_idx_ytension = np.where(cls.np_ly >= 1.)
        cls.np_idx_ycompression = np.where(cls.np_ly < 1.)

    def test_fung_tension_1d(self):
        """
        This material model must be fixed in the future

        Use this article to fix the material model:

        Spronck, Bart, and J. D. Humphrey. "Arterial stiffness: different metrics, different meanings."
        Journal of Biomechanical Engineering 141.9 (2019): 091004.
        """

        def_grad = self.mtx_def_grad[0:1, 0:1]

        # Regular Formulation
        # For a given a = 1, b = 1 (for simplicity), calculate the strain energy density
        # sym_psi_fg = fung(self.a_f, self.b_f, def_grad)
        # sym_psi_fg = fung(self.a_f, self.b_f, def_grad)
        # sym_psi_fg_iso = fung(self.a_f, self.b_f, def_grad, isochoric=True)

        # Compute strain energy density for normal and isochoric cases
        psi_fg = fung(1.0, 1.0, def_grad)
        pk1_fg = sy.diff(psi_fg, self.ar_def_grad[0])

        psi_fg_iso = fung(1., 1., def_grad, isochoric=True)
        pk1_fg_iso = psi_fg.diff(self.ar_def_grad[0], isochoric=True)

        lbf_psi_fg = _lambdify(self.ar_def_grad[0], psi_fg, module='numpy')
        lbf_pk1_fg = _lambdify(self.ar_def_grad[0], pk1_fg, module='numpy')

        lbf_psi_fg_iso = _lambdify(self.ar_def_grad[0], psi_fg_iso, module='numpy')
        lbf_pk1_fg_iso = _lambdify(self.ar_def_grad[0], pk1_fg_iso, module='numpy')

        np_ese = lbf_psi_fg(self.np_lx)
        np_pk1 = lbf_pk1_fg(self.np_lx)

        np_ese_iso = lbf_psi_fg_iso(self.np_lx)
        np_pk1_iso = lbf_pk1_fg_iso(self.np_lx)

        post_results = {"normal": [np_ese, np_pk1], "isochoric": [np_ese_iso, np_pk1_iso]}
        ltype = {"normal": "--", "isochoric": "-."}

        ptitle = "Fung material law"
        pfname = f"{work_path}/fung_1d.png"

        # Plot the results
        plot_test_1d(self.np_lx, ptitle, post_results, ltype, pfname)

    def test_fung_tension_2d(self):

        # plane strain configuration
        def_grad = self.mtx_def_grad.subs(self.lz, 1)

        psi_fg = fung(1., 1., def_grad)
        pk1_fg = sy.derive_by_array(psi_fg, self.ar_def_grad[:2])

        lbf_psi = _lambdify(self.ar_def_grad[:2], psi_fg, module='numpy')
        lbf_pk1 = _lambdify(self.ar_def_grad[:2], pk1_fg, module='numpy')

        np_ese = lbf_psi(self.np_lx, self.np_ly)
        np_pk1 = lbf_pk1(self.np_lx, self.np_ly)

        post_results = {"normal": [np_ese, np_pk1]}
        ltype = {"normal": "--", "isochoric": "-."}

        ################################################################################
        # Plot the results
        ptitle = "Fung material law: Plane Strain - stress plot"
        pfname = f"{work_path}/fung_pk1_plane_strain_2d.png"
        plot_test_2d(self.np_lx, self.np_ly, ptitle, post_results, ltype, pfname)

    def test_neo_hookean_tension_1d(self):
        """
        Test the Neo-Hookean material model in 1D tension.

        This test verifies that the strain energy and stress computed by the Neo-Hookean model are
        non-negative for tensile stretches and behave correctly under isochoric conditions.
        """

        def_grad = self.mtx_def_grad[0:1, 0:1]

        # Regular Formulation
        # For a given mu = 1.0 (for simplicity), calculate the strain energy density
        psi_nh = neo_hookean(self.mu.subs(self.mu, 1.), def_grad)
        pk1_nh = psi_nh.diff(self.ar_def_grad[0])

        lbf_psi_nh = _lambdify(self.ar_def_grad[0], psi_nh, module='numpy')
        lbf_pk1_nh = _lambdify(self.ar_def_grad[0], pk1_nh, module='numpy')

        np_ese = lbf_psi_nh(self.np_lx)
        np_pk1 = lbf_pk1_nh(self.np_lx)

        post_results = {"normal": [np_ese, np_pk1]}
        ltype = {"normal": "--", "isochoric": "-."}

        ptitle = "neo-hookean material law - volume strain plot"
        pfname = f"{work_path}/neo_hookean_1d.png"

        # Plot the results
        plot_test_1d(self.np_lx, ptitle, post_results, ltype, pfname)

        # Energy Strain
        # Energy should be non-negative
        self.assertTrue(np.all(np_ese >= 0), "Neo-Hookean model: Not all energy values are non-negative (normal)")

        # Stress Cases
        self.assertTrue(np.all(np_pk1[self.np_idx_xtension] >= 0), "Not all elements are non-negative")
        self.assertTrue(np.all(np_pk1[self.np_idx_xcompression] < 0), "Not all elements are negative")

    def test_neo_hookean_volume_1d(self):

        def_grad = self.mtx_def_grad[0:1, 0:1]
        # jr = sy.det(def_grad)
        # sym_vol_strain = volumetric_strain('simo92', jr)

        # Regular Formulation
        # For a given mu = 1.0 (for simplicity), calculate the strain energy density
        psi_nh = neo_hookean(self.mu.subs(self.mu, 1.), def_grad, volumetric=True)
        pk1_nh = psi_nh.diff(self.ar_def_grad[0])

        lbf_psi_nh = _lambdify(self.ar_def_grad[0], psi_nh, module='numpy')
        lbf_pk1_nh = _lambdify(self.ar_def_grad[0], pk1_nh, module='numpy')

        np_ese = lbf_psi_nh(self.np_lx)
        np_pk1 = lbf_pk1_nh(self.np_lx)

        # Isochoric Formulation
        psi_nh_iso = neo_hookean(self.mu.subs(self.mu, 1.), def_grad, isochoric=True, volumetric=True)
        # psi_nh_iso += sym_vol_strain
        pk1_nh_iso = psi_nh_iso.diff(self.ar_def_grad[0])

        lbf_psi_nh_iso = _lambdify(self.ar_def_grad[0], psi_nh_iso, module='numpy')
        lbf_pk1_nh_iso = _lambdify(self.ar_def_grad[0], pk1_nh_iso, module='numpy')

        np_ese_iso = lbf_psi_nh_iso(self.np_lx)
        np_pk1_iso = lbf_pk1_nh_iso(self.np_lx)

        post_results = {"normal": [np_ese, np_pk1], "isochoric": [np_ese_iso, np_pk1_iso]}
        ltype = {"normal": "--", "isochoric": "-."}

        ptitle = "neo-hookean material law - volume strain plot"
        pfname = f"{work_path}/neo_hookean_vol_1d.png"

        plot_test_1d(self.np_lx, ptitle, post_results, ltype, pfname)

        # Energy Strain
        self.assertTrue(np.all(np_ese >= 0), "Not all elements are non-negative")

        # Stress Cases
        self.assertTrue(np.all(np_pk1[self.np_idx_xtension] >= 0), "Not all elements are non-negative")
        self.assertTrue(np.all(np_pk1[self.np_idx_xcompression] < 0), "Not all elements are negative")

    def test_neo_hookian_tension_2d(self):
        """
        Test the Neo-Hookean material model in 2D plane strain configuration.

        This test verifies that the strain energy and principal stress computed by the Neo-Hookean model
        are consistent under plane strain conditions.
        """

        # plane strain configuration
        def_grad = self.mtx_def_grad.subs(self.lz, 1)

        # Regular Formulation
        # For a given mu = 1.0 (for simplicity), calculate the strain energy density
        psi_nh = neo_hookean(self.mu.subs(self.mu, 1.), def_grad, volumetric=True)
        pk1_nh = sy.derive_by_array(psi_nh, self.ar_def_grad[:2])

        lbf_psi_nh = _lambdify(self.ar_def_grad[:2], psi_nh, module='numpy')
        lbf_pk1_nh = _lambdify(self.ar_def_grad[:2], pk1_nh, module='numpy')

        np_ese = lbf_psi_nh(self.np_lx, self.np_ly)
        np_pk1 = lbf_pk1_nh(self.np_lx, self.np_ly)

        # Isochoric Formulation
        psi_nh_iso = neo_hookean(self.mu.subs(self.mu, 1.), def_grad, isochoric=True, volumetric=False)
        pk1_nh_iso = sy.derive_by_array(psi_nh_iso, self.ar_def_grad[:2])

        lbf_psi_nh_iso = _lambdify(self.ar_def_grad[:2], psi_nh_iso, module='numpy')
        lbf_pk1_nh_iso = _lambdify(self.ar_def_grad[:2], pk1_nh_iso, module='numpy')

        np_ese_iso = lbf_psi_nh_iso(self.np_lx, self.np_ly)
        np_pk1_iso = lbf_pk1_nh_iso(self.np_lx, self.np_ly)

        post_results = {"normal": [np_ese, np_pk1], "isochoric": [np_ese_iso, np_pk1_iso]}
        post_equations = {"normal": sy.latex(psi_nh), "isochoric": sy.latex(psi_nh_iso)}
        ltype = {"normal": "--", "isochoric": "-."}

        ################################################################################
        # Plot the results
        ptitle = "neo-hookean material law: Plane Strain - stress plot"
        pfname = f"{work_path}/neo_hookean_pk1_plane_strain_2d.png"
        plot_test_2d(self.np_lx, self.np_ly, ptitle, post_results, ltype, pfname, post_equations=post_equations)

        # Energy should be non-negative
        self.assertTrue(np.all(np_ese >= 0), "Neo-Hookean model (2D): Not all energy values are non-negative (normal)")
        self.assertTrue(np.all(np_ese_iso >= 0),
                        "Neo-Hookean model (2D): Not all energy values are non-negative (isochoric)")

        # Stress Cases
        self.assertTrue(np.all(np_pk1[0, self.np_idx_xtension] >= 0), "Not all elements are non-negative")
        self.assertTrue(np.all(np_pk1_iso[0, self.np_idx_xtension] >= 0), "Not all elements are non-negative")

        self.assertTrue(np.all(np_pk1[0, self.np_idx_xcompression] < 0), "Not all elements are negative")
        self.assertTrue(np.all(np_pk1_iso[0, self.np_idx_xcompression] < 0), "Not all elements are negative")

        self.assertTrue(np.all(np_pk1[1, self.np_idx_ytension] >= 0), "Not all elements are non-negative")
        self.assertTrue(np.all(np_pk1_iso[1, self.np_idx_ytension] >= 0), "Not all elements are non-negative")

        self.assertTrue(np.all(np_pk1[1, self.np_idx_ycompression] < 0), "Not all elements are negative")
        self.assertTrue(np.all(np_pk1_iso[1, self.np_idx_ycompression] < 0), "Not all elements are negative")

    def test_neo_hookian_volume_2d(self):

        def_grad = self.mtx_def_grad.subs(self.lz, 1)

        jr = sy.det(def_grad)
        sym_vol_strain = volumetric_strain('simo92', jr)

        # Regular Formulation
        # For a given mu = 1.0 (for simplicity), calculate the strain energy density
        psi_nh = neo_hookean(self.mu.subs(self.mu, 1.), def_grad, volumetric=True, isochoric=False)
        psi_nh += sym_vol_strain
        pk1_nh = sy.derive_by_array(psi_nh, self.ar_def_grad[:2])

        lbf_psi_nh = _lambdify(self.ar_def_grad[:2], psi_nh, module=module)
        lbf_pk1_nh = _lambdify(self.ar_def_grad[:2], pk1_nh, module=module)

        np_ese = lbf_psi_nh(self.np_lx, self.np_ly)
        np_pk1 = lbf_pk1_nh(self.np_lx, self.np_ly)

        # Isochoric Formulation
        psi_nh_iso = neo_hookean(self.mu.subs(self.mu, 1.), def_grad, volumetric=False, isochoric=True)
        psi_nh_iso += sym_vol_strain
        pk1_nh_iso = sy.derive_by_array(psi_nh_iso, self.ar_def_grad[:2])

        lbf_psi_nh_iso = _lambdify(self.ar_def_grad[:2], psi_nh_iso, module=module)
        lbf_pk1_nh_iso = _lambdify(self.ar_def_grad[:2], pk1_nh_iso, module=module)

        np_ese_iso = lbf_psi_nh_iso(self.np_lx, self.np_ly)
        np_pk1_iso = lbf_pk1_nh_iso(self.np_lx, self.np_ly)

        post_results = {"normal": [np_ese, np_pk1], "isochoric": [np_ese_iso, np_pk1_iso]}
        post_equations = {"normal": sy.latex(psi_nh), "isochoric": sy.latex(psi_nh_iso)}
        ltype = {"normal": "--", "isochoric": "-."}

        ################################################################################
        # Plot the results
        ptitle = "neo-hookean material law - stress plot"
        pfname = f"{work_path}/neo_hookean_pk1_vol_plane_strain_2d.png"
        plot_test_2d(self.np_lx, self.np_ly, ptitle, post_results, ltype, pfname, post_equations=post_equations)

        # Energy Strain
        self.assertTrue(np.all(np_ese >= 0),
                        "Neo-Hookean volumetric model (2D): Not all energy values are non-negative")
        self.assertTrue(np.all(np_ese_iso >= 0),
                        "Neo-Hookean volumetric model (2D): Not all energy values are non-negative (isochoric)")

        # Stress Cases
        self.assertTrue(np.all(np_pk1[0, self.np_idx_xtension] >= 0), "Not all elements are non-negative")
        self.assertTrue(np.all(np_pk1[0, self.np_idx_xcompression] < 0), "Not all elements are negative")

        self.assertTrue(np.all(np_pk1[1, self.np_idx_ytension] >= 0), "Not all elements are non-negative")
        self.assertTrue(np.all(np_pk1[1, self.np_idx_ycompression] < 0), "Not all elements are negative")

    def test_volumetric_strain(self):

        jr = sy.symbols('j^r', positive=True)

        np_jac = np.linspace(0.5, 2.5, 101)

        np_idx_tension = np.where(np_jac >= 1.)
        np_idx_compression = np.where(np_jac < 1.)

        list_vtypes = ['simo92', 'bathe87', 'hencky', 'liu', 'doll8']

        dict_vol_ener_strain = dict()

        # Plot the results
        fig, ax = plt.subplots(3, figsize=(16, 16), sharex=True, dpi=700)
        fig.suptitle(r"Volumetric Strain Energy $U(J^r)$", fontsize=14)

        for vtype_i in list_vtypes:
            # Energy Strain
            sym_vol_strain_i = volumetric_strain(vtype_i, jr)
            lbf_vol_strain_i = _lambdify(jr, sym_vol_strain_i, module=module)
            np_vol_ese_i = lbf_vol_strain_i(np_jac)

            self.assertTrue(np.all(np_vol_ese_i >= 0), "Not all elements are non-negative")

            ax[0].plot(np_jac, np_vol_ese_i, label=f'{vtype_i} formulation')

            # stress-free condition (volumetric stress)
            sym_dvol_strain_i = sym_vol_strain_i.diff(jr)
            lbf_dvol_strain_i = _lambdify(jr, sym_dvol_strain_i, modules=module)
            np_dvol_ese_i = lbf_dvol_strain_i(np_jac)

            # Stress Cases
            self.assertTrue(np.all(np_dvol_ese_i[np_idx_tension] >= 0), "Not all elements are non-negative")
            self.assertTrue(np.all(np_dvol_ese_i[np_idx_compression] < 0), "Not all elements are negative")

            ax[1].plot(np_jac, np_dvol_ese_i, label=f'{vtype_i} formulation')

            # requirement of polyconvexity of the strain energy function the volumetric,
            # part has to satisfy the convexity condition: d²U/dJ² >= 0
            sym_d2vol_strain_i = sym_dvol_strain_i.diff(jr)
            lbf_d2vol_strain_i = _lambdify(jr, sym_d2vol_strain_i, module)
            np_d2vol_ese_i = np.asarray([lbf_d2vol_strain_i(jac_i) for jac_i in np_jac])

            if isinstance(np_d2vol_ese_i, float):
                np_d2vol_ese_i = np.ones_like(np_jac)

            ax[2].plot(np_jac, np_d2vol_ese_i, label=f'{vtype_i} formulation')

            dict_vol_ener_strain[vtype_i] = dict(sympy=sym_vol_strain_i,
                                                 lambdfy=lbf_vol_strain_i,
                                                 ese=np_vol_ese_i,
                                                 )

        ax[0].axhline(y=0., color='r', linestyle=":")
        ax[1].axhline(y=0., color='r', linestyle=":")
        ax[2].axhline(y=1., color='r', linestyle=":")

        for ax_i in ax:
            ax_i.grid(which='minor', alpha=0.2)
            ax_i.grid(which='major', alpha=0.5)

            ax_i.axvline(x=1., color='k', linestyle=":")
            ax_i.legend()

        ax[0].set_ylabel(r'Volumetric Energy Strain Density, $U (J^r)$', fontsize=10)
        ax[1].set_ylabel(r'Engineering Volumetric Stress, ${\sigma}_{v} (J^r)$', fontsize=10)
        ax[2].set_ylabel(r'Volumetric Stiffness, $d^{2}U (J^r)$', fontsize=10)
        ax[-1].set_xlabel(r'$J^r$', fontsize=10)

        plt.tight_layout()
        fig.savefig(f"{work_path}/volumetric_strain.png")
        plt.close(fig)

    def test_anisotropic_invariant_2d(self):

        def_grad = self.mtx_def_grad[0:2, 0:2]

        np_alpha = np.pi * np.linspace(0.001, 0.499, num=8)
        np_alpha_deg = np.round(np.rad2deg(np_alpha), decimals=2)

        np_kappa = np.array([0., 0.1, 0.2, 0.3], dtype=float)
        list_was = [False, True]

        list_args = []

        for kappa_k in np_kappa:
            for was_n in list_was:
                list_args.append([kappa_k, was_n])

        ################################################################################
        # Looping of Anisotropic Invariant Tests

        for k, args_k in enumerate(list_args):
            ptitle_k = r"Anisotropic Unimodular Invariants, $\kappa$: " + f"{args_k[0]}, WAS: {args_k[1]}"
            pfname_k = f"{work_path}/anisotropic/invariants_2d_test_{k}.png"

            aniso_res_k = dict()
            for i, alpha_i in enumerate(np_alpha):
                list_fr_i = get_fiber_vector(alpha_i, dim=2)

                list_lbf_iv4_i = []
                list_val_iv4_i = []
                list_val_hv_i = []

                for fr_ij in list_fr_i:
                    args_ki = [def_grad, fr_ij] + args_k
                    sym_rc_ij, sym_iv4_ij = anisotropic_invariant(*args_ki)

                    # FIXME: heaviside function is not supported in jax
                    lbf_iv4_ij = _lambdify(self.ar_def_grad[:2], sym_iv4_ij, module="numpy")

                    np_iv4_ij = lbf_iv4_ij(self.np_lx, self.np_ly)
                    np_hv_ij = 0.8 * heaviside(np_iv4_ij) + i

                    list_lbf_iv4_i.append(lbf_iv4_ij)
                    list_val_iv4_i.append(np_iv4_ij)
                    list_val_hv_i.append(np_hv_ij)

                aniso_res_k[i] = dict(symbolic=list_lbf_iv4_i, values=list_val_iv4_i, hv=list_val_hv_i)

            plot_aniso_inv_test_2d(self.np_lx, np_alpha_deg, ptitle_k, aniso_res_k, pfname_k)

    @pytest.mark.xfail(reason="PIL _idat.fileno error with high-DPI PNG saves in some environments")
    def test_anisotropic_strain_2d(self):

        k1, k2 = 1., 1.
        def_grad = self.mtx_def_grad

        np_alpha = np.pi * np.linspace(0.001, 0.499, num=8, dtype=float)
        np_alpha_deg = np.round(np.rad2deg(np_alpha), decimals=2)

        np_kappa = np.array([0., 0.1, 0.2, 0.3], dtype=float)
        list_hv = [False, True]
        list_was = [False, True]

        # list_args = [[0.]]
        # list_kwargs = [{'hv': True, 'was': True}]
        list_args = []
        list_kwargs = []

        main_args = [k1, k2]

        for kappa_k in np_kappa:
            for hv_j in list_hv:
                for was_n in list_was:
                    list_args.append([kappa_k])
                    list_kwargs.append(dict(hv=hv_j, was=was_n))

        ################################################################################
        # Looping of Anisotropic Tests
        np_stretch = np.array([self.np_lx, self.np_ly, self.np_lz])

        for k, (args_k, kwargs_k) in enumerate(zip(list_args, list_kwargs)):
            ptitle_k = r"HGO anisotropic function, $\kappa$: " + \
                       f"{args_k[-1]}, WAS: {kwargs_k['was']}, HV: {kwargs_k['hv']}"

            aniso_res_k = dict()
            for i, alpha_i in enumerate(np_alpha):
                list_fr_i = get_fiber_vector(alpha_i, dim=3)

                list_iv4_i = []
                list_psi_ani_i = []
                list_heaviside_i = []

                for fr_ij in list_fr_i:
                    _, sym_iv4_ij = anisotropic_invariant(def_grad, fr_ij, args_k[-1], kwargs_k['was'])

                    args_ki = main_args + [sym_iv4_ij]
                    sym_psi_ani_ij = anisotropic_strain(*args_ki, hv=kwargs_k['hv'])

                    list_psi_ani_i.append(sym_psi_ani_ij)

                    lbf_iv4_ij = _lambdify(self.ar_def_grad, sym_iv4_ij, module=module)

                    sym_heaviside_ij = heaviside(sym_iv4_ij)
                    lbf_heaviside_ij = _lambdify(self.ar_def_grad, sym_heaviside_ij, module=module)

                    list_iv4_i.append(lbf_iv4_ij(self.np_lx, self.np_ly, self.np_lz))
                    np_heaviside_ij = lbf_heaviside_ij(self.np_lx, self.np_ly, self.np_lz)
                    list_heaviside_i.append(0.8 * np_heaviside_ij + i)

                ################################################################################
                # Energy Strain Density and Stress Calculation
                sym_psi_ani_i = sum(list_psi_ani_i)
                sym_pk1_ani_i = sy.derive_by_array(sym_psi_ani_i, self.ar_def_grad)

                lbf_psi_ani_i = _lambdify(self.ar_def_grad, sym_psi_ani_i, module=module)
                lbf_pk1_ani_i = _lambdify(self.ar_def_grad, sym_pk1_ani_i, module=module)

                np_psi_ani_i = lbf_psi_ani_i(self.np_lx, self.np_ly, self.np_lz)
                np_pk1_ani_i = np.array([lbf_pk1_ani_i(*np_stretch[:, n]) for n in range(self.num)]).T

                aniso_res_k[i] = dict(energy=np_psi_ani_i,
                                      stress=np_pk1_ani_i,
                                      iv4=list_iv4_i,
                                      heaviside=list_heaviside_i,
                                      )

            ################################################################################
            # Plot the results
            pfname_k1 = f"{work_path}/anisotropic/hgo_2d_stress_test_{k}.png"
            plot_aniso_stress_test_2d(self.np_lx, self.np_ly, np_alpha_deg, ptitle_k, aniso_res_k, pfname_k1)

            pfname_k2 = f"{work_path}/anisotropic/hgo_2d_strain_test_{k}.png"
            plot_aniso_strain_test_2d(self.np_lx, np_alpha_deg, ptitle_k, aniso_res_k, pfname_k2)

    @pytest.mark.xfail(reason="PIL _idat.fileno error with high-DPI PNG saves in some environments")
    def test_anisotropic_neo_hookean_function(self):
        """
        Test the combined anisotropic Neo-Hookean material function

        This test verifies that the combined isotropic and anisotropic strain energy and stress
        are computed correctly for various fiber orientations, anisotropy parameters, and flags.
        """

        def_grad = self.mtx_def_grad
        jr = sy.det(def_grad)

        ################################################################################
        # Main Parameters
        k1, k2 = 1., 1.

        # stretch in y
        np_ly = self.np_ly ** 1.2
        np_lz = 1. / (self.np_lx * np_ly)
        # np_jr = self.np_lx * np_ly * np_lz

        np_stretch = np.array([self.np_lx, np_ly, np_lz])

        # Compute isotropic strain energies
        sym_psi_nh = neo_hookean(self.mu.subs(self.mu, 1.), def_grad)
        sym_psi_vol = 100. * volumetric_strain('simo92', jr)

        sym_right_cauchy = right_cauchy_fun(def_grad)
        sym_iv1 = sym_right_cauchy.trace()
        lbf_iv1 = _lambdify(self.ar_def_grad, sym_iv1, module=module)

        ################################################################################
        np_alpha = np.pi * np.linspace(0.001, 0.499, num=8)
        np_alpha_deg = np.round(np.rad2deg(np_alpha), decimals=2)

        np_kappa = np.array([0., 0.1, 0.2, 0.3], dtype=float)
        list_hv = [False, True]
        list_was = [False, True]

        list_args = []
        list_kwargs = []

        main_args = [k1, k2]

        for kappa_k in np_kappa:
            for hv_j in list_hv:
                for was_n in list_was:
                    list_args.append([kappa_k])
                    list_kwargs.append(dict(hv=hv_j, was=was_n))

        ################################################################################
        # Isotropic Tests
        psi_sym_iso = dict(iso=sym_psi_nh, vol=sym_psi_vol)
        res_iso = dict()

        for key_j, val_j in psi_sym_iso.items():
            ese_j = f"energy_{key_j}"
            pk1_j = f"stress_{key_j}"

            lbf_psi_ij = _lambdify(self.ar_def_grad, val_j, module=module)

            sym_pk1_ij = sy.derive_by_array(val_j, self.ar_def_grad)
            lbf_pk1_ij = _lambdify(self.ar_def_grad, sym_pk1_ij, module=module)

            res_iso[ese_j] = lbf_psi_ij(self.np_lx, np_ly, np_lz)
            res_iso[pk1_j] = lbf_pk1_ij(self.np_lx, np_ly, np_lz)

        ################################################################################
        # Looping of Anisotropic Tests

        for k, (args_k, kwargs_k) in enumerate(zip(list_args, list_kwargs)):
            ptitle_k = r"neo-Hookean HGO anisotropic function, $\kappa$: " + \
                       f"{args_k[-1]}, WAS: {kwargs_k['was']}, HV: {kwargs_k['hv']}"

            aniso_res_k = dict()
            for i, alpha_i in enumerate(np_alpha):
                list_fr_i = get_fiber_vector(alpha_i, dim=3)

                list_iv4_i = []
                list_psi_ani_i = []
                list_heaviside_i = []

                for fr_ij in list_fr_i:
                    _, sym_iv4_ij = anisotropic_invariant(def_grad, fr_ij, args_k[-1], kwargs_k['was'])

                    args_ki = main_args + [sym_iv4_ij]
                    sym_psi_ani_ij = anisotropic_strain(*args_ki, hv=kwargs_k['hv'])

                    list_psi_ani_i.append(sym_psi_ani_ij)

                    lbf_iv4_ij = _lambdify(self.ar_def_grad, sym_iv4_ij, module=module)

                    sym_heaviside_ij = heaviside(sym_iv4_ij, 50.)
                    lbf_heaviside_ij = _lambdify(self.ar_def_grad, sym_heaviside_ij, module=module)

                    list_iv4_i.append(lbf_iv4_ij(self.np_lx, np_ly, np_lz))
                    list_heaviside_i.append(lbf_heaviside_ij(self.np_lx, np_ly, np_lz))

                ########################################
                # Energy Strain Density and Stress Calculation
                sym_psi_ani_i = sum(list_psi_ani_i)
                sym_psi_hgo_i = sym_psi_nh + sym_psi_vol + sym_psi_ani_i
                sym_pk1_hgo_i = sy.derive_by_array(sym_psi_hgo_i, self.ar_def_grad)

                psi_info_i = dict(ani=sym_psi_ani_i)

                lbf_psi_hgo_i = _lambdify(self.ar_def_grad, sym_psi_hgo_i, module=module)
                lbf_pk1_hgo_i = _lambdify(self.ar_def_grad, sym_pk1_hgo_i, module=module)

                aniso_res_k[i] = dict(energy=lbf_psi_hgo_i(self.np_lx, np_ly, np_lz),
                                      stress=lbf_pk1_hgo_i(self.np_lx, np_ly, np_lz),
                                      iv4=list_iv4_i, iv1=lbf_iv1(self.np_lx, np_ly, np_lz),
                                      heaviside=list_heaviside_i)

                for key_j, val_j in psi_info_i.items():
                    ese_j = f"energy_{key_j}"
                    pk1_j = f"stress_{key_j}"

                    lbf_psi_ij = _lambdify(self.ar_def_grad, val_j, module=module)
                    np_psi_ij = lbf_psi_ij(self.np_lx, np_ly, np_lz)

                    sym_pk1_ij = sy.derive_by_array(val_j, self.ar_def_grad)
                    lbf_pk1_ij = _lambdify(self.ar_def_grad, sym_pk1_ij, module=module)
                    np_pk1_ij = np.array([lbf_pk1_ij(*np_stretch[:, n]) for n in range(self.num)]).T

                    aniso_res_k[i][ese_j] = np_psi_ij
                    aniso_res_k[i][pk1_j] = np_pk1_ij
                    aniso_res_k[i].update(res_iso)

            pfname_k1 = f"{work_path}/hgo/2d_strain_test_{k}.png"
            plot_aniso_strain_test_2d(self.np_lx, np_alpha_deg, ptitle_k, aniso_res_k, pfname_k1)

            pfname_k2 = f"{work_path}/hgo/2d_stress_test_{k}.png"
            plot_aniso_stress_test_2d(self.np_lx, np_ly, np_alpha_deg, ptitle_k, aniso_res_k, pfname_k2)

            pfname_k3 = f"{work_path}/hgo/2d_stress_split_test_{k}.png"
            plot_aniso_stress_split_test_2d(self.np_lx, np_alpha_deg, ptitle_k, aniso_res_k, pfname_k3)


class TestVariationalFormulation(unittest.TestCase):

    def setUp(self):
        # Stretch (mix == 1) - Displacement formulation
        self.mix = [1, 2, 3]
        self.itype = 'nh'
        self.bulk_vals = 1000. * np.linspace(0.01, 1., num=10)

    def test_variational(self):

        lc_wpath = f"{work_path}/variational"
        list_latex_eqs = []

        # varfom = VariationalFormulation(self.itype, 3, False, False, 100., was=False, hv=True)
        # sympy2latex(varfom.latex_post(), f"varform_nh_m1_hv.tex", wpath=lc_wpath)

        for mi in [1, 2, 3]:
            for kappa_j in [False, True]:
                for dvol_k in [False, True]:
                    for was_l in [False, True]:
                        for hv_m in [False, True]:
                            varfom_ijk = VariationalFormulation(ds=1.,
                                                                itype=self.itype,
                                                                mix=mi,
                                                                kappa=kappa_j,
                                                                dvol=dvol_k,
                                                                bulk=100.,
                                                                was=was_l,
                                                                hv=hv_m)

                            latex_eqs_ijk = varfom_ijk.latex_post()
                            list_latex_eqs.append(latex_eqs_ijk)

                            fname_ijk = f"varform_nh_m{mi}"

                            if kappa_j:
                                fname_ijk += "_disp"

                            if dvol_k:
                                fname_ijk += "_vol"

                            if was_l:
                                fname_ijk += "_was"

                            if hv_m:
                                fname_ijk += "_hv"

                            sympy2latex(latex_eqs_ijk, f"{fname_ijk}.tex", wpath=lc_wpath)
                            gc.collect()


if __name__ == '__main__':
    unittest.main()
