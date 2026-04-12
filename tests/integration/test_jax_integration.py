import time
import unittest
import numpy as np
import pandas as pd
import sympy as sy
import pytest
# import jax
# import jax.numpy as jnp
from jax import config as jax_config

pytestmark = pytest.mark.integration

from dualmatfit.formulation.variational import VariationalFormulation
from dualmatfit.formulation.lambdify import LambdifyBuilder
from dualmatfit.solvers.extension import ExtensionSolution

# Ensure 64-bit for tighter numerical parity
jax_config.update("jax_enable_x64", True)

RTOL = 1e-6
ATOL = 1e-9


class TestJaxIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vf_args = dict(ds=0.1, itype='nh', mix=1, kappa=True, dvol=False, bulk=0.05)

        # mu, k_1, k_2, alpha, kappa
        mat_params = np.array([0.5, 0.6, 0.05, 5.0, 0.2], dtype=float)
        mat_params_lwr = 0.0001 * np.ones_like(mat_params)
        mat_params_upp = np.array([1., 100., 100., np.pi, (1 / 3.)], dtype=float)

        dsvars_data = {
            'values': mat_params,
            'variable': np.ones(mat_params.shape[0], dtype=bool),
            'lower': mat_params_lwr, 'upper': mat_params_upp
        }

        cls.mat_params = pd.DataFrame(dsvars_data, index=['mu', 'k_1', 'k_2', 'alpha', 'kappa'])

        cls.args_primal = np.array([1.1, 0.95, 1.0 / (1.1 * 0.95)], dtype=float)
        cls.var_form_np = VariationalFormulation(**cls.vf_args)
        cls.var_form_jax = VariationalFormulation(**cls.vf_args)
        cls.builder_np = LambdifyBuilder(cls.var_form_np, module='numpy')
        cls.builder_jax = LambdifyBuilder(cls.var_form_jax, module='jax')
        cls.inp_mat = cls.builder_jax.inp_material_keys

        cls.args_full = list(cls.args_primal) + list(cls.mat_params)
        cls.args_mat = cls.mat_params["values"].values.astype(float).copy()

        cls.builder_jax.fint(*cls.args_primal, *cls.args_mat)
        cls.builder_jax.jacobian(*cls.args_primal, *cls.args_mat)
        cls.builder_jax.hessian(*cls.args_primal, *cls.args_mat)

        for k in ['iso', 'vol', 'ani']:
            cls.builder_jax.dict_ese[k](*cls.args_primal, *cls.args_mat)
            cls.builder_jax.dict_pk1[k](*cls.args_primal, *cls.args_mat)

    def test_variational_form_outputs_numpy_vs_jax(self):
        np_res = {}; jax_res = {}
        for k in ['iso','vol','ani']:
            np_res[f"ese_{k}"] = np.array(self.builder_np.dict_ese[k](*self.args_primal, *self.args_mat), dtype=float)
            np_res[f"pk1_{k}"] = np.array(self.builder_np.dict_pk1[k](*self.args_primal, *self.args_mat), dtype=float)
            jax_res[f"ese_{k}"] = np.array(self.builder_jax.dict_ese[k](*self.args_primal, *self.args_mat), dtype=float)
            jax_res[f"pk1_{k}"] = np.array(self.builder_jax.dict_pk1[k](*self.args_primal, *self.args_mat), dtype=float)

        np_t0 = time.perf_counter()
        np_res['fint'] = np.array(self.builder_np.fint(*self.args_primal, *self.args_mat), dtype=float)
        np_res['res'] = np.array(self.builder_np.jacobian(*self.args_primal, *self.args_mat), dtype=float)
        np_res['hes'] = np.array(self.builder_np.hessian(*self.args_primal, *self.args_mat), dtype=float)
        np_t1 = time.perf_counter() - np_t0

        jax_t0 = time.perf_counter()
        jax_res['fint'] = np.array(self.builder_jax.fint(*self.args_primal, *self.args_mat), dtype=float)
        jax_res['res'] = np.array(self.builder_jax.jacobian(*self.args_primal, *self.args_mat), dtype=float)
        jax_res['hes'] = np.array(self.builder_jax.hessian(*self.args_primal, *self.args_mat), dtype=float)
        jax_t1 = time.perf_counter() - jax_t0

        print(f"\n\n Jacobian + Hessian Performance: \n  NumPy: {np_t1:.6f}s\n  JAX (JIT): {jax_t1:.6f}s")
        for key in np_res:
            self.assertTrue(np.allclose(np_res[key], jax_res[key], rtol=RTOL, atol=ATOL), f"Mismatch for {key}")

    def test_extension_solution_end_to_end_consistency(self):
        stretch_path = np.linspace(1.0, 1.2, 5)
        ext_np = ExtensionSolution(self.var_form_np, module='numpy')
        ext_jax = ExtensionSolution(self.var_form_jax, module='jax')

        sr_mat_params = self.mat_params.loc[self.inp_mat, 'values']

        res_np = ext_np.solve(sr_mat_params, stretch_path, max_iter=30, tol=1e-10, output=['stretch', 'fint'])
        res_jax = ext_jax.solve(sr_mat_params, stretch_path, max_iter=30, tol=1e-10, output=['stretch', 'fint'])

        self.assertEqual(res_np.stretch.shape, res_jax.stretch.shape)
        self.assertTrue(np.allclose(res_np.stretch, res_jax.stretch, rtol=1e-6, atol=1e-8))
        self.assertTrue(np.allclose(res_np.fint, res_jax.fint, rtol=1e-6, atol=1e-8))

    def test_extension_solution_performance(self):
        stretch_path = np.linspace(1.0, 1.3, 6)
        ext_np = ExtensionSolution(VariationalFormulation(**self.vf_args), module='numpy')
        ext_jax = ExtensionSolution(VariationalFormulation(**self.vf_args), module='jax')

        sr_mat_params = self.mat_params.loc[self.inp_mat, 'values']

        ext_jax.solve(sr_mat_params, stretch_path[:2], max_iter=20, tol=1e-9, output=['stretch'])

        t0 = time.perf_counter()
        ext_np.solve(sr_mat_params, stretch_path, max_iter=40, tol=1e-9, output=['stretch'])
        t_np = time.perf_counter() - t0

        t1 = time.perf_counter()
        ext_jax.solve(sr_mat_params, stretch_path, max_iter=40, tol=1e-9, output=['stretch'])
        t_jax = time.perf_counter() - t1

        print(f"\n\n Extension Solution Performance: \n  NumPy: {t_np:.6f}s\n  JAX (JIT): {t_jax:.6f}s")
        self.assertLessEqual(t_jax, 1.3 * t_np)

    def test_jax_compile_vs_execute_profile(self):
        a,b,c,d,e = sy.symbols('a b c d e', real=True)
        expr = sy.sin(a*b) + sy.exp(c) + d**2 + e
        symbols = [a,b,c,d,e]

        jax_scalar = LambdifyBuilder._build_jax_scalar(symbols, expr, jit=True)
        jax_grad = LambdifyBuilder._build_jax_gradient(symbols, expr, jit=True)
        jax_hess = LambdifyBuilder._build_jax_hessian(symbols, expr, jit=True)
        sample = [1.1, 0.9, 0.05, 2.0, -0.3]

        t0=time.perf_counter(); jax_scalar(*sample); t_scalar_compile=time.perf_counter()-t0
        t0=time.perf_counter(); jax_grad(*sample); t_grad_compile=time.perf_counter()-t0
        t0=time.perf_counter(); jax_hess(*sample); t_hess_compile=time.perf_counter()-t0

        def avg(fn):
            xs=[]
            for _ in range(5):
                s = time.perf_counter()
                fn(*sample)
                xs.append(time.perf_counter() - s)
            return sum(xs) / len(xs)

        t_scalar_exec=avg(jax_scalar); t_grad_exec=avg(jax_grad); t_hess_exec=avg(jax_hess)

        print(f"\n\n [JAX Compile/Exec] scalar {t_scalar_compile:.3e}/{t_scalar_exec:.3e} grad {t_grad_compile:.3e}/{t_grad_exec:.3e} hess {t_hess_compile:.3e}/{t_hess_exec:.3e}")
        self.assertGreater(t_scalar_compile, 5*t_scalar_exec)
        self.assertGreater(t_grad_compile, 5*t_grad_exec)
        self.assertGreater(t_hess_compile, 2*t_hess_exec)

        np_scalar = sy.lambdify(symbols, expr, modules='numpy')
        np_grad = sy.lambdify(symbols, sy.derive_by_array(expr, symbols), modules='numpy')
        np_hess = sy.lambdify(symbols, sy.hessian(expr, symbols), modules='numpy')

        self.assertTrue(np.allclose(np_scalar(*sample), jax_scalar(*sample), rtol=1e-9, atol=1e-12))
        self.assertTrue(np.allclose(np.array(np_grad(*sample), dtype=float), np.array(jax_grad(*sample)), rtol=1e-9, atol=1e-12))
        self.assertTrue(np.allclose(np.array(np_hess(*sample), dtype=float), np.array(jax_hess(*sample)), rtol=1e-9, atol=1e-12))


class TestNumPyConstantShortCircuit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.var_form = VariationalFormulation(ds=1.0, itype='nh', mix=1, kappa=False, dvol=False)
        cls.builder = LambdifyBuilder(cls.var_form, module='numpy')
        cls.symbols = cls.builder.inp_lbdf

    def test_constant_scalar_numpy(self):
        const_expr = sy.Integer(11)
        f_const = self.builder._lambdify(self.symbols, const_expr)
        args = [1.1, 0.9, 1 / (1.1 * 0.9)] + [0.5, 5.0, 0.2, 0.3]
        self.assertEqual(f_const(*args), 11)

    def test_variable_scalar_numpy(self):
        var_expr = self.var_form.primal_vars[0] + 2
        f_var = self.builder._lambdify(self.symbols, var_expr)
        args = [1.2, 0.9, 1 / (1.2*0.9)] + [0.5, 5.0, 0.2, 0.3]
        self.assertAlmostEqual(f_var(*args), 1.2 + 2, places=12)

    def test_compile_time_constant_faster_numpy(self):
        const_expr = sy.Integer(7)
        var_expr = self.var_form.primal_vars[0]*self.var_form.primal_vars[1] + sy.sin(self.var_form.primal_vars[2])
        n = 40

        t0 = time.perf_counter()
        for _ in range(n):
            self.builder._lambdify(self.symbols, const_expr)
        const_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n):
            self.builder._lambdify(self.symbols, var_expr)

        var_time = time.perf_counter() - t0
        print(f"\n\n NumPy short-circuit compile: const={const_time:.6f}s var={var_time:.6f}s ratio={const_time/var_time:.3f}")
        self.assertLess(const_time, var_time*0.9)


class TestLambdifyBuilderShortCircuit(unittest.TestCase):

    def test_constant_gradient_zero(self):
        a,b = sy.symbols('a b', real=True)
        g = LambdifyBuilder._build_jax_gradient([a,b], sy.Integer(7), diff_symbols=[a,b], jit=True)
        np.testing.assert_array_equal(np.array(g(0.0,10.0)), np.zeros(2))

    def test_partial_constant_gradient_zero_subset(self):
        a,b = sy.symbols('a b', real=True)
        g_subset = LambdifyBuilder._build_jax_gradient([a,b], a**2, diff_symbols=[b], jit=True)
        np.testing.assert_array_equal(np.array(g_subset(1.5,9.0)), np.zeros(1))

    def test_duplicate_diff_symbols_gradient_error(self):
        a,b = sy.symbols('a b', real=True)
        with self.assertRaises(ValueError):
            LambdifyBuilder._build_jax_gradient([a,b], a+b, diff_symbols=[a,a], jit=False)

    def test_duplicate_diff_symbols_hessian_error(self):
        a,b = sy.symbols('a b', real=True)
        with self.assertRaises(ValueError):
            LambdifyBuilder._build_jax_hessian([a,b], a*b, diff_symbols=[b,b], jit=False)


class TestJAXConstantShortCircuitProfiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.var_form = VariationalFormulation(ds=1.0, itype='nh', mix=1, kappa=False, dvol=False)
        cls.symbols = cls.var_form.primal_vars + cls.var_form.mat_vars

    # FIXME: Verify the purpose of this test
    def test_jax_constant_vs_variable_compile(self):
        const_expr = sy.Integer(13)
        var_expr = self.var_form.primal_vars[0] * self.var_form.primal_vars[1] * sy.exp(self.var_form.primal_vars[2])

        f_const = LambdifyBuilder._build_jax_scalar(self.symbols, const_expr, jit=True)
        f_var = LambdifyBuilder._build_jax_scalar(self.symbols, var_expr, jit=True)

        sample = [1.05, 0.95, 1. / (1.05 * 0.95)] + [0.6, 4.0, 0.3, 0.25]
        t0 = time.perf_counter()
        f_const(*sample)
        const_first = time.perf_counter() - t0

        t0 = time.perf_counter()
        f_var(*sample)
        var_first = time.perf_counter() - t0

        t0 = time.perf_counter()
        f_const(*sample)
        const_exec = time.perf_counter() - t0

        t0 = time.perf_counter()
        f_var(*sample)
        var_exec = time.perf_counter() - t0

        print(f"\n\n JAX constant compile vs variable: first_const={const_first:.6e}s first_var={var_first:.6e}s exec_const={const_exec:.6e}s exec_var={var_exec:.6e}s")
        self.assertLess(const_first, var_first)
        # self.assertLessEqual(const_exec, var_exec*1.5)
