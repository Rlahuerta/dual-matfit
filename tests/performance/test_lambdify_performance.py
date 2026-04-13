import unittest
import time
import numpy as np
import sympy as sy
import jax
import jax.numpy as jnp
import numba
import os
import warnings
import pytest

pytestmark = pytest.mark.slow

# Configure Numba to use workqueue threading layer to avoid TBB version warning
os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')

# Suppress Numba TBB warnings if they still appear
warnings.filterwarnings('ignore', message='.*TBB.*', category=numba.core.errors.NumbaWarning)
from jax import config as jax_config

# Enable 64-bit precision in JAX to match NumPy's float64 for tighter comparison tolerances
jax_config.update("jax_enable_x64", True)

from dualmatfit.formulation.lambdify import LambdifyBuilder


def bench(fn, args, reps=3):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


class TestLambdifyPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Define a complex symbolic expression for testing
        cls.lx, cls.ly, cls.lz = sy.symbols('l_x l_y l_z', real=True)
        cls.mu, cls.k1, cls.k2, cls.alpha, cls.kappa = sy.symbols('mu k_1 k_2 alpha kappa', positive=True)

        # A more complex expression inspired by material laws
        # This is a simplified representation for testing purposes
        F = sy.Matrix([[cls.lx, 0, 0], [0, cls.ly, 0], [0, 0, cls.lz]])
        C = F.T * F
        J = sy.det(F)
        Iv1 = sy.trace(C)
        Iv4 = (F.T * sy.Matrix([sy.cos(cls.alpha), sy.sin(cls.alpha), 0])).dot(F * sy.Matrix([sy.cos(cls.alpha), sy.sin(cls.alpha), 0]))

        # Neo-Hookean like part
        psi_iso = cls.mu / 2 * (Iv1 - 3) - cls.mu * sy.log(J)

        # Anisotropic part (simplified Fung-like or HGO-like)
        psi_ani = cls.k1 / (2 * cls.k2) * (sy.exp(cls.k2 * (Iv4 - 1)**2) - 1)

        cls.expression = psi_iso + psi_ani

        # Define input symbols for lambdify
        cls.symbols = [cls.lx, cls.ly, cls.lz, cls.mu, cls.k1, cls.k2, cls.alpha, cls.kappa]

        # Generate random numerical inputs
        cls.num_samples = 400
        # cls.num_samples = 20

        cls.inputs_lx = np.random.rand(cls.num_samples) * 0.5 + 1.0     # stretch_x between 1.0 and 1.5
        cls.inputs_ly = np.random.rand(cls.num_samples) * 0.2 + 0.8     # stretch_y between 0.8 and 1.0
        cls.inputs_lz = 1.0 / (cls.inputs_lx * cls.inputs_ly)       # maintain approximate incompressibility

        cls.inputs = {
            cls.lx: cls.inputs_lx,
            cls.ly: cls.inputs_ly,
            cls.lz: cls.inputs_lz,
            cls.mu: np.full(cls.num_samples, 0.1),
            cls.k1: np.full(cls.num_samples, 10.0),
            cls.k2: np.full(cls.num_samples, 0.5),
            cls.alpha: np.random.rand(cls.num_samples) * np.pi / 2, # angle between 0 and pi/2
            cls.kappa: np.full(cls.num_samples, 0.1)
        }

        cls.numerical_args = [cls.inputs[s] for s in cls.symbols]
        print(f"Length of numerical_args: {len(cls.numerical_args)}")
        print(f"Shape of each arg: {[arg.shape for arg in cls.numerical_args]}")

        # Lambdify functions
        cls.f_numpy = sy.lambdify(cls.symbols, cls.expression, modules='numpy')
        cls.f_numexpr = sy.lambdify(cls.symbols, cls.expression, modules='numexpr')
        
        # JAX lambdify for scalar inputs (reference vmapped path for correctness only)
        cls.f_jax_scalar = sy.lambdify(cls.symbols, cls.expression, modules='jax')
        cls.f_jax = jax.vmap(cls.f_jax_scalar)

        # JIT compile JAX function
        cls.f_jax_jit = jax.jit(cls.f_jax)

        # Numba JIT for numpy function
        cls.f_numpy_numba_jit = numba.jit(cls.f_numpy, nopython=True, parallel=True)

        # Warm-up JAX JIT and Numba JIT
        cls.f_jax_jit(*[jnp.array(arg) for arg in cls.numerical_args])
        cls.f_numpy_numba_jit(*cls.numerical_args)

        # --- Derivatives ---
        # NumPy derivatives: Lambdify symbolic derivatives, then manually vectorize
        cls.expression_diff1_sym = sy.derive_by_array(cls.expression, cls.symbols)
        cls.expression_diff2_sym = sy.hessian(cls.expression, cls.symbols)

        cls.f_diff1_numpy_scalar = sy.lambdify(cls.symbols, cls.expression_diff1_sym, modules='numpy')
        cls.f_diff2_numpy_scalar = sy.lambdify(cls.symbols, cls.expression_diff2_sym, modules='numpy')

        # Helper to vectorize NumPy scalar functions over batches
        def numpy_vectorized_wrapper(scalar_func, *batched_args):
            num_samples = batched_args[0].shape[0]
            results = []
            for i in range(num_samples):
                sample_args = [arg[i] for arg in batched_args]
                results.append(scalar_func(*sample_args))
            return np.array(results)

        cls.f_diff1_numpy = lambda *args: numpy_vectorized_wrapper(cls.f_diff1_numpy_scalar, *args)
        cls.f_diff2_numpy = lambda *args: numpy_vectorized_wrapper(cls.f_diff2_numpy_scalar, *args)

        # JAX derivatives: Use jax.grad/jax.hessian on the scalar JAX function, then vmap (reference implementation)

        # First derivative (gradient of scalar expression)
        def jax_grad_func_scalar(*args):
            grads_tuple = jax.grad(cls.f_jax_scalar, argnums=tuple(range(len(cls.symbols))))(*args)
            return jnp.stack(grads_tuple)

        cls.f_diff1_jax = jax.vmap(jax_grad_func_scalar)

        # Second derivative (Hessian of scalar expression)
        def jax_hessian_func_scalar(*args):
            hessian_blocks_tuple = jax.hessian(cls.f_jax_scalar, argnums=tuple(range(len(cls.symbols))))(*args)
            hessian_blocks_list = [list(row) for row in hessian_blocks_tuple]
            return jnp.block(hessian_blocks_list)

        cls.f_diff2_jax = jax.vmap(jax_hessian_func_scalar)

        # JIT compile JAX derivative functions
        cls.f_diff1_jax_jit = jax.jit(cls.f_diff1_jax)
        cls.f_diff2_jax_jit = jax.jit(cls.f_diff2_jax)

        # Warm-up JAX JIT derivative functions
        cls.f_diff1_jax_jit(*[jnp.array(arg) for arg in cls.numerical_args])
        cls.f_diff2_jax_jit(*[jnp.array(arg) for arg in cls.numerical_args])

        # ------------------------------------------------------------------
        # New focused JAX builders (scalar / gradient / hessian) under test
        # ------------------------------------------------------------------
        cls.jax_scalar_fn = LambdifyBuilder._build_jax_scalar(cls.symbols, cls.expression, jit=True)
        cls.jax_grad_fn = LambdifyBuilder._build_jax_gradient(cls.symbols, cls.expression, jit=True)
        cls.jax_hess_fn = LambdifyBuilder._build_jax_hessian(cls.symbols, cls.expression, jit=True)

        # Warm-up compile for scalar/grad/hess (single sample)
        _sample_args = [arg[0] for arg in cls.numerical_args]
        cls.jax_scalar_fn(*_sample_args)
        cls.jax_grad_fn(*_sample_args)
        cls.jax_hess_fn(*_sample_args)

    @staticmethod
    def _measure_time(func, args, name):
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"  {name} execution time: {elapsed_time:.6f} seconds")
        return result, elapsed_time

    def _loop_eval(self, fn, batched_args):
        out = []
        for i in range(self.num_samples):
            sample_args = [arg[i] for arg in batched_args]
            out.append(fn(*sample_args))
        return np.array(out)

    def test_jax_builder_single_sample(self):
        """Validate focused JAX builder scalar/gradient/hessian against NumPy/SymPy for one sample."""
        idx = 0
        sample_args = [arg[idx] for arg in self.numerical_args]
        # Reference computations
        ref_scalar = self.__class__.f_numpy(*sample_args)
        ref_grad = np.array(self.__class__.f_diff1_numpy_scalar(*sample_args), dtype=float)
        ref_hess = np.array(self.__class__.f_diff2_numpy_scalar(*sample_args), dtype=float)

        # JAX focused
        scalar_val = np.array(self.__class__.jax_scalar_fn(*sample_args))
        grad_val = np.array(self.__class__.jax_grad_fn(*sample_args))
        hess_val = np.array(self.__class__.jax_hess_fn(*sample_args))

        np.testing.assert_allclose(scalar_val, ref_scalar, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(grad_val, ref_grad, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(hess_val, ref_hess, rtol=1e-6, atol=1e-9)

    def test_jax_builder_loop_vs_vmap(self):
        """Compare loop-evaluated scalar/gradient/hessian vs vmapped JAX reference."""
        batched_args_np = self.numerical_args
        batched_args_jax = [jnp.array(arg) for arg in batched_args_np]

        # Scalar
        scalar_loop = self._loop_eval(self.__class__.jax_scalar_fn, batched_args_np)
        scalar_vmap = np.array(self.__class__.f_jax_jit(*batched_args_jax))
        np.testing.assert_allclose(scalar_loop, scalar_vmap, rtol=1e-6, atol=1e-9)

        # Gradient
        grad_loop = self._loop_eval(self.__class__.jax_grad_fn, batched_args_np)
        grad_vmap = np.array(self.__class__.f_diff1_jax_jit(*batched_args_jax))
        np.testing.assert_allclose(grad_loop, grad_vmap, rtol=1e-6, atol=1e-9)

        # Hessian
        hess_loop = self._loop_eval(self.__class__.jax_hess_fn, batched_args_np)
        hess_vmap = np.array(self.__class__.f_diff2_jax_jit(*batched_args_jax))
        np.testing.assert_allclose(hess_loop, hess_vmap, rtol=1e-6, atol=1e-9)

    def test_performance_comparison(self):
        print(f"\n\n--- Performance Comparison for {len(self.inputs[self.__class__.lx])} samples ---")

        # NumPy performance
        result_numpy, time_numpy = self._measure_time(self.__class__.f_numpy, self.numerical_args, "NumPy")

        # NumExpr performance
        result_numexpr, time_numexpr = self._measure_time(self.__class__.f_numexpr, self.numerical_args, "NumExpr")

        # JAX (non-JIT) performance
        result_jax, time_jax = self._measure_time(self.__class__.f_jax, [jnp.array(arg) for arg in self.numerical_args], "JAX (no JIT)")

        # JAX (JIT) performance
        result_jax_jit, time_jax_jit = self._measure_time(self.__class__.f_jax_jit, [jnp.array(arg) for arg in self.numerical_args], "JAX (JIT)")

        # Numba JIT for NumPy performance
        result_numpy_numba_jit, time_numpy_numba_jit = self._measure_time(self.__class__.f_numpy_numba_jit, self.numerical_args, "NumPy (Numba JIT)")

        # Assertions for correctness (results should be numerically close)
        np.testing.assert_allclose(result_numpy, result_numexpr, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(result_numpy, result_jax, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(result_numpy, result_jax_jit, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(result_numpy, result_numpy_numba_jit, rtol=1e-3, atol=1e-5)

        # JAX JIT expected faster than NumPy
        self.assertLessEqual(time_jax_jit, time_numpy * 1.25, "JAX (JIT) should not be significantly slower than NumPy (>25% slower)")

        print("\nPerformance Summary:")
        print(f"  NumPy: {time_numpy:.6f}s")
        print(f"  NumExpr: {time_numexpr:.6f}s")
        print(f"  JAX (no JIT): {time_jax:.6f}s")
        print(f"  JAX (JIT): {time_jax_jit:.6f}s (Fastest)")
        print(f"  NumPy (Numba JIT): {time_numpy_numba_jit:.6f}s")

    def test_first_derivative_performance(self):
        print(f"\n\n--- First Derivative Performance Comparison for {len(self.inputs[self.__class__.lx])} samples ---")

        # NumPy performance
        result_diff1_numpy, time_diff1_numpy = self._measure_time(self.__class__.f_diff1_numpy, self.numerical_args, "NumPy (Diff1)")

        list_jax_args = [jnp.array(arg) for arg in self.numerical_args]

        # JAX (non-JIT) performance
        result_diff1_jax, time_diff1_jax = self._measure_time(self.__class__.f_diff1_jax, list_jax_args, "JAX (Diff1, no JIT)")

        # JAX (JIT) performance
        result_diff1_jax_jit, time_diff1_jax_jit = self._measure_time(self.__class__.f_diff1_jax_jit, list_jax_args, "JAX (Diff1, JIT)")

        # Loop JAX gradient using focused builder
        def loop_grad():
            out = []
            for i in range(self.num_samples):
                sa = [arg[i] for arg in self.numerical_args]
                out.append(self.__class__.jax_grad_fn(*sa))
            return np.array(out)

        grad_loop_res, grad_loop_time = self._measure_time(loop_grad, (), "JAX (Diff1, loop scalar calls)")

        # Assertions for correctness
        np.testing.assert_allclose(result_diff1_numpy, result_diff1_jax, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(result_diff1_numpy, result_diff1_jax_jit, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(result_diff1_jax_jit, grad_loop_res, rtol=1e-3, atol=1e-5)

        # Assertions for performance
        self.assertLessEqual(time_diff1_jax_jit, time_diff1_numpy * 1.25, "JAX (Diff1, JIT) should not be >25% slower than NumPy (Diff1)")

        print("\nFirst Derivative Performance Summary:")
        print(f"  NumPy (Diff1): {time_diff1_numpy:.6f}s")
        print(f"  JAX (Diff1, no JIT): {time_diff1_jax:.6f}s")
        print(f"  JAX (Diff1, JIT): {time_diff1_jax_jit:.6f}s (Batched)")
        print(f"  JAX (Diff1, loop scalar calls): {grad_loop_time:.6f}s")

    def test_second_derivative_performance(self):
        print(f"\n\n--- Second Derivative Performance Comparison for {len(self.inputs[self.__class__.lx])} samples ---")

        # NumPy performance
        result_diff2_numpy, time_diff2_numpy = self._measure_time(self.__class__.f_diff2_numpy, self.numerical_args, "NumPy (Diff2)")

        # JAX (non-JIT) performance
        result_diff2_jax, time_diff2_jax = self._measure_time(self.__class__.f_diff2_jax, [jnp.array(arg) for arg in self.numerical_args], "JAX (Diff2, no JIT)")

        # JAX (JIT) performance
        result_diff2_jax_jit, time_diff2_jax_jit = self._measure_time(self.__class__.f_diff2_jax_jit, [jnp.array(arg) for arg in self.numerical_args], "JAX (Diff2, JIT)")

        # Loop JAX hessian using focused builder
        def loop_hess():
            out = []
            for i in range(self.num_samples):
                sa = [arg[i] for arg in self.numerical_args]
                out.append(self.__class__.jax_hess_fn(*sa))
            return np.array(out)

        hess_loop_res, hess_loop_time = self._measure_time(loop_hess, (), "JAX (Diff2, loop scalar calls)")

        # Assertions for correctness
        np.testing.assert_allclose(result_diff2_numpy, result_diff2_jax, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(result_diff2_numpy, result_diff2_jax_jit, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(result_diff2_jax_jit, hess_loop_res, rtol=1e-3, atol=1e-5)

        # Assertions for performance
        self.assertLessEqual(time_diff2_jax_jit, time_diff2_numpy * 1.25, "JAX (Diff2, JIT) should not be >25% slower than NumPy (Diff2)")

        print("\nSecond Derivative Performance Summary:")
        print(f"  NumPy (Diff2): {time_diff2_numpy:.6f}s")
        print(f"  JAX (Diff2, no JIT): {time_diff2_jax:.6f}s")
        print(f"  JAX (Diff2, JIT): {time_diff2_jax_jit:.6f}s (Batched)")
        print(f"  JAX (Diff2, loop scalar calls): {hess_loop_time:.6f}s")

    @pytest.mark.xfail(reason="Timing-sensitive: benchmark thresholds vary by system load")
    def test_unified_jax_builder_speedup(self):
        """Benchmark looped JAX gradient & Hessian vs NumPy loop derivatives for gradient & Hessian.

        Uses a reduced subset of samples to keep runtime reasonable in CI.
        Measures execution time (excluding JIT compile) and asserts JAX is not slower.
        """
        # Select subset for benchmarking
        max_samples = 250
        idx_stop = min(max_samples, self.num_samples)
        idx_slice = slice(0, idx_stop)
        np_subset_args = [arg[idx_slice] for arg in self.numerical_args]

        # Benchmark gradient (NumPy loop vs JAX loop)
        def np_grad_subset():
            return self.__class__.f_diff1_numpy(*np_subset_args)

        def jax_grad_subset():
            out = []
            for i in range(idx_stop):
                sa = [arg[i] for arg in np_subset_args]
                out.append(self.__class__.jax_grad_fn(*sa))
            return np.array(out)

        grad_time_np = bench(np_grad_subset, [])
        grad_time_jax = bench(jax_grad_subset, [])

        # Benchmark Hessian (reduce reps due to cost)
        def np_hess_subset():
            return self.__class__.f_diff2_numpy(*np_subset_args)

        def jax_hess_subset():
            out = []
            for i in range(idx_stop):
                sa = [arg[i] for arg in np_subset_args]
                out.append(self.__class__.jax_hess_fn(*sa))
            return np.array(out)

        hess_time_np = bench(np_hess_subset, [], reps=2)
        hess_time_jax = bench(jax_hess_subset, [], reps=2)

        print(f"\n\nLoop Builder Speed (subset {idx_stop} samples):")
        print(f"  Gradient NumPy loop: {grad_time_np:.6f}s | JAX loop: {grad_time_jax:.6f}s | Speedup x{grad_time_np/grad_time_jax:.2f}")
        print(f"  Hessian  NumPy loop: {hess_time_np:.6f}s | JAX loop: {hess_time_jax:.6f}s | Speedup x{hess_time_np/hess_time_jax:.2f}")

        # Assertions: JAX should be at least as fast (allow 15% tolerance)
        self.assertLessEqual(grad_time_jax, grad_time_np * 1.15, "JAX gradient loop should not be significantly slower.")
        self.assertLessEqual(hess_time_jax, hess_time_np * 1.15, "JAX hessian loop should not be significantly slower.")
