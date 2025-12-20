# -*- coding: utf-8 -*-
"""
Lambdify builder for converting symbolic expressions to numerical functions.

This module provides the LambdifyBuilder class for converting SymPy 
expressions to efficient numerical functions using JAX for automatic
differentiation.
"""
import warnings
import os
import jax
import numpy as np
import jax.numpy as jnp
import sympy as sy

from typing import List, Dict, Callable, Optional, Sequence
from dualmatfit.variational_form import VariationalFormulation

from dualmatfit.logging_config import get_logger
logger = get_logger('codegen')

__all__ = [
    'LambdifyBuilder',
]

_USE_JAX_JIT = os.environ.get('HGOMATFIT_JAX_JIT', '1') == '1'



def _get_block_array(block_matrix: List) -> List[int]:
    """Computes the indices for block matrices."""
    block_array = []
    m = 0
    for row_i in block_matrix:
        np_idx = np.arange(row_i.shape[0], dtype=int) + m
        m += np_idx.shape[0]
        block_array.append(np_idx)
    return block_array


def _is_constant_expr(expr, inputs) -> bool:
    try:
        expr_free = getattr(expr, 'free_symbols', set())
    except (AttributeError, TypeError) as e:
        # AttributeError: Object doesn't have free_symbols
        # TypeError: Unexpected type for getattr
        logger.debug(f"Warning: Failed to access free_symbols of expression: {e}")
        return False

    if isinstance(expr_free, Sequence):
        return len(expr_free.intersection(set(inputs))) == 0
    else:
        return len(expr_free) == 0


def _wrap_constant(expr, jax_mode: bool = False) -> Callable:
    """
    Convert a constant symbolic expression into a callable that returns its value.
    """

    try:
        if hasattr(expr, 'shape') and expr.shape not in ((), None):
            arr = np.array(expr, dtype=float)
            if jax_mode:
                const_val = jnp.array(arr)
            else:
                const_val = arr
        else:
            val = float(expr) if not isinstance(expr, (int, float)) else expr
            if jax_mode:
                const_val = jnp.array(val)
            else:
                const_val = val
    except (TypeError, ValueError) as e:
        # TypeError: Cannot convert expression to float/array
        # ValueError: Invalid value for conversion
        logger.debug(f"Warning: Failed to convert constant expression to array: {e}")
        const_val = expr

    return lambda *args, _v=const_val: _v


def _lambdify(inputs, expr, module: str = 'numpy', **kwargs) -> Callable:
    """Helper to lambdify an expression with the correct backend with constant short-circuit."""
    if _is_constant_expr(expr, inputs):
        fn = _wrap_constant(expr, jax_mode=(module == 'jax'))
        if module == 'jax' and _USE_JAX_JIT:
            fn = jax.jit(fn)
        return fn

    if module == 'jax':
        base_fn = sy.lambdify(inputs, expr, modules='jax')
        return jax.jit(base_fn) if _USE_JAX_JIT else base_fn

    return sy.lambdify(inputs, expr, modules=module)


class LambdifyBuilder:
    """
    Handles the conversion of symbolic expressions into numerical functions.

    This class takes a variational formulation, symbolic variables, and other
    parameters to build lambdified functions for residuals, Jacobians, and
    Hessians, which can be used with different numerical backends like NumPy or JAX.
    """

    def __init__(self,
                 var_form: VariationalFormulation,
                 module: str = 'numpy',
                 ):
        """
        Initializes the LambdifyBuilder.

        Args:
            var_form: An instance of the variational formulation.
            module: The backend for lambdification ('numpy' or 'jax').
        """
        self.var_form = var_form
        self.module = module

        # Main Symbolic expressions
        self.inp_lbdf = self.var_form.primal_vars + self.var_form.mat_vars

        # Cost function
        self.sym_inp_lsq = self.inp_lbdf + [self.var_form.fx]
        self.lsq_fun = None
        self.lsq_fun_diff = None

        # Volume Regularization
        self.Jvol = None

        # Anisotropic Functions
        self.dict_pk1_diff_x = {}
        self.aniso_inv_diff = {}
        self.aniso_inv = []

        self.block_array = _get_block_array(self.var_form.block_jacobian)
        self._build_block_vars()

        # Internal storage for compiled block jacobians
        # self._block_jacobian = None

        # Lambdify functions
        self._init_lambdify_functions()

        # Lambdify adjoint variables
        self._build_adjoint_lambdas()

        # Get input keys for material parameters (Design Parameters)
        self._get_input_keys()

    def _get_input_keys(self):
        """Generates string keys for input variables."""
        self.inp_material_keys = []

        for key_i, var_i in self.var_form.dict_mat_vars.items():
            for sym_k in self.var_form.mat_vars:
                if sym_k == var_i:
                    self.inp_material_keys.append(key_i)

    @property
    def block_jacobian(self):
        """Read-only access to compiled block Jacobian callables.
        Lazily initialized when _init_lambdify_functions runs."""
        return self._block_jacobian

    def _lambdify(self, inputs, expr, **kwargs):
        """Helper to lambdify an expression with the correct backend with constant short-circuit."""
        return _lambdify(inputs, expr, module=self.module)

    @staticmethod
    def _jax_lambdify_scalar(inputs, expr):
        return sy.lambdify(inputs, expr, modules='jax')

    def _jax_jit_lambdify_scalar(self, inputs, expr):
        base_fn = jax.vmap(self._jax_lambdify_scalar(inputs, expr))
        return jax.jit(base_fn) if _USE_JAX_JIT else base_fn

    def _jax_lambdify_jacobian(self, jax_lambdify_scalar, *args):
        grads_tuple = jax.grad(jax_lambdify_scalar, argnums=tuple(range(len(self.inp_lbdf))))(*args)
        return jnp.stack(grads_tuple)

    def _build_adjoint_lambdas(self):
        """Builds lambdified functions for the adjoint method."""

        # Residual forces in x-axis
        self.fint_x = _lambdify(self.inp_lbdf, self.var_form.fint_x)

        # Material derivatives (Adjoint Method)
        self.dfint_x_du = _lambdify(self.inp_lbdf, self.var_form.dfint_x_du)
        self.dfint_x_dm = _lambdify(self.inp_lbdf, self.var_form.dfint_x_dm)

        # Lambdify the Derivatives of residuals with respect to material parameters
        # self.dR_du = _lambdify(self.inp_lbdf, self.var_form.dR_du)
        self.dR_du = _lambdify(self.inp_lbdf, self.var_form.dR_du_full)

        self.dR_dm = _lambdify(self.inp_lbdf, self.var_form.dR_dm)

        # Lambdify the Derivatives of volumetric strain energy with respect to material parameters
        self.dJvol_du = _lambdify(self.inp_lbdf, self.var_form.dJvol_du)
        self.dJvol_dm = _lambdify(self.inp_lbdf, self.var_form.dJvol_dm)

        self.dIva_du = []
        self.dIva_dm = []

        for k, iv_k in enumerate(self.var_form.list_iv4):
            self.dIva_du.append(_lambdify(self.inp_lbdf, self.var_form.aniso_inv_derivatives['d_iva_du'][k, :]))
            self.dIva_dm.append(_lambdify(self.inp_lbdf, self.var_form.aniso_inv_derivatives['d_iva_dm'][k, :]))

    def build_cost_function_lambdas(self):
        """Builds lambdified functions for the cost function."""

        self.lsq_fun = _lambdify(self.sym_inp_lsq, self.var_form.lsq_fun)
        self.lsq_fun_diff = _lambdify(self.sym_inp_lsq, self.var_form.lsq_fun_diff)

        # Volumetric Strain Energy Function
        if hasattr(self.var_form, 'Jvol'):
            self.Jvol = _lambdify(self.inp_lbdf, self.var_form.Jvol)
        else:
            warnings.warn("Attribute Jvol not found in var_form. Volumetric calculations might fail.")
            self.Jvol = lambda *args: 0  # Placeholder

    def build_optimization_lambdas(self):
        """Builds lambdified functions for optimization."""

        # PK1 Stress Derivative in [x] direction (Lambdify the derivative function)
        for key_k in ['iso', 'vol', 'ani']:
            self.dict_pk1_diff_x[key_k] = _lambdify(self.inp_lbdf, self.var_form.dict_pk1_diff_x[key_k])

        self.aniso_inv = []
        for k, iv_k in enumerate(self.var_form.list_iv4):
            self.aniso_inv.append(_lambdify(self.inp_lbdf, iv_k))

    def _init_lambdify_functions(self):
        """Create lambdified functions for numerical computation."""
        self.dict_ese = {}
        self.dict_pk1 = {}

        for key_k in ['iso', 'vol', 'ani', 'total']:
            self.dict_ese[key_k] = _lambdify(self.inp_lbdf, self.var_form.dict_psi_sum[key_k])
            self.dict_pk1[key_k] = _lambdify(self.inp_lbdf, self.var_form.dict_pk1[key_k])

        self.dict_pk1['full'] = _lambdify(self.inp_lbdf, self.var_form.dict_pk1['full'])

        self.fint = _lambdify(self.inp_lbdf, self.var_form.residuum)
        self.jacobian = _lambdify(self.inp_lbdf, self.var_form.jacobian)
        self.hessian = _lambdify(self.inp_lbdf, self.var_form.hessian)

        self.fint_mat_diff = _lambdify(self.inp_lbdf, self.var_form.fint_mat_diff)
        self.energy_mat_diff = _lambdify(self.inp_lbdf, self.var_form.energy_mat_diff)

    def _build_block_vars(self):
        """Builds the block Hessian matrix and corresponding lambdified functions."""

        # Compile block jacobian
        self._block_jacobian = [_lambdify(self.inp_lbdf, sym_jac_i) for sym_jac_i in self.var_form.block_jacobian]

        # FIXME: Split this method with variational formulation
        block_sym_hes, block_lbdf_hes, block_shapes = [], [], []

        for i, (sym_hess_i, hess_type_i) in enumerate(zip(self.var_form.block_hessian, self.var_form.lbdf_block_hessian)):
            block_sym_ij, block_lbdf_ij = [], []

            for j, (sym_hess_j, hess_type_j) in enumerate(zip(sym_hess_i, hess_type_i)):
                if hess_type_j != 'lambdify':
                    block_lbdf_ij.append(np.asarray(sym_hess_j, dtype=float))
                else:
                    block_lbdf_ij.append(_lambdify(self.inp_lbdf, sym_hess_j))

                if i == 0:
                    block_shapes.append(sym_hess_j.shape)

            block_sym_hes.append(block_sym_ij)
            block_lbdf_hes.append(block_lbdf_ij)

        self.block_sym_hes = block_sym_hes
        self.block_hessian = block_lbdf_hes
        self.block_shapes = block_shapes

    @staticmethod
    def _build_jax_scalar(symbols: List[sy.Symbol],
                          expr: sy.Expr,
                          jit: bool = True) -> Callable:
        """Build a scalar (or vector) JAX callable for a SymPy expression.

        If jit is True the returned callable is JIT compiled on first use.
        No batching (vmap) is applied; intended for iterative nonlinear solves.
        """
        # Constant short-circuit
        expr_free = getattr(expr, 'free_symbols', set())
        if len(expr_free.intersection(set(symbols))) == 0:
            # constant expression
            const_fn = lambda *args, _v=expr: _v if isinstance(_v, (int, float)) else float(_v)
            return jax.jit(const_fn) if (jit and _USE_JAX_JIT) else const_fn

        base_fun = sy.lambdify(symbols, expr, modules='jax')

        if jit and _USE_JAX_JIT:
            base_fun = jax.jit(base_fun)

        return base_fun

    @staticmethod
    def _build_jax_gradient(symbols: List[sy.Symbol],
                             expr: sy.Expr,
                             diff_symbols: Optional[Sequence[sy.Symbol]] = None,
                             jit: bool = True) -> Callable:
        """Build gradient callable of scalar SymPy expression using JAX.

        diff_symbols: optional subset (and ordering) of symbols to differentiate w.r.t.
        Returns a function f(*args) -> (n_diff,) gradient vector.
        """
        # Validate scalar via safe shape access
        shape = getattr(expr, 'shape', None)
        if shape not in (None, (), (1,)):
            raise ValueError("Gradient requested for non-scalar expression.")

        if diff_symbols is None:
            diff_symbols = list(symbols)
        else:
            diff_symbols = list(diff_symbols)

        # Duplicate check
        if len(diff_symbols) != len(set(diff_symbols)):
            raise ValueError("diff_symbols contains duplicates; gradient rows ambiguous.")

        expr_free = getattr(expr, 'free_symbols', set())
        sym_to_idx = {s: i for i, s in enumerate(symbols)}
        try:
            argnums = tuple(sym_to_idx[s] for s in diff_symbols)
        except KeyError as e:
            missing = set(diff_symbols) - set(sym_to_idx)
            raise ValueError(f"diff_symbols contains symbols not in symbols list: {missing}") from e

        # Constant gradient shortcut
        if len(expr_free.intersection(set(diff_symbols))) == 0:
            zeros = jnp.zeros(len(diff_symbols))

            def grad_const(*args, _z=zeros):
                return _z

            return jax.jit(grad_const) if (jit and _USE_JAX_JIT) else grad_const

        base_fun = LambdifyBuilder._build_jax_scalar(symbols, expr, jit=False)

        def grad_fun(*args):
            g_tuple = jax.grad(base_fun, argnums=argnums)(*args)
            if len(argnums) == 1:
                return jnp.atleast_1d(g_tuple)
            return jnp.stack(g_tuple)

        if jit and _USE_JAX_JIT:
            grad_fun = jax.jit(grad_fun)
        return grad_fun

    @staticmethod
    def _build_jax_hessian(symbols: List[sy.Symbol],
                           expr: sy.Expr,
                           diff_symbols: Optional[Sequence[sy.Symbol]] = None,
                           jit: bool = True,
                           ) -> Callable:
        """Build Hessian callable of scalar SymPy expression using JAX.

        Returns f(*args) -> (n_diff, n_diff) Hessian matrix in order of diff_symbols.
        """
        # Validate scalar via safe shape access
        shape = getattr(expr, 'shape', None)
        if shape not in (None, (), (1,)):
            raise ValueError("Hessian requested for non-scalar expression.")

        if diff_symbols is None:
            diff_symbols = list(symbols)
        else:
            diff_symbols = list(diff_symbols)

        if len(diff_symbols) != len(set(diff_symbols)):
            raise ValueError("diff_symbols contains duplicates; hessian rows ambiguous.")

        expr_free = getattr(expr, 'free_symbols', set())
        sym_to_idx = {s: i for i, s in enumerate(symbols)}
        try:
            argnums = tuple(sym_to_idx[s] for s in diff_symbols)
        except KeyError as e:
            missing = set(diff_symbols) - set(sym_to_idx)
            raise ValueError(f"diff_symbols contains symbols not in symbols list: {missing}") from e

        if len(expr_free.intersection(set(diff_symbols))) == 0:
            zeros = jnp.zeros((len(diff_symbols), len(diff_symbols)))

            def hess_const(*args, _z=zeros):
                return _z

            return jax.jit(hess_const) if (jit and _USE_JAX_JIT) else hess_const

        base_fun = LambdifyBuilder._build_jax_scalar(symbols, expr, jit=False)

        def hess_fun(*args):
            h_blocks = jax.hessian(base_fun, argnums=argnums)(*args)
            if len(argnums) == 1:
                return jnp.atleast_2d(h_blocks)
            block_list = [list(row) for row in h_blocks]
            return jnp.block(block_list)

        if jit and _USE_JAX_JIT:
            hess_fun = jax.jit(hess_fun)

        return hess_fun
