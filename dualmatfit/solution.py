# -*- coding: utf-8 -*-
"""
Numerical solvers for nonlinear equations and optimization.

This module provides Newton-based solvers (Quasi-Newton Trust Region)
and utility functions for solving nonlinear systems arising in 
material mechanics.
"""
import copy
import numpy as np
# import jax.numpy as jnp
import scipy.linalg as la
# import scipy.sparse.linalg as sla
# import numexpr as ne

from typing import List, Optional, Dict, Callable, Union

from scipy import optimize
from scipy.interpolate import Rbf
# from jax.scipy.linalg import solve
from numba import jit

from dualmatfit.barrier import log_barrier, inv_barrier_function
from dualmatfit.numeric_utils import sanitize_array, has_nan, has_inf, is_finite

from dualmatfit.logging_config import get_logger
logger = get_logger('solvers')

__all__ = [
    'rescale',
    'Root',
    'QuasiNewtonTrustRegion',
]



def rescale(x: Union[float, np.ndarray],
            lb: Union[float, np.ndarray],
            ub: Union[float, np.ndarray],
            ) -> Optional[Union[float, np.ndarray]]:
    """
    Rescale a value or array of values to the range [0, 1] based on their original bounds.

    Parameters:
    - x (float or np.ndarray): The value or array of values to be rescaled.
    - lb (float or np.ndarray): The lower bound or array of lower bounds.
    - ub (float or np.ndarray): The upper bound or array of upper bounds.

    Returns:
    - float or np.ndarray: The rescaled value or array of values.

    Notes:
    - If a value's lower bound is equal to its upper bound, the rescaled value will be 0.5.
    - If the input is a float and its bounds are equal, the function will return 0.5.
    - If the input is an array, its shape must match the shapes of the bounds arrays.
    """

    if isinstance(x, np.ndarray):
        # Check the shape of the arrays
        if x.shape != lb.shape or x.shape != ub.shape:
            raise ValueError(
                f"The arrays do not have the same dimensions! "
                f"x.shape={x.shape}, lb.shape={lb.shape}, ub.shape={ub.shape}"
            )

        # Determine where lb is equal to ub
        mask_id = lb == ub

        # Rescale the values, using 0.5 for positions where lb equals ub
        x_primal = np.where(mask_id, 0.5, (x - lb) / (ub - lb))
        return x_primal

    elif isinstance(x, float):
        if lb == ub:
            return 0.5
        else:
            return (x - lb) / (ub - lb)

    return None


@jit(nopython=True)
def lbfgs_predictor(s_values: List[np.ndarray], y_values: List[np.ndarray], dfx: np.ndarray) -> np.ndarray:
    """
    Computes the L-BFGS approximation to the inverse Hessian-vector product using the two-loop recursion algorithm.

    Parameters:
    - s_values (np.ndarray): List of difference of consecutive x values.
    - y_values (np.ndarray): List of difference of consecutive gradient values.
    - dfx (np.ndarray): Gradient of the function at the current x value.

    Returns:
    - np.ndarray: Approximation of the inverse Hessian-vector product.
    """

    n = len(s_values)
    q = dfx.copy()
    alpha_values = np.empty(n)

    rho_values = np.empty(n)
    for i in range(n):
        dot_ys = np.dot(y_values[i], s_values[i])
        # Guard against division by zero
        rho_values[i] = 1. / dot_ys if dot_ys != 0 else 0.0

    for i in range(n - 1, -1, -1):
        s_i = s_values[i]
        y_i = y_values[i]
        rho_i = rho_values[i]
        alpha_i = rho_i * np.dot(s_i, q)
        alpha_values[i] = alpha_i
        q -= alpha_i * y_i

    if n > 0:
        dot_sy = np.dot(s_values[-1], y_values[-1])
        dot_yy = np.dot(y_values[-1], y_values[-1])
        # Guard against division by zero
        gamma_k = dot_sy / dot_yy if dot_yy != 0 else 1.0
        r_k = gamma_k * q
    else:
        r_k = q

    for i in range(n):
        s_i = s_values[i]
        y_i = y_values[i]
        rho_i = rho_values[i]
        alpha_i = alpha_values[i]
        beta_i = rho_i * np.dot(y_i, q)
        r_k += s_i * (alpha_i - beta_i)

    return -r_k


def linsolver(a: np.ndarray, b: np.ndarray, fast: bool = True) -> np.ndarray:
    identity = np.eye(a.shape[1])

    if fast:
        if la.det(a) < 0:
            reg_a = np.dot(a.T, a)
            reg_b = np.dot(a.T, b)
            x_res = la.lstsq(reg_a + 1.e-9 * identity, reg_b, cond=0.00001)
            x = x_res[0]

        else:
            np_q, np_r = np.linalg.qr(a)
            np_y = np.dot(np_q.T, b)

            x_res = la.lstsq(np_r, np_y, cond=0.000001)
            x = x_res[0]
            # x = solve(np_r + 1.e-6 * identity, np_y)

        return x

    else:
        return la.solve(a + 1.e-6 * identity, b)


@jit(nopython=True)
def jit_linsolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Parameters:
    -----------
    a : np.ndarray
        Coefficient matrix.
    b : np.ndarray
        Right-hand side vector.
    stab : float, optional
        Stabilization term to be added to the diagonal of matrix `a`. Default is 1.e-9.

    Returns:
    --------
    np.ndarray
        results from solved the linear system

    Note:
    -----
    This function is optimized using Numba's @njit decorator.
    """
    np_a_sym = (a + a.T) / 2

    return np.linalg.solve(np_a_sym, b)


class Root:
    """
    Root-finding solver using Newton's method, with support for mixed
    variational formulations and bound constraints via a barrier method.
    """

    MAX_INC = 300
    def __init__(self,
                 fun: Callable[[np.ndarray], np.ndarray],
                 jac: Callable[[np.ndarray], np.ndarray],
                 bounds: Optional[Dict[str, np.ndarray]] = None,
                 block_array: Optional[List[np.ndarray]] = None,
                 mu: float = None,
                 btype: str = None,
                 max_iter: int = 100,
                 tol: float = 1e-8,
                 disp: bool = False,
                 solver_type: str = 'newton',     # 'newton', 'least_squares', 'scipy_root'
                 use_line_search: bool = True,
                 ):
        """
        Initialize the Root solver.

        Args:
            fun (Callable): The function whose root is to be found.
            jac: The Jacobian of the function. Returns *either* a dense 2D NumPy array
                 (for single-field problems) *or* a list of lists of NumPy arrays,
                 representing the blocks of the Jacobian matrix (for mixed formulations).
            bounds: A dictionary with 'lower' and 'upper' keys, each containing a NumPy
                array of bounds.
            block_array:  If None, standard formulation.  If a list of NumPy arrays,
                mixed formulation.  Each array in the list contains the indices
                of the primal variables belonging to a block.
            btype (str, optional):      Barrier type ('log' or 'inv'). Default is 'log'.
            max_iter (int, optional):   Maximum number of iterations. Defaults to 100.
            tol (float, optional):      Tolerance for convergence. Defaults to 1e-8.
            disp (bool, optional):      Whether to display iteration information. Defaults to False.
            solver_type (str):          Which solver to use ('newton' or 'scipy').
            use_line_search (bool, optional): Whether to use line search. Defaults to True.
        """

        self.fun = fun
        self.jac = jac
        self.block_array = block_array

        self.bounds = bounds
        if bounds is not None:
            self.lb = bounds['lower']
            self.ub = bounds['upper']
        else:
            self.lb = None
            self.ub = None

        self.btype = btype
        self.it = 0
        self.inc = 0.9          # Decreasing rate of the barrier parameter
        self.alpha = 0.7        # Step size for Newton's method

        if mu is None:
            mu = 0.1

        self.imu = mu
        self.mu = mu
        self.stab = 1.e-6
        self.max_iter = max_iter
        self.tol = tol
        self.disp = disp
        self.solver_type = solver_type
        self.use_line_search = use_line_search
        self.is_mixed = False

        # Determine if it's a mixed formulation
        if block_array is None:
            block_array = []

        self.block_array = block_array

        if len(block_array) > 1:
            self.is_mixed = True

        # variables for solution looping
        self._xi = None
        self._dxi = None

        # Select barrier functions
        if btype == 'log':
            self.barrier_fun = log_barrier
        elif btype == 'inv':
            self.barrier_fun = inv_barrier_function
        elif btype is None:
            self.btype = None
        else:
            self.btype = None
            raise ValueError(f"Barrier type '{btype}' is not implemented. Supported types: 'log', 'inv', or None.")

    def _barrier_function(self, xi: np.ndarray) -> float:
        """Compute the barrier function value to penalize boundary violations.

        Args:
            xi (np.ndarray): The current variable values.

        Returns:
            float: The barrier function value.
        """

        return self.mu * self.barrier_fun(xi, self.lb, self.ub, dx=0) if self.lb is not None else 0.0

    def _barrier_jac(self, xi: np.ndarray) -> np.ndarray:
        """Compute the barrier function gradient to penalize boundary violations.

        Args:
            xi (np.ndarray): The current variable values.

        Returns:
            np.ndarray: The gradient of the barrier function.
        """

        if self.lb is not None and self.ub is not None:
            grad_phi = self.mu * self.barrier_fun(xi, self.lb, self.ub, dx=1)
            return grad_phi
        else:
            return np.zeros_like(xi)

    def _barrier_hessian(self, xi: np.ndarray) -> np.ndarray:
        """Compute the barrier function Hessian to penalize boundary violations.

        Args:
            xi (np.ndarray): The current variable values.

        Returns:
            np.ndarray: The Hessian of the barrier function.
        """

        if self.lb is not None and self.ub is not None:
            return self.mu * self.barrier_fun(xi, self.lb, self.ub, dx=2)
        else:
            return np.zeros_like(xi)

    def _fun(self, xi: np.ndarray) -> np.ndarray:
        """Compute the function value with barrier penalties added to residuals.

        Args:
            xi (np.ndarray): The current variable values.

        Returns:
            np.ndarray: The function residuals with barrier penalties.
        """

        fx_val = self.fun(xi)
        lsq_fx_val = 0.5 * np.dot(fx_val, fx_val)

        if self.btype is not None:
            barrier_residual = self._barrier_function(xi)
            return lsq_fx_val + barrier_residual
        else:
            return lsq_fx_val

    def _jac(self, xi: np.ndarray) -> np.ndarray:
        """Compute the Jacobian with barrier penalties.

        Args:
            x (np.ndarray): The current variable values.

        Returns:
            np.ndarray: The Jacobian matrix with barrier penalties.
        """

        fx_val = self.fun(xi)
        jac_val = self.jac(xi)
        lsq_jac_val = np.inner(fx_val, jac_val)

        if self.btype is not None:
            return lsq_jac_val + self._barrier_jac(xi)
        else:
            return lsq_jac_val

    def _hessian(self, xi: np.ndarray) -> np.ndarray:
        """Compute the hessian with barrier penalties.

        Args:
            x (np.ndarray): The current variable values.

        Returns:
            np.ndarray: The hessian matrix with barrier penalties.
        """

        jac_val = self.jac(xi)
        hess_val = jac_val.T @ jac_val

        if self.btype is not None:
            return hess_val + self._barrier_hessian(xi)
        else:
            return hess_val

    @staticmethod
    def _assemble_jacobian(jac_blocks: List[List[np.ndarray]]) -> np.ndarray:
        """Assembles the block Jacobian matrix from its blocks."""

        # if len(jac_blocks) == 2:

        return np.block(jac_blocks)

    def _split_solution(self, x: np.ndarray) -> List[np.ndarray]:
        """Splits the full solution vector into blocks."""
        if len(self.block_array) > 1:
            return [x[indices] for indices in self.block_array]
        else:
            return [x]  # Return as a single block if not mixed

    def _prepare_for_scipy(self, x0):
        """Prepare functions and initial guess for SciPy solvers."""
        if self.is_mixed:
            # Flatten the initial guess
            x0_flat = np.concatenate(x0, axis=0)

            def fun_flat(x_flat) -> np.ndarray:
                # Split the flattened x back into blocks
                x_blocks = self._split_solution(x_flat)
                # Call the original function with the blocked x
                fval = self.fun(x_blocks)

                if isinstance(fval, np.ndarray):
                    np_fval = fval.flatten()
                elif isinstance(fval, list):
                    np_fval = np.concatenate(fval)

                return sanitize_array(np_fval)

            def jac_flat(x_flat):
                # Split, call original jac, and reassemble
                x_blocks = self._split_solution(x_flat)
                jac_blocks = self.jac(x_blocks)
                jac_mtx = self._assemble_jacobian(jac_blocks)

                if len(jac_mtx.shape) == 3:
                    jac_mtx = jac_mtx[:, :, 0]

                return sanitize_array(jac_mtx)

            return x0_flat, fun_flat, jac_flat
        else:
            # For non-mixed, use the original functions directly
            return x0, self.fun, self.jac

    def _newton_solve(self, x0: np.ndarray, alpha: float = 0.7, restart: bool = True) -> optimize.OptimizeResult:
        """Solve using custom Newton's method with optional line search."""

        if restart:
            self.mu = copy.copy(self.imu)

        x_k = x0.copy()

        for k in range(self.max_iter):
            residual = self.fun(x_k)
            jacobian = self.jac(x_k)

            # Assemble the Jacobian if it's block-structured
            if isinstance(jacobian, list):
                jacobian = self._assemble_jacobian(jacobian)

            # --- Add Barrier Contributions ---
            if self.bounds is not None:
                residual += self._barrier_jac(x_k)              # Add barrier GRADIENT to RESIDUAL
                jacobian += self._barrier_hessian(x_k)          # Add barrier HESSIAN to JACOBIAN

            # Solve the linear system
            try:
                delta_x = -np.linalg.solve(jacobian, residual)
            except np.linalg.LinAlgError:
                return optimize.OptimizeResult(x=x_k,
                                               success=False,
                                               nit=(k + 1),
                                               message="Singular Jacobian encountered.")

            # Update the solution
            if self.use_line_search:
                alpha_k = 1.0
                # Backtracking line search.  Use _fun (which includes the barrier).
                while self._fun(x_k + alpha_k * delta_x) > self._fun(x_k) + 1e-4 * alpha_k * np.dot(self._jac(x_k).T, delta_x):
                    alpha_k *= 0.5
                    if alpha_k < 1e-4:  # Prevent infinite loops and very small steps
                        break
                if has_nan(np.atleast_1d(self._fun(x_k + alpha_k * delta_x))):
                    alpha_k = 1e-4
            else:
                alpha_k = alpha

            x_k = x_k + alpha_k * sanitize_array(delta_x)

            # --- Project to Bounds (Important!) ---
            if self.bounds is not None:  # Ensure we stay *strictly* within bounds
                x_k = np.maximum(self.lb + 1e-9, np.minimum(self.ub - 1e-9, x_k))

            # Check for convergence
            if np.linalg.norm(delta_x) < self.tol and k > 5:
                return optimize.OptimizeResult(x=x_k, fun=residual, success=True, nit=(k + 1), nfev=k+1, njev=k+1)

            # Decrease barrier parameter mu
            if self.bounds is not None and self.mu > 1.e-9:
                self.mu *= self.inc

        return optimize.OptimizeResult(x=x_k,
                                       fun=residual,
                                       success=False,
                                       nit=self.max_iter,
                                       message="Maximum iterations reached.")

    def _least_squares_solve(self, x0: np.ndarray, method: str = 'trf') -> optimize.OptimizeResult:
        """Solve using SciPy's least_squares."""

        x0_flat, fun_flat, jac_flat = self._prepare_for_scipy(x0)

        if method == 'lm':
            bounds = (None, None)
        else:
            # Default to unbounded
            bounds = (-np.inf, np.inf)
            if self.bounds is not None:
                bounds = (self.bounds['lower'], self.bounds['upper'])

        try:
            result = optimize.least_squares(
                fun=fun_flat,
                x0=x0_flat,
                jac=jac_flat,
                bounds=bounds,
                method=method,
                xtol=self.tol,
                ftol=self.tol,
                gtol=self.tol,
                max_nfev=self.max_iter,
            )
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
            # LinAlgError: Singular matrix or convergence issues
            # ValueError: Invalid input parameters
            # RuntimeError: Iteration limit or other runtime issues
            logger.debug(f"An error occurred during least_squares solution process: {e}, x0: {x0_flat}")
            logger.debug("Falling back to scipy.root")
            result = optimize.root(fun_flat, x0_flat, jac=jac_flat, method='hybr', tol=self.tol)

        return result

    def _scipy_root_solve(self, x0: np.ndarray) -> optimize.OptimizeResult:
        """Solve using SciPy's root function."""

        x0_flat, fun_flat, jac_flat = self._prepare_for_scipy(x0)

        # For mixed formulations, we need to use a method that handles jacobians as lists
        # 'hybr' is a good general-purpose choice.
        result = optimize.root(fun_flat, x0_flat, jac=jac_flat, method='hybr', tol=self.tol)
        return result

    def solve(self, x0: Union[List[np.ndarray], np.ndarray], **kwargs) -> optimize.OptimizeResult:
        """Solve the root-finding problem using the specified solver."""

        if not isinstance(x0, (np.ndarray, list)):
            raise TypeError("x0 must be a list or numpy array")

        if self.solver_type == 'newton':
            return self._newton_solve(x0, **kwargs)
        elif self.solver_type == 'least_squares':
            return self._least_squares_solve(x0)
        elif self.solver_type == 'scipy_root':
            return self._scipy_root_solve(x0)
        else:
            raise ValueError(f"Invalid solver_type: {self.solver_type}")
