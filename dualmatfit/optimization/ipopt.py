# -*- coding: utf-8 -*-
"""
IPOPT optimizer interface.

This module provides the IpyoptMinimizer class for interfacing with the
IPOPT interior-point optimizer for constrained nonlinear optimization.
"""
from __future__ import annotations

import numpy as np
import ipyopt
from enum import IntEnum
from typing import Sequence, Callable
from scipy.optimize import OptimizeResult, rosen, rosen_der

from dualmatfit.optimization.cost import CostFunction
from dualmatfit.utils.numeric import sanitize_gradient, is_finite

from dualmatfit.utils.logging_config import get_logger
logger = get_logger('optimization')

__all__ = [
    'IpyoptMinimizer',
    'IpoptStatus',
]


class IpoptStatus(IntEnum):
    """
    IPOPT solver return status codes.
    
    Based on Ipopt::ApplicationReturnStatus enumeration from the IPOPT library.
    See: https://coin-or.github.io/Ipopt/classIpopt_1_1ApplicationReturnStatus.html
    """
    # Success states
    SOLVE_SUCCEEDED = 0
    SOLVED_TO_ACCEPTABLE_LEVEL = 1
    
    # Non-optimal termination states
    INFEASIBLE_PROBLEM_DETECTED = 2
    SEARCH_DIRECTION_TOO_SMALL = 3
    DIVERGING_ITERATES = 4
    USER_REQUESTED_STOP = 5
    FEASIBLE_POINT_FOUND = 6
    
    # Limit exceeded states (partial success)
    MAXIMUM_ITERATIONS_EXCEEDED = -1
    RESTORATION_FAILED = -2
    ERROR_IN_STEP_COMPUTATION = -3
    MAXIMUM_CPUTIME_EXCEEDED = -4
    MAXIMUM_WALLTIME_EXCEEDED = -5
    
    # Error states
    NOT_ENOUGH_DEGREES_OF_FREEDOM = -10
    INVALID_PROBLEM_DEFINITION = -11
    INVALID_OPTION = -12
    INVALID_NUMBER_DETECTED = -13
    
    # Fatal error states
    UNRECOVERABLE_EXCEPTION = -100
    NON_IPOPT_EXCEPTION = -101
    INSUFFICIENT_MEMORY = -102
    INTERNAL_ERROR = -199
    
    @classmethod
    def is_success(cls, status: int) -> bool:
        """Check if status indicates optimization success."""
        return status in (cls.SOLVE_SUCCEEDED, cls.SOLVED_TO_ACCEPTABLE_LEVEL)
    
    @classmethod
    def is_acceptable(cls, status: int) -> bool:
        """Check if status indicates an acceptable result (success or iteration limit)."""
        return status in (
            cls.SOLVE_SUCCEEDED, 
            cls.SOLVED_TO_ACCEPTABLE_LEVEL,
            cls.MAXIMUM_ITERATIONS_EXCEEDED,
            cls.FEASIBLE_POINT_FOUND
        )
    
    @classmethod
    def get_message(cls, status: int) -> str:
        """Get human-readable message for status code."""
        try:
            status_enum = cls(status)
            return status_enum.name.replace('_', ' ').title()
        except ValueError:
            return f"Unknown status code: {status}"



def eval_ipopt_g(np_xi: np.ndarray, np_out: np.ndarray):
    """

    :param np_xi:
    :param np_out:
    :return:
    """
    return


def eval_ipopt_jac_g(np_xi: np.ndarray, np_out: np.ndarray):
    """

    :param np_xi:
    :param np_out:
    :return:
    """
    return


def ipopt_trial(warm_start: bool,
                lc_seed: int,
                xi_lwr: np.ndarray,
                xi_upp: np.ndarray,
                ini_dsvars: np.ndarray,
                xi_dt: np.ndarray,
                nvar: int,
                obj_fun: Callable | CostFunction,
                ncon: int = 0.,
                g_lwr: np.ndarray = None,
                g_upp: np.ndarray = None,
                cst_fun: Callable = None,
                cst_dfun: Callable = None,
                kwargs_set: dict = None,
                perturbation: float = 0.,
                mult_g: np.ndarray = None,
                mult_x_l: np.ndarray = None,
                mult_x_u: np.ndarray = None,
                seed: int = 0,
                ) -> (float, np.ndarray):

    opt_kwargs = kwargs_set.copy()
    np_vars = ini_dsvars.copy()

    if perturbation > 0.:
        np.random.seed(seed + lc_seed)
        np_rand = np.random.uniform(-1., 1., nvar)
        np_pert = perturbation * np_rand
        np_vars_dt = ini_dsvars.copy() + np_pert * xi_dt
        np_chk = (xi_lwr <= np_vars_dt) & (np_vars_dt < xi_upp)
        np_vars[np_chk] = np_vars_dt[np_chk]

    if g_lwr is None:
        g_lwr = np.array([], dtype=float)

    if g_upp is None:
        g_upp = np.array([], dtype=float)

    if cst_fun is None:
        jac_sparsity_indices = (np.array([]), np.array([]))
        cst_dfun = eval_ipopt_jac_g

    nlp = ipyopt.Problem(nvar, xi_lwr, xi_upp, ncon, g_lwr, g_upp, jac_sparsity_indices, 0,
                         obj_fun, obj_fun.derivative, cst_fun, cst_dfun)

    if warm_start:
        opt_kwargs.update({"warm_start_init_point": "yes", "warm_start_bound_push": 1.e-8,
                               "warm_start_slack_bound_push": 1.e-8, "warm_start_mult_bound_push": 1.e-8})

    nlp.set(**opt_kwargs)
    np_vars, obj, _ = nlp.solve(np_vars.astype(float), mult_g=mult_g, mult_x_L=mult_x_l, mult_x_U=mult_x_u)

    return np_vars, obj


class IpyoptMinimizer:
    """
    A generic local minimizer that calls ipyopt internally.
    You can adapt the eval_f, eval_grad_f, etc. or pass them in.
    """

    def __init__(self,
                 x_l: np.ndarray,
                 x_u: np.ndarray,
                 obj_fun: Callable | CostFunction,
                 obj_grad_fun: Callable = None,
                 ncon: int = 0,
                 g_l: np.ndarray = None,
                 g_u: np.ndarray = None,
                 eval_jac_g_sparsity_indices: Sequence[np.ndarray] = None,
                 eval_h_sparsity_indices: int = 0,
                 eval_g: Callable = None,
                 eval_jac_g: Callable = None,
                 ipyopt_options: dict = None,
                 ):
        """
        Parameters
        ----------
        x_l, x_u : np.ndarray
            Lower and upper bounds for the x variables.
        ncon : int
            Number of constraints.
        g_l, g_u : np.ndarray
            Lower/upper bounds on the constraints.
        eval_jac_g_sparsity_indices : tuple of np.ndarrays
            Row/col indices of the nonzero pattern in the constraint Jacobian.
        eval_h_sparsity_indices : int
            Placeholder flag for Hessian sparsity support. The current
            implementation uses ``0`` to indicate that no explicit Hessian
            sparsity pattern is supplied.
        obj_fun : callable
            Objective function ``f(x)`` or ``CostFunction`` instance providing
            the objective and its gradient.
        eval_g : callable, optional
            Constraint functions g(x).
        eval_jac_g : callable, optional
            Jacobian of the constraints.
        ipyopt_options : dict, optional
            Additional options to set on the ipyopt solver.
        """
        self.nvar = x_l.shape[0]

        self.x_l = x_l
        self.x_u = x_u

        # Store user-supplied callback cost function
        self.eval_f = obj_fun

        if obj_grad_fun is None:
            self._original_grad_f = obj_fun.derivative
        else:
            self._original_grad_f = obj_grad_fun

        def safe_gradient_wrapper(x, out=None):
            """Wrapper to catch and handle gradient evaluation errors."""
            try:
                grad = self._original_grad_f(x)
                if not is_finite(grad):
                    logger.warning(f"Gradient contains NaN/Inf at x={x}")
                    grad = sanitize_gradient(grad, log_warning=False)
                if out is not None:
                    out[:] = grad
                    return out
                return grad
            except (RuntimeError, ValueError) as e:
                logger.error(f"Gradient evaluation failed: {e}")
                fallback_grad = np.full(x.shape[0], 1e10)
                if out is not None:
                    out[:] = fallback_grad
                    return out
                return fallback_grad

        self.eval_grad_f = safe_gradient_wrapper

        # Store user-supplied constraint function
        self.ncon = ncon

        if g_l is None:
            g_l = np.array([], dtype=float)

        if g_u is None:
            g_u = np.array([], dtype=float)

        self.g_l = g_l
        self.g_u = g_u

        if eval_jac_g_sparsity_indices is None:
            eval_jac_g_sparsity_indices = (np.array([]), np.array([]))

        self.eval_jac_g_sparsity_indices = eval_jac_g_sparsity_indices

        if eval_h_sparsity_indices is None:
            eval_h_sparsity_indices = 0

        self.eval_h_sparsity_indices = eval_h_sparsity_indices

        if eval_g is None:
            eval_g = eval_ipopt_g

        if eval_jac_g is None:
            eval_jac_g = eval_ipopt_jac_g

        self.eval_g = eval_g
        self.eval_jac_g = eval_jac_g

        # Build the ipyopt problem
        self.nlp = ipyopt.Problem(
            self.nvar,
            self.x_l,
            self.x_u,
            self.ncon,
            self.g_l,
            self.g_u,
            eval_jac_g_sparsity_indices,
            eval_h_sparsity_indices,
            self.eval_f,
            self.eval_grad_f,
            self.eval_g if self.eval_g else (lambda x, out: None),  # no constraints
            self.eval_jac_g if self.eval_jac_g else (lambda x, out: None),  # no constraints
        )

        self.mult_x_l = np.zeros(self.nvar, dtype=float)
        self.mult_x_u = np.zeros(self.nvar, dtype=float)
        self.mult_g = np.zeros(self.ncon, dtype=float)
        self.ipopt_options = ipyopt_options

        # Optional: set ipyopt solver options if desired
        if ipyopt_options is not None:
            self.nlp.set(**ipyopt_options)

    def __call__(self,
                 x0: np.ndarray,
                 mult_g: np.ndarray = None,
                 mult_x_l: np.ndarray = None,
                 mult_x_u: np.ndarray = None,
                 ):
        """
        Run ipyopt from the given starting point x0,
        and return a scipy-style OptimizeResult.
        """
        # We will solve from x0 with no warm-start multipliers
        if mult_x_l is None:
            mult_x_l = self.mult_x_l

        if mult_x_u is None:
            mult_x_u = self.mult_x_u

        if mult_g is None:
            mult_g = self.mult_g

        # Solve
        x_sol, obj_val, status = self.nlp.solve(x0.copy(), mult_g=mult_g, mult_x_L=mult_x_l, mult_x_U=mult_x_u)

        # Construct a SciPy-like result
        res = OptimizeResult()
        res.x = x_sol
        res.fun = obj_val
        res.status = status

        # Use IpoptStatus enum for proper success determination
        res.success = IpoptStatus.is_acceptable(status)
        res.message = f"IPOPT: {IpoptStatus.get_message(status)}"
        logger.debug(f"\n Local Optimal: fval {res.fun:.4f},  DS Vars: {res.x.round(5)} \n")

        res.nfev = 1  # Approximation, ipyopt might have its own iteration count
        res.njev = 1  # Approximation
        res.nhev = 1  # Approximation

        return res


if __name__ == '__main__':
    # Here we optimize the Rosenbrock function.
    # Define the objective and its gradient.
    def rosen_fun(x):
        return rosen(x)


    def rosen_grad(x, out: np.ndarray = None):
        dfx = rosen_der(x)
        if out is not None:
            out[:] = dfx.astype(float)

        return dfx.astype(float)


    # Initial guess.
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    x0_lwr = np.zeros_like(x0)
    x0_upp = np.ones_like(x0) * 100.

    # (Optional) ipyopt options – for example, suppress printing:
    ipyopt_opts = {"print_level": 0, "max_iter": 1000,}

    min_ipopt = IpyoptMinimizer(x0_lwr, x0_upp, rosen_fun, obj_grad_fun=rosen_grad, ipyopt_options=ipyopt_opts)
    solution = min_ipopt(x0.copy())

    logger.debug(f"IPOPT result: {solution.message}")
    logger.debug(f"IPOPT solution: {solution}")

    test = 1.
