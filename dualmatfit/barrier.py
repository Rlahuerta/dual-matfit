# -*- coding: utf-8 -*-
"""
Barrier functions for constrained optimization.

This module provides logarithmic and inverse barrier functions
for handling bound constraints in optimization algorithms.
"""
import numpy as np

from typing import Union
from numba import jit

__all__ = [
    'safe_log_barrier_function',
    'safe_log',
    'log_barrier',
    'inv_barrier_function',
]


@jit(nopython=True)
def safe_log_barrier_function(lower: np.ndarray,
                              upper: np.ndarray,
                              log_lower: np.ndarray,
                              log_upper: np.ndarray,
                              dx: int = 0,
                              ) -> np.ndarray:
    """Compute the log barrier function for bound constraints."""

    if dx == 0:
        return np.array([[np.sum(log_lower + log_upper)]])

    elif dx == 1:
        # Guard against division by zero
        lower_safe = np.where(lower == 0, 1e-12, lower)
        upper_safe = np.where(upper == 0, 1e-12, upper)
        return (1. / lower_safe - 1. / upper_safe).reshape(-1, 1)

    elif dx == 2:
        # Guard against division by zero
        lower_safe = np.where(lower == 0, 1e-12, lower)
        upper_safe = np.where(upper == 0, 1e-12, upper)
        np_diag_fdx2 = - 1. / (lower_safe ** 2) - 1. / (upper_safe ** 2)
        return np.diagflat(np_diag_fdx2)

    else:
        raise NotImplementedError


def safe_log(x: np.ndarray) -> np.ndarray:
    """
    Computes the natural logarithm of x, handling potential negative or zero values.

    Args:
        x: Input array.

    Returns:
        Logarithm of x, with -inf for non-positive values.  Uses where to avoid
        warnings.
    """
    return np.log(x, out=np.full_like(x, -np.inf, dtype=float), where=(x > 0))


def log_barrier(xi: np.ndarray,
                lb: np.ndarray,
                lu: np.ndarray,
                dx: int = 0,
                epsilon: float = 1.e-9,
                ) -> Union[float, np.ndarray]:
    """
    Compute the logarithmic barrier function and its derivatives for variables with lower and upper bounds.

    Parameters:
    - xi (np.ndarray):  The current variable values.
    - lb (np.ndarray):  Lower bounds (use -np.inf for no lower bound).
    - lu (np.ndarray):  Upper bounds (use np.inf for no upper bound).
    - dx (int):         Derivative order (0: function value, 1: gradient, 2: Hessian).
    - epsilon (float):  Small positive number to avoid division by zero or log of zero.

    Returns:
    - Union[float, np.ndarray]: The barrier function value or its derivatives.
        - If dx == 0: returns a scalar barrier function value.
        - If dx == 1: returns a gradient vector.
        - If dx == 2: returns a Hessian matrix.

    Raises:
        ValueError: If dx is not 0, 1, or 2.
    """

    xi = np.asarray(xi, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(lu, dtype=float)

    # Ensure xi, lb, and lu have the same shape
    if xi.shape != lb.shape or xi.shape != lu.shape:
        raise ValueError("xi, lb, and lu must have the same shape.")

    

    # Clip the differences to be at least epsilon, *before* any calculations.
    d_lower = np.maximum(xi - lb, epsilon)
    d_upper = np.maximum(ub - xi, epsilon)

    if dx == 0:
        # Barrier function value
        return -np.sum(safe_log(d_lower)) - np.sum(safe_log(d_upper))

    elif dx == 1:
        # Gradient
        return (1.0 / d_upper) - (1.0 / d_lower)

    elif dx == 2:
        # Hessian (diagonal matrix)
        hess_diag = (1.0 / d_upper ** 2) + (1.0 / d_lower ** 2)
        return np.diag(hess_diag)

    else:
        raise ValueError("Invalid value for dx. Must be 0, 1, or 2.")


def inv_barrier_function(xi: np.ndarray,
                         lb: np.ndarray,
                         ub: np.ndarray,
                         dx: int = 0,
                         epsilon: float = 1.e-12,
                         ):
    """
    Inverse barrier function for constraints lb <= xi <= ub.

    Args:
        xi (np.ndarray): Variable values.
        lb (np.ndarray): Lower bounds.
        ub (np.ndarray): Upper bounds.
        dx (int): 0 for function value, 1 for gradient, 2 for Hessian.
        epsilon (float): Small value to prevent division by zero.

    Returns:
        Depending on dx, returns the function value, gradient, or Hessian.
    """

    # Guard against division by zero with epsilon clipping
    np_lwr = np.maximum(xi - lb, epsilon)
    np_upp = np.maximum(ub - xi, epsilon)

    if dx == 0:
        np_lwr_inv = 1. / np_lwr
        np_upp_inv = 1. / np_upp

        return np.array([[np.sum(np_lwr_inv + np_upp_inv)]])

    elif dx == 1:
        np_barrier_lwr = 1. / np_lwr ** 2
        np_barrier_upp = 1. / np_upp ** 2

        return (-np_barrier_lwr + np_barrier_upp).reshape(-1, 1)

    elif dx == 2:
        np_barrier_lwr = 2. / np_lwr ** 3
        np_barrier_upp = 2. / np_upp ** 3

        return np.diagflat(np_barrier_lwr + np_barrier_upp)

    else:
        raise ValueError("Invalid value for dx. Must be 0, 1, or 2.")
