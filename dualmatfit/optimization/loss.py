# -*- coding: utf-8 -*-
"""
Cost functions for optimization in material fitting.

This module provides various loss functions (LSQ, Cauchy, Huber, Log-Cosh, etc.)
and their derivatives for use in optimization algorithms.
"""
from __future__ import annotations

import numpy as np

from dualmatfit.utils.numeric import safe_divide

__all__ = [
    # Least squares
    'lsq_fval',
    'lsq_dfval',
    'lsq_wise_fval',
    'lsq_wise_dfval',
    # Cauchy loss
    'cauchy_fval',
    'cauchy_dfval',
    # Huber loss
    'huber_fval',
    'huber_dfval',
    # Log-Cosh loss
    'logcosh_fval',
    'logcosh_dfval',
    # Logarithmic loss
    'ln_fval',
    'ln_dfval',
    'ln_lsq_fun',
    'ln_lsq_fun_diff',
    # Sum-of-squares helpers
    'sum_lsq_fun',
    'sum_lsq_fun_diff',
]


def _ensure_2d_residuum(arr: np.ndarray) -> np.ndarray:
    """
    Ensure residuum array is at least 2D by expanding first axis if needed.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array, either 1D (n_residuals,) or 2D (n_rows, n_residuals).
        
    Returns
    -------
    np.ndarray
        2D array with shape (n_rows, n_residuals). If input was 1D, 
        n_rows=1 after expansion.
    """
    arr = arr.copy()
    return np.expand_dims(arr, axis=0) if arr.ndim == 1 else arr


def _ensure_3d_jacobian(arr: np.ndarray) -> np.ndarray:
    """
    Ensure Jacobian array is at least 3D by expanding first axis if needed.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array, either 2D (n_residuals, n_vars) or 3D (n_rows, n_residuals, n_vars).
        
    Returns
    -------
    np.ndarray
        3D array with shape (n_rows, n_residuals, n_vars). If input was 2D,
        n_rows=1 after expansion.
    """
    arr = arr.copy()
    return np.expand_dims(arr, axis=0) if arr.ndim == 2 else arr


def lsq_fval(residuum: np.ndarray, **kwargs) -> float:
    """
    Compute the least squares function value.

    Args:
        residuum (np.ndarray): Residual vector.

    Returns:
        float: Function value.
    """
    residuum_in = _ensure_2d_residuum(residuum)
    return np.sum(np.linalg.norm(residuum_in, axis=1))


def lsq_dfval(residuum: np.ndarray, residuum_diff: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute the derivative of the least squares function.

    Args:
        residuum (np.ndarray): Residual vector.
        residuum_diff (np.ndarray): Derivative of residuals.

    Returns:
        np.ndarray: Gradient vector.
    """
    residuum_in = _ensure_2d_residuum(residuum)
    residuum_diff_in = _ensure_3d_jacobian(residuum_diff)

    norms = np.linalg.norm(residuum_in, axis=1)

    # Einsum for batched dot product
    dfvals = np.einsum('ij,ijk->ik', residuum_in, residuum_diff_in)

    # Normalize by the norm of each residual vector (guard against zero norms)
    dfvals_normalized = safe_divide(dfvals, norms[:, np.newaxis], default=0.0)

    return np.sum(dfvals_normalized, axis=0)


def lsq_wise_fval(residuum: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute the least squares element-wise function value.

    Args:
        residuum (np.ndarray): Residual vector.

    Returns:
        float: Function value.
    """
    residuum_in = _ensure_2d_residuum(residuum)

    list_fvals = []
    for i in range(residuum_in.shape[0]):
        residuum2_i = residuum_in[i, :] ** 2
        list_fvals.append(residuum2_i)

    return np.concatenate(list_fvals, axis=0)


def lsq_wise_dfval(residuum: np.ndarray, residuum_diff: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute the derivative of the least squares element-wise function.

    Args:
        residuum (np.ndarray): Residual vector.
        residuum_diff (np.ndarray): Derivative of residuals.

    Returns:
        np.ndarray: Gradient vector.
    """
    residuum_in = _ensure_2d_residuum(residuum)
    residuum_diff_in = _ensure_3d_jacobian(residuum_diff)

    # Vectorized computation of the derivative
    # residuum_in shape: (n_rows, n_residuals) -> (n_rows, n_residuals, 1)
    # residuum_diff_in shape: (n_rows, n_residuals, n_vars)
    # result shape: (n_rows, n_residuals, n_vars)
    d_res2 = 2. * residuum_in[..., np.newaxis] * residuum_diff_in

    # Reshape to (n_rows * n_residuals, n_vars)
    return d_res2.reshape(-1, d_res2.shape[-1])


def cauchy_fval(residuum: np.ndarray, c: float, **kwargs) -> float:
    """
    Compute the Cauchy loss function value.

    Args:
        residuum (np.ndarray): Residual vector.
        c (float): Scaling parameter.

    Returns:
        float: Function value.
    """
    residuum_in = _ensure_2d_residuum(residuum)

    list_fvals = []
    for i in range(residuum_in.shape[0]):
        residuum2_i = (residuum_in[i, :] / c) ** 2
        fval_i = 0.5 * c ** 2 * np.sum(np.log1p(residuum2_i))
        list_fvals.append(fval_i)

    return sum(list_fvals)


def cauchy_dfval(residuum: np.ndarray, residuum_diff: np.ndarray, c: float, **kwargs) -> np.ndarray:
    """
    Compute the derivative of the Cauchy loss function.

    Args:
        residuum (np.ndarray): Residual vector.
        residuum_diff (np.ndarray): Derivative of residuals (Jacobian).
        c (float): Scaling parameter.

    Returns:
        np.ndarray: Gradient vector.
    """
    residuum_in = _ensure_2d_residuum(residuum)
    residuum_diff_in = _ensure_3d_jacobian(residuum_diff)

    list_dfvals = []
    for i in range(residuum_in.shape[0]):
        residuum2_i = (residuum_in[i, :] / c) ** 2
        denom_i = 1. + residuum2_i

        weights_i = residuum_in[i, :] / denom_i                     # Shape: (n_residuals,)
        np_dfval_i = weights_i @ residuum_diff_in[i, :, :]          # Shape: (n_vars,)
        list_dfvals.append(np_dfval_i)

    return np.sum(list_dfvals, axis=0)


def huber_fval(residuum: np.ndarray, delta: float = 1., **kwargs) -> float:
    """
    Compute the Huber loss function value.

    Args:
        residuum (np.ndarray): Residual vector.
        delta (float): Threshold parameter.

    Returns:
        float: Function value.
    """
    residuum_in = _ensure_2d_residuum(residuum)

    list_fvals = []
    for i in range(residuum_in.shape[0]):
        abs_res_i = np.abs(residuum_in[i, :])
        resid2_i = 0.5 * residuum_in[i, :] ** 2
        linear_i = delta * (abs_res_i - 0.5 * delta)
        fval_i = np.where(abs_res_i <= delta, resid2_i, linear_i)
        list_fvals.append(np.sum(fval_i))

    return sum(list_fvals)


def huber_dfval(residuum: np.ndarray, residuum_diff: np.ndarray, delta: float = 1., **kwargs) -> np.ndarray:
    """
    Compute the derivative of the Huber loss function.

    Args:
        residuum (np.ndarray): Residual vector.
        residuum_diff (np.ndarray): Derivative of residuals.
        delta (float): Threshold parameter.

    Returns:
        np.ndarray: Gradient vector.
    """
    residuum_in = _ensure_2d_residuum(residuum)
    residuum_diff_in = _ensure_3d_jacobian(residuum_diff)

    list_dfvals = []
    for i in range(residuum_in.shape[0]):
        abs_res_i = np.abs(residuum_in[i, :])
        np_grad_i = np.where(abs_res_i <= delta, residuum_in[i, :], delta * np.sign(residuum_in[i, :]))
        np_dfval_i = np_grad_i @ residuum_diff_in[i, :, :]
        list_dfvals.append(np_dfval_i)

    return np.sum(list_dfvals, axis=0)


def logcosh_fval(residuum: np.ndarray, **kwargs) -> float:
    """
    Compute the Log-Cosh function value.

    Args:
        residuum (np.ndarray): Residual vector.

    Returns:
        float: Function value.
    """
    residuum_in = _ensure_2d_residuum(residuum)

    list_fvals = []
    for i in range(residuum_in.shape[0]):
        list_fvals.append(np.log(np.cosh(residuum_in[i, :])))

    return sum(list_fvals)


def logcosh_dfval(residuum: np.ndarray, residuum_diff: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute the derivative of the Log-Cosh function.

    Args:
        residuum (np.ndarray): Residual vector.
        residuum_diff (np.ndarray): Derivative of residuals.

    Returns:
        np.ndarray: Gradient vector.
    """
    residuum_in = _ensure_2d_residuum(residuum)
    residuum_diff_in = _ensure_3d_jacobian(residuum_diff)

    list_dfvals = []
    for i in range(residuum_in.shape[0]):
        tanh_residuum_i = np.tanh(residuum_in[i, :])
        list_dfvals.append(tanh_residuum_i @ residuum_diff_in[i, :, :])

    return np.sum(list_dfvals, axis=0)


def ln_fval(residuum: np.ndarray, **kwargs) -> float:
    """
    Compute the logarithm of the sum of squares function value.

    Args:
        residuum (np.ndarray): Residual vector.

    Returns:
        float: Function value.
    """
    residuum_in = _ensure_2d_residuum(residuum)

    list_fvals = []
    for i in range(residuum_in.shape[0]):
        residuum2_i = np.dot(residuum_in[i, :], residuum_in[i, :])
        fval_i = np.log1p(residuum2_i)
        list_fvals.append(fval_i)

    return sum(list_fvals)


def ln_dfval(residuum: np.ndarray, residuum_diff: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute the derivative of the logarithm of the sum of squares function.

    Args:
        residuum (np.ndarray): Residual vector.
        residuum_diff (np.ndarray): Derivative of residuals.

    Returns:
        np.ndarray: Gradient vector.
    """
    residuum_in = _ensure_2d_residuum(residuum)
    residuum_diff_in = _ensure_3d_jacobian(residuum_diff)

    list_dfvals = []
    for i in range(residuum_in.shape[0]):
        residuum2_i = np.dot(residuum_in[i, :], residuum_in[i, :])
        np_dfval_i = (2. / (residuum2_i + 1.)) * np.dot(residuum_in[i, :], residuum_diff_in[i, :, :])
        list_dfvals.append(np_dfval_i)

    return np.sum(list_dfvals, axis=0)


def sum_lsq_fun(y: np.ndarray) -> float:
    return np.dot(y, y)


def sum_lsq_fun_diff(y: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return 2. * np.dot(y, dy)


def ln_lsq_fun(y: np.ndarray) -> float:
    y2_1 = np.dot(y, y)
    return np.log1p(y2_1)


def ln_lsq_fun_diff(y: np.ndarray, dy: np.ndarray) -> np.ndarray:
    y2_1 = np.dot(y, y) + 1.
    return (2. / y2_1) * np.dot(y, dy)
