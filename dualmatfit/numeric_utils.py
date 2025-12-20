"""
Numeric Utilities for NaN/Inf Handling.

This module provides centralized, consistent functions for handling
NaN and Inf values in numerical computations across the codebase.

Patterns standardized:
- `sanitize_array`: Replace NaN/Inf with safe defaults (standard replacement values)
- `sanitize_gradient`: Replace NaN/Inf with large finite values for optimization
- `has_nan`: Check if array contains NaN values
- `has_inf`: Check if array contains Inf values
- `is_finite`: Check if all values in array are finite
"""

from __future__ import annotations

import logging
from typing import Union, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    'sanitize_array',
    'sanitize_gradient',
    'has_nan',
    'has_inf',
    'is_finite',
    'has_non_finite',
    'safe_divide',
]

# --- Default replacement values ---
DEFAULT_NAN_REPLACEMENT = 0.0
DEFAULT_POSINF_REPLACEMENT = 0.0
DEFAULT_NEGINF_REPLACEMENT = 0.0

# --- Gradient-specific replacement values (for optimization) ---
GRADIENT_NAN_REPLACEMENT = 1e10
GRADIENT_POSINF_REPLACEMENT = 1e10
GRADIENT_NEGINF_REPLACEMENT = -1e10


def sanitize_array(
    arr: np.ndarray,
    nan: float = DEFAULT_NAN_REPLACEMENT,
    posinf: float = DEFAULT_POSINF_REPLACEMENT,
    neginf: float = DEFAULT_NEGINF_REPLACEMENT,
    copy: bool = True,
    log_warning: bool = False
) -> np.ndarray:
    """
    Replace NaN and Inf values in an array with safe defaults.
    
    This is the standard pattern for sanitizing numerical arrays used in
    residuum computation, volume regularization, and general numerical outputs.
    
    Args:
        arr: Input numpy array to sanitize.
        nan: Replacement value for NaN. Default is 0.0.
        posinf: Replacement value for positive infinity. Default is 0.0.
        neginf: Replacement value for negative infinity. Default is 0.0.
        copy: If True (default), returns a copy. If False, may modify in-place.
        log_warning: If True, logs a warning when NaN/Inf values are found.
        
    Returns:
        Sanitized numpy array with NaN/Inf replaced.
        
    Examples:
        >>> arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
        >>> sanitize_array(arr)
        array([1., 0., 0., 0., 2.])
    """
    if log_warning and not np.all(np.isfinite(arr)):
        logger.warning("Array contains NaN/Inf values, replacing with defaults")
    
    return np.nan_to_num(arr, copy=copy, nan=nan, posinf=posinf, neginf=neginf)


def sanitize_gradient(
    grad: np.ndarray,
    nan: float = GRADIENT_NAN_REPLACEMENT,
    posinf: float = GRADIENT_POSINF_REPLACEMENT,
    neginf: float = GRADIENT_NEGINF_REPLACEMENT,
    copy: bool = True,
    log_warning: bool = True
) -> np.ndarray:
    """
    Replace NaN and Inf values in a gradient array with large finite values.
    
    This pattern is used specifically for optimization gradients where replacing
    with zero would cause issues. Large values effectively penalize the direction.
    
    Args:
        grad: Input gradient array to sanitize.
        nan: Replacement value for NaN. Default is 1e10.
        posinf: Replacement value for positive infinity. Default is 1e10.
        neginf: Replacement value for negative infinity. Default is -1e10.
        copy: If True (default), returns a copy. If False, may modify in-place.
        log_warning: If True (default), logs a warning when NaN/Inf values are found.
        
    Returns:
        Sanitized gradient array.
        
    Examples:
        >>> grad = np.array([1.0, np.nan, np.inf])
        >>> sanitize_gradient(grad)
        array([1.e+00, 1.e+10, 1.e+10])
    """
    if log_warning and not np.all(np.isfinite(grad)):
        logger.warning("Gradient contains NaN/Inf values, replacing with large finite values")
    
    return np.nan_to_num(grad, copy=copy, nan=nan, posinf=posinf, neginf=neginf)


def has_nan(arr: np.ndarray) -> bool:
    """
    Check if array contains any NaN values.
    
    More efficient than `np.isnan(arr).sum() > 0` pattern used in some places.
    
    Args:
        arr: Input numpy array.
        
    Returns:
        True if array contains at least one NaN value.
        
    Examples:
        >>> has_nan(np.array([1.0, 2.0, 3.0]))
        False
        >>> has_nan(np.array([1.0, np.nan, 3.0]))
        True
    """
    return np.any(np.isnan(arr))


def has_inf(arr: np.ndarray) -> bool:
    """
    Check if array contains any infinite values.
    
    Args:
        arr: Input numpy array.
        
    Returns:
        True if array contains at least one infinite value.
        
    Examples:
        >>> has_inf(np.array([1.0, 2.0, 3.0]))
        False
        >>> has_inf(np.array([1.0, np.inf, 3.0]))
        True
    """
    return np.any(np.isinf(arr))


def is_finite(arr: np.ndarray) -> bool:
    """
    Check if all values in array are finite (not NaN and not Inf).
    
    This is a convenience wrapper around np.all(np.isfinite(arr)).
    
    Args:
        arr: Input numpy array.
        
    Returns:
        True if all values are finite.
        
    Examples:
        >>> is_finite(np.array([1.0, 2.0, 3.0]))
        True
        >>> is_finite(np.array([1.0, np.nan, 3.0]))
        False
        >>> is_finite(np.array([1.0, np.inf, 3.0]))
        False
    """
    return np.all(np.isfinite(arr))


def has_non_finite(arr: np.ndarray) -> bool:
    """
    Check if array contains any non-finite values (NaN or Inf).
    
    Equivalent to `not is_finite(arr)` but more readable in certain contexts.
    
    Args:
        arr: Input numpy array.
        
    Returns:
        True if array contains at least one NaN or Inf value.
        
    Examples:
        >>> has_non_finite(np.array([1.0, 2.0, 3.0]))
        False
        >>> has_non_finite(np.array([1.0, np.nan, 3.0]))
        True
    """
    return not np.all(np.isfinite(arr))


def count_non_finite(arr: np.ndarray) -> int:
    """
    Count the number of non-finite values in an array.
    
    Args:
        arr: Input numpy array.
        
    Returns:
        Number of NaN and Inf values in the array.
        
    Examples:
        >>> count_non_finite(np.array([1.0, np.nan, np.inf, 2.0]))
        2
    """
    return int(np.sum(~np.isfinite(arr)))


def safe_divide(
    numerator: Union[np.ndarray, float],
    denominator: Union[np.ndarray, float],
    default: float = 0.0
) -> Union[np.ndarray, float]:
    """
    Perform division with safe handling of division by zero.
    
    Args:
        numerator: Numerator value(s).
        denominator: Denominator value(s).
        default: Value to use when division by zero occurs. Default is 0.0.
        
    Returns:
        Result of division with zeros replaced by default value.
        
    Examples:
        >>> safe_divide(1.0, 0.0)
        0.0
        >>> safe_divide(np.array([1.0, 2.0]), np.array([2.0, 0.0]))
        array([0.5, 0. ])
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(numerator, denominator)
        if isinstance(result, np.ndarray):
            result = sanitize_array(result, nan=default, posinf=default, neginf=default)
        elif not np.isfinite(result):
            result = default
    return result
