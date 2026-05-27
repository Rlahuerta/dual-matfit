# -*- coding: utf-8 -*-
"""
Kreisselmeier-Steinhauser (KS) aggregation functions for constraint handling.

Provides smooth approximations of min/max operations for gradient-based
optimization of constrained problems.
"""
import numpy as np

__all__ = [
    'min_ks',
    'max_ks',
]


def min_ks(g: np.ndarray, rho: float = 10.0) -> float:
    """
    Kreisselmeier-Steinhauser (KS) smooth approximation of the minimum.

    Parameters
    ----------
    g : np.ndarray
        Array of constraint values.
    rho : float, optional
        Aggregation parameter (default 10.0). Higher values give tighter
        approximation to the true minimum.

    Returns
    -------
    float
        KS approximation of min(g).
    """
    g_min = np.min(g)
    return g_min - (1.0 / rho) * np.log(np.sum(np.exp(-rho * (g - g_min))))


def max_ks(g: np.ndarray, rho: float = 10.0) -> float:
    """
    Kreisselmeier-Steinhauser (KS) smooth approximation of the maximum.

    Parameters
    ----------
    g : np.ndarray
        Array of constraint values.
    rho : float, optional
        Aggregation parameter (default 10.0). Higher values give tighter
        approximation to the true maximum.

    Returns
    -------
    float
        KS approximation of max(g).
    """
    g_max = np.max(g)
    return g_max + (1.0 / rho) * np.log(np.sum(np.exp(rho * (g - g_max))))