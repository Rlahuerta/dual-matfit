# -*- coding: utf-8 -*-
"""
Regularization components for cost function optimization.

This module provides regularization strategies (L2/Tikhonov and volume-based)
extracted from CostIntegrator to follow the Single Responsibility Principle.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Callable, List, TYPE_CHECKING

import numpy as np

from dualmatfit.solvers.derivative import _fdm
from dualmatfit.utils.numeric import sanitize_array

if TYPE_CHECKING:
    from dualmatfit.optimization.cost import CostFunction, LSQFit

__all__ = [
    'RegularizationStrategy',
    'L2Regularization',
    'VolumeRegularization',
    'CompositeRegularization',
    '_inv_weight',
    '_rsc_weight',
]


def _inv_weight(dxi: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """
    Compute inverse weighting for regularization (rsc2).
    
    Parameters
    ----------
    dxi : np.ndarray
        Difference vector (xi - xi_ref).
    beta : float, default=2.0
        Weighting exponent.
        
    Returns
    -------
    np.ndarray
        Normalized inverse weights.
    """
    np_ws = 1.0 / (1.0 + np.abs(dxi) ** beta)
    np_ws /= np.sum(np_ws)
    return np_ws


def _rsc_weight(dxi: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """
    Compute direct rescaling weight for regularization (rsc3).
    
    Parameters
    ----------
    dxi : np.ndarray
        Difference vector (xi - xi_ref).
    beta : float, default=2.0
        Weighting exponent.
        
    Returns
    -------
    np.ndarray
        Normalized rescaled weights.
    """
    np_ws = 1.0 + np.abs(dxi) ** beta
    np_ws /= np.sum(np_ws)
    return np_ws


class RegularizationStrategy(ABC):
    """Abstract base class for regularization strategies."""
    
    @abstractmethod
    def value(self, xi: np.ndarray) -> float:
        """Compute the regularization term value."""
        pass
    
    @abstractmethod
    def gradient(self, xi: np.ndarray, fdm: bool = False, **kwargs) -> np.ndarray:
        """Compute the gradient of the regularization term."""
        pass


class L2Regularization(RegularizationStrategy):
    """
    L2 (Tikhonov) regularization: alpha/2 * ||xi - xi_ref||^2.
    
    Supports optional rescaling weights for adaptive regularization.
    
    Parameters
    ----------
    xi_ref : np.ndarray
        Reference parameter values to regularize towards.
    alpha : float, default=0.0
        Regularization strength.
    rescale : str, optional
        Rescaling method ('direct', 'inverse', or None).
    beta : float, default=2.0
        Exponent for rescaling weights.
    xi_bounds : list, optional
        Parameter bounds for FDM fallback.
    multi_objective : bool, default=False
        If True, regularize towards zero instead of xi_ref.
    """
    
    def __init__(
        self,
        xi_ref: np.ndarray,
        alpha: float = 0.0,
        rescale: Optional[str] = None,
        beta: float = 2.0,
        xi_bounds: Optional[List] = None,
        multi_objective: bool = False,
    ):
        self._xi_ref = xi_ref.copy()
        self._alpha = alpha if alpha is not None else 0.0
        self._rescale = rescale
        self._beta = beta
        self._xi_bounds = xi_bounds
        self._multi_objective = multi_objective
    
    @property
    def alpha(self) -> float:
        """Get the regularization strength."""
        return self._alpha
    
    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set the regularization strength."""
        self._alpha = value if value is not None else 0.0
    
    def _compute_weights(self, dxi: np.ndarray) -> np.ndarray:
        """Compute rescaling weights based on the configured method."""
        if self._rescale == "direct":
            return _rsc_weight(dxi, self._beta)
        elif self._rescale == "inverse":
            return _inv_weight(dxi, self._beta)
        else:
            return np.ones_like(dxi)
    
    def _compute_delta(self, xi: np.ndarray) -> np.ndarray:
        """Compute the difference vector for regularization."""
        if self._multi_objective:
            return xi - np.zeros_like(self._xi_ref)
        return xi - self._xi_ref
    
    def value(self, xi: np.ndarray) -> float:
        """
        Compute the L2 regularization term value.
        
        Parameters
        ----------
        xi : np.ndarray
            Current parameter values.
            
        Returns
        -------
        float
            Regularization term: alpha/2 * ||W @ (xi - xi_ref)||^2.
        """
        if self._alpha <= 0.0:
            return 0.0
        
        dxi = self._compute_delta(xi)
        weights = self._compute_weights(dxi)
        weighted_delta = weights * dxi
        
        return self._alpha * 0.5 * np.dot(weighted_delta, weighted_delta)
    
    def gradient(self, xi: np.ndarray, fdm: bool = False, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the L2 regularization term.
        
        Parameters
        ----------
        xi : np.ndarray
            Current parameter values.
        fdm : bool, default=False
            If True, use finite difference method.
        **kwargs
            Additional arguments for FDM.
            
        Returns
        -------
        np.ndarray
            Gradient of the regularization term.
        """
        if self._alpha <= 0.0:
            return np.zeros_like(xi)
        
        if fdm:
            return self._alpha * _fdm(
                lambda x: self.value(x) / self._alpha,
                xi,
                xi_bounds=self._xi_bounds,
                **kwargs
            )
        
        dxi = self._compute_delta(xi)
        weights = self._compute_weights(dxi)
        
        return self._alpha * weights * dxi


class VolumeRegularization(RegularizationStrategy):
    """
    Volume-based regularization using strain energy volume terms.
    
    Aggregates volume regularization terms from multiple cost functions.
    
    Parameters
    ----------
    cost_functions : list
        List of CostFunction objects that provide volume methods.
    epsilon : float, default=0.0
        Regularization strength.
    xi_bounds : list, optional
        Parameter bounds for FDM fallback.
    cache : CostCache, optional
        Cache for storing computed values.
    """
    
    def __init__(
        self,
        cost_functions: List["CostFunction"],
        epsilon: float = 0.0,
        xi_bounds: Optional[List] = None,
        cache: Optional["CostCache"] = None,
    ):
        self._cost_functions = cost_functions
        self._epsilon = epsilon if epsilon is not None else 0.0
        self._xi_bounds = xi_bounds
        self._cache = cache
    
    @property
    def epsilon(self) -> float:
        """Get the regularization strength."""
        return self._epsilon
    
    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """Set the regularization strength."""
        self._epsilon = value if value is not None else 0.0
    
    def value(self, xi: np.ndarray) -> float:
        """
        Compute the volume regularization term value.
        
        Parameters
        ----------
        xi : np.ndarray
            Current parameter values.
            
        Returns
        -------
        float
            Sum of volume terms from all cost functions, scaled by epsilon.
        """
        if self._epsilon <= 0.0:
            return 0.0
        
        cache_key = tuple(xi.flatten())
        
        # Check cache
        if self._cache is not None and cache_key in self._cache.volume:
            return self._epsilon * self._cache.volume[cache_key]
        
        # Compute volume from all functions
        vol_sum = 0.0
        for fun_i in self._cost_functions:
            if hasattr(fun_i, 'volume'):
                vol_i = sanitize_array(fun_i.volume(xi))
                vol_sum += np.sum(vol_i)
        
        # Cache result
        if self._cache is not None:
            self._cache.volume[cache_key] = vol_sum
        
        return self._epsilon * vol_sum
    
    def gradient(self, xi: np.ndarray, fdm: bool = False, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the volume regularization term.
        
        Parameters
        ----------
        xi : np.ndarray
            Current parameter values.
        fdm : bool, default=False
            If True, use finite difference method.
        **kwargs
            Additional arguments for FDM.
            
        Returns
        -------
        np.ndarray
            Gradient of the volume regularization term.
        """
        if self._epsilon <= 0.0:
            return np.zeros_like(xi)
        
        if fdm:
            return _fdm(self.value, xi, xi_bounds=self._xi_bounds, **kwargs)
        
        cache_key = tuple(xi.flatten())
        
        # Check cache
        if self._cache is not None and cache_key in self._cache.volume_diff:
            return self._epsilon * self._cache.volume_diff[cache_key]
        
        # Compute volume derivative from all functions
        vol_diff_list = []
        for fun_i in self._cost_functions:
            if hasattr(fun_i, 'volume_diff'):
                vol_diff_i = fun_i.volume_diff(xi, fdm=fdm, **kwargs)
                vol_diff_list.append(sanitize_array(vol_diff_i))
        
        if vol_diff_list:
            vol_diff_sum = np.asarray(vol_diff_list, dtype=float).sum(axis=(0, 1))
        else:
            vol_diff_sum = np.zeros_like(xi)
        
        # Cache result
        if self._cache is not None:
            self._cache.volume_diff[cache_key] = vol_diff_sum.copy()
        
        return self._epsilon * vol_diff_sum


class CompositeRegularization(RegularizationStrategy):
    """
    Combines multiple regularization strategies.
    
    Parameters
    ----------
    strategies : list of RegularizationStrategy
        List of regularization strategies to combine.
    """
    
    def __init__(self, strategies: Optional[List[RegularizationStrategy]] = None):
        self._strategies = strategies if strategies is not None else []
    
    def add_strategy(self, strategy: RegularizationStrategy) -> None:
        """Add a regularization strategy to the composite."""
        self._strategies.append(strategy)
    
    def value(self, xi: np.ndarray) -> float:
        """
        Compute the combined regularization term value.
        
        Parameters
        ----------
        xi : np.ndarray
            Current parameter values.
            
        Returns
        -------
        float
            Sum of all regularization term values.
        """
        return sum(s.value(xi) for s in self._strategies)
    
    def gradient(self, xi: np.ndarray, fdm: bool = False, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the combined regularization terms.
        
        Parameters
        ----------
        xi : np.ndarray
            Current parameter values.
        fdm : bool, default=False
            If True, use finite difference method.
        **kwargs
            Additional arguments for FDM.
            
        Returns
        -------
        np.ndarray
            Sum of all regularization gradients.
        """
        if not self._strategies:
            return np.zeros_like(xi)
        
        return sum(s.gradient(xi, fdm=fdm, **kwargs) for s in self._strategies)
