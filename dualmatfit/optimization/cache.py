# -*- coding: utf-8 -*-
"""
Caching utilities for cost function optimization.

This module provides caching strategies for residuum, derivative, and volume
computations to avoid redundant calculations during optimization.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any

import numpy as np

__all__ = [
    'CostCache',
    'LimitedOrderedDict',
]


class LimitedOrderedDict(OrderedDict):
    """
    OrderedDict with a maximum capacity that removes oldest items when full.
    
    Parameters
    ----------
    capacity : int
        Maximum number of items to store.
    """
    
    def __init__(self, capacity: int):
        super().__init__()
        self._capacity = capacity

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        if len(self) > self._capacity:
            self.popitem(last=False)  # Remove the oldest item


class CostCache:
    """
    Cache manager for cost function computations.
    
    Provides separate caches for residuum, residuum derivatives, volume,
    and volume derivatives to avoid redundant computations during optimization.
    
    Parameters
    ----------
    cache_size : int, default=128
        Maximum number of entries in each cache.
        
    Attributes
    ----------
    residuum : LimitedOrderedDict
        Cache for residuum values.
    residuum_diff : LimitedOrderedDict
        Cache for residuum derivative values.
    volume : LimitedOrderedDict
        Cache for volume values.
    volume_diff : LimitedOrderedDict
        Cache for volume derivative values.
    
    Examples
    --------
    >>> cache = CostCache(cache_size=64)
    >>> xi = np.array([1.0, 2.0, 3.0])
    >>> key = cache.make_key(xi)
    >>> cache.residuum[key] = np.array([0.1, 0.2])
    >>> cache.get_residuum(xi)
    array([0.1, 0.2])
    """
    
    def __init__(self, cache_size: int = 128):
        self._cache_size = cache_size
        self._residuum = LimitedOrderedDict(cache_size)
        self._residuum_diff = LimitedOrderedDict(cache_size)
        self._volume = LimitedOrderedDict(cache_size)
        self._volume_diff = LimitedOrderedDict(cache_size)
    
    @property
    def cache_size(self) -> int:
        """Get the maximum cache size."""
        return self._cache_size
    
    @property
    def residuum(self) -> LimitedOrderedDict:
        """Access the residuum cache."""
        return self._residuum
    
    @property
    def residuum_diff(self) -> LimitedOrderedDict:
        """Access the residuum derivative cache."""
        return self._residuum_diff
    
    @property
    def volume(self) -> LimitedOrderedDict:
        """Access the volume cache."""
        return self._volume
    
    @property
    def volume_diff(self) -> LimitedOrderedDict:
        """Access the volume derivative cache."""
        return self._volume_diff
    
    @staticmethod
    def make_key(xi: np.ndarray) -> Tuple[float, ...]:
        """
        Create a hashable cache key from a numpy array.
        
        Parameters
        ----------
        xi : np.ndarray
            Parameter values array.
            
        Returns
        -------
        tuple of float
            Hashable tuple representation of the array.
        """
        return tuple(xi.flatten())
    
    def get_residuum(self, xi: np.ndarray) -> Optional[np.ndarray]:
        """
        Get cached residuum value if available.
        
        Parameters
        ----------
        xi : np.ndarray
            Parameter values.
            
        Returns
        -------
        np.ndarray or None
            Cached residuum array or None if not cached.
        """
        key = self.make_key(xi)
        if key in self._residuum:
            return self._residuum[key].copy()
        return None
    
    def set_residuum(self, xi: np.ndarray, value: np.ndarray) -> None:
        """
        Store residuum value in cache.
        
        Parameters
        ----------
        xi : np.ndarray
            Parameter values.
        value : np.ndarray
            Residuum value to cache.
        """
        key = self.make_key(xi)
        self._residuum[key] = value.copy()
    
    def get_residuum_diff(self, xi: np.ndarray) -> Optional[np.ndarray]:
        """
        Get cached residuum derivative if available.
        
        Parameters
        ----------
        xi : np.ndarray
            Parameter values.
            
        Returns
        -------
        np.ndarray or None
            Cached residuum derivative or None if not cached.
        """
        key = self.make_key(xi)
        if key in self._residuum_diff:
            return self._residuum_diff[key].copy()
        return None
    
    def set_residuum_diff(self, xi: np.ndarray, value: np.ndarray) -> None:
        """
        Store residuum derivative in cache.
        
        Parameters
        ----------
        xi : np.ndarray
            Parameter values.
        value : np.ndarray
            Residuum derivative to cache.
        """
        key = self.make_key(xi)
        self._residuum_diff[key] = value.copy()
    
    def get_volume(self, xi: np.ndarray) -> Optional[float]:
        """
        Get cached volume value if available.
        
        Parameters
        ----------
        xi : np.ndarray
            Parameter values.
            
        Returns
        -------
        float or None
            Cached volume value or None if not cached.
        """
        key = self.make_key(xi)
        if key in self._volume:
            return self._volume[key]
        return None
    
    def set_volume(self, xi: np.ndarray, value: float) -> None:
        """
        Store volume value in cache.
        
        Parameters
        ----------
        xi : np.ndarray
            Parameter values.
        value : float
            Volume value to cache.
        """
        key = self.make_key(xi)
        self._volume[key] = value
    
    def get_volume_diff(self, xi: np.ndarray) -> Optional[np.ndarray]:
        """
        Get cached volume derivative if available.
        
        Parameters
        ----------
        xi : np.ndarray
            Parameter values.
            
        Returns
        -------
        np.ndarray or None
            Cached volume derivative or None if not cached.
        """
        key = self.make_key(xi)
        if key in self._volume_diff:
            return self._volume_diff[key].copy()
        return None
    
    def set_volume_diff(self, xi: np.ndarray, value: np.ndarray) -> None:
        """
        Store volume derivative in cache.
        
        Parameters
        ----------
        xi : np.ndarray
            Parameter values.
        value : np.ndarray
            Volume derivative to cache.
        """
        key = self.make_key(xi)
        self._volume_diff[key] = value.copy()
    
    def clear(self) -> None:
        """Clear all caches."""
        self._residuum.clear()
        self._residuum_diff.clear()
        self._volume.clear()
        self._volume_diff.clear()
    
    def clear_residuum(self) -> None:
        """Clear only residuum caches."""
        self._residuum.clear()
        self._residuum_diff.clear()
    
    def clear_volume(self) -> None:
        """Clear only volume caches."""
        self._volume.clear()
        self._volume_diff.clear()
    
    def stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns
        -------
        dict
            Dictionary with cache sizes.
        """
        return {
            'residuum_entries': len(self._residuum),
            'residuum_diff_entries': len(self._residuum_diff),
            'volume_entries': len(self._volume),
            'volume_diff_entries': len(self._volume_diff),
            'max_capacity': self._cache_size,
        }
    
    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"CostCache(capacity={stats['max_capacity']}, "
            f"residuum={stats['residuum_entries']}, "
            f"residuum_diff={stats['residuum_diff_entries']}, "
            f"volume={stats['volume_entries']}, "
            f"volume_diff={stats['volume_diff_entries']})"
        )
