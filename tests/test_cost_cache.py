# -*- coding: utf-8 -*-
"""
Tests for cost_cache.py - CostCache and LimitedOrderedDict classes.
"""
import pytest
import numpy as np

from dualmatfit.optimization.cache import CostCache, LimitedOrderedDict


class TestLimitedOrderedDict:
    """Tests for LimitedOrderedDict capacity-limited dictionary."""

    def test_init_with_capacity(self):
        """Test initialization with specified capacity."""
        d = LimitedOrderedDict(5)
        assert d._capacity == 5
        assert len(d) == 0

    def test_items_within_capacity(self):
        """Test adding items within capacity limit."""
        d = LimitedOrderedDict(3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        assert len(d) == 3
        assert list(d.keys()) == ["a", "b", "c"]

    def test_oldest_removed_when_exceeds_capacity(self):
        """Test that oldest item is removed when capacity exceeded."""
        d = LimitedOrderedDict(3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        d["d"] = 4  # Should remove "a"
        
        assert len(d) == 3
        assert "a" not in d
        assert list(d.keys()) == ["b", "c", "d"]

    def test_order_preserved(self):
        """Test that insertion order is preserved."""
        d = LimitedOrderedDict(5)
        for i in range(5):
            d[f"key_{i}"] = i
        
        assert list(d.keys()) == [f"key_{i}" for i in range(5)]


class TestCostCache:
    """Tests for CostCache utility class."""

    def test_init_default_capacity(self):
        """Test initialization with default capacity."""
        cache = CostCache()
        assert cache.cache_size == 128

    def test_init_custom_capacity(self):
        """Test initialization with custom capacity."""
        cache = CostCache(cache_size=64)
        assert cache.cache_size == 64

    def test_make_key(self):
        """Test key generation from numpy array."""
        xi = np.array([1.0, 2.0, 3.0])
        key = CostCache.make_key(xi)
        assert key == (1.0, 2.0, 3.0)
        assert isinstance(key, tuple)

    def test_residuum_cache_set_get(self):
        """Test residuum cache set and get operations."""
        cache = CostCache(cache_size=10)
        xi = np.array([1.0, 2.0])
        residuum = np.array([0.1, 0.2, 0.3])
        
        # Initially empty
        assert cache.get_residuum(xi) is None
        
        # Set and get
        cache.set_residuum(xi, residuum)
        result = cache.get_residuum(xi)
        
        np.testing.assert_array_equal(result, residuum)

    def test_residuum_diff_cache_set_get(self):
        """Test residuum derivative cache operations."""
        cache = CostCache(cache_size=10)
        xi = np.array([1.0, 2.0])
        residuum_diff = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        cache.set_residuum_diff(xi, residuum_diff)
        result = cache.get_residuum_diff(xi)
        
        np.testing.assert_array_equal(result, residuum_diff)

    def test_volume_cache_set_get(self):
        """Test volume cache operations."""
        cache = CostCache(cache_size=10)
        xi = np.array([1.0, 2.0])
        volume = 1.5
        
        cache.set_volume(xi, volume)
        result = cache.get_volume(xi)
        
        assert result == volume

    def test_volume_diff_cache_set_get(self):
        """Test volume derivative cache operations."""
        cache = CostCache(cache_size=10)
        xi = np.array([1.0, 2.0])
        volume_diff = np.array([0.5, 0.6])
        
        cache.set_volume_diff(xi, volume_diff)
        result = cache.get_volume_diff(xi)
        
        np.testing.assert_array_equal(result, volume_diff)

    def test_clear_all(self):
        """Test clearing all caches."""
        cache = CostCache(cache_size=10)
        xi = np.array([1.0, 2.0])
        
        cache.set_residuum(xi, np.array([0.1]))
        cache.set_residuum_diff(xi, np.array([[0.1]]))
        cache.set_volume(xi, 1.0)
        cache.set_volume_diff(xi, np.array([0.1]))
        
        cache.clear()
        
        assert cache.get_residuum(xi) is None
        assert cache.get_residuum_diff(xi) is None
        assert cache.get_volume(xi) is None
        assert cache.get_volume_diff(xi) is None

    def test_clear_residuum(self):
        """Test clearing only residuum caches."""
        cache = CostCache(cache_size=10)
        xi = np.array([1.0, 2.0])
        
        cache.set_residuum(xi, np.array([0.1]))
        cache.set_volume(xi, 1.0)
        
        cache.clear_residuum()
        
        assert cache.get_residuum(xi) is None
        assert cache.get_volume(xi) == 1.0

    def test_clear_volume(self):
        """Test clearing only volume caches."""
        cache = CostCache(cache_size=10)
        xi = np.array([1.0, 2.0])
        
        cache.set_residuum(xi, np.array([0.1]))
        cache.set_volume(xi, 1.0)
        
        cache.clear_volume()
        
        assert cache.get_residuum(xi) is not None
        assert cache.get_volume(xi) is None

    def test_stats(self):
        """Test cache statistics."""
        cache = CostCache(cache_size=10)
        
        stats = cache.stats()
        assert stats['max_capacity'] == 10
        assert stats['residuum_entries'] == 0
        
        cache.set_residuum(np.array([1.0]), np.array([0.1]))
        cache.set_residuum(np.array([2.0]), np.array([0.2]))
        
        stats = cache.stats()
        assert stats['residuum_entries'] == 2

    def test_repr(self):
        """Test string representation."""
        cache = CostCache(cache_size=10)
        repr_str = repr(cache)
        
        assert "CostCache" in repr_str
        assert "capacity=10" in repr_str

    def test_cached_values_are_copies(self):
        """Test that cached values are copies, not references."""
        cache = CostCache(cache_size=10)
        xi = np.array([1.0, 2.0])
        original = np.array([0.1, 0.2, 0.3])
        
        cache.set_residuum(xi, original)
        
        # Modify the original
        original[0] = 999.0
        
        # Cached value should be unaffected
        result = cache.get_residuum(xi)
        assert result[0] == 0.1

    def test_get_returns_copies(self):
        """Test that get returns copies, not references."""
        cache = CostCache(cache_size=10)
        xi = np.array([1.0, 2.0])
        cache.set_residuum(xi, np.array([0.1, 0.2, 0.3]))
        
        result1 = cache.get_residuum(xi)
        result1[0] = 999.0
        
        # Second get should be unaffected
        result2 = cache.get_residuum(xi)
        assert result2[0] == 0.1
