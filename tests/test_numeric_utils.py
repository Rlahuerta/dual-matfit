# -*- coding: utf-8 -*-
"""Tests for numeric_utils module."""

import pytest
import numpy as np

from dualmatfit.numeric_utils import (
    sanitize_array,
    sanitize_gradient,
    has_nan,
    has_inf,
    is_finite,
    has_non_finite,
    count_non_finite,
    safe_divide,
    DEFAULT_NAN_REPLACEMENT,
    DEFAULT_POSINF_REPLACEMENT,
    DEFAULT_NEGINF_REPLACEMENT,
    GRADIENT_NAN_REPLACEMENT,
    GRADIENT_POSINF_REPLACEMENT,
    GRADIENT_NEGINF_REPLACEMENT,
)


class TestSanitizeArray:
    """Tests for sanitize_array function."""
    
    def test_no_nan_inf(self):
        """Array with no NaN/Inf should remain unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        result = sanitize_array(arr)
        np.testing.assert_array_equal(result, arr)
    
    def test_nan_replacement(self):
        """NaN values should be replaced with default (0.0)."""
        arr = np.array([1.0, np.nan, 3.0])
        result = sanitize_array(arr)
        expected = np.array([1.0, 0.0, 3.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_posinf_replacement(self):
        """Positive infinity should be replaced with default (0.0)."""
        arr = np.array([1.0, np.inf, 3.0])
        result = sanitize_array(arr)
        expected = np.array([1.0, 0.0, 3.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_neginf_replacement(self):
        """Negative infinity should be replaced with default (0.0)."""
        arr = np.array([1.0, -np.inf, 3.0])
        result = sanitize_array(arr)
        expected = np.array([1.0, 0.0, 3.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_custom_replacement_values(self):
        """Custom replacement values should be used."""
        arr = np.array([1.0, np.nan, np.inf, -np.inf])
        result = sanitize_array(arr, nan=99.0, posinf=88.0, neginf=-88.0)
        expected = np.array([1.0, 99.0, 88.0, -88.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_copy_behavior(self):
        """Original array should not be modified when copy=True."""
        arr = np.array([1.0, np.nan, 3.0])
        result = sanitize_array(arr, copy=True)
        assert np.isnan(arr[1])  # Original still has NaN
        assert result[1] == 0.0  # Result has replacement
    
    def test_2d_array(self):
        """Should work with 2D arrays."""
        arr = np.array([[1.0, np.nan], [np.inf, 2.0]])
        result = sanitize_array(arr)
        expected = np.array([[1.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_equal(result, expected)


class TestSanitizeGradient:
    """Tests for sanitize_gradient function."""
    
    def test_no_nan_inf(self):
        """Gradient with no NaN/Inf should remain unchanged."""
        grad = np.array([1.0, 2.0, 3.0])
        result = sanitize_gradient(grad, log_warning=False)
        np.testing.assert_array_equal(result, grad)
    
    def test_nan_replacement(self):
        """NaN values should be replaced with 1e10."""
        grad = np.array([1.0, np.nan, 3.0])
        result = sanitize_gradient(grad, log_warning=False)
        expected = np.array([1.0, 1e10, 3.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_posinf_replacement(self):
        """Positive infinity should be replaced with 1e10."""
        grad = np.array([1.0, np.inf, 3.0])
        result = sanitize_gradient(grad, log_warning=False)
        expected = np.array([1.0, 1e10, 3.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_neginf_replacement(self):
        """Negative infinity should be replaced with -1e10."""
        grad = np.array([1.0, -np.inf, 3.0])
        result = sanitize_gradient(grad, log_warning=False)
        expected = np.array([1.0, -1e10, 3.0])
        np.testing.assert_array_equal(result, expected)


class TestHasNan:
    """Tests for has_nan function."""
    
    def test_no_nan(self):
        """Array without NaN should return False."""
        assert not has_nan(np.array([1.0, 2.0, 3.0]))
    
    def test_with_nan(self):
        """Array with NaN should return True."""
        assert has_nan(np.array([1.0, np.nan, 3.0]))
    
    def test_with_inf_no_nan(self):
        """Array with Inf but no NaN should return False."""
        assert not has_nan(np.array([1.0, np.inf, 3.0]))
    
    def test_2d_array(self):
        """Should work with 2D arrays."""
        assert has_nan(np.array([[1.0, np.nan], [2.0, 3.0]]))
        assert not has_nan(np.array([[1.0, 2.0], [3.0, 4.0]]))


class TestHasInf:
    """Tests for has_inf function."""
    
    def test_no_inf(self):
        """Array without Inf should return False."""
        assert not has_inf(np.array([1.0, 2.0, 3.0]))
    
    def test_with_posinf(self):
        """Array with positive Inf should return True."""
        assert has_inf(np.array([1.0, np.inf, 3.0]))
    
    def test_with_neginf(self):
        """Array with negative Inf should return True."""
        assert has_inf(np.array([1.0, -np.inf, 3.0]))
    
    def test_with_nan_no_inf(self):
        """Array with NaN but no Inf should return False."""
        assert not has_inf(np.array([1.0, np.nan, 3.0]))


class TestIsFinite:
    """Tests for is_finite function."""
    
    def test_all_finite(self):
        """Array with all finite values should return True."""
        assert is_finite(np.array([1.0, 2.0, 3.0]))
    
    def test_with_nan(self):
        """Array with NaN should return False."""
        assert not is_finite(np.array([1.0, np.nan, 3.0]))
    
    def test_with_inf(self):
        """Array with Inf should return False."""
        assert not is_finite(np.array([1.0, np.inf, 3.0]))


class TestHasNonFinite:
    """Tests for has_non_finite function."""
    
    def test_all_finite(self):
        """Array with all finite values should return False."""
        assert not has_non_finite(np.array([1.0, 2.0, 3.0]))
    
    def test_with_nan(self):
        """Array with NaN should return True."""
        assert has_non_finite(np.array([1.0, np.nan, 3.0]))
    
    def test_with_inf(self):
        """Array with Inf should return True."""
        assert has_non_finite(np.array([1.0, np.inf, 3.0]))


class TestCountNonFinite:
    """Tests for count_non_finite function."""
    
    def test_no_non_finite(self):
        """Array with no non-finite values should return 0."""
        assert count_non_finite(np.array([1.0, 2.0, 3.0])) == 0
    
    def test_one_nan(self):
        """Array with one NaN should return 1."""
        assert count_non_finite(np.array([1.0, np.nan, 3.0])) == 1
    
    def test_multiple_non_finite(self):
        """Array with multiple non-finite values should return correct count."""
        arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
        assert count_non_finite(arr) == 3


class TestSafeDivide:
    """Tests for safe_divide function."""
    
    def test_normal_division(self):
        """Normal division should work as expected."""
        result = safe_divide(4.0, 2.0)
        assert result == 2.0
    
    def test_division_by_zero_scalar(self):
        """Division by zero should return default."""
        result = safe_divide(1.0, 0.0)
        assert result == 0.0
    
    def test_division_by_zero_array(self):
        """Division by zero in array should return default for those elements."""
        result = safe_divide(np.array([1.0, 2.0]), np.array([2.0, 0.0]))
        np.testing.assert_array_equal(result, np.array([0.5, 0.0]))
    
    def test_custom_default(self):
        """Custom default value should be used."""
        result = safe_divide(1.0, 0.0, default=-1.0)
        assert result == -1.0


class TestConstants:
    """Tests for module constants."""
    
    def test_default_replacement_values(self):
        """Default replacement values should be 0.0."""
        assert DEFAULT_NAN_REPLACEMENT == 0.0
        assert DEFAULT_POSINF_REPLACEMENT == 0.0
        assert DEFAULT_NEGINF_REPLACEMENT == 0.0
    
    def test_gradient_replacement_values(self):
        """Gradient replacement values should be large finite values."""
        assert GRADIENT_NAN_REPLACEMENT == 1e10
        assert GRADIENT_POSINF_REPLACEMENT == 1e10
        assert GRADIENT_NEGINF_REPLACEMENT == -1e10
