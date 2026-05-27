# -*- coding: utf-8 -*-
"""
Tests for regularization.py - Regularization strategy classes.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from dualmatfit.optimization.regularization import (
    _inv_weight,
    _rsc_weight,
    L2Regularization,
    VolumeRegularization,
    CompositeRegularization,
)
from dualmatfit.optimization.cache import CostCache


class TestWeightFunctions:
    """Tests for weight computation functions."""

    def test_inv_weight_basic(self):
        """Test inverse weight computation."""
        dxi = np.array([1.0, 2.0, 3.0])
        weights = _inv_weight(dxi, beta=2.0)
        
        # Weights should sum to 1
        assert np.isclose(np.sum(weights), 1.0)
        
        # Larger values should have smaller weights
        assert weights[0] > weights[1] > weights[2]

    def test_inv_weight_zero_input(self):
        """Test inverse weight with zero input."""
        dxi = np.array([0.0, 0.0])
        weights = _inv_weight(dxi, beta=2.0)
        
        # Should be equal weights
        np.testing.assert_array_almost_equal(weights, [0.5, 0.5])

    def test_rsc_weight_basic(self):
        """Test rescaling weight computation."""
        dxi = np.array([1.0, 2.0, 3.0])
        weights = _rsc_weight(dxi, beta=2.0)
        
        # Weights should sum to 1
        assert np.isclose(np.sum(weights), 1.0)
        
        # Larger values should have larger weights (opposite of inv_weight)
        assert weights[0] < weights[1] < weights[2]

    def test_rsc_weight_zero_input(self):
        """Test rescaling weight with zero input."""
        dxi = np.array([0.0, 0.0])
        weights = _rsc_weight(dxi, beta=2.0)
        
        # Should be equal weights
        np.testing.assert_array_almost_equal(weights, [0.5, 0.5])


class TestL2Regularization:
    """Tests for L2 (Tikhonov) regularization."""

    def test_init(self):
        """Test initialization."""
        xi_ref = np.array([1.0, 2.0, 3.0])
        reg = L2Regularization(xi_ref=xi_ref, alpha=0.1)
        
        assert reg.alpha == 0.1
        np.testing.assert_array_equal(reg._xi_ref, xi_ref)

    def test_value_zero_alpha(self):
        """Test value returns 0 when alpha is 0."""
        xi_ref = np.array([1.0, 2.0])
        reg = L2Regularization(xi_ref=xi_ref, alpha=0.0)
        
        xi = np.array([5.0, 10.0])
        assert reg.value(xi) == 0.0

    def test_value_at_reference(self):
        """Test value is 0 when xi equals reference."""
        xi_ref = np.array([1.0, 2.0])
        reg = L2Regularization(xi_ref=xi_ref, alpha=0.1)
        
        assert reg.value(xi_ref) == 0.0

    def test_value_positive(self):
        """Test value is positive when xi differs from reference."""
        xi_ref = np.array([1.0, 2.0])
        reg = L2Regularization(xi_ref=xi_ref, alpha=1.0)
        
        xi = np.array([2.0, 3.0])
        # Without rescaling: 0.5 * alpha * ||xi - xi_ref||^2 = 0.5 * 1.0 * 2 = 1.0
        assert reg.value(xi) == pytest.approx(1.0)

    def test_gradient_zero_alpha(self):
        """Test gradient returns zeros when alpha is 0."""
        xi_ref = np.array([1.0, 2.0])
        reg = L2Regularization(xi_ref=xi_ref, alpha=0.0)
        
        xi = np.array([5.0, 10.0])
        grad = reg.gradient(xi)
        
        np.testing.assert_array_equal(grad, np.zeros(2))

    def test_gradient_at_reference(self):
        """Test gradient is 0 when xi equals reference."""
        xi_ref = np.array([1.0, 2.0])
        reg = L2Regularization(xi_ref=xi_ref, alpha=0.1)
        
        grad = reg.gradient(xi_ref)
        np.testing.assert_array_almost_equal(grad, np.zeros(2))

    def test_gradient_direction(self):
        """Test gradient points towards reference."""
        xi_ref = np.array([0.0, 0.0])
        reg = L2Regularization(xi_ref=xi_ref, alpha=1.0)
        
        xi = np.array([1.0, 1.0])
        grad = reg.gradient(xi)
        
        # Gradient should be alpha * (xi - xi_ref) = [1, 1]
        np.testing.assert_array_almost_equal(grad, [1.0, 1.0])

    def test_alpha_setter(self):
        """Test alpha property setter."""
        reg = L2Regularization(xi_ref=np.array([1.0]), alpha=0.1)
        
        reg.alpha = 0.5
        assert reg.alpha == 0.5
        
        reg.alpha = None
        assert reg.alpha == 0.0

    def test_multi_objective_regularizes_to_zero(self):
        """Test multi-objective mode regularizes towards zero."""
        xi_ref = np.array([1.0, 2.0])
        reg = L2Regularization(xi_ref=xi_ref, alpha=1.0, multi_objective=True)
        
        xi = np.array([1.0, 2.0])
        # Should not be zero since we regularize towards zero, not xi_ref
        assert reg.value(xi) > 0.0

    def test_rescale_direct(self):
        """Test direct rescaling option."""
        xi_ref = np.array([0.0, 0.0])
        reg = L2Regularization(xi_ref=xi_ref, alpha=1.0, rescale="direct")
        
        xi = np.array([1.0, 2.0])
        value = reg.value(xi)
        
        # Value should be different from no rescaling
        reg_no_scale = L2Regularization(xi_ref=xi_ref, alpha=1.0, rescale=None)
        value_no_scale = reg_no_scale.value(xi)
        
        # Both should be positive but may differ due to rescaling
        assert value > 0
        assert value_no_scale > 0

    def test_rescale_inverse(self):
        """Test inverse rescaling option."""
        xi_ref = np.array([0.0, 0.0])
        reg = L2Regularization(xi_ref=xi_ref, alpha=1.0, rescale="inverse")
        
        xi = np.array([1.0, 2.0])
        value = reg.value(xi)
        
        assert value > 0


class TestVolumeRegularization:
    """Tests for volume-based regularization."""

    def test_init(self):
        """Test initialization."""
        mock_cost_fn = Mock()
        reg = VolumeRegularization(cost_functions=[mock_cost_fn], epsilon=0.1)
        
        assert reg.epsilon == 0.1

    def test_value_zero_epsilon(self):
        """Test value returns 0 when epsilon is 0."""
        mock_cost_fn = Mock()
        mock_cost_fn.volume = Mock(return_value=np.array([10.0]))
        
        reg = VolumeRegularization(cost_functions=[mock_cost_fn], epsilon=0.0)
        
        xi = np.array([1.0])
        assert reg.value(xi) == 0.0
        mock_cost_fn.volume.assert_not_called()

    def test_value_aggregates_volumes(self):
        """Test value aggregates volume from all cost functions."""
        mock1 = Mock()
        mock1.volume = Mock(return_value=np.array([1.0, 2.0]))
        
        mock2 = Mock()
        mock2.volume = Mock(return_value=np.array([3.0, 4.0]))
        
        reg = VolumeRegularization(cost_functions=[mock1, mock2], epsilon=1.0)
        
        xi = np.array([1.0])
        value = reg.value(xi)
        
        # Sum of all volumes: 1 + 2 + 3 + 4 = 10
        assert value == pytest.approx(10.0)

    def test_gradient_zero_epsilon(self):
        """Test gradient returns zeros when epsilon is 0."""
        mock_cost_fn = Mock()
        
        reg = VolumeRegularization(cost_functions=[mock_cost_fn], epsilon=0.0)
        
        xi = np.array([1.0, 2.0])
        grad = reg.gradient(xi)
        
        np.testing.assert_array_equal(grad, np.zeros(2))

    def test_epsilon_setter(self):
        """Test epsilon property setter."""
        reg = VolumeRegularization(cost_functions=[], epsilon=0.1)
        
        reg.epsilon = 0.5
        assert reg.epsilon == 0.5
        
        reg.epsilon = None
        assert reg.epsilon == 0.0

    def test_with_cache(self):
        """Test volume regularization with cache."""
        mock_cost_fn = Mock()
        mock_cost_fn.volume = Mock(return_value=np.array([5.0]))
        
        cache = CostCache(cache_size=10)
        reg = VolumeRegularization(
            cost_functions=[mock_cost_fn], 
            epsilon=1.0, 
            cache=cache
        )
        
        xi = np.array([1.0])
        
        # First call should compute
        value1 = reg.value(xi)
        assert mock_cost_fn.volume.call_count == 1
        
        # Second call should use cache
        value2 = reg.value(xi)
        assert mock_cost_fn.volume.call_count == 1
        
        assert value1 == value2


class TestCompositeRegularization:
    """Tests for composite regularization strategy."""

    def test_init_empty(self):
        """Test initialization with no strategies."""
        reg = CompositeRegularization()
        
        xi = np.array([1.0, 2.0])
        assert reg.value(xi) == 0.0
        np.testing.assert_array_equal(reg.gradient(xi), np.zeros(2))

    def test_add_strategy(self):
        """Test adding strategies."""
        reg = CompositeRegularization()
        
        l2 = L2Regularization(xi_ref=np.array([0.0]), alpha=1.0)
        reg.add_strategy(l2)
        
        xi = np.array([2.0])
        assert reg.value(xi) > 0

    def test_combines_values(self):
        """Test that composite combines values from all strategies."""
        l2_1 = L2Regularization(xi_ref=np.array([0.0, 0.0]), alpha=1.0)
        l2_2 = L2Regularization(xi_ref=np.array([0.0, 0.0]), alpha=1.0)
        
        reg = CompositeRegularization([l2_1, l2_2])
        
        xi = np.array([1.0, 0.0])
        
        # Each L2 should contribute 0.5 * 1.0 * 1.0 = 0.5
        # Total should be 1.0
        assert reg.value(xi) == pytest.approx(1.0)

    def test_combines_gradients(self):
        """Test that composite combines gradients from all strategies."""
        l2_1 = L2Regularization(xi_ref=np.array([0.0, 0.0]), alpha=1.0)
        l2_2 = L2Regularization(xi_ref=np.array([0.0, 0.0]), alpha=1.0)
        
        reg = CompositeRegularization([l2_1, l2_2])
        
        xi = np.array([1.0, 2.0])
        grad = reg.gradient(xi)
        
        # Each L2 contributes [1.0, 2.0], total [2.0, 4.0]
        np.testing.assert_array_almost_equal(grad, [2.0, 4.0])

    def test_init_with_strategies(self):
        """Test initialization with list of strategies."""
        l2 = L2Regularization(xi_ref=np.array([0.0]), alpha=0.5)
        
        reg = CompositeRegularization([l2])
        
        xi = np.array([2.0])
        # 0.5 * 0.5 * 4 = 1.0
        assert reg.value(xi) == pytest.approx(1.0)
