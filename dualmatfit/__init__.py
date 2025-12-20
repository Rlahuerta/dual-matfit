# -*- coding: utf-8 -*-
"""
DualMatFit - Dual Material Fitting

A Python package for fitting hyperelastic material models to experimental data,
with a focus on dual material formulations for arterial tissues.
"""

# Initialize logging on package import
from dualmatfit.logging_config import setup_logging

# Setup logging with default configuration
# Users can reconfigure by calling setup_logging() with custom parameters
setup_logging()

# Export extracted utility classes
from dualmatfit.cost_cache import CostCache, LimitedOrderedDict
from dualmatfit.regularization import (
    RegularizationStrategy,
    L2Regularization,
    VolumeRegularization,
    CompositeRegularization,
)
from dualmatfit.numeric_utils import (
    sanitize_array,
    sanitize_gradient,
    has_nan,
    has_inf,
    is_finite,
    has_non_finite,
    safe_divide,
)

__version__ = "0.1.0"

__all__ = [
    # Cache utilities
    "CostCache",
    "LimitedOrderedDict",
    # Regularization strategies
    "RegularizationStrategy",
    "L2Regularization",
    "VolumeRegularization",
    "CompositeRegularization",
    # Numeric utilities
    "sanitize_array",
    "sanitize_gradient",
    "has_nan",
    "has_inf",
    "is_finite",
    "has_non_finite",
    "safe_divide",
]
