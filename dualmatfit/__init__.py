# -*- coding: utf-8 -*-
"""Dual Material Fitting for 1D."""

__version__ = "0.1.0"

import importlib

# Initialize logging on import (lightweight, no heavy deps)
from dualmatfit.utils.logging_config import setup_logging
setup_logging()

__all__ = [
    "__version__",
    "CostCache",
    "LimitedOrderedDict",
    "RegularizationStrategy",
    "L2Regularization",
    "VolumeRegularization",
    "CompositeRegularization",
    "sanitize_array",
    "sanitize_gradient",
    "has_nan",
    "has_inf",
    "is_finite",
    "has_non_finite",
    "safe_divide",
]

_SUBPKG_MAP = {
    "CostCache": "optimization.cache",
    "LimitedOrderedDict": "optimization.cache",
    "RegularizationStrategy": "optimization.regularization",
    "L2Regularization": "optimization.regularization",
    "VolumeRegularization": "optimization.regularization",
    "CompositeRegularization": "optimization.regularization",
    "sanitize_array": "utils.numeric",
    "sanitize_gradient": "utils.numeric",
    "has_nan": "utils.numeric",
    "has_inf": "utils.numeric",
    "is_finite": "utils.numeric",
    "has_non_finite": "utils.numeric",
    "safe_divide": "utils.numeric",
}


def __getattr__(name):
    if name in _SUBPKG_MAP:
        module = importlib.import_module(f"dualmatfit.{_SUBPKG_MAP[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")