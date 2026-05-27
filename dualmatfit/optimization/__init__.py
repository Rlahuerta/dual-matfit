# -*- coding: utf-8 -*-
"""Optimization: cost functions, loss functions, regularization, drivers."""

import importlib

__all__ = [
    "CostFunction", "CostIntegrator", "LSQFit",
    "CostCache", "LimitedOrderedDict",
    "RegularizationStrategy", "L2Regularization",
    "VolumeRegularization", "CompositeRegularization",
]

_MODULE_MAP = {
    "CostFunction": "cost",
    "CostIntegrator": "cost",
    "LSQFit": "cost",
    "CostCache": "cache",
    "LimitedOrderedDict": "cache",
    "RegularizationStrategy": "regularization",
    "L2Regularization": "regularization",
    "VolumeRegularization": "regularization",
    "CompositeRegularization": "regularization",
}


def __getattr__(name):
    if name in _MODULE_MAP:
        module = importlib.import_module(f".{_MODULE_MAP[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")