# -*- coding: utf-8 -*-
"""Symbolic formulation: material laws, variational forms, tensor algebra, lambdify."""

import importlib

__all__ = ["VariationalFormulation", "LambdifyBuilder", "TensorManager"]

_MODULE_MAP = {
    "VariationalFormulation": "variational",
    "LambdifyBuilder": "lambdify",
    "TensorManager": "tensor",
}


def __getattr__(name):
    if name in _MODULE_MAP:
        module = importlib.import_module(f".{_MODULE_MAP[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")