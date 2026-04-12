# -*- coding: utf-8 -*-
"""Experimental data loading and processing."""

import importlib

__all__ = ["InstronData"]

_MODULE_MAP = {
    "InstronData": "experimental",
}


def __getattr__(name):
    if name in _MODULE_MAP:
        module = importlib.import_module(f".{_MODULE_MAP[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")