# -*- coding: utf-8 -*-
"""Numerical solvers: root-finding, extension solution, derivatives, barrier methods."""

import importlib

__all__ = ["Root", "ExtensionSolution", "DesignVariablesMixin", "check_dsvars"]

_MODULE_MAP = {
    "Root": "solution",
    "ExtensionSolution": "extension",
    "DesignVariablesMixin": "extension",
    "check_dsvars": "extension",
}


def __getattr__(name):
    if name in _MODULE_MAP:
        module = importlib.import_module(f".{_MODULE_MAP[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")