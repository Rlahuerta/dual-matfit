# -*- coding: utf-8 -*-
"""Material parameter fitting: orchestration, optimization, persistence, visualization."""

import importlib

__all__ = ["AnisoModelSolve", "AnisoMaterialFit"]

_MODULE_MAP = {
    "AnisoModelSolve": "core",
    "AnisoMaterialFit": "core",
}


def __getattr__(name):
    if name in _MODULE_MAP:
        module = importlib.import_module(f".{_MODULE_MAP[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")