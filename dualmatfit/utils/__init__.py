# -*- coding: utf-8 -*-
"""Cross-cutting utilities: logging, numeric helpers, I/O, path management."""

import importlib

__all__ = [
    "get_logger", "setup_logging",
    "sanitize_array", "sanitize_gradient", "has_nan", "has_inf",
    "is_finite", "has_non_finite", "safe_divide",
    "PathConfiguration", "PathManager",
    "min_ks", "max_ks",
]

_MODULE_MAP = {
    "get_logger": "logging_config",
    "setup_logging": "logging_config",
    "sanitize_array": "numeric",
    "sanitize_gradient": "numeric",
    "has_nan": "numeric",
    "has_inf": "numeric",
    "is_finite": "numeric",
    "has_non_finite": "numeric",
    "safe_divide": "numeric",
    "PathConfiguration": "path_manager",
    "PathManager": "path_manager",
    "min_ks": "ks",
    "max_ks": "ks",
}


def __getattr__(name):
    if name in _MODULE_MAP:
        module = importlib.import_module(f".{_MODULE_MAP[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")