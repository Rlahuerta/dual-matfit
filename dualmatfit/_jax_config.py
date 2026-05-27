# -*- coding: utf-8 -*-
"""Shared JAX configuration for dualmatfit."""

from __future__ import annotations

import os
from typing import Any

_CPU_PLATFORM = "cpu"


def configure_jax_environment() -> None:
    """Default JAX to CPU unless the caller explicitly selected platforms."""
    os.environ.setdefault("JAX_PLATFORMS", _CPU_PLATFORM)


def configure_jax() -> Any:
    """Configure JAX consistently before numerical modules use it."""
    configure_jax_environment()

    import jax

    jax.config.update("jax_enable_x64", True)
    return jax

