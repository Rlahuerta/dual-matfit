# -*- coding: utf-8 -*-
"""
Optimization utilities and constraint handling.

This module provides constraint aggregation methods (KS aggregation)
and validation helpers for constrained optimization problems.
"""
import numpy as np
import pandas as pd
import sympy as sy

from functools import lru_cache
from collections import OrderedDict
from typing import Tuple, List, Dict, Union, Optional

from dualmatfit.formulation.variational import VariationalFormulation
from dualmatfit.solvers.extension import ExtensionSolution
from dualmatfit.solvers.derivative import _fdm, adjoint_derivative
from dualmatfit.utils.numeric import safe_divide

__all__ = [
    'ConstraintAggregation',
    'cst_sparsity_indices',
]


def _validate_and_fill_gval(
    gval: Optional[np.ndarray],
    values: np.ndarray,
    expected_shape: int,
) -> None:
    """
    Validate gval shape and fill it with values if provided.
    
    Parameters
    ----------
    gval : np.ndarray or None
        Output array to fill (modified in-place).
    values : np.ndarray
        Values to store in gval.
    expected_shape : int
        Expected first dimension of gval.
        
    Raises
    ------
    ValueError
        If gval shape doesn't match expected dimensions.
    """
    if gval is not None:
        if values.shape[0] != expected_shape:
            raise ValueError(
                f"Constraint array dimension mismatch: expected {expected_shape}, "
                f"got {values.shape[0]}."
            )
        gval[:] = values


def cst_sparsity_indices(nvars, cst_num) -> Tuple[np.ndarray, np.ndarray]:
    """
    Define the nonzero slots in the jacobian, there are no nonzeros in the constraint jacobian, this function
    returns an empty tuple if there are no constraints. Otherwise, it creates two arrays that list the row and
    column indices for the nonzeros in the Jacobian. The row index repeats for every variable since each variable
    has a nonzero entry in each constraint. The column index corresponds to the position of each variable in the
    optimization problem. The function returns a tuple of two arrays, where the first array contains the row indices
    and the second array contains the column indices of the nonzeros in the Jacobian.

    Returns:
    -------
        Tuple[NDArray, NDArray]: A tuple of two arrays, where the first array contains the row indices and the
        second array contains the column indices of the nonzeros in the Jacobian.
    """

    if cst_num == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    list_rows, list_cols = [], []
    cols_i = np.arange(0, nvars)

    for i in range(cst_num):
        list_rows += [i] * nvars
        list_cols += cols_i.tolist()

    return np.array(list_rows, dtype=int), np.array(list_cols, dtype=int)
