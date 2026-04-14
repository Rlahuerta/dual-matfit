"""
Utilities for local identifiability and conditioning diagnostics.

This module provides small numerical helpers used by the reviewer-facing
identifiability workflow. The core idea is to inspect the local residual
Jacobian ``J`` and the Gauss-Newton normal matrix ``J^T J`` at a fitted
parameter vector, then summarize how well-conditioned that local linearization
is.

Typical library usage::

    report = conditioning_report(
        jacobian=jacobian_array,
        residual=residual_vector,
        param_names=("mu", "k_1", "alpha"),
    )

Typical integration usage::

    report = analyze_cost_integrator(
        integrator,
        xi,
        fiber_angle_key="alpha",
        stiffness_key="k_1",
    )

The resulting :class:`ConditioningReport` contains global conditioning metrics
for ``J`` and ``J^T J``, plus a focused diagnostic on the coupling between the
fiber-angle parameter and the stiffness parameter. The reported
``beta_variance_proxy`` is derived from the pseudoinverse of the unregularized
normal matrix and should be interpreted only as a local conditioning proxy, not
as a formal measurement-only covariance estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
# from typing import Sequence
# from dualmatfit.optimization.cost import CostIntegrator
# Deferred import to avoid circular dependency (covariance ↔ identifiability)
# build_gauss_newton_hessian is imported lazily inside functions that need it.

import numpy as np

__all__ = [
    "ConditioningReport",
    "as_1d_residual",
    "as_2d_jacobian",
    "analyze_cost_integrator",
    "beta_variance_proxy",
    "cosine_similarity",
]


@dataclass(slots=True)
class ConditioningReport:
    """
    Summary of local conditioning metrics for one fitted parameter state.

    Instances are usually created by :func:`conditioning_report` or
    :func:`analyze_cost_integrator` and then consumed by higher-level scripts
    that export reviewer tables or Markdown summaries.

    Attributes
    ----------
    param_names:
        Parameter names associated with the Jacobian columns.
    cost:
        Cost function Value
    jacobian:
        Shape of the normalized 2D Jacobian used for the analysis.
    singular_values:
        Singular values of the normalized Jacobian.
    condition_number_jtj:
        Condition number of the Gauss-Newton normal matrix ``J^T J``.
    beta_k1_condition_number:
        Condition number of the two-column submatrix formed by the fiber-angle
        and stiffness parameters when both are present.
    beta_k1_cosine_similarity:
        Cosine similarity between the fiber-angle and stiffness Jacobian
        columns, useful for detecting near-collinearity.
    smallest_singular_value:
        Smallest singular value of the Jacobian.
    beta_variance_proxy:
        Diagonal entry of the pseudoinverse of ``J^T J`` for the fiber-angle
        parameter. This is only a local conditioning proxy.
    omega_tik:
        Optional Tikhonov regularization weight reported as contextual
        metadata.
    """

    param_names: tuple[str, ...]
    cost: float
    jacobian: np.ndarray
    singular_values: np.ndarray
    covariance_matrix: np.ndarray
    standard_error: np.ndarray
    hessian: np.ndarray
    condition_number_jtj: float
    beta_k1_condition_number: float | None
    beta_k1_cosine_similarity: float | None
    smallest_singular_value: float
    beta_variance_proxy: float | None
    omega_tik: float | None


def as_2d_jacobian(jacobian: np.ndarray) -> np.ndarray:
    """
    Normalize a Jacobian-like array to a 2D ``(n_residuals, n_parameters)`` view.

    This helper accepts the common Jacobian shapes produced by the numerical
    stack:

    - already-2D arrays are returned unchanged;
    - ``(1, m, n)`` arrays are unwrapped to ``(m, n)``;
    - general 3D arrays are flattened over the leading dimensions.

    Parameters
    ----------
    jacobian:
        Jacobian array produced by a solver or cost integrator.

    Returns
    -------
    numpy.ndarray
        A float-valued 2D Jacobian matrix.
    """

    array = np.asarray(jacobian, dtype=float)
    if array.ndim == 2:
        return array

    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]

    if array.ndim == 3:
        return array.reshape(-1, array.shape[-1])

    raise ValueError(f"Expected 2D or 3D Jacobian, got shape {array.shape}")


def as_1d_residual(residual: np.ndarray) -> np.ndarray:
    """
    Normalize a residual-like array to a flat 1D vector.

    Parameters
    ----------
    residual:
        Residual array produced by a solver or cost integrator.

    Returns
    -------
    numpy.ndarray
        A float-valued residual vector with shape ``(n_residuals,)``.
    """

    return np.asarray(residual, dtype=float).reshape(-1)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two Jacobian columns.

    Values near ``1`` or ``-1`` indicate that the two parameter directions are
    nearly collinear in the local linearization, which is a common signature of
    weak identifiability.

    Parameters
    ----------
    vec_a, vec_b:
        vector to compare.
    """

    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)

    if denom == 0.0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / denom)


def beta_variance_proxy(normal: np.ndarray, parameter_index: int) -> float:
    """
    Extract a non-negative diagonal proxy from the pseudoinverse of ``J^T J``.

    This helper is intentionally conservative: it does not claim to compute a
    formal covariance matrix. Instead, it returns the diagonal entry associated
    with one parameter in the pseudoinverse of the unregularized normal matrix,
    which is useful as a local conditioning indicator.

    Parameters
    ----------
    normal:
        Square Gauss-Newton normal matrix.
    parameter_index:
        Index of the parameter of interest within ``normal``.
    """

    matrix = np.asarray(normal, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square normal matrix, got shape {matrix.shape}")

    if not 0 <= parameter_index < matrix.shape[0]:
        raise IndexError(f"parameter_index {parameter_index} is out of bounds for shape {matrix.shape}")

    covariance_like = np.linalg.pinv(matrix)
    value = float(covariance_like[parameter_index, parameter_index])
    return max(value, 0.)


def analyze_cost_integrator(
    integrator,
    xi: np.ndarray,
    *,
    fiber_angle_key: str = "alpha",
    stiffness_key: str = "k_1",
) -> ConditioningReport:
    """
    Evaluate a cost integrator and return its local conditioning report.

    This is the high-level convenience entrypoint used by reviewer-facing
    scripts. It evaluates the integrator residual and Jacobian at ``xi``,
    normalizes those arrays, and forwards them into :func:`conditioning_report`.

    Parameters
    ----------
    integrator:
        Cost-integrator-like object exposing ``_residuum(xi)``,
        ``_residuum_diff(xi)``, and ``inp_mat_keys``.
    xi:
        Parameter vector at which the local linearization is evaluated.
    fiber_angle_key, stiffness_key:
        Parameter names used for the focused two-parameter coupling summary.
    """

    np_fval = integrator._cost_function(xi)
    np_jacobian_raw = integrator._residuum_diff(xi)

    # Reshape 3D (n_funcs, n_control, n_params) to 2D (n_obs, n_params)
    if np_jacobian_raw.ndim == 3:
        np_jacobian = np_jacobian_raw.reshape(-1, np_jacobian_raw.shape[-1])
    elif np_jacobian_raw.ndim == 2:
        np_jacobian = np_jacobian_raw
    else:
        np_jacobian = np_jacobian_raw.reshape(1, -1)

    param_names = integrator.inp_mat_keys
    param_idx = np.array(
        [param_names.index(param_i) for param_i in [fiber_angle_key, stiffness_key]],
        dtype=int,
    )

    from dualmatfit.fitting.covariance import build_gauss_newton_hessian  # deferred
    gauss_newton_hessian = build_gauss_newton_hessian(np_jacobian, nvars=len(param_names))
    gauss_newton_hessian_bk1 = gauss_newton_hessian[param_idx, :][:, param_idx]

    singular_values = np.asarray(np.linalg.svd(gauss_newton_hessian_bk1, compute_uv=False), dtype=float)
    smallest_singular_value = float(singular_values[-1]) if singular_values.size else 0.0
    est_res_variance = 2 * np.sum(np_fval) / (integrator.cost_functions[0].ncontrol - param_idx.shape[0])

    covar_matrix = np.sqrt(est_res_variance) * np.linalg.pinv(gauss_newton_hessian_bk1)
    std_error = np.sqrt(np.linalg.diagonal(covar_matrix))

    beta_k1_condition_number: float | None = None
    beta_k1_cosine: float | None = None
    beta_variance: float | None = None

    if fiber_angle_key in param_names and stiffness_key in param_names:
        fiber_index = param_names.index(fiber_angle_key)
        stiffness_index = param_names.index(stiffness_key)
        beta_k1_matrix = np_jacobian[:, param_idx]
        beta_k1_condition_number = float(np.linalg.cond(beta_k1_matrix))
        beta_k1_cosine = cosine_similarity(np_jacobian[fiber_index], np_jacobian[stiffness_index])
        beta_variance = beta_variance_proxy(gauss_newton_hessian, fiber_index)

    conditioning_report = ConditioningReport(
        param_names=param_names,
        cost=np.sum(np_fval),
        jacobian=np_jacobian.copy(),
        singular_values=singular_values.copy(),
        hessian=gauss_newton_hessian_bk1.copy(),
        covariance_matrix=covar_matrix.copy(),
        standard_error=std_error.copy(),
        condition_number_jtj=float(np.linalg.cond(gauss_newton_hessian_bk1.copy())),
        beta_k1_condition_number=beta_k1_condition_number,
        beta_k1_cosine_similarity=beta_k1_cosine,
        smallest_singular_value=smallest_singular_value,
        beta_variance_proxy=beta_variance,
        omega_tik=float(integrator._alpha) if integrator._alpha is not None else None,
    )

    return conditioning_report