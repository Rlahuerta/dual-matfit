"""
Accurate covariance matrix estimation for fitted model parameters.

Implements a full covariance pipeline for penalised M-estimators:

1. **Hessian computation** — Ridders' method for diagonal curvatures and
   Richardson extrapolation for off-diagonal elements [Baker, 2021].
2. **Eigenvalue polish** — recompute eigenvalues in eigenvector directions
   via Ridders' extrapolation to ensure positive-definiteness [Baker, 2021].
3. **NPD calibration** — condition-number-aware eigenvalue thresholding
   based on Huang et al. (2017) with a ``lambda_max / kappa_target`` floor.
4. **Sandwich covariance** — ``V = H^{-1} B H^{-1}`` where ``B`` is the meat
   matrix assembled from per-section score vectors [White, 1980; Huber
   and Ronchetti, 2009, Sec.7.6].
5. **Confidence intervals** — Wald-type ``xi pm t_{alpha/2, dof} x SE`` with
   optional clipping to design-variable box constraints [Seber and Wild,
   2003, Sec.5.2].

References
----------
 - [1] Baker, R. (2021). Estimating accurate covariance matrices on
       fitted model parameters. *arXiv:2105.04829v1*.
 - [2] White, H. (1980). A heteroskedasticity-consistent covariance
       matrix estimator. *Econometrica*, 48(4):817-838.
 - [3] Huber, P. J. and Ronchetti, E. M. (2009). *Robust Statistics*,
       2nd ed. Wiley. Sec.7.6 (sandwich formula for M-estimators).
 - [4] Huang, C., Farewell, D. and Pan, J. (2017). A calibration method
       for non-positive definite covariance matrix. *J. Multivariate
       Analysis*, 157:45-52.
 - [5] Seber, G. A. F. and Wild, C. J. (2003). *Nonlinear Regression*.
       Wiley. Sec.5.2 (finite-sample t-based confidence intervals).
 - [6] Press, W. H., et al. (2007). *Numerical Recipes*, 3rd ed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

from scipy.stats import t as t_dist

from dualmatfit.utils.logging_config import get_logger
from dualmatfit.fitting.identifiability import as_2d_jacobian, as_1d_residual

logger = get_logger('covariance')

__all__ = [
    'find_initial_step',
    'ridders_curvature',
    'central_diff_cross',
    'build_gauss_newton_hessian',
    'richardson_off_diagonal',
    'accurate_hessian',
    'eigenvalue_polish',
    'huang_calibration',
    'CovarianceReport',
    'frobenius_distance',
    'correlation_distance',
    'g_metric',
    'robust_covariance_from_cost',
    'mle_covariance',
    'covariance_from_gauss_newton',
    'save_covariance_report',
    'load_covariance_report',
    'format_params_with_uncertainty',
    'build_covariance_summary_table',
]

# Type alias for objective functions: f(x: ndarray) -> float
ObjectiveFunc = Callable[[np.ndarray], float]

# Machine epsilon for float64
_EPS = np.finfo(np.float64).eps


# ---------------------------------------------------------------------------
# Phase 1: Ridders' method for diagonal curvatures
# ---------------------------------------------------------------------------

def find_initial_step(
    cost_fun: ObjectiveFunc,
    x0: np.ndarray,
    i: int,
    eps: float = float(_EPS),
    max_doublings: int = 50,
) -> Tuple[float, float]:
    """Find an initial step size suitable for curvature estimation along axis *i*.

    Starting from ``h = eps^{1/4}``, doubles *h* until the central-difference
    curvature estimate ``(f(x0+h) - 2f(x0) + f(x0-h)) / h^2`` is finite and
    positive. Falls back to the bracket condition
    ``f(x0+h) > f(x0) and f(x0-h) > f(x0)`` when possible.

    Parameters
    ----------
    cost_fun : callable
        Objective function ``f(x) -> float``.
    x0 : ndarray
        Point at which to evaluate.
    i : int
        Coordinate index along which to probe.
    eps : float
        Machine epsilon (default: ``np.finfo(float).eps``).
    max_doublings : int
        Safety limit on the number of doublings.

    Returns
    -------
    h : float
        Step size suitable for curvature estimation.
    f0 : float
        Function value ``f(x0)`` (cached to avoid redundant evaluation).
    """
    h = eps ** 0.25
    f0 = cost_fun(x0)
    e_i = np.zeros_like(x0)
    e_i[i] = 1.0

    for _ in range(max_doublings):
        fp = cost_fun(x0 + h * e_i)
        fm = cost_fun(x0 - h * e_i)
        if not (np.isfinite(fp) and np.isfinite(fm)):
            h *= 0.5
            continue
        cd = fp - 2.0 * f0 + fm
        if cd > 0.0 and np.isfinite(cd / (h * h)):
            return h, f0
        h *= 2.0

    logger.warning(
        "find_initial_step: suitable step not found after %d iterations "
        "for axis %d; using last h=%.4e", max_doublings, i, h,
    )
    return h, f0


def ridders_curvature(
    cost_fun: ObjectiveFunc,
    x0: np.ndarray,
    i: int,
    beta: float = 1.4,
    max_tableau: int = 10,
) -> Tuple[float, float]:
    """Compute the diagonal curvature ``d^2f/dx_i^2`` using Ridders' method.

    Implements Steps 1-3 of Baker (2021, Section 2):
    1. Find initial step via :func:`find_initial_step`.
    2. Estimate curvature scale ``sigma = h / sqrt(f(x0+h) - 2f(x0) + f(x0-h))``.
    3. Apply Neville-tableau Richardson extrapolation (Ridders' method)
       starting at ``h = sigma / 2``, scaling down by *beta* each row.

    Parameters
    ----------
    cost_fun : callable
        Objective function.
    x0 : ndarray
        Evaluation point.
    i : int
        Coordinate index for the curvature.
    beta : float
        Step-size reduction factor per tableau row (paper recommends sqrt(2) ~ 1.4).
    max_tableau : int
        Maximum number of Neville-tableau rows.

    Returns
    -------
    curvature : float
        Estimated second derivative ``d^2f/dx_i^2``.
    h_opt : float
        Optimal step size at which the best estimate was obtained.
    """
    h_init, f0 = find_initial_step(cost_fun, x0, i)

    e_i = np.zeros_like(x0)
    e_i[i] = 1.0

    # Central-difference curvature estimate at h_init
    fp = cost_fun(x0 + h_init * e_i)
    fm = cost_fun(x0 - h_init * e_i)
    cd = fp - 2.0 * f0 + fm

    if cd <= 0.0:
        # Concave or flat — fall back to simple central difference
        h_start = h_init
    else:
        sigma = h_init / np.sqrt(cd)
        h_start = sigma / 2.0

    # Ridders' Neville tableau
    beta2 = beta * beta
    tableau = np.zeros((max_tableau, max_tableau))
    h = h_start
    best_err = np.inf
    best_val = 0.0
    best_h = h

    for row in range(max_tableau):
        fp = cost_fun(x0 + h * e_i)
        fm = cost_fun(x0 - h * e_i)
        tableau[row, 0] = (fp - 2.0 * f0 + fm) / (h * h)

        fac = beta2
        for col in range(1, row + 1):
            tableau[row, col] = (
                fac * tableau[row, col - 1] - tableau[row - 1, col - 1]
            ) / (fac - 1.0)
            fac *= beta2

            err = max(
                abs(tableau[row, col] - tableau[row, col - 1]),
                abs(tableau[row, col] - tableau[row - 1, col - 1]),
            )
            if err < best_err:
                best_err = err
                best_val = tableau[row, col]
                best_h = h

        # If the latest diagonal is drifting, stop early
        if row > 0:
            delta = abs(tableau[row, row] - tableau[row - 1, row - 1])
            if delta > 2.0 * best_err:
                break

        h /= beta

    return float(best_val), float(best_h)


# ---------------------------------------------------------------------------
# Phase 2: Richardson extrapolation for off-diagonal elements
# ---------------------------------------------------------------------------

def central_diff_cross(
    f: ObjectiveFunc,
    x0: np.ndarray,
    i: int,
    j: int,
    hi: float,
    hj: float,
) -> float:
    """Central difference approximation to the cross-derivative d^2f/dx_i dx_j.

    .. math::

        D(h_i, h_j) = \\frac{f(x_0+h_i e_i+h_j e_j) + f(x_0-h_i e_i-h_j e_j)
        - f(x_0+h_i e_i-h_j e_j) - f(x_0-h_i e_i+h_j e_j)}{4 h_i h_j}

    This has error ``O(h_i h_j)``.
    """
    e_i = np.zeros_like(x0)
    e_j = np.zeros_like(x0)
    e_i[i] = hi
    e_j[j] = hj

    fpp = f(x0 + e_i + e_j)
    fmm = f(x0 - e_i - e_j)
    fpm = f(x0 + e_i - e_j)
    fmp = f(x0 - e_i + e_j)

    return (fpp + fmm - fpm - fmp) / (4.0 * hi * hj)


def build_gauss_newton_hessian(jacobian: np.ndarray, nvars: int) -> np.ndarray:
    """
    Build the Gauss-Newton normal matrix ``J^T J`` from a Jacobian vector.

    Parameters
    ----------
    jacobian:
        Jacobian matrix or a Jacobian-like array accepted by
        :func:`as_2d_jacobian`.

    nvars:
        Number of design variable
    """

    if len(jacobian.shape) == 1:
        assert jacobian.shape[0] == nvars, f"Wrong Jacobian Array Dimension {jacobian.shape[0]} != {nvars}"

        return np.linalg.outer(jacobian, jacobian)
    else:
        assert nvars in jacobian.shape, f"Wrong Jacobian Array Dimension {jacobian.shape[0]} != {nvars}"

        if jacobian.shape[0] == nvars:
            return jacobian @ jacobian.T
        else:
            return jacobian.T @ jacobian


def richardson_off_diagonal(
    cost_fun: ObjectiveFunc,
    x0: np.ndarray,
    i: int,
    j: int,
    hi: float,
    hj: float,
) -> float:
    """Off-diagonal Hessian element with one Richardson extrapolation step.

    .. math::

        H_{ij} \\approx \\frac{4 D(h_i/2, h_j/2) - D(h_i, h_j)}{3}

    This improves the error from ``O(h^2)`` to ``O(h^4)``.
    Uses 8 function evaluations total (4 per central difference call).
    """
    d_full = central_diff_cross(cost_fun, x0, i, j, hi, hj)
    d_half = central_diff_cross(cost_fun, x0, i, j, hi / 2.0, hj / 2.0)
    return (4.0 * d_half - d_full) / 3.0


# ---------------------------------------------------------------------------
# Phase 3: Full accurate Hessian assembly
# ---------------------------------------------------------------------------

def accurate_hessian(
    cost_fun: ObjectiveFunc,
    x0: np.ndarray,
    beta: float = 1.4,
    max_tableau: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the full Hessian matrix using Ridders' + Richardson.

    Diagonal elements are computed with Ridders' method; off-diagonal
    elements use central differences with one Richardson extrapolation step,
    using the optimal step sizes from the diagonal computation.

    Parameters
    ----------
    cost_fun : callable
        Objective function ``f(x) -> float``.
    x0 : ndarray, shape (n,)
        Point at which to evaluate the Hessian.
    beta : float
        Ridders' step-size reduction factor.
    max_tableau : int
        Maximum Neville-tableau rows for Ridders' method.

    Returns
    -------
    H : ndarray, shape (n, n)
        Hessian matrix.
    h_opt : ndarray, shape (n,)
        Optimal step sizes per parameter (from Ridders').
    """
    n = len(x0)
    H = np.zeros((n, n))
    h_opt = np.zeros(n)

    # Diagonal elements via Ridders'
    for i in range(n):
        H[i, i], h_opt[i] = ridders_curvature(cost_fun, x0, i, beta=beta, max_tableau=max_tableau)

    # Off-diagonal elements via Richardson extrapolation
    for i in range(n):
        for j in range(i + 1, n):
            hij = richardson_off_diagonal(cost_fun, x0, i, j, h_opt[i], h_opt[j])
            H[i, j] = hij
            H[j, i] = hij

    return H, h_opt


# ---------------------------------------------------------------------------
# Phase 4: Eigenvalue polish
# ---------------------------------------------------------------------------

def eigenvalue_polish(
    hessian: np.ndarray,
    cost_fun: ObjectiveFunc,
    x0: np.ndarray,
    recompute: bool = True,
    beta: float = 1.4,
    max_tableau: int = 10,
    min_eigenvalue: float = 1e-12,
) -> np.ndarray:
    """Polish eigenvalues to ensure positive-definiteness.

    Diagonalizes ``H = Q D Q^T`` (via ``eigh`` for symmetric matrices) and
    optionally recomputes each eigenvalue by applying Ridders' method along
    the corresponding eigenvector direction.  Negative or tiny eigenvalues
    are clamped to *min_eigenvalue*.

    References
    ----------
    Baker, R. (2021). Estimating accurate covariance matrices on fitted
    model parameters. *arXiv:2105.04829v1*, Section 3.

    Parameters
    ----------
    hessian : ndarray, shape (n, n)
        Hessian matrix to polish.
    cost_fun : callable
        Objective function (needed for recomputation).
    x0 : ndarray
        Point at which to recompute eigenvalues.
    recompute : bool
        If True, recompute eigenvalues via Ridders' in eigenvector directions.
    beta, max_tableau : float, int
        Ridders' method parameters (only used if *recompute* is True).
    min_eigenvalue : float
        Floor for eigenvalues.

    Returns
    -------
    H_polished : ndarray, shape (n, n)
        Polished Hessian with all eigenvalues positive.
    """
    eigen_values, eigen_vectors = np.linalg.eigh(hessian)

    if recompute:
        for k in range(len(eigen_values)):
            v_k = eigen_vectors[:, k].copy()

            def f_along_eigvec(t_arr, _v=v_k):
                return cost_fun(x0 + t_arr[0] * _v)

            curv, _ = ridders_curvature(f_along_eigvec, np.asarray([0.0]), 0,
                                        beta=beta, max_tableau=max_tableau)
            eigen_values[k] = curv

    eigen_values = np.maximum(eigen_values, min_eigenvalue)
    return eigen_vectors @ np.diag(eigen_values) @ eigen_vectors.T


# ---------------------------------------------------------------------------
# Phase 4b: Huang et al. (2017) NPD calibration
# ---------------------------------------------------------------------------

def huang_calibration(
        matrix: np.ndarray,
        atol: float = 0.0,
        kappa_target: float = 1e8,
) -> np.ndarray:
    """Calibrate a matrix to a well-conditioned positive-definite surrogate.

    Uses direct eigenvalue thresholding via a condition-number target.
    Every eigenvalue below ``c* = lambda_max / kappa_target`` is lifted to ``c*``,
    guaranteeing ``kappa(H_PD) <= kappa_target`` with minimal Frobenius distortion.

    Parameters
    ----------
    matrix : ndarray, shape (n, n)
        Symmetric matrix (possibly NPD or ill-conditioned) to calibrate.
    atol : float
        Eigenvalues <= *atol* are treated as non-positive.
    kappa_target : float
        Maximum acceptable condition number for the calibrated matrix.
        The eigenvalue floor is ``lambda_max / kappa_target``.
        Set to ``np.inf`` to disable the floor (eigenvalues are still
        lifted to *atol* to ensure positivity).

    Returns
    -------
    H_pd : ndarray, shape (n, n)
        Calibrated positive-definite matrix with ``kappa <= kappa_target``.

    References
    ----------
    Lopez C., D. C. et al. (2015). A computational framework for
    identifiability and ill-conditioning analysis of lithium-ion battery
    models. *SIAM/ASA J. Uncertainty Quantification*, 3(1):464-504.

    Seber, G. A. F. and Wild, C. J. (2003). *Nonlinear Regression*.
    Wiley. Section 5.1 — uses ``tol x lambda_max`` as eigenvalue floor.
    """
    eigen_values, eigen_vectors = np.linalg.eigh(matrix)
    lambda_max = eigen_values.max()

    # Direct threshold: c* = lambda_max / kappa_target
    if np.isfinite(kappa_target) and lambda_max > 0:
        c_star = lambda_max / kappa_target
    elif lambda_max > 0:
        # kappa_target=inf: no conditioning control, but ensure positivity
        c_star = lambda_max * 1e-12
    else:
        # All-negative: anchor to |lambda_max|
        c_star = np.abs(eigen_values).max() * 1e-6

    # Ensure at least atol for positivity
    c_star = max(c_star, atol)

    # Early exit: already well-conditioned PD
    if eigen_values.min() >= c_star:
        kappa_actual = lambda_max / eigen_values.min() if eigen_values.min() > 0 else np.inf
        logger.debug(
            "huang_calibration: matrix already PD with kappa=%.2e <= kappa_target=%.2e",
            kappa_actual, kappa_target,
        )
        return matrix

    n_flipped = int(np.sum(eigen_values < c_star))

    # Iterative calibration: progressively tighten kappa_target if needed
    c_star_i = c_star
    kappa_target_i = kappa_target
    eigen_vectors_i = eigen_vectors

    for iteration in range(10):
        calibrated_eigs = np.maximum(eigen_values, c_star_i)
        matrix_cal = eigen_vectors_i @ np.diag(calibrated_eigs) @ eigen_vectors_i.T

        # Verify result via fresh decomposition (catches numerical drift)
        eigen_values_check, eigen_vectors_i = np.linalg.eigh(matrix_cal)

        if eigen_values_check.min() >= atol:
            break

        # Tighten: halve kappa_target (handles finite targets only)
        if np.isfinite(kappa_target_i):
            kappa_target_i /= 2.0
            c_star_i = lambda_max / kappa_target_i
        else:
            # kappa_target=inf: fall back to atol as hard floor
            c_star_i = max(atol, lambda_max * 1e-10)
            break

        # Use the verified eigenvalues for the next iteration
        eigen_values = eigen_values_check

    kappa_result = calibrated_eigs.max() / calibrated_eigs.min()
    logger.info(
        "huang_calibration: c*=%.3e  flipped %d/%d eigenvalues  kappa_result=%.2e",
        c_star_i, n_flipped, len(eigen_values), kappa_result,
    )

    return matrix_cal


# ---------------------------------------------------------------------------
# Phase 5: MLE covariance and CovarianceReport
# ---------------------------------------------------------------------------

@dataclass
class CovarianceReport:
    """Results of MLE covariance estimation.

    Attributes
    ----------
    param_names : tuple of str
        Names of the fitted parameters.
    covariance_matrix : ndarray, shape (n, n)
        Estimated covariance matrix ``V = H^{-1}``.
    standard_errors : ndarray, shape (n,)
        Standard errors ``sigma_i = sqrt(V_{ii})``.
    correlation_matrix : ndarray, shape (n, n)
        Correlation matrix ``R_{ij} = V_{ij} / (sigma_i sigma_j)``.
    hessian_matrix : ndarray, shape (n, n)
        The computed (and optionally polished) Hessian.
    eigenvalues : ndarray, shape (n,)
        Eigenvalues of the Hessian (sorted ascending from ``eigh``).
    hessian_diagonal : Series, shape (n,)
        Per-parameter Hessian curvature ``H_{ii}``, indexed by parameter
        name.  Unlike eigenvalues (which are rotated mixtures of
        parameters), the diagonal gives an unambiguous per-parameter
        stiffness measure for identifying weakly-determined parameters.
    confidence_interval : DataFrame, shape (n, 3)
        Per-parameter confidence interval (``lower``, ``value``,
        ``upper``) at the ``confidence_level`` significance level,
        indexed by parameter name.  Computed as
        ``xi pm t_{alpha/2, dof} x SE`` where *n_params* is the **total**
        number of fitted parameters (not just those in *param_names*)
        and *dof = n_obs - n_params*.  The raw interval is clipped to
        the design-variable box constraints when available, so
        ``lower >= lb`` and ``upper <= ub`` for every parameter.
    confidence_level : float
        Confidence level used for ``confidence_interval`` (e.g. 0.95).
    n_function_evals : int
        Total number of objective-function evaluations.
    polished : bool
        Whether eigenvalue polish was applied.
    calibrated : bool
        Whether Huang et al. (2017) NPD calibration was applied.
    method : str
        Estimation method: ``'accurate'`` or ``'gauss_newton'``.
    """
    param_names: Sequence
    param_idx: Sequence
    covariance_matrix: np.ndarray | pd.DataFrame
    standard_errors: np.ndarray
    correlation_matrix: np.ndarray | pd.DataFrame
    hessian_matrix: np.ndarray
    hessian_condition: float
    eigenvalues: np.ndarray | pd.Series
    hessian_diagonal: pd.Series
    confidence_interval: pd.DataFrame
    confidence_level: float
    n_function_evals: int
    polished: bool
    calibrated: bool
    method: str


class _EvalCounter:
    """Transparent wrapper that counts objective-function evaluations."""

    def __init__(self, f: ObjectiveFunc):
        self._f = f
        self.count: int = 0

    def __call__(self, x: np.ndarray) -> float:
        self.count += 1
        return self._f(x)


# ---------------------------------------------------------------------------
# Accuracy metrics (Paper Section 5)
# ---------------------------------------------------------------------------

def frobenius_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Mean absolute element-wise distance (Paper Eq. for F).

    ``F = Sigma|A_ij - B_ij| / n^2``
    """
    n = vec_a.shape[0]
    return float(np.sum(np.abs(vec_a - vec_b)) / (n * n))


def correlation_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Frobenius-style distance between two correlation matrices."""
    return frobenius_distance(_to_corr(vec_a), _to_corr(vec_b))


def g_metric(vec_comp: np.ndarray, vec_ref: np.ndarray) -> float:
    """Average percentage error on standard errors (Paper metric G).

    ``G = 100 Sigma|sqrt(V_ii) - sqrt(C_ii)| / (n sqrt(C_ii))``
    """
    se_comp = np.sqrt(np.diag(vec_comp))
    se_ref = np.sqrt(np.diag(vec_ref))
    n = len(se_ref)

    return float(100.0 * np.sum(np.abs(se_comp - se_ref) / se_ref) / n)


def _to_corr(cov_mat: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix."""
    np_cov_diag = np.diag(cov_mat)
    se = np.emath.sqrt(np_cov_diag)

    outer = np.outer(se, se)
    outer = np.where(outer == 0.0, 1.0, outer)

    return np.real(cov_mat / outer)


def _compute_confidence_interval(
    xi: np.ndarray,
    se: np.ndarray,
    n_obs: int,
    n_params: int,
    param_names: Sequence[str],
    confidence_level: float = 0.95,
    bounds: Optional[Sequence[Sequence[float]]] = None,
) -> pd.DataFrame:
    """Compute per-parameter confidence intervals.

    CI = xi pm t_{alpha/2, dof} x SE  where dof = n_obs - n_params.

    When *bounds* are supplied the raw interval is clipped so that
    ``lower >= lb`` and ``upper <= ub`` for each parameter, ensuring the
    CI stays inside the physically admissible domain.

    References
    ----------
    Seber, G. A. F. and Wild, C. J. (2003). *Nonlinear Regression*.
    Wiley. Section 5.2 — finite-sample t-based confidence intervals.

    Parameters
    ----------
    xi : ndarray, shape (n,)
        Fitted parameter vector.
    se : ndarray, shape (n,)
        Standard errors (sqrt(diag(V))).
    n_obs : int
        Total number of observations across all sections.
    n_params : int
        Total number of fitted parameters (degrees-of-freedom denominator).
    param_names : sequence of str
        Parameter labels for the DataFrame index.
    confidence_level : float
        Desired confidence level (default 0.95).
    bounds : sequence of [lower, upper] or None
        Per-parameter box constraints.  Each entry is ``[lb, ub]``.
        Pass *None* to leave the interval unconstrained.
    """
    dof = max(n_obs - n_params, 1)
    alpha = 1.0 - confidence_level
    t_crit = float(t_dist.ppf(1.0 - alpha / 2.0, dof))

    half_width = t_crit * se
    ci_lower = xi - half_width
    ci_upper = xi + half_width

    if bounds is not None:
        for i, (lb, ub) in enumerate(bounds):
            if lb is not None and np.isfinite(lb):
                ci_lower[i] = max(ci_lower[i], lb)
            if ub is not None and np.isfinite(ub):
                ci_upper[i] = min(ci_upper[i], ub)

    return pd.DataFrame(
        {'lower': ci_lower, 'value': xi, 'upper': ci_upper},
        index=param_names,
    )


# ---------------------------------------------------------------------------
# Phase 6: CostIntegrator / CostFunction integration
# ---------------------------------------------------------------------------

def robust_covariance_from_cost(
        integrator,
        xi: np.ndarray,
        param_names: Optional[Sequence[str]] = None,
        polish: bool = True,
        calibrate: bool = True,
        confidence_level: float = 0.95,
        beta: float = 1.4,
        max_tableau: int = 10,
) -> CovarianceReport:
    """Compute covariance from a cost integrator using the accurate Hessian.

    The resulting sandwich covariance is ``V = H^{-1} B H^{-1}`` where ``H``
    is the Hessian of *integrator._cost_function* at *xi* and ``B`` is
    the meat matrix built from per-section score vectors.

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
    estimator. *Econometrica*, 48(4):817-838.

    Huber, P. J. and Ronchetti, E. M. (2009). *Robust Statistics*, 2nd ed.
    Wiley. Section 7.6 (sandwich formula for M-estimators).

    Parameters
    ----------
    integrator
        Any object exposing ``_cost_function(xi) -> float`` and
        ``inp_mat_keys`` (list of parameter names).  Both
        :class:`~dualmatfit.cost_fitting.CostIntegrator` and
        :class:`~dualmatfit.cost_fitting.CostFunction` satisfy this.
    xi : ndarray
        Fitted parameter vector (at the optimum).
    param_names :
        Material parameter names.  When *None* (default), names are taken
        from ``integrator.inp_mat_keys``.
    polish : bool
        Whether to apply eigenvalue polish.
    calibrate : bool
        Whether to apply Huang et al. (2017) NPD calibration when the
        Hessian remains non-positive-definite after polishing.
    confidence_level : float
        Confidence level for per-parameter intervals (default 0.95).
    beta : float
        Ridders' step-size reduction factor.
    max_tableau : int
        Maximum Neville-tableau rows.

    Returns
    -------
    CovarianceReport
        Report with ``method='accurate'``.
    """

    if param_names is None:
        param_names = integrator.inp_mat_keys

    cost_fun = integrator._cost_function
    cost_jacobian = integrator._cost_function_diff

    param_idx = np.asarray([integrator.inp_mat_keys.index(param_i) for param_i in param_names], dtype=int)  # noqa: F841

    eval_counter = _EvalCounter(cost_fun)

    # Hessian Calculation
    np_hessian, h_opt = accurate_hessian(eval_counter, xi, beta=beta, max_tableau=max_tableau)

    if polish:
        np_hessian = eigenvalue_polish(np_hessian, eval_counter, xi, recompute=True, beta=beta, max_tableau=max_tableau)

    # Huang et al. (2017) NPD calibration as fallback
    calibrated = False
    atol = 1.e-6
    if calibrate:
        hessian_min_eig = np.linalg.eigvalsh(np_hessian).min()
        if hessian_min_eig <= atol:
            n_npd = int(np.sum(np.linalg.eigvalsh(np_hessian) <= atol))
            logger.warning(
                "Hessian still has %d non-positive eigenvalue(s) after polish; "
                "applying Huang et al. (2017) calibration",
                n_npd,
            )
            np_hessian = huang_calibration(np_hessian, atol=atol)
            calibrated = True

    # Per-section score vectors for meat matrix.
    try:
        np_scores = cost_jacobian(xi, fsum=False, freg=False)
    except TypeError:
        np_scores = cost_jacobian(xi, freg=False)

    if np_scores.ndim == 1:
        np_scores = np_scores[np.newaxis, :]

    # Meat matrix: B = Sigma_i s_i s_i^T = S.T @ S  (rank <= n_functions)
    np_meat = np_scores.T @ np_scores

    # Sandwich: V = H^{-1} B H^{-1}  (solve twice, transposing between)
    H_inv_B = np.linalg.solve(np_hessian, np_meat)
    np_covariance = np.linalg.solve(np_hessian, H_inv_B.T).T

    # eigh for symmetric matrices: returns real eigenvalues in ascending order
    np_eigen_val, np_eigen_vec = np.linalg.eigh(np_hessian)
    hess_cond = np_eigen_val[-1] / np_eigen_val[0] if np_eigen_val[0] != 0 else np.inf

    se = np.sqrt(np.abs(np.diag(np_covariance)))
    correlation_matrix = _to_corr(np_covariance)

    pf_correlation = pd.DataFrame(correlation_matrix,
                                  index=integrator.inp_mat_keys,
                                  columns=integrator.inp_mat_keys)

    pd_covariance = pd.DataFrame(np_covariance,
                                 index=integrator.inp_mat_keys,
                                 columns=integrator.inp_mat_keys)

    # Eigenvalues are sorted ascending from eigh — keep them unlabelled
    # (they are rotated mixtures of parameters, labelling is misleading
    # for coupled modes).  For per-parameter identification use the
    # Hessian diagonal H[i,i] which is unambiguous.
    sr_eigen_values = pd.Series(
        np_eigen_val,
        index=np.arange(len(np_eigen_val)),
        name='Hessian Eigenvalues',
    )
    hess_diag = pd.Series(
        np.diag(np_hessian),
        index=integrator.inp_mat_keys,
        name='Hessian Diagonal',
    )

    # Confidence intervals: xi pm t_{alpha/2, dof} x SE, clipped to design bounds
    n_params = len(integrator.inp_mat_keys)
    try:
        n_obs = sum(cf.ncontrol for cf in integrator.cost_function)
    except (AttributeError, TypeError):
        n_obs = n_params + 1  # minimal fallback

    df_ci = _compute_confidence_interval(
        xi, se, n_obs, n_params, integrator.inp_mat_keys, confidence_level,
        bounds=getattr(integrator, 'xi_bounds', None),
    )

    return CovarianceReport(
        param_names=param_names,
        param_idx=param_idx,
        covariance_matrix=pd_covariance,
        standard_errors=se,
        correlation_matrix=pf_correlation,
        hessian_matrix=np_hessian,
        hessian_condition=hess_cond,
        eigenvalues=sr_eigen_values,
        hessian_diagonal=hess_diag,
        confidence_interval=df_ci,
        confidence_level=confidence_level,
        n_function_evals=eval_counter.count,
        polished=polish,
        calibrated=calibrated,
        method='accurate',
    )


#: Backward-compatible alias for :func:`covariance_from_cost`.
mle_covariance = robust_covariance_from_cost


def covariance_from_gauss_newton(
        integrator,
        xi: np.ndarray,
        param_names: Optional[Sequence[str]] = None,
) -> CovarianceReport:
    """Compute covariance using the Gauss-Newton approximation.

    Uses the Jacobian of the residuals to form the normal matrix
    ``J^T J`` and estimates the residual variance as ``s^2 = RSS / (m - n)``
    where ``RSS = Sigma r_i^2`` and ``m``, ``n`` are the number of residuals and
    parameters, respectively.

    ``V = s^2 (J^T J)^{-1}``

    Parameters
    ----------
    integrator
        Object exposing ``_residuum(xi)``, ``_residuum_diff(xi)``, and
        ``inp_mat_keys``.
    xi : ndarray
        Fitted parameter vector.
    param_names :
        Material Parameters

    Returns
    -------
    CovarianceReport
        Report with ``method='gauss_newton'``.
    """

    xi = np.asarray(xi, dtype=float)
    np_res = as_1d_residual(integrator._residuum(xi))
    np_jac = as_2d_jacobian(integrator._residuum_diff(xi))

    m, n = np_jac.shape
    RSS = float(np.sum(np_res ** 2))
    dof = max(m - n, 1)
    s2 = RSS / dof

    JtJ = np_jac.T @ np_jac
    JtJ_inv = np.linalg.pinv(JtJ)
    V = s2 * JtJ_inv

    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    se_outer = np.outer(se, se)
    se_outer = np.where(se_outer == 0.0, 1.0, se_outer)
    corr = V / se_outer

    eigenvalues = np.linalg.eigvalsh(JtJ)

    # Confidence intervals (dof already computed above from Jacobian shape)
    bounds = getattr(integrator, 'xi_bounds', None)
    ci_df = _compute_confidence_interval(
        xi, se, m, n, integrator.inp_mat_keys, bounds=bounds,
    )

    return CovarianceReport(
        param_names=tuple(integrator.inp_mat_keys),
        param_idx=np.arange(n),
        covariance_matrix=V,
        standard_errors=se,
        correlation_matrix=corr,
        hessian_matrix=JtJ,
        hessian_condition=float(eigenvalues.max() / max(eigenvalues.min(), _EPS)),
        eigenvalues=eigenvalues,
        hessian_diagonal=pd.Series(np.diag(JtJ), index=integrator.inp_mat_keys, name='Hessian Diagonal',),
        confidence_interval=ci_df,
        confidence_level=0.95,
        n_function_evals=0,
        polished=False,
        calibrated=False,
        method='gauss_newton',
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_covariance_report(report: CovarianceReport, path: Union[str, Path]) -> None:
    """Save a :class:`CovarianceReport` to a compressed ``.npz`` file.

    Parameters
    ----------
    report : CovarianceReport
        Report to persist.
    path : str or Path
        Destination file path (should end in ``.npz``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        param_names=np.asarray(report.param_names, dtype=str),
        param_idx=report.param_idx,
        covariance_matrix=report.covariance_matrix,
        standard_errors=report.standard_errors,
        correlation_matrix=report.correlation_matrix,
        hessian_matrix=report.hessian_matrix,
        hessian_condition=np.asarray(report.hessian_condition),
        eigenvalues=report.eigenvalues,
        hessian_diagonal=report.hessian_diagonal,
        hessian_diagonal_index=np.asarray(list(report.hessian_diagonal.index), dtype=str,),
        confidence_interval_lower=report.confidence_interval['lower'].values,
        confidence_interval_value=report.confidence_interval['value'].values,
        confidence_interval_upper=report.confidence_interval['upper'].values,
        confidence_interval_index=np.asarray(list(report.confidence_interval.index), dtype=str,),
        confidence_level=np.asarray(report.confidence_level),
        n_function_evals=np.asarray(report.n_function_evals),
        polished=np.asarray(report.polished),
        calibrated=np.asarray(report.calibrated),
        method=np.asarray(str(report.method)),
    )
    logger.debug("Saved CovarianceReport to %s", path)


def load_covariance_report(path: Union[str, Path]) -> CovarianceReport:
    """Load a :class:`CovarianceReport` from a ``.npz`` file.

    Parameters
    ----------
    path : str or Path
        File path to load.

    Returns
    -------
    CovarianceReport
    """
    path = Path(path)
    data = np.load(str(path), allow_pickle=False)
    param_names = tuple(data['param_names'].tolist())

    # Confidence interval (backward-compatible: missing -> zeros)
    if 'confidence_interval_lower' in data:
        ci_index = list(data['confidence_interval_index'])
        ci_value = data['confidence_interval_value'] if 'confidence_interval_value' in data else np.zeros(len(ci_index))
        ci_df = pd.DataFrame(
            {'lower': data['confidence_interval_lower'],
             'value': ci_value,
             'upper': data['confidence_interval_upper']},
            index=ci_index,
        )
        cl = float(data['confidence_level'])
    else:
        ci_df = pd.DataFrame(
            {'lower': np.zeros(len(param_names)),
             'value': np.zeros(len(param_names)),
             'upper': np.zeros(len(param_names))},
            index=list(param_names),
        )
        cl = 0.95

    return CovarianceReport(
        param_names=param_names,
        param_idx=data.get('param_idx', np.arange(len(param_names))),
        covariance_matrix=data['covariance_matrix'],
        standard_errors=data['standard_errors'],
        correlation_matrix=data['correlation_matrix'],
        hessian_matrix=data['hessian_matrix'],
        hessian_condition=float(data['hessian_condition']) if 'hessian_condition' in data else 0.0,
        eigenvalues=data['eigenvalues'],
        hessian_diagonal=pd.Series(
            data['hessian_diagonal'],
            index=list(data['hessian_diagonal_index']),
            name='Hessian Diagonal',
        ) if 'hessian_diagonal' in data else pd.Series(
            np.diag(data['hessian_matrix']),
            index=list(param_names),
            name='Hessian Diagonal',
        ),
        confidence_interval=ci_df,
        confidence_level=cl,
        n_function_evals=int(data['n_function_evals']),
        polished=bool(data['polished']),
        calibrated=bool(data['calibrated']) if 'calibrated' in data else False,
        method=str(data['method']),
    )


# ---------------------------------------------------------------------------
# Phase 8: Formatting helpers for AnisoMaterialFit integration
# ---------------------------------------------------------------------------

def format_params_with_uncertainty(
    params: pd.Series,
    report: CovarianceReport,
) -> pd.DataFrame:
    """Create a single-row DataFrame with parameter values and +/- sigma columns.

    Parameters
    ----------
    params : pd.Series
        Fitted parameter values, indexed by parameter name.
    report : CovarianceReport
        Covariance report whose ``param_names`` must be a subset of
        *params* index.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns ``[p1, p1 +/- sigma, p2, p2 +/- sigma, ...]``.

    Raises
    ------
    ValueError
        If the report's ``param_names`` are not all present in *params*.
    """
    missing = set(report.param_names) - set(params.index)
    if missing:
        raise ValueError(
            f"param_names in the CovarianceReport are not present in params: {missing}"
        )

    row: dict = {}
    for name, se in zip(report.param_names, report.standard_errors):
        row[name] = params[name]
        row[f'{name} +/- sigma'] = float(se)

    return pd.DataFrame([row])


def build_covariance_summary_table(
    params_dict: dict[str, pd.Series],
    reports_dict: dict[str, CovarianceReport],
) -> pd.DataFrame:
    """Aggregate per-section parameters and uncertainties into one table.

    Parameters
    ----------
    params_dict : dict
        ``{section_label: pd.Series}`` of fitted parameter values.
    reports_dict : dict
        ``{section_label: CovarianceReport}`` matching *params_dict* keys.

    Returns
    -------
    pd.DataFrame
        Rows indexed by section label, columns interleave values and SE.
    """
    if not params_dict:
        return pd.DataFrame()

    rows = []
    for label in params_dict:
        row_df = format_params_with_uncertainty(params_dict[label],
                                                reports_dict[label])
        row_df.index = [label]
        rows.append(row_df)

    return pd.concat(rows)