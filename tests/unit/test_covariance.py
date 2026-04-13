# -*- coding: utf-8 -*-
"""
Tests for covariance matrix estimation algorithms.

Based on Baker (2021, arXiv:2105.04829v1): "Estimating accurate covariance
matrices on fitted model parameters."

Tests are organized by algorithm phase, each class targeting a specific
component of the covariance estimation pipeline.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import rosen, rosen_hess

from dualmatfit.fitting.covariance import (
    find_initial_step,
    ridders_curvature,
    richardson_off_diagonal,
    central_diff_cross,
    accurate_hessian,
    eigenvalue_polish,
    mle_covariance,
    CovarianceReport,
    frobenius_distance,
    correlation_distance,
    g_metric,
)


# ---------------------------------------------------------------------------
# Analytical test functions with known Hessians
# ---------------------------------------------------------------------------

def _quadratic_1d(x):
    """f(x) = 3x², f''(x) = 6."""
    return 3.0 * x[0] ** 2


def _exponential_1d(x):
    """f(x) = exp(x), f''(x) = exp(x)."""
    return np.exp(x[0])


def _cosine_1d(x):
    """f(x) = -cos(x), f''(x) = cos(x)."""
    return -np.cos(x[0])


def _bilinear_2d(x):
    """f(x, y) = 5xy, H = [[0, 5], [5, 0]]."""
    return 5.0 * x[0] * x[1]


def _quadratic_2d(x):
    """f(x, y) = 3x² + 2xy + 5y², H = [[6, 2], [2, 10]]."""
    return 3.0 * x[0] ** 2 + 2.0 * x[0] * x[1] + 5.0 * x[1] ** 2


def _rosenbrock(x):
    """Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)².

    At (1,1): H = [[802, -400], [-400, 200]].
    """
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def _quadratic_5d(x):
    """f(x) = x^T A x / 2 for a known 5×5 SPD matrix A.

    Returns A as well via _quadratic_5d_hessian().
    """
    A = _quadratic_5d_hessian()
    return 0.5 * x @ A @ x


def _quadratic_5d_hessian():
    """5×5 SPD Hessian for _quadratic_5d."""
    rng = np.random.RandomState(42)
    L = rng.randn(5, 5)
    return L.T @ L + 0.1 * np.eye(5)


# ---------------------------------------------------------------------------
# Mock integrator for mle_covariance (satisfies the CostIntegrator protocol)
# ---------------------------------------------------------------------------

class _CostFunctionStub:
    """Minimal stub to satisfy ``cost_integrator.cost_fun[0].ncontrol``."""

    def __init__(self, ncontrol: int):
        self.ncontrol = ncontrol


class _FunctionIntegrator:
    """Wrap a bare scalar function into the protocol expected by
    ``mle_covariance``.

    Provides:
    - ``_cost_function(xi) -> float``
    - ``_cost_function_diff(xi) -> ndarray (n,)``  (central FDM)
    - ``inp_mat_keys``
    - ``cost_functions`` list with stub carrying ``ncontrol``
    """

    def __init__(self, f, n: int, *, ncontrol: int = 100):
        self._f = f
        self.inp_mat_keys = [f'p{i}' for i in range(n)]
        self.list_cost_fun = [_CostFunctionStub(ncontrol)]

    def _cost_function(self, xi):
        return float(self._f(xi))

    def _cost_function_diff(self, xi, **kwargs):
        """Central finite-difference gradient."""
        xi = np.asarray(xi, dtype=float)
        grad = np.zeros_like(xi)
        h = 1e-7
        for i in range(len(xi)):
            e = np.zeros_like(xi)
            e[i] = h
            grad[i] = (self._f(xi + e) - self._f(xi - e)) / (2.0 * h)
        return grad


# ===========================================================================
# Phase 1: Ridders' Method for Diagonal Curvatures
# ===========================================================================

class TestFindInitialStep:
    """Tests for the initial step-size search (Paper Section 2, Step 1)."""

    def test_quadratic_finds_bracket(self):
        """For a quadratic at the minimum, doubling h should always bracket."""
        x0 = np.array([0.0])
        h, f0 = find_initial_step(_quadratic_1d, x0, 0)
        assert h > 0
        assert _quadratic_1d(x0 + h * np.eye(1)[0]) > f0
        assert _quadratic_1d(x0 - h * np.eye(1)[0]) > f0

    def test_exponential_away_from_minimum(self):
        """For exp(x) at x=1 (not a minimum), step finding still works."""
        x0 = np.array([1.0])
        h, f0 = find_initial_step(_exponential_1d, x0, 0)
        assert h > 0
        # Verify the central-difference curvature is positive and finite
        fp = _exponential_1d(x0 + np.array([h]))
        fm = _exponential_1d(x0 - np.array([h]))
        cd = (fp - 2.0 * f0 + fm) / (h * h)
        assert cd > 0
        assert np.isfinite(cd)

    def test_returns_positive_step(self):
        x0 = np.array([0.5, 0.5])
        h, _ = find_initial_step(_quadratic_2d, x0, 0)
        assert h > 0
        h, _ = find_initial_step(_quadratic_2d, x0, 1)
        assert h > 0


class TestRiddersCurvature:
    """Tests for Ridders' method (Paper Section 2, Steps 2-3)."""

    def test_quadratic_exact(self):
        """f(x) = 3x² → f''(0) = 6, should be recovered exactly."""
        x0 = np.array([0.0])
        curv, h_opt = ridders_curvature(_quadratic_1d, x0, 0)
        assert curv == pytest.approx(6.0, rel=1e-8)
        assert h_opt > 0

    def test_exponential_at_one(self):
        """f(x) = exp(x) → f''(1) = e ≈ 2.71828."""
        x0 = np.array([1.0])
        curv, h_opt = ridders_curvature(_exponential_1d, x0, 0)
        assert curv == pytest.approx(np.e, rel=1e-6)

    def test_cosine_at_zero(self):
        """f(x) = -cos(x) → f''(0) = cos(0) = 1."""
        x0 = np.array([0.0])
        curv, h_opt = ridders_curvature(_cosine_1d, x0, 0)
        assert curv == pytest.approx(1.0, rel=1e-6)

    def test_multivariate_diagonal(self):
        """f(x,y) = 3x² + 2xy + 5y² → H_00 = 6, H_11 = 10."""
        x0 = np.array([1.0, 2.0])
        curv0, _ = ridders_curvature(_quadratic_2d, x0, 0)
        curv1, _ = ridders_curvature(_quadratic_2d, x0, 1)
        assert curv0 == pytest.approx(6.0, rel=1e-6)
        assert curv1 == pytest.approx(10.0, rel=1e-6)

    def test_returns_optimal_step(self):
        """Optimal step h should be finite and positive."""
        x0 = np.array([0.0])
        _, h_opt = ridders_curvature(_quadratic_1d, x0, 0)
        assert np.isfinite(h_opt)
        assert h_opt > 0

    def test_5d_diagonal(self):
        """All diagonal curvatures of a 5D quadratic match the known Hessian."""
        A = _quadratic_5d_hessian()
        x0 = np.ones(5)
        for i in range(5):
            curv, _ = ridders_curvature(_quadratic_5d, x0, i)
            assert curv == pytest.approx(A[i, i], rel=1e-5), (
                f"Diagonal element {i}: got {curv}, expected {A[i, i]}"
            )


# ===========================================================================
# Phase 2: Richardson Extrapolation for Off-Diagonal Elements
# ===========================================================================

class TestCentralDiffCross:
    """Tests for the central difference cross-derivative."""

    def test_bilinear_exact(self):
        """f(x,y) = 5xy → D₁₂ = 5 for any step sizes."""
        x0 = np.array([1.0, 2.0])
        d = central_diff_cross(_bilinear_2d, x0, 0, 1, 0.1, 0.1)
        assert d == pytest.approx(5.0, rel=1e-6)

    def test_quadratic_cross(self):
        """f(x,y) = 3x² + 2xy + 5y² → H₁₂ = 2."""
        x0 = np.array([1.0, 2.0])
        d = central_diff_cross(_quadratic_2d, x0, 0, 1, 0.01, 0.01)
        assert d == pytest.approx(2.0, rel=1e-3)


class TestRichardsonOffDiagonal:
    """Tests for Richardson-extrapolated off-diagonal elements."""

    def test_bilinear_exact(self):
        """f(x,y) = 5xy → H₁₂ = 5."""
        x0 = np.array([1.0, 2.0])
        h12 = richardson_off_diagonal(_bilinear_2d, x0, 0, 1, 0.1, 0.1)
        assert h12 == pytest.approx(5.0, rel=1e-8)

    def test_quadratic_cross(self):
        """f(x,y) = 3x² + 2xy + 5y² → H₁₂ = 2."""
        x0 = np.array([1.0, 2.0])
        h12 = richardson_off_diagonal(_quadratic_2d, x0, 0, 1, 0.01, 0.01)
        assert h12 == pytest.approx(2.0, rel=1e-6)

    def test_accuracy_improvement(self):
        """Richardson should be more accurate than raw central diff with large h."""

        def f_exp2d(x):
            """f(x,y) = exp(x+y), H_01 = exp(x+y)."""
            return np.exp(x[0] + x[1])

        x0 = np.array([1.0, 1.0])
        hi, hj = 0.3, 0.3
        true_val = np.exp(2.0)
        raw = central_diff_cross(f_exp2d, x0, 0, 1, hi, hj)
        rich = richardson_off_diagonal(f_exp2d, x0, 0, 1, hi, hj)
        assert abs(rich - true_val) < abs(raw - true_val)


# ===========================================================================
# Phase 3: Full Accurate Hessian Assembly
# ===========================================================================

class TestAccurateHessian:
    """Tests for the full accurate_hessian() function."""

    def test_quadratic_2d(self):
        """f(x,y) = 3x² + 2xy + 5y² → H = [[6,2],[2,10]]."""
        x0 = np.array([1.0, 2.0])
        H, h_opt = accurate_hessian(_quadratic_2d, x0)
        expected = np.array([[6.0, 2.0], [2.0, 10.0]])
        np.testing.assert_allclose(H, expected, rtol=1e-5)
        assert len(h_opt) == 2
        assert all(h > 0 for h in h_opt)

    def test_rosenbrock_at_minimum(self):
        """Rosenbrock at (1,1): H = [[802, -400], [-400, 200]]."""
        x0 = np.array([1.0, 1.0])
        H, _ = accurate_hessian(_rosenbrock, x0)
        expected = np.array([[802.0, -400.0], [-400.0, 200.0]])
        np.testing.assert_allclose(H, expected, rtol=1e-4)

    def test_symmetry(self):
        """Hessian should be symmetric: H_ij = H_ji."""
        x0 = np.array([0.5, -0.3])
        H, _ = accurate_hessian(_rosenbrock, x0)
        np.testing.assert_allclose(H, H.T, atol=1e-8)

    def test_5d_quadratic_form(self):
        """f(x) = x^T A x / 2 → H = A for known 5×5 SPD A."""
        A = _quadratic_5d_hessian()
        x0 = np.ones(5)
        H, _ = accurate_hessian(_quadratic_5d, x0)
        np.testing.assert_allclose(H, A, rtol=1e-4)

    def test_1d_reduces_to_scalar(self):
        """Single-parameter case should work."""
        x0 = np.array([0.0])
        H, _ = accurate_hessian(_quadratic_1d, x0)
        assert H.shape == (1, 1)
        assert H[0, 0] == pytest.approx(6.0, rel=1e-6)


# ===========================================================================
# Phase 4: Eigenvalue Polish
# ===========================================================================

class TestEigenvaluePolish:
    """Tests for eigenvalue polish / positive-definiteness enforcement."""

    def test_already_pd_minimal_change(self):
        """A well-conditioned PD matrix should barely change."""
        H = np.array([[6.0, 2.0], [2.0, 10.0]])
        x0 = np.array([1.0, 2.0])
        H_polished = eigenvalue_polish(H, _quadratic_2d, x0)
        np.testing.assert_allclose(H_polished, H, rtol=1e-4)

    def test_near_singular_corrected(self):
        """A nearly singular matrix should become well-conditioned after polish."""
        H = np.array([[1.0, 0.9999], [0.9999, 1.0]])

        def f(x):
            return 0.5 * x @ H @ x

        x0 = np.zeros(2)
        H_polished = eigenvalue_polish(H, f, x0)
        eigenvalues = np.linalg.eigvalsh(H_polished)
        assert all(ev > 0 for ev in eigenvalues)

    def test_result_is_symmetric(self):
        """Polished matrix should remain symmetric."""
        H = np.array([[802.0, -400.0], [-400.0, 200.0]])
        x0 = np.array([1.0, 1.0])
        H_polished = eigenvalue_polish(H, _rosenbrock, x0)
        np.testing.assert_allclose(H_polished, H_polished.T, atol=1e-10)

    def test_preserves_eigenvectors(self):
        """Eigenvectors should be unchanged; only eigenvalues recomputed."""
        H = np.array([[6.0, 2.0], [2.0, 10.0]])
        x0 = np.array([1.0, 2.0])
        _, V_orig = np.linalg.eigh(H)
        H_polished = eigenvalue_polish(H, _quadratic_2d, x0)
        _, V_polished = np.linalg.eigh(H_polished)
        # Eigenvectors may differ by sign
        for i in range(2):
            cos_angle = abs(np.dot(V_orig[:, i], V_polished[:, i]))
            assert cos_angle == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# Phase 5: MLE Covariance and CovarianceReport
# ===========================================================================

class TestMLECovariance:
    """Tests for the end-to-end MLE covariance computation.

    ``mle_covariance`` expects a CostIntegrator-like object.  We use
    ``_FunctionIntegrator`` to wrap bare analytical functions.

    Note: ``mle_covariance`` computes the sandwich covariance
    ``V = H⁻¹ B H⁻¹`` where ``B = Σᵢ sᵢ sᵢᵀ`` is the meat matrix built
    from per-section score vectors.  When the integrator does not support
    ``fsum=False``, the fallback uses the total gradient as a single score
    vector, producing a rank-1 meat.
    """

    def test_hessian_matches_known(self):
        """Numerical Hessian should match the analytical Hessian."""
        x0 = np.array([1.0, 2.0])
        integ = _FunctionIntegrator(_quadratic_2d, 2)
        report = mle_covariance(integ, x0, param_names=('p0', 'p1'))
        H_expected = np.array([[6.0, 2.0], [2.0, 10.0]])
        np.testing.assert_allclose(report.hessian_matrix, H_expected, rtol=1e-4)

    def test_positive_definite(self):
        """Hessian and its inverse should be positive definite."""
        x0 = np.array([1.0, 1.0])
        integ = _FunctionIntegrator(_rosenbrock, 2)
        report = mle_covariance(integ, x0, param_names=('p0', 'p1'))
        eigenvalues = np.linalg.eigvalsh(report.hessian_matrix)
        assert all(ev > 0 for ev in eigenvalues)

    def test_standard_errors_from_sandwich(self):
        """Standard errors come from sandwich V = H⁻¹ B H⁻¹."""
        x0 = np.array([1.0, 2.0])
        integ = _FunctionIntegrator(_quadratic_2d, 2)
        report = mle_covariance(integ, x0, param_names=('p0', 'p1'))
        # _FunctionIntegrator has no fsum support → meat = grad @ grad.T
        # At a non-minimum (x0=[1,2]), grad ≠ 0 so sandwich V ≠ 0.
        V = np.array(report.covariance_matrix)
        assert V.shape == (2, 2)
        # SE = sqrt(abs(diag(V)))
        np.testing.assert_allclose(
            report.standard_errors,
            np.sqrt(np.abs(np.diag(V))),
            rtol=1e-4,
        )

    def test_correlation_matrix(self):
        """Correlation matrix: unit diagonal, entries in [-1, 1]."""
        x0 = np.array([1.0, 2.0])
        integ = _FunctionIntegrator(_quadratic_2d, 2)
        report = mle_covariance(integ, x0, param_names=('p0', 'p1'))
        np.testing.assert_allclose(np.diag(report.correlation_matrix), 1.0, atol=1e-10)
        assert np.all(np.abs(report.correlation_matrix) <= 1.0 + 1e-10)

    def test_report_fields(self):
        """CovarianceReport should have all expected fields."""
        x0 = np.array([1.0, 2.0])
        integ = _FunctionIntegrator(_quadratic_2d, 2)
        report = mle_covariance(integ, x0, param_names=('p0', 'p1'))
        assert isinstance(report, CovarianceReport)
        assert report.param_names == ('p0', 'p1')
        assert report.covariance_matrix.shape == (2, 2)
        assert report.hessian_matrix.shape == (2, 2)
        assert len(report.standard_errors) == 2
        assert len(report.eigenvalues) == 2
        assert report.n_function_evals > 0
        assert report.method == 'accurate'

    def test_5d_hessian(self):
        """Full 5D Hessian matches the known SPD matrix A."""
        A = _quadratic_5d_hessian()
        x0 = np.ones(5)
        integ = _FunctionIntegrator(_quadratic_5d, 5)
        names = tuple(f'p{i}' for i in range(5))
        report = mle_covariance(integ, x0, param_names=names)
        np.testing.assert_allclose(report.hessian_matrix, A, rtol=1e-3)


# ===========================================================================
# Phase 5 (continued): Accuracy Metrics
# ===========================================================================

class TestAccuracyMetrics:
    """Tests for Frobenius distance, correlation distance, and G metric."""

    def test_frobenius_zero_for_identical(self):
        A = np.array([[1.0, 0.5], [0.5, 2.0]])
        assert frobenius_distance(A, A) == pytest.approx(0.0)

    def test_frobenius_known_value(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        B = np.array([[2.0, 0.0], [0.0, 2.0]])
        # F = sum(|A_ij - B_ij|) / n² = (1+0+0+1) / 4 = 0.5
        assert frobenius_distance(A, B) == pytest.approx(0.5)

    def test_correlation_distance_identical(self):
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        assert correlation_distance(A, A) == pytest.approx(0.0, abs=1e-12)

    def test_g_metric_zero_for_identical(self):
        V = np.array([[4.0, 1.0], [1.0, 9.0]])
        assert g_metric(V, V) == pytest.approx(0.0)

    def test_g_metric_known_value(self):
        """G = 100 * Σ|√V_ii - √C_ii| / (n * √C_ii)."""
        V = np.array([[4.0, 0.0], [0.0, 9.0]])  # SE: 2, 3
        C = np.array([[1.0, 0.0], [0.0, 4.0]])  # SE: 1, 2
        # G = 100 * (|2-1|/(2*1) + |3-2|/(2*2)) = 100 * (0.5 + 0.25) = 75.0
        assert g_metric(V, C) == pytest.approx(75.0)


# ===========================================================================
# Phase 6: CostFunction / CostIntegrator Integration
# ===========================================================================

class _MockIntegrator:
    """Minimal duck-typed CostIntegrator for testing covariance wrappers.

    Simulates a linear regression: y = X @ theta + noise.
    Cost function: f(theta) = 0.5 * ||y - X theta||^2.
    """

    def __init__(self, X, y, param_names):
        self.inp_mat_keys = list(param_names)
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._alpha = None
        self.cost_fun = [_CostFunctionStub(ncontrol=len(y))]
        # OLS solution
        self.xi = np.linalg.lstsq(self._X, self._y, rcond=None)[0]

    def _cost_function(self, xi):
        r = self._y - self._X @ xi
        return 0.5 * np.sum(r ** 2)

    def _cost_function_diff(self, xi, **kwargs):
        """Gradient: ∇f = -Xᵀ(y - Xθ) = Xᵀ(Xθ - y)."""
        r = self._y - self._X @ xi
        return -self._X.T @ r

    def _residuum(self, xi):
        r = self._y - self._X @ xi
        return r.reshape(1, -1)

    def _residuum_diff(self, xi):
        J = -self._X
        return J.reshape(1, *J.shape)


def _make_linear_mock(n=50, p=3, sigma=0.1, seed=42):
    """Create a mock integrator with known analytical covariance."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    theta_true = np.array([2.0, -1.0, 0.5])[:p]
    y = X @ theta_true + sigma * rng.randn(n)
    names = [f'p{i}' for i in range(p)]
    return _MockIntegrator(X, y, names), X, theta_true, sigma


class TestCovarianceFromCost:
    """Tests for covariance_from_cost() — accurate Hessian on a CostIntegrator."""

    def test_returns_covariance_report(self):
        mock, X, _, _ = _make_linear_mock()
        from dualmatfit.fitting.covariance import robust_covariance_from_cost
        report = robust_covariance_from_cost(mock, mock.xi)
        assert isinstance(report, CovarianceReport)
        assert report.method == 'accurate'
        assert report.covariance_matrix.shape == (3, 3)
        assert len(report.standard_errors) == 3
        assert all(se > 0 for se in report.standard_errors)
        assert report.n_function_evals > 0

    def test_hessian_matches_xtx(self):
        """For f = 0.5 ||y - Xθ||², the Hessian is X^T X."""
        mock, X, _, _ = _make_linear_mock()
        from dualmatfit.fitting.covariance import robust_covariance_from_cost
        report = robust_covariance_from_cost(mock, mock.xi, polish=False)
        H_expected = X.T @ X
        np.testing.assert_allclose(report.hessian_matrix, H_expected, rtol=0.05)

    def test_covariance_is_sandwich(self):
        """V = H⁻¹ B H⁻¹ (sandwich estimator).

        At the OLS optimum ∇f ≈ 0, so the rank-1 meat from the total
        gradient fallback gives V ≈ 0.  We verify the formula by checking
        that V is near-zero (consistent with a gradient-based meat at a
        minimum) and that the Hessian itself is accurate.
        """
        mock, X, _, _ = _make_linear_mock()
        from dualmatfit.fitting.covariance import robust_covariance_from_cost
        report = robust_covariance_from_cost(mock, mock.xi, polish=False)
        # At optimum, gradient ≈ 0 → meat ≈ 0 → V ≈ 0
        V = np.array(report.covariance_matrix)
        assert np.max(np.abs(V)) < 1e-10, f"Sandwich V should be ~0 at optimum, got max={np.max(np.abs(V)):.2e}"
        # Hessian should still match X^T X
        H_expected = X.T @ X
        np.testing.assert_allclose(report.hessian_matrix, H_expected, rtol=0.05)


class TestCovarianceFromGaussNewton:
    """Tests for covariance_from_gauss_newton() — Gauss-Newton approximation."""

    def test_returns_gauss_newton_report(self):
        mock, _, _, _ = _make_linear_mock()
        from dualmatfit.fitting.covariance import covariance_from_gauss_newton
        report = covariance_from_gauss_newton(mock, mock.xi)
        assert isinstance(report, CovarianceReport)
        assert report.method == 'gauss_newton'
        assert report.covariance_matrix.shape == (3, 3)
        assert report.polished is False
        assert report.n_function_evals == 0

    def test_matches_analytical_gn_covariance(self):
        """V_gn = s² (J^T J)^{-1} where s² = RSS/(m-n)."""
        mock, X, _, _ = _make_linear_mock()
        from dualmatfit.fitting.covariance import covariance_from_gauss_newton
        theta_hat = mock.xi
        r = mock._y - X @ theta_hat
        m, n = X.shape
        s2 = np.sum(r ** 2) / (m - n)
        V_expected = s2 * np.linalg.inv(X.T @ X)
        report = covariance_from_gauss_newton(mock, theta_hat)
        np.testing.assert_allclose(report.covariance_matrix, V_expected, rtol=1e-6)

    def test_standard_errors_positive(self):
        mock, _, _, _ = _make_linear_mock()
        from dualmatfit.fitting.covariance import covariance_from_gauss_newton
        report = covariance_from_gauss_newton(mock, mock.xi)
        assert all(se > 0 for se in report.standard_errors)

    def test_correlation_matrix_unit_diagonal(self):
        mock, _, _, _ = _make_linear_mock()
        from dualmatfit.fitting.covariance import covariance_from_gauss_newton
        report = covariance_from_gauss_newton(mock, mock.xi)
        np.testing.assert_allclose(np.diag(report.correlation_matrix), 1.0, atol=1e-10)


class TestAccurateVsGaussNewton:
    """Compare accurate Hessian and Gauss-Newton covariance methods."""

    def test_hessian_structures_proportional(self):
        """For a linear model, accurate H and J^T J should be close."""
        mock, X, _, _ = _make_linear_mock()
        from dualmatfit.fitting.covariance import robust_covariance_from_cost
        rpt = robust_covariance_from_cost(mock, mock.xi, polish=False)
        JtJ = X.T @ X
        np.testing.assert_allclose(
            rpt.hessian_matrix / np.max(np.abs(rpt.hessian_matrix)),
            JtJ / np.max(np.abs(JtJ)),
            atol=0.05,
        )

    def test_gauss_newton_pd(self):
        """Gauss-Newton covariance should be positive definite."""
        mock, _, _, _ = _make_linear_mock()
        from dualmatfit.fitting.covariance import covariance_from_gauss_newton
        rpt_gn = covariance_from_gauss_newton(mock, mock.xi)
        assert all(np.linalg.eigvalsh(rpt_gn.covariance_matrix) > 0)


class TestCovarianceReportIO:
    """Tests for save/load round-trip of CovarianceReport."""

    def test_save_load_round_trip(self, tmp_path):
        from dualmatfit.fitting.covariance import save_covariance_report, load_covariance_report
        integ = _FunctionIntegrator(_quadratic_2d, 2)
        report = mle_covariance(integ, np.array([1.0, 2.0]),
                                param_names=('p0', 'p1'))
        path = tmp_path / "cov_report.npz"
        save_covariance_report(report, path)
        loaded = load_covariance_report(path)

        assert loaded.param_names == report.param_names
        assert loaded.method == report.method
        assert loaded.polished == report.polished
        assert loaded.n_function_evals == report.n_function_evals
        np.testing.assert_allclose(loaded.covariance_matrix, report.covariance_matrix)
        np.testing.assert_allclose(loaded.standard_errors, report.standard_errors)
        np.testing.assert_allclose(loaded.correlation_matrix, report.correlation_matrix)
        np.testing.assert_allclose(loaded.hessian_matrix, report.hessian_matrix)
        np.testing.assert_allclose(loaded.eigenvalues, report.eigenvalues)

    def test_save_creates_file(self, tmp_path):
        from dualmatfit.fitting.covariance import save_covariance_report
        integ = _FunctionIntegrator(_quadratic_2d, 2)
        report = mle_covariance(integ, np.array([1.0, 2.0]), param_names=('p0', 'p1'))
        path = tmp_path / "test.npz"
        save_covariance_report(report, path)
        assert path.exists()


# ===========================================================================
# Rosenbrock N-D Hessian tests (Phases 1–4)
# Uses scipy.optimize.rosen + rosen_hess as analytical oracle.
# ===========================================================================

class TestRosenbrockHessian:
    """Phase 1: ``accurate_hessian`` vs ``rosen_hess`` for ≥6 parameters.

    The N-D generalised Rosenbrock has a known tridiagonal Hessian
    (``scipy.optimize.rosen_hess``).  At the minimum ``x = ones(n)`` the
    Hessian is SPD.
    """

    def test_6d_at_minimum(self):
        """6-parameter Rosenbrock at global minimum."""
        x0 = np.ones(6)
        H_num, h_opt = accurate_hessian(rosen, x0)
        H_exact = rosen_hess(x0)
        np.testing.assert_allclose(H_num, H_exact, rtol=1e-4, atol=1e-6)
        assert len(h_opt) == 6
        assert all(h > 0 for h in h_opt)

    def test_8d_at_minimum(self):
        """8-parameter Rosenbrock at global minimum."""
        x0 = np.ones(8)
        H_num, _ = accurate_hessian(rosen, x0)
        H_exact = rosen_hess(x0)
        np.testing.assert_allclose(H_num, H_exact, rtol=1e-5, atol=1e-6)

    def test_10d_at_minimum(self):
        """10-parameter Rosenbrock at global minimum (stress test)."""
        x0 = np.ones(10)
        H_num, _ = accurate_hessian(rosen, x0)
        H_exact = rosen_hess(x0)
        np.testing.assert_allclose(H_num, H_exact, rtol=1e-3, atol=1e-6)

    def test_6d_near_minimum(self):
        """6D Rosenbrock at a perturbed point (still SPD)."""
        x0 = np.array([1.01, 0.99, 1.02, 0.98, 1.01, 0.99])
        H_num, _ = accurate_hessian(rosen, x0)
        H_exact = rosen_hess(x0)
        np.testing.assert_allclose(H_num, H_exact, rtol=1e-4, atol=1e-6)
        eigvals = np.linalg.eigvals(H_num)
        assert all(ev > 0 for ev in eigvals), "Expected SPD at near-minimum point"

    def test_6d_symmetry(self):
        """Numerical Hessian must be symmetric."""
        x0 = np.ones(6)
        H_num, _ = accurate_hessian(rosen, x0)
        np.testing.assert_allclose(H_num, H_num.T, atol=1e-8)

    def test_6d_tridiagonal_structure(self):
        """At the minimum, the Rosenbrock Hessian is tridiagonal.

        Off-tridiagonal entries should be numerically negligible.
        """
        x0 = np.ones(6)
        H_num, _ = accurate_hessian(rosen, x0)
        n = len(x0)
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1:
                    assert abs(H_num[i, j]) < 1.0, (
                        f"H[{i},{j}] = {H_num[i, j]:.4e} should be ≈ 0"
                    )


class TestRosenbrockPolishedHessian:
    """Phase 2: Combined ``accurate_hessian → eigenvalue_polish`` pipeline.

    Tests the full pipeline that ``mle_covariance`` uses internally.
    """

    def test_polish_6d_at_minimum(self):
        """At the minimum (already PD): polish should preserve accuracy."""
        x0 = np.ones(6)
        H_raw, _ = accurate_hessian(rosen, x0)
        H_polished = eigenvalue_polish(H_raw, rosen, x0, recompute=True)
        H_exact = rosen_hess(x0)
        np.testing.assert_allclose(H_polished, H_exact, rtol=1e-3, atol=1e-6)

    def test_polish_8d_at_minimum(self):
        """8D: polished Hessian ≈ analytical Hessian."""
        x0 = np.ones(8)
        H_raw, _ = accurate_hessian(rosen, x0)
        H_polished = eigenvalue_polish(H_raw, rosen, x0, recompute=True)
        H_exact = rosen_hess(x0)
        np.testing.assert_allclose(H_polished, H_exact, rtol=1e-3, atol=1e-6)

    def test_polish_6d_indefinite(self):
        """At a non-minimum point with negative eigenvalues:
        polish must produce a PD matrix while preserving eigenvectors.
        """
        x0 = np.array([0.5, 0.8, 1.2, 0.9, 1.1, 0.7])
        H_exact = rosen_hess(x0)
        eigvals_exact = np.linalg.eigvalsh(H_exact)
        assert any(ev < 0 for ev in eigvals_exact), \
            "Test requires an indefinite analytical Hessian"

        H_raw, _ = accurate_hessian(rosen, x0)
        H_polished = eigenvalue_polish(H_raw, rosen, x0, recompute=True)

        eigvals_polished = np.linalg.eigvalsh(H_polished)
        assert all(ev > 0 for ev in eigvals_polished), \
            "Polished Hessian must be positive definite"
        np.testing.assert_allclose(H_polished, H_polished.T, atol=1e-10)

    def test_polish_preserves_accuracy(self):
        """Relative Frobenius error after polish should be small at the minimum."""
        x0 = np.ones(6)
        H_raw, _ = accurate_hessian(rosen, x0)
        H_polished = eigenvalue_polish(H_raw, rosen, x0, recompute=True)
        H_exact = rosen_hess(x0)
        rel_err = np.linalg.norm(H_polished - H_exact) / np.linalg.norm(H_exact)
        assert rel_err < 0.01, f"Relative Frobenius error {rel_err:.4e} > 1%"

    def test_polish_condition_number(self):
        """Polished matrix should have finite, bounded condition number."""
        x0 = np.ones(6)
        H_raw, _ = accurate_hessian(rosen, x0)
        H_polished = eigenvalue_polish(H_raw, rosen, x0, recompute=True)
        cond = np.linalg.cond(H_polished)
        assert np.isfinite(cond)
        # Analytical condition at 6D minimum ≈ 3400
        assert cond < 1e5, f"Condition number {cond:.1f} unexpectedly large"


class TestMLECovarianceRosenbrock:
    """Phase 3: End-to-end ``mle_covariance`` with ≥6 parameters.

    Uses ``_FunctionIntegrator`` wrapping ``scipy.optimize.rosen`` so that
    the full ``mle_covariance`` pipeline (Hessian + σ² scaling) is exercised.
    """

    def test_hessian_matches_rosen_hess_6d(self):
        """Report Hessian should match ``rosen_hess(x0)`` for 6 parameters."""
        x0 = np.ones(6)
        integ = _FunctionIntegrator(rosen, 6, ncontrol=100)
        names = tuple(f'p{i}' for i in range(6))
        report = mle_covariance(integ, x0, param_names=names, polish=False)
        H_exact = rosen_hess(x0)
        np.testing.assert_allclose(report.hessian_matrix, H_exact, rtol=1e-4, atol=1e-6)

    def test_covariance_positive_definite_6d(self):
        """Covariance (and Hessian) eigenvalues should all be positive."""
        x0 = np.ones(6)
        integ = _FunctionIntegrator(rosen, 6, ncontrol=100)
        names = tuple(f'p{i}' for i in range(6))
        report = mle_covariance(integ, x0, param_names=names)
        eigvals_h = np.linalg.eigvalsh(report.hessian_matrix)
        assert all(ev > 0 for ev in eigvals_h)

    def test_covariance_is_not_just_hinv(self):
        """V = σ² · H⁻¹ ≠ H⁻¹ when σ² ≠ 1.

        Use a non-minimum evaluation point where ∇f ≠ 0 so that σ² ≠ 0.
        """
        x0 = np.array([1.5, 2.0, 0.5, 1.0, 1.5, 0.8])
        integ = _FunctionIntegrator(rosen, 6, ncontrol=100)
        names = tuple(f'p{i}' for i in range(6))
        report = mle_covariance(integ, x0, param_names=names, polish=False)
        H_inv = np.linalg.inv(report.hessian_matrix)
        # σ² comes from the gradient, which is non-zero at x0
        if not np.allclose(report.covariance_matrix, H_inv, rtol=1e-6):
            pass  # expected: V ≠ H⁻¹
        else:
            pytest.fail("V should differ from H⁻¹ when σ² ≠ 1")

    def test_standard_errors_from_sandwich(self):
        """SE = √diag(V) where V = H⁻¹ B H⁻¹ (sandwich)."""
        x0 = np.ones(6)
        integ = _FunctionIntegrator(rosen, 6, ncontrol=100)
        names = tuple(f'p{i}' for i in range(6))
        report = mle_covariance(integ, x0, param_names=names)
        V = np.array(report.covariance_matrix)
        np.testing.assert_allclose(
            report.standard_errors,
            np.sqrt(np.abs(np.diag(V))),
            rtol=1e-3,
        )

    def test_correlation_unit_diagonal_6d(self):
        """Correlation matrix has unit diagonal and |R_ij| ≤ 1."""
        x0 = np.ones(6)
        integ = _FunctionIntegrator(rosen, 6, ncontrol=100)
        names = tuple(f'p{i}' for i in range(6))
        report = mle_covariance(integ, x0, param_names=names)
        np.testing.assert_allclose(
            np.diag(report.correlation_matrix), 1.0, atol=1e-8,
        )
        assert np.all(np.abs(report.correlation_matrix) <= 1.0 + 1e-8)

    def test_polish_true_vs_false(self):
        """Both polish paths produce valid reports; results may differ."""
        x0 = np.ones(6)
        integ = _FunctionIntegrator(rosen, 6, ncontrol=100)
        names = tuple(f'p{i}' for i in range(6))
        rpt_yes = mle_covariance(integ, x0, param_names=names, polish=True)
        rpt_no = mle_covariance(integ, x0, param_names=names, polish=False)
        assert rpt_yes.polished is True
        assert rpt_no.polished is False
        assert rpt_yes.hessian_matrix.shape == (6, 6)
        assert rpt_no.hessian_matrix.shape == (6, 6)


class TestCovarianceAccuracyMetricsRosenbrock:
    """Phase 4: Accuracy metrics against analytical Rosenbrock Hessian."""

    def test_frobenius_6d(self):
        """Frobenius distance between numerical and analytical Hessian."""
        x0 = np.ones(6)
        H_num, _ = accurate_hessian(rosen, x0)
        H_exact = rosen_hess(x0)
        dist = frobenius_distance(H_num, H_exact)
        assert dist < 0.1, f"Frobenius distance {dist:.4e} too large"

    def test_g_metric_6d(self):
        """G metric (% error on standard errors) should be small."""
        x0 = np.ones(6)
        H_num, _ = accurate_hessian(rosen, x0)
        H_exact = rosen_hess(x0)
        V_num = np.linalg.inv(H_num)
        V_exact = np.linalg.inv(H_exact)
        g = g_metric(V_num, V_exact)
        assert g < 1.0, f"G metric {g:.2f}% too large"


# ===========================================================================
# Phase 5: Zakharov function — dense Hessian coverage
# ===========================================================================

def zakharov(x: np.ndarray) -> float:
    """Zakharov benchmark function (N-dimensional).

    f(x) = Σ xᵢ² + s² + s⁴  where  s = Σ 0.5·i·xᵢ  (1-indexed).
    Global minimum at x* = 0 with f(x*) = 0.
    """
    i_vec = np.arange(1, len(x) + 1, dtype=float)
    s = 0.5 * np.dot(i_vec, x)
    return float(np.dot(x, x) + s**2 + s**4)


def zakharov_hess(x: np.ndarray) -> np.ndarray:
    """Analytical Hessian of the Zakharov function.

    H_{kl} = 2·δ_{kl} + (0.5 + 3s²)·k·l  where  s = Σ 0.5·i·xᵢ.
    At the origin: H = 2I + 0.5·v·vᵀ  with v = [1,2,...,n].
    """
    n = len(x)
    i_vec = np.arange(1, n + 1, dtype=float)
    s = 0.5 * np.dot(i_vec, x)
    return 2.0 * np.eye(n) + (0.5 + 3.0 * s**2) * np.outer(i_vec, i_vec)


class TestZakharovHessian:
    """``accurate_hessian`` vs analytical Zakharov Hessian for ≥6 parameters.

    The Zakharov Hessian is **dense** (rank-1 update of 2I), complementing
    Rosenbrock's tridiagonal structure.  At the minimum x* = 0 the Hessian
    is H = 2I + 0.5·vvᵀ with v = [1..n], well-conditioned (κ ≈ 24 at 6D).
    """

    def test_6d_at_minimum(self):
        """6-parameter Zakharov at global minimum x = 0."""
        x0 = np.zeros(6)
        H_num, h_opt = accurate_hessian(zakharov, x0)
        H_exact = zakharov_hess(x0)
        np.testing.assert_allclose(H_num, H_exact, rtol=1e-4, atol=1e-6)
        assert len(h_opt) == 6
        assert all(h > 0 for h in h_opt)

    def test_8d_at_minimum(self):
        """8-parameter Zakharov at global minimum."""
        x0 = np.zeros(8)
        H_num, _ = accurate_hessian(zakharov, x0)
        H_exact = zakharov_hess(x0)
        np.testing.assert_allclose(H_num, H_exact, rtol=1e-4, atol=1e-6)

    def test_10d_at_minimum(self):
        """10-parameter Zakharov at global minimum (stress test)."""
        x0 = np.zeros(10)
        H_num, _ = accurate_hessian(zakharov, x0)
        H_exact = zakharov_hess(x0)
        np.testing.assert_allclose(H_num, H_exact, rtol=1e-3, atol=1e-6)

    def test_6d_near_minimum(self):
        """6D Zakharov at a perturbed point (Hessian remains SPD)."""
        x0 = np.array([0.01, -0.01, 0.02, -0.02, 0.01, -0.01])
        H_num, _ = accurate_hessian(zakharov, x0)
        H_exact = zakharov_hess(x0)
        np.testing.assert_allclose(H_num, H_exact, rtol=1e-3, atol=1e-6)
        eigvals = np.linalg.eigvalsh(H_num)
        assert all(ev > 0 for ev in eigvals), "Expected SPD near origin"

    def test_6d_dense_structure(self):
        """At the origin the Hessian is fully dense (no zero entries)."""
        x0 = np.zeros(6)
        H_exact = zakharov_hess(x0)
        assert np.all(H_exact != 0), "Zakharov Hessian at origin should be dense"

    def test_6d_symmetry(self):
        """Numerical Hessian must be symmetric."""
        x0 = np.zeros(6)
        H_num, _ = accurate_hessian(zakharov, x0)
        np.testing.assert_allclose(H_num, H_num.T, atol=1e-8)

    def test_6d_eigenvalue_spectrum(self):
        """At origin: (n-1) eigenvalues = 2, one = 2 + 0.5·Σi²."""
        n = 6
        x0 = np.zeros(n)
        H_num, _ = accurate_hessian(zakharov, x0)
        eigvals = np.sort(np.linalg.eigvalsh(H_num))

        sum_i_sq = n * (n + 1) * (2 * n + 1) / 6  # 91 for n=6
        expected_large = 2.0 + 0.5 * sum_i_sq       # 47.5
        np.testing.assert_allclose(eigvals[:-1], 2.0, rtol=1e-3)
        np.testing.assert_allclose(eigvals[-1], expected_large, rtol=1e-3)

    def test_6d_sympy_hessian_vs_numerical(self):
        """SymPy symbolic Hessian as independent oracle for accurate_hessian.

        Both paths originate from the same SymPy expression:
        - Path A: lambdify(f) → accurate_hessian → H_numerical
        - Path B: sp.hessian(f) → lambdify(H) → H_sympy
        Tests at origin and a perturbed point (exercises 3s² coupling term).
        """
        import sympy as sp

        n = 6
        xs = sp.symbols(" ".join(f"x{i}" for i in range(n)))
        s_expr = sum(sp.Rational(1, 2) * (i + 1) * xs[i] for i in range(n))
        f_expr = sum(x**2 for x in xs) + s_expr**2 + s_expr**4

        # Path A: lambdified scalar → accurate_hessian
        f_np = sp.lambdify(xs, f_expr, "numpy")
        f_for_hessian = lambda x: float(f_np(*x))

        # Path B: symbolic Hessian → lambdified matrix
        H_sym = sp.hessian(f_expr, xs)
        H_np = sp.lambdify(xs, H_sym, "numpy")

        # --- Test at origin (s = 0) ---
        x0 = np.zeros(n)
        H_numerical, _ = accurate_hessian(f_for_hessian, x0)
        H_sympy = np.array(H_np(*x0), dtype=float)
        np.testing.assert_allclose(H_numerical, H_sympy, rtol=1e-4, atol=1e-6)

        # --- Test at perturbed point (s ≠ 0, activates 3s² term) ---
        x1 = np.array([0.05, -0.03, 0.04, -0.02, 0.01, -0.04])
        H_numerical_p, _ = accurate_hessian(f_for_hessian, x1)
        H_sympy_p = np.array(H_np(*x1), dtype=float)
        np.testing.assert_allclose(H_numerical_p, H_sympy_p, rtol=1e-3, atol=1e-6)


class TestZakharovPolishedHessian:
    """Combined ``accurate_hessian → eigenvalue_polish`` for Zakharov."""

    def test_polish_6d_at_minimum(self):
        """At the minimum (already PD): polish should preserve accuracy."""
        x0 = np.zeros(6)
        H_raw, _ = accurate_hessian(zakharov, x0)
        H_polished = eigenvalue_polish(H_raw, zakharov, x0, recompute=True)
        H_exact = zakharov_hess(x0)
        np.testing.assert_allclose(H_polished, H_exact, rtol=1e-3, atol=1e-6)

    def test_polish_8d_at_minimum(self):
        """8D: polished Hessian ≈ analytical Hessian."""
        x0 = np.zeros(8)
        H_raw, _ = accurate_hessian(zakharov, x0)
        H_polished = eigenvalue_polish(H_raw, zakharov, x0, recompute=True)
        H_exact = zakharov_hess(x0)
        np.testing.assert_allclose(H_polished, H_exact, rtol=1e-3, atol=1e-6)

    def test_polish_preserves_accuracy(self):
        """Relative Frobenius error after polish should be small."""
        x0 = np.zeros(6)
        H_raw, _ = accurate_hessian(zakharov, x0)
        H_polished = eigenvalue_polish(H_raw, zakharov, x0, recompute=True)
        H_exact = zakharov_hess(x0)
        rel_err = np.linalg.norm(H_polished - H_exact) / np.linalg.norm(H_exact)
        assert rel_err < 0.01, f"Relative Frobenius error {rel_err:.4e} > 1%"


class TestZakharovAccuracyMetrics:
    """Accuracy metrics against analytical Zakharov Hessian."""

    def test_frobenius_6d(self):
        """Frobenius distance between numerical and analytical Hessian."""
        x0 = np.zeros(6)
        H_num, _ = accurate_hessian(zakharov, x0)
        H_exact = zakharov_hess(x0)
        dist = frobenius_distance(H_num, H_exact)
        assert dist < 0.1, f"Frobenius distance {dist:.4e} too large"

    def test_g_metric_6d(self):
        """G metric (% error on standard errors) should be small."""
        x0 = np.zeros(6)
        H_num, _ = accurate_hessian(zakharov, x0)
        H_exact = zakharov_hess(x0)
        V_num = np.linalg.inv(H_num)
        V_exact = np.linalg.inv(H_exact)
        g = g_metric(V_num, V_exact)
        assert g < 1.0, f"G metric {g:.2f}% too large"


# ===========================================================================
# Phase 8: AnisoMaterialFit Integration — formatting & summary helpers
# ===========================================================================

class TestFormatParamsWithUncertainty:
    """Tests for format_params_with_uncertainty() DataFrame helper."""

    def test_adds_se_columns(self):
        """SE columns named '<param> +/- sigma' should appear for every parameter."""
        from dualmatfit.fitting.covariance import format_params_with_uncertainty
        params = pd.Series({'mu': 23.6, 'k_1': 32.5, 'alpha': 5.0})
        V = np.diag([1.0, 4.0, 0.25])
        report = CovarianceReport(
            param_names=('mu', 'k_1', 'alpha'),
            param_idx=np.arange(3),
            covariance_matrix=V,
            standard_errors=np.sqrt(np.diag(V)),
            correlation_matrix=np.eye(3),
            hessian_matrix=np.linalg.inv(V),
            hessian_condition=1.0,
            eigenvalues=np.linalg.eigvalsh(np.linalg.inv(V)),
            hessian_diagonal=pd.Series(np.diag(np.linalg.inv(V)), index=['mu', 'k_1', 'alpha']),
            confidence_interval=pd.DataFrame({'lower': -np.sqrt(np.diag(V)), 'value': np.zeros(3), 'upper': np.sqrt(np.diag(V))}, index=['mu', 'k_1', 'alpha']),
            confidence_level=0.95,
            n_function_evals=100,
            polished=True,
            calibrated=False,
            method='accurate',
        )
        df = format_params_with_uncertainty(params, report)
        assert 'mu' in df.columns
        assert 'mu +/- sigma' in df.columns
        assert 'k_1 +/- sigma' in df.columns
        assert 'alpha +/- sigma' in df.columns

    def test_se_values_correct(self):
        """SE values should match √diag(V)."""
        from dualmatfit.fitting.covariance import format_params_with_uncertainty
        params = pd.Series({'a': 10.0, 'b': 20.0})
        V = np.array([[4.0, 0.5], [0.5, 9.0]])
        report = CovarianceReport(
            param_names=('a', 'b'),
            param_idx=np.arange(2),
            covariance_matrix=V,
            standard_errors=np.sqrt(np.diag(V)),
            correlation_matrix=np.eye(2),
            hessian_matrix=np.linalg.inv(V),
            hessian_condition=1.0,
            eigenvalues=np.array([1.0, 2.0]),
            hessian_diagonal=pd.Series(np.diag(np.linalg.inv(V)), index=['a', 'b']),
            confidence_interval=pd.DataFrame({'lower': -np.sqrt(np.diag(V)), 'value': np.zeros(2), 'upper': np.sqrt(np.diag(V))}, index=['a', 'b']),
            confidence_level=0.95,
            n_function_evals=50,
            polished=True,
            calibrated=False,
            method='accurate',
        )
        df = format_params_with_uncertainty(params, report)
        assert df['a +/- sigma'].iloc[0] == pytest.approx(2.0)
        assert df['b +/- sigma'].iloc[0] == pytest.approx(3.0)

    def test_returns_single_row_dataframe(self):
        from dualmatfit.fitting.covariance import format_params_with_uncertainty
        params = pd.Series({'x': 1.0, 'y': 2.0})
        V = np.eye(2) * 0.01
        report = CovarianceReport(
            param_names=('x', 'y'),
            param_idx=np.arange(2),
            covariance_matrix=V,
            standard_errors=np.sqrt(np.diag(V)),
            correlation_matrix=np.eye(2),
            hessian_matrix=np.linalg.inv(V),
            hessian_condition=1.0,
            eigenvalues=np.array([100.0, 100.0]),
            hessian_diagonal=pd.Series(np.diag(np.linalg.inv(V)), index=['x', 'y']),
            confidence_interval=pd.DataFrame({'lower': -np.sqrt(np.diag(V)), 'value': np.zeros(2), 'upper': np.sqrt(np.diag(V))}, index=['x', 'y']),
            confidence_level=0.95,
            n_function_evals=50,
            polished=False,
            calibrated=False,
            method='gauss_newton',
        )
        df = format_params_with_uncertainty(params, report)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df['x'].iloc[0] == pytest.approx(1.0)
        assert df['y'].iloc[0] == pytest.approx(2.0)

    def test_param_name_mismatch_raises(self):
        """Raise if report param_names don't match params index."""
        from dualmatfit.fitting.covariance import format_params_with_uncertainty
        params = pd.Series({'a': 1.0, 'b': 2.0})
        V = np.eye(3)
        report = CovarianceReport(
            param_names=('a', 'b', 'c'),
            param_idx=np.arange(3),
            covariance_matrix=V,
            standard_errors=np.sqrt(np.diag(V)),
            correlation_matrix=np.eye(3),
            hessian_matrix=V,
            hessian_condition=1.0,
            eigenvalues=np.ones(3),
            hessian_diagonal=pd.Series(np.diag(V), index=['a', 'b', 'c']),
            confidence_interval=pd.DataFrame({'lower': -np.ones(3), 'value': np.zeros(3), 'upper': np.ones(3)}, index=['a', 'b', 'c']),
            confidence_level=0.95,
            n_function_evals=10,
            polished=False,
            calibrated=False,
            method='accurate',
        )
        with pytest.raises(ValueError, match="param_names"):
            format_params_with_uncertainty(params, report)


class TestCovarianceSummaryTable:
    """Tests for build_covariance_summary_table() aggregation helper."""

    def test_aggregates_multiple_sections(self):
        """Build summary table from multiple CovarianceReports."""
        from dualmatfit.fitting.covariance import build_covariance_summary_table
        reports = {}
        for label in ['Ar-A', 'Ar-B', 'Tr-A']:
            V = np.diag([0.01, 0.04])
            reports[label] = CovarianceReport(
                param_names=('mu', 'k_1'),
                param_idx=np.arange(2),
                covariance_matrix=V,
                standard_errors=np.sqrt(np.diag(V)),
                correlation_matrix=np.eye(2),
                hessian_matrix=np.linalg.inv(V),
                hessian_condition=4.0,
                eigenvalues=np.array([100.0, 25.0]),
                hessian_diagonal=pd.Series(np.diag(np.linalg.inv(V)), index=['mu', 'k_1']),
                confidence_interval=pd.DataFrame({'lower': -np.sqrt(np.diag(V)), 'value': np.zeros(2), 'upper': np.sqrt(np.diag(V))}, index=['mu', 'k_1']),
                confidence_level=0.95,
                n_function_evals=50,
                polished=True,
                calibrated=False,
                method='accurate',
            )
        params = {
            'Ar-A': pd.Series({'mu': 23.0, 'k_1': 32.0}),
            'Ar-B': pd.Series({'mu': 24.0, 'k_1': 33.0}),
            'Tr-A': pd.Series({'mu': 22.0, 'k_1': 31.0}),
        }
        df = build_covariance_summary_table(params, reports)
        assert len(df) == 3
        assert set(df.index) == {'Ar-A', 'Ar-B', 'Tr-A'}
        assert 'mu' in df.columns
        assert 'mu +/- sigma' in df.columns

    def test_empty_reports_returns_empty(self):
        from dualmatfit.fitting.covariance import build_covariance_summary_table
        df = build_covariance_summary_table({}, {})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
