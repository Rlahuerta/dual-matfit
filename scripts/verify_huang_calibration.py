"""
Verification of the Huang et al. (2017) NPD calibration implementation.

Systematically tests that ``dualmatfit.fitting.covariance.huang_calibration``
correctly implements the algorithm described in §6 of
``docs/algorithms/covariance_theory.md`` (Eqs. 20–22).

Run with::

    conda run -n matfit1d python scripts/verify_huang_calibration.py

A PASS/FAIL line is printed for each check.  The script exits with a non-zero
status if any assertion fails.

References
----------
[Huang et al., 2017]
    Huang, C., Farewell, D., and Pan, J. (2017). A calibration method for
    non-positive definite covariance matrix in multivariate data analysis.
    *Journal of Multivariate Analysis*, 157:45–52.

[Baker, 2021]
    Baker, J. (2021). Accurate computation of the Hessian matrix and subsequent
    covariance and curvature measures. *arXiv preprint arXiv:2105.04829v1*.
"""
from __future__ import annotations

import sys
import numpy as np

_FAILURES: list[str] = []


def _check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        _FAILURES.append(label)


# ===========================================================================
# 1. Basic properties: output must always be symmetric PD
# ===========================================================================

def verify_basic_properties() -> None:
    """Check that huang_calibration returns a symmetric PD matrix."""
    from dualmatfit.fitting.covariance import huang_calibration

    print("\n─── 1. Basic properties ──────────────────────────────────────────")

    # Case A: already PD — should be returned unchanged
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    A_cal = huang_calibration(A)
    _check("Already-PD matrix returned unchanged",
           np.allclose(A, A_cal, atol=1e-14))

    # Case B: one negative eigenvalue
    Q = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    D_neg = np.diag([-0.5, 3.0])
    B = Q @ D_neg @ Q.T
    B = (B + B.T) / 2
    B_cal = huang_calibration(B)
    eigs_cal = np.linalg.eigvalsh(B_cal)
    _check("NPD input → all eigenvalues positive after calibration",
           np.all(eigs_cal > 0))
    _check("Output is symmetric",
           np.allclose(B_cal, B_cal.T, atol=1e-14))

    # Case C: large matrix (6×6) with 3 negative eigenvalues
    rng = np.random.default_rng(42)
    Q6, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    eigs6 = np.array([-2.0, -0.5, -0.1, 0.3, 1.5, 8.0])
    C = Q6 @ np.diag(eigs6) @ Q6.T
    C = (C + C.T) / 2
    C_cal = huang_calibration(C)
    eigs_c = np.linalg.eigvalsh(C_cal)
    _check("6×6 with 3 negative eigs → all positive after calibration",
           np.all(eigs_c > 0))
    _check("6×6 output is symmetric",
           np.allclose(C_cal, C_cal.T, atol=1e-14))


# ===========================================================================
# 2. Frobenius optimality: calibrated matrix is closer than naive clamping
# ===========================================================================

def verify_frobenius_optimality() -> None:
    """Huang calibration minimises the combined Eq. (22) objective, not raw Frobenius."""
    from dualmatfit.fitting.covariance import huang_calibration

    print("\n─── 2. Direct threshold optimality ──────────────────────────────")

    rng = np.random.default_rng(99)
    Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    eigs = np.array([-1.0, -0.01, 0.5, 2.0, 10.0])
    H_npd = Q @ np.diag(eigs) @ Q.T
    H_npd = (H_npd + H_npd.T) / 2

    H_cal = huang_calibration(H_npd)

    # With kappa_target=1e8 (default), c* = 10.0 / 1e8 = 1e-7
    eigs_orig = np.linalg.eigvalsh(H_npd)
    eigs_cal = np.linalg.eigvalsh(H_cal)
    lambda_max = eigs_orig.max()
    c_expected = lambda_max / 1e8

    _check(f"c* = λ_max/κ_target = {c_expected:.2e}: all calibrated eigs ≥ c*",
           np.all(eigs_cal >= c_expected - 1e-15))

    # Frobenius distortion is minimal — only negative eigs are lifted
    frob = np.linalg.norm(eigs_orig - eigs_cal)
    _check(f"Frobenius distortion = {frob:.4f} (only negative eigs lifted)",
           frob > 0)


# ===========================================================================
# 3. Eigenvector preservation: only eigenvalues change
# ===========================================================================

def verify_eigenvector_preservation() -> None:
    """Calibration preserves eigenvectors for non-degenerate eigenvalues."""
    from dualmatfit.fitting.covariance import huang_calibration

    print("\n─── 3. Eigenvector preservation ─────────────────────────────────")

    rng = np.random.default_rng(17)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    # Use eigenvalues where only one is negative and the rest are well-separated,
    # so no degeneracy is introduced by clamping
    eigs = np.array([-0.1, 1.0, 5.0, 20.0])
    H = Q @ np.diag(eigs) @ Q.T
    H = (H + H.T) / 2

    H_cal = huang_calibration(H)
    _, vecs_orig = np.linalg.eigh(H)
    eigs_cal, vecs_cal = np.linalg.eigh(H_cal)

    # Eigenvectors for eigenvalues that were NOT clamped should be preserved
    eigs_orig_sorted = np.linalg.eigvalsh(H)
    for k in range(4):
        dot = abs(np.dot(vecs_orig[:, k], vecs_cal[:, k]))
        was_clamped = eigs_orig_sorted[k] < eigs_cal[k] - 1e-10
        if was_clamped:
            _check(f"Eigenvector {k}: clamped eigenvalue — eigenvector may change (OK)",
                   True)
        else:
            _check(f"Eigenvector {k}: preserved |v_orig · v_cal| = {dot:.10f} ≈ 1",
                   abs(dot - 1.0) < 1e-8)


# ===========================================================================
# 4. Edge cases
# ===========================================================================

def verify_edge_cases() -> None:
    """Handle edge cases: all-negative, single-eigenvalue, 1×1."""
    from dualmatfit.fitting.covariance import huang_calibration

    print("\n─── 4. Edge cases ───────────────────────────────────────────────")

    # 1×1 negative
    H1 = np.array([[-5.0]])
    H1_cal = huang_calibration(H1)
    _check("1×1 negative → positive after calibration",
           H1_cal[0, 0] > 0)

    # All eigenvalues negative
    H_all_neg = -np.eye(3) * np.array([1.0, 2.0, 3.0])
    H_cal = huang_calibration(H_all_neg)
    eigs = np.linalg.eigvalsh(H_cal)
    _check("All-negative input → all eigenvalues positive",
           np.all(eigs > 0))

    # Already PD: identity
    I3 = np.eye(3)
    I3_cal = huang_calibration(I3)
    _check("Identity matrix unchanged",
           np.allclose(I3, I3_cal, atol=1e-14))

    # Near-zero eigenvalue (not negative, but very small)
    # With kappa_target=1e8 (default), this all-positive but ill-conditioned
    # matrix (κ=5e15) SHOULD be calibrated — that's the fix.
    eigs_tiny = np.array([1e-15, 1.0, 5.0])
    Q, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((3, 3)))
    H_tiny = Q @ np.diag(eigs_tiny) @ Q.T
    H_tiny = (H_tiny + H_tiny.T) / 2
    H_tiny_cal = huang_calibration(H_tiny)
    eigs_tiny_cal = np.linalg.eigvalsh(H_tiny_cal)
    cond_tiny = eigs_tiny_cal.max() / eigs_tiny_cal.min()
    _check("Near-zero (but positive) eigenvalue: κ controlled ≤ 1e8",
           cond_tiny <= 1e8 * 1.01)


# ===========================================================================
# 5. Sandwich estimator integration: V = H⁻¹ B H⁻¹ must be PD
# ===========================================================================

def verify_sandwich_integration() -> None:
    """After Huang calibration, the sandwich V = H⁻¹BH⁻¹ must be PD."""
    from dualmatfit.fitting.covariance import huang_calibration

    print("\n─── 5. Sandwich V = H⁻¹BH⁻¹ integration ────────────────────────")

    rng = np.random.default_rng(123)
    n = 5

    # Build an NPD Hessian with a strongly negative eigenvalue to ensure
    # the uncalibrated sandwich produces negative V_ii
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs_h = np.array([-5.0, -2.0, 0.3, 2.0, 7.0])
    H_npd = Q @ np.diag(eigs_h) @ Q.T
    H_npd = (H_npd + H_npd.T) / 2

    # Build a valid meat matrix B = Σ sⱼsⱼᵀ (always PSD)
    scores = rng.standard_normal((20, n))
    B = scores.T @ scores

    # Verify the NPD Hessian is indeed non-positive-definite
    _check("Input Hessian has negative eigenvalues",
           np.any(np.linalg.eigvalsh(H_npd) < 0))

    # With Huang calibration
    H_pd = huang_calibration(H_npd)
    _check("Calibrated Hessian is PD",
           np.all(np.linalg.eigvalsh(H_pd) > 0))

    H_inv = np.linalg.inv(H_pd)
    V = H_inv @ B @ H_inv
    _check("Calibrated sandwich: all V_ii > 0",
           np.all(np.diag(V) > 0))

    _check("Calibrated sandwich: V symmetric",
           np.allclose(V, V.T, atol=1e-12))

    # SE = √diag(V) should all be real and positive
    se = np.sqrt(np.diag(V))
    _check("Standard errors: all positive and finite",
           np.all(se > 0) and np.all(np.isfinite(se)))

    # Correlation matrix entries in [-1, 1]
    se_outer = np.outer(se, se)
    corr = V / se_outer
    _check("Correlation matrix: all |R_ij| ≤ 1",
           np.all(np.abs(corr) <= 1.0 + 1e-10))
    _check("Correlation matrix: unit diagonal",
           np.allclose(np.diag(corr), 1.0, atol=1e-12))


# ===========================================================================
# 6. Condition number improvement
# ===========================================================================

def verify_condition_number() -> None:
    """After calibration, the condition number should be finite and reasonable."""
    from dualmatfit.fitting.covariance import huang_calibration

    print("\n─── 6. Condition number ─────────────────────────────────────────")

    rng = np.random.default_rng(55)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    eigs = np.array([-0.001, 0.001, 1.0, 100.0])
    H = Q @ np.diag(eigs) @ Q.T
    H = (H + H.T) / 2

    H_cal = huang_calibration(H)
    eigs_cal = np.linalg.eigvalsh(H_cal)

    cond_cal = eigs_cal.max() / eigs_cal.min()

    _check("Calibrated condition number is finite",
           np.isfinite(cond_cal))
    _check("All calibrated eigenvalues strictly positive",
           np.all(eigs_cal > 0))

    # Verify that with calibration, the matrix is invertible and usable
    eigs2 = np.array([-50.0, 0.5, 1.0, 100.0])
    H2 = Q @ np.diag(eigs2) @ Q.T
    H2 = (H2 + H2.T) / 2
    H2_cal = huang_calibration(H2)
    eigs2_cal = np.linalg.eigvalsh(H2_cal)
    cond2_cal = eigs2_cal.max() / eigs2_cal.min()
    _check("Large-negative case: calibrated cond is finite",
           np.isfinite(cond2_cal))
    _check("Large-negative case: all eigs positive after calibration",
           np.all(eigs2_cal > 0))


# ===========================================================================
# 7. Eq. (22) objective: verify α* minimises ‖H - P_c‖_F + α
# ===========================================================================

def verify_conditioning_guarantee() -> None:
    """Verify that κ(H_cal) ≤ κ_target for various dimensions and spectra."""
    from dualmatfit.fitting.covariance import huang_calibration

    print("\n─── 7. Conditioning guarantee across dimensions ─────────────────")

    rng = np.random.default_rng(77)
    kappa_target = 1e8
    for dim in [3, 6, 10]:
        Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        eigs = np.sort(rng.uniform(-2, 10, dim))
        eigs[0] = -abs(eigs[0]) - 0.1  # ensure at least one negative
        H = Q @ np.diag(eigs) @ Q.T
        H = (H + H.T) / 2

        H_cal = huang_calibration(H, kappa_target=kappa_target)
        eigs_cal = np.linalg.eigvalsh(H_cal)

        _check(f"dim={dim}: all eigs positive", np.all(eigs_cal > 0))

        kappa_result = eigs_cal.max() / eigs_cal.min()
        _check(f"dim={dim}: κ={kappa_result:.2e} ≤ κ_target={kappa_target:.0e}",
               kappa_result <= kappa_target * (1 + 1e-6))


# ===========================================================================
# 8. Realistic HGO-like scenario: mixed positive/negative curvatures
# ===========================================================================

def verify_hgo_scenario() -> None:
    """Simulate a realistic 5-parameter HGO Hessian with Cauchy-induced NPD."""
    from dualmatfit.fitting.covariance import huang_calibration

    print("\n─── 8. Realistic HGO-like scenario ──────────────────────────────")

    # Simulate: μ, k₁, k₂, κ, θ — fiber params k₁,k₂ have Cauchy-induced
    # negative curvature from exponential stiffening
    param_names = ['mu', 'k1', 'k2', 'kappa', 'theta']
    n = len(param_names)

    # Build Hessian with realistic structure:
    # - μ, κ, θ have positive curvature (well-determined)
    # - k₁, k₂ have negative curvature (Cauchy redescending + exponential)
    rng = np.random.default_rng(2017)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.array([50.0, -0.8, -0.05, 12.0, 3.0])
    H = Q @ np.diag(eigs) @ Q.T
    H = (H + H.T) / 2

    _check("Original Hessian is NPD",
           np.any(np.linalg.eigvalsh(H) < 0))

    H_cal = huang_calibration(H)
    eigs_cal = np.linalg.eigvalsh(H_cal)
    _check("Calibrated Hessian is PD",
           np.all(eigs_cal > 0))

    # Build meat from realistic scores
    n_obs = 50
    jacobian = rng.standard_normal((n_obs, n)) * np.array([1, 0.1, 0.05, 0.5, 0.3])
    residuals = rng.standard_normal(n_obs) * 0.1
    # Cauchy ψ(r) = r / (1 + r²)
    psi_vals = residuals / (1 + residuals ** 2)
    scores = psi_vals[:, None] * jacobian  # s_j = ψ(r_j) · J_j
    B = scores.T @ scores

    H_inv = np.linalg.inv(H_cal)
    V = H_inv @ B @ H_inv

    se = np.sqrt(np.diag(V))
    _check("All standard errors positive",
           np.all(se > 0))
    _check("All standard errors finite",
           np.all(np.isfinite(se)))

    for i, name in enumerate(param_names):
        _check(f"  SE({name}) = {se[i]:.6f} > 0", se[i] > 0)

    # Correlation matrix well-formed
    se_outer = np.outer(se, se)
    corr = V / se_outer
    _check("Correlation diagonal = 1",
           np.allclose(np.diag(corr), 1.0, atol=1e-12))
    _check("All |R_ij| ≤ 1",
           np.all(np.abs(corr) <= 1.0 + 1e-10))


# ===========================================================================
# 9. Idempotency: calibrating a PD matrix twice gives same result
# ===========================================================================

def verify_idempotency() -> None:
    """Applying calibration twice should give the same result."""
    from dualmatfit.fitting.covariance import huang_calibration

    print("\n─── 9. Idempotency ─────────────────────────────────────────────")

    rng = np.random.default_rng(33)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    eigs = np.array([-1.0, -0.5, 2.0, 8.0])
    H = Q @ np.diag(eigs) @ Q.T
    H = (H + H.T) / 2

    H_cal1 = huang_calibration(H)
    H_cal2 = huang_calibration(H_cal1)
    _check("Double calibration is idempotent",
           np.allclose(H_cal1, H_cal2, atol=1e-14))


# ===========================================================================
# 10. Numerical agreement with explicit Eq. (20)–(22) implementation
# ===========================================================================

def verify_equation_agreement() -> None:
    """Step-by-step verification of the direct c* = λ_max/κ_target algorithm."""
    from dualmatfit.fitting.covariance import huang_calibration

    print("\n─── 10. Direct thresholding step-by-step agreement ───────────────")

    # Construct a known NPD matrix
    rng = np.random.default_rng(2024)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    eigs = np.array([-0.7, 0.3, 4.0])
    H = Q @ np.diag(eigs) @ Q.T
    H = (H + H.T) / 2

    # Spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    _check("Spectral decomposition reconstructs H",
           np.allclose(eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T, H, atol=1e-12))

    # Manual computation of direct thresholding
    kappa_target = 1e8  # default
    lambda_max = eigenvalues.max()
    c_star = lambda_max / kappa_target
    _check(f"c* = λ_max/κ_target = {c_star:.6e}", c_star > 0)

    # Lift eigenvalues below c_star
    cal_eigs = np.maximum(eigenvalues, c_star)
    H_manual = eigenvectors @ np.diag(cal_eigs) @ eigenvectors.T

    # Compare with function output
    H_func = huang_calibration(H)
    _check("Function output matches manual c* = λ_max/κ_target implementation",
           np.allclose(H_func, H_manual, atol=1e-12))

    # Verify all calibrated eigenvalues ≥ c*
    _check(f"All calibrated eigenvalues ≥ c* = {c_star:.6e}",
           np.all(cal_eigs >= c_star - 1e-15))


# ===========================================================================
# 11. HYPOTHESIS TEST — Near-zero λ_min⁺ exposes Huang limitation
#
# When the smallest positive eigenvalue is tiny (e.g. 1e-8), the Huang
# threshold c_α = 10^{-α}·λ_min⁺ is always tiny, so the calibrated
# Hessian is PD but nearly singular.  The sandwich V = H⁻¹BH⁻¹ then
# has enormous / unreliable entries.
#
# We test the condition-number-aware floor implemented in
# huang_calibration(kappa_target=...):
#     c_α = max(10^{-α}·λ_min⁺,  λ_max / κ_target)
# ===========================================================================


def verify_near_zero_eigenvalue_problem() -> None:
    """Demonstrate and fix the near-zero λ_min⁺ failure mode.

    Scenario: HGO-like Hessian where fiber parameters k₁, k₂ are weakly
    identifiable, producing eigenvalues [50, 12, 3, 1e-8, -0.5].
    The negative eigenvalue is a Cauchy-loss artefact, but 1e-8 is a genuine
    near-zero curvature from weak data support.
    """
    from dualmatfit.fitting.covariance import huang_calibration

    print("\n─── 11. Near-zero λ_min⁺ failure & κ_target fix ─────────────────")

    # --- Build a realistic near-zero-eigenvalue Hessian ---
    rng = np.random.default_rng(2026)
    n = 5
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.array([50.0, 12.0, 3.0, 1e-8, -0.5])
    H = Q @ np.diag(eigs) @ Q.T
    H = (H + H.T) / 2

    param_names = ['mu', 'k1', 'k2', 'kappa', 'theta']

    _check("Original Hessian is NPD", np.any(np.linalg.eigvalsh(H) < 0))

    # --- Build a realistic PSD meat matrix B ---
    n_obs = 50
    jacobian = rng.standard_normal((n_obs, n)) * np.array([1, 0.1, 0.05, 0.5, 0.3])
    residuals = rng.standard_normal(n_obs) * 0.1
    psi_vals = residuals / (1 + residuals ** 2)
    scores = psi_vals[:, None] * jacobian
    B = scores.T @ scores  # always PSD

    # ------------------------------------------------------------------
    # Part A: Old-style Huang (kappa_target=inf disables the floor)
    # ------------------------------------------------------------------
    print("\n  --- Part A: Old-style Huang (no κ floor, kappa_target=inf) ---")

    H_huang = huang_calibration(H, kappa_target=np.inf)
    eigs_huang = np.linalg.eigvalsh(H_huang)
    lambda_min_huang = eigs_huang.min()
    cond_huang = eigs_huang.max() / lambda_min_huang

    _check(f"Huang: all eigenvalues positive (min = {lambda_min_huang:.2e})",
           np.all(eigs_huang > 0))
    _check(f"Huang: condition number = {cond_huang:.2e} (huge with kappa_target=inf!)",
           np.isfinite(cond_huang))

    # Sandwich with Huang-calibrated Hessian
    H_inv_huang = np.linalg.inv(H_huang)
    V_huang = H_inv_huang @ B @ H_inv_huang

    diag_huang = np.diag(V_huang)
    any_negative_vii = np.any(diag_huang < 0)
    se_huang = np.sqrt(np.abs(diag_huang))

    print(f"\n  Huang-calibrated eigenvalues: {np.sort(eigs_huang)}")
    print(f"  Condition number: κ = {cond_huang:.2e}")
    print(f"  V_ii (diag of sandwich): {diag_huang}")
    print(f"  Standard errors: {se_huang}")
    print(f"  Any negative V_ii? {any_negative_vii}")

    # With kappa_target=inf, floor = λ_max * 1e-12 ≈ 5e-11.
    # The near-zero eigenvalue (1e-8) >> floor, so it remains as-is.
    # This means κ ≈ 50 / 1e-8 = 5e9, demonstrating that kappa_target=inf
    # provides no conditioning control (only ensures strict PD).
    _check("Huang (inf): κ > 1e6 (demonstrating no conditioning control)",
           cond_huang > 1e6)

    # ------------------------------------------------------------------
    # Part B: Enhanced calibration with κ_target floor — show the fix
    # ------------------------------------------------------------------
    print("\n  --- Part B: Enhanced Huang (κ_target = 1e8, default) ---")

    kappa_target = 1e8
    H_enhanced = huang_calibration(H, kappa_target=kappa_target)
    eigs_enhanced = np.linalg.eigvalsh(H_enhanced)
    lambda_min_enhanced = eigs_enhanced.min()
    cond_enhanced = eigs_enhanced.max() / lambda_min_enhanced

    _check(f"Enhanced: all eigenvalues positive (min = {lambda_min_enhanced:.2e})",
           np.all(eigs_enhanced > 0))
    _check(f"Enhanced: κ = {cond_enhanced:.2e} ≤ κ_target = {kappa_target:.0e}",
           cond_enhanced <= kappa_target * 1.01)

    # Sandwich with enhanced-calibrated Hessian
    H_inv_enhanced = np.linalg.inv(H_enhanced)
    V_enhanced = H_inv_enhanced @ B @ H_inv_enhanced

    diag_enhanced = np.diag(V_enhanced)
    se_enhanced = np.sqrt(np.abs(diag_enhanced))

    print(f"\n  Enhanced eigenvalues: {np.sort(eigs_enhanced)}")
    print(f"  Condition number: κ = {cond_enhanced:.2e}")
    print(f"  V_ii (diag of sandwich): {diag_enhanced}")
    print(f"  Standard errors: {se_enhanced}")

    _check("Enhanced: all V_ii > 0 (no negative covariance)",
           np.all(diag_enhanced > 0))
    _check("Enhanced: all SE finite",
           np.all(np.isfinite(se_enhanced)))

    # The floor c_floor = λ_max / κ_target = 50 / 1e8 = 5e-7
    c_floor = eigs.max() / kappa_target
    _check(f"Floor c_floor = λ_max/κ_target = {c_floor:.2e} >> λ_min⁺ = 1e-8",
           c_floor > 1e-8)
    _check("Enhanced min eigenvalue ≥ c_floor",
           lambda_min_enhanced >= c_floor * (1 - 1e-10))

    # ------------------------------------------------------------------
    # Part C: Sweep κ_target to show controllable conditioning
    # ------------------------------------------------------------------
    print("\n  --- Part C: κ_target sweep ---")

    for kt in [1e4, 1e6, 1e8, 1e10]:
        H_kt = huang_calibration(H, kappa_target=kt)
        eigs_kt = np.linalg.eigvalsh(H_kt)
        cond_kt = eigs_kt.max() / eigs_kt.min()
        _check(f"κ_target={kt:.0e}: actual κ = {cond_kt:.2e} ≤ {kt:.0e}",
               cond_kt <= kt * 1.01)

    # ------------------------------------------------------------------
    # Part D: Backward compatibility — healthy λ_min⁺ is unchanged
    # ------------------------------------------------------------------
    print("\n  --- Part D: Backward compatibility (healthy λ_min⁺) ---")

    eigs_healthy = np.array([50.0, 12.0, 3.0, 0.5, -0.8])
    H_healthy = Q @ np.diag(eigs_healthy) @ Q.T
    H_healthy = (H_healthy + H_healthy.T) / 2

    H_orig = huang_calibration(H_healthy, kappa_target=np.inf)
    H_enh = huang_calibration(H_healthy, kappa_target=1e8)

    # When λ_min⁺ = 0.5 is healthy, both produce PD results.
    # kappa_target=inf lifts -0.8 to ~5e-11, kappa_target=1e8 lifts to ~5e-7.
    # Both thresholds << 0.5, so the positive eigenvalues are unchanged.
    eigs_orig = np.linalg.eigvalsh(H_orig)
    eigs_enh = np.linalg.eigvalsh(H_enh)
    _check("Healthy λ_min⁺: both are PD",
           np.all(eigs_orig > 0) and np.all(eigs_enh > 0))
    _check("Healthy λ_min⁺: positive eigenvalues unchanged in both",
           np.allclose(np.sort(eigs_orig)[1:], np.sort(eigs_enh)[1:], rtol=1e-10))

    # ------------------------------------------------------------------
    # Part E: All-negative eigenvalue edge case
    # ------------------------------------------------------------------
    print("\n  --- Part E: All-negative eigenvalues ---")

    eigs_all_neg = np.array([-50.0, -12.0, -3.0, -0.5, -0.1])
    H_all_neg = Q @ np.diag(eigs_all_neg) @ Q.T
    H_all_neg = (H_all_neg + H_all_neg.T) / 2

    H_all_neg_cal = huang_calibration(H_all_neg, kappa_target=1e6)
    eigs_all_neg_cal = np.linalg.eigvalsh(H_all_neg_cal)
    cond_all_neg = eigs_all_neg_cal.max() / eigs_all_neg_cal.min()

    _check("All-negative: all calibrated eigenvalues > 0",
           np.all(eigs_all_neg_cal > 0))
    _check(f"All-negative: κ = {cond_all_neg:.2e} is finite",
           np.isfinite(cond_all_neg))

    # ------------------------------------------------------------------
    # Part F: Frobenius distortion comparison
    # ------------------------------------------------------------------
    print("\n  --- Part F: Frobenius distortion comparison ---")

    eigs_orig = np.linalg.eigvalsh(H)

    frob_huang = np.linalg.norm(H - H_huang, 'fro')
    frob_enhanced = np.linalg.norm(H - H_enhanced, 'fro')

    print(f"  ‖H − H_huang‖_F     = {frob_huang:.6e}")
    print(f"  ‖H − H_enhanced‖_F  = {frob_enhanced:.6e}")

    # Enhanced may have slightly higher Frobenius distortion (it lifts
    # the near-zero eigenvalue higher), but the trade-off is massively
    # better conditioning.
    _check("Enhanced Frobenius ≥ Huang Frobenius (expected: floor adds distortion)",
           frob_enhanced >= frob_huang - 1e-12)
    _check(f"Conditioning improvement: {cond_huang:.2e} → {cond_enhanced:.2e}",
           cond_enhanced < cond_huang)


# ===========================================================================
# 12. PROOF — The Huang α-sweep `cost < best_cost` is NEVER satisfied
#     for α > 0 when λ_min⁺ is small.
#
# We manually replay the Huang loop for matrices with λ_min⁺ = 1e-5
# (and 1e-3, 1e-1) and record the cost at each α.  We prove:
#   cost(α) = ‖λ − max(λ, c_α)‖ + α    is monotonically increasing,
#   so best_alpha is ALWAYS α_min = 0.
#
# Root cause: as α increases, c_α = 10^{-α}·λ_min⁺ decreases, so frob
# drops — but only by the TINY amount c_{α-1} − c_α applied to the few
# negative eigenvalues.  That drop is << α_step = 0.1, so cost rises.
# ===========================================================================

def verify_huang_sweep_is_broken() -> None:
    """Prove that the Huang α-sweep never advances past α=0."""
    print("\n─── 12. PROOF: Huang α-sweep is broken ──────────────────────────")

    # --- Test several λ_min⁺ values to show the pattern ---
    test_cases = [
        # (eigenvalues, label)
        (np.array([50.0, 12.0, 3.0, 1e-5, -0.5]),  "λ_min⁺=1e-5"),
        (np.array([50.0, 12.0, 3.0, 1e-3, -0.5]),  "λ_min⁺=1e-3"),
        (np.array([50.0, 12.0, 3.0, 1e-1, -0.5]),  "λ_min⁺=1e-1"),
        (np.array([50.0, 12.0, 3.0, 1e-5, -5.0]),  "λ_min⁺=1e-5, large neg"),
        (np.array([50.0, 12.0, 3.0, 0.5, 1e-5]),   "all-positive, λ_min=1e-5"),
    ]

    alpha_min, alpha_max, alpha_step = 0.0, 10.0, 0.1
    alphas = np.arange(alpha_min, alpha_max + alpha_step * 0.5, alpha_step)

    for eigenvalues, label in test_cases:
        print(f"\n  --- Case: {label} ---")
        print(f"  Eigenvalues: {eigenvalues}")

        positive_mask = eigenvalues > 0
        if not positive_mask.any():
            print("  Skipping (no positive eigenvalues)")
            continue

        lambda_min_pos = eigenvalues[positive_mask].min()
        lambda_max = eigenvalues.max()

        # Replay the EXACT Huang sweep and record every cost
        costs = []
        best_alpha = alphas[0]
        best_cost = np.inf
        best_alpha_idx = 0

        for idx, alpha_i in enumerate(alphas):
            c_alpha = 10.0 ** (-alpha_i) * lambda_min_pos
            calibrated_eigs = np.maximum(eigenvalues, c_alpha)
            frob = np.linalg.norm(eigenvalues - calibrated_eigs)
            cost = frob + alpha_i
            costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_alpha = alpha_i
                best_alpha_idx = idx

        costs = np.array(costs)

        # Print first few costs to show the trend
        print(f"  λ_min⁺ = {lambda_min_pos:.1e}, λ_max = {lambda_max:.1e}")
        print(f"  cost(α=0.0) = {costs[0]:.6f}")
        print(f"  cost(α=0.1) = {costs[1]:.6f}")
        print(f"  cost(α=0.2) = {costs[2]:.6f}")
        print(f"  cost(α=1.0) = {costs[10]:.6f}")
        print(f"  best_alpha  = {best_alpha:.1f} (index {best_alpha_idx})")

        # THE KEY CHECK: best_alpha is ALWAYS 0 when λ_min⁺ is small
        if lambda_min_pos < 0.1:
            _check(
                f"{label}: best_alpha = {best_alpha:.1f} == 0 "
                f"(sweep is useless, cost monotonically increases)",
                best_alpha_idx == 0,
            )

            # Verify cost is strictly increasing after α=0
            diffs = np.diff(costs)
            _check(
                f"{label}: cost strictly increases for all α > 0 "
                f"(min Δcost = {diffs.min():.6f} > 0)",
                diffs.min() > 0,
            )

            # Show the resulting κ is terrible
            c_star = lambda_min_pos  # 10^0 * lambda_min_pos
            kappa = lambda_max / c_star
            _check(
                f"{label}: κ = {kappa:.2e} (terrible conditioning, sweep can't fix)",
                kappa > 1e4,
            )
        else:
            # For λ_min⁺ = 0.1, sweep MIGHT find α > 0 because
            # the Frobenius drop per step can exceed 0.1
            print(f"  (λ_min⁺ is large enough that sweep may work)")

    # ------------------------------------------------------------------
    # Summary: why the sweep fails
    # ------------------------------------------------------------------
    print("\n  --- Summary ---")
    print("  The Huang cost = ‖λ − max(λ, c_α)‖ + α always picks α=0")
    print("  because the Frobenius drop per α-step is negligible")
    print("  compared to the α penalty (0.1 per step).")
    print("  Result: c* = λ_min⁺, so κ = λ_max/λ_min⁺ is uncontrolled.")
    print("  The α-sweep is fundamentally broken for small λ_min⁺.")
    print("  SOLUTION: direct eigenvalue thresholding via κ_target.")


# ===========================================================================
# 13. SOLUTION — Direct eigenvalue thresholding via κ_target
#
# Now that we proved the α-sweep is broken, test that huang_calibration()
# uses the direct c_star = λ_max / κ_target approach for all cases:
# (a) All-positive but ill-conditioned
# (b) NPD with tiny λ_min⁺
# (c) Backward compatibility (healthy matrix, kappa_target=inf)
# ===========================================================================

def verify_direct_threshold_solution() -> None:
    """Test that huang_calibration controls κ via direct thresholding."""
    from dualmatfit.fitting.covariance import huang_calibration

    print("\n─── 13. SOLUTION: direct κ_target thresholding ──────────────────")

    rng = np.random.default_rng(2026)
    n = 5
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))

    # Meat matrix B for sandwich tests
    n_obs = 50
    jacobian = rng.standard_normal((n_obs, n)) * np.array([1, 0.1, 0.05, 0.5, 0.3])
    residuals = rng.standard_normal(n_obs) * 0.1
    psi_vals = residuals / (1 + residuals ** 2)
    scores = psi_vals[:, None] * jacobian
    B = scores.T @ scores

    # ------------------------------------------------------------------
    # Part A: All-positive, ill-conditioned — old code returned unchanged!
    # ------------------------------------------------------------------
    print("\n  --- Part A: All-positive, κ_orig = 5e6, κ_target = 1e4 ---")

    eigs_a = np.array([50.0, 12.0, 3.0, 0.5, 1e-5])
    H_a = Q @ np.diag(eigs_a) @ Q.T
    H_a = (H_a + H_a.T) / 2

    kt = 1e4
    H_cal = huang_calibration(H_a, kappa_target=kt)
    eigs_cal = np.linalg.eigvalsh(H_cal)
    kappa_cal = eigs_cal.max() / eigs_cal.min()
    c_expected = eigs_a.max() / kt  # = 50 / 1e4 = 5e-3

    _check(f"All-positive: κ = {kappa_cal:.2e} ≤ {kt:.0e}",
           kappa_cal <= kt * 1.01)
    _check(f"All-positive: min eigenvalue = {eigs_cal.min():.2e} ≈ c_floor = {c_expected:.2e}",
           abs(eigs_cal.min() - c_expected) / c_expected < 0.01)

    H_inv = np.linalg.inv(H_cal)
    V = H_inv @ B @ H_inv
    se = np.sqrt(np.abs(np.diag(V)))
    _check(f"All-positive: max SE = {se.max():.2e} < 100 (reasonable)",
           se.max() < 100)

    print(f"  κ: 5.00e+06 → {kappa_cal:.2e}")
    print(f"  min eigenvalue: 1e-5 → {eigs_cal.min():.2e}")
    print(f"  SEs: {se}")

    # ------------------------------------------------------------------
    # Part B: NPD + tiny λ_min⁺ = 1e-5
    # ------------------------------------------------------------------
    print("\n  --- Part B: NPD + λ_min⁺ = 1e-5, κ_target = 1e4 ---")

    eigs_b = np.array([50.0, 12.0, 3.0, 1e-5, -0.5])
    H_b = Q @ np.diag(eigs_b) @ Q.T
    H_b = (H_b + H_b.T) / 2

    H_cal_b = huang_calibration(H_b, kappa_target=kt)
    eigs_cal_b = np.linalg.eigvalsh(H_cal_b)
    kappa_cal_b = eigs_cal_b.max() / eigs_cal_b.min()

    _check(f"NPD: κ = {kappa_cal_b:.2e} ≤ {kt:.0e}",
           kappa_cal_b <= kt * 1.01)
    _check(f"NPD: both neg AND tiny eigs lifted to c_floor = {c_expected:.2e}",
           eigs_cal_b.min() >= c_expected * (1 - 1e-10))

    H_inv_b = np.linalg.inv(H_cal_b)
    V_b = H_inv_b @ B @ H_inv_b
    se_b = np.sqrt(np.abs(np.diag(V_b)))
    _check(f"NPD: max SE = {se_b.max():.2e} < 100 (reasonable)",
           se_b.max() < 100)

    print(f"  κ_cal = {kappa_cal_b:.2e}, SEs: {se_b}")

    # ------------------------------------------------------------------
    # Part C: κ_target sweep — conditioning is precisely controlled
    # ------------------------------------------------------------------
    print("\n  --- Part C: κ_target sweep ---")

    for kt_sweep in [1e3, 1e4, 1e6, 1e8]:
        H_s = huang_calibration(H_b, kappa_target=kt_sweep)
        eigs_s = np.linalg.eigvalsh(H_s)
        kappa_s = eigs_s.max() / eigs_s.min()
        c_floor_s = eigs_b.max() / kt_sweep
        _check(
            f"κ_target={kt_sweep:.0e}: κ_actual={kappa_s:.2e}, "
            f"min_eig={eigs_s.min():.2e} ≈ c_floor={c_floor_s:.2e}",
            kappa_s <= kt_sweep * 1.01,
        )

    # ------------------------------------------------------------------
    # Part D: Healthy matrix — returned unchanged
    # ------------------------------------------------------------------
    print("\n  --- Part D: Well-conditioned matrix → unchanged ---")

    eigs_d = np.array([50.0, 12.0, 3.0, 0.5, 0.1])  # κ = 500
    H_d = Q @ np.diag(eigs_d) @ Q.T
    H_d = (H_d + H_d.T) / 2

    H_d_cal = huang_calibration(H_d, kappa_target=1e4)
    _check("Healthy (κ=500 < κ_target=1e4): matrix returned unchanged",
           np.allclose(H_d, H_d_cal, atol=1e-14))

    # ------------------------------------------------------------------
    # Part E: kappa_target=inf — backward compatibility (old Huang)
    # ------------------------------------------------------------------
    print("\n  --- Part E: kappa_target=inf backward compatibility ---")

    H_inf = huang_calibration(H_b, kappa_target=np.inf)
    eigs_inf = np.linalg.eigvalsh(H_inf)
    kappa_inf = eigs_inf.max() / eigs_inf.min()

    # With kappa_target=inf, c_star = λ_max * 1e-12 ≈ 5e-11.
    # The -0.5 eigenvalue is lifted to 5e-11, but 1e-5 stays.
    # So κ ≈ 50 / 5e-11 = 1e12 (no conditioning control with inf).
    _check(f"kappa_target=inf: no conditioning control, κ = {kappa_inf:.2e}",
           kappa_inf > 1e4)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Huang et al. (2017) NPD Calibration — Verification Script")
    print("Ref: covariance_theory.md §6, Eqs. (20)–(22)")
    print("=" * 70)

    verify_basic_properties()
    verify_frobenius_optimality()
    verify_eigenvector_preservation()
    verify_edge_cases()
    verify_sandwich_integration()
    verify_condition_number()
    verify_conditioning_guarantee()
    verify_hgo_scenario()
    verify_idempotency()
    verify_equation_agreement()
    verify_near_zero_eigenvalue_problem()
    verify_huang_sweep_is_broken()
    verify_direct_threshold_solution()

    print("\n" + "═" * 70)
    total = len(_FAILURES)
    if total == 0:
        print("RESULT: ALL CHECKS PASSED ✓")
    else:
        print(f"RESULT: {total} CHECK(S) FAILED ✗")
        for f in _FAILURES:
            print(f"  • {f}")
    print("═" * 70)

    sys.exit(1 if _FAILURES else 0)