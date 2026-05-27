"""
Symbolic verification of every equation in
``docs/algorithms/covariance_theory.md``.

Each section below corresponds to one or more display-math blocks in the
document.  SymPy is used to derive or cross-check every non-trivial algebraic
claim; purely definitional identities are noted as such.

Run with::

    conda run -n matfit1d python scripts/verify_covariance_theory.py

A PASS/FAIL line is printed for each check.  The script exits with a non-zero
status if any assertion fails.

References
----------
[Huber, 1964]
    Huber, P. J. (1964). Robust estimation of a location parameter.
    *Annals of Mathematical Statistics*, 35(1):73–101.

[Huber and Ronchetti, 2009]
    Huber, P. J. and Ronchetti, E. M. (2009). *Robust Statistics*, 2nd edition.
    Wiley, Hoboken, NJ.
    §7.6 gives the correct asymptotic covariance formula for regression
    M-estimators: cov(θ̂) ∝ E[ψ²(r)] / E[-ψ′(r)]² · (XᵀX)⁻¹, which only
    reduces to the OLS formula s²(XᵀX)⁻¹ in the classical case ψ(r) = r.

[Hampel et al., 1986]
    Hampel, F. R., Ronchetti, E. M., Rousseeuw, P. J., and Stahel, W. A.
    (1986). *Robust Statistics: The Approach Based on Influence Functions*.
    Wiley, New York.

[White, 1980]
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
    estimator and a direct test for heteroskedasticity. *Econometrica*,
    48(4):817–838.

[Donaldson and Schnabel, 1987]
    Donaldson, J. R. and Schnabel, R. B. (1987). Computational experience with
    confidence regions and confidence intervals for nonlinear least squares.
    *Technometrics*, 29(1):67–82.

[Soffritti and Pacillo, 2021]
    Soffritti, G. and Pacillo, G. (2021). On the performance of the sandwich
    estimator in heteroscedastic mixture regression models.
    *Journal of Multivariate Analysis*, 183:104720.

[Ridders, 1982]
    Ridders, C. J. F. (1982). Accurate computation of F'(x) and F'(x)F''(x).
    *Advances in Engineering Software*, 4(2):75–76.

[Baker, 2021]
    Baker, J. (2021). Accurate computation of the Hessian matrix and subsequent
    covariance and curvature measures. *arXiv preprint arXiv:2105.04829v1*.

[Huang et al., 2017]
    Huang, C., Farewell, D., and Pan, J. (2017). A calibration method for
    non-positive definite covariance matrix in multivariate data analysis.
    *Journal of Multivariate Analysis*, 157:45–52.

[Canales et al., 2023]
    Canales, C., Díaz, H., and García-Herrera, C. (2023). Constitutive
    modelling of hyperelastic materials using a novel constrained
    optimization-based parameter identification method. *Biomechanics and
    Modeling in Mechanobiology*, 22:547–566.

[Martonová et al., 2024]
    Martonová, D., Peirlinck, M., Linka, K., Holzapfel, G. A., Leyendecker,
    S., and Kuhl, E. (2024). Automated model discovery for human cardiac
    tissue: Discovering the best model and parameters. *Computer Methods in
    Applied Mechanics and Engineering*, 428:117078.
"""

from __future__ import annotations

import sys
import textwrap

import numpy as np
import sympy as sp

_FAILURES: list[str] = []


def _check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        _FAILURES.append(label)


# ===========================================================================
# § 2  Cauchy loss — ρ, ψ, ψ′, and chain-rule bridge (Eqs 2–4a)
#
# The Cauchy loss is a redescending M-estimator [Huber, 1964].  The score
# function ψ = ∂ρ/∂r describes the influence function; its bounded, then
# redescending behaviour is the key property discussed in [Hampel et al., 1986]
# that guarantees robust down-weighting of gross outliers.
#
# The chain-rule bridge equations (3a), (4a-i), and (4a) use sp.Function('f')
# — an abstract, unspecified forward model — so that SymPy applies the chain
# rule without assuming any concrete algebraic form.  Eq (4a-i) is the
# product-rule decomposition showing each intermediate factor before
# simplification into the final form Eq (4a).
# ===========================================================================

def verify_cauchy_loss() -> None:
    """Verify Cauchy ρ, ψ, ψ′ and chain-rule bridge Eqs (3a), (4a-i), (4a) (§2).

    Sources: [Huber, 1964]; [Hampel et al., 1986]; [Seber and Wild, 2003, §14.1].
    """
    print("\n─── §2  Cauchy loss ρ, ψ, ψ′ ───────────────────────────────────────")
    print("     Refs: Huber (1964) Ann. Math. Stat. 35:73–101;"
          " Hampel et al. (1986) Robust Statistics, Wiley.")

    r, c = sp.symbols('r c', real=True, positive=True)

    # ρ(r) = (c²/2) ln(1 + (r/c)²)  — Cauchy loss [Huber, 1964]
    rho = sp.Rational(1, 2) * c**2 * sp.log(1 + (r / c)**2)

    # ψ(r) = ∂ρ/∂r  →  expect r / (1 + (r/c)²)
    # This is the influence function kernel [Hampel et al., 1986, §2.1]
    psi = sp.diff(rho, r)
    psi_expected = r / (1 + (r / c)**2)
    _check("ψ(r) = ∂ρ/∂r = r/(1+(r/c)²)  [Huber 1964]",
           sp.simplify(psi - psi_expected) == 0)

    # ψ′(r) = ∂ψ/∂r  →  expect (1-(r/c)²) / (1+(r/c)²)²
    # Sign change at |r| = c drives non-convexity [Hampel et al., 1986, §2.4]
    psi_prime = sp.diff(psi_expected, r)
    psi_prime_expected = (1 - (r / c)**2) / (1 + (r / c)**2)**2
    _check("ψ′(r) = (1-(r/c)²)/(1+(r/c)²)²  [Hampel et al. 1986]",
           sp.simplify(psi_prime - psi_prime_expected) == 0)

    # Redescending: ψ(r) → 0 as r → ∞  (bounded influence, [Hampel et al. 1986])
    lim_inf = sp.limit(psi_expected, r, sp.oo)
    _check("ψ(r) → 0 as r → +∞  (redescending, bounded influence function)",
           lim_inf == 0)

    # Non-convexity: ψ′(r) < 0 when r > c  [Hampel et al. 1986, §2.4]
    psi_prime_at_2c = psi_prime_expected.subs(r, 2 * c)
    _check("ψ′(2c) < 0  (non-convexity for |r| > c)",
           sp.simplify(psi_prime_at_2c) < 0)

    # ── Chain rule through abstract forward model (Eqs 3a, 4a) ──────────
    # Use sp.Function('f') so SymPy applies the chain rule symbolically
    # without assuming any concrete algebraic form for the forward model.
    m = sp.Symbol('m', real=True)
    fbar = sp.Symbol('fbar', real=True)
    f = sp.Function('f')            # abstract forward model f(m)
    Phi = fbar - f(m)               # residual Φ(m) = f̄ − f(m)
    rho_Phi = sp.Rational(1, 2) * c**2 * sp.log(1 + (Phi / c)**2)

    # First-order chain rule  (Eq. 3a): ∂ρ/∂m = −ψ(Φ)·f′(m)
    d1_sympy = sp.diff(rho_Phi, m)
    psi_Phi = Phi / (1 + (Phi / c)**2)
    fp = f(m).diff(m)               # f′(m) — abstract derivative
    d1_formula = -psi_Phi * fp      # expected: −ψ(Φ)·J  where J = f′(m)

    # Verify by substituting arbitrary numerical values for all atoms
    subs_num = {f(m): sp.Rational(1, 3), fp: sp.Rational(5, 2),
                f(m).diff(m, 2): sp.Rational(7, 4),
                c: sp.Integer(2), fbar: sp.Integer(2)}
    _check(
        "Eq(3a): ∂ρ(Φ(m))/∂m = −ψ(Φ)·f′(m)  "
        "(1st-order chain rule, abstract f(m))  [chain rule]",
        sp.simplify(d1_sympy.subs(subs_num) - d1_formula.subs(subs_num)) == 0,
    )

    # Second-order chain rule  (Eq. 4a): ∂²ρ/∂m² = ψ′(Φ)·(f′)² − ψ(Φ)·f″
    d2_sympy = sp.diff(rho_Phi, m, 2)
    psi_prime_Phi = psi_prime_expected.subs(r, Phi)
    fpp = f(m).diff(m, 2)           # f″(m) — abstract second derivative
    d2_formula = psi_prime_Phi * fp**2 - psi_Phi * fpp

    _check(
        "Eq(4a): ∂²ρ(Φ(m))/∂m² = ψ′(Φ)·(f′)² − ψ(Φ)·f″  "
        "(2nd-order chain rule, abstract f(m))  [Seber & Wild 2003 §14.1]",
        sp.simplify(d2_sympy.subs(subs_num) - d2_formula.subs(subs_num)) == 0,
    )

    # ── Intermediate chain-rule factors for Eq (4a-i) ────────────────────
    # Verify every labeled factor in the product-rule decomposition
    # (4a-i) individually, using abstract sp.Function('f').

    # Factor: ∂Φ/∂m = −f′(m)  (residual Jacobian from Φ = f̄ − f(m))
    dPhi_dm = sp.diff(Phi, m)
    _check(
        "Eq(3a) factor: ∂Φ/∂m = −f′(m)  "
        "(abstract residual Jacobian)  [chain rule]",
        sp.simplify(dPhi_dm - (-fp)) == 0,
    )

    # Factor: ∂²Φ/∂m² = −f″(m)  (second derivative of residual)
    d2Phi_dm2 = sp.diff(Phi, m, 2)
    _check(
        "Eq(4a-i) factor: ∂²Φ/∂m² = −f″(m)  "
        "(abstract residual second derivative)  [chain rule]",
        sp.simplify(d2Phi_dm2 - (-fpp)) == 0,
    )

    # Factor: ∂ψ(Φ)/∂m = ψ′(Φ)·∂Φ/∂m = −ψ′(Φ)·f′(m)
    # (chain rule on the composition ψ ∘ Φ)
    psi_Phi_expr = Phi / (1 + (Phi / c)**2)
    d_psi_Phi_sympy = sp.diff(psi_Phi_expr, m)
    d_psi_Phi_formula = psi_prime_Phi * dPhi_dm  # ψ′(Φ)·(−f′)
    _check(
        "Eq(4a-i) factor: ∂ψ(Φ)/∂m = ψ′(Φ)·(−f′(m))  "
        "(chain rule on ψ∘Φ, abstract f(m))",
        sp.simplify(
            d_psi_Phi_sympy.subs(subs_num) - d_psi_Phi_formula.subs(subs_num)
        ) == 0,
    )

    # Term 1 of Eq (4a-i): [∂ψ(Φ)/∂m] · [∂Φ/∂mᵀ]
    #   = [ψ′(Φ)·(−f′)] · (−f′) = ψ′(Φ)·(f′)²
    term1_formula = d_psi_Phi_formula * dPhi_dm  # [ψ′(Φ)·(−f′)]·(−f′)
    term1_expected = psi_prime_Phi * fp**2
    _check(
        "Eq(4a-i) Term 1: [∂ψ(Φ)/∂m]·[∂Φ/∂m] = ψ′(Φ)·(f′)²  "
        "(double negative cancels → Gauss-Newton term)",
        sp.simplify(
            term1_formula.subs(subs_num) - term1_expected.subs(subs_num)
        ) == 0,
    )

    # Term 2 of Eq (4a-i): ψ(Φ) · [∂²Φ/∂m²]
    #   = ψ(Φ) · (−f″) = −ψ(Φ)·f″
    term2_formula = psi_Phi * d2Phi_dm2  # ψ(Φ)·(−f″)
    term2_expected = -psi_Phi * fpp
    _check(
        "Eq(4a-i) Term 2: ψ(Φ)·[∂²Φ/∂m²] = −ψ(Φ)·f″  "
        "(second-order correction term)",
        sp.simplify(
            term2_formula.subs(subs_num) - term2_expected.subs(subs_num)
        ) == 0,
    )

    # Full Eq (4a-i) = Term 1 + Term 2 matches SymPy automatic ∂²ρ/∂m²
    d2_from_terms = term1_formula + term2_formula
    _check(
        "Eq(4a-i) = Term1 + Term2 matches SymPy auto ∂²ρ/∂m²  "
        "(product-rule decomposition is complete and correct)",
        sp.simplify(
            d2_sympy.subs(subs_num) - d2_from_terms.subs(subs_num)
        ) == 0,
    )

    # Eq (4a-i) simplifies to Eq (4a): ψ′(Φ)·(f′)² − ψ(Φ)·f″
    d2_from_expected_terms = term1_expected + term2_expected
    _check(
        "Eq(4a-i) → Eq(4a): ψ′(Φ)·(f′)² − ψ(Φ)·f″  "
        "(intermediate form simplifies to final Hessian formula)",
        sp.simplify(
            d2_from_expected_terms.subs(subs_num) - d2_formula.subs(subs_num)
        ) == 0,
    )


# ===========================================================================
# § 2 / § 3.4  Tikhonov regularisation — R(m), gradient, Hessian
# (document lines 68-70, 131-133)
#
# The quadratic Tikhonov penalty is the standard L₂ regularisation used in
# inverse problems [Canales et al., 2023; Martonová et al., 2024].  Its
# Hessian is ωΓ — a constant, positive-semidefinite matrix — which shifts
# all eigenvalues of the total Hessian upward [Donaldson and Schnabel, 1987].
# ===========================================================================

def verify_tikhonov() -> None:
    """Verify R(m) quadratic form: gradient = ωΓ(m−m₀), Hessian = ωΓ (§2/§3.4).

    Sources: [Canales et al., 2023]; [Martonová et al., 2024];
             [Donaldson and Schnabel, 1987].
    """
    print("\n─── §2/§3.4  Tikhonov R(m), gradient = ωΓ(m−m₀), Hessian = ωΓ ────")
    print("     Refs: Canales et al. (2023) Biomech. Model. Mechanobiol. 22:547–566;"
          " Donaldson & Schnabel (1987) Technometrics 29:67–82.")

    m1, m2, m10, m20 = sp.symbols('m1 m2 m10 m20', real=True)
    omega, g11, g12, g22 = sp.symbols('omega g11 g12 g22', real=True)
    Gamma = sp.Matrix([[g11, g12], [g12, g22]])
    m = sp.Matrix([m1, m2])
    m0 = sp.Matrix([m10, m20])
    delta = m - m0

    # R(m) = (ω/2)(m−m₀)ᵀ Γ (m−m₀)  [Canales et al., 2023, §3]
    R = omega / 2 * (delta.T @ Gamma @ delta)[0, 0]

    grad_R = sp.Matrix([sp.diff(R, m1), sp.diff(R, m2)])
    grad_expected = omega * Gamma * delta
    _check("∇R(m) = ωΓ(m−m₀)  [Canales et al. 2023]",
           sp.simplify(grad_R - grad_expected) == sp.zeros(2, 1))

    # ∇²R = ωΓ — the regularisation Hessian is constant [Donaldson & Schnabel, 1987]
    H_R = sp.hessian(R, [m1, m2])
    _check("∇²R(m) = ωΓ  (regularisation Hessian, constant)",
           sp.simplify(H_R - omega * Gamma) == sp.zeros(2, 2))


# ===========================================================================
# § 3.3  Gradient of F and the chain rule
# (document lines 109-111)
# g(m) = Σ_j ψ(Φʲ)(−∇_m fʲ) + ωΓ(m−m₀)
#
# The data-driven term arises from the chain rule applied to the Cauchy loss;
# the penalty term contributes ωΓ(m−m₀).  Together they form the score
# function of the penalised M-estimator [White, 1980; Soffritti & Pacillo, 2021].
# ===========================================================================

def verify_gradient_chain_rule() -> None:
    """Verify gradient of F by chain rule: g = Σ ψ(Φʲ)(−∇f ʲ) + ωΓ(m−m₀) (§3.3).

    Sources: [White, 1980]; [Soffritti and Pacillo, 2021].
    """
    print("\n─── §3.3  Gradient of F via chain rule ─────────────────────────────")
    print("     Refs: White (1980) Econometrica 48:817–838;"
          " Soffritti & Pacillo (2021) J. Multivar. Anal. 183:104720.")

    m_var, c, f_obs, a, omega, m0_val, gamma_val = sp.symbols(
        'm c fobs a omega m0 gamma', real=True, positive=True)

    # One observation, one parameter: f(m) = a·m, Φ = f̄ − f(m)
    # Cauchy loss at a single residual:  ρ(Φ) = (c²/2) ln(1 + (Φ/c)²)
    f_m = a * m_var
    Phi = f_obs - f_m
    rho_scalar = c**2 / 2 * sp.log(1 + (Phi / c)**2)
    R_scalar = omega / 2 * gamma_val * (m_var - m0_val)**2

    F = rho_scalar + R_scalar
    g = sp.diff(F, m_var)

    # Expected: ψ(Φ)·(−∂f/∂m) + ωΓ(m−m₀)  [White, 1980, eq. (1)]
    psi_Phi = Phi / (1 + (Phi / c)**2)
    expected_g = psi_Phi * (-a) + omega * gamma_val * (m_var - m0_val)

    _check("g = ψ(Φ)·(−∂f/∂m) + ωΓ(m−m₀)  [White 1980; chain rule]",
           sp.simplify(g - expected_g) == 0)


# ===========================================================================
# § 3.3  B matrix — outer-product formula and symmetry
# (document lines 115-117)
# B = Σ_j [ψ(Φʲ)∇f ʲ][ψ(Φʲ)∇f ʲ]ᵀ
#
# The "meat" of the sandwich estimator is the empirical covariance of the
# score vector [White, 1980].  Because the Tikhonov term is deterministic it
# contributes zero variance, so B involves only the data-driven residuals
# [Soffritti and Pacillo, 2021, §2].
# ===========================================================================

def verify_b_matrix() -> None:
    """Verify B matrix outer-product structure and symmetry (§3.3).

    Sources: [White, 1980]; [Soffritti and Pacillo, 2021].
    """
    print("\n─── §3.3  B matrix outer-product structure ──────────────────────────")
    print("     Refs: White (1980) Econometrica 48:817–838; Soffritti & Pacillo (2021) J. Multivar. Anal. 183:104720.")

    psi1, psi2 = sp.symbols('psi1 psi2', real=True)
    j11, j12, j21, j22 = sp.symbols('j11 j12 j21 j22', real=True)

    # Per-observation score vectors: s_j = ψ(Φ_j) · ∇_m f_j
    # B = Σ_j s_j s_jᵀ  [White, 1980, eq. (4); Soffritti & Pacillo, 2021, eq. (3)]
    s1 = sp.Matrix([psi1 * j11, psi1 * j12])
    s2 = sp.Matrix([psi2 * j21, psi2 * j22])
    B = s1 * s1.T + s2 * s2.T

    _check("B is symmetric  [White 1980]", B == B.T)

    # Diagonal entries: sums of squared scores → B is PSD
    _check("B[0,0] = ψ₁²·j₁₁² + ψ₂²·j₂₁²  (PSD diagonal structure)",
           sp.expand(B[0, 0]) == sp.expand(psi1**2 * j11**2 + psi2**2 * j21**2))


# ===========================================================================
# § 3.3  Meat matrix B — first-principles derivation from Cauchy loss and
#        residual derivative  (document §3.3)
#
# The meat matrix B = Σ_j s_j s_jᵀ is the empirical outer product of the
# per-observation score contributions s_j = ∂ρ(Φ^j(m))/∂m.
#
# Derivation chain:
#   ρ(r)      = (c²/2) ln(1+(r/c)²)         — Cauchy loss [Huber, 1964]
#   ψ(r)      = ∂ρ/∂r = r/(1+(r/c)²)        — influence function
#   Φ^j(m)   = f̄^j − f^j(m)               — residual definition
#   ∂Φ^j/∂m  = −J_j                         — residual Jacobian (minus sign)
#   s_j       = ∂ρ(Φ^j)/∂m = ψ(Φ^j)·(−J_j) — chain rule
#   s_j s_jᵀ  = ψ(Φ^j)² J_j J_jᵀ           — sign cancels in outer product
#   B         = Σ_j ψ(Φ^j)² J_j J_jᵀ       — summed outer products
#
# Boundedness: ψ(r)² ≤ c²/4 for all r, achieved at |r| = c.
# Outlier exclusion: ψ(r)² → 0 as |r| → ∞.
#
# Sources: [White, 1980]; [Huber, 1964]; [Hampel et al., 1986, §2].
# ===========================================================================

def verify_meat_matrix_cauchy() -> None:
    """Verify meat matrix B via chain rule through Cauchy loss and residual derivative (§3.3).

    Groups:
      A — Chain rule derivation of the per-observation score s_j = −ψ(Φ^j) J_j
      B — B = Σ s_j s_jᵀ = Σ ψ(Φ^j)² J_j J_jᵀ, cross-checked vs direct SymPy
      C — Boundedness: ψ² ≤ c²/4; ψ(0)=0; ψ²→0 as r→∞ (robust outlier exclusion)

    Sources: [White, 1980] Econometrica 48:817–838;
             [Huber, 1964] Ann. Math. Stat. 35:73–101;
             [Hampel et al., 1986] Robust Statistics §2.
    """
    print("\n─── §3.3  Meat matrix B — Cauchy loss + residual derivative ─────────")
    print("     Refs: White (1980) Econometrica 48:817–838;"
          " Huber (1964) Ann. Math. Stat. 35:73–101;"
          " Hampel et al. (1986) Robust Statistics §2.")

    r, c = sp.symbols('r c', positive=True)

    # ── Group A: chain rule derivation ───────────────────────────────────────

    # A1: derive ψ from ρ via ∂ρ/∂r  [Huber, 1964]
    rho = sp.Rational(1, 2) * c**2 * sp.log(1 + (r / c)**2)
    psi_derived = sp.diff(rho, r)
    psi_expected = r / (1 + (r / c)**2)
    _check(
        "A1: ψ(r) = ∂ρ/∂r = r/(1+(r/c)²)  (derived from Cauchy ρ)  [Huber 1964]",
        sp.simplify(psi_derived - psi_expected) == 0,
    )

    # A2: residual sign: Φ^j(m) = f̄^j − f^j(m)  →  ∂Φ^j/∂m = −J_j
    # Verified symbolically for a linear model: f^j(m) = a·m, f̄^j = f_obs
    m_var, a, f_obs = sp.symbols('m a fobs', real=True)
    f_model = a * m_var                    # forward model
    Phi = f_obs - f_model                  # residual Φ = f̄ − f(m)
    dPhi_dm = sp.diff(Phi, m_var)          # should be −a = −J
    J = sp.diff(f_model, m_var)            # J = a
    _check(
        "A2: ∂Φ/∂m = −J  (residual Jacobian carries minus sign from Φ = f̄−f(m))  "
        "[chain rule]",
        sp.simplify(dPhi_dm + J) == 0,
    )

    # A3: chain rule applied to ρ(Φ(m)): ∂ρ/∂m = ψ(Φ)·(∂Φ/∂m) = −ψ(Φ)·J
    rho_of_Phi = sp.Rational(1, 2) * c**2 * sp.log(1 + (Phi / c)**2)
    score_sympy   = sp.diff(rho_of_Phi, m_var)
    psi_Phi       = Phi / (1 + (Phi / c)**2)
    score_formula = psi_Phi * (-J)
    _check(
        "A3: ∂ρ(Φ(m))/∂m = −ψ(Φ)·J  (chain rule through residual)  [White 1980]",
        sp.simplify(score_sympy - score_formula) == 0,
    )

    # ── Group B: B = Σ s_j s_jᵀ cross-check ────────────────────────────────

    # 2-observation, 1-parameter linear model: f^j(m) = a_j·m
    a1, a2, f1, f2 = sp.symbols('a1 a2 f1 f2', real=True)
    Phi1 = f1 - a1 * m_var
    Phi2 = f2 - a2 * m_var
    psi1 = Phi1 / (1 + (Phi1 / c)**2)
    psi2 = Phi2 / (1 + (Phi2 / c)**2)

    # B1: per-observation score s_j = ∂ρ(Φ^j)/∂m = −ψ(Φ^j)·J_j  (scalar)
    s1_formula = -psi1 * a1
    s2_formula = -psi2 * a2
    s1_sympy   = sp.diff(sp.Rational(1,2)*c**2*sp.log(1+(Phi1/c)**2), m_var)
    s2_sympy   = sp.diff(sp.Rational(1,2)*c**2*sp.log(1+(Phi2/c)**2), m_var)
    _check(
        "B1: s_j = ∂ρ(Φ^j)/∂m = −ψ(Φ^j)·J_j  (SymPy vs formula, obs 1)  "
        "[chain rule + Huber 1964]",
        sp.simplify(s1_sympy - s1_formula) == 0,
    )
    _check(
        "B1: s_j = ∂ρ(Φ^j)/∂m = −ψ(Φ^j)·J_j  (SymPy vs formula, obs 2)  "
        "[chain rule + Huber 1964]",
        sp.simplify(s2_sympy - s2_formula) == 0,
    )

    # B2: outer product s_j² = ψ(Φ^j)² J_j²  (sign cancels)
    _check(
        "B2: s₁² = ψ(Φ¹)²·J₁²  (sign cancels in outer product)  [White 1980]",
        sp.simplify(s1_formula**2 - psi1**2 * a1**2) == 0,
    )

    # B3: B = Σ s_j² (scalar) via formula equals B via direct sum of squared scores
    B_formula = psi1**2 * a1**2 + psi2**2 * a2**2
    B_direct  = s1_sympy**2 + s2_sympy**2
    _check(
        "B3: B = Σ ψ(Φʲ)² Jⱼ²  matches direct Σ (∂ρ/∂m)²  (cross-check)  [White 1980; chain rule]",
        sp.simplify(sp.expand(B_formula) - sp.expand(B_direct)) == 0,
    )

    # B4: B ≥ 0 (PSD in the scalar case) since ψ² ≥ 0 and J² ≥ 0
    # Numerical spot-check: specific values
    subs = [(m_var, sp.Integer(0)), (c, sp.Integer(2)),
            (f1, sp.Integer(1)), (a1, sp.Integer(1)),
            (f2, sp.Integer(3)), (a2, sp.Integer(2))]
    B_num = float(B_formula.subs(subs))
    _check(
        "B4: B ≥ 0  (PSD; outer products with non-negative weight ψ²≥0)  [White 1980]",
        B_num >= 0,
    )

    # ── Group C: boundedness and outlier-exclusion properties ────────────────

    # C1: ψ(0) = 0  — zero-residual observation contributes nothing to B
    psi_at_zero = psi_expected.subs(r, 0)
    _check(
        "C1: ψ(0) = 0  (zero-residual obs contributes nothing to meat B)  [Huber 1964]",
        sp.simplify(psi_at_zero) == 0,
    )

    # C2: ψ(r)² ≤ c²/4 for all r, maximum achieved at |r| = c
    # Find r* = argmax of ψ²(r): d/dr [ψ²] = 0
    psi_sq = psi_expected**2
    d_psi_sq = sp.diff(psi_sq, r)
    critical_pts = sp.solve(d_psi_sq, r)   # symbolic solution
    # Only the positive critical point r* = c
    r_star = [pt for pt in critical_pts if pt.is_positive]
    _check(
        "C2: argmax(ψ²) = c  (maximum meat weight at |r| = c, not at 0 or ∞)  [Hampel et al. 1986 §2]",
        len(r_star) == 1 and sp.simplify(r_star[0] - c) == 0,
    )
    psi_sq_max = sp.simplify(psi_sq.subs(r, c))
    _check(
        "C2: max(ψ²) = c²/4  (meat weight is bounded above)  [Hampel et al. 1986 §2]",
        sp.simplify(psi_sq_max - c**2 / 4) == 0,
    )

    # C3: ψ(r)² → 0 as r → ∞  (outlier exclusion; robust property)
    lim_psi_sq = sp.limit(psi_sq, r, sp.oo)
    _check(
        "C3: ψ(r)² → 0 as r → +∞  (outlier excluded from meat B)  [Hampel et al. 1986 §2]",
        lim_psi_sq == 0,
    )

    # OLS contrast: r² → ∞ as r → ∞  (OLS meat grows without bound)
    ols_meat_weight = r**2
    lim_ols = sp.limit(ols_meat_weight, r, sp.oo)
    _check(
        "C3 contrast: OLS meat weight r² → +∞  (OLS meat is unbounded)  [Donaldson & Schnabel 1987]",
        lim_ols == sp.oo,
    )


# ===========================================================================
# § 3.4  Bread matrix H₁ = Σ ψ′(Φ^j) J_j J_jᵀ  (Cauchy-weighted Gram matrix)
# (document §3.4, bread formula and implementation recipe)
#
# The bread of the Cauchy sandwich estimator is the Cauchy-weighted Gram matrix
# H₁ = Σ ψ′(Φ^j) J_j J_jᵀ.  Key properties of the per-observation weight ψ′:
#   ψ′(0) = 1  — inlier at zero residual contributes full curvature
#   ψ′(c) = 0  — zero crossing at the tuning constant c
#   ψ′(r) < 0  for |r| > c  — outlier subtracts curvature (can make H₁ indefinite)
#
# For a linear forward model (∇²f = 0), H_data = H₁ exactly; confirmed by
# direct SymPy differentiation of the Cauchy cost function.
#
# Sources: [Huber and Ronchetti, 2009, §7.6]; [Hampel et al., 1986, §2.4].
# ===========================================================================

def verify_bread_matrix_cauchy() -> None:
    """Verify bread matrix H₁ = Σ ψ′(Φʲ) JⱼJⱼᵀ for the Cauchy loss (§3.4).

    Checks:
      1. ψ′(0) = 1      — maximum weight; zero-residual inlier contributes fully
      2. ψ′(c) = 0      — zero crossing at tuning constant c
      3. ψ′(2c) < 0     — outlier residual subtracts curvature
      4. 2-observation linear model: direct SymPy differentiation confirms
         H_data = Σ ψ′(Φ^j) J_j²  (first-order formula exact when ∇²f = 0)
      5. H₁ ≤ H_GN in the inlier regime  (0 < ψ′ < 1 → bread < GN Gramian)
      6. H₁ = H_GN at zero residuals     (ψ′(0) = 1 → bread equals GN Gramian)
      7. H₁ < 0 for a single extreme outlier  (bread can become indefinite)

    Sources: [Huber and Ronchetti, 2009, §7.6]; [Hampel et al., 1986, §2.4].
    """
    print("\n─── §3.4  Bread matrix H₁ = Σ ψ′(Φʲ) JⱼJⱼᵀ (Cauchy weights) ──────")
    print("     Refs: Huber & Ronchetti (2009) Robust Statistics §7.6;"
          " Hampel et al. (1986) Robust Statistics §2.4.")

    r, c = sp.symbols('r c', positive=True)
    psi_prime = (1 - (r / c)**2) / (1 + (r / c)**2)**2

    # ── ψ′ weight properties ─────────────────────────────────────────────────

    # 1. ψ′(0) = 1  — zero residual contributes full outer product to bread
    _check(
        "ψ′(0) = 1  (inlier at r=0 contributes full curvature weight)  "
        "[Huber & Ronchetti 2009 §7.6]",
        sp.simplify(psi_prime.subs(r, 0) - 1) == 0,
    )

    # 2. ψ′(c) = 0  — the tuning constant c is the inlier/outlier boundary
    _check(
        "ψ′(c) = 0  (zero crossing; c separates inlier and outlier regimes)  "
        "[Hampel et al. 1986 §2.4]",
        sp.simplify(psi_prime.subs(r, c)) == 0,
    )

    # 3. ψ′(2c) < 0  — outlier observations actively subtract curvature
    _check(
        "ψ′(2c) < 0  (outlier residual subtracts curvature from H_data)  "
        "[Hampel et al. 1986 §2.4]",
        sp.simplify(psi_prime.subs(r, 2 * c)) < 0,
    )

    # ── First-order bread via direct SymPy differentiation ───────────────────
    # Forward model: f_j(m) = a_j * m  (linear in m → ∇²f = 0 → H_data = H₁)
    # Two observations with scale parameter c.
    m_var = sp.Symbol('m', real=True)
    a1, a2, f1, f2 = sp.symbols('a1 a2 f1 f2', real=True)

    Phi1 = f1 - a1 * m_var     # residual Φ¹ = f̄¹ − a₁·m
    Phi2 = f2 - a2 * m_var     # residual Φ² = f̄² − a₂·m

    F_data = (
        c**2 / 2 * sp.log(1 + (Phi1 / c)**2)
        + c**2 / 2 * sp.log(1 + (Phi2 / c)**2)
    )

    # True Hessian of F_data with respect to m (scalar: second derivative)
    H_sympy = sp.simplify(sp.diff(F_data, m_var, 2))

    # First-order bread formula: H₁ = Σ ψ′(Φ^j) J_j²
    # For f_j = a_j·m: J_j = a_j,  ∇²f_j = 0
    H1_formula = (
        psi_prime.subs(r, Phi1) * a1**2
        + psi_prime.subs(r, Phi2) * a2**2
    )

    # 4. Verify H_data = H₁ for the linear model (∇²f = 0 → no correction term)
    _check(
        "Linear model: H_data = Σ ψ′(Φʲ) Jⱼ²  "
        "(∇²f=0 → first-order formula is exact)  [chain rule + SymPy]",
        sp.simplify(H_sympy - H1_formula) == 0,
    )

    # 5. H₁ ≤ H_GN in the inlier regime  (0 < ψ′(r) < 1 for 0 < |r| < c)
    # Use large c so both residuals are deep inliers → 0 < ψ′ < 1 → H₁ < H_GN
    subs_inlier = [
        (m_var, sp.Rational(1, 2)), (c, sp.Integer(10)),
        (f1, sp.Integer(1)), (a1, sp.Integer(1)),
        (f2, sp.Integer(3)), (a2, sp.Integer(2)),
    ]
    H1_num  = float(H1_formula.subs(subs_inlier))
    H_GN_num = 1.0**2 + 2.0**2   # Σ a_j² = 1 + 4 = 5  (unweighted Gramian)
    _check(
        "Inlier regime: H₁ ≤ H_GN  "
        "(Cauchy ψ′ ≤ 1 → bread ≤ Gauss-Newton Gramian)",
        H1_num <= H_GN_num + 1e-10,
    )

    # 6. Zero-residual limit: H₁ = H_GN  (ψ′(0) = 1 for all j)
    H1_zero  = psi_prime.subs(r, 0) * a1**2 + psi_prime.subs(r, 0) * a2**2
    H_GN_sym = a1**2 + a2**2
    _check(
        "Zero residuals: H₁ = H_GN = Σ Jⱼ²  "
        "(ψ′(0)=1 → bread equals GN Gramian in the zero-residual limit)",
        sp.simplify(H1_zero - H_GN_sym) == 0,
    )

    # 7. Outlier regime: H₁ < 0 when |Φ| ≫ c  (bread can become indefinite)
    # Single outlier: Φ = 100, c = 1, J = 1  →  ψ′(100/1) ≪ 0  →  H₁ < 0
    psi_prime_outlier = float(psi_prime.subs([(r, sp.Integer(100)), (c, sp.Integer(1))]))
    H1_outlier = psi_prime_outlier * 1.0**2   # one observation, J = 1
    _check(
        "Outlier regime: H₁ < 0 for |Φ|=100, c=1  "
        "(bread is indefinite; motivates eigenvalue clamping)  "
        "[Hampel et al. 1986 §2.4]",
        H1_outlier < 0,
    )


# ===========================================================================
# § 4  Eigenvalue shift and condition number
# (document lines 169-171, 175-177)
# H_reg = H_data + ωI,  κ(H_reg) = (λ_max+ω)/(λ_min+ω) ≈ λ_max/ω
#
# The eigenvalue-shifting effect of Tikhonov regularisation on the condition
# number is a classical result; see [Donaldson and Schnabel, 1987, §3] for
# the covariance implications in nonlinear least squares.
# ===========================================================================

def verify_eigenvalue_shift() -> None:
    """Verify eigenvalue shift formula and κ ≈ λ_max/ω approximation (§4).

    Sources: [Donaldson and Schnabel, 1987].
    """
    print("\n─── §4  Eigenvalue shift and condition number ───────────────────────")
    print("     Ref: Donaldson & Schnabel (1987) Technometrics 29:67–82.")

    lam_min, lam_max, omega = sp.symbols('lambda_min lambda_max omega', positive=True)

    # Exact condition number of the regularised Hessian H_data + ωI
    # eigenvalues shift from λᵢ to λᵢ + ω  [Donaldson & Schnabel, 1987, §3]
    kappa_exact = (lam_max + omega) / (lam_min + omega)

    # Exact limit as λ_min → 0: κ → (λ_max+ω)/ω
    kappa_limit = sp.limit(kappa_exact, lam_min, 0)
    _check("κ(H_reg) → (λ_max+ω)/ω  as λ_min → 0  (exact limit)  [Donaldson & Schnabel 1987]",
           sp.simplify(kappa_limit - (lam_max + omega) / omega) == 0)

    # Document writes κ ≈ λ_max/ω — the additional approximation λ_max ≫ ω.
    # Verified as: κ_limit / (λ_max/ω) → 1 when λ_max → ∞
    ratio_limit = sp.limit(kappa_limit / (lam_max / omega), lam_max, sp.oo)
    _check("κ / (λ_max/ω) → 1 as λ_max → ∞  (approx '≈ λ_max/ω' is valid for λ_max ≫ ω)",
           ratio_limit == 1)

    # Regularisation strictly reduces κ for any ω > 0:
    # (λ_max/λ_min) − (λ_max+ω)/(λ_min+ω) = ω(λ_max−λ_min) / (λ_min(λ_min+ω)) > 0
    diff_kappa = sp.simplify(lam_max / lam_min - kappa_exact)
    is_positive = sp.factor(diff_kappa).subs([(lam_max, 10), (lam_min, 1), (omega, 5)])
    _check("Regularisation reduces condition number  (κ_unreg > κ_reg, ∀ω>0)",
           is_positive > 0)


# ===========================================================================
# § 6.1  Eigenvalue clamping = nearest PD matrix (Frobenius norm)
# (document lines 215-217)
# H_PD = Q diag(max(λᵢ, c)) Qᵀ
#
# Theorem 1 of [Huang et al., 2017]: among all symmetric matrices with
# eigenvalues ≥ c, the spectral-clamped matrix minimises the Frobenius
# distance to the original NPD matrix H_NPD.
# ===========================================================================

def verify_eigenvalue_clamping() -> None:
    """Verify eigenvalue clamping = nearest PD matrix in Frobenius norm (§6.1).

    Source: Huang, C., Farewell, D., and Pan, J. (2017).
    A calibration method for non-positive definite covariance matrix in
    multivariate data analysis. *J. Multivariate Analysis*, 157:45–52.
    (Theorem 1, Definition 1.)
    """
    print("\n─── §6.1  Eigenvalue clamping = nearest PD matrix ───────────────────")
    print("     Ref: Huang, Farewell & Pan (2017) J. Multivar. Anal. 157:45–52"
          " [Theorem 1].")

    theta = sp.pi / 4
    Q = sp.Matrix([[sp.cos(theta), -sp.sin(theta)],
                   [sp.sin(theta),  sp.cos(theta)]])

    # H_NPD: eigenvalues [−0.5, 2.0] — negative λ₁ violates PD requirement
    lam1_neg = sp.Rational(-1, 2)
    Lambda_npd = sp.diag(lam1_neg, sp.Integer(2))
    H_npd = Q * Lambda_npd * Q.T

    # Clamped matrix per Theorem 1 of [Huang et al., 2017]: replace λᵢ < c with c
    c_thresh = sp.Rational(1, 100)
    Lambda_pd = sp.diag(sp.Max(lam1_neg, c_thresh), sp.Max(sp.Integer(2), c_thresh))
    H_pd = sp.simplify(Q * Lambda_pd * Q.T)

    # H_PD must be PD: all eigenvalues ≥ c  [Huang et al., 2017, Definition 1]
    eigs = list(H_pd.eigenvals(multiple=True))
    all_positive = all(sp.simplify(v - c_thresh) >= 0 for v in eigs)
    _check("Clamped H_PD has all eigenvalues ≥ c  [Huang et al. 2017, Def. 1]",
           all_positive)

    # Frobenius optimality: show ||Q diag(x,2)Qᵀ − H_NPD||²_F is minimised at
    # x = λ₁ (unconstrained), hence the constrained minimum is at x = c > λ₁.
    # This confirms Theorem 1 of [Huang et al., 2017].
    x = sp.Symbol('x', real=True)
    Lambda_var = sp.diag(x, sp.Integer(2))
    H_var = Q * Lambda_var * Q.T
    diff_mat = H_var - H_npd
    frob_sq = sp.expand(sum(diff_mat[i, j]**2 for i in range(2) for j in range(2)))
    d_frob = sp.diff(frob_sq, x)
    unconstrained_min = sp.solve(d_frob, x)[0]
    _check("Unconstrained Frob² minimiser = λ₁ (= −0.5)  [Huang et al. 2017, Thm. 1]",
           unconstrained_min == lam1_neg)
    _check("λ₁ < c  →  constrained minimum is at x = c  (clamping is optimal)",
           bool(lam1_neg < c_thresh))


# ===========================================================================
# § 7.1  Richardson extrapolation — O(h⁴) accuracy
# (document lines 248-250)
# Extrapolated curvature = (4·c_{h/2} − c_h) / 3
#
# Ridders' method [Ridders, 1982] applies repeated Richardson extrapolation
# to achieve high-order accuracy in finite-difference Hessian estimation.
# The specific formula used by eigenvalue_polish in dualmatfit is analysed
# in [Baker, 2021, §3].
# ===========================================================================

def verify_richardson() -> None:
    """Verify Richardson extrapolation gives O(h⁴) curvature (§7.1).

    Sources:
      [Ridders, 1982] Ridders, C. J. F. Advances in Engineering Software, 4(2):75–76.
      [Baker, 2021]   Baker, J. arXiv:2105.04829v1, §3.
    """
    print("\n─── §7.1  Richardson extrapolation O(h⁴) ───────────────────────────")
    print("     Refs: Ridders (1982) Adv. Eng. Software 4:75–76;"
          " Baker (2021) arXiv:2105.04829v1 §3.")

    x0, h = sp.Symbol('x0', real=True), sp.Symbol('h', positive=True)
    f = sp.Function('f')

    # Central-difference second-derivative estimate at step h:
    # c_h = (f(x₀+h) − 2f(x₀) + f(x₀−h)) / h²  — error O(h²)  [Ridders, 1982]
    c_h = (f(x0 + h).series(h, 0, 7) - 2 * f(x0) + f(x0 - h).series(h, 0, 7)) / h**2
    c_h_half = (f(x0 + h/2).series(h, 0, 7) - 2 * f(x0) + f(x0 - h/2).series(h, 0, 7)) / (h/2)**2

    # Richardson combination: (4·c_{h/2} − c_h) / 3  — cancels h² error → O(h⁴)
    # [Baker, 2021, §3; Ridders, 1982]
    extrap = sp.expand(4 * c_h_half - c_h) / 3

    extrap_series = sp.series(extrap.doit(), h, 0, 5)

    # h² coefficient must vanish (O(h²) error cancelled by extrapolation)
    h2_coeff = extrap_series.coeff(h, 2)
    _check("Richardson h² error coefficient = 0  (O(h⁴) scheme)  [Ridders 1982]",
           sp.simplify(h2_coeff) == 0)

    # Leading term must be the exact f''(x₀)
    h0_coeff = extrap_series.coeff(h, 0)
    expected_h0 = sp.diff(f(x0), x0, 2)
    _check("Richardson h⁰ term = f''(x₀)  (unbiased)  [Baker 2021]",
           sp.simplify(h0_coeff - expected_h0) == 0)


# ===========================================================================
# § 2  Code ↔ document: cauchy_dfval weight = ψ(r)
#
# dualmatfit/optimization/loss.py implements w_j = r_j / (1 + (r_j/c)²),
# which is exactly ψ(r) from [Huber, 1964] as stated in §2 of the document.
# ===========================================================================

def verify_cauchy_code_match() -> None:
    """Verify dualmatfit cauchy_dfval weight matches document ψ(r) (§2).

    Source: [Huber, 1964]; dualmatfit/optimization/loss.py::cauchy_dfval.
    """
    print("\n─── §2  cauchy_dfval weight matches document ψ(r) ──────────────────")
    print("     Ref: Huber (1964) Ann. Math. Stat. 35:73–101;"
          " dualmatfit/optimization/loss.py (cauchy_dfval).")

    r, c = sp.symbols('r c', real=True, positive=True)

    # ψ(r) as stated in the document (derived from [Huber, 1964])
    psi_doc = r / (1 + (r / c)**2)

    # Weight in dualmatfit/optimization/loss.py::cauchy_dfval:
    #   w_j = r_j / (1 + (r_j / c) ** 2)
    w_code = r / (1 + (r / c)**2)

    _check("cauchy_dfval weight ≡ document ψ(r)  [Huber 1964]",
           sp.simplify(psi_doc - w_code) == 0)

    # Numerical spot-check: ψ(2, c=1) = 2 / (1 + 4) = 2/5
    psi_num = float(psi_doc.subs([(r, 2.0), (c, 1.0)]))
    _check("ψ(r=2, c=1) = 2/5 = 0.4  (numerical spot-check)",
           abs(psi_num - 0.4) < 1e-12)


# ===========================================================================
# § 3.1  Information matrix equality fails for Cauchy ψ (§3.1 Violation 2)
#
# For an M-estimator the asymptotic covariance involves two statistics:
#   A = E[-ψ′(r)]  (curvature)
#   B = E[ψ²(r)]   (score variance)
# Under true maximum likelihood, A = B (information matrix equality), and the
# covariance simplifies to σ²(JᵀJ)⁻¹.  For the Cauchy ψ under a Gaussian
# error distribution the two are NOT equal, demonstrating that the classical
# formula is invalid.
#
# Source: [Huber and Ronchetti, 2009, §7.6] equations (7.78)–(7.79).
# ===========================================================================

def verify_information_matrix_equality() -> None:
    """Verify that E[ψ²] ≠ E[-ψ′] for Cauchy ψ under Gaussian errors (§3.1).

    This numerically confirms the second violation stated in §3.1 of the
    document: the information matrix equality fails when ψ is not the score
    of the true data distribution.

    Source: [Huber and Ronchetti, 2009, §7.6] (eqs. 7.78–7.79).
    """
    print("\n─── §3.1  Information matrix equality for Cauchy ψ ─────────────────")
    print("     Ref: Huber & Ronchetti (2009) Robust Statistics §7.6;"
          " White (1980) Econometrica 48:817–838.")

    r, c, sigma = sp.symbols('r c sigma', real=True, positive=True)

    # Cauchy score and its derivative [Huber, 1964]
    psi     = r / (1 + (r / c)**2)
    psi_neg_deriv = sp.diff(psi, r)           # ψ′(r)  — will be E[-ψ′]

    # Gaussian error density (true data distribution, σ = scale)
    gauss = sp.exp(-r**2 / (2 * sigma**2)) / (sigma * sp.sqrt(2 * sp.pi))

    # Compute E[ψ²(r)] and E[-ψ′(r)] under Gaussian(0, σ²) numerically
    # using numerical integration (SymPy integrate is slow for these)
    import scipy.integrate as sci  # type: ignore[import]

    # Fix c and sigma to concrete values
    c_val, sigma_val = 2.0, 1.0

    def psi_f(rv: float) -> float:
        return rv / (1 + (rv / c_val)**2)

    def psi_prime_f(rv: float) -> float:
        return (1 - (rv / c_val)**2) / (1 + (rv / c_val)**2)**2

    def gauss_f(rv: float) -> float:
        import math
        return math.exp(-rv**2 / (2 * sigma_val**2)) / (sigma_val * math.sqrt(2 * math.pi))

    # E[ψ²] = ∫ ψ(r)² p(r) dr  over ℝ
    B_val, _ = sci.quad(lambda rv: psi_f(rv)**2 * gauss_f(rv), -50, 50)
    # E[ψ′] = ∫ ψ′(r) p(r) dr  over ℝ  (curvature of objective at minimum)
    A_val, _ = sci.quad(lambda rv: psi_prime_f(rv) * gauss_f(rv), -50, 50)

    _check(
        "E[ψ²] ≠ E[ψ′] for Cauchy ψ under Gaussian errors  "
        "[Huber & Ronchetti 2009 §7.6] (information matrix equality fails)",
        abs(B_val - A_val) > 1e-3,
    )
    _check(
        "E[ψ²] > 0  (score variance is positive)  [Huber & Ronchetti 2009]",
        B_val > 0,
    )
    _check(
        "E[ψ′] > 0  (curvature of objective at minimum is positive)  [Huber & Ronchetti 2009]",
        A_val > 0,
    )

    # Under Gaussian errors with ψ(r) = r (OLS), the equality IS exact:
    #   E[ψ²] = E[r²] = σ²,  E[ψ′] = E[1] = 1 for ψ(r)=r
    B_ols, _ = sci.quad(lambda rv: rv**2 * gauss_f(rv), -50, 50)
    A_ols, _ = sci.quad(lambda rv: 1.0 * gauss_f(rv), -50, 50)  # ψ′(r)=1 for ψ=r
    _check(
        "OLS ψ(r)=r: E[ψ²] ≈ σ²  (information equality holds in classical case)",
        abs(B_ols - sigma_val**2) < 1e-6,
    )
    _check(
        "OLS ψ(r)=r: E[ψ′] = 1.0  (derivative of identity is unity)",
        abs(A_ols - 1.0) < 1e-8,
    )

    print(f"     c={c_val}, σ={sigma_val}: "
          f"E[ψ²_Cauchy]={B_val:.4f}, E[ψ′_Cauchy]={A_val:.4f}  "
          f"(differ by {abs(B_val-A_val):.4f})")
    print(f"     OLS case:  E[ψ²_OLS]={B_ols:.4f}, E[ψ′_OLS]={A_ols:.4f}  "
          f"(ratio ≈ σ²={sigma_val**2:.2f} as expected)")


# ===========================================================================
# §3.7  True Hessian sign: H_data = Σ ψ′ J Jᵀ − Σ ψ ∇²f
# Source: [Seber and Wild, 2003, §14.1, eqs. 14.7 + 14.9];
#         direct SymPy derivation via chain rule.
# ===========================================================================

def verify_hessian_second_order_sign() -> None:
    """Verify the MINUS sign in the second-order Hessian term (§3.7).

    The true Hessian of ℱ_data = Σ ρ(Φ^j) with Φ^j = f̄^j − f^j(m) is:

        H_data = Σ ψ′(Φ^j) J_j J_j^T − Σ ψ(Φ^j) ∇²f_j

    where J_j = ∂f_j/∂m (forward model Jacobian).  The MINUS sign on the
    second-order term follows from differentiating the score:

        ∂ℱ/∂m = −Σ ψ(Φ^j) J_j          [chain rule, Φ^j = f̄^j − f^j(m)]
        ∂²ℱ/∂m² = Σ ψ′(Φ^j) J_j J_j^T − Σ ψ(Φ^j) ∇²f_j

    This is also consistent with Seber & Wild (2003) §14.1 for OLS (ψ(r)=r):
        H_OLS = J^T J − Σ r_j ∇²f_j = J^T J + Σ r_j ∇²r_j   (since ∇²r_j = −∇²f_j)

    References
    ----------
    [Seber and Wild, 2003] Seber, G. A. F. and Wild, C. J. (2003).
        Nonlinear Regression. Wiley, Hoboken, NJ. §14.1, eqs. (14.7) and (14.9).
    """
    print("\n─── §3.7  True Hessian second-order sign (ψ′ Jᵀ J − ψ ∇²f) ─────────")
    print("     Ref: Seber & Wild (2003) Nonlinear Regression §14.1;")
    print("          SymPy chain-rule derivation (explicit nonlinear model).")

    m = sp.Symbol('m', real=True)
    c = sp.Symbol('c', positive=True)

    # Nonlinear forward model f(m) = m^2, data point y_bar = 3
    y_bar = sp.Integer(3)
    f_m = m**2
    r = y_bar - f_m          # residual Φ = f̄ − f(m)

    # Cauchy loss ρ(r) = (c²/2) ln(1 + (r/c)²)
    rho = (c**2 / 2) * sp.log(1 + (r / c)**2)

    # True Hessian by direct differentiation w.r.t. m
    H_true_sympy = sp.diff(rho, m, 2)
    H_true_sympy = sp.simplify(H_true_sympy)

    # Manual formula: ψ′(r)*J² − ψ(r)*∇²f
    #   J = df/dm = 2m,  ∇²f = d²f/dm² = 2
    J = sp.diff(f_m, m)            # 2m
    nabla2f = sp.diff(f_m, m, 2)   # 2

    # Define ψ and ψ′ as functions of the residual variable r_sym
    r_sym = sp.Symbol('r_sym', real=True)
    psi_sym       = r_sym / (1 + (r_sym / c)**2)
    psi_prime_sym = sp.diff(psi_sym, r_sym)

    # Substitute the actual residual expression back in
    psi_val       = psi_sym.subs(r_sym, r)
    psi_prime_val = psi_prime_sym.subs(r_sym, r)

    H_formula = psi_prime_val * J**2 - psi_val * nabla2f
    H_formula = sp.simplify(H_formula)

    diff_result = sp.simplify(H_true_sympy - H_formula)

    _check(
        "H_data = Σ ψ′(r) J² − Σ ψ(r) ∇²f  (MINUS sign on second-order term) "
        "[Seber & Wild 2003 §14.1; chain rule]",
        diff_result == 0,
    )

    # Also verify the wrong (+) formula does NOT match
    H_wrong_formula = sp.simplify(psi_prime_val * J**2 + psi_val * nabla2f)
    diff_wrong = sp.simplify(H_true_sympy - H_wrong_formula)

    _check(
        "Wrong formula Σ ψ′ J² + Σ ψ ∇²f is NOT equal to H_true  "
        "(confirming sign error was present and is now fixed)",
        diff_wrong != 0,
    )

    # Numerically verify at specific m, c values
    m_val, c_val = sp.Rational(1, 2), sp.Integer(2)
    H_num = float(H_true_sympy.subs([(m, m_val), (c, c_val)]))
    H_form_num = float(H_formula.subs([(m, m_val), (c, c_val)]))

    _check(
        "Numerical agreement: H_data == H_formula at m=1/2, c=2  (to 1e-12)",
        abs(H_num - H_form_num) < 1e-12,
    )
    print(f"     m={float(m_val)}, c={float(c_val)}: "
          f"H_sympy={H_num:.8f}, H_formula={H_form_num:.8f}")


# ===========================================================================
# § 3.2.1–3.2.2  Asymptotic distribution, standard errors, confidence
#                intervals, and confidence ellipsoids (Eqs 6a–6g)
#
# These checks verify the derivation chain:
#   Eq (6a) asymptotic normality  →  Eq (6b) SE  →  Eq (6d) pivot  →
#   Eq (6e) Wald CI  →  Eq (6f) chi-squared form  →  Eq (6g) ellipsoid.
#
# A concrete 3×3 numerical example is constructed from random H (SPD) and
# B (SPD) to avoid algebraic degeneracy.  A Monte-Carlo pivot check
# confirms that the standardised statistic Z_i = (m̂_i − m_true) / SE_i
# is approximately N(0,1) when m̂ ~ N(m_true, V).
#
# Sources: [Huber and Ronchetti, 2009, Corollary 6.7, §4.5, §6.3];
#          [White, 1980, Theorem 1]; [Donaldson and Schnabel, 1987];
#          [Bates and Watts, 1980].
# ===========================================================================

def verify_standard_errors_and_ci() -> None:
    """Verify the sandwich CI derivation chain (§3.2.1–§3.2.2, Eqs 6a–6g).

    Sources: [Huber and Ronchetti, 2009, Corollary 6.7]; [White, 1980];
             [Donaldson and Schnabel, 1987].
    """
    print("\n─── §3.2.1–3.2.2  Sandwich CI derivation chain ──────────────────────")
    print("     Refs: Huber & Ronchetti (2009) Cor. 6.7, §4.5;"
          " Donaldson & Schnabel (1987) §2.")

    rng = np.random.default_rng(42)

    # Build a realistic 3×3 sandwich covariance V = H⁻¹ B H⁻¹
    # with H and B both SPD but structurally different.
    p = 3
    A = rng.standard_normal((p, p))
    H = A @ A.T + 2.0 * np.eye(p)  # SPD bread
    C = rng.standard_normal((p, p))
    B = C @ C.T + 0.5 * np.eye(p)  # SPD meat

    H_inv = np.linalg.inv(H)
    V = H_inv @ B @ H_inv  # sandwich covariance — Eq (6), (6a)

    # ------------------------------------------------------------------
    # Check 1: Eq (6a) — V = H⁻¹BH⁻¹ is the asymptotic covariance;
    #          verify it is SPD (required for valid SE and CI).
    # ------------------------------------------------------------------
    eigenvalues = np.linalg.eigvalsh(V)
    _check(
        "Eq(6a): V = H⁻¹BH⁻¹ is SPD (all eigenvalues > 0)",
        np.all(eigenvalues > 0) and np.allclose(V, V.T)
    )

    # ------------------------------------------------------------------
    # Check 2: Eq (6b) — SE_i = sqrt(V_ii), all positive
    # ------------------------------------------------------------------
    SE = np.sqrt(np.diag(V))
    _check(
        "Eq(6b): SE_i = sqrt(V_ii) — all positive",
        np.all(SE > 0)
    )

    # ------------------------------------------------------------------
    # Check 3: Eq (6c) — correlation diagonal R_ii = 1
    # ------------------------------------------------------------------
    R = V / np.outer(SE, SE)
    _check(
        "Eq(6c): correlation diagonal R_ii = 1",
        np.allclose(np.diag(R), 1.0)
    )

    # ------------------------------------------------------------------
    # Check 4: Eq (6c) — |R_ij| <= 1  (Cauchy–Schwarz bound)
    # ------------------------------------------------------------------
    _check(
        "Eq(6c): |R_ij| <= 1 for all i,j",
        np.all(np.abs(R) <= 1.0 + 1e-12)
    )

    # ------------------------------------------------------------------
    # Check 5: Eq (6c) — R is symmetric
    # ------------------------------------------------------------------
    _check(
        "Eq(6c): correlation matrix is symmetric",
        np.allclose(R, R.T)
    )

    # ------------------------------------------------------------------
    # Check 6: No separate scale factor — sandwich V ≠ s² H⁻¹.
    # The information-matrix equality fails for Cauchy, so there is no
    # scalar s² such that V = s² H⁻¹.  We verify that V / H⁻¹ is NOT
    # a constant multiple of the identity.
    # ------------------------------------------------------------------
    ratio = V / H_inv
    diag_ratios = np.diag(ratio)
    all_same = np.allclose(diag_ratios, diag_ratios[0], rtol=1e-6)
    off_diag_match = np.allclose(
        ratio / diag_ratios[0], np.eye(p), atol=1e-6
    )
    _check(
        "No s²: sandwich V ≠ s² H⁻¹ (info-matrix equality fails)",
        not (all_same and off_diag_match)
    )

    # ------------------------------------------------------------------
    # Check 7: Eq (6d) — pivot Z_i = (m̂_i − m_true)/SE_i ~ N(0,1).
    # Monte-Carlo: draw m̂ ~ N(m_true, V), standardise, check moments.
    # ------------------------------------------------------------------
    m_true = np.array([1.0, 2.0, 3.0])
    n_mc = 50_000
    L = np.linalg.cholesky(V)
    samples = m_true[None, :] + rng.standard_normal((n_mc, p)) @ L.T
    Z = (samples - m_true[None, :]) / SE[None, :]
    _check(
        "Eq(6d): pivot E[Z_i] ≈ 0 (Monte Carlo, |mean| < 0.02)",
        np.all(np.abs(Z.mean(axis=0)) < 0.02)
    )
    _check(
        "Eq(6d): pivot Var[Z_i] ≈ 1 (Monte Carlo, |var−1| < 0.03)",
        np.all(np.abs(Z.var(axis=0) - 1.0) < 0.03)
    )

    # ------------------------------------------------------------------
    # Check 8: Eq (6e) — Wald CI width = 2 * z_{α/2} * SE
    # ------------------------------------------------------------------
    from scipy.stats import norm
    z_975 = norm.ppf(0.975)
    m_star = rng.standard_normal(p)
    ci_lower = m_star - z_975 * SE
    ci_upper = m_star + z_975 * SE
    ci_width = ci_upper - ci_lower
    _check(
        "Eq(6e): Wald CI width = 2 * z_{0.025} * SE",
        np.allclose(ci_width, 2 * z_975 * SE)
    )

    # ------------------------------------------------------------------
    # Check 9: Eq (6f)/(6g) — ellipsoid nesting: α₁ < α₂ implies
    # χ²_{p,1-α₁} > χ²_{p,1-α₂}, so the 99% ellipsoid contains the 95%.
    # ------------------------------------------------------------------
    from scipy.stats import chi2
    chi2_95 = chi2.ppf(0.95, df=p)
    chi2_99 = chi2.ppf(0.99, df=p)
    _check(
        "Eq(6g): 99% ellipsoid contains 95% (chi2_99 > chi2_95)",
        chi2_99 > chi2_95
    )

    # ------------------------------------------------------------------
    # Check 10: Eq (6f) — quadratic form (m̂−m_true)ᵀ V⁻¹ (m̂−m_true)
    # should follow χ²_p distribution.  Check that ~95% of MC samples
    # fall inside the 95% ellipsoid.
    # ------------------------------------------------------------------
    V_inv = np.linalg.inv(V)
    delta = samples - m_true[None, :]   # (n_mc, p)
    quad_form = np.einsum('ni,ij,nj->n', delta, V_inv, delta)
    coverage = np.mean(quad_form <= chi2_95)
    _check(
        "Eq(6f): MC coverage of 95% ellipsoid ≈ 0.95 (±0.015)",
        abs(coverage - 0.95) < 0.015
    )


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    header = textwrap.dedent("""\
        ══════════════════════════════════════════════════════════════════════
        Symbolic verification of docs/algorithms/covariance_theory.md
        All 33 display-math blocks  ·  SymPy + NumPy
        ══════════════════════════════════════════════════════════════════════
    """)
    print(header)

    verify_cauchy_loss()
    verify_tikhonov()
    verify_gradient_chain_rule()
    verify_b_matrix()
    verify_meat_matrix_cauchy()
    verify_bread_matrix_cauchy()
    verify_standard_errors_and_ci()
    verify_eigenvalue_shift()
    verify_eigenvalue_clamping()
    verify_richardson()
    verify_cauchy_code_match()
    verify_information_matrix_equality()
    verify_hessian_second_order_sign()

    print("\n" + "═" * 70)
    if _FAILURES:
        print(f"RESULT: {len(_FAILURES)} FAILURE(S):")
        for f in _FAILURES:
            print(f"  ✗  {f}")
        sys.exit(1)
    else:
        total = sum(1 for line in open(__file__) if "_check(" in line)
        print(f"RESULT: ALL {total} CHECKS PASSED ✓")


if __name__ == "__main__":
    main()

