# -*- coding: utf-8 -*-
"""
Variational formulation for hyperelastic material models.

This module provides the VariationalFormulation class for constructing
strain energy functions and their derivatives for the Holzapfel-Gasser-Ogden
(HGO) and related hyperelastic material models.
"""
import numpy as np
import sympy as sy

from typing import Tuple, List, Dict, Union, Any
from sympy import latex
from dualmatfit.formulation.tensor import TensorManager, safe_simplify
from dualmatfit.formulation.material_law import (neo_hookean, fung, volumetric_strain, anisotropic_invariant,
                                    anisotropic_strain, get_fiber_vector, right_cauchy_fun)

from dualmatfit.utils.logging_config import get_logger
logger = get_logger('variational_form')

__all__ = [
    'VariationalFormulation',
    'ring_geom',
    'isotropic_strain_energy',
    'NEO_HOOKEAN',
    'FUNG',
    'SUPPORTED_ITYPES',
]

# Material model constants
NEO_HOOKEAN = 'neo_hookean'
FUNG = 'fung'
SUPPORTED_ITYPES = {'nh', NEO_HOOKEAN, FUNG}

# See Article: "On the Compressibility of Arterial Tissue"
# lwr: 42.14, upp: 99.03 (kPa)
bulk_val = 99.03 / 1000.            # Median Value
# bulk_val = 56.67 / 1000.          # Median Value [MPa]


def ring_geom(info_data, pos) -> float:
    """Calculates cross-sectional area 'ds' and mean perimeter 'dp' for a ring section."""
    dl = info_data[pos]['len']
    de = info_data['thick']
    ds = dl * de
    info_data[pos]['ds'] = ds
    info_data[pos]['dp'] = np.pi * info_data['dia']
    return ds


def isotropic_strain_energy(material_type: str,
                            iso_split: bool,
                            mat_vars: Dict[str, Union[sy.symbols, sy.Expr, float]],
                            tensor_manager: TensorManager,
                            ) -> Union[sy.symbols, sy.Expr, float]:
    """
    Construct the isotropic part of the strain energy function (Ψ_iso).

    This component models the mechanical behavior of the non-collagenous ground
    substance (elastin, proteoglycans, water) that provides the isotropic baseline
    response of the tissue matrix.

    Mathematical Formulation
    ------------------------
    Two isotropic models are supported:

    **Neo-Hookean Model** (material_type='nh' or 'neo_hookean', default):
        Ψ_iso = (μ/2)(Ī₁ - 3)

        where:
        - μ: Shear modulus [MPa]
        - Ī₁ = J⁻²/³ I₁: Isochoric first invariant (if iso_split=True)
        - I₁ = tr(C): First invariant of right Cauchy-Green tensor

        Typical range: μ = 0.01 - 0.1 MPa for arterial tissue

    **Fung Model** (material_type='fung'):
        Ψ_iso = (aₑ/2bₑ)[exp(bₑ(Ī₁ - 3)) - 1]

        where:
        - aₑ: Material parameter [MPa]
        - bₑ: Dimensionless exponential parameter [-]

        The Fung model provides exponential strain-stiffening for the matrix.

    Isochoric-Volumetric Split
    ---------------------------
    When iso_split=True, the strain energy is evaluated using the isochoric
    invariant:
        Ī₁ = J⁻²/³ I₁

    This multiplicative decomposition improves numerical conditioning for nearly
    incompressible materials by decoupling volumetric and deviatoric responses:
        F = J^(1/3) F̄    (where det(F̄) = 1)

    Physical Interpretation
    -----------------------
    The shear modulus μ controls:
    - Initial stiffness in simple shear
    - Baseline tissue compliance
    - Ground substance contribution relative to fibers

    For arterial tissue:
    - Intima: μ ≈ 0.01 - 0.03 MPa
    - Media: μ ≈ 0.01 - 0.05 MPa
    - Adventitia: μ ≈ 0.02 - 0.1 MPa

    The ground substance is softer than collagen fibers (k₁), providing
    extensibility at low strains before fiber engagement.

    References
    ----------
    .. [1] Holzapfel, G. A., & Ogden, R. W. (2010). Constitutive modelling of 
           arteries. Proceedings of the Royal Society A, 466(2118), 1551-1597.
           DOI: 10.1098/rspa.2010.0058

    .. [2] Holzapfel, G. A., et al. (2005). Determination of layer-specific 
           mechanical properties of human coronary arteries with nonatherosclerotic 
           intimal thickening. American Journal of Physiology-Heart and 
           Circulatory Physiology, 289(5), H2048-H2058.
           DOI: 10.1152/ajpheart.00934.2004

    .. [3] Fung, Y. C. (1993). Biomechanics: Mechanical Properties of Living 
           Tissues (2nd ed.). Springer-Verlag, New York.

    Parameters
    ----------
    material_type : str
        Isotropic constitutive model:
        - 'nh' or 'neo_hookean': Neo-Hookean model (default, recommended)
        - 'fung': Fung exponential model
    iso_split : bool
        Apply isochoric-volumetric split:
        - True: Use Ī₁ = J⁻²/³ I₁ (recommended for ν > 0.45)
        - False: Use I₁ directly (for compressible materials)
    mat_vars : Dict[str, Union[sy.Symbol, sy.Expr, float]]
        Dictionary of material parameters:
        - For Neo-Hookean: 'mu' (shear modulus) [MPa]
        - For Fung: 'a_f', 'b_f' (material parameters)
    tensor_manager : TensorManager
        Manages symbolic tensor expressions and provides access to F, I₁, J.

    Returns
    -------
    Union[sy.Symbol, sy.Expr, float]
        Symbolic expression for isotropic strain energy Ψ_iso.

    Raises
    ------
    NotImplementedError
        If an unsupported material_type is specified.

    Notes
    -----
    - Neo-Hookean is the simplest hyperelastic model and works well for most applications
    - Fung model provides additional strain-stiffening for the ground substance
    - The isochoric split is essential for nearly incompressible materials (ν ≈ 0.5)
    - When iso_split=False, a volumetric stabilization term may be added automatically
    - See docs/algorithms/hgo_model.md for detailed mathematical derivations

    See Also
    --------
    volumetric_strain_energy : Volumetric contribution (Ψ_vol)
    anisotropic_strain_energy : Fiber contribution (Ψ_ani)
    VariationalFormulation : Main class for complete strain energy

    """
    vol_stabilization = not iso_split

    if material_type in {'nh', NEO_HOOKEAN}:
        psi_iso = neo_hookean(mat_vars["mu"],
                              tensor_manager.get_symbol_by_index("F"),
                              isochoric=iso_split,
                              volumetric=vol_stabilization,
                              )

    elif material_type == FUNG:
        a_f, b_f = sy.symbols('a_f b_f', positive=True)
        psi_iso = fung(a_f, b_f, tensor_manager.get_symbol_by_index("F"), isochoric=iso_split)

    else:
        raise NotImplementedError("This material type is not implemented.")

    for sym_idx in ["Iv_1", "J"]:
        psi_iso = psi_iso.subs(tensor_manager.get_expression_by_index(sym_idx),
                               tensor_manager.get_symbol_by_index(sym_idx))

    return psi_iso


def volumetric_strain_energy(vol_type: str,
                             tensor_manager: TensorManager,
                             ) -> Union[sy.Symbol, sy.Expr, float]:
    """
    Construct the volumetric part of the strain energy function (Ψ_vol).

    The volumetric energy component penalizes deviations from incompressibility
    in nearly incompressible materials. Different formulations provide varying
    numerical characteristics and stability properties.

    Mathematical Formulation
    ------------------------
    The volumetric strain energy takes the general form:

        Ψ_vol = Ψ_vol(J)

    where J = det(F) is the Jacobian (volume ratio). Three formulations are supported:

    **Simo 1992** (vol_type='simo92', default):
        Ψ_vol = (D/4)(J² - 1 - 2ln(J))

    **Sussman-Bathe 1987** (vol_type='bathe87'):
        Ψ_vol = (D/2)(J - 1)²

    **Doll-Schweizerhof 2000** (vol_type='doll8'):
        Ψ_vol = D[(J - 1)²/2 + (J⁻¹ - 1)²/2]

    where D = 2/κ with κ being the bulk modulus.

    Physical Interpretation
    -----------------------
    - D → 0 (κ → ∞): Incompressible limit
    - D small: Nearly incompressible (typical for soft tissues)
    - For arterial tissue: κ ≈ 50-100 MPa, D ≈ 0.02-0.04 MPa⁻¹

    The choice of volumetric form affects:
    1. Numerical stability near J = 1
    2. Behavior under extreme compression/expansion
    3. Conditioning of the tangent stiffness matrix

    Selection Guidelines
    --------------------
    - **simo92**: Best overall performance, symmetric behavior
    - **bathe87**: Simpler form, good for mild compressibility
    - **doll8**: Most stable for large volume changes, bidirectional penalty

    References
    ----------
    .. [1] Simo, J. C., & Taylor, R. L. (1991). Quasi-incompressible finite 
           elasticity in principal stretches. Computer Methods in Applied 
           Mechanics and Engineering, 85(3), 273-310.

    .. [2] Sussman, T., & Bathe, K. J. (1987). A finite element formulation for 
           nonlinear incompressible elastic and inelastic analysis. Computers & 
           Structures, 26(1-2), 357-409. DOI: 10.1016/0045-7949(87)90265-3

    .. [3] Doll, S., & Schweizerhof, K. (2000). On the development of volumetric 
           strain energy functions. Journal of Applied Mechanics, 67(1), 17-21.

    .. [4] Holzapfel, G. A., & Ogden, R. W. (2010). Constitutive modelling of 
           arteries. Proceedings of the Royal Society A, 466(2118), 1551-1597.
           DOI: 10.1098/rspa.2010.0058

    Parameters
    ----------
    vol_type : str
        Type of volumetric strain energy formulation:
        - 'simo92': Simo (1992) form (default, recommended)
        - 'bathe87': Sussman-Bathe (1987) form
        - 'doll8': Doll-Schweizerhof (2000) form
    tensor_manager : TensorManager
        Manages symbolic tensor expressions and provides access to J (Jacobian).

    Returns
    -------
    Union[sy.Symbol, sy.Expr, float]
        Symbolic expression for volumetric strain energy Ψ_vol(J).

    Notes
    -----
    - The volumetric term is decoupled from isochoric response via isochoric-volumetric split
    - Stabilization may be needed for κ → ∞ (see docs/algorithms/stabilization.md)
    - The Jacobian J must remain positive (J > 0) to maintain material stability

    See Also
    --------
    isotropic_strain_energy : Isochoric isotropic contribution
    mixed_strain_energy_functional : Complete strain energy with mixed formulations
    VariationalFormulation : Main class for strain energy construction

    """

    return volumetric_strain(vol_type, tensor_manager.get_symbol_by_index("J"))


def anisotropic_strain_energy(dim:int,
                              was: bool,
                              mat_vars: Dict[str, Union[sy.symbols, sy.Expr, float]],
                              tensor_manager: TensorManager,
                              hv: bool = False,
                              ) -> Tuple[List[Any], List[Any]]:
    """
    Construct the anisotropic (fiber) part of the strain energy function (Ψ_ani).

    This component models the mechanical contribution of embedded collagen fiber
    families in soft biological tissues, giving rise to characteristic anisotropic
    behavior. The implementation follows the Holzapfel-Gasser-Ogden (HGO) model.

    Mathematical Formulation
    ------------------------
    The anisotropic strain energy for N_f fiber families is:

        Ψ_ani = Σᵢ₌₁ᴺᶠ (k₁/2k₂)[exp(k₂⟨Ēᵢ⟩²) - 1]

    where the pseudo-invariant for fiber family i depends on fiber dispersion:

    **Without dispersion** (κ = 0, was=False):
        Ēᵢ = I₄ᵢ - 1 = C:(aᵢ ⊗ aᵢ) - 1

    **With dispersion** (κ > 0, was=True):
        Ēᵢ = κ I₁ + (1 - 3κ)Ī₄ᵢ - 1

    where:
    - I₁ = tr(C): First invariant
    - I₄ᵢ = C:(aᵢ ⊗ aᵢ): Anisotropic invariant for fiber direction aᵢ
    - Ī₄ᵢ = J⁻²/³I₄ᵢ: Isochoric anisotropic invariant
    - κ ∈ [0, 1/3]: Fiber dispersion parameter
      * κ = 0: Perfect alignment (original HGO)
      * κ = 1/3: Isotropic distribution
      * Typical for arteries: κ = 0.1 - 0.2
    - ⟨·⟩ = max(·, 0): Macaulay brackets or Heaviside(·-1) activation

    Fiber Orientation
    -----------------
    For typical arterial tissue with two fiber families:
        a₁ = [cos(α), sin(α), 0]ᵀ
        a₂ = [cos(α), -sin(α), 0]ᵀ

    where α is the fiber angle from the circumferential direction (typically 30°-50°).

    Physical Interpretation
    -----------------------
    - **k₁** [MPa]: Fiber stiffness, controls initial fiber response
      Range: 0.5 - 50 MPa (tissue dependent)
    - **k₂** [-]: Exponential parameter, controls nonlinearity
      Range: 1 - 10 (higher = stiffer at high strain)
    - **α** [deg]: Fiber angle, determines mechanical anisotropy
      Range: 30° - 50° for arteries
    - **κ** [-]: Dispersion, accounts for fiber distribution
      Range: 0.0 - 0.33 (0.1 - 0.2 typical)

    Fiber Activation
    ----------------
    The Heaviside function H(I₄ - 1) ensures fibers only contribute in tension:
    - I₄ > 1: Fiber stretched → contributes to stress
    - I₄ ≤ 1: Fiber compressed → no contribution

    When hv=False, fibers are always active (more stable numerically).

    References
    ----------
    .. [1] Holzapfel, G. A., Gasser, T. C., & Ogden, R. W. (2000). A new 
           constitutive framework for arterial wall mechanics and a comparative 
           study of material models. Journal of Elasticity, 61(1-3), 1-48.
           DOI: 10.1023/A:1010835316564

    .. [2] Gasser, T. C., Ogden, R. W., & Holzapfel, G. A. (2006). Hyperelastic 
           modelling of arterial layers with distributed collagen fibre 
           orientations. Journal of the Royal Society Interface, 3(6), 15-35.
           DOI: 10.1098/rsif.2005.0073

    .. [3] Holzapfel, G. A., & Ogden, R. W. (2010). Constitutive modelling of 
           arteries. Proceedings of the Royal Society A, 466(2118), 1551-1597.
           DOI: 10.1098/rspa.2010.0058

    .. [4] Nolan, D. R., et al. (2014). A robust anisotropic hyperelastic 
           formulation for the modelling of soft tissue. Journal of the 
           Mechanical Behavior of Biomedical Materials, 39, 48-60.

    Parameters
    ----------
    dim : int
        Spatial dimension (3 for 3D continuum mechanics).
    was : bool
        Weighted Average Structure (WAS) flag:
        - True: Fully incompressible formulation (Ī₄ᵢ with J⁻²/³)
        - False: Compressible formulation (I₄ᵢ directly)
        See Nolan et al. (2014) for compressible vs incompressible discussion.
    mat_vars : Dict[str, Union[sy.Symbol, sy.Expr, float]]
        Dictionary of material parameters:
        - 'k_1': Fiber stiffness [MPa]
        - 'k_2': Exponential parameter [-]
        - 'alpha': Fiber angle [rad]
        - 'kappa': Fiber dispersion parameter [-] (optional, default=0)
    tensor_manager : TensorManager
        Manages symbolic tensor expressions and provides deformation gradient F,
        right Cauchy-Green tensor C, and Jacobian J.
    hv : bool, default=False
        Apply Heaviside function H(I₄ - 1) to enforce tension-only fiber activation:
        - True: Fibers only active in tension (physically accurate)
        - False: Fibers always active (numerically more stable, default)

    Returns
    -------
    Tuple[List[Any], List[Any]]
        - list_psi_ani: List of anisotropic strain energy expressions for each 
          fiber family [Ψ_ani_1, Ψ_ani_2, ...]
        - list_iv4: List of pseudo-invariant expressions for each fiber family
          [Ē₁, Ē₂, ...]

    Notes
    -----
    - Two fiber families are typical for arterial walls (N_f = 2)
    - The exponential form captures strain-stiffening behavior of collagen
    - Dispersion (κ > 0) provides smoother stress-strain response
    - For numerical stability, hv=False is recommended initially
    - See docs/algorithms/hgo_model.md for complete mathematical derivation

    See Also
    --------
    volumetric_strain_energy : Volumetric contribution
    isotropic_strain_energy : Isotropic ground substance contribution
    VariationalFormulation : Main class coordinating all energy components

    """

    if mat_vars.get("kappa") is not None:
        kappa_value = mat_vars["kappa"]
    else:
        kappa_value = 0.0

    def_grad = tensor_manager.get_symbol_by_index("F")

    # Substitute known tensor values into the expression
    substitutions = {tensor_manager.get_expression_by_index("C"): tensor_manager.get_symbol_by_index("C")}

    if not was:
        substitutions[tensor_manager.get_expression_by_index("J")] = tensor_manager.get_symbol_by_index("J")

    list_fr = get_fiber_vector(mat_vars["alpha"], dim=dim)
    list_iv4 = []
    list_psi_ani = []

    # Iterate over fiber directions and compute the associated strains
    for i, fr_i in enumerate(list_fr):

        # Define symbolic variables and tensors related to fiber direction
        nm_fr_i = f"f^r_{i + 1}"
        sy_fr_i = sy.MatrixSymbol(f"f^r_{i + 1}", dim, 1)
        tensor_manager.add(nm_fr_i, sy_fr_i, fr_i)

        # Calculate anisotropic invariants
        nm_mr_i = f"M^r_{i + 1}"
        sy_mr_i = sy.MatrixSymbol(f"M^r_{i + 1}", dim, dim)
        tensor_manager.add(nm_mr_i, sy_mr_i, (sy.tensorproduct(fr_i, fr_i)).tomatrix())

        nm_iv4_i = f"Iv_{4 + 2 * i}"
        sy_iv4_i = sy.symbols(nm_iv4_i, real=True)

        _, expr_iv4_i = anisotropic_invariant(def_grad, sy_mr_i, kappa_value, was, stabilization=False,)

        # Calculation of the Anisotropic Strain Energy in fiber direction "i"
        expr_iv4_i = expr_iv4_i.subs(substitutions)

        tensor_manager.add(nm_iv4_i, sy_iv4_i, expr_iv4_i)
        list_iv4.append(tensor_manager.get_concrete_expression_by_symbol(sy_iv4_i))

        expr_psi_ani_i = anisotropic_strain(mat_vars["k_1"], mat_vars["k_2"], sy_iv4_i, hv=hv)
        list_psi_ani.append(expr_psi_ani_i)

    return list_psi_ani, list_iv4


def mixed_strain_energy_functional(
        mix: int,
        primal_vars: Dict[str, Union[sy.Symbol, sy.Expr, float]],
        mat_vars: Dict[str, Union[sy.symbols, sy.Expr, float]],
        bulk: float,
        vol_flg: bool,
        tensor_manager: TensorManager,
        psi_iso: Union[sy.symbols, sy.Expr, float],
        psi_vol: Union[sy.symbols, sy.Expr, float],
        psi_ani: Union[sy.symbols, sy.Expr, float],
        subs: bool = False,
):
    """
    Assembly full functional
    """

    if vol_flg is True and mat_vars.get("D") is not None:
        bulk_value = mat_vars["D"]
    else:
        bulk_value = bulk

    fields = {nm_i: sy.symbols(idf_i, real=True) for nm_i, idf_i in zip(["iso", "vol", "ani"], ["i", "v", "a"])}

    if mix == 1:
        psi_total = fields["iso"] * psi_iso + fields["vol"] * bulk_value * psi_vol + fields["ani"] * psi_ani

    elif mix == 2:
        psi_vol_m2 = primal_vars["p"][0] * psi_vol - (1. / (2 * bulk_value)) * primal_vars["p"][0] ** 2
        psi_total = fields["iso"] * psi_iso + fields["vol"] * psi_vol_m2 + fields["ani"] * psi_ani

    elif mix == 3:
        psi_vol = (bulk_value * psi_vol.subs(tensor_manager.get_concrete_expression_by_index("J"), primal_vars["t"][0]) +
                   primal_vars["p"][0] * (tensor_manager.get_concrete_expression_by_index("J") - primal_vars["t"][0]))

        psi_total = fields["iso"] * psi_iso  + fields["vol"] * psi_vol + fields["ani"] * psi_ani

    else:
        raise NotImplementedError(f"This mixed variational form is not yet supported: {mix}")

    if subs:
        subs_vars = dict.fromkeys(fields.values(), 1.)
        return psi_total.subs(subs_vars)

    else:
        return psi_total, fields


def _subs_and_eval(expression, symbols, tensor_manager):
    """Helper method to substitute tensor symbols and evaluate the expression."""
    for sym_i in symbols:
        expression = expression.subs(
            tensor_manager.get_symbol_by_index(sym_i),
            tensor_manager.get_expression_by_index(sym_i)
        )
    return expression.doit()


class VariationalFormulation:
    """
    Constructs and manages the symbolic variational formulation for hyperelastic material models.

    This class implements the Holzapfel-Gasser-Ogden (HGO) constitutive model for
    fiber-reinforced biological tissues, particularly arterial walls. It orchestrates
    the symbolic construction of the strain energy function and its derivatives.

    Mathematical Framework
    ----------------------
    The total strain energy density is decomposed as:
    
        Ψ = Ψ_iso + Ψ_vol + Ψ_ani
    
    where:
        - Ψ_iso: Isotropic contribution (ground substance)
        - Ψ_vol: Volumetric contribution (near-incompressibility)
        - Ψ_ani: Anisotropic contribution (collagen fibers)

    References
    ----------
    .. [1] Holzapfel, G. A., Gasser, T. C., & Ogden, R. W. (2000).
           A new constitutive framework for arterial wall mechanics and a
           comparative study of material models. Journal of Elasticity, 61(1-3), 1-48.
           DOI: 10.1023/A:1010835316564

    .. [2] Gasser, T. C., Ogden, R. W., & Holzapfel, G. A. (2006).
           Hyperelastic modelling of arterial layers with distributed collagen
           fibre orientations. Journal of the Royal Society Interface, 3(6), 15-35.
           DOI: 10.1098/rsif.2005.0073

    .. [3] Holzapfel, G. A., & Ogden, R. W. (2010).
           Constitutive modelling of arteries. Proceedings of the Royal Society A,
           466(2118), 1551-1597. DOI: 10.1098/rspa.2010.0058

    .. [4] Simo, J. C., Taylor, R. L., & Pister, K. S. (1985).
           Variational and projection methods for the volume constraint in finite
           deformation elasto-plasticity. Computer Methods in Applied Mechanics
           and Engineering, 51(1-3), 177-208.

    .. [5] Nolan, D. R., et al. (2014). A robust anisotropic hyperelastic
           formulation for the modelling of soft tissue. Journal of the Mechanical
           Behavior of Biomedical Materials, 39, 48-60.

    Parameters
    ----------
    ds : float
        Cross-sectional area of the specimen [mm²].
    itype : str
        Isotropic material model:
        - 'nh' or 'neo_hookean': Neo-Hookean model (default) [1]
        - 'fung': Fung-type exponential model
    mix : int
        Mixed formulation type for near-incompressibility [4]:
        - 1: Standard displacement (u)
        - 2: Two-field displacement-pressure (u-p)
        - 3: Three-field displacement-pressure-dilatation (u-p-θ)
    kappa : bool
        Enable fiber dispersion model [2]. If True, the dispersion parameter
        κ ∈ [0, 1/3] is included as a material parameter.
    dvol : bool
        Treat bulk modulus D as a fitting parameter. If False, uses fixed value.
    bulk : float, optional
        Bulk modulus κ [MPa]. Typical range: 50-100 MPa for arterial tissue.
        Used only if dvol=False.
    iso_split : bool, default=False
        Perform volumetric-isochoric split: Ψ_iso = Ψ_iso(Ī₁) where Ī₁ = J^(-2/3)I₁.
        Improves conditioning for nearly incompressible materials.
    vol_type : str, default='simo92'
        Volumetric strain energy form:
        - 'simo92': Simo et al. (1992) - Ψ_vol = (D/4)(J² - 1 - 2ln(J))
        - 'bathe87': Sussman-Bathe (1987) - Ψ_vol = (D/2)(J - 1)²
        - 'doll8': Doll-Schweizerhof (2000)
        When ``mix == 2``, this argument is ignored and the formulation uses
        the internal ``'Sussman'`` volumetric form.
    hv : bool, default=False
        Apply Heaviside function H(I₄ - 1) to anisotropic energy. Ensures
        fibers only contribute in tension (I₄ > 1).
    was : bool, default=False
        Use Without Anisotropic Split (WAS or Modified Anisotropic) formulation for anisotropic
        invariants. If True, fully incompressible; if False, compressible
        formulation following Nolan et al. (2014) [5].
    dim : int, default=3
        Spatial dimensions (always 3 for 3D continuum mechanics).

    Attributes
    ----------
    dict_psi : dict
        Symbolic strain energy components: 'iso', 'vol', 'ani', 'total'.
    dict_pk1 : dict
        First Piola-Kirchhoff stress tensor components.
    dict_hessian : dict
        Tangent stiffness (elasticity tensor) components.
    mat_vars : list of sympy.Symbol
        Material parameter symbols: [μ, k₁, k₂, α, ...].
    primal_vars : list of sympy.Symbol
        Kinematic variables: [lₓ, l_y, l_z, ...].

    Notes
    -----
    - For nearly incompressible materials (ν ≈ 0.5), use mix=2 or mix=3
    - The dispersion parameter κ typically ranges from 0.1-0.25 for arteries
    - Stabilization parameters are crucial for convergence (see docs/algorithms/stabilization.md)
    - The symbolic formulation is built at initialization and cached for performance

    See Also
    --------
    ExtensionSolution : Solves the nonlinear equilibrium equations
    AnisoMaterialFit : Parameter identification framework

    Examples
    --------
    >>> # Standard two-fiber family HGO model with dispersion
    >>> var_form = VariationalFormulation(
    ...     ds=1.5,              # Cross-sectional area [mm²]
    ...     itype='nh',          # Neo-Hookean ground substance
    ...     mix=2,               # u-p formulation
    ...     kappa=True,          # Enable fiber dispersion
    ...     dvol=True,           # Fit bulk modulus
    ...     bulk=56.67e-3,       # Initial bulk modulus [MPa]
    ...     was=True             # Incompressible fibers
    ... )
    >>> # Access strain energy
    >>> psi_total = var_form.dict_psi['total']
    >>> # Access material parameters
    >>> mu, k1, k2, alpha = var_form.mat_vars[:4]
    """
    __name__ = 'VariationalFormulation'

    def __init__(self,
                 ds: float,
                 itype: str,
                 mix: int,
                 kappa: bool,
                 dvol: bool,
                 bulk: float = None,
                 iso_split: bool = False,
                 vol_type: str = 'simo92',
                 hv: bool = False,
                 was: bool = False,
                 dim: int = 3,
                 simplify_tensors: bool = False,
                 simplify_timeout: int = 10,
                 **kwargs,
                 ):
        """
        Initialize the VariationalFormulation.

        Performance Parameters
        ----------------------
        :param simplify_tensors: Enable symbolic simplification of tensor expressions.
                                 When False (default), expressions remain unsimplified but
                                 initialization is ~15x faster. Enable for production use
                                 when simplified expressions are needed.
        :param simplify_timeout: Maximum time (seconds) for each simplification attempt.
                                Lower values speed up initialization but may leave complex
                                expressions unsimplified. Default: 10.

        Example - Fast initialization for testing:
            >>> vf = VariationalFormulation(ds=0.5, itype='nh', mix=1, kappa=False, 
            ...                            dvol=False, bulk=1.0)

        Example - Thorough initialization for production:
            >>> vf = VariationalFormulation(ds=0.5, itype='nh', mix=1, kappa=False,
            ...                            dvol=False, bulk=1.0, simplify_tensors=True,
            ...                            simplify_timeout=10)
        """

        if iso_split and not was:
            import warnings
            warnings.warn(
                "Using `iso_split=True` with `was=False` is not recommended for "
                "compressible materials as it may lead to unphysical results. "
                "See Nolan et al. (2020) for details.", UserWarning)

        self.ds = ds
        self.dim = dim
        self.mix = mix
        self.itype = itype

        # Volume Strain Variables
        if self.mix == 2:
            self._vol_type = 'Sussman'
        else:
            self._vol_type = vol_type

        self.iso_split = iso_split
        self._vol_stabilization = not self.iso_split
        self._vol_flg = dvol

        if bulk is None:
            self._bulk = bulk_val
        else:
            self._bulk = bulk

        # Anisotropic Strain Variables
        self._hv = hv
        self._was = was
        self._kappa_flg = kappa

        # Performance parameters
        self._simplify_tensors = simplify_tensors
        self._simplify_timeout = simplify_timeout

        self.tensor_manager = TensorManager(
            simplify_concrete_expressions=simplify_tensors,
            simplify_intermediate_steps=False,  # Never simplify intermediate steps
            simplify_timeout=simplify_timeout,
        )

        # Variables Initialization
        self._init_kinematic_vars()
        self._init_material_vars()

        # Strain Energy Variations
        self._init_strain_energy()

        # Calculate the derivatives to solve PDEs
        self._compute_derivatives()

        # Calculate the cost function variables
        self._compute_cost_functional()

        # Calculate the adjoint variables
        self._compute_adjoint_variable()

        # initialize block variables, like jacobian and hessian blocks
        self._initialize_blocks()

    def _init_kinematic_vars(self):
        self.lx, self.ly, self.lz = sy.symbols('l_x l_y l_z', real=True)
        self.primal_vars = [self.lx, self.ly, self.lz]
        self.dict_primal_vars = dict(u=sy.Array(self.primal_vars))
        self.ar_def_grad = self.dict_primal_vars['u']
        self.mtx_def_grad = sy.Matrix([[self.lx, 0, 0], [0, self.ly, 0], [0, 0, self.lz]])

        if self.mix == 2:
            self.p = sy.symbols('p', real=True)
            self.primal_vars.append(self.p)
            self.dict_primal_vars['p'] = sy.Array([self.p])

        elif self.mix == 3:
            self.p = sy.symbols('p', real=True)
            self.t = sy.symbols('theta', real=True)
            self.primal_vars.extend([self.p, self.t])
            self.dict_primal_vars['p'] = sy.Array([self.p])
            self.dict_primal_vars['t'] = sy.Array([self.t])

        # Add more variables/ tensors and invariants
        def_grad = sy.MatrixSymbol("F", self.dim, self.dim)
        right_cauchy = sy.MatrixSymbol("C", self.dim, self.dim)
        iv_1, jr = sy.symbols("Iv_1 j^r", positive=True)

        rcauchy = right_cauchy_fun(def_grad)
        jacobian = sy.Determinant(def_grad)
        iv1 = sy.Trace(rcauchy)

        self.tensor_manager.add("F", def_grad, self.mtx_def_grad)
        self.tensor_manager.add("C", right_cauchy, rcauchy)
        self.tensor_manager.add("Iv_1", iv_1, iv1)
        self.tensor_manager.add("J", jr, jacobian)

    def _init_material_vars(self):

        # Isotropic Material Variables
        if self.itype in {'nh', NEO_HOOKEAN}:
            mu, lbd = sy.symbols(r'mu \lambda', positive=True)
            self.mat_iso = [mu]
            self.dict_mat_vars = dict(mu=mu, lambda_=lbd)

        elif self.itype == FUNG:
            a_f, b_f = sy.symbols('a_f b_f', positive=True)
            self.mat_iso = [a_f, b_f]
            self.dict_mat_vars = dict(a_f=a_f, b_f=b_f)

        else:
            raise ValueError(f"This isotropic material type is not implemented: {self.itype}")

        # Isotropic Volumetric Material Variables
        dbk = sy.symbols('D', positive=True)

        if self.vol_flg:
            self.mat_iso += [dbk]
            self.dict_mat_vars["D"] = dbk

        # Anisotropic Material Variables
        alpha = sy.symbols('alpha', positive=True)
        self.mat_ani = [alpha]
        self.dict_mat_vars["alpha"] = alpha

        # If kappa is enabled, we include the dispersion parameter
        if self._kappa_flg:
            ka = sy.symbols('kappa', positive=True)
            self.mat_ani.append(ka)
            self.dict_mat_vars["kappa"] = ka

        k_1, k_2 = sy.symbols('k_1 k_2', positive=True)
        self.mat_ani.extend([k_1, k_2])
        self.dict_mat_vars["k_1"] = k_1
        self.dict_mat_vars["k_2"] = k_2

        # Material Variables
        self.mat_vars = self.mat_iso + self.mat_ani

    def _init_strain_energy(self):

        # Isotropic (Isochoric) Strain Energy
        _ref_iso = ["Iv_1", "C", "J", "F"]
        _psi_iso = isotropic_strain_energy(self.itype, self.iso_split, self.dict_mat_vars, self.tensor_manager)
        self._psi_iso = _subs_and_eval(_psi_iso, _ref_iso, self.tensor_manager)

        # Isotropic (Volumetric) Strain Energy
        _ref_vol = ["Iv_1", "C", "J", "F"]
        _psi_vol = volumetric_strain_energy(self._vol_type, self.tensor_manager)
        psi_vol = _subs_and_eval(_psi_vol, _ref_vol, self.tensor_manager)

        if self._vol_flg:
            psi_vol_penal = self.dict_mat_vars["D"] * psi_vol
        else:
            psi_vol_penal = self._bulk * psi_vol

        self._psi_vol = psi_vol_penal

        # Anisotropic Strain Energy
        _ref_ani = ["Iv_4", "Iv_6", "M^r_1", "M^r_2", "C", "J", "F"]
        list_psi_ani, list_iv4 = anisotropic_strain_energy(self.dim, self._was, self.dict_mat_vars,
                                                           self.tensor_manager, hv=self._hv)
        _psi_ani = sum(list_psi_ani)
        self._psi_ani = _subs_and_eval(_psi_ani, _ref_ani, self.tensor_manager)

        self.list_iv4 = []
        for i, iv4_i in enumerate(list_iv4):
            self.list_iv4.append(_subs_and_eval(iv4_i, _ref_ani, self.tensor_manager))

        # Total Strain Energy Functional
        _psi_total, self._psi_fields = mixed_strain_energy_functional(
            mix=self.mix,
            primal_vars=self.dict_primal_vars,
            mat_vars=self.dict_mat_vars,
            bulk=self._bulk,
            vol_flg=self._vol_flg,
            tensor_manager=self.tensor_manager,
            psi_iso=self._psi_iso,
            psi_vol=psi_vol,
            psi_ani=self._psi_ani,
            subs=False,
        )

        self._psi_total = _subs_and_eval(_psi_total, ["J", "F"], self.tensor_manager)

        # strain energy - [Components - Deformation + Material Behavior]
        self.dict_psi = {key: self._psi_total.diff(value) for key, value in self._psi_fields.items()}
        self.dict_psi['total'] = self._psi_total.subs(dict.fromkeys(self._psi_fields.values(), 1.))

        self.dict_psi_sum = {key: self.ds * self._psi_total.diff(value) for key, value in self._psi_fields.items()}
        self.dict_psi_sum['total'] = self.ds * self.dict_psi['total']
        self.psi_sum = self.ds * self.dict_psi['total']

    def _compute_derivatives(self):

        # #######################################################################################3
        # First Variation (PK1)
        self.dict_pk1 = {key: sy.derive_by_array(self.dict_psi[key], self.ar_def_grad)
                         for key, value in self._psi_fields.items()}

        # derivative by kinematic variables: u, p, t
        self.dict_pk1.update({key: sy.derive_by_array(self.dict_psi['total'], value)
                              for key, value in self.dict_primal_vars.items()})

        # derivative by kinematic variables: u
        self.dict_pk1['total'] = sy.derive_by_array(self.dict_psi['total'], self.ar_def_grad)
        self.dict_pk1['full'] = sy.derive_by_array(self.dict_psi['total'], self.primal_vars)

        self.dict_residuum = {key_i: self.ds * pk1_i for key_i, pk1_i in self.dict_pk1.items()}

        self.residuum = self.dict_residuum['full']

        # #######################################################################################3
        # Second Variation
        self.dict_hessian = {key: sy.hessian(self.dict_psi[key], self.ar_def_grad)
                             for key, value in self._psi_fields.items()}

        self.dict_hessian.update({key: sy.hessian(self.dict_psi['total'], value)
                                  for key, value in self.dict_primal_vars.items()})

        self.dict_hessian['total'] = sy.hessian(self.dict_psi['total'], self.ar_def_grad)
        self.dict_hessian['full'] = sy.hessian(self.dict_psi['total'], self.primal_vars)

        # self.hessian = self.ds * self.dict_hessian['u'][1:, 1:]

        # #######################################################################################3
        # Cross Variation
        self.fint_mat_diff = sy.derive_by_array(self.residuum[0], self.mat_vars)
        self.energy_mat_diff = sy.derive_by_array(self.ds * self.dict_psi['total'], self.mat_vars)

    def _compute_cost_functional(self):

        self.fx = sy.symbols('f_x', real=True)
        self.fx_diff = self.residuum[0] - self.fx

        self.lsq_fun = 0.5 * self.fx_diff ** 2
        self.lsq_fun_diff = sy.derive_by_array(self.lsq_fun, self.mat_vars)

    def _compute_adjoint_variable(self):

        # Material derivatives (Adjoint Method)
        self.fint_x = self.residuum[0]

        self.dfint_x_du = sy.derive_by_array(self.fint_x, self.primal_vars)
        self.dfint_x_dm = sy.derive_by_array(self.fint_x, self.mat_vars)

        # Lambdify the Derivatives of residuals with respect to material parameters
        self.dR_du = self.ds * self.dict_hessian['u'][:, 1:]

        self.dR_du_full = self._build_full_hessian().transpose()
        
        # Material derivatives - conditionally simplify
        sym_dR_dm = self.ds * sy.derive_by_array(self.dict_pk1['full'], self.mat_vars).transpose()
        if self._simplify_tensors:
            sym_dR_dm = safe_simplify(sym_dR_dm, timeout=self._simplify_timeout)
        self.dR_dm = sym_dR_dm

        # Create the Adjoint Derivative of the volumetric strain energy functional
        jr = sy.Determinant(self.tensor_manager.get_expression_by_index("F")).doit()

        if self.mix == 1 or self.mix == 3:
            self.Jvol = volumetric_strain('simo92', jr)
        elif self.mix == 2:
            self.Jvol = 0.5 * self.p ** 2.

        if self._vol_flg:
            self.dJvol_du = sy.derive_by_array(self.Jvol, self.primal_vars)
            self.dJvol_dm = sy.derive_by_array(self.Jvol, self.mat_vars)
        else:
            self.dJvol_du = sy.Array(np.zeros(len(self.primal_vars)))
            self.dJvol_dm = sy.Array(np.zeros(len(self.mat_vars)))

        self.dict_pk1_diff_x = {}
        for key_k in ['iso', 'vol', 'ani']:
            sym_pk1 = self.dict_pk1[key_k][0]  # Assuming dof=0

            # Compute derivative with respect to material variables
            self.dict_pk1_diff_x[key_k] = sy.derive_by_array(sym_pk1, self.mat_vars)

        # Derivatives of anisotropic invariants
        self.aniso_inv_derivatives = {
            'd_iva_du': sy.Transpose(sy.derive_by_array(self.list_iv4, self.dict_primal_vars['u'])),
            'd_iva_dm': sy.Transpose(sy.derive_by_array(self.list_iv4, self.mat_vars)),
        }

    def _initialize_blocks(self):
        """
        Initializes the blocks for mixed formulations.

        Article:
            On some mixed finite element methods for incompressible and nearly incompressible finite elasticity
        """

        # FIXME: fix the block variables definition
        self.block_hessian, self.lbdf_block_hessian, self.hessian_block_shapes = [], [], []

        if self.mix in [1, 2, 3]:
            self.primal_block = [self.primal_vars[1:]]
            
            # Jacobian - conditionally simplify
            sym_jacobian = self.ds * self.dict_pk1['full'][1:]
            if self._simplify_tensors:
                sym_jacobian = safe_simplify(sym_jacobian, timeout=self._simplify_timeout)
            self.jacobian = sym_jacobian
            self.block_jacobian = [self.jacobian]

            self.hessian = self._build_full_hessian()[1:, 1:]
            self.block_hessian.append([self.hessian])
            self.lbdf_block_hessian.append(['lambdify'])

        else:
            raise NotImplementedError(f"This mixed variational form is not yet supported: {self.mix}")

    def _build_full_hessian(self):
        
        # Hessian - conditionally simplify
        sym_hessian = self.ds * self.dict_hessian['full']
        if self._simplify_tensors:
            sym_hessian = safe_simplify(sym_hessian, timeout=self._simplify_timeout)
        
        if self.mix == 3:
            if self.dict_mat_vars.get('D') is not None:
                stab = 1.e-6 * self.ds / self.dict_mat_vars['D']
            else:
                stab = 1.e-6 * self.ds / self._bulk
            sym_hessian[3, 3] = stab

        elif self.mix not in [1, 2]:
            raise NotImplementedError(f"This mixed variational form is not yet supported: {self.mix}")

        return sym_hessian

    def latex_post(self) -> List[str]:
        list_latex_eqs = []
        for index_i, symbol_i, expression_i in self.tensor_manager:
            latex_sym_i = latex(symbol_i)
            latex_expr_i = latex(expression_i)
            latex_eqs_i = "\\begin{equation}\\n" + latex_sym_i + " = " + latex_expr_i + "\\n" + "\\end{equation}\\n"
            list_latex_eqs.append(latex_eqs_i)

        list_latex_eqs.append(" \\n")
        list_latex_eqs.append(" Total strain energy density function:")
        list_latex_eqs.append("\\n \\begin{dmath}\\n" + "\\psi = " + latex(self._psi_total) + "\\n \\end{dmath}\\n")
        list_latex_eqs.append(" \\n")
        list_latex_eqs.append(" Isotropic (Shear) strain energy density function:")
        list_latex_eqs.append("\\n \\begin{dmath}\\n" + "{\\psi}_{iso} = " + latex(self._psi_iso) + "\\n \\end{dmath}\\n")
        list_latex_eqs.append(" \\n")
        list_latex_eqs.append(" Isotropic (Volumetric) strain energy density function:")
        list_latex_eqs.append("\\n \\begin{dmath}\\n" + "{\\psi}_{vol} = " + latex(self._psi_vol) + "\\n \\end{dmath}\\n")
        list_latex_eqs.append(" \\n")
        list_latex_eqs.append(" Isotropic (Anisotropic) strain energy density function:")
        list_latex_eqs.append("\\n \\begin{dmath}\\n" + "{\\psi}_{ani} = " + latex(self._psi_ani) + "\\n \\end{dmath}\\n")

        return list_latex_eqs

    @property
    def vol_flg(self):
        return self._vol_flg

    def print_configuration(self, verbose: bool = False) -> None:
        """
        Print the internal configuration of the VariationalFormulation.
        
        This method displays all key settings, material parameters, and computed
        quantities to aid in verification and debugging. Output is sent to the
        logger at INFO level.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, includes additional details like tensor expressions and 
            symbolic derivatives. Default is False.
            
        Examples
        --------
        >>> var_form = VariationalFormulation(...)
        >>> var_form.print_configuration()
        >>> var_form.print_configuration(verbose=True)
        """
        
        lines = []
        lines.append("")
        lines.append("=" * 80)
        lines.append("VARIATIONAL FORMULATION CONFIGURATION")
        lines.append("=" * 80)
        lines.append("")
        
        # --- Basic Settings ---
        lines.append("─" * 80)
        lines.append("BASIC SETTINGS (Initialization)")
        lines.append("─" * 80)
        lines.append(f"  Material Type (itype):            {self.itype}")
        lines.append(f"  Mix Formulation (mix):            {self.mix}")
        lines.append(f"  Cross-sectional Area (ds):        {self.ds:.6f}")
        lines.append(f"  Bulk Modulus (bulk):              {self._bulk:.6f} MPa")
        lines.append(f"  Isochoric Split (iso_split):      {self.iso_split}")
        lines.append(f"  Modified Anisotropic (WAS):       {self._was}")
        lines.append(f"  Volumetric Flag (vol_flg):        {self._vol_flg}")
        lines.append(f"  Volumetric Function (vol_type):   {self._vol_type}")
        lines.append(f"  Kappa (fiber dispersion):         {self._kappa_flg}")
        lines.append(f"  Heaviside Function (fiber):       {self._hv}")
        lines.append(f"  Simplify Tensors Flag:            {self._simplify_tensors}")
        if self._simplify_tensors:
            lines.append(f"  Simplification Timeout:           {self._simplify_timeout}s")
        lines.append("")
        
        # --- Material Parameters ---
        lines.append("─" * 80)
        lines.append("MATERIAL PARAMETERS")
        lines.append("─" * 80)
        lines.append(f"  Number of parameters:             {len(self.mat_vars)}")
        lines.append(f"  Parameter names:                  {list(self.dict_mat_vars.keys())}")
        lines.append("")
        for name, symbol in self.dict_mat_vars.items():
            lines.append(f"    {name:10s} = {symbol}")
        lines.append("")
        
        # --- Primal Variables ---
        lines.append("─" * 80)
        lines.append("PRIMAL VARIABLES")
        lines.append("─" * 80)
        lines.append(f"  Number of DOFs:                   {len(self.primal_vars)}")
        lines.append(f"  Variable names:                   {list(self.dict_primal_vars.keys())}")
        lines.append("")
        if verbose:
            for name, symbol in self.dict_primal_vars.items():
                lines.append(f"    {name:10s} = {symbol}")
            lines.append("")
        
        # --- Fiber Configuration ---
        lines.append("─" * 80)
        lines.append("FIBER CONFIGURATION")
        lines.append("─" * 80)
        
        num_families = len(self.list_iv4) if hasattr(self, 'list_iv4') and self.list_iv4 else 0
        lines.append(f"  Number of fiber families:         {num_families}")
        
        if hasattr(self, 'dict_mat_vars') and 'alpha' in self.dict_mat_vars:
            lines.append(f"  Fiber angles (alpha):             {self.dict_mat_vars['alpha']}")
        else:
            lines.append(f"  Fiber angles (alpha):             Not defined")
        
        if hasattr(self, 'list_iv4') and self.list_iv4:
            lines.append(f"  Anisotropic invariants (I4):      Defined ({len(self.list_iv4)} families)")
            if verbose:
                for i, inv in enumerate(self.list_iv4):
                    lines.append(f"    I4_{i} = {inv}")
        else:
            lines.append(f"  Anisotropic invariants (I4):      Not defined")
        lines.append("")
        
        # --- Strain Energy Components ---
        lines.append("─" * 80)
        lines.append("STRAIN ENERGY COMPONENTS")
        lines.append("─" * 80)
        lines.append(f"  Isotropic (ψ_iso):                {'Defined' if hasattr(self, '_psi_iso') else 'Not defined'}")
        lines.append(f"  Volumetric (ψ_vol):               {'Defined' if hasattr(self, '_psi_vol') else 'Not defined'}")
        lines.append(f"  Anisotropic (ψ_ani):              {'Defined' if hasattr(self, '_psi_ani') else 'Not defined'}")
        lines.append(f"  Total (ψ_total):                  {'Defined' if hasattr(self, '_psi_total') else 'Not defined'}")
        
        if verbose and hasattr(self, '_psi_total'):
            lines.append("")
            lines.append("  Total Strain Energy Expression:")
            lines.append(f"    ψ_total = {self._psi_total}")
        lines.append("")
        
        # --- Tensor Manager ---
        lines.append("─" * 80)
        lines.append("TENSOR MANAGER")
        lines.append("─" * 80)
        if hasattr(self, 'tensor_manager'):
            tensor_count = len(list(self.tensor_manager))
            lines.append(f"  Number of tensors:                {tensor_count}")
            tensor_names = [name for _, name, _ in self.tensor_manager]
            tensor_list = ", ".join([str(t) for t in tensor_names[:5]])
            if len(tensor_names) > 5:
                tensor_list += f" ... (+{len(tensor_names)-5} more)"
            lines.append(f"  Registered tensors:               {tensor_list}")
        else:
            lines.append(f"  Tensor Manager:                   Not initialized")
        lines.append("")
        
        # --- Computed Quantities ---
        lines.append("─" * 80)
        lines.append("COMPUTED QUANTITIES")
        lines.append("─" * 80)
        
        if hasattr(self, 'dict_pk1'):
            pk1_keys = list(self.dict_pk1.keys())
            lines.append(f"  PK1 Stress components:            {', '.join(pk1_keys)}")
            if verbose:
                for key in pk1_keys:
                    shape = self.dict_pk1[key].shape if hasattr(self.dict_pk1[key], 'shape') else 'scalar'
                    lines.append(f"    {key:10s}: shape = {shape}")
        
        if hasattr(self, 'fint_x'):
            try:
                shape = self.fint_x.shape
                lines.append(f"  Internal Force (fint_x):          Defined, shape = {shape}")
            except AttributeError:
                lines.append(f"  Internal Force (fint_x):          Defined (symbolic)")
        
        if hasattr(self, 'dict_hessian'):
            hessian_keys = list(self.dict_hessian.keys())
            lines.append(f"  Hessian components:               {', '.join(hessian_keys)}")
            if verbose and 'full' in self.dict_hessian:
                try:
                    lines.append(f"    Full Hessian shape:             {self.dict_hessian['full'].shape}")
                except AttributeError:
                    lines.append(f"    Full Hessian:                   Symbolic expression")
        
        if hasattr(self, 'jacobian'):
            try:
                shape = self.jacobian.shape
                lines.append(f"  Jacobian (mixed form):            Defined, shape = {shape}")
            except AttributeError:
                lines.append(f"  Jacobian (mixed form):            Defined (symbolic)")
        
        if hasattr(self, 'dR_dm'):
            try:
                shape = self.dR_dm.shape
                lines.append(f"  Material derivatives (dR_dm):     Defined, shape = {shape}")
            except AttributeError:
                lines.append(f"  Material derivatives (dR_dm):     Defined (symbolic)")
        
        if hasattr(self, 'dJvol_du'):
            lines.append(f"  Volumetric derivatives:           Defined")
        
        if hasattr(self, 'list_iv4') and self.list_iv4:
            lines.append(f"  Anisotropic invariants:           {len(self.list_iv4)} computed")
        
        lines.append("")
        
        # --- Mixed Formulation Details ---
        if self.mix in [1, 2, 3]:
            lines.append("─" * 80)
            lines.append("VARIATIONAL FORMULATION DETAILS")
            lines.append("─" * 80)
            lines.append(f"  Mix Type:                         {self.mix}")
            if self.mix == 1:
                lines.append(f"  Description:                      u formulation (displacement)")
            elif self.mix == 2:
                lines.append(f"  Description:                      u-p formulation (displacement-pressure)")
            elif self.mix == 3:
                lines.append(f"  Description:                      u-p-t formulation (displacement-pressure-dilatation)")
            
            if hasattr(self, 'primal_block'):
                lines.append(f"  Primal blocks:                    {len(self.primal_block)}")
            if hasattr(self, 'block_hessian'):
                lines.append(f"  Hessian blocks:                   {len(self.block_hessian)}")
            if hasattr(self, 'block_jacobian'):
                lines.append(f"  Jacobian blocks:                  {len(self.block_jacobian)}")
            lines.append("")
        
        # --- Summary ---
        lines.append("─" * 80)
        lines.append("SUMMARY")
        lines.append("─" * 80)
        
        num_families = len(self.list_iv4) if hasattr(self, 'list_iv4') and self.list_iv4 else 0
        
        lines.append(f"  ✓ Configuration complete")
        lines.append(f"  ✓ Material parameters: {len(self.mat_vars)}")
        lines.append(f"  ✓ DOFs: {len(self.primal_vars)}")
        lines.append(f"  ✓ Fiber families: {num_families}")
        lines.append(f"  ✓ Formulation: {'Mixed' if self.mix in [1,2,3] else 'Standard'}")
        lines.append("=" * 80)
        lines.append("")
        
        # Output all lines via logger
        for line in lines:
            logger.info(line)
