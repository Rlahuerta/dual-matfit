# -*- coding: utf-8 -*-
"""
Material law functions for hyperelastic models.

This module provides strain energy density functions and related utilities
for Neo-Hookean, Fung, and anisotropic material models used in soft tissue
mechanics.
"""
import numpy as np
import sympy as sy

from sympy.core.function import Function, ArgumentIndexError
from typing import Union, Tuple
from dualmatfit.formulation.simplify import safe_simplify

__all__ = [
    'HeavisideFunction',
    'neo_hookean',
    'fung',
    'fung_general',
    'volumetric_strain',
    'anisotropic_invariant',
    'anisotropic_strain',
    'get_fiber_vector',
    'right_cauchy_fun',
]


class HeavisideFunction(Function):
    """
    Heaviside function class using a smooth approximation.

    Options for the Heaviside function include:
    - Logistic function: 1 / (1 + exp(-2k(x - 1)))
    - Standard Heaviside step function: Heaviside(x - 1)
    - Hyperbolic tangent approximation: 0.5 + 0.5 * tanh(k * (x - 1))

    """
    nargs = 2

    @classmethod
    def eval(cls, x, k):
        return sy.Pow(1 + sy.exp(-2. * k * (x - 1)), -1)

    def fdiff(self, argindex=1):
        """
        Computes the derivative with respect to x.

        Parameters
        ==========

        argindex : integer
            degree of derivative

        """

        if argindex == 1:
            x, k = self.args
            f_prime = self.func(x, k)
            return 2 * k * f_prime * (1- f_prime)

        else:
            raise ArgumentIndexError(self, argindex)


def heaviside(x: Union[sy.Symbol, np.ndarray, float], k: float = 150.) -> HeavisideFunction:
    """

    :param x:
    :param k:

    :return:        heaviside function
    """

    if isinstance(x, np.ndarray) and isinstance(k, float):
        return (np.power(1. + np.exp(-2. * k * (x - 1.)), -1)).astype(float)

    else:
        return HeavisideFunction(x, k)


def right_cauchy_fun(def_grad: Union[sy.Symbol, sy.Matrix, sy.Array, np.ndarray],
                     isochoric: bool = False,
                     ) -> Union[sy.Matrix, sy.Array, np.ndarray]:
    """
    Compute the right Cauchy deformation tensor.

    :param def_grad:    Deformation gradient
    :param isochoric:   Whether to compute the isochoric part of the tensor

    :return: Right Cauchy deformation tensor
    """

    if isinstance(def_grad, (sy.Matrix, sy.MatrixSymbol)):
        # If sympy Matrix
        right_cauchy = def_grad.T * def_grad

        if isochoric:
            # Volumetric-isochoric split := DISTORTIONAL COMPONENT OF THE DEFORMATION GRADIENT
            jac = def_grad.det()
            spl_def_grad = jac ** (-sy.Rational(1, 3)) * def_grad
            return spl_def_grad.T * spl_def_grad

        return right_cauchy

    elif isinstance(def_grad, np.ndarray):
        # If numpy ndarray
        right_cauchy = def_grad.T @ def_grad

        if isochoric:
            # Volumetric-isochoric split := DISTORTIONAL COMPONENT OF THE DEFORMATION GRADIENT
            jac = np.linalg.det(def_grad)

            if jac <= 0:
                raise ValueError(
                    "The determinant of the deformation gradient should be positive for isochoric deformations.")

            spl_def_grad = jac ** (-1 / 3) * def_grad
            return spl_def_grad.T @ spl_def_grad

        return right_cauchy

    else:
        raise NotImplementedError("This array type is not supported yet...")


def neo_hookean(mu: Union[sy.Basic, np.number],
                def_grad: Union[sy.Basic, sy.Matrix, sy.Array, np.number],
                isochoric: bool = False,
                volumetric: bool = True,
                ) -> Union[sy.Symbol, np.number, float]:
    """
    Neo-Hookean material model, this material exhibits characteristics that can be identified with the familiar
    material parameters found in linear elastic analysis.

    args:
    :param mu:                      Material coefficient
    :param def_grad:                Deformation Gradient
    :param isochoric:               Isochoric Split on the Deformation Gradient
    :param volumetric:              Volumetric Stabilization via adding mu * ln(J)

    :return:                        The stored energy function
    """

    dim = max(def_grad.shape)

    right_cauchy = right_cauchy_fun(def_grad, isochoric=isochoric)
    inv_rc = safe_simplify(sy.Trace(right_cauchy))

    psi = sy.Rational(1, 2) * mu * (inv_rc - dim)

    if volumetric:
        jac = sy.det(def_grad)
        psi -= mu * sy.ln(jac)

    return psi


def fung(a_f: Union[sy.Basic, float],
         b_f: Union[sy.Basic, float],
         def_grad: Union[sy.Basic, sy.Matrix, sy.Array, np.number],
         isochoric: bool = False,
         ) -> Union[sy.Symbol, sy.Expr, np.number]:
    """
    Fung-type strain-energy function
    Article:        Pseudo-elasticity of arteries and the choice of its mathematical expression.
    Article:        An orthotropic viscoelastic model for the passive myocardium: continuum basis and numerical
                treatment. (eq. 37)
    Article:        Constitutive modelling of passive myocardium: a structurally based framework for material
                characterization. (eq. 5.38)
    Article*:       A New Constitutive Framework for Arterial Wall Mechanics and a Comparative Study of Material
                Models (eq. 33)
    Article:        Residual strain effects on the stress field in a thick wall finite element model of the human
                carotid bifurcation (eq. 1) [Artigo Original]

    :param a_f:                 Material Parameters having dimension of stress
    :param b_f:                 Dimensionless material constant
    :param def_grad:            Deformation Gradient
    :param isochoric:           Isochoric Split on the Deformation Gradient

    :return:            The stored energy function
    """

    dim = max(def_grad.shape)

    right_cauchy = safe_simplify(def_grad.T * def_grad)
    # green_lag_strain = 0.5 * (right_cauchy - sy.eye(dim))
    # inv_gl = sy.trace(green_lag_strain)
    inv_c = sy.trace(right_cauchy)

    if isochoric:
        jac = sy.det(def_grad)
        spl_def_grad = sy.Pow(jac, -sy.Rational(1, 3)) * def_grad
        spl_right_cauchy = safe_simplify(spl_def_grad.T * spl_def_grad)
        spl_inv_rc = sy.trace(spl_right_cauchy)
        # spl_green_lag_strain = 0.5 * (spl_def_grad - sy.eye(dim))
        # spl_inv_gl = sy.trace(spl_green_lag_strain)

        return (a_f / b_f) * (sy.exp(sy.Rational(1, 2) * b_f * (spl_inv_rc - dim)) - 1)

    else:
        return (a_f / b_f) * (sy.exp(sy.Rational(1, 2) * b_f * (inv_c - dim)) - 1)


def fung_general(a_f: Union[sy.Basic, float],
                 q_f: Union[sy.Basic, float],
                 def_grad: Union[sy.Basic, sy.Matrix, sy.Array, np.ndarray],
                 isochoric: bool = False,
                 ) -> Union[sy.Symbol, float]:
    """
    Fung-type strain-energy general function
    Article:        Convex Fung-type potentials for biological tissues

    :param a_f:                 Material Parameters having dimension of stress
    :param q_f:                 Dimensionless material parameter
    :param def_grad:            Deformation Gradient tensor
    :param isochoric:           Isochoric Split on the Deformation Gradient

    :return:
        The stored energy function
    """

    right_cauchy = safe_simplify(def_grad.T * def_grad)
    inv_rc = sy.trace(right_cauchy)

    if isochoric:
        jac = sy.det(def_grad)
        spl_def_grad = sy.Pow(jac, -sy.Rational(1, 3)) * def_grad
        spl_right_cauchy = safe_simplify(spl_def_grad.T * spl_def_grad)
        spl_inv_rc = sy.trace(spl_right_cauchy)

        return a_f * sy.Rational(1, 2) * (sy.exp(q_f * spl_inv_rc) - 1)

    else:
        return a_f * sy.Rational(1, 2) * (sy.exp(q_f * inv_rc) - 1)


def volumetric_strain(vol_type: str, jr: Union[sy.Basic, float]) -> Union[sy.Expr, float]:
    """
    The scalar function which is a purely volumetric contribution to the stored energy function

    Article:            On the Development of Volumetric Strain Energy Functions

    Args:
        vol_type (str): Volumetric Strain Energy Type
        jr (Union[sy.Basic, float]): Jacobian: measure the volume change between the initial and current configuration

    Returns:
        Union[sy.Expr, float]: The volumetric strain energy
    """

    # Validate vol_type
    valid_vol_types = ['simo92', 'Sussman', 'bathe87', 'hencky', 'liu', 'doll8']
    if vol_type not in valid_vol_types:
        raise ValueError(f"Unsupported vol_type '{vol_type}'. Supported types are: {valid_vol_types}")

    # Validate jr
    if isinstance(jr, (int, float)):
        if jr <= 0:
            raise ValueError("The Jacobian determinant 'jr' must be positive.")
    elif isinstance(jr, sy.Basic):
        # For symbolic expressions, ensure jr is positive if possible
        # Note: Symbolic validation is limited; consider adding assumptions during symbol definition
        if jr.is_positive is not None:
            if jr.is_negative:
                raise ValueError("The Jacobian determinant 'jr' must be positive.")
    else:
        raise TypeError(f"Unsupported type for 'jr': {type(jr)}. Expected float or sympy expression.")

    if vol_type == 'simo92':
        psi_vol = 0.25 * ((jr - 1)**2 + sy.ln(jr)**2)
    elif vol_type == 'Sussman':
        # Sussman, T.; Bathe, K.-J. (1987): A finite element formulation for nonlinear incompressible elastic and
        # inelastic analysis. Comp. & Struct. 26:357-409
        psi_vol = 1. * (jr - 1.)
    elif vol_type == 'bathe87':
        psi_vol = 0.5 * (jr - 1)**2
    elif vol_type == 'hencky':
        psi_vol = 0.5 * sy.ln(jr)**2
    elif vol_type == 'liu':
        # 3D finite elem ent analysis of rubber-like materials at finite strains
        psi_vol = jr * sy.ln(jr) - jr + 1
    elif vol_type == 'doll8':
        psi_vol = 0.5 * (jr - 1) * sy.ln(jr)
    else:
        raise NotImplementedError(f"Volumetric strain type '{vol_type}' is not implemented.")

    return psi_vol


def get_fiber_vector(alpha: Union[float, np.ndarray, sy.Basic, sy.MatrixSymbol], size: int = 2, dim: int = 3, plane=(0, 1)) -> [sy.Array]:
    """
    Generate a list of fiber vectors oriented at specified angles within a given plane.

    This function creates fiber direction vectors based on an angle `alpha` within a specified
    plane in a multidimensional space. The fibers are symmetrically distributed around the
    primary axis of the plane.

    Parameters:
    ----------
    alpha : sympy.Symbol
        The angle (in radians) defining the orientation of the fibers within the specified plane.
        It should be a SymPy symbolic variable to allow for symbolic computations.

    size : int, optional (default=2)
        The number of fiber vectors to generate. Typically, an even number ensures symmetric
        distribution around the primary axis. For example:
            - `size=2` generates two fibers at +alpha and -alpha.
            - `size=4` generates four fibers at +alpha, -alpha, +alpha, -alpha (additional pairs can be implemented as needed).

    dim : int, optional (default=3)
        The dimensionality of the space. Commonly, 2D or 3D spaces are used.

    plane : tuple of two ints, optional (default=(0, 1))
        A tuple specifying the indices of the axes that define the plane in which the fibers lie.
        For example:
            - `(0, 1)` corresponds to the x-y plane.
            - `(0, 2)` corresponds to the x-z plane.
            - `(1, 2)` corresponds to the y-z plane.
        The indices should be within the range `[0, dim-1]`.

    Returns:
    -------
    List[sympy.Array]
        A list of SymPy Array objects representing the fiber vectors. Each vector is a combination
        of cosine and sine components based on the angle `alpha` within the specified plane.

    Raises:
    ------
    ValueError
        If the `plane` indices are out of bounds for the given `dim`.
    TypeError
        If the input types do not match the expected types.

    Examples:
    --------
    >>> alpha = sy.Symbol('alpha')
    >>> fibers = get_fiber_vector(alpha, size=2, dim=3, plane=(0, 1))
    >>> fibers
    [Matrix([
    [cos(alpha)],
    [sin(alpha)],
    [0]
    ]), Matrix([
    [cos(alpha)],
    [-sin(alpha)],
    [0]
    ])]
    """

    # Input Validation
    if type(alpha) in [float, np.ndarray, np.float64, sy.Basic, sy.MatrixSymbol] == False:
        raise TypeError("Parameter 'alpha' must be a SymPy Symbol.")

    if not isinstance(size, int) or size <= 0:
        raise ValueError("Parameter 'size' must be a positive integer.")

    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("Parameter 'dim' must be a positive integer.")

    if (not isinstance(plane, tuple) or len(plane) != 2 or
            not all(isinstance(axis, int) for axis in plane)):
        raise TypeError("Parameter 'plane' must be a tuple of two integers.")

    if not all(0 <= axis < dim for axis in plane):
        raise ValueError(f"Plane indices {plane} are out of bounds for dimension {dim}.")

    list_fr = []
    sy_eye = sy.Array(sy.eye(dim))
    e1 = sy_eye[:, plane[0]]
    e2 = sy_eye[:, plane[1]]

    for i in range(size):
        sign_i = (-1) ** (i + 1)
        vector_i = sy.cos(alpha) * e1 + sign_i * sy.sin(alpha) * e2
        list_fr.append(vector_i)

    return list_fr


def anisotropic_invariant(def_grad: Union[sy.Basic, np.ndarray, sy.Matrix, sy.MatrixSymbol],
                          fr: Union[sy.Basic, np.ndarray],
                          kappa: Union[sy.Symbol, float],
                          was: bool,
                          stabilization: bool = False,
                          ) -> Tuple[Union[sy.Basic, float], Union[sy.Basic, float]]:
    """
    Calculates the anisotropic invariant I4 (or I6, etc.).

    This function computes the fourth invariant (I4), which represents the square of
    the stretch in a specific fiber direction. It can also incorporate fiber dispersion
    effects through the `kappa` parameter, as described by Gasser et al. (2006).

    The `was` flag is critical as it determines whether the calculation is based on
    the full deformation gradient (for compressible materials) or the isochoric part
    (for incompressible materials). Using the isochoric invariant for a compressible
    material model can lead to unphysical results, as highlighted by Nolan et al. (2020).

    Args:
        def_grad: The deformation gradient tensor (F).
        fr: The fiber direction vector in the reference configuration.
        kappa: The fiber dispersion parameter (0 <= kappa <= 1/3).
        was: "With Anisotropic Split" flag.
             - If `True`, uses the full deformation gradient (F), calculating the
               full invariant I4. This is physically correct for compressible models.
             - If `False`, uses the isochoric part of the deformation gradient (F_bar),
               calculating the isochoric invariant I_bar_4.
        stabilization: A flag to add a small stabilization term.

    Returns:
        A tuple containing the first invariant (I1) and the fourth invariant (I4),
        either full or isochoric depending on the `was` flag.
    """

    if isinstance(fr, sy.Array):
        mr = (sy.tensorproduct(fr, fr)).tomatrix()

    elif isinstance(fr, sy.Matrix) is True or isinstance(fr, sy.MatrixSymbol) is True:
        if fr.shape[1] == 1:
            mr = fr * fr.T
        else:
            mr = fr.copy()

    else:
        raise NotImplementedError(" This type of fr vector type is not supported... ")

    dim = def_grad.shape[0]
    jac = sy.det(def_grad)

    if not was:
        spl_def_grad = sy.Pow(jac, -sy.Rational(1, 3)) * def_grad
    else:
        spl_def_grad = def_grad

    right_cauchy = safe_simplify(spl_def_grad.T * spl_def_grad)
    ani_cauchy = right_cauchy.T * mr

    if isinstance(kappa, sy.Symbol) is True or kappa > 0.:
        disp_factor = (1. - 3. * kappa)
        ani_cauchy_eqv = (3 / dim) * kappa * right_cauchy + disp_factor * ani_cauchy
    else:
        ani_cauchy_eqv = ani_cauchy

    inv_rc = safe_simplify(sy.Trace(right_cauchy))
    iv4 = safe_simplify(sy.Trace(ani_cauchy_eqv))

    # Add stabilization Parameter
    if max(def_grad.shape) == 3 and stabilization is True:
        iv4 += 1.e-9 * inv_rc

    return inv_rc, iv4


def anisotropic_strain(k1: Union[sy.Symbol, float],
                       k2: Union[sy.Symbol, float],
                       iv4: Union[sy.Symbol, float, sy.Expr],
                       hv: bool = False,
                       ) -> Union[sy.Symbol, sy.Expr, float]:
    """

    Article:    On the quasi-incompressible finite element analysis of anisotropic hyperelastic materials

    :param k1:                  stress-like material parameter
    :param k2:                  dimensionless material parameter
    :param iv4:                 Anisotropic Invariant
    :param hv:                  Heaviside Function

    :return:
    """

    psi_ani = (k1 / (2. * k2)) * (sy.exp(k2 * sy.Pow(iv4 - 1., 2)) - 1.)

    if hv:
        psi_ani = heaviside(iv4, k=32) * psi_ani

    return psi_ani
