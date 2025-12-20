# -*- coding: utf-8 -*-
"""
Extension solution for uniaxial material testing.

This module provides the ExtensionSolution class for solving the
equilibrium equations of a uniaxial extension test and computing
state variables and their sensitivities.
"""
import warnings
import numpy as np
# import jax.numpy as jnp

from typing import Union, List, Callable, Tuple

import pandas as pd
from scipy.optimize import OptimizeResult
from dualmatfit.solution import Root
from dualmatfit.variational_form import VariationalFormulation
from dualmatfit.lambdify_builder import LambdifyBuilder
from dualmatfit.utils import check_dsvars
from dualmatfit.numeric_utils import sanitize_array, has_nan

__all__ = [
    'ExtensionSolution',
    'DesignVariablesMixin',
]


def _normalize_primal_input(
    xi: Union[list, np.ndarray],
    np_primal: np.ndarray,
) -> np.ndarray:
    """
    Normalize xi input and align with primal variable shape.
    
    Parameters
    ----------
    xi : list or np.ndarray
        Input design variables (may be list of arrays or single array).
    np_primal : np.ndarray
        Reference primal variable array for shape validation.
        
    Returns
    -------
    np.ndarray
        Primal array with xi values properly assigned.
        
    Raises
    ------
    TypeError
        If xi is not a list or ndarray.
    ValueError
        If shape mismatch cannot be reconciled.
    """
    xi_primal = np_primal.copy()
    
    if isinstance(xi, list):
        xi_in = (np.concatenate([item.flatten() for item in xi]) 
                 if all(isinstance(item, np.ndarray) for item in xi) 
                 else np.array(xi))
    elif isinstance(xi, np.ndarray):
        xi_in = xi.copy()
    else:
        raise TypeError(f"Unsupported type for xi: {type(xi)}")
    
    if xi_primal.shape[0] == xi_in.shape[0]:
        xi_primal[:] = xi_in
    elif xi_primal.shape[0] - 1 == xi_in.shape[0]:
        xi_primal[1:] = xi_in
    else:
        raise ValueError(
            f"Shape mismatch: primal has {xi_primal.shape[0]} elements, "
            f"xi has {xi_in.shape[0]} elements."
        )
    
    return xi_primal


class DesignVariablesMixin:
    """
    Mixin class providing common design variables initialization logic.
    
    This mixin encapsulates the repeated pattern of validating and initializing
    design variables from a DataFrame/Series against material variable keys
    from a LambdifyBuilder instance.
    
    Classes using this mixin must have:
        - self.lbdf_builder: LambdifyBuilder instance with inp_material_keys attribute
        - self.var_form: VariationalFormulation instance
    
    After calling _init_design_variables(), the following attributes are set:
        - self.dsvars: DataFrame/Series of design variables (reindexed)
        - self.inp_mat_keys: List of material variable keys
        - self.run_dsvars: Boolean array indicating active variables
        - self.nvars: Number of active design variables
        - self.xi: Array of active design variable values
        - self.xi_ref: Array of all design variable values (reference)
        - self.xi_bounds: List of [lower, upper] bounds for active variables
    """
    
    def _init_design_variables(
        self,
        dsvars: Union[pd.DataFrame, pd.Series],
    ) -> None:
        """
        Initialize design variables from a DataFrame or Series.
        
        This method validates the design variables against the material variables
        defined in the variational formulation and sets up the optimization-related
        attributes.
        
        Parameters
        ----------
        dsvars : pd.DataFrame or pd.Series
            Design variables table with index matching material variable names.
            If DataFrame, must have columns: 'variable', 'values', 'lower', 'upper'.
            If Series, used for simpler access patterns.
            
        Raises
        ------
        ValueError
            If the number of design variable keys doesn't match the expected
            material variable keys from the LambdifyBuilder.
        TypeError
            If dsvars is neither a DataFrame nor a Series.
        """
        mat_vars_keys, dsvars = check_dsvars(self.var_form, dsvars)
        inp_mat_keys = self.lbdf_builder.inp_material_keys
        
        if len(mat_vars_keys) != len(inp_mat_keys):
            raise ValueError(
                f"Mismatch in material variable keys length: "
                f"got {len(mat_vars_keys)}, expected {len(inp_mat_keys)}"
            )
        
        if isinstance(dsvars, pd.DataFrame):
            self.dsvars = dsvars.loc[inp_mat_keys, :]
        elif isinstance(dsvars, pd.Series):
            self.dsvars = dsvars[inp_mat_keys]
        else:
            raise TypeError("dsvars must be a pandas DataFrame or Series.")
        
        self.inp_mat_keys = mat_vars_keys
        
        # Extract optimization-related arrays
        self.run_dsvars = self.dsvars["variable"].values.astype(bool)
        self.nvars = np.count_nonzero(self.run_dsvars)
        
        self.xi = self.dsvars["values"][self.run_dsvars].values.astype(float)
        self.xi_ref = self.dsvars["values"].values.astype(float)
        self.xi_bounds = dsvars[self.run_dsvars][["lower", "upper"]].values.tolist()

    def _update_design_variables(self, xi: np.ndarray) -> pd.Series:
        """
        Update design variables with new values.
        
        Parameters
        ----------
        xi : np.ndarray
            New values for active design variables.
            
        Returns
        -------
        pd.Series
            Updated material parameter series with xi values applied.
        """
        self.xi_ref[self.run_dsvars] = xi
        sr_xi_mat = self.dsvars["values"].copy().astype(float)
        sr_xi_mat[self.run_dsvars] = xi
        return sr_xi_mat


class NumericalProblem:
    """
    Encapsulates the numerical problem for the nonlinear solver.

    This class provides the functions required by the nonlinear solver, namely
    the function to compute the residual of the equilibrium equations (the
    Jacobian of the strain energy) and the function to compute the Jacobian
    of the residual (the Hessian of the strain energy). It uses the fast
    numerical functions generated by `LambdifyBuilder`.
    """

    def __init__(self,
                 lambdify_builder: LambdifyBuilder,
                 var_form: VariationalFormulation,
                 np_primal: np.ndarray,
                 np_mat_props: np.ndarray,
                 ):
        """
        Initializes the NumericalProblem.

        Args:
            lambdify_builder: An instance of the LambdifyBuilder.
            var_form:       An instance of the variational formulation.
            np_primal:      NumPy array of primal variables.
            np_mat_props:   NumPy array of material properties.
        """
        self.np_primal = np_primal
        self.np_mat_props = np_mat_props

        self.var_form = var_form
        self.builder = lambdify_builder

    def build_jacobian(self,
                       xi: np.ndarray,
                       xi_mat: np.ndarray = None,
                       **kwargs,
                       ):
        """Builds the Jacobian matrix."""
        xi_primal_in = _normalize_primal_input(xi, self.np_primal)
        xi_mat = xi_mat if xi_mat is not None else self.np_mat_props

        if self.var_form.mix in [1, 2, 3]:
            jac_ou = sanitize_array(self.builder.block_jacobian[0](*xi_primal_in, *xi_mat),
                                   nan=1.e9, posinf=1.e16, neginf=-1.e16)
        else:
            raise NotImplementedError(f"Not implemented for mix type {self.var_form.mix}")

        return sanitize_array(jac_ou)

    def build_hessian(self,
                      xi: np.ndarray,
                      xi_mat: np.ndarray = None,
                      **kwargs,
                      ) -> Union[np.ndarray, list]:
        """Builds the Hessian matrix."""
        xi_primal_in = _normalize_primal_input(xi, self.np_primal)
        xi_mat = xi_mat if xi_mat is not None else self.np_mat_props

        if self.var_form.mix in [1, 2, 3]:
            hessian = np.asarray(self.builder.block_hessian[0][0](*xi_primal_in, *xi_mat), dtype=float)
            if has_nan(hessian) and np.all(np.isnan(hessian)):
                warnings.warn("NaN values encountered in Hessian calculation.", RuntimeWarning)
        else:
            raise NotImplementedError(f"Not implemented for mix type {self.var_form.mix}")

        return sanitize_array(hessian)


def _get_initial_guess(lx: float | np.ndarray,
                       mix: int,
                       incompressible: bool = False,
                       ) -> np.ndarray:
    """
    Provides an improved initial guess based on the stretch value and mix formulation.

    Args:
        lx: Current axial stretch value

    Returns:
        Initial guess array for primal variables (excluding lx)
    """
    if np.isclose(lx, 1.0, atol=1e-6):
        # No motion case: all stretches should be 1, pressure and theta have specific values
        if mix == 1:
            # ly, lz
            return np.array([1.0, 1.0], dtype=float)

        elif mix == 2:
            # ly, lz, p
            return np.array([1.0, 1.0, 1e-2], dtype=float)

        elif mix == 3:
            # ly, lz, p, theta
            return np.array([1.0, 1.0, 1e-2, 1.0], dtype=float)

        else:
            raise NotImplementedError(f"Mix type {mix} not implemented for initial guess.")

    elif incompressible:
        # For non-unity stretches, use incompressibility-based guess
        if mix == 1:
            # Incompressible: ly * lz = 1 / lx, assume ly = lz
            ly_lz_guess = 1.0 / np.sqrt(lx)
            return np.array([ly_lz_guess, ly_lz_guess], dtype=float)

        elif mix == 2:
            # Slightly compressible behavior
            ly_lz_guess = 1.0 / np.sqrt(lx * 1.001)  # Small volume change
            p_guess = 0.1 * (lx - 1.0)  # Pressure proportional to deformation
            return np.array([ly_lz_guess, ly_lz_guess, p_guess], dtype=float)

        elif mix == 3:
            # Mixed formulation with theta
            ly_lz_guess = 1.0 / np.sqrt(lx * 1.001)
            p_guess = 0.1 * (lx - 1.0)
            theta_guess = lx * ly_lz_guess * ly_lz_guess  # Volume ratio
            return np.array([ly_lz_guess, ly_lz_guess, p_guess, theta_guess], dtype=float)
        else:
            raise NotImplementedError(f"Mix type {mix} not implemented for initial guess.")

    else:
        # Compressible case: use previous values as guess
        if mix == 1:
            return np.ones(2, dtype=float)
        elif mix == 2:
            return np.ones(3, dtype=float)
        elif mix == 3:
            return np.ones(4, dtype=float)
        else:
            raise NotImplementedError(f"Mix type {mix} not implemented for initial guess.")


class ExtensionSolution:
    """
    Performs an uniaxial extension test simulation for a given material model.

    This class orchestrates the solution of the nonlinear equilibrium problem
    for an uniaxial extension test. It takes a `VariationalFormulation` object,
    which defines the constitutive model, and iteratively solves for the
    transverse stretches for a prescribed axial stretch.

    The class is designed with SOLID principles in mind, decomposing the
    problem into smaller, more focused components:
    - `LambdifyBuilder`: Handles the conversion of symbolic SymPy expressions
      into fast numerical functions.
    - `NumericalProblem`: Defines the numerical problem for the solver, i.e.,
      it provides the functions to compute the residual and the Jacobian.
    - `Root`: The nonlinear solver that finds the roots of the residual equations.
    - `ResultFormatter`: Formats and aggregates the results from the solver into
      a comprehensive output object.
    """
    __name__ = 'ExtensionSolution'

    MAX_STRETCH: float = 5.
    MIN_STRETCH: float = 0.2
    STABILITY_THRESHOLD: float = 1.e-6
    MAX_INC: int = 100

    def __init__(self,
                 var_form: VariationalFormulation,
                 module: str = 'numpy',
                 solver: Callable = Root,
                 **kwargs,
                 ):
        """
        Initializes the ExtensionSolution.

        Args:
            var_form: An instance of the `VariationalFormulation` class that
                defines the material model.
            ds: The cross-sectional area of the specimen.
            module: The numerical backend for the lambdified functions
                ('numpy' or 'jax').
            solver: The solver class to be used for the nonlinear equilibrium problem.
                Defaults to `Root`.
            **kwargs: Additional keyword arguments for the solver.
        """
        self.var_form = var_form
        self.module = module
        self.solver_class = solver
        self.solver_type = kwargs.get("solver_type", "least_squares")

        self.nmat_vars = len(self.var_form.mat_vars)
        self.nprm_vars = len(self.var_form.primal_vars)
        self.np_primal = np.ones(self.nprm_vars, dtype=float)
        self.np_mat_props = np.zeros(self.nmat_vars, dtype=float)

        self._init_bounds()
        self.lbdf_builder = LambdifyBuilder(self.var_form, self.module)
        self.num_problem = NumericalProblem(self.lbdf_builder, self.var_form, self.np_primal, self.np_mat_props)

    def _init_bounds(self):
        """Initializes the bounds for primal variables."""
        # l_y, l_z
        np_bounds_lwr = self.MIN_STRETCH * np.ones(2, dtype=float)
        np_bounds_upp = self.MAX_STRETCH * np.ones(2, dtype=float)

        if self.var_form.mix == 2:
            # p
            np_bounds_lwr = np.concatenate([np_bounds_lwr, [-100.]])
            np_bounds_upp = np.concatenate([np_bounds_upp, [100.]])
        elif self.var_form.mix == 3:
            # theta
            np_bounds_lwr = np.concatenate([np_bounds_lwr, [-1., 0.1]])
            np_bounds_upp = np.concatenate([np_bounds_upp, [100., 10.]])

        self.primal_bounds = {"lower": np_bounds_lwr, "upper": np_bounds_upp}

    def _update_results(self,
                        stretch_x: np.ndarray,
                        mat_params: pd.Series,
                        results: List[OptimizeResult],
                        output_keys: List[str] = [],
                        ) -> OptimizeResult:
        """Aggregates results from multiple optimization steps."""

        list_detF = []
        list_fint = []
        list_stretch = []
        list_success = []
        list_fun = []
        list_nfev = []
        list_njev = []
        list_message = []

        dict_stress = {k: [] for k in ['iso', 'vol', 'ani', 'total', 'full']}
        dict_ese = {k: [] for k in ['iso', 'vol', 'ani', 'total']}

        for i, (lx_i, res_i) in enumerate(zip(stretch_x, results)):
            input_primal_i = np.concatenate(([lx_i], res_i.x))
            input_i = np.concatenate(([lx_i], res_i.x, mat_params.values))

            if "volume" in output_keys:
                list_detF.append(np.prod(input_i[:3]))

            if "fint" in output_keys:
                list_fint.append(self.lbdf_builder.fint(*input_i))

            if "stress" in output_keys:
                for k, lambd_k in self.lbdf_builder.dict_pk1.items():
                    dict_stress[k].append(lambd_k(*input_i))

            if "ese" in output_keys:
                for k, lambd_k in self.lbdf_builder.dict_ese.items():
                    dict_ese[k].append(lambd_k(*input_i))

            list_fun.append(res_i.fun)

            if hasattr(res_i, "nfev"):
                list_nfev.append(res_i.nfev)

            if hasattr(res_i, "njev"):
                list_njev.append(res_i.njev)

            list_success.append(res_i.success)
            list_message.append(res_i.message)
            list_stretch.append(input_primal_i)

        kwargs_result = {}

        if "volume" in output_keys:
            kwargs_result["detF"] = np.array(list_detF, dtype=float)
            kwargs_result["volume"] = np.array(list_detF, dtype=float)

        if "fint" in output_keys:
            kwargs_result["fint"] = np.array(list_fint, dtype=float)

        if "stress" in output_keys:
            kwargs_result["stress"] = {}
            for k, list_stress_k in dict_stress.items():
                kwargs_result["stress"][k] = np.array(list_stress_k, dtype=float)

        if "ese" in output_keys:
            kwargs_result["ese"] = {}
            for k, list_ese_k in dict_ese.items():
                kwargs_result["ese"][k] = np.array(list_ese_k, dtype=float)

        np_stretch = np.array(list_stretch, dtype=float)
        kwargs_result["stretch"] = np_stretch

        if len(list_nfev) > 0:
            kwargs_result["nfev"] = sum(list_nfev)

        if len(list_njev) > 0:
            kwargs_result["njev"] = sum(list_njev)

        cumulative_result = OptimizeResult(x=np_stretch,
                                           x_mat=mat_params,
                                           fun=list_fun,
                                           success=list_success,
                                           message=list_message,
                                           **kwargs_result)

        return cumulative_result

    def solve(self,
              mat_params: pd.Series,
              stretch_x: Union[float, np.ndarray],
              **kwargs,
              ) -> OptimizeResult:
        """
        Solves the nonlinear equilibrium problem for a given axial stretch.

        This method iteratively solves for the transverse stretches (ly and lz) that satisfy the equilibrium equations
        for each prescribed axial stretch (lx) in the `stretch_x` array.

        Args:
            mat_params: A NumPy array of material parameters.
            stretch_x: A scalar or 1D NumPy array of prescribed axial stretches.
            **kwargs: Additional keyword arguments for the solver. Supported options
                include 'output', 'max_iter', and 'tol'.

        Returns:
            A `scipy.optimize.OptimizeResult` object containing the solution, including the computed stretches,
            stresses, and strain energy densities.
        """
        if not isinstance(mat_params, pd.Series):
            raise TypeError("mat_params must be a Pandas Series Array.")

        if mat_params.shape[0] != self.nmat_vars:
            raise ValueError(
                f"Material parameter array size mismatch: "
                f"got {mat_params.shape[0]}, expected {self.nmat_vars}"
            )

        output = kwargs.get('output', ["stretch", "fint", "stress", "ese", "volume"])
        max_iter = kwargs.get('max_iter', 100)
        tol = kwargs.get('tol', 1e-9)

        stretch_x = np.atleast_1d(stretch_x)
        if stretch_x.ndim > 1:
            raise ValueError("stretch_x must be a scalar or a 1D array.")

        in_mat_params = mat_params[self.lbdf_builder.inp_material_keys]
        self.np_mat_props[:] = in_mat_params.values

        kwargs_root = {"bounds": self.primal_bounds,
                       "block_array": self.lbdf_builder.block_array,
                       "max_iter": max_iter,
                       "solver_type": self.solver_type,
                       "tol": tol}

        solver = self.solver_class(
            fun=self.num_problem.build_jacobian,
            jac=self.num_problem.build_hessian,
            **kwargs_root
        )

        opt_res_i = OptimizeResult()
        self.np_primal[1:] = _get_initial_guess(1., self.var_form.mix)
        list_results = []

        for i, lx_i in enumerate(stretch_x):
            self.np_primal[0] = lx_i

            if self.var_form.mix == 1:
                opt_res_i = solver.solve(x0=self.np_primal[1:])
            elif self.var_form.mix == 2:
                opt_res_i = solver.solve(x0=self.np_primal[1:], alpha=1.)
            elif self.var_form.mix == 3:
                opt_res_i = solver.solve(x0=self.np_primal[1:], alpha=0.7)

            self.np_primal[1:] = opt_res_i.x
            list_results.append(opt_res_i)

            if has_nan(self.np_primal) or not opt_res_i.success:
                warnings.warn(f"Solver did not converge at stretch step {i} (lx = {lx_i}).")

        return self._update_results(stretch_x, in_mat_params, list_results, output)
