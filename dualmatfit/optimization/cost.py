# -*- coding: utf-8 -*-
"""
Least squares fitting and cost function integration.

This module provides classes for least squares optimization (LSQFit),
cost function evaluation (CostFunction), and integrated cost computation
(CostIntegrator) for material parameter fitting.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from dualmatfit._jax_config import configure_jax

jax = configure_jax()
from jax import jacobian

from functools import lru_cache
from typing import Sequence, Callable, Union, Optional, Any
from scipy import optimize

from dualmatfit.formulation.variational import VariationalFormulation
from dualmatfit.solvers.extension import ExtensionSolution, DesignVariablesMixin
from dualmatfit.solvers.derivative import _fdm, adjoint_derivative
from dualmatfit.optimization.cache import CostCache
from dualmatfit.utils.numeric import sanitize_array, has_nan, is_finite, safe_divide
from dualmatfit.optimization.regularization import (
    L2Regularization,
    VolumeRegularization,
    CompositeRegularization,
)
from dualmatfit.optimization.loss import (lsq_fval,
                                      lsq_dfval,
                                      ln_fval,
                                      ln_dfval,
                                      logcosh_fval,
                                      logcosh_dfval,
                                      huber_fval,
                                      huber_dfval,
                                      cauchy_fval,
                                      cauchy_dfval,
                                      )

__all__ = [
    'LSQFit',
    'CostFunction',
    'CostIntegrator',
]


class LSQFit:
    def __init__(self,
                 ncontrol: int,
                 delta: float,
                 dsvars: pd.DataFrame,
                 cost_function: Callable,
                 ftype: str = 'lsq',
                 seed: int = 10,
                 ):
        """
        Initializes the LSQFit class, which models a least squares fitting problem with generated data.

        This class creates synthetic data for a given cost function based on a specified number of control points,
        adding random noise to simulate measurement errors. It also initializes the parameters required for
        evaluating the cost function and its derivative.

        Parameters:
        ----------
        ncontrol : int
            Number of control points for generating data. This determines the length of the input data array.

        delta : float
            Standard deviation of the noise added to the generated reference data, simulating measurement uncertainty.

        xi_ini : np.ndarray
            Initial values for the parameters of the cost function. These values serve as the starting
            point for fitting.

        cost_function : Callable
            The function to be fitted, which should take `x` (input data) and `xi` (parameters) as arguments
            and return the computed output. This is the target function for least squares fitting.

        ftype : str, optional, default='lsq'
            Type of least squares fitting to perform. Options include:
                - 'lsq': Basic least squares fitting (default).
                - 'ln': Logarithmic least squares fitting.

        seed : int, optional, default=10
            Seed for the random number generator, ensuring reproducibility of the generated noisy data.
        """

        # --- Initialize Function Type ---
        self._ftype = ftype

        self._cost_function = cost_function

        # --- Setup Derivatives (JAX) ---
        self._cost_function_diff = jacobian(cost_function, argnums=1)

        # --- Initialize Random Generator ---
        self._rng = np.random.default_rng(seed)

        # --- Setup Design Variables ---
        self.inp_mat_keys = dsvars.index.tolist()
        self.dsvars = dsvars
        self.nvars = dsvars["variable"].values.shape[0]
        self.xi = dsvars["variable"].values.astype(float).copy()
        self.xi_ref = dsvars["values"].values.astype(float).copy()
        self.xi_bounds = dsvars[["lower", "upper"]].values.tolist()

        # --- Generate Synthetic Data ---
        self._xdata = np.linspace(0., 4., ncontrol)
        self.ydata_ref = cost_function(self._xdata, self.xi)
        self.ydata_pertub = delta * np.random.default_rng().normal(size=self._xdata.size)
        self.ydata_fix = self.ydata_ref + self.ydata_pertub
        self.ncontrol = self.ydata_ref.shape[0]

    def residuum(self, xi: np.ndarray) -> np.ndarray:
        """
        Compute the residuum between fitted and reference data.
        
        Parameters
        ----------
        xi : np.ndarray
            Parameter values for the cost function.
            
        Returns
        -------
        np.ndarray
            Residuum array (reference - computed).
        """
        np_fval = self._cost_function(self._xdata, xi)
        return np.array(self.ydata_fix - np_fval, dtype=float)

    def residuum_diff(self, xi: np.ndarray, fdm: bool = False, **kwargs) -> np.ndarray:
        """
        Compute the derivative of the residuum with respect to parameters.
        
        Parameters
        ----------
        xi : np.ndarray
            Parameter values for the cost function.
        fdm : bool, default=False
            Ignored. Present for interface compatibility with CostFunction.
        **kwargs : dict
            Ignored. Present for interface compatibility with CostFunction.
            
        Returns
        -------
        np.ndarray
            Jacobian of the residuum with respect to parameters.
            
        Notes
        -----
        The `fdm` and `**kwargs` parameters are accepted for polymorphic
        compatibility with `CostFunction.residuum_diff`, but are ignored
        since `LSQFit` uses analytical derivatives from the cost function.
        """
        return np.array(self._cost_function_diff(self._xdata, xi), dtype=float)

    def _adjoint(self, xi: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, xi: np.ndarray) -> float:

        np_res = self.residuum(xi)

        if self._ftype == 'lsq':
            return lsq_fval(np_res)
        elif self._ftype == 'ln':
            return ln_fval(np_res)
        else:
            raise NotImplementedError

    def derivative(self, xi: np.ndarray) -> np.ndarray:

        np_res = self.residuum(xi)
        np_dfval = self.residuum_diff(xi)

        if self._ftype == 'lsq':
            return lsq_dfval(np_res, np_dfval).sum(axis=0)
        elif self._ftype == 'ln':
            return ln_dfval(np_res, np_dfval).sum(axis=0)
        else:
            raise NotImplementedError


class CostFunction(DesignVariablesMixin, ExtensionSolution):
    """
    Cost Function Class to define the minimization problem
    """

    __name__ = 'CostFunction'

    def __init__(self,
                 var_form: VariationalFormulation,
                 load_ref: np.ndarray,
                 stretch_x: np.ndarray,
                 dsvars: pd.DataFrame,
                 ftype: str = 'lsq',
                 module: str = 'numpy',
                 dtype: str = "adjoint",
                 cache_size: Optional[int] = 128  # Add cache_size parameter
                 ):
        """
        Initializes the CostFunction class.

        Parameters:
        -----------
        var_form : VariationalFormulation
            Variational Formulation containing strain energy, stress, and variables.
        ds : float
            Cross-sectional area.
        load_ref : np.ndarray
            Reference load values (experimental data).
        stretch_ref : np.ndarray
            Reference stretch values along the x-axis (experimental data).
        dsvars : pd.DataFrame
            Design variables information table.
        ftype : str, optional
            Type of function to minimize ('lsq' or 'ln'). Defaults to 'lsq'.
        module : str, optional
            Numeric library for Sympy lambdify function. Defaults to 'numpy'.
        dtype : str, optional
            Derivative method ('fdm' or 'adjoint'). Defaults to 'fdm'.
        """

        # --- Initialize Base Class ---
        super().__init__(var_form, module)

        # --- Initialize Design Variable Placeholders ---
        self.nvars = None
        self.dsvars = None

        self.xi = None
        self.xi_ref = None
        self.xi_bounds = None
        self.run_dsvars = None

        self._init_design_variables(dsvars)

        # --- Setup Reference Data ---
        self.stretch_x = stretch_x
        self.np_load_ref = load_ref
        self.ncontrol = load_ref.shape[0]
        self._ftype = ftype
        self._dtype = dtype
        self._cache_size = cache_size

        # --- Build Lambdify Functions ---
        self.lbdf_builder.build_cost_function_lambdas()

        # --- Apply LRU Cache Decorator ---
        if self._cache_size is not None and self._cache_size > 0:
            self._cached_solve = lru_cache(maxsize=self._cache_size)(self._solve_implementation)
        else:
            self._cached_solve = self._solve_implementation

    def update_variables(self, dsvars: pd.DataFrame) -> None:
        """
        Updates the design variables.

        Parameters:
        -----------
        dsvars : pd.DataFrame
            Updated design variables information table.
        """

        self.dsvars = dsvars.loc[self.inp_mat_keys, :]
        self.xi_ref = self.dsvars["values"].values.astype(float).copy()
        self.run_dsvars = self.dsvars["variable"].values.astype(bool)

        self.nvars = np.count_nonzero(self.run_dsvars)
        self.xi = self.xi_ref[self.run_dsvars]
        self.xi_bounds = self.dsvars[["lower", "upper"]][self.run_dsvars].values.tolist()

        # Clear the cache when design variables change
        self.clear_cache()

    def clear_cache(self) -> None:
        """Clears the solution cache."""
        if self._cache_size is not None and self._cache_size > 0:
            self._cached_solve.cache_clear()

    def _solve_implementation(self, xi_mat: tuple, stretch_x: tuple, **kwargs) -> optimize.OptimizeResult:
        """Internal method to perform the actual solve (called by _cached_solve)."""
        # Convert tuples back to NumPy arrays

        np_xi_mat = np.array(xi_mat, dtype=float).reshape(self.xi_ref.shape)
        np_stretch_x = np.sort(np.array(stretch_x, dtype=float))

        # Build a writable Series from the design variables slice
        _sr = self.dsvars["values"][self.run_dsvars]
        sr_xi_mat = pd.Series(_sr.values.astype(float).copy(), index=_sr.index)
        sr_xi_mat[:] = np_xi_mat

        # Solve the problem using the parent class's solve method
        result = self.solve(mat_params=sr_xi_mat, stretch_x=np_stretch_x, max_iter=1000, **kwargs)
        if not result.success:
            warnings.warn(f"Solver failed to converge. Mat. Params: {np_xi_mat}", RuntimeWarning)

        return result

    def _chk_solve(self, xi: np.ndarray, **kwargs) -> optimize.OptimizeResult:
        """
        Solves the problem with the given material parameters.

        Parameters:
        -----------
        xi : np.ndarray
            Material parameter values.

        Returns:
        --------
        Dict[str, np.ndarray]
            Solution output from the `solve` method.
        """
        # Update design variables (also updates self.xi_ref)
        self._update_design_variables(xi)

        # Use the cached solver
        # Convert arrays to tuples for hashable cache key
        xi_ref_tuple = tuple(np.asarray(self.xi_ref, dtype=float).flatten().tolist())
        stretch_x_tuple = tuple(np.asarray(self.stretch_x, dtype=float).flatten().tolist())
        return self._cached_solve(xi_ref_tuple, stretch_x_tuple, **kwargs)

    def _fint_x(self, xi: np.ndarray) -> np.ndarray:
        result = self._chk_solve(xi)
        return result.fint[:, 0]

    def _fint_x_diff(self, xi: np.ndarray, fdm: bool = False, **kwargs) -> np.ndarray:

        if self._dtype == 'adjoint' and fdm is False:
            result = self._chk_solve(xi)

            # Get the solution
            list_fwr_dfin_dm = []
            for xi_primal_i in result.stretch:
                # dJ/dm
                dfin_dm_i = self.lbdf_builder.dfint_x_dm(*xi_primal_i, *xi)

                # dJ/du
                dfin_du_i = self.lbdf_builder.dfint_x_du(*xi_primal_i, *xi)

                # ∂R/∂m [dim_m x dim_u]
                dR_dm_i = self.lbdf_builder.dR_dm(*xi_primal_i, *xi)

                # ∂R/∂u (reduced Hessian if driver variable eliminated)
                dR_du_i = self.lbdf_builder.dR_du(*xi_primal_i, *xi)

                list_fwr_dfin_dm.append(adjoint_derivative(dfin_du_i, dfin_dm_i, dR_dm_i, dR_du_i))
            return np.array(list_fwr_dfin_dm, dtype=float)

        elif self._dtype == 'fdm' or fdm is True:
            return _fdm(self.residuum, xi, xi_bounds=self.xi_bounds, **kwargs)

        return np.asarray([])

    def residuum(self, xi: np.ndarray) -> np.ndarray:
        """
        Computes the residuum between computed and reference load values.

        Parameters:
        -----------
        xi : np.ndarray
            Material parameter values.

        Returns:
        --------
        np.ndarray
            Residuum array.
        """
        xi = np.asarray(xi, dtype=float)
        return self._fint_x(xi) - self.np_load_ref

    def residuum_diff(self, xi: np.ndarray, fdm: bool = False, **kwargs) -> np.ndarray:
        """Return ``dJ/dm`` assuming that ``R(x, m) = 0``
        where J = Res, the residuum function, future improvement will be the use of generic function J(x, m)

        Parameters
        xi (m):    Design Variables that the derivative is being taken with respect to

        **Cost Function:** `J = J(u, p, θ, m)`      (Note: In your specific example, `J` might not directly depend on `p` or `θ`, but in general, it *could*.)
        **State Variables:** `x = [u, p, θ]`        (where `u` represents the displacement-related variables `lx, ly, lz`)
        **Design Variables:** `m`                   (a vector of material parameters, including `D`)
        **Residual Equations:**                     `R(x, d) = 0`, where `R = [R_u, R_p, R_θ]` for the three-field formulation.

        We want to find `dJ/dm`, using the chain rule (total derivative)

        This is the key result.  The total derivative `dJ/dm` is now expressed in terms of:

        1.  `∂J/∂d`: The *direct* dependence of `J` on `m`.
        2.  `∂R/∂d`: The *direct* dependence of the residuals on `d`.
        3.  `λ`: The adjoint variable, which captures the *indirect* dependence through the state variables.
        """

        xi = np.asarray(xi, dtype=float)

        if self._dtype == 'adjoint' and fdm is False:
            result = self._chk_solve(xi)

            # Get the solution
            list_fwr_dfin_dm = []
            for xi_primal_i in result.stretch:
                # dJ/dm
                dfin_dm_i = self.lbdf_builder.dfint_x_dm(*xi_primal_i, *xi)

                # dJ/du
                dfin_du_i = self.lbdf_builder.dfint_x_du(*xi_primal_i, *xi)

                # ∂R/∂m [dim_m x dim_u]
                dR_dm_i = self.lbdf_builder.dR_dm(*xi_primal_i, *xi)

                # ∂R/∂u (reduced Hessian if driver variable eliminated)
                dR_du_i = self.lbdf_builder.dR_du(*xi_primal_i, *xi)

                list_fwr_dfin_dm.append(adjoint_derivative(dfin_du_i, dfin_dm_i, dR_dm_i, dR_du_i))
            return np.array(list_fwr_dfin_dm, dtype=float)

        elif self._dtype == 'fdm' or fdm is True:
            return _fdm(self.residuum, xi, xi_bounds=self.xi_bounds, **kwargs)

        else:
            raise NotImplementedError("Not implemented")

    def volume(self, xi: np.ndarray) -> np.ndarray:
        """
        Isotropic Volume regularization function

        Parameters
            xi (m):    Design Variables that the derivative is being taken with respect to
        """

        xi = np.asarray(xi, dtype=float)

        result = self._chk_solve(xi)

        # Get the solution
        list_fwr_volume = []
        for xi_primal_i in result.stretch:
            list_fwr_volume.append(self.lbdf_builder.Jvol(*xi_primal_i, *xi))

        return np.asarray(list_fwr_volume, dtype=float)

    def volume_diff(self, xi: np.ndarray, fdm: bool = False, **kwargs) -> np.ndarray:
        """Return ``dJ/dm`` assuming that ``R(x, m) = 0``
        where J = Res, the residuum function, future improvement will be the use of generic function J(x, m)

        Parameters
        xi (m):    Design Variables that the derivative is being taken with respect to

        **Cost Function:** `J = J(u, p, θ, m)`      (Note: In your specific example, `J` might not directly depend on `p` or `θ`, but in general, it *could*.)
        **State Variables:** `x = [u, p, θ]`        (where `u` represents the displacement-related variables `lx, ly, lz`)
        **Design Variables:** `m`                   (a vector of material parameters, including `D`)
        **Residual Equations:**                     `R(x, d) = 0`, where `R = [R_u, R_p, R_θ]` for the three-field formulation.

        We want to find `dJ/dm`, using the chain rule (total derivative)

        This is the key result.  The total derivative `dJ/dm` is now expressed in terms of:

        1.  `∂J/∂d`: The *direct* dependence of `J` on `m`.
        2.  `∂R/∂d`: The *direct* dependence of the residuals on `d`.
        3.  `λ`: The adjoint variable, which captures the *indirect* dependence through the state variables.
        """

        xi = np.asarray(xi, dtype=float)

        if self._dtype == 'adjoint' and fdm is False:
            result = self._chk_solve(xi)

            # Get the solution
            list_fwr_volume_dm = []
            for xi_primal_i in result.stretch:
                # dJ/dm
                dvol_dm_i = self.lbdf_builder.dJvol_dm(*xi_primal_i, *xi)

                # dJ/du
                dvol_du_i = self.lbdf_builder.dJvol_du(*xi_primal_i, *xi)

                # ∂R/∂m [dim_m x dim_u]
                dR_dm_i = self.lbdf_builder.dR_dm(*xi_primal_i, *xi)

                # ∂R/∂u (reduced Hessian if driver variable eliminated)
                dR_du_i = self.lbdf_builder.dR_du(*xi_primal_i, *xi)

                list_fwr_volume_dm.append(adjoint_derivative(dvol_du_i, dvol_dm_i, dR_dm_i, dR_du_i))
            return np.asarray(list_fwr_volume_dm, dtype=float)

        elif self._dtype == 'fdm' or fdm is True:
            return _fdm(self.volume, xi, xi_bounds=self.xi_bounds, **kwargs)

        else:
            raise NotImplementedError("Not implemented")

    def __call__(self, xi: np.ndarray) -> float | None | Any:
        """
        Computes the objective function value.

        Parameters:
        -----------
        xi : np.ndarray
            Material parameter values.

        Returns:
        --------
        float
            Objective function value.
        """

        np_rx = self.residuum(xi)
        np_rx2 = np.inner(np_rx, np_rx)

        if has_nan(np.atleast_1d(np_rx2)):
            return 1.e6

        elif np.isscalar(np_rx2):
            if self._ftype == 'lsq':
                return np.dot(np_rx, np_rx)
            elif self._ftype == 'ln':
                return np.log(np.dot(np_rx, np_rx) + 1.)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError("Objective function value type not implemented.")

    def derivative(self, xi: np.ndarray, np_out: np.ndarray = None) -> np.ndarray:
        """
        Computes the derivative of the objective function.

        Parameters:
        -----------
        xi : np.ndarray
            Material parameter values.
        np_out : np.ndarray, optional
            Output array to store the derivative. Defaults to None.

        Returns:
        --------
        np.ndarray
            Derivative array.
        """

        np_resid = self.residuum(xi)

        if self._dtype == "adjoint":
            np_dfin_dxi = self.residuum_diff(xi)

        elif self._dtype == "fdm":
            np_dfin_dxi = self.residuum_diff(xi, fdm=True, h=1.e-5)

        else:
            raise NotImplementedError(f"Derivative type '{self._dtype}' is not implemented.")

        np_dlsq_dxi = np_resid @ np_dfin_dxi

        if self._ftype == 'ln':
            lsq_val = 0.5 * np.inner(np_resid, np_resid)
            np_dlsq_dxi = 2. * np_dlsq_dxi / (lsq_val + 1.)

        elif self._ftype != 'lsq':
            raise NotImplementedError(f"Function type '{self._ftype}' is not implemented.")

        if np_out is not None:
            if np_out.shape[0] != np_dlsq_dxi.shape[0]:
                raise ValueError(
                    f"Output array shape mismatch: "
                    f"np_out.shape[0]={np_out.shape[0]} != np_dlsq_dxi.shape[0]={np_dlsq_dxi.shape[0]}"
                )
            np_out[:] = np_dlsq_dxi.astype(float)

        return np_dlsq_dxi.astype(float)


class CostIntegrator:
    """
    Integrates multiple Cost Functions (LSQ) and provides methods for calculating the combined cost function and its
    derivative, with optional regularization and caching.
    
    This class uses extracted components for caching (CostCache) and regularization
    (L2Regularization, VolumeRegularization) to follow the Single Responsibility Principle.
    """

    def __init__(self,
                 mat_cost_fun: Sequence[CostFunction | LSQFit],
                 ftype: str = 'lsq',
                 fid: Optional[int] = None,
                 vol_reg: bool = False,
                 rescale: str = None,
                 cache_size: int = 128,
                 **kwargs,
                 ):
        """
        Initialize the CostIntegrator.

        Args:
            xi: Initial design variables array.
            mat_cost_fun: Sequence of Cost Function objects.
            ftype: Type of the cost function ('lsq', 'ln', 'cauchy', etc.).
            fid: Index of the primary function to focus on (if stab < 1).
            vol_reg: Flag to include volume regularization.
            rescale: Rescaling method for Tikhonov regularization ('direct', 'inverse', 'inverse_nrs', None).
            cache_size: Maximum number of results to store in the LRU cache.
            **kwargs: Additional keyword arguments for cost functions (e.g., 'c', 'rho', 'alpha', 'beta', 'vol').

        Raises:
            ValueError: If bounds are incorrectly specified.
        """
        # --- Validate Inputs ---
        if not isinstance(mat_cost_fun, list):
            raise TypeError(
                f'LSQ function must be a list of LSQ function objects, '
                f'got {type(mat_cost_fun).__name__}'
            )

        if not mat_cost_fun:
            raise ValueError('mat_cost_fun list cannot be empty - at least one LSQ function is required')

        # --- Initialize Design Variables ---
        self.xi = mat_cost_fun[0].xi.copy()
        self.xi_ref = mat_cost_fun[0].xi_ref.copy()
        self.nvars = self.xi.shape[0]
        self.inp_mat_keys = mat_cost_fun[0].inp_mat_keys.copy()

        self.cost_functions = mat_cost_fun
        self._ftype = ftype

        # --- Setup Bounds ---
        self.xi_bounds = None
        self._check_bounds()

        # --- Configure Function Indices ---
        if fid is None:
            self._fid = np.arange(len(mat_cost_fun))
        elif isinstance(fid, int):
            if not 0 <= fid < len(mat_cost_fun):
                raise ValueError(f"fid {fid} is out of range for mat_cost_fun list of size {len(mat_cost_fun)}")
            self._fid = np.array([fid])
        else:
            raise ValueError(
                f'fid must be an integer or None, got {type(fid).__name__}'
            )

        # --- Configure Regularization Parameters ---
        self._mobj = len(mat_cost_fun) > 0
        self._iter = 0
        self._rescale = rescale
        self._dvol = vol_reg

        self._check_kwargs(kwargs)

        # --- Initialize Cache (using extracted CostCache) ---
        self.cache_size = cache_size
        self._cache = CostCache(cache_size)
        
        # Legacy cache references for backward compatibility
        self._cache_residuum = self._cache.residuum
        self._cache_residuum_diff = self._cache.residuum_diff
        self._cache_volume = self._cache.volume
        self._cache_volume_diff = self._cache.volume_diff
        
        # --- Initialize Regularization Strategies ---
        epsilon = kwargs.get("epsilon", 0.) or 0.
        self._regularization = self._build_regularization(vol_reg, epsilon)

    def _build_regularization(self, vol_reg: bool, epsilon: float) -> CompositeRegularization:
        """Build the composite regularization strategy."""
        regularization = CompositeRegularization()

        # Add L2 (Tikhonov) regularization if alpha > 0
        if self._alpha > 0:
            l2_reg = L2Regularization(
                xi_ref=self.xi_ref,
                alpha=self._alpha,
                rescale=self._rescale,
                beta=self._beta,
                xi_bounds=self.xi_bounds,
                multi_objective=(len(self.cost_functions) > 1),
            )
            regularization.add_strategy(l2_reg)

        # Add volume regularization if enabled
        if vol_reg and epsilon > 0.:
            vol_strategy = VolumeRegularization(
                cost_functions=self.cost_functions,
                epsilon=epsilon,
                xi_bounds=self.xi_bounds,
                cache=self._cache,
            )
            regularization.add_strategy(vol_strategy)
        
        return regularization

    def _check_bounds(self) -> None:
        # bounds search
        for fun_i in self.cost_functions:
            if fun_i.xi_bounds is not None:
                if self.xi_bounds is None:
                    self.xi_bounds = fun_i.xi_bounds.copy()
                else:
                    # Compute the intersection of bounds
                    for j in range(len(self.xi_bounds)):
                        self.xi_bounds[j][0] = max(self.xi_bounds[j][0], fun_i.xi_bounds[j][0])
                        self.xi_bounds[j][1] = min(self.xi_bounds[j][1], fun_i.xi_bounds[j][1])
                        if self.xi_bounds[j][0] > self.xi_bounds[j][1]:
                            raise ValueError(f"Conflicting bounds for variable {j}: "
                                             f"[{self.xi_bounds[j][0]}, {self.xi_bounds[j][1]}]")

    def _check_kwargs(self, kwargs) -> None:

        # TODO: check each variable has been deprecated, update fitting/core.py
        self._c = kwargs.get("c", 10.0)                 # Cauchy parameter
        self._alpha = kwargs.get("alpha", 0.)           # Tikhonov regularization scaling parameter
        if self._alpha is None:
            self._alpha = 0.

        self._beta = kwargs.get("beta", 2.0)            # Tikhonov rescaling parameter
        self._epsilon = kwargs.get("epsilon", 0.)       # Volume Strain Energy regularization scaling parameter
        if self._epsilon is None:
            self._epsilon = 0.

    def clear_cache(self) -> None:
        """
        Clear all cached values.
        
        This should be called when design variables or cost functions change
        to ensure fresh computations.
        """
        self._cache.clear()

    def cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns
        -------
        dict
            Dictionary with cache entry counts and capacity.
        """
        return self._cache.stats()

    def _residuum(self, xi: np.ndarray) -> np.ndarray:
        """
        Aggregate residua from all cost functions with caching.
        
        This method collects residuum values from each cost function in
        `mat_cost_fun` and caches the result for efficiency.
        
        Parameters
        ----------
        xi : np.ndarray
            Design variable values (material parameters).
            
        Returns
        -------
        np.ndarray
            Stacked residua from all cost functions, shape (n_functions, n_control).
            NaN, inf values are replaced with 0.
        """
        # Check cache first using CostCache API
        np_resi = self._cache.get_residuum(xi)
        if np_resi is not None:
            return np_resi

        # Compute residuum for all cost functions
        list_resi = []
        for fun_i in self.cost_functions:
            list_resi.append(sanitize_array(fun_i.residuum(xi)))

        np_resi = np.asarray(list_resi, dtype=float)
        self._cache.set_residuum(xi, np_resi)

        return np_resi

    def _residuum_diff(self, xi: np.ndarray, fdm: bool = False, **kwargs) -> np.ndarray:
        """
        Compute the derivative of residua from all cost functions with caching.
        
        Parameters
        ----------
        xi : np.ndarray
            Design variable values (material parameters).
        fdm : bool, default=False
            If True, use finite difference method.
        **kwargs
            Additional arguments for FDM computation.
            
        Returns
        -------
        np.ndarray
            Stacked residuum derivatives from all cost functions.
        """
        # Check cache for analytical (non-fdm) path
        if not fdm:
            np_resi_diff = self._cache.get_residuum_diff(xi)
            if np_resi_diff is not None:
                return np_resi_diff

        # Compute residuum derivative for all cost functions
        list_resi_diff = []
        for fun_i in self.cost_functions:
            resi_diff_i = sanitize_array(fun_i.residuum_diff(xi, fdm=fdm, **kwargs))
            list_resi_diff.append(resi_diff_i)

        np_resi_diff = np.asarray(list_resi_diff, dtype=float)
        
        # Store in cache only for the analytical (non-fdm) path
        if not fdm:
            self._cache.set_residuum_diff(xi, np_resi_diff)

        return np_resi_diff

    def _function_type(self, residuum: np.ndarray, **kwargs) -> Union[float, np.ndarray]:

        if 'ln' in self._ftype:
            return ln_fval(residuum, **kwargs)
        elif 'logcosh' in self._ftype:
            return logcosh_fval(residuum, **kwargs)
        elif 'huber' in self._ftype:
            return huber_fval(residuum, **kwargs)
        elif 'lsq' in self._ftype:
            return lsq_fval(residuum, **kwargs)
        elif 'cauchy' in self._ftype:
            return cauchy_fval(residuum, c=self._c, **kwargs)
        else:
            raise ValueError(f"Invalid ftype '{self._ftype}' specified.")

    def _function_type_diff(self, residuum: np.ndarray, residuum_diff: np.ndarray, **kwargs) -> np.ndarray:

        if 'ln' in self._ftype:
            return ln_dfval(residuum, residuum_diff, **kwargs)
        elif 'logcosh' in self._ftype:
            return logcosh_dfval(residuum, residuum_diff, **kwargs)
        elif 'huber' in self._ftype:
            return huber_dfval(residuum, residuum_diff, **kwargs)
        elif 'lsq' in self._ftype:
            return lsq_dfval(residuum, residuum_diff, **kwargs)
        elif 'cauchy' in self._ftype:
            return cauchy_dfval(residuum, residuum_diff, c=self._c, **kwargs)
        else:
            raise ValueError(f"Invalid ftype '{self._ftype}' specified.")

    @staticmethod
    def _map_residuum_function(args) -> tuple[np.ndarray, np.ndarray]:
        i, fid, fun, xi, diff = args

        np_residuum_diff_i = np.zeros((fun.ncontrol, fun.nvars), dtype=float)
        np_residuum_i = sanitize_array(fun.residuum(xi))

        if diff == 1:
            np_residuum_diff_i = sanitize_array(fun.residuum_diff(xi))

        return np_residuum_i, np_residuum_diff_i

    def _sum_function(self, residuum: np.ndarray) -> float:
        return self._function_type(residuum).sum().item()

    def _sum_function_diff(self, residuum: np.ndarray, residuum_diff: np.ndarray) -> np.ndarray:
        return sanitize_array(self._function_type_diff(residuum, residuum_diff).sum(axis=0))

    def _cost_function(self, xi: np.ndarray, fsum: bool = True) -> float | np.ndarray:
        """
        Compute the cost function value.

        Args:
            xi (np.ndarray): Current value of variables.
            fsum (bool): If True (default), return a scalar total cost. If False,
                return a per-function array so callers can inspect individual
                contributions (e.g. for residual variance estimation).

        Returns:
            float | np.ndarray: Scalar cost (fsum=True) or per-function array (fsum=False).

        Raises:
            ValueError: If invalid ftype is specified.
        """

        np_resi = self._residuum(xi)
        reg_value = self._regularization.value(xi)

        if fsum:
            cost_fval = self._sum_function(np_resi)
            return cost_fval + reg_value
        else:
            cost_fval_array = self._function_type(np_resi)
            return cost_fval_array + reg_value / cost_fval_array.shape[0]

    def _cost_function_diff(self, xi: np.ndarray, fdm: bool = False,
                            fsum: bool = True, freg: bool = True, **kwargs) -> np.ndarray:

        if not isinstance(xi, np.ndarray):
            raise TypeError(
                f"xi must be a numpy array, got {type(xi).__name__}"
            )

        if not fdm:
            np_resi = self._residuum(xi)
            np_resi_diff = self._residuum_diff(xi)
            np_reg_grad = self._regularization.gradient(xi, fdm=fdm, **kwargs)

            if fsum:
                np_cost_dfval = self._sum_function_diff(np_resi, np_resi_diff)
            else:
                np_cost_dfval = self._function_type_diff(np_resi, np_resi_diff)
                n_funcs = np_cost_dfval.shape[0]
                np_reg_grad = np.asarray([np_reg_grad for _ in range(n_funcs)]) / n_funcs

            if freg:
                np_cost_dfval = np_cost_dfval + np_reg_grad

        else:
            np_cost_dfval = _fdm(self._cost_function, xi, xi_bounds=self.xi_bounds, **kwargs)

        if not is_finite(np_cost_dfval):
            nan_mask = ~np.isfinite(np_cost_dfval)
            nan_indices = np.where(nan_mask)[0]
            raise ValueError(
                f"Gradient contains NaN/Inf values at indices {nan_indices.tolist()}. "
                f"xi = {xi}, gradient = {np_cost_dfval}"
            )

        return np_cost_dfval

    def mse(self, xi: np.ndarray) -> float:
        """
        Compute the Mean-Squared Error (MSE) of the residuals at xi.

        Args:
            xi (np.ndarray): Current value of variables.

        Returns:
            float: MSE value computed among all data points.
        """
        # FIXME: This function is not tested yet and need to be reviewed

        if xi.shape[0] != self.nvars:
            raise ValueError(f"Expected xi of shape ({self.nvars},), got {xi.shape}.")

        # If self._mobj is True, we have multiple cost functions
        # We need to gather all residuals from each function
        if self._mobj:
            # FIXME: Remove _map_residuum_function, use a loop instead
            list_args = [(i, self._fid, fun_i, xi, 0) for i, fun_i in enumerate(self.cost_functions)]
            results = map(self._map_residuum_function, list_args)
            list_residuum, _ = zip(*results)  # We don't need the derivative part

            np_resid = np.array(list_residuum, dtype=float)

            # Flatten to get all residuals from all subproblems:
            all_resid = np_resid.ravel()
        else:
            # Single cost function
            all_resid = self.cost_functions[-1].residuum(xi)

        # Compute MSE
        # 1) sum of squares
        sum_of_squares = np.sum(all_resid ** 2)

        # 2) number of points
        n_total = all_resid.shape[0]

        # 3) mean (guard against zero points)
        mse_val = safe_divide(sum_of_squares, n_total, default=0.0)

        return mse_val

    def __call__(self, xi: np.ndarray) -> float:
        """
        Compute the cost function value at xi.

        Args:
            xi (np.ndarray): Current value of variables.

        Returns:
            float: Cost function value.

        Raises:
            ValueError: If xi has incorrect shape or invalid barrier type.
        """
        if not isinstance(xi, np.ndarray):
            raise ValueError(f"Expected xi as numpy array, got {type(xi)}.")

        if xi.shape[0] != self.nvars:
            raise ValueError(f"Expected xi of shape ({self.nvars},), got {xi.shape}.")

        if self._mobj:
            fval = self._cost_function(xi)
        else:
            fval = self.cost_functions[-1](xi)

        return float(fval)

    def derivative(self, xi: np.ndarray, df_xi: np.ndarray = None) -> np.ndarray:
        """
        Compute the derivative of the cost function at xi.

        Args:
            xi (np.ndarray): Current value of variables.
            df_xi (np.ndarray, optional): Array to store the derivative.

        Returns:
            np.ndarray: Gradient vector.

        Raises:
            ValueError: If xi has incorrect shape or invalid barrier type.
        """

        if xi.shape[0] != self.nvars:
            raise ValueError(f"Expected xi of shape ({self.nvars},), got {xi.shape}.")

        self._iter += 1

        if df_xi is not None and isinstance(df_xi, np.ndarray):
            if df_xi.shape[0] != self.nvars:
                raise ValueError(f"Expected df_xi of shape ({self.nvars},), got {df_xi.shape}.")

        # Compute derivative of the least squares function
        if self._mobj:
            np_eqv_dfun = self._cost_function_diff(xi)
        else:
            np_eqv_dfun = self.cost_functions[-1].derivative(xi)

        if df_xi is not None:
            df_xi[:] = np_eqv_dfun.astype(float)

        return np_eqv_dfun.astype(float)
