# -*- coding: utf-8 -*-
"""
Optimization mixin for AnisoMaterialFit.

Provides baseline and section-specific parameter optimization methods.
"""
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from dualmatfit.optimization.cost import CostFunction, CostIntegrator
from dualmatfit.optimization.drivers import opt_solvers
from dualmatfit.data.experimental import InstronData
from dualmatfit.plotting.experimental_visuals import plot_material_fit, exp_test_plot, stress_plot
from dualmatfit.fitting.constants import (
    DEFAULT_BASELINE_OPTIMIZATION_ITERATIONS,
    DEFAULT_LOCAL_OPTIMIZATION_ITERATIONS,
    DEFAULT_GLOBAL_ITERATIONS,
    HIGH_RESOLUTION_NCONTROL,
    UNSTRETCHED_STATE,
    DEFAULT_PLOT_LIMITS,
)

from dualmatfit.utils.logging_config import get_logger
logger = get_logger('fitting.optimization')

__all__ = ['FitOptimizationMixin']


class FitOptimizationMixin:
    """Optimization methods for AnisoMaterialFit."""

    def find_baseline_parameters(self,
                                 ftype: Optional[str] = None,
                                 miter: int = None,
                                 **kwargs,
                                 ) -> pd.DataFrame:
        """
        Find baseline (global) material parameters across all test cases.

        Performs global optimization to find a single set of material parameters
        that best fits all experimental data simultaneously. This is the primary
        method for material parameter identification in the fitting workflow.

        Parameters
        ----------
        ftype : str, optional
            Cost function type for least squares fitting.
            Options:
                - 'cauchy_robust': Robust Cauchy loss (default, recommended)
                - 'l2': Standard L2 norm (least squares)
                - 'huber': Huber loss (robust)
                - 'soft_l1': Soft L1 loss
            If None, uses 'cauchy_robust'.

        miter : int, optional
            Maximum number of iterations for local optimization.
            If None, uses DEFAULT_BASELINE_OPTIMIZATION_ITERATIONS.
            Larger values allow more thorough convergence.

        **kwargs : dict
            Additional optimization parameters:

            rho : float, optional
                KS aggregation parameter for constraint handling.

            c : float, default=40.0
                Robustness parameter for robust loss functions.
                Smaller values = more robust to outliers.

            giter : int, default=3
                Number of global iterations (basin-hopping).

            alpha : float, optional
                Fiber angle constraint (radians).
                If provided, fixes fiber angle to this value.

            rescale : str, optional
                Stress rescaling method ('std', 'minmax', None).

            dvol : bool, default=True
                Include volumetric terms in optimization.

            epsilon : float, optional
                Regularization parameter for cost function.

            xi : np.ndarray, optional
                Initial parameter guess [mu, D, k1, k2, alpha, kappa].

            bh_step : str, default='random_displacement'
                Basin-hopping step strategy.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row containing optimized baseline parameters.
            Columns: Parameter names (mu, D, k_1, k_2, alpha, kappa, etc.)
            Values: Optimized parameter values

        Notes
        -----
        This method:
        1. Aggregates all experimental data into single cost function
        2. Performs global optimization (if opt_glb=True)
        3. Returns single parameter set representing baseline tissue properties

        For section-specific parameters, use find_optimal_parameters() instead.

        Examples
        --------
        Basic usage with defaults:

        >>> fit = AnisoMaterialFit(selection, opt_type='ipopt', opt_glb=True)
        >>> df_baseline = fit.find_baseline_parameters()

        With custom settings:

        >>> df_baseline = fit.find_baseline_parameters(
        ...     ftype='cauchy_robust',
        ...     miter=100,
        ...     giter=5,
        ...     c=50.0
        ... )

        See Also
        --------
        find_optimal_parameters : Section-specific optimization
        """

        # --- Extract Optimization Parameters ---
        # KS function parameter
        if kwargs.get("rho") is not None:
            rho = kwargs.get("rho")
        else:
            rho = None

        # Cauchy Loss function parameter
        if kwargs.get("c") is not None:
            cm = kwargs.get("c")
        else:
            cm = None

        # L2: Adding Tikhonov Regularization
        if kwargs.get("alpha") is not None:
            alpha = kwargs.get("alpha")
        else:
            alpha = None

        # Regularization Rescaling method ('direct' or 'inverse'). Defaults to 'direct'.
        if kwargs.get("rescale") is not None:
            rescale = kwargs.get("rescale")
        else:
            rescale = None

        # Volume Regularization
        if kwargs.get("epsilon") is not None:
            epsilon = kwargs.get("epsilon")
        else:
            epsilon = None

        if kwargs.get("giter") is not None:
            giter = kwargs.get("giter")
        else:
            giter = DEFAULT_GLOBAL_ITERATIONS

        # Basinhopping Step parameter
        if kwargs.get("bh_step") is not None:
            bh_step = kwargs.get("bh_step")
        else:
            bh_step = "random_displacement"

        # Volume Stabilization Parameter
        if kwargs.get("dvol") is not None:
            dvol = kwargs.get("dvol")
        else:
            dvol = False

        # Set default miter if not provided
        if miter is None:
            miter = DEFAULT_BASELINE_OPTIMIZATION_ITERATIONS

        # Cost function type
        if ftype is not None:
            self._baseline_ftype = ftype

        # --- Initialize Design Variables ---
        df_mat_params = self.ds_vars[["values", "lower", "upper", "limit", "variable"]].copy(deep=True)

        # Initial Guess for the design variable
        if kwargs.get("xi") is not None:
            xi = kwargs.get("xi")
            np_xi = np.asarray(xi)

            if df_mat_params["values"].values.shape == np_xi.shape:
                df_mat_params.loc[:, "values"] = np_xi

        # --- Run Global Optimization ---
        for key_rnm_i, val_i in self.model_opt_res.items():
            list_lsq_i = [val_ik['lsq'] for key_ik, val_ik in val_i.items()]
            lsq_fun_i = CostIntegrator(mat_cost_fun=list_lsq_i,
                                       ftype=ftype,
                                       c=cm,
                                       rho=rho,
                                       alpha=alpha,
                                       rescale=rescale,
                                       vol_reg=dvol,
                                       epsilon=epsilon
                                       )

            opt_args_i = [self.opt_type, lsq_fun_i, df_mat_params.loc[lsq_fun_i.inp_mat_keys, :]]

            with (warnings.catch_warnings()):
                logger.info(f'\n ==> Solving Global Baseline Parameters: {key_rnm_i}')
                opt_res_i = opt_solvers(*opt_args_i,
                                        miter=miter,
                                        giter=giter,
                                        glb=self.opt_glb,
                                        bh_step=bh_step,
                                        )

                if len(opt_res_i.x) < self.ds_vars.index.shape[0]:
                    raise ValueError(
                        f"Error with Baseline optimization process: "
                        f"result length {len(opt_res_i.x)} < expected {self.ds_vars.index.shape[0]}"
                    )

                self.baseline_ds_vars.loc[key_rnm_i] = opt_res_i.series

        return self.baseline_ds_vars

    def _eval_problem(self,
                      min_lx: float,
                      opt_material: OptimizeResult,
                      instron_data: InstronData,
                      lsq: CostFunction,
                      plot_fname: str = "",
                      ncontrol: int = None,
                      ) -> Tuple[Dict[str, Any], pd.Series]:
        """
        Evaluate optimization problem and generate fit plots.

        Computes model solution with optimized parameters, generates comparison
        plots with experimental data, and extracts strain energy components.

        Parameters
        ----------
        min_lx : float
            Maximum stretch ratio to evaluate (lambda_max)
        opt_material : OptimizeResult
            Optimization result containing fitted material parameters
        instron_data : InstronData
            Experimental data object with measured force-stretch curves
        lsq : CostFunction
            Cost function object for computing model predictions
        plot_fname : str, optional
            Filename for saving comparison plot, by default ""
        ncontrol : int, optional
            Number of points for solution evaluation. If None, uses self.ncontrol

        Returns
        -------
        opt_solution : Dict[str, Any]
            Dictionary containing:
            - 'pk1': First Piola-Kirchhoff stress components
            - 'cauchy': Cauchy stress components
            - 'ese': Strain energy density components
            - 'lx': Stretch ratios evaluated
        sr_dsvars : pd.Series
            Series of optimized material parameter values

        Notes
        -----
        This method:
        1. Sets bulk modulus if volumetric effects are disabled
        2. Computes model solution at specified stretch ratios
        3. Generates comparison plots (experimental vs model)
        4. Extracts and sums strain energy components

        The plot shows:
        - Experimental data points
        - Model prediction curve
        - Component breakdown (isotropic, anisotropic, volumetric)

        Strain Energy Components:
        - 'iso':    Isotropic (elastin) contribution
        - 'ani':    Anisotropic (collagen) contribution
        - 'vol':    Volumetric constraint/ contribution
        - 'full':   Total strain energy density

        See Also
        --------
        plot_material_fit : Generate comparison plots
        CostFunction.solve : Compute model solution
        """

        # --- Validate Inputs ---
        if ncontrol is None:
            ncontrol = self.ncontrol

        if not self.dvol:
            opt_material.series["bulk"] = self.bulk

        # --- Compute Model Solution ---
        kwargs_sol = {}
        if min_lx > UNSTRETCHED_STATE:
            np_lx = np.linspace(UNSTRETCHED_STATE, min_lx, num=ncontrol)
            kwargs_sol["stretch_x"] = np_lx

        opt_solution = lsq.solve(opt_material.series, **kwargs_sol)

        # --- Generate Comparison Plots ---
        plot_path = Path(plot_fname)
        fname = plot_path.stem
        path_loc = str(plot_path.parent)

        plot_material_fit(instron_data, opt_solution, path_loc, filename_prefix=fname)

        # --- Extract Strain Energy Components ---
        for key_l in self.ese_types:
            if opt_solution["ese"].get(key_l) is not None:
                opt_material[key_l] = sum(opt_solution["ese"][key_l])

        return opt_solution, opt_material.series

    def _configure_optimization_parameters(self,
                                        miter: Optional[int],
                                        ftype: Optional[str],
                                        sections: Optional[List[str]],
                                        **kwargs,
                                           ) -> Dict[str, Any]:
        """
        Configure and validate optimization parameters.

        Extracts optimization settings from kwargs, applies defaults,
        and validates parameter ranges. Centralizes all parameter handling
        for reproducibility and maintainability.

        Parameters
        ----------
        miter : int, optional
            Maximum iterations for local optimization
        ftype : str, optional
            Cost function type
        sections : List[str], optional
            Sections to optimize (if None, optimizes all)
        **kwargs : dict
            Additional optimization parameters

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary with keys:
            - miter: int - Maximum iterations
            - ftype: str - Cost function type
            - sections: List[str] - Sections to optimize
            - cm: float - Cauchy parameter (c)
            - rho: float - KS aggregation parameter
            - alpha: float - Tikhonov regularization
            - beta: float - Rescaling regularization parameter
            - rescale: str - Rescaling method
            - epsilon: float - Volume regularization
            - giter: int - Global iterations (basin-hopping)
            - dvol: bool - Volume stabilization flag

        Notes
        -----
        - Applies sensible defaults from module constants
        - Validates parameter ranges
        - Logs configuration for reproducibility
        - Updates instance variable self._local_ftype if ftype provided

        """
        # --- Initialize Configuration ---
        config = {'miter': miter if miter is not None else DEFAULT_LOCAL_OPTIMIZATION_ITERATIONS,
                  'giter': kwargs.get('giter', DEFAULT_GLOBAL_ITERATIONS)}

        # --- Cost Function Settings ---
        if ftype is not None:
            self._local_ftype = ftype

        config['ftype'] = ftype

        # --- Loss Function Parameters ---
        config['cm'] = kwargs.get('c')           # Cauchy parameter
        config['rho'] = kwargs.get('rho')        # KS function parameter
        config['alpha'] = kwargs.get('alpha')    # Tikhonov regularization
        config['beta'] = kwargs.get('beta')      # Rescaling regularization
        config['rescale'] = kwargs.get('rescale')  # Rescaling method
        config['epsilon'] = kwargs.get('epsilon')  # Volume regularization

        # --- Stabilization Settings ---
        config['dvol'] = kwargs.get('dvol', False)

        # --- Section Configuration ---
        if sections is None:
            config['sections'] = [f"{sec}-{pos}" for sec in ["Ar", "Tr", "Ab"] for pos in ["A", "B", "C"]]
        else:
            config['sections'] = sections

        # --- Validation ---
        if config['miter'] < 1:
            raise ValueError(f"miter must be positive integer, got {config['miter']}")

        if config['giter'] < 1:
            raise ValueError(f"giter must be positive integer, got {config['giter']}")

        # Log configuration for reproducibility
        logger.info(f"Optimization configuration: miter={config['miter']}, "
                   f"giter={config['giter']}, sections={len(config['sections'])}, "
                   f"ftype={config['ftype']}, dvol={config['dvol']}")

        return config

    def _get_local_solution_path(self, rat_id: str, local_path: Optional[str] = None) -> Path:
        """
        Get absolute path for saving local optimization solutions.

        Constructs the full path for saving optimization results, optionally
        appending a subdirectory to organize multiple optimization runs.

        Parameters
        ----------
        rat_id : str
            Rat identifier (e.g., 'rato-17')
        local_path : str, optional
            Subdirectory within the rat's solution path.
            If None, uses the base solution path.
            Useful for organizing multiple optimization iterations.

        Returns
        -------
        Path
            Absolute path for saving results. Resolved to avoid symlink issues.

        Notes
        -----
        - Always returns absolute path via resolve()
        - Creates parent directories if needed (via ensure_directory_exists)
        - Path object allows easy concatenation with / operator

        See Also
        --------
        PathManager : Central path management system
        solution_path : Dictionary of rat solution paths
        """
        base_path = self.solution_path[rat_id]
        if local_path is not None:
            return (base_path / local_path).resolve()

        return base_path

    @staticmethod
    def _prepare_cost_and_data(rat_data: Dict[str, Any],
                               df_mat_params: pd.DataFrame,
                               ) -> Tuple[List[CostFunction], List[InstronData]]:
        """
        Prepare cost functions and experimental data for optimization.

        Extracts cost function and experimental data objects from rat-specific
        data, updating cost functions with current material parameters.

        Parameters
        ----------
        rat_data : Dict[str, Any]
            Rat-specific data dictionary from model_opt_res containing:
            - Keys: Section-position codes (e.g., 'Ar-A', 'Tr-B')
            - Values: Dict with 'lsq', 'instron', 'ext_solution' keys
        df_mat_params : pd.DataFrame
            Material parameter DataFrame with current baseline values.
            Used to update cost function variables.

        Returns
        -------
        cost_functions : List[CostFunction]
            List of cost function objects, updated with current parameters.
            Order matches the iteration order of rat_data.values()
        instron_data_list : List[InstronData]
            List of experimental data objects.
            Order matches cost_functions list.

        Notes
        -----
        This method:
        1. Iterates through rat_data values
        2. Updates each cost function with df_mat_params
        3. Collects updated cost functions in order
        4. Collects corresponding experimental data

        The order preservation is critical - cost_functions[i] corresponds
        to instron_data_list[i].

        Side Effects
        ------------
        - Modifies cost function objects in rat_data via update_variables()

        See Also
        --------
        CostFunction.update_variables : Update cost function parameters
        model_opt_res : Dictionary of per-rat optimization data
        """
        list_cost_functions = []
        list_instron_data = []

        for val_k in rat_data.values():
            # Update cost function with current material parameters
            val_k['lsq'].update_variables(df_mat_params)
            list_cost_functions.append(val_k['lsq'])
            list_instron_data.append(val_k['instron'])

        return list_cost_functions, list_instron_data

    def _optimize_single_position(self,
                                  rat_id: str,
                                  section_key: str,
                                  section_name: str,
                                  position_code: str,
                                  cost_function: CostFunction,
                                  instron_data: InstronData,
                                  max_stretch: float,
                                  df_mat_params: pd.DataFrame,
                                  local_solution_path: Path,
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize material parameters for a single section-position.

        This is the atomic optimization unit, handling cost function setup,
        optimization execution, result evaluation, and plot generation for
        one anatomical position.

        Parameters
        ----------
        rat_id : str
            Rat identifier (e.g., 'rato-17') for logging
        section_key : str
            Full section-position key (e.g., 'Ar-A', 'Tr-B')
        section_name : str
            Section code only (e.g., 'Ar', 'Tr', 'Ab')
        position_code : str
            Position letter ('A', 'B', or 'C')
        cost_function : CostFunction
            Cost function object for this position
        instron_data : InstronData
            Experimental data for this position
        max_stretch : float
            Maximum stretch ratio to evaluate (lambda_max)
        df_mat_params : pd.DataFrame
            Material parameter DataFrame (baseline values)
        local_solution_path : Path
            Base path for saving results
        config : Dict[str, Any]
            Optimization configuration from _configure_optimization_parameters()

        Returns
        -------
        Dict[str, Any]
            Result dictionary with keys:
            - 'pk1': First Piola-Kirchhoff stress components
            - 'cauchy': Cauchy stress components
            - 'ese': Strain energy density components
            - 'lx': Stretch ratios evaluated
            - 'mat_param': Optimized material parameters (pd.Series)

        Notes
        -----
        Processing steps:
        1. Log optimization start
        2. Create CostIntegrator with single cost function
        3. Run optimization (or use baseline if not in config['sections'])
        4. Evaluate model at specified stretch ratios
        5. Generate comparison plot
        6. Return complete results dictionary

        The method decides whether to optimize based on config['sections']:
        - If section_key in list: Run full optimization
        - Otherwise: Use baseline parameters (empty=True flag)

        Side Effects
        ------------
        - Generates PNG plot file in section subdirectory
        - Logs optimization progress

        See Also
        --------
        CostIntegrator : Aggregates cost functions
        opt_solvers : Optimization solver interface
        _eval_problem : Evaluates and plots optimization result
        """
        # --- Log and Initialize ---
        logger.info(f'\n {rat_id}, Solving Local Parameters, Section: {section_name}, Position: {position_code}')

        df_mat_params_copy = df_mat_params.copy(deep=True)

        # --- Setup Cost Integrator ---
        integrator = CostIntegrator(
            [cost_function],
            ftype=config['ftype'],
            c=config['cm'],
            rho=config['rho'],
            alpha=config['alpha'],
            epsilon=config['epsilon'],
            beta=config['beta'],
            rescale=config['rescale'],
            vol_reg=config['dvol']
        )

        # --- Prepare Optimization Arguments ---
        opt_params = df_mat_params_copy.loc[integrator.inp_mat_keys, :].copy(deep=True)
        opt_args = [self.opt_type, integrator, opt_params]

        # --- Run Optimization ---
        if section_key in config['sections']:
            opt_result = opt_solvers(
                *opt_args,
                miter=config['miter'],
                giter=config['giter'],
                glb=False
            )
        else:
            # Use baseline parameters without optimization
            opt_result = opt_solvers(
                *opt_args,
                miter=config['miter'],
                giter=config['giter'],
                empty=True
            )

        # --- Generate Results and Plot ---
        plot_path = (local_solution_path / section_name / f"{self.opt_type}_{position_code}.png").resolve()

        # Evaluate model and generate plot
        result_dict, optimized_params = self._eval_problem(
            min_lx=max_stretch,
            opt_material=opt_result,
            instron_data=instron_data,
            lsq=cost_function,
            plot_fname=str(plot_path),
            ncontrol=HIGH_RESOLUTION_NCONTROL
        )

        # Add optimized parameters to result
        result_dict["mat_param"] = optimized_params

        return result_dict

    def _optimize_section_positions(self,
                                    rat_id: str,
                                    section_info: pd.DataFrame,
                                    cost_functions: List[CostFunction],
                                    instron_data_list: List[InstronData],
                                    max_stretches: pd.Series,
                                    df_mat_params: pd.DataFrame,
                                    local_solution_path: Path,
                                    config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Optimize all section-position combinations for a rat.

        Iterates through anatomical sections (Ar, Tr, Ab) and their positions
        (A, B, C), running optimization for each valid combination using the
        atomic optimization method.

        Parameters
        ----------
        rat_id : str
            Rat identifier (e.g., 'rato-17') for logging
        section_info : pd.DataFrame
            Boolean DataFrame indicating which section-positions are valid.
            Index: section names (e.g., 'Ar', 'Tr', 'Ab')
            Columns: position codes (e.g., 'Ar-A', 'Tr-B', 'Ab-C')
            Values: True if position should be optimized, False otherwise
        cost_functions : List[CostFunction]
            List of cost function objects, one per section-position.
            Order matches the iteration order of section_info.
        instron_data_list : List[InstronData]
            List of experimental data objects, one per section-position.
            Order matches cost_functions.
        max_stretches : pd.Series
            Maximum stretch ratios (lambda_max) for each section-position.
            Index: section-position codes (e.g., 'Ar-A', 'Tr-B')
        df_mat_params : pd.DataFrame
            Material parameter DataFrame with baseline values
        local_solution_path : Path
            Base path for saving optimization results
        config : Dict[str, Any]
            Optimization configuration from _configure_optimization_parameters()

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Nested dictionary of results:
            - Keys: section-position codes (e.g., 'Ar-A', 'Tr-B')
            - Values: Result dictionaries from _optimize_single_position()
              containing 'pk1', 'cauchy', 'ese', 'lx', 'mat_param'

        Notes
        -----
        Iteration Strategy:
        1. Outer loop: Anatomical sections (rows of section_info)
        2. Inner loop: Positions within each section (columns)
        3. For each valid position (val_k == True):
           - Call _optimize_single_position()
           - Store results in output dictionary

        The section_info DataFrame controls which positions are processed.
        Invalid positions (val_k == False) are skipped automatically.

        Index Correspondence:
        - `ijk` is the linear index into cost_functions and instron_data_list
        - Increments for each column in section_info, row by row
        - Example: For 3 sections x 3 positions, ijk goes 0->8

        Side Effects
        ------------
        - Generates plot files via _optimize_single_position()
        - Updates self.optimal_ds_vars with optimized parameters
        - Logs progress for each position

        Examples
        --------
        >>> results = self._optimize_section_positions(
        ...     rat_id='rato-17',
        ...     section_info=df_sec_info,
        ...     cost_functions=cost_fns,
        ...     instron_data_list=instron_list,
        ...     max_stretches=max_stretch_series,
        ...     df_mat_params=params_df,
        ...     local_solution_path=Path('/results'),
        ...     config=config
        ... )
        >>> len(results)  # Number of optimized positions
        9
        >>> 'Ar-A' in results
        True

        See Also
        --------
        _optimize_single_position : Atomic optimization for one position
        _get_index : Generates section_info DataFrame
        """
        # --- Initialize Results Container ---
        model_results = {}

        # --- Iterate Over Sections and Positions ---
        for ij, (section_name, position_row) in enumerate(section_info.iterrows()):
            for ijk, (position_key, is_valid) in enumerate(position_row.items()):
                if is_valid:
                    # --- Optimize Single Position ---
                    result_dict = self._optimize_single_position(
                        rat_id=rat_id,
                        section_key=position_key,
                        section_name=section_name,
                        position_code=position_key[-1],
                        cost_function=cost_functions[ijk],
                        instron_data=instron_data_list[ijk],
                        max_stretch=max_stretches[position_key].item(),
                        df_mat_params=df_mat_params,
                        local_solution_path=local_solution_path,
                        config=config
                    )

                    # --- Store Results ---
                    model_results[position_key] = result_dict
                    self.optimal_ds_vars[rat_id].loc[position_key, result_dict["mat_param"].index] = result_dict["mat_param"]

        return model_results

    def _save_rat_optimization_results(self,
                                       rat_id: str,
                                       model_results: Dict[str, Dict[str, Any]],
                                       df_mat_params: pd.DataFrame,
                                       local_solution_path: Path,
                                       plot_ese: bool = False) -> None:
        """
        Save and plot optimization results for a single rat.

        Performs post-optimization tasks: saves parameter data to files,
        generates comparison plots (stress, force, energy patterns), and
        stores results in instance variables.

        Parameters
        ----------
        rat_id : str
            Rat identifier (e.g., 'rato-17')
        model_results : Dict[str, Dict[str, Any]]
            Optimization results from _optimize_section_positions()
            Keys: section-position codes
            Values: Result dictionaries with stress/strain data
        df_mat_params : pd.DataFrame
            Material parameter DataFrame with baseline values
        local_solution_path : Path
            Path for saving result files
        plot_ese : bool, optional
            If True, generate strain energy pattern plot, by default False

        Side Effects
        ------------
        - Saves Excel and Parquet files with optimized parameters
        - Generates PNG plot files (stress, force, optionally energy)
        - Updates self.aorta_model_results dictionary
        - Closes matplotlib figures

        Generated Files
        ---------------
        - opt_mat_param_{opt_type}.xlsx : Parameter table
        - opt_mat_param_{opt_type}.parquet.gzip : Compressed data
        - {opt_type}_pk1_stress_*.png : Stress component plots
        - {opt_type}_uniaxial_test.png : Force-stretch curves
        - energy_strain_pattern_{opt_type}.png : Energy plots (if plot_ese=True)

        Notes
        -----
        Data Saving Strategy:
        1. Concatenate baseline parameters with optimized values
        2. Save to both Excel (human-readable) and Parquet (efficient)
        3. Generate comparison plots showing experimental vs fitted data

        Plot Generation:
        - Stress plots: PK1 stress components vs stretch
        - Force plots: Force vs stretch for uniaxial test
        - Energy plots: Strain energy density patterns (optional)

        Examples
        --------
        >>> self._save_rat_optimization_results(
        ...     rat_id='rato-17',
        ...     model_results=results_dict,
        ...     df_mat_params=params_df,
        ...     local_solution_path=Path('/results/rato-17'),
        ...     plot_ese=True
        ... )
        # Files created in /results/rato-17/

        See Also
        --------
        _save_data : Low-level file saving method
        _setup_plot_fit : Prepares data for plotting
        _plot_stress : Generates stress component plots
        exp_test_plot : Generates force-stretch plots
        _plot_ese : Generates energy pattern plots
        """
        import matplotlib.pyplot as plt

        # --- Store Results ---
        if len(model_results) > 0:
            self.aorta_model_results[rat_id] = model_results

        # --- Prepare Combined DataFrame ---
        df_baseline = pd.concat([
            self.baseline_ds_vars.loc[rat_id].to_frame(name='baseline'),
            df_mat_params[["lower", "upper"]]
        ], axis=1)

        df_opt_combined = pd.concat([df_baseline.T, self.optimal_ds_vars[rat_id]], axis=0)

        # --- Save Data Files ---
        self._save_data(local_solution_path, df_opt_combined)

        # --- Generate Plots ---
        plot_data = self._setup_plot_fit({rat_id: model_results})

        # Stress component plots
        self._plot_stress(plot_data, DEFAULT_PLOT_LIMITS, str(self.solution_path[rat_id]))

        # Force-stretch plot
        fig_force = exp_test_plot(plot_data, limits=DEFAULT_PLOT_LIMITS, lines=self.plt_lines)
        fig_force.savefig(f'{local_solution_path}/{self.opt_type}_uniaxial_test.png')
        plt.close(fig_force)

        # --- Optional Energy Pattern Plot ---
        if plot_ese:
            fig_energy = self._plot_ese(model_results, self.optimal_ds_vars[rat_id])
            fig_energy.savefig(f'{local_solution_path}/energy_strain_pattern_{self.opt_type}.png')
            plt.close(fig_energy)

    def _generate_aggregate_plots(self, full_solution: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Generate aggregate plots combining all rats.

        Creates summary plots showing combined results from all optimized rats,
        useful for cross-rat comparison and overall assessment.

        Parameters
        ----------
        full_solution : Dict[str, Dict[str, Dict[str, Any]]]
            Complete optimization results for all rats.
            Structure:
            {
                'rato-17': {
                    'Ar-A': {result_dict},
                    'Tr-B': {result_dict},
                    ...
                },
                'rato-18': {...},
                ...
            }

        Side Effects
        ------------
        - Generates aggregate PNG plot files in main results directory
        - Closes matplotlib figures after saving

        Generated Files
        ---------------
        - {opt_type}_uniaxial_test.png : Combined force-stretch curves
        - {opt_type}_pk1_stress_*.png : Combined stress component plots

        Notes
        -----
        Aggregate plots overlay data from all rats, allowing visual comparison
        of material behavior across specimens. Useful for identifying outliers
        and assessing parameter consistency.

        Examples
        --------
        >>> full_sol = {
        ...     'rato-17': {...},
        ...     'rato-18': {...}
        ... }
        >>> self._generate_aggregate_plots(full_sol)
        # Files created in self.path_main/

        See Also
        --------
        _setup_plot_fit : Prepares combined data for plotting
        exp_test_plot : Generates force-stretch plots
        _plot_stress : Generates stress component plots
        """
        import matplotlib.pyplot as plt

        # Setup combined plot data
        plot_data = self._setup_plot_fit(full_solution)

        # Aggregate force-stretch plot
        fig_force = exp_test_plot(plot_data, limits=DEFAULT_PLOT_LIMITS, lines=self.plt_lines)
        fig_force.savefig(f'{self.path_main}/{self.opt_type}_uniaxial_test.png')
        plt.close(fig_force)

        # Aggregate stress plots
        self._plot_stress(plot_data, DEFAULT_PLOT_LIMITS, str(self.path_main))

    def find_optimal_parameters(self,
                                miter: int = None,
                                plot: bool = False,
                                ftype: str = None,
                                sections: List[str] = None,
                                local_path: str = None,
                                **kwargs
                                ) -> pd.DataFrame:
        """
        Find optimal section-specific material parameters.

        Performs local optimization for each anatomical section and position,
        allowing parameters to vary spatially along the aorta. This captures
        regional heterogeneity in material properties.

        Parameters
        ----------
        miter : int, optional
            Maximum iterations for local optimization.
            If None, uses DEFAULT_LOCAL_OPTIMIZATION_ITERATIONS (100).
        plot : bool, optional
            If True, generate strain energy pattern plots, by default False
        ftype : str, optional
            Cost function type for optimization:
            - 'lsq': Standard least squares
            - 'lsq_sum': Sum of least squares across sections
            - 'cauchy_robust': Robust Cauchy loss
            If None, uses self._local_ftype
        sections : List[str], optional
            List of section-position keys to optimize (e.g., ['Ar-A', 'Tr-B']).
            If None, optimizes all sections: Ar/Tr/Ab x A/B/C = 9 combinations
        local_path : str, optional
            Subdirectory within solution path for saving results.
            If None, saves directly to solution path
        **kwargs : dict
            Additional optimization parameters:
            - c (float):        Robustness parameter for robust loss functions
            - rho (float):      KS aggregation parameter
            - alpha (float):    Tikhonov regularization parameter
            - beta (float):     Re-scaling regularization parameter
            - rescale (str):    Rescaling method ('std', 'minmax', None)
            - epsilon (float):  Volumetric regularization parameter
            - giter (int):      Global iterations (basin-hopping)
            - dvol (bool):      Include volumetric stabilization

        Returns
        -------
        pd.DataFrame
            Empty DataFrame (optimization results stored in self.optimal_ds_vars)

        Notes
        -----
        Optimization Strategy:
        1. Uses baseline parameters as initial guess for each section
        2. Optimizes each section-position independently
        3. Generates comparison plots (experimental vs fitted)
        4. Saves results (Excel + Parquet formats)

        The method processes sections in order: Ar -> Tr -> Ab, with positions
        A, B, C for each. Only sections in the `sections` list are optimized;
        others use baseline parameter values.

        Generated Files (per rat):
        - opt_mat_param_{opt_type}.xlsx: Parameter table
        - opt_mat_param_{opt_type}.parquet.gzip: Compressed data
        - {opt_type}_pk1_stress_{section}.png: Stress plots
        - {opt_type}_uniaxial_test.png: Force-stretch plots

        See Also
        --------
        find_baseline_parameters : Global parameter optimization
        _eval_problem : Evaluate and plot single optimization
        _save_data : Save optimization results

        Examples
        --------
        Optimize all sections:

        >>> fit = AnisoMaterialFit(selection, opt_type='ipopt')
        >>> fit.exp_test_eval()
        >>> fit.find_baseline_parameters()  # Get initial guess
        >>> fit.find_optimal_parameters(miter=100, plot=True)

        Optimize specific sections only:

        >>> sections_to_fit = ['Ar-A', 'Ar-B', 'Tr-A']
        >>> fit.find_optimal_parameters(sections=sections_to_fit, miter=50)
        """

        # --- Configure Optimization Parameters ---
        config = self._configure_optimization_parameters(miter, ftype, sections, **kwargs)

        # --- Initialize Output Variables ---
        df_opt_ds_vars_cfg_i = pd.DataFrame()
        df_mat_params = self.ds_vars[["values", "lower", "upper", "limit", "variable"]].copy()
        full_solution = {}

        # --- Optimize Each Rat ---
        for key_i, val_i in self.model_opt_res.items():
            df_mat_params_i = df_mat_params.copy(deep=True)
            df_mat_params_i.loc[:, "values"] = self.baseline_ds_vars.loc[key_i, :]

            # --- Setup Paths ---
            local_solution_path_i = self._get_local_solution_path(key_i, local_path)

            df_sec_info_i = self._get_index(val_i)

            # --- Prepare Cost Functions ---
            list_lsq_i, list_instron_i = self._prepare_cost_and_data(val_i, df_mat_params_i)

            min_mlx_i = self.optimal_ds_vars[key_i].loc[:, "mlx"]

            # --- Optimize Section Positions ---
            model_res_ij = self._optimize_section_positions(
                rat_id=key_i,
                section_info=df_sec_info_i,
                cost_functions=list_lsq_i,
                instron_data_list=list_instron_i,
                max_stretches=min_mlx_i,
                df_mat_params=df_mat_params_i,
                local_solution_path=local_solution_path_i,
                config=config
            )

            # --- Save Results ---
            self._save_rat_optimization_results(
                rat_id=key_i,
                model_results=model_res_ij,
                df_mat_params=df_mat_params_i,
                local_solution_path=local_solution_path_i,
                plot_ese=plot
            )

            full_solution[key_i] = model_res_ij

        # --- Generate Aggregate Plots ---
        self._generate_aggregate_plots(full_solution)

        return df_opt_ds_vars_cfg_i