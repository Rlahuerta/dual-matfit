# -*- coding: utf-8 -*-
"""
Persistence mixin for AnisoMaterialFit.

Provides methods for saving and loading optimization results.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import numpy as np
import pandas as pd

from dualmatfit.utils.io_utils import MaterialFitIO, load_parquet_results
from dualmatfit.utils.path_manager import PathLike
from dualmatfit.fitting.covariance import save_covariance_report
from dualmatfit.fitting.constants import (
    UNSTRETCHED_STATE,
    HIGH_RESOLUTION_NCONTROL,
)

from dualmatfit.utils.logging_config import get_logger
logger = get_logger('fitting.persistence')

__all__ = ['FitPersistenceMixin']


class FitPersistenceMixin:
    """Persistence methods for AnisoMaterialFit."""

    def _save_data(self, file_path: PathLike, dsvars: pd.DataFrame) -> None:
        """
        Save material parameters to multiple file formats.

        Exports optimization results in both Excel and compressed Parquet formats
        for different use cases (human-readable vs efficient storage).

        Parameters
        ----------
        file_path : PathLike
            Base directory path for saving files (str or Path)
        dsvars : pd.DataFrame
            DataFrame containing material parameters with columns:
            - Parameter names as index
            - 'Baseline': Baseline parameter values
            - Section-specific values (e.g., 'Ar-A', 'Tr-B')
            - 'lower', 'upper': Parameter bounds

        Returns
        -------
        None
            Files are saved to disk:
            - {file_path}/opt_mat_param_{opt_type}.xlsx
            - {file_path}/opt_mat_param_{opt_type}.parquet.gzip

        Notes
        -----
        File Formats:
        - Excel (.xlsx): Human-readable, good for inspection and sharing
        - Parquet (.gzip): Compressed binary, efficient for large datasets

        The directory is created automatically if it doesn't exist.

        Future Enhancement:
        - TODO: Add joblib format for complete solution objects

        See Also
        --------
        save_data : Public method for saving all results
        load_results : Load previously saved results
        """
        # Delegate to IO handler
        self.io_handler.save_optimization_results(dsvars, file_path)

    def save_data(self):
        """
        Save optimization results to disk.

        Saves all material parameters, optimization results, and model predictions
        to the configured output paths. Data is saved in Parquet format for
        efficient storage and loading.

        File Structure
        --------------
        For each rat ID, creates:
        - ``{rat_id}/optimal_dsvars.parquet``: Section-specific optimized parameters
        - ``{rat_id}/baseline_dsvars.parquet``: Baseline parameters across sections
        - Model results cached for quick loading

        Notes
        -----
        This method should be called after optimization to persist results.
        The saved data can be reloaded using load_results().

        Saved data includes:
        - Optimized material parameters (mu, D, k1, k2, alpha, kappa)
        - Parameter bounds (lower/upper)
        - Maximum stretch values for each section
        - Optimization metadata (iterations, convergence, cost)

        Examples
        --------
        After parameter optimization:

        >>> fit = AnisoMaterialFit(selection)
        >>> df_baseline = fit.find_baseline_parameters()
        >>> fit.find_optimal_parameters()
        >>> fit.save_data()  # Save all results

        See Also
        --------
        load_results : Load previously saved results
        _save_data : Internal method for saving individual files
        """

        for key_i in self.model_opt_res.keys():
            local_path_i = f'{self.solution_path[key_i]}'
            pd_baseline_dsvars_i = self.baseline_ds_vars.loc[key_i].to_frame(name='baseline')
            pd_opt_dsvars_i = pd.concat([pd_baseline_dsvars_i.T, self.optimal_ds_vars[key_i]], axis=0)

            self._save_data(local_path_i, pd_opt_dsvars_i)

            # Persist covariance reports if available
            self._save_covariance_for_rat(key_i, local_path_i)

    def _save_covariance_for_rat(self, rat_id: str, local_path: str) -> None:
        """Save covariance reports for one rat (if previously computed)."""
        if not hasattr(self, 'covariance_reports'):
            return

        out_dir = Path(local_path)

        # Baseline covariance
        baseline_reports = self.covariance_reports.get('baseline', {})
        if rat_id in baseline_reports:
            save_covariance_report(
                baseline_reports[rat_id],
                out_dir / f'covariance_baseline_{self.opt_type}.npz',
            )
            logger.info("Saved baseline covariance for %s", rat_id)

        # Optimal covariance (per section)
        optimal_reports = self.covariance_reports.get('optimal', {})
        if rat_id in optimal_reports:
            for sec_pos, report in optimal_reports[rat_id].items():
                save_covariance_report(
                    report,
                    out_dir / f'covariance_{sec_pos}_{self.opt_type}.npz',
                )
            logger.info("Saved optimal covariance for %s", rat_id)

    def load_results(self, run: bool = False):
        """
        Load previously saved optimization results from disk.

        Restores material parameters, optimization results, and optionally
        recomputes model solutions from saved parameters.

        Parameters
        ----------
        run : bool, default=False
            If True, recomputes model solutions using loaded parameters.
            If False, only loads parameters without recomputing solutions.

        Notes
        -----
        This method loads data saved by save_data(). It's useful for:
        - Resuming interrupted workflows
        - Generating plots from previous optimizations
        - Comparing results across different runs
        - Avoiding re-running expensive optimizations

        The method automatically detects available saved files and loads:
        - Optimal design variables (section-specific parameters)
        - Baseline design variables (average parameters)
        - Maximum stretch values for each section

        If run=True, also computes:
        - Stress-stretch curves for all sections
        - Model predictions at experimental stretch points
        - Comparison with experimental data

        Examples
        --------
        Load results without recomputing:

        >>> fit = AnisoMaterialFit(selection)
        >>> fit.load_results(run=False)
        >>> # Can now plot saved results
        >>> fit.plot_fit()

        Load and recompute solutions:

        >>> fit = AnisoMaterialFit(selection)
        >>> fit.load_results(run=True)
        >>> # Solutions recomputed with loaded parameters
        >>> fit.plot_fit()

        Raises
        ------
        FileNotFoundError
            If no saved results are found in the expected paths

        See Also
        --------
        save_data : Save optimization results
        """

        # --- Initialize Index List ---
        index_ds_vars = self.ds_vars.index.to_list()

        # --- Iterate Over Saved Results ---
        for rat_i, sec_info_i in self.path_local_solution.items():
            # --- Find Data Directory ---
            path_list = [Path(path_ik) for path_ik in sec_info_i.values()]
            if path_list:
                common_dir = path_list[0].parent
            else:
                continue

            joblib_file_i = common_dir / f"opt_xyz_mat_param_{self.opt_type}.joblib"
            parquet_file_i = common_dir / f"opt_mat_param_{self.opt_type}.parquet.gzip"

            # --- Load From File ---
            if joblib_file_i.is_file():
                logger.info(f" Not implemented load feature for joblib file: {str(joblib_file_i)}")

            elif parquet_file_i.is_file():
                # Use centralized loading function
                df_optimal_i = load_parquet_results(parquet_file_i, index_ds_vars)

                if df_optimal_i is None:
                    sr_optimal_baseline_i = self.ds_vars.loc[:, "ini"]
                    continue

                # --- Extract Baseline Parameters ---
                if 'mean' in df_optimal_i.index:
                    sr_optimal_baseline_i = df_optimal_i.loc["mean", index_ds_vars]
                    sr_optimal_baseline_i.name = "baseline"

                elif 'baseline' in df_optimal_i.index:
                    sr_optimal_baseline_i = df_optimal_i.loc["baseline", index_ds_vars]

                else:
                    sr_optimal_baseline_i = self.ds_vars["values"].copy()
                    sr_optimal_baseline_i.name = "baseline"
                    logger.warning(f"Baseline row not found in {str(parquet_file_i)}! Using 'ini' values.")

                # --- Update Instance State ---
                df_optimal_i = df_optimal_i[df_optimal_i.columns.intersection(self.optimal_ds_vars[rat_i].columns)]
                df_optimal_i = df_optimal_i.loc[df_optimal_i.index.intersection(self.optimal_ds_vars[rat_i].index)]

                self.optimal_ds_vars[rat_i].update(df_optimal_i)
                self.baseline_ds_vars.loc[rat_i, :] = sr_optimal_baseline_i

            else:
                sr_optimal_baseline_i = self.ds_vars.loc[:, "ini"]

            # --- Optionally Recompute Solutions ---
            model_res_i = {}

            for sec_key_k, models_k in self.model_opt_res[rat_i].items():
                if run:
                    mlx_k = self.optimal_ds_vars[rat_i].loc[sec_key_k, "mlx"]
                    np_lx = np.linspace(UNSTRETCHED_STATE, mlx_k, num=HIGH_RESOLUTION_NCONTROL)
                    model_res_i[sec_key_k] = models_k['lsq'].solve(sr_optimal_baseline_i, stretch_x=np_lx)

                self.aorta_model_results[rat_i] = model_res_i

        logger.info(" Previously saved results have been loaded!")