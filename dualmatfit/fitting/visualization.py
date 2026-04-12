# -*- coding: utf-8 -*-
"""
Visualization mixin for AnisoMaterialFit.

Provides plotting and visualization methods for material fitting results.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dualmatfit.plotting.experimental_visuals import exp_test_plot, stress_plot
from dualmatfit.plotting.parameters import NAME_SECTIONS
from dualmatfit.fitting.constants import (
    DEFAULT_PLOT_LIMITS,
    UNSTRETCHED_STATE,
    HIGH_RESOLUTION_NCONTROL,
)

from dualmatfit.utils.logging_config import get_logger
logger = get_logger('fitting.visualization')

__all__ = ['FitVisualizationMixin']


class FitVisualizationMixin:
    """Visualization methods for AnisoMaterialFit."""

    def _setup_plot_fit(self, model_fit: Dict[str, Any]) -> Dict[str, Dict[str, List[Any]]]:
        """
        Organize model results for plotting by anatomical section.

        Groups model solutions and experimental data by section (Ar, Tr, Ab) to
        facilitate section-wise comparison plots.

        Parameters
        ----------
        model_fit : Dict[str, Any]
            Dictionary mapping rat IDs to their model results.
            Structure: {rat_id: {section_key: solution_dict}}

        Returns
        -------
        Dict[str, Dict[str, List[Any]]]
            Organized plot data with structure:
            {
                'Ar': {'data': [...], 'test': [...], 'name': [...]},
                'Tr': {'data': [...], 'test': [...], 'name': [...]},
                'Ab': {'data': [...], 'test': [...], 'name': [...]}
            }
            where:
            - 'data': List of model solution dictionaries
            - 'test': List of InstronData objects (experimental)
            - 'name': List of [rat_id, section_key] identifiers

        Notes
        -----
        This organization allows plotting all data from a specific anatomical
        section together, making it easy to compare:
        - Different specimens from the same section
        - Different positions (A, B, C) within a section
        - Model predictions vs experimental data

        The grouping is based on section prefix:
        - 'Ar': Ascending aorta (root)
        - 'Tr': Transverse arch
        - 'Ab': Abdominal aorta

        See Also
        --------
        _plot_stress : Generate stress plots for organized data
        stress_plot : Plotting function for stress components
        """

        plot_solution = {key_i: {"data": [], "test": [], "name": []} for key_i in ['Ar', 'Tr', 'Ab']}

        for seg_k in ['Ar', 'Tr', 'Ab']:
            for key_i, model_i in model_fit.items():
                for sec_j, solu_j in model_i.items():
                    if seg_k in sec_j:
                        plot_solution[seg_k]["name"].append([key_i, sec_j])
                        plot_solution[seg_k]["data"].append(solu_j)
                        plot_solution[seg_k]["test"].append(self.model_opt_res[key_i][sec_j]["instron"])

        return plot_solution

    def _plot_stress(self,
                     plot_solution: Dict[str, Dict[str, List[Any]]],
                     plot_limits: Dict[str, List[float]],
                     plot_path: str,
                     ) -> None:
        """
        Generate and save stress component plots for each anatomical section.

        Creates separate plots showing PK1 stress components (isotropic,
        anisotropic, total) for each section and saves them to disk.

        Parameters
        ----------
        plot_solution : Dict[str, Dict[str, List[Any]]]
            Organized plot data from _setup_plot_fit(), grouped by section
        plot_limits : Dict[str, List[float]]
            Axis limits for stress plots with keys:
            - 'iso': [min, max] for isotropic stress [MPa]
            - 'ani': [min, max] for anisotropic stress [MPa]
            - 'sum': [min, max] for total stress [MPa]
            - 'lx': [min, max] for stretch ratio
        plot_path : str
            Directory path for saving plot files

        Returns
        -------
        None
            Plots are saved as PNG files:
            - {plot_path}/{opt_type}_pk1_stress_Ar.png
            - {plot_path}/{opt_type}_pk1_stress_Tr.png
            - {plot_path}/{opt_type}_pk1_stress_Ab.png

        Notes
        -----
        Each plot shows:
        - Experimental data points (markers)
        - Model predictions (lines with different styles for positions A/B/C)
        - Three subplots: isotropic, anisotropic, and total stress
        - Stretch ratio on x-axis, stress on y-axis

        Line styles are defined in self.plt_lines:
        - 'A': solid line
        - 'B': dashed line
        - 'C': dash-dot line

        See Also
        --------
        stress_plot : Core plotting function
        NAME_SECTIONS : Dictionary mapping section codes to full names
        """

        for seg_k, solution_k in plot_solution.items():
            fig_stress_k = stress_plot(solution_k,
                                       limits=DEFAULT_PLOT_LIMITS,
                                       ptitle=f'{NAME_SECTIONS[seg_k]} Segments',
                                       lines=self.plt_lines)

            plot_fname_k = f'{plot_path}/{self.opt_type}_pk1_stress_{seg_k}.png'
            fig_stress_k.savefig(plot_fname_k)
            plt.close(fig_stress_k)

    def plot_fit(self,
                 global_opt: bool = False,
                 plot_path: Optional[str] = None,
                 ):
        """
        Generate plots comparing experimental data with model predictions.

        Creates comprehensive visualization of the fitting results including:
        - Stress-stretch curves for each anatomical section
        - Experimental data points vs. model predictions
        - Section-wise and combined plots

        Parameters
        ----------
        global_opt : bool, default=False
            If True, generates plots using baseline (global) parameters.
            If False, uses section-specific optimized parameters.

        plot_path : str, optional
            Custom directory path for saving plots.
            If None, uses self.path_main.
            Directory is created if it doesn't exist.

        Notes
        -----
        Generated Plots
        ---------------
        For each anatomical section (Ar, Tr, Ab):

        1. **PK1 Stress Plots**: ``{opt_type}_pk1_stress_{section}.png``
           - Shows First Piola-Kirchhoff stress components
           - Includes isotropic, anisotropic, and total contributions

        2. **Force-Stretch Plots**: ``{opt_type}_uniaxial_test.png``
           - Compares experimental force data with model predictions
           - One subplot per anatomical section
           - Includes all specimen positions (A, B, C)

        3. **Global Fit Plots** (if global_opt=True):
           - Shows quality of baseline parameter fit across all sections
           - Saved as ``{rat_id}_{opt_type}_uniaxial_glb_test.png``

        Plot Features
        -------------
        - Experimental data: Markers/points
        - Model predictions: Solid lines
        - Different colors for different specimens
        - Anatomical section labels
        - Stress units: MPa or kPa
        - Stretch units: mm/mm (dimensionless)

        Examples
        --------
        Plot with section-specific parameters:

        >>> fit = AnisoMaterialFit(selection)
        >>> fit.find_optimal_parameters()
        >>> fit.plot_fit(global_opt=False)

        Plot with baseline parameters:

        >>> fit.find_baseline_parameters()
        >>> fit.plot_fit(global_opt=True)

        Custom output location:

        >>> fit.plot_fit(plot_path='/path/to/plots', global_opt=False)

        See Also
        --------
        _plot_stress : Generate stress component plots
        _setup_plot_fit : Prepare plot data structure

        References
        ----------
        .. [1] NAME_SECTIONS dict maps section codes to full names:
               'Ar' -> 'AoA' (Ascending Aorta)
               'Tr' -> 'DTAo' (Descending Thoracic Aorta)
               'Ab' -> 'DAAo' (Descending Abdominal Aorta)
        """

        # --- Evaluate Global Model ---
        if global_opt:
            for rat_i, sec_info_i in self.model_opt_res.items():
                # --- Setup Output Path ---
                if plot_path is not None:
                    fig_path_loc_i = Path(plot_path) / f'{rat_i}_{self.opt_type}_uniaxial_glb_test.png'
                else:
                    fig_path_loc_i = list(self.solution_path.values())[-1] / f'{self.opt_type}_uniaxial_glb_test.png'

                self.path_manager.ensure_parent_dir(fig_path_loc_i)
                ds_baseline_vars_i = self.baseline_ds_vars.loc[rat_i, :]

                # --- Compute Model Solutions ---
                if self.aorta_model_results.get(rat_i) is None:
                    ext_solu_j = {}
                else:
                    ext_solu_j = self.aorta_model_results[rat_i]

                for sec_j, solu_j in sec_info_i.items():
                    np_lx_j = np.linspace(UNSTRETCHED_STATE, self.optimal_ds_vars[rat_i].loc[sec_j, "mlx"], num=HIGH_RESOLUTION_NCONTROL)
                    ext_solu_j[sec_j] = solu_j['ext_solution'].solve(ds_baseline_vars_i, np_lx_j)

                self.aorta_model_results[rat_i] = ext_solu_j

                # --- Generate Global Fit Plot ---
                plot_solu_i = self._setup_plot_fit({rat_i: ext_solu_j})
                try:
                    fig_force_i = exp_test_plot(plot_solu_i, limits=DEFAULT_PLOT_LIMITS, lines=self.plt_lines)
                    fig_force_i.savefig(fig_path_loc_i)
                    plt.close(fig_force_i)
                except (ValueError, TypeError, IOError, OSError) as e:
                    logger.info(f" An error occurred while plotting the global fit for {rat_i}: {e}")
                    continue

        # --- Setup Plot Output ---
        fig_name = f'{self.opt_type}_uniaxial_test.png'

        if plot_path is None:
            plot_path = self.path_main

        plot_path = Path(plot_path)
        fig_path_loc = plot_path / fig_name
        plot_solution = self._setup_plot_fit(self.aorta_model_results)

        # --- Generate Stress Component Plots ---
        self._plot_stress(plot_solution, DEFAULT_PLOT_LIMITS, str(plot_path))

        # --- Generate Force-Stretch Plot ---
        fig_force = exp_test_plot(plot_solution, limits=DEFAULT_PLOT_LIMITS, lines=self.plt_lines)
        fig_force.savefig(fig_path_loc)
        plt.close(fig_force)

    def correlation(self, plot: bool = True) -> None:
        """
        Compute parameter correlations (not yet implemented).

        This method will analyze correlations between optimized material
        parameters across different sections and specimens.

        Parameters
        ----------
        plot : bool, optional
            If True, generate correlation plots, by default True

        Raises
        ------
        NotImplementedError
            This method is not yet implemented

        Notes
        -----
        Planned functionality:
        - Compute pairwise correlation matrix for material parameters
        - Identify highly correlated parameters (redundancy)
        - Generate correlation heatmaps
        - Analyze parameter sensitivity and identifiability

        This analysis would be useful for:
        - Understanding parameter coupling
        - Reducing model complexity
        - Improving optimization strategies
        - Validating parameter uniqueness
        """
        raise NotImplementedError