# -*- coding: utf-8 -*-
"""
Experimental data visualization functions.

This module provides plotting functions for visualizing experimental
data from Instron tests and material fitting results.
"""
# import os
import re
import numpy as np
import pandas as pd
import sympy as sy
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pathlib import Path
from scipy.optimize import OptimizeResult
from typing import Optional, Union, Dict, Tuple

from dualmatfit.rato_info import excel_data
from dualmatfit.experimental import InstronData
from dualmatfit.plotting.plot_helpers import PlotHelper
from dualmatfit.plotting.parameters import COLORS

from dualmatfit.logging_config import get_logger
logger = get_logger('plotting')

__all__ = [
    'plot_raw_signals',
    'plot_material_fit',
    'article_post',
]



def plot_raw_signals(instron_data: InstronData,
                     save_dir: str,
                     filename_prefix: Optional[str] = None,
                     ylim_load: Optional[Tuple[float, float]] = None,
                     xlim_time: Optional[Tuple[float, float]] = None,
                     xlim_ext: Optional[Tuple[float, float]] = None,
                     plot_cfg: Dict[str, int] = None,
                     plot_figsize: Optional[Tuple[float, float]] = None,
                     sharex: bool = False,
                     sharey: bool = False,
                     # plot_control_region: bool = False,
                     plot_title: Optional[str] = None,
                     ):
    """
    Plots the raw Time-Extension, Time-Load, and Extension-Load signals.
    If distinct control points were generated (instron_data.ncontrol > 0),
    they are overlaid.

    Args:
        instron_data (InstronData): Object containing processed experimental data.
        save_dir (str): Directory to save the plot.
        filename_prefix (str, optional): Prefix for the output plot filename.
                                         If None, a sanitized sample_id is used.
        ylim_load (Optional[Tuple[float, float]]): Optional y-axis limits for load plots.
        xlim_time (Optional[Tuple[float, float]]): Optional x-axis limits for time-based plots.
        xlim_ext (Optional[Tuple[float, float]]): Optional x-axis limits for extension-based plots.
        plot_cfg (Dict): Configuration for subplots (nrows, ncols).
        plot_figsize (Optional[Tuple[float, float]]): Figure size. Defaults to (18, 5.5).
        sharex (bool): Whether to share x-axes among subplots.
        sharey (bool): Whether to share y-axes among subplots.
        plot_title (Optional[str]): Custom title for the entire figure.
                                    Defaults to 'Raw Experimental Data - [sample_id]'.
    """
    if plot_cfg is None:
        plot_cfg = {"nrows": 1, "ncols": 3}

    if instron_data is None:
        logger.error(" InstronData object is None. Cannot plot raw signals.")
        return

    if plot_figsize is None:
        plot_figsize = (18., 6.)

    np_time, np_extn, np_load = instron_data.get_raw_data()
    sample_id = instron_data.get_sample_id()

    if np_time.size < 2 or np_extn.size < 2 or np_load.size < 2:
        logger.debug(f"Warning: Insufficient raw data for {sample_id}. Skipping raw signal plot.")
        return

    if plot_title is None:
        plot_title = f'Raw Instron Data - {sample_id}'

    plot_helper = PlotHelper(use_latex=False)

    fig, axes = plot_helper.setup_figure(**plot_cfg, figsize=plot_figsize, sharex=sharex, sharey=sharey)

    if not isinstance(axes, np.ndarray) or axes.ndim == 0: # Ensure axes is iterable
        axes = np.array([axes])
    axes = axes.flatten() # Ensure 1D array for easy indexing if ncols > 1

    fig.suptitle(plot_title, fontsize=16)
    idx_limit = instron_data.get_plot_limit_index() # Get dynamic limit

    plot_control_overlay = False
    np_tinc, np_textn, np_tload = None, None, None # Initialize to None

    if instron_data.ncontrol > 0:
        np_tinc_cand, np_textn_cand, np_tload_cand = instron_data.get_control_points()
        # A simple check: if number of control points is different from raw points,
        # or if they exist and are not identical (which would be the case if ncontrol=0 led to mirroring)
        if np_tinc_cand is not None and \
           (len(np_tinc_cand) != len(np_time) or not np.array_equal(np_tinc_cand, np_time)):
            np_tinc, np_textn, np_tload = np_tinc_cand, np_textn_cand, np_tload_cand
            plot_control_overlay = True

    # --- Plot 1: Time vs Extension ---
    if len(axes) > 0:
        ax = axes[0]
        ax.plot(np_time[:idx_limit], np_extn[:idx_limit], label='Experimental Data', color=COLORS[0], linewidth=1.5)
        if plot_control_overlay and np_tinc is not None and np_textn is not None:
            ax.plot(np_tinc, np_textn, 'o', color=COLORS[1], markersize=5, label='Control Points', alpha=0.7)
        plot_helper.set_labels_title(ax, xlabel='Time [s]', ylabel='Extension [mm]', title='Time vs Extension')
        plot_helper.set_limits_ticks(ax, xdata=np_time[:idx_limit], ydata=np_extn[:idx_limit], xlims=xlim_time)
        ax.legend()

    # --- Plot 2: Time vs Load ---
    if len(axes) > 1:
        ax = axes[1]
        ax.plot(np_time[:idx_limit], np_load[:idx_limit], label='Experimental Data', color=COLORS[0], linewidth=1.5)
        if plot_control_overlay and np_tinc is not None and np_tload is not None:
            ax.plot(np_tinc, np_tload, 'o', color=COLORS[1], markersize=5, label='Control Points', alpha=0.7)
        plot_helper.set_labels_title(ax, xlabel='Time [s]', ylabel='Load [N]', title='Time vs Load')
        plot_helper.set_limits_ticks(ax, xdata=np_time[:idx_limit], ydata=np_load[:idx_limit], xlims=xlim_time, ylims=ylim_load)
        ax.legend()

    # --- Plot 3: Extension vs Load ---
    if len(axes) > 2:
        ax = axes[2]
        ax.plot(np_extn[:idx_limit], np_load[:idx_limit], label='Experimental Data', color=COLORS[0], linewidth=1.5)
        if plot_control_overlay and np_textn is not None and np_tload is not None:
            ax.plot(np_textn, np_tload, 'o', color=COLORS[1], markersize=5, label='Control Points', alpha=0.7)
        plot_helper.set_labels_title(ax, xlabel='Extension [mm]', ylabel='Load [N]', title='Extension vs Load')
        plot_helper.set_limits_ticks(ax, xdata=np_extn[:idx_limit], ydata=np_load[:idx_limit], xlims=xlim_ext, ylims=ylim_load)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])  # Adjusted rect for suptitle

    if filename_prefix is None:
        base_name_for_file = sample_id.replace('-', '_')  # Use sample_id if no prefix
    else:
        base_name_for_file = filename_prefix

    safe_filename_base = "".join([c if c.isalnum() or c in ['_', '.'] else '_' for c in base_name_for_file])
    filename = f"{safe_filename_base}_raw_signals.png"  # Add a standard suffix

    plot_helper.save_plot(fig, filename, save_dir)


def plot_material_fit(instron_data: InstronData,
                      solution: OptimizeResult,
                      save_dir: str,
                      figsize: Optional[Tuple[float, float]] = (20, 20),
                      filename_prefix: str = None,
                      ):
    """
    Plots the comparison between experimental data and the fitted model results.
    Includes Reaction Force, Volume Change (det(F)), and PK1 Stress plots.

    Args:
        instron_data (InstronData): Object containing processed experimental data.
        solution (Dict): Dictionary containing model evaluation results (stretch, stress, fint, detF, x_mat).
                         Expected keys: 'stretch', 'stress', 'fint', 'detF', 'x_mat'.
        save_dir (str): Directory to save the plot.
        figsize:
        filename_prefix (str): Prefix for the output plot filename.
    """
    if instron_data is None:
        logger.error(" InstronData object is None. Cannot plot material fit.")
        return

    if solution is None or not all(k in solution for k in ['stretch', 'stress', 'fint', 'detF', 'x_mat']):
        logger.error(" Solution dictionary is missing required keys ('stretch', 'stress', 'fint', 'detF', 'x_mat'). "
              "Cannot plot material fit.")
        return

    sample_id = instron_data.get_sample_id()

    # Get high-resolution experimental data for comparison
    np_lx_hg, np_force_hg, np_pk1_hg = instron_data.get_high_res_data_relative()

    # Get control points for overlay
    np_tstretch_ref, np_tload_ref, np_tpk1_ref = instron_data.get_control_data_relative()

    if np_lx_hg is None or np_force_hg is None or np_pk1_hg is None:
        logger.debug(f"Warning: High-resolution relative data not available for {sample_id}. Skipping material fit plot.")
        return

    # Model results
    np_stretch_x = solution['stretch'][:, 0]            # Assuming x-stretch is the primary one
    np_force_x = solution.fint[:, 0]                    # Assuming fint[:, 0] is the relevant force component
    np_pk1_iso_x = solution['stress']['iso'][:, 0]
    np_pk1_vol_x = solution['stress']['vol'][:, 0]
    np_pk1_ani_x = solution['stress']['ani'][:, 0]
    model_pk1_tot = solution['stress']['full'][:, 0]
    np_detF = solution['detF']
    # mat_params = solution.get('x_mat', {})

    np_tload_ring_ref = 2. * np_tload_ref
    np_force_ring_ref = 2. * np_force_x

    # Generate plot configurations
    max_lx = np.around(np_tstretch_ref.max(), 1) + 0.1
    max_force_ref = max(np.around(1.3 * np_tload_ring_ref.max(), 1).item(), 0.1)
    max_force_md = max(np.around(1.3 * np_force_ring_ref.max(), 1).item(), 0.1)

    x_major_ticks = np.arange(1., max_lx, 0.05)
    x_minor_ticks = np.arange(1., max_lx, 0.01)

    nymajor_div = np.around(max(max_force_ref, max_force_md) / 10., 1)

    if nymajor_div > 0.:
        y_major_load_ticks = np.arange(0, max(max_force_ref, max_force_md), nymajor_div)
        y_minor_load_ticks = np.arange(0, max(max_force_ref, max_force_md), nymajor_div / 4)

    else:
        nymajor_div = np.around(max(max_force_ref, max_force_md) / 10., 2)
        y_major_load_ticks = np.arange(0, max(max_force_ref, max_force_md), nymajor_div)
        y_minor_load_ticks = np.arange(0, max(max_force_ref, max_force_md), nymajor_div / 4)

    plot_helper = PlotHelper(use_latex=False)

    fig, axes = plot_helper.setup_figure(nrows=3, ncols=1, figsize=figsize, sharex=True, dpi=200)
    fig.suptitle(f'Material Fit Comparison - {sample_id}', fontsize=16)

    # --- Plot 1: Reaction Force ---
    axes[0].plot(np_lx_hg, np_force_hg, label='Instron')

    if np_tstretch_ref is not None and np_tload_ref is not None:
        axes[0].plot(np_tstretch_ref, np_tload_ring_ref, 'o', color='red', markersize=8, label='Control Points')

    lx_max_idx = np.where(np_stretch_x <= 1.02 * np_tstretch_ref.max())[0]

    axes[0].plot(np_stretch_x[lx_max_idx], np_force_ring_ref[lx_max_idx], linestyle='--', color='black', label='Model Prediction')
    axes[0].axvline(1.0, color='red', linestyle=':', linewidth=1)       # zero deformation
    axes[0].legend()

    plot_helper.set_labels_title(axes[0], ylabel='Force [N]')
    plot_helper.set_axis_ticks(axes[0],
                               x_major_ticks,
                               x_minor_ticks,
                               y_major_load_ticks,
                               y_minor_load_ticks)

    # --- Plot 2: Volume Change ---
    axes[1].plot(np_stretch_x[lx_max_idx], solution['stretch'][:, 0][lx_max_idx], '--', label=r'$\lambda_x$',color=COLORS[1])
    axes[1].plot(np_stretch_x[lx_max_idx], solution['stretch'][:, 1][lx_max_idx], '--', label=r'$\lambda_y$', color=COLORS[2])
    axes[1].plot(np_stretch_x[lx_max_idx], solution['stretch'][:, 2][lx_max_idx], '--', label=r'$\lambda_z$', color=COLORS[3])

    detF_eq_label = r'$J =\lambda_x \lambda_y \lambda_z = det(F)$'
    axes[1].plot(np_stretch_x[lx_max_idx], np_detF[lx_max_idx], label=detF_eq_label, color='black', linewidth=1.5)
    axes[1].yaxis.set_minor_locator(mticker.AutoMinorLocator())
    plot_helper.set_labels_title(axes[1], ylabel='Stretch / Volume Ratio [-]')
    plot_helper.set_limits_ticks(axes[1], ylims=(0.2, 3.))              # Base limits on J

    axes[1].axvline(1.0, color='red', linestyle=':', linewidth=1)       # zero deformation
    axes[1].axhline(1.0, color='grey', linestyle=':', linewidth=1)      # Line at J=1
    axes[1].legend()

    # --- Plot 3: PK1 Stress ---
    axes[2].plot(np_lx_hg, np_pk1_hg, label='$P_{11}$ Instron')

    if np_tstretch_ref is not None and np_tpk1_ref is not None:
        axes[2].plot(np_tstretch_ref, 2. * np_tpk1_ref, 'o', color='red', markersize=8, label='Control Points')

    # Plot model components if desired
    axes[2].plot(np_stretch_x[lx_max_idx], 2 * np_pk1_iso_x[lx_max_idx], '--', label='$P_{11}$ iso')
    axes[2].plot(np_stretch_x[lx_max_idx], 2 * np_pk1_vol_x[lx_max_idx], '--', label='$P_{11}$ vol')
    axes[2].plot(np_stretch_x[lx_max_idx], 2 * np_pk1_ani_x[lx_max_idx], '--', label='$P_{11}$ ani')
    axes[2].plot(np_stretch_x[lx_max_idx], 2 * model_pk1_tot[lx_max_idx], linewidth=1.5, label='$P_{11}$ Total')

    axes[2].axvline(1.0, color='red', linestyle=':', linewidth=1)
    plot_helper.set_labels_title(axes[2], xlabel='Stretch in [x] [mm/mm]', ylabel='Engineering Stress [MPa]')
    axes[2].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    axes[2].yaxis.set_minor_locator(mticker.AutoMinorLocator())
    axes[2].legend()

    # --- Add Material Parameters Text ---

    # if solution.get("x_mat") is not None:
    if hasattr(solution, "x_mat"):
        mat_vars_dims = {'mu': "MPa", 'D': "MPa", 'k_1': "MPa", 'k_2': "-", 'alpha': "rad", 'kappa': "-"}
        param_text = "Material Parameters: "
        params_list = []

        for symbol_i, value_i in solution.x_mat.items():
            str_symbol_i = str(symbol_i)
            unit = mat_vars_dims.get(str_symbol_i, "-")
            params_list.append(f"${sy.latex(symbol_i)}$: {value_i:.4f} [{unit}]")

        param_text += ", ".join(params_list)

        # Get the position of the x-axis label of the last subplot
        xlabel_x, xlabel_y = axes[-1].xaxis.label.get_position()

        # Add text below the plots
        fig.text(xlabel_x, xlabel_y + 0.08,
                 param_text,
                 ha='center', va='bottom', fontsize=15, wrap=True,
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5),
                 )
    fig.subplots_adjust(bottom=0.15, hspace=0.3)  # Adjust bottom margin and spacing

    if filename_prefix is None:
        filename_prefix = f"material_fit_{sample_id}.png"

    plot_helper.save_plot(fig, filename_prefix, save_dir)


def article_post(
        h5_input_path: Union[str, Path, None] = None,
        plot_output_root_dir: Union[str, Path, None] = None,
        rats_ids_to_process: Optional[list] = None,
        # plot_control_region: bool = True,
):
    """
    Processes experimental data from an H5 file and generates raw signal plots for each relevant section and position,
    saving them into structured directories.

    Args:
        h5_input_path (Union[str, Path, None], optional):
            Path to the HDF5 data file OR the directory containing 'final_data.h5'.
            If None, it assumes 'DualMatFit/instron_data/final_data.h5'.
            The project root is inferred assuming this script is in 'DualMatFit/dualmatfit/plot.py'.
            Defaults to None.
        plot_output_root_dir (Union[str, Path, None], optional):
            The root directory where plot subdirectories will be created.
            If None, it defaults to 'DualMatFit/Results/article_plots_output'.
            Defaults to None.
        rats_ids_to_process (Optional[list], optional):
            List of specific rat IDs to process (e.g., ['rato_1', 'rato_2']).

    Notes:
        Instron Experimental Data default location:

        DualMatFit/
        ├── dualmatfit/  (package)
        │   └── plot.py
        ├── instron_data/
        │   └── final_data.h5
        └── Results/

    """
    h5_file_name = 'final_data.h5'
    script_dir = Path(__file__).resolve().parent.parent     # DualMatFit/dualmatfit/plotting/
    project_package_dir = script_dir.parent                 # DualMatFit/dualmatfit/

    # Determine HDF5 file path
    if h5_input_path is None:
        h5_file_path = (project_package_dir / 'instron_data' / h5_file_name).resolve()
    else:
        h5_input_path_resolved = Path(h5_input_path).resolve()
        if h5_input_path_resolved.is_dir():
            h5_file_path = (h5_input_path_resolved / h5_file_name).resolve()
        elif h5_input_path_resolved.is_file():
            h5_file_path = h5_input_path_resolved
        else:
            # Try one level up if a directory was given but the file wasn't directly in it
            h5_file_path_check = (h5_input_path_resolved.parent / h5_file_name).resolve()
            if h5_file_path_check.is_file():
                h5_file_path = h5_file_path_check
            else:
                logger.debug(f"Error: Provided HDF5 input path is neither a valid file nor a directory containing '{h5_file_name}': {h5_input_path}")
                return

    if not h5_file_path.is_file():
        logger.debug(f"Error: HDF5 data file not found at: {h5_file_path}")
        return

    # Determine plot output directory
    if plot_output_root_dir is None:
        plot_output_dir_base = (project_package_dir / 'Results' / 'instron_plots').resolve()
    else:
        plot_output_dir_base = Path(plot_output_root_dir).resolve()

    plot_output_dir_base.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Plot output base directory: {plot_output_dir_base}")

    dict_info_data = excel_data()
    logger.debug(f"Processing HDF5 file: {h5_file_path}")

    column_name_pattern = re.compile(r"\(([A-Za-z]{2})-([A-Za-z0-9])\)\s+(?:Time|Extension|Load)(?:\s*\[.*\])?")

    try:
        with pd.HDFStore(str(h5_file_path), mode='r') as h5_store:
            if rats_ids_to_process is None:
                h5_keys_to_process = [key for key in h5_store.keys() if key.lower().startswith('/rato')]
            else:
                h5_keys_to_process = [f"/{rat_id.lower()}" for rat_id in rats_ids_to_process if f"/{rat_id.lower()}" in h5_store.keys()]

            if not h5_keys_to_process:
                logger.debug(f"No keys starting with '/rato' found in {h5_file_path.name}. Nothing to process.")
                return

            for h5_key_i in h5_keys_to_process:
                logger.debug(f"\n  Processing HDF5 key: {h5_key_i}")
                rat_group_key_i = h5_key_i.lstrip('/').replace('_', '-')

                pd_exp_data_group_i = h5_store[h5_key_i]
                all_columns_in_group_i = pd_exp_data_group_i.columns
                num_columns_i = len(all_columns_in_group_i)

                logger.debug(f"    Output directory for this rat group: {plot_output_dir_base}")

                for col_group_start_k in range(0, num_columns_i, 3):
                    current_col_group_names_k = all_columns_in_group_i[col_group_start_k : col_group_start_k + 3]
                    if len(current_col_group_names_k) < 3:
                        logger.debug(f"    Skipping incomplete column group: {current_col_group_names_k.tolist()}")
                        continue

                    pd_sample_data_k = pd_exp_data_group_i[current_col_group_names_k].copy()
                    pd_sample_data_k.dropna(inplace=True)

                    if pd_sample_data_k.empty:
                        logger.debug(f"    Skipping empty data for column group: {current_col_group_names_k.tolist()}")
                        continue

                    first_col_name_k = current_col_group_names_k[0]
                    name_parts_match_k = column_name_pattern.match(first_col_name_k)

                    if not name_parts_match_k:
                        logger.debug(f"    Warning: Could not parse section/position from column name '{first_col_name_k}' for rat group '{rat_group_key_i}'. Skipping this sample.")
                        continue

                    section_key_from_col_k = name_parts_match_k.group(1)
                    position_key_from_col_k = name_parts_match_k.group(2) # This can be alphanumeric
                    sample_id_k = f"{rat_group_key_i}-{section_key_from_col_k}-{position_key_from_col_k}"
                    logger.debug(f"      Processing sample: {sample_id_k}")

                    if rat_group_key_i not in dict_info_data:
                        logger.debug(f"      Warning: Metadata key '{rat_group_key_i}' not found in dict_info_data. Skipping sample {sample_id_k}.")
                        continue

                    current_rat_specific_info_k = dict_info_data[rat_group_key_i]

                    if section_key_from_col_k not in current_rat_specific_info_k:
                        logger.debug(f"      Warning: Metadata key '{section_key_from_col_k}' not found for rat '{rat_group_key_i}'. Skipping sample {sample_id_k}.")
                        continue

                    current_section_info_k = current_rat_specific_info_k[section_key_from_col_k]

                    if position_key_from_col_k not in current_section_info_k:
                        logger.debug(f"      Warning: Metadata key '{position_key_from_col_k}' not found for section '{section_key_from_col_k}' of rat '{rat_group_key_i}'. Skipping sample {sample_id_k}.")
                        continue

                    current_sample_info_k = current_section_info_k[position_key_from_col_k].copy()
                    current_sample_info_k['sample_id'] = sample_id_k

                    try:
                        length_k = current_sample_info_k['len']
                        thickness_k = current_section_info_k['thick']
                        diameter_k = current_section_info_k['dia']

                        current_sample_info_k['ds'] = length_k * thickness_k
                        current_sample_info_k['dp'] = np.pi * (diameter_k - thickness_k / 2.0)

                    except KeyError as e:
                        logger.debug(f"      Warning: Missing geometric key ({e}) for sample {sample_id_k}. Cannot calculate ds/dp. Skipping.")
                        continue
                    except TypeError as e:
                        logger.debug(f"      Warning: TypeError during geometric calculation for sample {sample_id_k} (key: {e}). Check data types. Skipping.")
                        continue

                    instron_obj_k = InstronData(df_data=pd_sample_data_k, info_data=current_sample_info_k, ncontrol=3)

                    # Using sample_id as filename_prefix for clarity
                    dt_ext_k = instron_obj_k.np_extn[-1] * 0.05
                    xlim_time_k = (0., instron_obj_k.np_time[-1] * 1.05)
                    xlim_ext_k = (instron_obj_k.np_extn[0] - dt_ext_k, instron_obj_k.np_extn[-1] + dt_ext_k)
                    save_dir_i = (plot_output_dir_base / h5_key_i[1:]).resolve()
                    filename_prefix_k = sample_id_k.replace('-', '_')

                    plot_raw_signals(instron_data=instron_obj_k,
                                     xlim_time=xlim_time_k,
                                     xlim_ext=xlim_ext_k,
                                     save_dir=str(save_dir_i),
                                     filename_prefix=filename_prefix_k
                                     )
    except FileNotFoundError:
        logger.debug(f"Error: HDF5 data file not found after checks. Path was: {h5_file_path}")
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"An error occurred while processing HDF5 data: {e}")

    logger.debug("\narticle_post processing complete.")


if __name__ == "__main__":

    # list_selection = ['rato-1', 'rato-2', 'rato-3', 'rato-4', 'rato-5', 'rato-6', 'rato-7', 'rato-8', 'rato-9',
    #                   'rato-10', 'rato-11', 'rato-12', 'rato-13', 'rato-14', 'rato-15', 'rato-16', 'rato-17',
    #                   'rato-18', 'rato-19', 'rato-20', 'rato-21', 'rato-22', 'rato-23', 'rato-24', 'rato-25',
    #                   'rato-26', 'rato-27']

    # list_selection = ['rato-idoso-1', 'rato-idoso-2', 'rato-idoso-3', 'rato-idoso-4', 'rato-idoso-5', 'rato-idoso-6',
    #                   'rato-idoso-7', 'rato-idoso-8', 'rato-idoso-9', 'rato-idoso-10', 'rato-idoso-11',
    #                   'rato-idoso-12', 'rato-idoso-13']

    # list_selection = ['rato-17', 'rato-18', 'rato-20', 'rato-21', 'rato-22', 'rato-23', 'rato-24']

    article_post(rats_ids_to_process=['rato_wt_184041'])
    # article_post()