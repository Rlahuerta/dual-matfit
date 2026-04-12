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
from matplotlib.ticker import FormatStrFormatter

from pathlib import Path
from scipy import interpolate
from scipy.optimize import OptimizeResult
from typing import Optional, Union, Dict, Tuple, Any

from dualmatfit.data.rato_info import excel_data
from dualmatfit.data.experimental import InstronData
from dualmatfit.plotting.plot_helpers import PlotHelper, get_colors, plt_assign_sec, set_axis_labels, set_axis_ticks
from dualmatfit.plotting.parameters import COLORS, rats_ids, stress_dim

from dualmatfit.utils.logging_config import get_logger
logger = get_logger('plotting')

__all__ = [
    'plot_raw_signals',
    'plot_material_fit',
    'stress_plot',
    'exp_test_plot',
    'plot_time_extension',
    'plot_time_load',
    'plot_extension_load',
    'plot_reaction_force',
    'plot_volume_change',
    'plot_pk1_stress',
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

# --- Functions moved from plot.py ---

# Module-level configuration from plot.py
ticks_fontsize = 20
label_fontsize = 20

# Instantiate a default helper
plot_helper = PlotHelper(use_latex=False)


def stress_plot(
        dict_data: Dict[str, Any],
        ptitle: str = '',
        limits: Dict = None,
        lines: Dict = None,
):
    """
    Plot the fit data

    Args:
    :param dict_data:

    Kwargs:
    :param limits:
    :param ptitle:
    :param lines:           line types

    :return:
    """

    if limits is None:
        limits = {}

    if lines is None:
        lines = {}

    fig, ax = plt.subplots(3, figsize=(18, 20), sharex=True, dpi=200)
    fig.suptitle(f'Aorta PK1 Stress Plot - {ptitle}', x=0.5, y=0.99, fontweight="bold", fontsize=30)

    list_max_lx, list_max_ani = [], []
    list_min_iso, list_max_iso = [], []
    list_min_sum, list_max_sum = [], []
    list_yticklabels = []
    ninc = 4

    ldata = len(dict_data['name'])
    list_colors = get_colors()

    for i in range(ldata):
        # Get Rat name and ID
        data_nm_i = dict_data['name'][i]
        id_i = ''
        for xi in data_nm_i[0]:
            if xi.isdigit():
                id_i += xi

        nm_id_i = int(id_i)

        if rats_ids.get(nm_id_i) is not None:
            lc_id_i = rats_ids[nm_id_i]['id']
            color_i = rats_ids[nm_id_i]['color']

            data_nm_rs_i = f'rat-{lc_id_i}'
        else:
            color_i = list_colors[i]
            data_nm_rs_i = data_nm_i[0]

        data_sb_i = data_nm_i[-1][-1]
        name_i = f'{data_nm_rs_i}-{data_sb_i}'

        np_stretch_i = dict_data['data'][i]['stretch'][:, 0]
        list_max_lx.append(np_stretch_i.max())

        ############################################################################
        # Isotropic Contribution
        np_iso_x_i = dict_data['data'][i]['stress']['iso'][:, 0]
        np_vol_x_i = dict_data['data'][i]['stress']['vol'][:, 0]
        np_iso_i = np_iso_x_i + np_vol_x_i

        list_min_iso.append(np_iso_i.min())
        list_max_iso.append(np_iso_i.max())

        plt_kwargs_i = {'color': color_i, 'label': name_i}

        if lines.get(data_sb_i) is not None:
            plt_kwargs_i['linestyle'] = lines[data_sb_i]

        ax[0].plot(np_stretch_i, np_iso_i, **plt_kwargs_i)
        list_yticklabels.append(np_iso_i)

        # section info config
        plt_assign_sec(ax[0], np_stretch_i[-1], np_iso_i[-1], data_sb_i, color_i)

        ############################################################################
        # Anisotropic Contribution

        np_ani_x_i = dict_data['data'][i]['stress']['ani'][:, 0]
        list_max_ani.append(np_ani_x_i.max())

        ax[1].plot(np_stretch_i, np_ani_x_i, **plt_kwargs_i)
        list_yticklabels.append(np_ani_x_i)

        # section info config
        plt_assign_sec(ax[1], np_stretch_i[-1], np_ani_x_i[-1], data_sb_i, color_i)

        ############################################################################
        # PK1[0, 0], P_11

        # PK1 Stress Sum
        np_pk1_sum_x_i = np_iso_x_i + np_vol_x_i + np_ani_x_i

        list_min_sum.append(np_pk1_sum_x_i.min())
        list_max_sum.append(np_pk1_sum_x_i.max())

        ax[2].plot(np_stretch_i, np_pk1_sum_x_i, **plt_kwargs_i)
        list_yticklabels.append(np_pk1_sum_x_i)

        # section info config
        plt_assign_sec(ax[2], np_stretch_i[-1], np_pk1_sum_x_i[-1], data_sb_i, color_i)

    ax[0].set_ylabel(r'${\sigma}_{x}$ - Isotropic Contribution ' + f"[{stress_dim}]", fontsize=label_fontsize)
    ax[1].set_ylabel(r'${\sigma}_{x}$ - Collagen Fibers Contribution ' + f"[{stress_dim}]", fontsize=label_fontsize)
    ax[2].set_ylabel(r'${\sigma}_{x}$ - Total Stress ' + f"[{stress_dim}]", fontsize=label_fontsize)
    ax[2].set_xlabel(r'${\lambda}_{x}$ - Stretch [mm/mm]', fontsize=label_fontsize)

    if ldata > 0:
        ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=False, shadow=True, ncol=7, fontsize=18)

    ############################################################################
    # L_x
    if limits.get('lx') is None:
        lx_max = np.around(1.1 * np.max(list_max_lx), 1)
        lx_major_inc = lx_max / 20
        lx_minor_inc = lx_major_inc / ninc

        xmajor_lx_ticks = np.arange(1., lx_max, lx_major_inc)
        xminor_lx_ticks = np.arange(1., lx_max, lx_minor_inc)

    else:
        lx_max = limits['lx'][1]
        lx_major_inc = 0.1
        lx_minor_inc = lx_major_inc / ninc
        xmajor_lx_ticks = np.around(np.arange(limits['lx'][0], lx_max + lx_major_inc, lx_major_inc), decimals=3)
        xminor_lx_ticks = np.around(np.arange(limits['lx'][0], lx_max + lx_major_inc, lx_minor_inc), decimals=3)

    for i in range(3):
        ax[i].grid(which='minor', alpha=0.2)
        ax[i].grid(which='major', alpha=0.5)

        ax[i].set_xticks(xmajor_lx_ticks)
        ax[i].set_xticks(xminor_lx_ticks, minor=True)
        ax[i].set_xlim([0.9, lx_max])
        ax[i].set_xticklabels(xmajor_lx_ticks)

        ax[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # ax[i].set_yticklabels(list_yticklabels[i], fontsize=ticks_fontsize)

    ############################################################################
    for i, key_i in enumerate(['iso', 'ani', 'sum']):
        ymajor_ticks_i = None
        yminor_ticks_i = None
        yminmax_i = None

        if limits.get(key_i) is None:
            if i == 0:
                # Isotropic
                iso_min = np.around(np.min(list_min_iso), 1)
                iso_max = np.around(1.1 * np.max(list_max_iso), 1)

                iso_major_inc = np.around((iso_max - iso_min) / 5, 1)
                iso_minor_inc = iso_major_inc / ninc

                if iso_major_inc == 0.:
                    iso_major_inc = np.around((iso_max - iso_min) / 5, 2)
                    iso_minor_inc = iso_major_inc / ninc

                if iso_min < 0.:
                    iso_min = np.around(1.1 * np.min(list_min_iso), 1)
                else:
                    iso_min = -iso_major_inc

                ymajor_ticks_i = np.arange(iso_min, iso_max + iso_major_inc, iso_major_inc)
                yminor_ticks_i = np.arange(iso_min, iso_max + iso_minor_inc, iso_minor_inc)
                yminmax_i = [iso_min, iso_max]

            elif i == 1:
                # Anisotropic
                ani_max = np.around(1.1 * np.max(list_max_ani), 1)

                ani_major_inc = np.around(ani_max / ninc, 1)
                ani_minor_inc = ani_major_inc / ninc

                ymajor_ticks_i = np.arange(-ani_major_inc, ani_max, ani_major_inc)
                yminor_ticks_i = np.arange(-ani_major_inc, ani_max, ani_minor_inc)
                yminmax_i = [-ani_major_inc, ani_max]

            elif i == 2:
                # SUM / Total
                pk1_max = np.around(1.1 * np.max(list_max_sum), 1)
                pk1_major_inc = np.around(pk1_max / ninc, 1)
                pk1_minor_inc = pk1_major_inc / ninc

                ymajor_ticks_i = np.arange(-pk1_major_inc, pk1_max, pk1_major_inc)
                yminor_ticks_i = np.arange(-pk1_major_inc, pk1_max, pk1_minor_inc)
                yminmax_i = [-pk1_major_inc, pk1_max]

        else:
            px_max_i = limits[key_i][1]
            px_min_i = limits[key_i][0]

            px_major_inc = 0.5
            px_minor_inc = px_major_inc / ninc

            px_max_upp_i = px_max_i + px_major_inc
            px_max_lwr_i = limits[key_i][0] - px_major_inc

            if limits[key_i][0] < 0:
                ymajor_ticks_upp_i = np.arange(0., px_max_upp_i, px_major_inc)
                ymajor_ticks_lwr_i = np.flip(np.arange(0., px_max_lwr_i, -px_major_inc))
                ymajor_ticks_i = np.round(np.concatenate([ymajor_ticks_lwr_i[:-1], ymajor_ticks_upp_i], axis=0),
                                          decimals=3)

                yminor_ticks_upp_i = np.arange(0., px_max_upp_i, px_minor_inc)
                yminor_ticks_lwr_i = np.flip(np.arange(0., px_max_lwr_i, -px_minor_inc))
                yminor_ticks_i = np.round(np.concatenate([yminor_ticks_lwr_i[:-1], yminor_ticks_upp_i], axis=0),
                                          decimals=3)

            else:
                ymajor_ticks_i = np.round(np.arange(0., px_max_upp_i, px_major_inc), decimals=3)
                yminor_ticks_i = np.round(np.arange(0., px_max_upp_i, px_minor_inc), decimals=3)

            yminmax_i = (px_min_i, px_max_i)

        ax[i].set_yticks(ymajor_ticks_i)
        ax[i].set_yticks(yminor_ticks_i, minor=True)
        ax[i].set_ylim(yminmax_i)

    fig.set_layout_engine('tight')

    return fig


def exp_test_plot(
        exp_data: Dict[str, Any],
        limits: Dict = None,
        lines: Dict = None,
):
    """
    Plot the fit data

    Args:
    :param exp_data:

    Kwargs:
    :param limits:
    :param lines:           line types

    :return:
    """

    if limits is None:
        limits = {}

    fig, ax = plt.subplots(3, figsize=(18, 20), sharex=True, dpi=200)

    segm_ax = {'Ar': ax[0], 'Tr': ax[1], 'Ab': ax[2]}
    list_max_lx = []
    list_colors = get_colors()

    # Aorta Segments Min Max Values
    segm_cfg = {'Ar': {'min': [], 'max': []}, 'Tr': {'min': [], 'max': []}, 'Ab': {'min': [], 'max': []}}
    segm_info = {'Ar': r'AoA - $f_x$ - Force [N]',
                     'Tr': r'DTAo - $f_x$ - Force [N]',
                     'Ab': r'DAAo - $f_x$ - Force [N]'}

    # HGO model legend
    line_leg_mod = {key_i: {'line': [], 'label': []} for key_i in ['Ar', 'Tr', 'Ab']}

    # Instron legend
    line_leg_tst = {key_i: {'line': [], 'label': []} for key_i in ['Ar', 'Tr', 'Ab']}

    for key_k, data_k in exp_data.items():
        ldata_k = len(data_k['name'])

        # Main Plotting Looping
        for i in range(ldata_k):
            # Get Rat name and ID
            data_nm_i = data_k['name'][i]
            id_i = ''
            for xi in data_nm_i[0]:
                if xi.isdigit():
                    id_i += xi

            nm_id_i = int(id_i)

            if rats_ids.get(nm_id_i) is not None:
                lc_id_i = rats_ids[nm_id_i]['id']
                color_i = rats_ids[nm_id_i]['color']
                data_nm_rs_i = f'rat-{lc_id_i}'

            else:
                color_i = list_colors[i]
                data_nm_rs_i = data_nm_i[0]

            data_sg_i = data_nm_i[-1][:2]
            data_sb_i = data_nm_i[-1][-1]
            name_i = f'{data_nm_rs_i}-{data_sb_i}'

            np_stretch_i = data_k['data'][i]['stretch'][:, 0]
            list_max_lx.append(np_stretch_i.max())

            ############################################################################
            plt_kwargs_i = {'color': color_i, 'label': name_i}

            if lines.get(data_sb_i) is not None:
                plt_kwargs_i['linestyle'] = lines[data_sb_i]

            np_force_x_i = 2. * data_k['data'][i].fint[:, 0]

            segm_ax_k = segm_ax[data_sg_i]
            data_test_ki = data_k['test'][i]

            line_leg_mod[data_sg_i]['line'].extend(segm_ax_k.plot(np_stretch_i, np_force_x_i, **plt_kwargs_i))
            line_leg_mod[data_sg_i]['label'].append(name_i)

            # Instron Test (experimental)
            line_leg_tst[data_sg_i]['line'].append(segm_ax_k.plot(data_test_ki.high_res_stretch,
                                                                  data_test_ki.high_res_force,
                                                                  color='k', linestyle='dotted'))
            plt_assign_sec(segm_ax_k, np_stretch_i[-1], data_test_ki.high_res_force[-1], data_sb_i, 'k')

            segm_cfg[data_sg_i]['min'].append(min(np_force_x_i.min(), data_test_ki.high_res_force.min()))
            segm_cfg[data_sg_i]['max'].append(max(np_force_x_i.max(), data_test_ki.high_res_force.max()))

            # section info config (model)
            plt_assign_sec(segm_ax_k, np_stretch_i[-1], np_force_x_i[-1], data_sb_i, color_i)

    ############################################################################
    # L_x
    if limits.get('lx') is None:
        lx_max = np.around(1.1 * np.max(list_max_lx), 1)
        lx_major_inc = lx_max / 20
        lx_minor_inc = lx_major_inc / 5

        xmajor_lx_ticks = np.round(np.arange(1., lx_max, lx_major_inc), decimals=3)
        xminor_lx_ticks = np.round(np.arange(1., lx_max, lx_minor_inc), decimals=3)

    else:
        lx_max = limits['lx'][1]
        lx_major_inc = 0.1
        lx_minor_inc = lx_major_inc / 5
        xmajor_lx_ticks = np.round(np.arange(limits['lx'][0], lx_max + lx_major_inc, lx_major_inc), decimals=3)
        xminor_lx_ticks = np.round(np.arange(limits['lx'][0], lx_max + lx_major_inc, lx_minor_inc), decimals=3)

    ############################################################################
    # Configuration Setup

    for key_i, ax_i in segm_ax.items():
        ax_i.grid(which='minor', alpha=0.2)
        ax_i.grid(which='major', alpha=0.5)

        ax_i.set_xticks(xmajor_lx_ticks)
        ax_i.set_xticks(xminor_lx_ticks, minor=True)
        ax_i.set_xlim([0.95, lx_max])

        ax_i.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_i.set_ylabel(segm_info[key_i], fontsize=label_fontsize)

        ax_i.axvline(x=1., color='k', linestyle=":")
        ax_i.axhline(y=0., color='r', linestyle=":")

        if len(segm_cfg[key_i]['max']) > 0 and len(segm_cfg[key_i]['min']) > 0:
            if limits.get(key_i) is None:
                max_i = np.around(1.1 * np.max(segm_cfg[key_i]['max']), 1)
                major_inc_i = max((np.around(max_i / 10, 1), 0.1))
                minor_inc_i = major_inc_i / 5

                ymajor_ticks_i = np.arange(-major_inc_i, max_i, major_inc_i)
                yminor_ticks_i = np.arange(-major_inc_i, max_i, minor_inc_i)
                yminmax_i = [-major_inc_i, max_i]

            else:
                max_i = limits[key_i][1]
                major_inc = 0.2
                minor_inc = major_inc / 5

                ymajor_ticks_i = np.arange(limits[key_i][0], max_i + major_inc, major_inc)
                yminor_ticks_i = np.arange(limits[key_i][0], max_i + major_inc, minor_inc)
                yminmax_i = limits[key_i]

            ax_i.set_xticklabels(np.round(xmajor_lx_ticks, decimals=3))
            
            # Set ticks before labels to avoid warning
            ax_i.set_yticks(np.round(ymajor_ticks_i, decimals=3))
            ax_i.set_yticklabels(np.round(ymajor_ticks_i, decimals=3))
            ax_i.set_yticks(np.round(yminor_ticks_i, decimals=3), minor=True)
            ax_i.set_ylim(yminmax_i)

            ax_i.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax_i.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

            ax_i.legend(line_leg_mod[key_i]['line'], line_leg_mod[key_i]['label'],
                        loc='lower center', fancybox=False, shadow=True, ncol=7, fontsize=18)

    ax[-1].set_xlabel(r'${\lambda}_{x}$ - Stretch (mm/mm)', fontsize=label_fontsize)

    fig.set_layout_engine('tight')

    return fig


def plot_time_extension(ax: plt.Axes,
                        time_data: np.ndarray,
                        ext_data: np.ndarray,
                        control_time: Optional[np.ndarray] = None,
                        control_ext: Optional[np.ndarray] = None,
                        ):
    """
    Plot Time vs Extension.

    Parameters:
    - ax (plt.Axes): Axis to plot on.
    - time_data (np.ndarray): Raw time data.
    - ext_data (np.ndarray): Raw extension data.
    - control_time (np.ndarray, optional): Time points for control data.
    - control_ext (np.ndarray, optional): Extension points for control data.
    """
    ax.plot(time_data, ext_data, label='Signal')
    if control_time is not None and control_ext is not None:
        ax.plot(control_time, control_ext, 'o', color='red', label='Control')
    set_axis_labels(ax, 'Time [s]', 'Extension [mm]')
    set_axis_ticks(ax, time_data, ext_data)
    ax.legend()


def plot_time_load(ax: plt.Axes,
                   time_data: np.ndarray,
                   load_data: np.ndarray,
                   control_time: Optional[np.ndarray] = None,
                   control_load: Optional[np.ndarray] = None,
                   ylim: Optional[Tuple[float, float]] = None,
                   ):
    """
    Plot Time vs Load.

    Parameters:
    - ax (plt.Axes): Axis to plot on.
    - time_data (np.ndarray): Raw time data.
    - load_data (np.ndarray): Raw load data.
    - control_time (np.ndarray, optional): Time points for control data.
    - control_load (np.ndarray, optional): Load points for control data.
    - ylim (Optional[Tuple[float, float]]): Y-axis limits for the plot.
    """
    ax.plot(time_data, load_data, label='Signal')
    if control_time is not None and control_load is not None:
        ax.plot(control_time, control_load, 'o', color='red', label='Control')
    set_axis_labels(ax, 'Time [s]', 'Force [N]')
    set_axis_ticks(ax, time_data, load_data)
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)


def plot_extension_load(ax: plt.Axes,
                        ext_data: np.ndarray,
                        load_data: np.ndarray,
                        control_ext: Optional[np.ndarray] = None,
                        control_load: Optional[np.ndarray] = None,
                        ylim: Optional[Tuple[float, float]] = None,
                        ):
    """
    Plot Extension vs Load.

    Parameters:
    - ax (plt.Axes): Axis to plot on.
    - ext_data (np.ndarray): Raw extension data.
    - load_data (np.ndarray): Raw load data.
    - control_ext (np.ndarray, optional): Extension points for control data.
    - control_load (np.ndarray, optional): Load points for control data.
    - ylim (Optional[Tuple[float, float]]): Y-axis limits for the plot.
    """
    ax.plot(ext_data, load_data, label='Signal')
    if control_ext is not None and control_load is not None:
        ax.plot(control_ext, control_load, 'o', color='red', label='Control')
    set_axis_labels(ax, 'Extension [mm]', 'Force [N]')
    set_axis_ticks(ax, ext_data, load_data)
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)


def plot_reaction_force(ax: plt.Axes,
                        exp_stretch: np.ndarray,
                        exp_load_ref: np.ndarray,
                        control_stretch_ref: Optional[np.ndarray] = None,
                        control_load_ref: Optional[np.ndarray] = None,
                        model_stretch_hg: np.ndarray = None,
                        model_force_int: np.ndarray = None,
                        ring_factor: float = 1.0,
                        ):
    """
    Plot Reaction Force vs Stretch (Model and Experimental).

    Parameters:
    - ax (plt.Axes): Axis to plot on.
    - exp_stretch (np.ndarray): Experimental stretch data (raw or high-res interpolated).
    - exp_load_ref (np.ndarray): Experimental load data (ref adjusted, raw or high-res).
    - control_stretch_ref (np.ndarray, optional): Control stretch points (ref adjusted).
    - control_load_ref (np.ndarray, optional): Control load points (ref adjusted).
    - model_stretch_hg (np.ndarray, optional): High-resolution model stretch values.
    - model_force_int (np.ndarray, optional): Model internal force values corresponding to model_stretch_hg.
    - ring_factor (float): Factor to scale force (e.g., 2.0 for tape test).
    """
    ax.plot(exp_stretch, exp_load_ref, label='Instron')

    if control_stretch_ref is not None and control_load_ref is not None:
        ax.plot(control_stretch_ref, ring_factor * control_load_ref, 'o', color='red', label='Control')

    if model_stretch_hg is not None and model_force_int is not None:
        ax.plot(model_stretch_hg, ring_factor * model_force_int[:, 0], color='black', label='Model Prediction')

    # Determine axis limits and ticks based on combined data
    all_x = np.concatenate([exp_stretch, control_stretch_ref if control_stretch_ref is not None else np.array([]), model_stretch_hg if model_stretch_hg is not None else np.array([])])
    all_y = np.concatenate([exp_load_ref, ring_factor * control_load_ref if control_load_ref is not None else np.array([]), ring_factor * model_force_int[:, 0] if model_force_int is not None else np.array([])])

    # Handle potential empty or constant data
    if all_x.size > 0 and all_y.size > 0:
        max_y = np.around(all_y.max() * 1.2, 1) if all_y.max() > 0 else (0.1 if all_y.min() <= 0 else all_y.min() * 0.8)
        min_y = np.around(all_y.min() * 1.2, 1) if all_y.min() < 0 else (0.1 if all_y.max() <= 0 else all_y.max() * 0.8)
        if max_y <= min_y: # Handle case where all values are close to 0 or constant
             buffer = abs(all_y[0]) * 0.2 + 0.1 if all_y.size > 0 else 0.1
             min_y = all_y[0] - buffer if all_y.size > 0 else -0.1
             max_y = all_y[0] + buffer if all_y.size > 0 else 0.1

        ax.set_ylim([min_y, max_y]) # Set limits based on combined data

        # X-ticks handled by sharex in orchestrator, Y-ticks calculated here
        y_major_load_ticks = np.linspace(min_y, max_y, num=10)
        if max_y - min_y > 1e-9:
             y_minor_load_ticks = np.linspace(min_y, max_y, num=50)
        else:
             y_minor_load_ticks = y_major_load_ticks

        ax.set_yticks(y_major_load_ticks)
        ax.set_yticks(y_minor_load_ticks, minor=True)

    set_axis_labels(ax, '', 'Force [N]') # X-label will be on the bottom plot
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.legend()


def plot_volume_change(ax: plt.Axes,
                       model_stretch: np.ndarray = None,
                       model_detJ: np.ndarray = None,
                       ):
    """
    Plot Volume Change (detJ) and individual stretches vs Stretch.

    Parameters:
    - ax (plt.Axes): Axis to plot on.
    - model_stretch (np.ndarray, optional): Model stretch values (shape (N, 3)).
    - model_detJ (np.ndarray, optional): Model determinant of F values (shape (N,)).
    """
    if model_stretch is not None and model_stretch.shape[1] >= 3:
        model_stretch_x = model_stretch[:, 0]
        model_stretch_y = model_stretch[:, 1]
        model_stretch_z = model_stretch[:, 2]
        ax.plot(model_stretch_x, model_stretch_x, '--', label=r'$\lambda_x$') # Plot lambda_x vs lambda_x (identity)
        ax.plot(model_stretch_x, model_stretch_y, '--', label=r'$\lambda_y$')
        ax.plot(model_stretch_x, model_stretch_z, '--', label=r'$\lambda_z$')
    else:
        model_stretch_x = None

    if model_stretch_x is not None and model_detJ is not None:
         ax.plot(model_stretch_x, model_detJ, label='J - Volume Ratio')

    # Determine axis limits and ticks based on model solution data
    all_x = model_stretch_x if model_stretch_x is not None else np.array([])
    all_y_list = []
    if model_stretch is not None: all_y_list.append(model_stretch)
    if model_detJ is not None: all_y_list.append(model_detJ.reshape(-1,1))
    all_y = np.concatenate(all_y_list, axis=1) if all_y_list else np.array([])

    if all_x.size > 0 and all_y.size > 0:
        max_y = np.around(all_y.max() * 1.1, 1) if all_y.max() > 0 else 1.1
        min_y = np.around(all_y.min() * 0.9, 1) if all_y.min() < 1 else 0.9
        if max_y <= min_y:
             buffer = abs(all_y[0]) * 0.2 + 0.1 if all_y.size > 0 else 0.1
             min_y = all_y[0] - buffer if all_y.size > 0 else 0.9
             max_y = all_y[0] + buffer if all_y.size > 0 else 1.1
             if min_y > 1 and max_y > 1: min_y = 1.0

        ax.set_ylim([min_y, max_y])

        # X-ticks handled by sharex in orchestrator, Y-ticks calculated here
        y_major_ticks = np.linspace(min_y, max_y, num=10)
        if max_y - min_y > 1e-9:
            y_minor_ticks = np.linspace(min_y, max_y, num=50)
        else:
             y_minor_ticks = y_major_ticks

        ax.set_yticks(y_major_ticks)
        ax.set_yticks(y_minor_ticks, minor=True)

    set_axis_labels(ax, '', 'det(F)') # X-label will be on the bottom plot
    ax.legend()
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)


def plot_pk1_stress(ax: plt.Axes,
                    exp_stretch_ref: Optional[np.ndarray] = None,
                    exp_pk1_ref: Optional[np.ndarray] = None,
                    model_stretch_hg: np.ndarray = None,
                    model_stress_iso: np.ndarray = None,
                    model_stress_vol: np.ndarray = None,
                    model_stress_ani: np.ndarray = None,
                    ):
    """
    Plot PK1 Stress vs Stretch (Model and Experimental).

    Parameters:
    - ax (plt.Axes): Axis to plot on.
    - exp_stretch_ref (np.ndarray, optional): Experimental stretch points (ref adjusted).
    - exp_pk1_ref (np.ndarray, optional): Experimental PK1 points (ref adjusted).
    - model_stretch_hg (np.ndarray, optional): High-resolution model stretch values.
    - model_stress_iso (np.ndarray, optional): Model isotropic stress.
    - model_stress_vol (np.ndarray, optional): Model volumetric stress.
    - model_stress_ani (np.ndarray, optional): Model anisotropic stress.
    """
    model_stress_total = None
    if model_stress_iso is not None and model_stress_vol is not None and model_stress_ani is not None:
         model_stress_total = model_stress_iso + model_stress_vol + model_stress_ani

    if model_stretch_hg is not None:
        if model_stress_iso is not None: ax.plot(model_stretch_hg, model_stress_iso[:, 0], '--', label='PK1 Iso')
        if model_stress_vol is not None: ax.plot(model_stretch_hg, model_stress_vol[:, 0], '--', label='PK1 Vol')
        if model_stress_ani is not None: ax.plot(model_stretch_hg, model_stress_ani[:, 0], '--', label='PK1 Ani')
        if model_stress_total is not None: ax.plot(model_stretch_hg, model_stress_total[:, 0], label='PK1 Total')

    if exp_stretch_ref is not None and exp_pk1_ref is not None:
         ax.plot(exp_stretch_ref, exp_pk1_ref, 'o', color='red', label='PK1 Reference')

    # Determine axis limits and ticks based on combined data
    all_x = np.concatenate([exp_stretch_ref if exp_stretch_ref is not None else np.array([]), model_stretch_hg if model_stretch_hg is not None else np.array([])])
    all_y = np.concatenate([exp_pk1_ref if exp_pk1_ref is not None else np.array([]), model_stress_total[:, 0] if model_stress_total is not None else np.array([])])


    if all_x.size > 0 and all_y.size > 0:
        max_y = np.around(all_y.max() * 1.2, 2) if all_y.max() > 0 else (0.1 if all_y.min() <= 0 else all_y.min() * 0.8)
        min_y = np.around(all_y.min() * 1.2, 2) if all_y.min() < 0 else (0.1 if all_y.max() <= 0 else all_y.max() * 0.8)
        if max_y <= min_y:
             buffer = abs(all_y[0]) * 0.2 + 0.1 if all_y.size > 0 else 0.1
             min_y = all_y[0] - buffer if all_y.size > 0 else -0.1
             max_y = all_y[0] + buffer if all_y.size > 0 else 0.1

        ax.set_ylim([min_y, max_y])

        # X-ticks handled by sharex in orchestrator, Y-ticks calculated here
        y_major_ticks = np.linspace(min_y, max_y, num=10)
        if max_y - min_y > 1e-9:
            y_minor_ticks = np.linspace(min_y, max_y, num=50)
        else:
            y_minor_ticks = y_major_ticks

        ax.set_yticks(y_major_ticks)
        ax.set_yticks(y_minor_ticks, minor=True)

    set_axis_labels(ax, 'Stretch in [x] (mm/mm)', 'Stress [KPa]') # X-label is on this bottom plot
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.legend()
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

