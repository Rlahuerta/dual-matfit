# -*- coding: utf-8 -*-
"""
Analytical model visualization functions.

This module provides plotting functions for visualizing analytical
model results, stress-strain comparisons, and parameter studies.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional, Union, Dict, List, Any, Tuple
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec

from dualmatfit.logging_config import get_logger
from dualmatfit.rato_info import excel_data
from dualmatfit.drivers import opt_solvers
from dualmatfit.experimental import InstronData, MaterialSetup
from dualmatfit.variational_form import VariationalFormulation, ring_geom
from dualmatfit.extension_solution import ExtensionSolution
from dualmatfit.least_square import CostFunction, CostIntegrator
from dualmatfit.plotting.plot_helpers import get_x_stretch, get_y_stretch, plt_assign_sec, PlotHelper
from dualmatfit.plotting.parameters import (COLORS, NAME_SECTIONS, SEGMENT_LINESTYLES,
                                           RATS_STYLES, DEFAULT_PLOT_LIMITS)
from dualmatfit.io_utils import load_excel_params, load_hdf5_data


logger = get_logger('plotting')

__all__ = [
    'plot_optimization_history',
    'plot_segment_force_curves',
    'plot_segment_stress_curves',
    'plot_mean_stress_curves',
    'plot_curves_from_xlsx',
]

# Default configuration for the VariationalFormulation if not fully specified by CSV
DEFAULT_VAR_FORM_CONFIG = {
    'itype': 'nh',
    'mix': 1,       # Displacement-based formulation typically
    'kappa': True,  # Assume kappa is a parameter
    'dvol': True,   # Assume bulk modulus is a parameter
    # 'bulk': 0.05, # This will be read from CSV if dvol is True
    'iso_split': False,
    'vol_type': 'simo92',
    'hv': False,
    'was': False,
}

# See Article: "On the Compressibility of Arterial Tissue"
# lwr: 42.14, upp: 99.03 (kPa)
# bulk_val = 99.03 / 1000.            # Median Value
bulk_val = 56.67 / 1000.          # Median Value [MPa]

# Default geometric parameters for ExtensionSolution if not available otherwise
# These are placeholders and ideally should come from a more specific source
# or be arguments to the plotting function if they vary.
DEFAULT_DS = 2.0        # example cross-sectional area (mm^2)
DEFAULT_LX_R = 10.0     # example reference length for stretch calculation (mm)


def plot_test_1d(
        lx: np.ndarray,
        title: str,
        post_results: dict,
        ltype: dict,
        fname: str,
):

    # Plot the results
    fig, ax = plt.subplots(2, figsize=(16, 12), sharex=True, dpi=700)
    fig.suptitle(title, fontsize=16)

    ltx_energy = r"$\psi$"
    ltx_stress = r"$\sigma_x$"

    for i, (key_i, val_i) in enumerate(post_results.items()):
        ax[0].plot(lx, val_i[0], ltype[key_i], label=f'{ltx_energy} - {key_i}')
        ax[1].plot(lx, val_i[1], ltype[key_i], label=f'{ltx_stress} - {key_i}')

    ax[0].set_ylabel(r'Strain Energy Density, $\psi (l_x)$', fontsize=10)
    ax[0].set_title('Strain Energy Density vs Stretch Ratio', fontsize=10)

    ax[0].grid(which='minor', alpha=0.2)
    ax[0].grid(which='major', alpha=0.5)
    ax[0].legend()

    ax[1].set_xlabel(r'Stretch Ratio, ($l_x$)', fontsize=10)
    ax[1].set_ylabel(f'Engineering Stress, {ltx_stress}', fontsize=10)
    ax[1].set_title('Stress vs Stretch Ratio', fontsize=10)

    ax[1].grid(which='minor', alpha=0.2)
    ax[1].grid(which='major', alpha=0.5)
    ax[1].legend()

    for ax_i in ax:
        ax_i.axvline(x=1., color='k', linestyle=":")
        ax_i.axhline(y=0., color='r', linestyle=":")

    fig.savefig(fname)
    plt.close(fig)


def plot_test_2d(lx: np.ndarray,
                 ly: np.ndarray,
                 title: str,
                 post_results: dict,
                 ltype: dict,
                 fname: str,
                 post_equations: dict = None,
                 ):
    if post_equations is None:
        post_equations = {}

    fig = plt.figure(figsize=(16, 12), dpi=700)
    gs = GridSpec(nrows=2, ncols=2)
    ax = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    ltx_energy = r"$\psi$"
    ltx_stress_x = r"$\sigma_x$"
    ltx_stress_y = r"$\sigma_y$"

    fig.suptitle(title, fontsize=16)

    eq_inc = 4.5
    for i, (key_i, val_i) in enumerate(post_results.items()):
        if post_equations.get(key_i) is not None:
            ax[0].text(1.1, eq_inc, f'{ltx_energy} - {key_i} = ${post_equations.get(key_i)}$', fontsize=13, ha='left')
            eq_inc -= 1.

        ax[0].plot(lx, val_i[0], ltype[key_i], label=f'{ltx_energy} - {key_i} formulation')

        ax[1].plot(lx, val_i[1][0, :], ltype[key_i], label=f'{ltx_stress_x} - {key_i} formulation')
        ax[1].plot(lx, val_i[1][1, :], ltype[key_i], label=f'{ltx_stress_y} - {key_i} formulation')

        ax[2].plot(ly, val_i[1][0, :], ltype[key_i], label=f'{ltx_stress_x} - {key_i} formulation')
        ax[2].plot(ly, val_i[1][1, :], ltype[key_i], label=f'{ltx_stress_y} - {key_i} formulation')

    ax[0].set_xlabel(r'Stretch Ratio, $l_x$', fontsize=10)
    ax[0].set_ylabel(r'Strain Energy, $\psi (l_x)$', fontsize=10)

    ax[1].set_xlabel(r'Stretch Ratio, $l_x$', fontsize=10)
    ax[1].set_ylabel(f'Engineering Stress, {ltx_stress_x}', fontsize=10)

    ax[2].set_xlabel(r'Stretch Ratio, $l_y$', fontsize=10)
    ax[2].set_ylabel(f'Engineering Stress, {ltx_stress_y}', fontsize=10)

    for ax_i in ax:
        ax_i.axvline(x=1., color='k', linestyle=":")
        ax_i.axhline(y=0., color='r', linestyle=":")

        ax_i.grid(which='minor', alpha=0.2)
        ax_i.grid(which='major', alpha=0.5)
        ax_i.legend()

    fig.savefig(fname)
    plt.close(fig)


def plot_aniso_strain_test_2d(
        lx: np.ndarray,
        alpha_deg: np.ndarray,
        title: str,
        post_results: dict,
        fname: str,
):

    fig = plt.figure(figsize=(18, 15), dpi=800)
    gs = GridSpec(nrows=3, ncols=2)
    ax = [fig.add_subplot(gs[0, :]),
          fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
          fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

    labels_y = [r"Strain Energy, $\psi$",
                r"Anisotropic Unimodular Invariants - $I_{v4} (\alpha)$",
                r"Anisotropic Unimodular Invariants - $I_{v6} (\alpha)$",
                r"Heaviside function $\phi (I_{v4})$",
                r"Heaviside function $\phi (I_{v6})$"]

    labels_x = [r"Stretch Ratio, $l_x$", r"Stretch Ratio, $l_y$"]

    list_ylimits = [[-5., 50.], [-1., 17.], [-1., 17.], [-0.2, 8.2], [-0.2, 8.2]]

    fig.suptitle(title, fontsize=16)
    alpha_latex = r"$\alpha$"

    for i, (key_i, val_i) in enumerate(post_results.items()):
        angle_i = alpha_deg[i]

        ax[0].plot(lx, val_i['energy'], label=f'{alpha_latex} - {angle_i}')

        ax[1].plot(lx, val_i['iv4'][0], label=f'{alpha_latex} - {angle_i}')
        ax[2].plot(lx, val_i['iv4'][1], label=f'{alpha_latex} - {angle_i}')

        ax[3].plot(lx, val_i['heaviside'][0], label=f'{alpha_latex} - {angle_i}')
        ax[4].plot(lx, val_i['heaviside'][1], label=f'{alpha_latex} - {angle_i}')

    for i, ax_i in enumerate(ax):
        if len(list_ylimits[i]) > 0:
            ax_i.set_ylim(list_ylimits[i])

        ax_i.axvline(x=1., color='k', linestyle=":")
        if i == 1 or i == 2:
            ax_i.axhline(y=1., color='r', linestyle=":")

        ax_i.set_xlabel(labels_x[0], fontsize=10)
        ax_i.set_ylabel(labels_y[i], fontsize=10)

        ax_i.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax_i.grid(which='minor', alpha=0.2)
        ax_i.grid(which='major', alpha=0.5)

        if i == 0:
            ax_i.legend()

        sec_ax_i = ax_i.secondary_xaxis('top', functions=(get_x_stretch, get_y_stretch))
        sec_ax_i.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.tight_layout()

    fig.savefig(fname)
    plt.close(fig)


def plot_aniso_stress_test_2d(lx: np.ndarray,
                              ly: np.ndarray,
                              alpha_deg: np.ndarray,
                              title: str,
                              post_results: dict,
                              fname: str,
                              ):

    fig = plt.figure(figsize=(18, 15), dpi=800)
    gs = GridSpec(nrows=3, ncols=2)
    ax = [fig.add_subplot(gs[0, :]),
          fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
          fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

    list_pairs = [(r"Strain Energy, "
                   r"$\psi (l_x, l_y, \kappa, \alpha)$",
                   "Stretch Ratio, $l_x$",
                   "Stretch Ratio, $l_y$"),
                  (r"Engineering Stress, $\sigma_x$", "Stretch Ratio, $l_x$"),
                  (r"Engineering Stress, $\sigma_y$", "Stretch Ratio, $l_x$"),
                  (r"Engineering Stress, $\sigma_x$", "Stretch Ratio, $l_y$"),
                  (r"Engineering Stress, $\sigma_y$", "Stretch Ratio, $l_y$")]

    fig.suptitle(title, fontsize=16)
    alpha_latex = r"$\alpha$"

    for i, (key_i, val_i) in enumerate(post_results.items()):
        angle_i = alpha_deg[i]

        ax[0].plot(lx, val_i['energy'], label=f'{alpha_latex} - {angle_i}')

        ax[1].plot(lx, val_i['stress'][0, :], label=f'{alpha_latex} - {angle_i}')
        ax[2].plot(lx, val_i['stress'][1, :], label=f'{alpha_latex} - {angle_i}')

        ax[3].plot(ly, val_i['stress'][0, :], label=f'{alpha_latex} - {angle_i}')
        ax[4].plot(ly, val_i['stress'][1, :], label=f'{alpha_latex} - {angle_i}')

    for i, par_i in enumerate(list_pairs):
        ax[i].set_ylim([-10., 50.])
        ax[i].axvline(x=1., color='k', linestyle=":")

        sec_axi = ax[i].secondary_xaxis('top', functions=(get_x_stretch, get_y_stretch))
        if i == 0:
            sec_axi.set_xlabel(par_i[2], fontsize=10)

        sec_axi.grid(which='minor', alpha=0.2)
        sec_axi.grid(which='major', alpha=0.5)

        ax[i].set_xlabel(par_i[1], fontsize=10)
        ax[i].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        sec_axi.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax[i].set_ylabel(par_i[0], fontsize=10)
        ax[i].grid(which='minor', alpha=0.2)
        ax[i].grid(which='major', alpha=0.5)
        ax[i].legend()

    plt.tight_layout()

    fig.savefig(fname)
    plt.close(fig)


def plot_aniso_stress_split_test_2d(lx: np.ndarray,
                                    alpha_deg: np.ndarray,
                                    title: str,
                                    post_results: dict,
                                    fname: str,
                                    ):
    """
    Plots anisotropic stress split test results in 2D.

    Args:
        lx (np.ndarray): Array of stretch ratios.
        alpha_deg (np.ndarray): Array of fiber angles in degrees.
        title (str): Title of the figure.
        post_results (dict): Dictionary containing the results to plot.
        fname (str): File name to save the figure.
    """

    # Ensure that alpha_deg and post_results have the same length
    if len(alpha_deg) != len(post_results):
        raise ValueError("Length of alpha_deg must match the number of entries in post_results.")

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 15), dpi=800)
    ax = axs.flatten()

    # Define labels for y-axes and x-axes
    labels_y = [r"${\psi}_{iso} (l_x, l_y)$", r"${\psi}_{vol} (l_x, l_y)$",
                r"${\psi}_{ani} (l_x, l_y, \kappa, \alpha)$", r"$I_{1} (l_x, l_y)$",
                r"$\sigma_x$ - Isochoric", r"$\sigma_x$ - Volumetric", r"$\sigma_x$ - Anisotropic",
                r"${I}_{v4}, {I}_{v6}$",
                r"$\sigma_y$ - Isochoric", r"$\sigma_y$ - Volumetric", r"$\sigma_y$ - Anisotropic",
                r"Heaviside $\phi ({I}_{v4}, {I}_{v6})$"]

    labels_x = [None, None, None, "Stretch Ratio, $l_x$",
                None, None, None, "Stretch Ratio, $l_x$",
                None, None, None, "Stretch Ratio, $l_x$"]

    # Set x and y limits for each subplot
    list_xlimits = [[0.1, 2.6] for _ in range(len(ax))]

    list_ylimits = [[-5., 50.], [-5., 50.], [-5., 50.], [-1., 17.],
                    [-10., 10.], [-20., 50.], [-20., 50.], [-1., 17.],
                    [-10., 10.], [-20., 50.], [-20., 50.], [-0.2, 8.2]]

    list_haxis = [0., 0., 0., 2.,
                  0., 0., 0., 1.,
                  0., 0., 0., None]

    fig.suptitle(title, fontsize=16)
    alpha_latex = r"$\alpha$"
    list_lines = []
    list_labels = []

    # Define mapping of subplot indices to data keys and data extraction functions
    plot_mappings = [
        (0, 'energy_iso', None),
        (1, 'energy_vol', None),
        (2, 'energy_ani', None),
        (3, 'iv1', None),
        (4, 'stress_iso', lambda v: v[0, :]),
        (5, 'stress_vol', lambda v: v[0, :]),
        (6, 'stress_ani', lambda v: v[0, :]),
        (7, 'iv4', lambda v: v[0]),
        (8, 'stress_iso', lambda v: v[1, :]),
        (9, 'stress_vol', lambda v: v[1, :]),
        (10, 'stress_ani', lambda v: v[1, :]),
        (11, 'heaviside', lambda v: v[0]),
    ]

    # Iterate over each angle and corresponding results
    for i, (key_i, val_i) in enumerate(post_results.items()):
        angle_i = alpha_deg[i]
        labels_i = f'{alpha_latex} = {angle_i}°'
        list_labels.append(labels_i)

        # Plot data on each subplot according to the mapping
        for idx, data_key, data_func in plot_mappings:
            ax_i = ax[idx]
            y_data = val_i[data_key]
            if data_func:
                y_data = data_func(y_data)
            line_i = ax_i.plot(lx, y_data, label=labels_i)
            if idx == 0:
                # For the legend, collect lines from the first plot
                list_lines.append(line_i[0])

    # Add legend to the figure
    fig.legend(handles=list_lines, labels=list_labels, title="Fiber Angle", loc='lower center', ncol=4, shadow=True)

    for i, ax_i in enumerate(ax):
        ax_i.set_xlim(list_xlimits[i])
        ax_i.set_ylim(list_ylimits[i])

        ax_i.axvline(x=1., color='k', linestyle=":")

        if list_haxis[i] is not None:
            ax_i.axhline(y=list_haxis[i], color='r', linestyle=":")

        # Set the secondary x-axis if needed
        if 'get_x_stretch' in globals() and 'get_y_stretch' in globals():
            sec_axi = ax_i.secondary_xaxis('top', functions=(get_x_stretch, get_y_stretch))
            sec_axi.grid(which='minor', alpha=0.2)
            sec_axi.grid(which='major', alpha=0.5)

        ax_i.set_ylabel(labels_y[i], fontsize=10)

        if labels_x[i] is not None:
            ax_i.set_xlabel(labels_x[i], fontsize=10)

        ax_i.grid(which='minor', alpha=0.2)
        ax_i.grid(which='major', alpha=0.5)

    plt.tight_layout()

    fig.savefig(fname)
    plt.close(fig)


def plot_aniso_inv_test_2d(
        lx: np.ndarray,
        alpha_deg: np.ndarray,
        title: str,
        post_results: dict,
        fname: str,
):

    fig, ax = plt.subplots(2, 2, figsize=(18, 15), sharex=True, dpi=800)
    fig.suptitle(title, fontsize=18)

    latex_alpha = r"$\alpha$"

    labels_y = [r"$I_{v4} (\alpha)$",
                r"$I_{v6} (\alpha)$",
                r"Heaviside $\phi (I_{v4})$",
                r"Heaviside $\phi (I_{v6})$"]

    labels_x = [r"Stretch Ratio, $l_x$", r"Stretch Ratio, $l_y$"]

    list_ylimits_iv = [[-1., 10.], [-1., 10.]]
    list_ylimits_hv = [[-0.2, 8.2], [-0.2, 8.2]]

    for i, (key_i, val_i) in enumerate(post_results.items()):
        angle_i = alpha_deg[i]

        for k, val_ik in enumerate(val_i['values']):
            ax[0, k].plot(lx, val_ik, label=f'{latex_alpha} - {angle_i}')

        for k, val_ik in enumerate(val_i['hv']):
            ax[1, k].plot(lx, val_ik, label=f'{latex_alpha} - {angle_i}')

    for i, ax_i in enumerate(ax[0, :]):
        sec_axi = ax_i.secondary_xaxis('top', functions=(get_x_stretch, get_y_stretch))
        sec_axi.set_xlabel(labels_x[1], fontsize=16)

        ax_i.grid(which='minor', alpha=0.2)
        ax_i.grid(which='major', alpha=0.5)

        ax_i.set_ylim(list_ylimits_iv[i])
        ax_i.axvline(x=1., color='k', linestyle=":")
        ax_i.axhline(y=1., color='r', linestyle=":")

        ax_i.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        sec_axi.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax_i.set_xlabel(labels_x[0], fontsize=16)
        ax_i.set_ylabel(labels_y[i], fontsize=16)
        ax_i.legend()

    for i, ax_i in enumerate(ax[1, :]):
        sec_axi = ax_i.secondary_xaxis('top', functions=(get_x_stretch, get_y_stretch))
        # sec_axi.set_xlabel(labels_x[1], fontsize=16)
        ax_i.axvline(x=1., color='k', linestyle=":")

        ax_i.grid(which='minor', alpha=0.2)
        ax_i.grid(which='major', alpha=0.5)

        ax_i.set_ylim(list_ylimits_hv[i])
        ax_i.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        sec_axi.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax_i.set_xlabel(labels_x[0], fontsize=16)
        ax_i.set_ylabel(labels_y[2 + i], fontsize=16)

    plt.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def plot_optimization_history(history: List[Dict],
                              param_names: List[str],
                              save_dir: str,
                              filename: str = "optimization_history.png"):
    """
    Plots the evolution of parameters and function value during optimization.

    Args:
        history (List[Dict]): A list of dictionaries, each containing 'x' (parameters)
                              and 'fun' (function value) for an iteration.
        param_names (List[str]): List of parameter names corresponding to the elements in 'x'.
        save_dir (str): Directory to save the plot.
        filename (str): Name for the output plot file.
    """
    plot_helper = PlotHelper(use_latex=False)

    if not history:
        logger.warning(" Optimization history is empty. Skipping plot.")
        return

    iterations = list(range(len(history)))
    fun_values = [h['fun'] for h in history]
    param_values = np.array([h['x'] for h in history])
    n_params = param_values.shape[1]

    if len(param_names) != n_params:
         logger.debug(f"Warning: Number of param_names ({len(param_names)}) does not match number of parameters ({n_params}). Using generic names.")
         param_names = [f'Param {i+1}' for i in range(n_params)]

    fig, axes = plot_helper.setup_figure(nrows=n_params + 1, ncols=1, figsize=(10, 3 * (n_params + 1)), sharex=True)
    fig.suptitle('Optimization History')

    # Plot function value
    ax = axes[0]
    ax.plot(iterations, fun_values, marker='.', linestyle='-', color=COLORS[0])
    plot_helper.set_labels_title(ax, ylabel='Objective Function Value')
    plot_helper.set_limits_ticks(ax, xdata=np.array(iterations), ydata=np.array(fun_values))
    ax.set_yscale('log') # Often useful for objective functions

    # Plot parameters
    for i in range(n_params):
        ax = axes[i+1]
        ax.plot(iterations, param_values[:, i], marker='.', linestyle='-', color=COLORS[(i+1) % len(COLORS)])
        plot_helper.set_labels_title(ax, ylabel=f'{param_names[i]} Value')
        plot_helper.set_limits_ticks(ax, xdata=np.array(iterations), ydata=param_values[:, i])

    axes[-1].set_xlabel('Iteration') # Set xlabel only on the last subplot
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout
    plot_helper.save_plot(fig, filename, save_dir)


def get_plot_style_for_sample(sample_name: str) -> Dict:
    """
    Determines plot style (color, linestyle, label) based on sample name parts.
    Example sample_name_parts: ['rat_17', 'Ar-A'] or ['rat_17', 'Ar', 'A']
    """
    if "idoso" in sample_name or "wt" in sample_name or "ko" in sample_name:
        rat_idx = sample_name.split('-')
    else:
        rat_idx = sample_name.split('-', 2)

    style = {'color': 'gray', 'linestyle': '-', 'linewidth': 1.5, 'label': 'rat-0', 'idx': '0-A'}  # Default

    str_rat_id_i = None
    segment_char_i = None
    rat_prefix_i = None

    # Try to extract rat ID and segment character
    for i, part_i in enumerate(rat_idx):
        if part_i.isdigit():
            str_rat_id_i = "_".join(rat_idx[:3])

            # FIXME: hot fix
            for seg_k in ['_A', '_B', '_C']:
                if seg_k in str_rat_id_i:
                    str_rat_id_i = str_rat_id_i.replace(seg_k, "")

            if RATS_STYLES.get(str_rat_id_i) is not None:
                rat_prefix_i = RATS_STYLES[str_rat_id_i]['id_prefix'].split('-')
            else:
                rat_prefix_i = ""

        if part_i in SEGMENT_LINESTYLES:  # e.g. 'A', 'B', 'C'
            segment_char_i = part_i
        elif len(part_i) == 1 and part_i.isalpha() and part_i.isupper():  # Fallback if not in dict but looks like a segment
            segment_char_i = part_i

    if str_rat_id_i and str_rat_id_i in RATS_STYLES:
        style['color'] = RATS_STYLES[str_rat_id_i]['color']
        style['label'] = f"{RATS_STYLES[str_rat_id_i]['id_prefix']}"
        style['idx'] = f"{rat_prefix_i[1]}-{segment_char_i}"

        if segment_char_i:
            style['label'] += f"-{segment_char_i}"

    elif segment_char_i:  # If only segment is identified
        style['label'] = f"Seg-{segment_char_i}"

    if segment_char_i and segment_char_i in SEGMENT_LINESTYLES:
        style['linestyle'] = SEGMENT_LINESTYLES[segment_char_i]

    return style


def generate_plot_data_from_xlsx(
        h5_path: Union[str, Path],
        xlsx_path: Union[str, Path],
        var_form_config: Dict = None,
        ncontrol: int = 15,
        list_rats: List[str] = None,
        rerun: bool = False,
) -> Dict[str, Any]:
    """
    Reads material parameters from a CSV file, generates theoretical force-stretch curves.

    Args:
        xlsx_path (Union[str, Path]):       Path to the XLSX file with material parameters.
        h5_path (Union[str, Path]):
        var_form_config (Dict, optional):   Configuration for VariationalFormulation. Defaults to DEFAULT_VAR_FORM_CONFIG.
        ncontrol (int, optional):           Number of control points from the cost function
        list_rats:
        rerun:

    Returns:
        Dict[str, List[Dict]]: A dictionary where keys are aortic regions ('Ar', 'Tr', 'Ab')
                               and values are lists of dictionaries, each containing
                               'name', 'stretch', 'force', 'position_id'.
    """
    if var_form_config is None:
        var_form_config = DEFAULT_VAR_FORM_CONFIG.copy()

    # ##################################################################################
    # Load Excel and HDF5 files
    df_params = load_excel_params(xlsx_path)
    if df_params is None:
        return {}

    if list_rats is None:
        list_rats = list(df_params.keys())

    h5_data = load_hdf5_data(h5_path)
    if h5_data is None:
        return {}

    # ##################################################################################
    # Main
    dict_info_data = excel_data()

    results_by_region = {'Ar': {}, 'Tr': {}, 'Ab': {}, 'baseline': {}}
    np_stretch_bsl = np.linspace(1., 1.5, num=ncontrol)

    if rerun:
        material_setup = MaterialSetup(itype=var_form_config['itype'],
                                       bulk=bulk_val,
                                       dvol=var_form_config['dvol'],
                                       kappa=var_form_config['kappa'],
                                       )
        ds_vars, aorta_seq, _, _ = material_setup()


    for i, (rat_i, df_param_i) in enumerate(df_params.items()):
        rat_rs_i = rat_i.replace('_', '-')

        if rat_i in list_rats:
            df_sample_data_i = h5_data[rat_i]
            sample_columns_group_i = list(df_sample_data_i.columns)

            # get the material parameters
            df_params_i = df_params[rat_i]
            df_params_i = df_params_i.rename(columns={'bulk': 'D'})
            rat_params_i = {loc_j: dict_info_data[rat_rs_i].get(loc_j) for loc_j in ['Ar', 'Tr', 'Ab'] if dict_info_data[rat_rs_i].get(loc_j) is not None}
            rat_ring_params_i = {}

            # ##############################################################################################
            # Evaluate for Each Section
            for section_j, info_j in rat_params_i.items():
                sections_info_j = info_j.copy()
                for loc_k in ["A", "B", "C"]:
                    if info_j.get(loc_k) is not None:
                        ring_geom(sections_info_j, loc_k)
                sections_info_j.pop('dia', None)
                sections_info_j.pop('thick', None)

                for loc_k, loc_params_k in sections_info_j.items():
                    rat_ik = f"{rat_rs_i}-{loc_k}"
                    section_jk = f"{section_j}-{loc_k}"
                    mat_param_jk = df_params_i.loc[section_jk, :]

                    if mat_param_jk.sum() > 0.:
                        try:
                            # sample_info_jk = dict_info_data[rat_rs_i][section_j][loc_k].copy()
                            # sample_info_jk['ds'] = loc_params_k['len'] * info_j['thick']
                            # sample_info_jk['dp'] = np.pi * (info_j['dia'] - info_j['thick'] / 2.)  # mean diameter
                            rat_ring_params_i[section_jk] = loc_params_k

                            section_exp_idx_jk = np.array([section_jk in clm_l for clm_l in sample_columns_group_i])
                            instron_jk = InstronData(df_data=df_sample_data_i.iloc[:, section_exp_idx_jk],
                                                    info_data=loc_params_k,
                                                    ncontrol=ncontrol,
                                                    )

                            vf_kwargs_jk = var_form_config.copy()
                            vf_kwargs_jk["ds"] = loc_params_k['ds']
                            var_form_jk = VariationalFormulation(**vf_kwargs_jk)
                            # var_form_jk._bulk = 0.056670
                            extension_solution_jk = ExtensionSolution(var_form_jk, module='jax')

                            # ReRun Optimization
                            if rerun:
                                ds_vars_jk = ds_vars.copy()
                                ds_vars_jk.loc[mat_param_jk.index, ["ini", "values"]] = mat_param_jk

                                cost_solution_jk = CostFunction(
                                    var_form=var_form_jk,
                                    load_ref=instron_jk.np_tload_ref,
                                    stretch_x=instron_jk.np_tstretch_ref,
                                    dsvars=ds_vars_jk,
                                    module='jax',
                                    dtype='adjoint',
                                    )

                                cost_int_jk = CostIntegrator([cost_solution_jk],
                                                           ftype='cauchy_robust',
                                                           c=40.,
                                                           rho=None,
                                                           alpha=0.01,              # L2: Adding Tikhonov Regularization
                                                           epsilon=1.e-3,           # Volume Regularization
                                                           beta=None,
                                                           rescale=None,
                                                           vol_reg=True,
                                                           )

                                opt_args_jk = ['ipopt', cost_int_jk, ds_vars_jk]
                                opt_res_jk = opt_solvers(*opt_args_jk, miter=200, giter=4, glb=False)
                                mat_param_jk[mat_param_jk.index] = opt_res_jk.series[mat_param_jk.index]

                            # TODO: Recreate high-res stretch from Instron data
                            primal_solution_jk = extension_solution_jk.solve(mat_params=mat_param_jk,
                                                                             stretch_x=instron_jk.high_res_stretch,
                                                                             # stretch_x=instron_jk.np_tstretch_ref,
                                                                             )

                            primal_solution_jk.name = section_jk
                            results_by_region[section_j][rat_ik] = {"experimental": instron_jk, "model": primal_solution_jk}

                        except (KeyError, ValueError, TypeError, IndexError) as e:
                            logger.debug(f"Error processing '{rat_rs_i}:{section_jk}': {e}")
                            continue

            # ##############################################################################################
            # Evaluate Baseline parameter
            np_ds_i = np.asarray([sec_k['ds'] for key_k, sec_k in rat_ring_params_i.items()])

            if 'mean' in df_params_i.index:
                mat_param_i = df_params_i.loc["mean", :]
            elif 'baseline' in df_params_i.index:
                mat_param_i = df_params_i.loc["baseline", :]
            else:
                mat_param_i = df_params_i.mean(axis=0)

            vf_kwargs_jk = var_form_config.copy()
            vf_kwargs_jk["ds"] = np.median(np_ds_i).item()
            var_form_i = VariationalFormulation(**vf_kwargs_jk)

            extension_solution_i = ExtensionSolution(var_form_i, module='jax')
            primal_solution_i = extension_solution_i.solve(mat_params=mat_param_i, stretch_x=np_stretch_bsl, )

            primal_solution_i.name = "baseline"
            results_by_region['baseline'][rat_rs_i] = primal_solution_i

    return results_by_region


def _get_nice_tick_step(val_range: float, num_steps_target: int) -> float:
    """
    Determines a 'nice' step size (e.g., 1, 2, 2.5, 5 * 10^n) for ticks,
    aiming for approximately num_steps_target intervals.

    Args:
        val_range: The approximate range the ticks should cover.
        num_steps_target: The desired number of intervals/steps.

    Returns:
        A 'nice' step size.
    """
    if num_steps_target <= 0:
        num_steps_target = 1

    if np.isclose(val_range, 0.0):
        # If range is zero, a default step is needed if we want >1 tick.
        # This value might need adjustment based on typical data scales.
        return 0.1

        # Estimate raw interval
    rough_step = val_range / num_steps_target

    if np.isclose(rough_step, 0.0):  # Avoid log(0) if val_range is tiny but not zero
        return 0.1  # Fallback if rough_step becomes ~0

    exponent = np.floor(np.log10(rough_step))
    mantissa = rough_step / (10 ** exponent)

    # "Nice" mantissas for intervals.
    # Common choices are 1, 2, 5, 10. 2.5 can be added for finer control.
    # Order matters: find the smallest nice_mantissa >= current mantissa.
    if mantissa <= 1.0:
        nice_mantissa = 1.0
    elif mantissa <= 2.0:
        nice_mantissa = 2.0
    # elif mantissa <= 2.5: nice_mantissa = 2.5 # Optional: for finer ticks
    elif mantissa <= 5.0:
        nice_mantissa = 5.0
    else:
        nice_mantissa = 10.0  # This will become 1.0 of the next exponent

    return nice_mantissa * (10 ** exponent)


def _calc_plot_ticks_with_zero(
        min_val: float,
        max_val: float,
        num_major_intervals_target: int = 5,  # Target number of major intervals
        major_tick_step_param: Optional[float] = None,  # User-specified major_inc
        num_minor_intervals_per_major: int = 5  # Number of minor intervals per major interval
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates major and minor tick positions for a Matplotlib plot,
    ensuring that the ticks include or span the zero value.

    Args:
        min_val: The minimum value of the data.
        max_val: The maximum value of the data.
        num_major_intervals_target: The desired number of major tick intervals.
        major_tick_step_param: Optional. If provided, this exact major tick step is used.
                               Overrides calculation based on num_major_intervals_target.
        num_minor_intervals_per_major: Number of minor intervals within each major interval.

    Returns:
        A tuple containing (minor_ticks_array, major_ticks_array).
    """
    if not isinstance(num_major_intervals_target, int) or num_major_intervals_target <= 0:
        num_major_intervals_target = 5  # Default if invalid
    if not isinstance(num_minor_intervals_per_major, int) or num_minor_intervals_per_major <= 0:
        num_minor_intervals_per_major = 5  # Default

    # 1. Determine Major Tick Step (major_step)
    if major_tick_step_param is not None and major_tick_step_param > 0:
        major_step = major_tick_step_param
    else:
        data_true_span = max_val - min_val
        if np.isclose(data_true_span, 0.0):  # min_val is very close to max_val
            if np.isclose(min_val, 0.0):  # Data is effectively at zero
                major_step = 0.2  # Default nice step around zero (e.g., for 5 intervals, range is -0.4 to 0.4)
            else:  # Data is a single point not at zero
                # Choose major_step to give a few ticks around this point AND ensure zero is crossed
                # Consider the larger of distance to zero or a small default span
                effective_span_for_step = max(abs(min_val), 0.2)  # Ensure some span if min_val is tiny
                major_step = _get_nice_tick_step(effective_span_for_step, max(1, num_major_intervals_target // 2))
        else:  # Normal case with a data range
            major_step = _get_nice_tick_step(data_true_span, num_major_intervals_target)

    if np.isclose(major_step, 0.0):  # Final fallback if calculations led to zero
        major_step = 0.1

    # Effective bounds that the ticks must span
    span_min = min(min_val, 0.0)
    span_max = max(max_val, 0.0)

    # Use a small epsilon related to major_step to handle floating point issues in floor/ceil
    epsilon = 1e-9 * major_step

    tick_start = np.floor(span_min / major_step - epsilon) * major_step
    tick_end = np.ceil(span_max / major_step + epsilon) * major_step

    # Ensure the original data min_val and max_val are also within the generated tick range
    tick_start = min(tick_start, np.floor(min_val / major_step - epsilon) * major_step)
    tick_end = max(tick_end, np.ceil(max_val / major_step + epsilon) * major_step)

    # Handle cases where the calculated range is degenerate (e.g., min_val=max_val=0)
    if np.isclose(tick_start, tick_end):
        if np.isclose(tick_start, 0.0):  # Effectively, data is at 0 or range is tiny around 0
            tick_start = -major_step * (num_major_intervals_target // 2 if num_major_intervals_target > 1 else 1)
            tick_end = major_step * (num_major_intervals_target // 2 if num_major_intervals_target > 1 else 1)
            # Ensure start and end are different if num_major_intervals_target was 1
            if np.isclose(tick_start, tick_end):
                tick_start = -major_step
                tick_end = major_step
        else:  # Single data point not at zero, or very narrow range not at zero
            tick_start = min(tick_start, 0.0) - major_step  # Ensure zero is spanned
            tick_end = max(tick_end, 0.0) + major_step  # Ensure zero is spanned

    # Ensure tick_start is not greater than tick_end (can happen if major_step is too large for a tiny range)
    if tick_start > tick_end:
        tick_start, tick_end = tick_end, tick_start  # Swap
        if np.isclose(tick_start, tick_end):  # If still degenerate after swap (e.g. both zero)
            tick_start -= major_step
            tick_end += major_step

    # 3. Generate major ticks
    major_ticks = np.arange(tick_start, tick_end + major_step * 0.5, major_step)

    # 4. Snap to zero if a major tick is very close (aesthetic)
    if len(major_ticks) > 0 and not np.isclose(0.0, major_ticks).any():
        closest_idx_to_zero = np.argmin(np.abs(major_ticks - 0.0))
        # Snap if within 10% of a major_step from zero (can be adjusted)
        if np.isclose(major_ticks[closest_idx_to_zero], 0.0, atol=major_step * 0.1):
            major_ticks[closest_idx_to_zero] = 0.0
    major_ticks = np.unique(major_ticks)  # Sorts and removes duplicates

    # Ensure there are at least two distinct major ticks if possible, especially if spanning zero
    if len(major_ticks) < 2:
        if np.isclose(0.0, major_ticks).any():  # If only [0.0]
            major_ticks = np.array([-major_step, 0.0, major_step])
        elif len(major_ticks) == 1:  # Single tick not at zero
            val = major_ticks[0]
            # Ensure ticks span this value and zero
            temp_ticks = [val - major_step, val, val + major_step, 0.0, 0.0 - major_step, 0.0 + major_step]
            major_ticks = np.unique(np.sort(temp_ticks))
            # Filter to a reasonable number around val and 0
            major_ticks = major_ticks[(major_ticks >= val - major_step * 2) & (major_ticks <= val + major_step * 2) | \
                                      (major_ticks >= 0.0 - major_step * 2) & (major_ticks <= 0.0 + major_step * 2)]
            major_ticks = np.unique(np.sort(major_ticks))
            if len(major_ticks) < 2:  # Final fallback
                major_ticks = np.array([min(val, 0.0) - major_step, max(val, 0.0) + major_step])

    # 5. Calculate minor ticks
    minor_step = major_step / num_minor_intervals_per_major
    # Ensure minor_step is not zero if major_step was very small
    if np.isclose(minor_step, 0.0):
        minor_step = major_step / 2.0 if not np.isclose(major_step, 0.0) else 0.01

    if len(major_ticks) > 0 and not np.isclose(minor_step, 0.0):
        # Generate minor ticks to span the actual range of generated major ticks
        minor_ticks_start = major_ticks[0]
        minor_ticks_end = major_ticks[-1]
        minor_ticks = np.arange(minor_ticks_start, minor_ticks_end + minor_step * 0.5, minor_step)
    else:
        # Fallback if major_ticks ended up empty or minor_step is zero
        # This case should ideally be prevented by robust major tick generation.
        if np.isclose(major_step, 0): major_step = 0.1  # ensure major_step is not zero for fallback
        if np.isclose(minor_step, 0): minor_step = major_step / 5.0

        major_ticks = np.array([-major_step, 0.0, major_step]) if len(major_ticks) < 2 else major_ticks
        minor_ticks = np.arange(major_ticks[0], major_ticks[-1] + minor_step * 0.5, minor_step)

    return minor_ticks, major_ticks


def plot_segment_force_curves(
        data_by_region: Dict[str, Any],
        output_dir: Union[str, Path],
        config_name: str,
        plot_title_suffix: str = "Uniaxial Force-Stretch Response",
        plot_limits: Optional[Dict[str, Tuple[float, float]]] = None,
):
    """
    Plots force-stretch curves for aortic segments, grouped by region, using PlotHelper.

    Args:
        data_by_region (Dict): Data structured by region.
            Example: {'Ar': {'name': [['rat_17','Ar-A'], ...],
                             'data': [model_solution_ArA, ...],
                             'test': [instron_data_ArA, ...]}, ...}
        output_dir (Union[str, Path]): Directory to save the plot.
        config_name (str): A name for this configuration (used in filename).
        plot_title_suffix (str): Suffix for the plot title.
        plot_limits (Optional[Dict]): Custom plot limits overriding defaults.
    """
    plot_helper = PlotHelper(use_latex=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    numbers_format = '%.2f'
    ninc = 4

    current_limits = DEFAULT_PLOT_LIMITS.copy()
    if plot_limits:
        current_limits.update(plot_limits)

    # ##################################################################################
    # Aorta Segments Min Max Values
    regions = ['Ar', 'Tr', 'Ab']
    segm_cfg = {key_i: dict(min=[], max=[]) for key_i in regions}
    y_labels_map = dict(
        Ar=r'AoA - $f_x (\textbf{\textit{m}})$ - Force [N]',
        Tr=r'DTAo - $f_x (\textbf{\textit{m}})$ - Force [N]',
        Ab=r'DAAo - $f_x (\textbf{\textit{m}})$ - Force [N]',
                     )
    common_x_label = r'$\lambda_x$ - Stretch [mm/mm]'

    # HGO model legend
    line_leg_mod = {key_i: dict(line=[], label=[]) for key_i in regions}

    list_max_lx = []
    label_fontsize = 20

    fig, segms_axes = plot_helper.setup_multi_region_plot(
        regions=regions,
        main_title=f"{plot_title_suffix}",
        y_labels_map=y_labels_map,
        x_label_common=common_x_label,
        figsize=(18, 20),
        dpi=200,
    )

    for region_key, region_ax in segms_axes.items():
        region_specific_data = data_by_region.get(region_key, {})
        if len(region_specific_data) > 0: # Skip if no data for this region
            for k, (key_k, data_k) in enumerate(region_specific_data.items()):
                style_info_k = get_plot_style_for_sample(key_k)

                # Experimental Data
                np_exp_stretch_k = data_k['experimental'].high_res_stretch
                np_exp_force_k = data_k['experimental'].high_res_force
                list_max_lx.append(np_exp_stretch_k.max())

                # Plot the Experimental lines
                exp_line_k, = region_ax.plot(np_exp_stretch_k, np_exp_force_k, linestyle='dotted', color='k',
                                             linewidth=1.5)

                plt_assign_sec(region_ax,
                               np_exp_stretch_k[-1], np_exp_force_k[-1],
                               style_info_k.get('idx'), 'k', size=800,
                               )

                # Model Data
                np_mdl_stretch_k = data_k['model'].stretch[:, 0]
                np_mdl_force_k = 2. * data_k['model'].fint[:, 0]

                style_info_k.pop('idx')
                mod_line_k, = region_ax.plot(np_mdl_stretch_k, np_mdl_force_k, **style_info_k)

                # Legend items for this subplot
                line_leg_mod[region_key]['line'].append(mod_line_k)
                line_leg_mod[region_key]['label'].append(style_info_k['label'])

                segm_cfg[region_key]['min'].append(min(np_mdl_force_k.min(), np_exp_force_k.min()))
                segm_cfg[region_key]['max'].append(max(np_mdl_force_k.max(), np_exp_force_k.max()))

    # ##################################################################################
    # Evaluate the limits for L_x
    lx_max = np.around(1.05 * max(list_max_lx), 1)
    lx_major_inc = lx_max / 20.
    lx_minor_inc = lx_major_inc / ninc

    xmajor_lx_ticks = np.round(np.arange(1., lx_max, lx_major_inc), decimals=3)
    xminor_lx_ticks = np.round(np.arange(1., lx_max, lx_minor_inc), decimals=3)

    # ##################################################################################
    # Set common X-axis label and limits
    segms_axes['Ab'].set_xlabel(common_x_label, fontsize=label_fontsize)

    for i, (key_i, ax_i) in enumerate(segms_axes.items()):
        ax_i.set_xticks(xmajor_lx_ticks)
        ax_i.set_xticks(xminor_lx_ticks, minor=True)
        ax_i.set_xticklabels(np.round(xmajor_lx_ticks, decimals=2))

        ax_i.set_xlim([0.95, lx_max])
        ax_i.xaxis.set_major_formatter(FormatStrFormatter(numbers_format))

        if len(segm_cfg[key_i]['max']) > 0 and len(segm_cfg[key_i]['min']) > 0:
            max_i = np.around(1.1 * max(segm_cfg[key_i]['max']), 1)
            min_i = -np.around(max_i / 10., 1) / ninc

            y_minor_ticks_i, y_major_ticks_i = _calc_plot_ticks_with_zero(min_i,
                                                                          max_i,
                                                                          num_minor_intervals_per_major=ninc,
                                                                          num_major_intervals_target=ninc,
                                                                          )

            ax_i.set_yticks(y_major_ticks_i)
            ax_i.set_yticks(y_minor_ticks_i, minor=True)
            ax_i.set_yticklabels(y_major_ticks_i)
            ax_i.set_ylim([y_minor_ticks_i[2], y_minor_ticks_i[-3]])

        ax_i.yaxis.set_major_formatter(FormatStrFormatter(numbers_format))
        ax_i.set_ylabel(y_labels_map[key_i], fontsize=label_fontsize)

        if i == 2:
            bbox_i = (0.5, -0.40)
        else:
            bbox_i = (0.5, -0.25)

        ax_i.legend(line_leg_mod[key_i]['line'], line_leg_mod[key_i]['label'],
                    bbox_to_anchor=bbox_i, loc='lower center', shadow=True, ncol=8, fontsize=16,
                    # fancybox=False,
                    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f"{config_name}_segment_forces_curves.png"
    plot_helper.save_plot(fig, plot_filename, str(output_dir))


def plot_segment_stress_curves(
        data_by_region: Dict[str, Any],
        output_dir: Union[str, Path],
        config_name: str,
        plot_title_suffix: str = "Uniaxial Stress-Stretch Response",
        plot_limits: Optional[Dict[str, Tuple[float, float]]] = None,
):
    """
    Plots stress-stretch curves for aortic segments, grouped by region, using PlotHelper.
    Allows plotting multiple stress components.

    Args:
        data_by_region (Dict): Data structured by region (similar to plot_segment_curves).
        output_dir (Union[str, Path]): Directory to save the plot.
        config_name (str): A name for this configuration (used in filename).
        plot_title_suffix (str): Suffix for the plot title.
        plot_limits (Optional[Dict]): Custom plot limits overriding defaults.

    """

    plot_helper = PlotHelper(use_latex=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ndiv = 4

    current_limits = DEFAULT_PLOT_LIMITS.copy()
    if plot_limits:
        current_limits.update(plot_limits)

    # ##################################################################################
    regions = ['Ar', 'Tr', 'Ab']                    # Aorta Segments Min Max Values
    stress_comps = ['iso', 'ani', 'total']          # Stress Componentes
    numbers_format = '%.2f'

    y_labels_map = dict(iso=r'$P_{11} (\textbf{\textit{m}})$ - Isotropic Stress Component [KPa]',
                        ani=r'$P_{11} (\textbf{\textit{m}})$ - Anisotropic Stress Component [KPa]',
                        total=r'$P_{11} (\textbf{\textit{m}})$ - Total Stress [KPa]',
                        )
    common_x_label = r'$\lambda_x$ - Stretch [mm/mm]'
    label_fontsize = 16

    for region_key_i in regions:
        fig_i, comp_ax_i = plot_helper.setup_multi_region_plot(
            regions=stress_comps,
            main_title=f"{plot_title_suffix}: {NAME_SECTIONS[region_key_i]}",
            y_labels_map=y_labels_map,
            x_label_common=common_x_label,
            figsize=(18, 20),
            dpi=200,
        )

        list_max_lx_i = []
        comps_cfg_i = {key_i: dict(min=[], max=[]) for key_i in stress_comps}
        line_leg_mod_i = {}

        region_specific_data = data_by_region.get(region_key_i, {})
        if len(region_specific_data) > 0:  # Skip if no data for this region
            for k, (key_k, data_k) in enumerate(region_specific_data.items()):
                style_info_k = get_plot_style_for_sample(key_k)
                sec_idx_k = style_info_k.get('idx')
                style_info_k.pop('idx')

                # Plot the Model lines
                np_mdl_stretch_k = data_k['model'].stretch[:, 0]
                list_max_lx_i.append(np_mdl_stretch_k.max())

                for comp_j in stress_comps:
                    np_mdl_pk1_kj = data_k['model'].stress[comp_j][:, 0]
                    if comp_j == 'iso':
                         np_mdl_pk1_kj = np_mdl_pk1_kj + data_k['model'].stress["vol"][:, 0]

                    plt_assign_sec(comp_ax_i[comp_j],
                                   np_mdl_stretch_k[-1],
                                   np_mdl_pk1_kj[-1],
                                   sec_idx_k,
                                   style_info_k.get('color'),
                                   size=800,
                                   )

                    mdl_line_kj, = comp_ax_i[comp_j].plot(np_mdl_stretch_k, np_mdl_pk1_kj, **style_info_k)

                    # Legend items for this subplot
                    if line_leg_mod_i.get(style_info_k['label']) is None:
                        line_leg_mod_i[style_info_k['label']] = mdl_line_kj

                    comps_cfg_i[comp_j]['min'].append(np_mdl_pk1_kj.min())
                    comps_cfg_i[comp_j]['max'].append(np_mdl_pk1_kj.max())

        # ##################################################################################
        # Evaluate the limits for L_x
        lx_max_i = np.around(1.1 * max(list_max_lx_i), 1)
        lx_major_inc_i = lx_max_i / 20
        lx_minor_inc_i = lx_major_inc_i / ndiv

        xmajor_lx_ticks_i = np.round(np.arange(1., lx_max_i, lx_major_inc_i), decimals=3)
        xminor_lx_ticks_i = np.round(np.arange(1., lx_max_i, lx_minor_inc_i), decimals=3)

        # ##################################################################################
        # Set common X-axis label and limits
        comp_ax_i['total'].set_xlabel(common_x_label, fontsize=label_fontsize)

        list_min_pk1 = []
        list_max_pk1 = []

        list_min_pk1_iso = []
        list_max_pk1_iso = []

        # X-axis pre-processing
        for comp_key_k, ax_k in comp_ax_i.items():
            ax_k.set_xticks(xmajor_lx_ticks_i)
            ax_k.set_xticks(xminor_lx_ticks_i, minor=True)
            ax_k.set_xticklabels(np.round(xmajor_lx_ticks_i, decimals=3))

            ax_k.set_xlim([0.95, lx_max_i])
            ax_k.xaxis.set_major_formatter(FormatStrFormatter(numbers_format))

            if comp_key_k == 'iso':
                list_min_pk1_iso.append(min(comps_cfg_i[comp_key_k]['min']))
                list_max_pk1_iso.append(max(comps_cfg_i[comp_key_k]['max']))
            else:
                list_min_pk1.append(min(comps_cfg_i[comp_key_k]['min']))
                list_max_pk1.append(max(comps_cfg_i[comp_key_k]['max']))

        # Y-axis pre-processing
        for comp_key_k, ax_k in comp_ax_i.items():
            if comp_key_k == 'iso':
                y_min_i = np.around(min(list_min_pk1_iso), 2)
                # y_max_i = np.around(1.1 * max(list_max_pk1_iso), 1)
                y_max_i = 0.1

                if y_max_i <= 0.:
                    y_max_i = np.around(0.2 * max(list_max_pk1), 2)

                if y_min_i == 0.:
                    y_min_shfit_i = -np.around(1.75 * y_max_i / ndiv, 2)
                else:
                    y_min_shfit_i = np.around(1.75 * y_min_i, 2)

                y_minor_ticks_i, y_major_ticks_i = _calc_plot_ticks_with_zero(y_min_shfit_i,
                                                                              y_max_i,
                                                                              num_minor_intervals_per_major=ndiv,
                                                                              num_major_intervals_target=ndiv,
                                                                              )

            else:
                y_min_i = np.around(min(list_min_pk1), 1)
                y_max_i = np.around(1.1 * max(list_max_pk1), 1)

                y_minor_ticks_i, y_major_ticks_i = _calc_plot_ticks_with_zero(y_min_i,
                                                                              y_max_i,
                                                                              num_minor_intervals_per_major=ndiv,
                                                                              num_major_intervals_target=ndiv,
                                                                              )

                y_min_shfit_i = -np.diff(y_major_ticks_i)[0] / 4

            ax_k.set_yticks(np.round(y_minor_ticks_i, decimals=3), minor=True)
            ax_k.set_yticks(np.round(y_major_ticks_i, decimals=3))
            ax_k.set_yticklabels(np.round(y_major_ticks_i, decimals=3))
            ax_k.set_ylabel(y_labels_map[comp_key_k], fontsize=label_fontsize)
            ax_k.set_ylim([y_min_shfit_i, y_max_i])
            ax_k.yaxis.set_major_formatter(FormatStrFormatter(numbers_format))

        comp_ax_i['total'].legend(list(line_leg_mod_i.values()), list(line_leg_mod_i.keys()),
                    bbox_to_anchor=(0.5, -0.35), loc='lower center', shadow=True, ncol=8, fontsize=16,
                    # fancybox=False,
                    )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f"{config_name}_{NAME_SECTIONS[region_key_i]}_stress_curves.png"
        plot_helper.save_plot(fig_i, plot_filename, str(output_dir))


def plot_mean_stress_curves(
        data_by_region: Dict[str, Any],
        output_dir: Union[str, Path],
        config_name: str,
        plot_title_suffix: str = "Uniaxial Stress-Stretch Baseline Response",
        plot_limits: Optional[Dict[str, Tuple[float, float]]] = None,
):
    plot_helper = PlotHelper(use_latex=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ndiv = 4

    current_limits = DEFAULT_PLOT_LIMITS.copy()
    if plot_limits:
        current_limits.update(plot_limits)

    # ##################################################################################
    stress_comps = ['iso', 'ani', 'total']          # Stress Componentes
    numbers_format = '%.2f'

    y_labels_map = dict(iso=r'$P_{11} (\textbf{\textit{m}})$ - Isotropic Stress Component [KPa]',
                        ani=r'$P_{11} (\textbf{\textit{m}})$ - Anisotropic Stress Component [KPa]',
                        total=r'$P_{11} (\textbf{\textit{m}})$ - Total Stress [KPa]',
                        )
    common_x_label = r'$\lambda_x$ - Stretch [mm/mm]'
    label_fontsize = 16

    fig, comp_ax = plot_helper.setup_multi_region_plot(
        regions=stress_comps,
        main_title=f"{plot_title_suffix}",
        y_labels_map=y_labels_map,
        x_label_common=common_x_label,
        figsize=(18, 20),
        dpi=200,
    )

    list_max_lx = []
    comps_cfg = {key_i: dict(min=[], max=[]) for key_i in stress_comps}
    line_leg_mod = {}

    region_specific_data = data_by_region.get("baseline", {})

    for k, (key_k, data_k) in enumerate(region_specific_data.items()):
        style_info_k = get_plot_style_for_sample(key_k)
        style_info_k.pop('idx')

        # Plot the Model lines
        np_mdl_stretch_k = data_k.stretch[:, 0]
        list_max_lx.append(np_mdl_stretch_k.max())

        for comp_j in stress_comps:
            np_mdl_pk1_kj = data_k.stress[comp_j][:, 0]
            if comp_j == 'iso':
                np_mdl_pk1_kj = np_mdl_pk1_kj + data_k.stress["vol"][:, 0]

            plt_assign_sec(comp_ax[comp_j],
                           np_mdl_stretch_k[-1],
                           np_mdl_pk1_kj[-1],
                           style_info_k.get('label'),
                           style_info_k.get('color'),
                           size=800,
                           )

            mdl_line_kj, = comp_ax[comp_j].plot(np_mdl_stretch_k, np_mdl_pk1_kj, **style_info_k)

            # Legend items for this subplot
            if line_leg_mod.get(style_info_k['label']) is None:
                line_leg_mod[style_info_k['label']] = mdl_line_kj

            comps_cfg[comp_j]['min'].append(np_mdl_pk1_kj.min())
            comps_cfg[comp_j]['max'].append(np_mdl_pk1_kj.max())

        # ##################################################################################
        # Evaluate the limits for L_x
        lx_max_i = np.around(1.1 * max(list_max_lx), 1)
        lx_major_inc_i = lx_max_i / 20
        lx_minor_inc_i = lx_major_inc_i / ndiv

        xmajor_lx_ticks_i = np.round(np.arange(1., lx_max_i, lx_major_inc_i), decimals=3)
        xminor_lx_ticks_i = np.round(np.arange(1., lx_max_i, lx_minor_inc_i), decimals=3)

        # ##################################################################################
        # Set common X-axis label and limits
        comp_ax['total'].set_xlabel(common_x_label, fontsize=label_fontsize)

        list_min_pk1 = []
        list_max_pk1 = []

        list_min_pk1_iso = []
        list_max_pk1_iso = []

        # X-axis pre-processing
        for comp_key_k, ax_k in comp_ax.items():
            ax_k.set_xticks(xmajor_lx_ticks_i)
            ax_k.set_xticks(xminor_lx_ticks_i, minor=True)
            ax_k.set_xticklabels(np.round(xmajor_lx_ticks_i, decimals=3))

            ax_k.set_xlim([0.95, lx_max_i])
            ax_k.xaxis.set_major_formatter(FormatStrFormatter(numbers_format))

            if comp_key_k == 'iso':
                list_min_pk1_iso.append(min(comps_cfg[comp_key_k]['min']))
                list_max_pk1_iso.append(max(comps_cfg[comp_key_k]['max']))
            else:
                list_min_pk1.append(min(comps_cfg[comp_key_k]['min']))
                list_max_pk1.append(max(comps_cfg[comp_key_k]['max']))

        # Y-axis pre-processing
        for comp_key_k, ax_k in comp_ax.items():
            if comp_key_k == 'iso':
                y_min_i = np.around(min(list_min_pk1_iso), 2)
                y_max_i = np.around(1.1 * max(list_max_pk1_iso), 1)

                if y_max_i <= 0.:
                    y_max_i = np.around(0.2 * max(list_max_pk1), 2)

                if y_min_i == 0.:
                    y_min_shfit_i = -np.around(1.75 * y_max_i / ndiv, 2)
                else:
                    y_min_shfit_i = np.around(1.75 * y_min_i, 2)

                y_minor_ticks_i, y_major_ticks_i = _calc_plot_ticks_with_zero(y_min_shfit_i,
                                                                              y_max_i,
                                                                              num_minor_intervals_per_major=ndiv,
                                                                              num_major_intervals_target=ndiv,
                                                                              )

            else:
                y_min_i = np.around(min(list_min_pk1), 1)
                y_max_i = np.around(1.1 * max(list_max_pk1), 1)

                y_minor_ticks_i, y_major_ticks_i = _calc_plot_ticks_with_zero(y_min_i,
                                                                              y_max_i,
                                                                              num_minor_intervals_per_major=ndiv,
                                                                              num_major_intervals_target=ndiv,
                                                                              )

                y_min_shfit_i = -np.diff(y_major_ticks_i)[0] / 4

            ax_k.set_yticks(np.round(y_minor_ticks_i, decimals=3), minor=True)
            ax_k.set_yticks(np.round(y_major_ticks_i, decimals=3))
            ax_k.set_yticklabels(np.round(y_major_ticks_i, decimals=3))
            ax_k.set_ylabel(y_labels_map[comp_key_k], fontsize=label_fontsize)
            ax_k.set_ylim([y_min_shfit_i, y_max_i])
            ax_k.yaxis.set_major_formatter(FormatStrFormatter(numbers_format))

        comp_ax['total'].legend(list(line_leg_mod.values()), list(line_leg_mod.keys()),
                                bbox_to_anchor=(0.5, -0.25), loc='lower center', shadow=True, ncol=8, fontsize=16,)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f"{config_name}_baseline_stress_curves.png"
    plot_helper.save_plot(fig, plot_filename, str(output_dir))


def plot_curves_from_xlsx(
        h5_path: Union[str, Path],
        xlsx_path: Union[str, Path],
        output_dir: Union[str, Path],
        ncontrol: int,
        config_name: str = "global_params_plot",
        var_form_cfg: Optional[Dict] = None,
):
    """
    Main function to read parameters from CSV, generate curves, and plot them.

    Args:
        h5_path:
        xlsx_path:          Path to the CSV file.
        output_dir:         Directory to save the plot.
        ncontrol:           Number of control points for the spline fitting.
        config_name:        Name for this plot configuration (used in filename).
        var_form_cfg:       Optional dictionary to override default VariationalFormulation settings.
    """

    vf_config = DEFAULT_VAR_FORM_CONFIG.copy()
    if var_form_cfg:
        vf_config.update(var_form_cfg)

    list_rats2plot = None
    # list_rats2plot = []
    # list_rats2plot += ["rato_wt_184085", "rato_wt_184012", "rato_wt_183964", "rato_wt_183918"]
    # list_rats2plot = ['rato_ko_184030', 'rato_ko_184058']
    # list_rats2plot = ['rato_17']

    # ###################################################################################33
    # test plot data generation
    # inp_keys = ['rato-ko-184030-A', 'rato-ko-184030-B', 'rato-ko-184030-C', 'rato-ko-184058-A', 'rato-ko-184058-B',
    #             'rato-ko-184058-C']
    #
    # list_plot_info = []
    # for key_i in inp_keys:
    #     style_info_i = get_plot_style_for_sample(key_i)
    #     list_plot_info.append((key_i, style_info_i))

    # ###################################################################################33
    logger.debug(f"Generating plot data for '{config_name}' from: {xlsx_path}")

    plot_data = generate_plot_data_from_xlsx(
        h5_path=h5_path,
        xlsx_path=xlsx_path,
        var_form_config=vf_config,
        ncontrol=ncontrol,
        list_rats=list_rats2plot,
        # rerun=True,
        rerun=False,
    )

    if plot_data:
        logger.debug(f"Plotting data for '{config_name}'")
        plot_segment_force_curves(
            data_by_region=plot_data,
            output_dir=output_dir,
            config_name=config_name
        )

        # Regular Stress local sections plots
        plot_segment_stress_curves(
            data_by_region=plot_data,
            output_dir=output_dir,
            config_name=config_name
        )

        # Baseline Stress plots
        plot_mean_stress_curves(
            data_by_region=plot_data,
            output_dir=output_dir,
            config_name=config_name
        )

    else:
        logger.debug(f"No data generated for plotting from {xlsx_path}")


if __name__ == "__main__":

    script_dir = Path(__file__).resolve().parent.parent

    xlsx_file = (script_dir.parent / "Results/M3-nh-ka-vol-glb/glb_opt_mat_param_ipopt_v01.xlsx").resolve()
    # xlsx_file = (script_dir.parent / "Results/M3-nh-ka-vol-glb/glb_opt_mat_param_ipopt_debug.xlsx").resolve()
    # xlsx_file = (script_dir.parent / "Results/M3-nh-ka-vol-glb/opt_mat_param_ipopt_ko.xlsx").resolve()

    h5_file = (script_dir.parent / 'instron_data' / 'final_data.h5').resolve()

    output_plot_dir = (xlsx_file.parent / "plots").resolve()
    output_plot_dir.mkdir(parents=True, exist_ok=True)

    custom_vf_config = {
        'itype': 'nh',
        'mix': 3,
        'kappa': True,
        'dvol': True,
        # 'iso_split': True,
        'iso_split': False,
        'vol_type': 'simo92',
        'hv': False,
        'was': True,
    }

    plot_curves_from_xlsx(
        h5_path=str(h5_file),
        xlsx_path=str(xlsx_file),
        output_dir=str(output_plot_dir),
        ncontrol=15,
        config_name="glb_ipopt_params_mix3_was_isosplit",
        var_form_cfg=custom_vf_config,
    )
