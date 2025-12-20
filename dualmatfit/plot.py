# -*- coding: utf-8 -*-
"""
Basic plotting utilities for material fitting.

This module provides simple plotting functions for stress-strain curves
and experimental test data visualization.
"""
import numpy as np
import pandas as pd
# import sympy as sy

# from pathlib import Path
from scipy import interpolate
from scipy.optimize import OptimizeResult
from typing import Optional, Dict, Any, Tuple


import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# from matplotlib.gridspec import GridSpec

from dualmatfit.plotting.plot_helpers import PlotHelper, get_colors, plt_assign_sec, set_axis_labels, set_axis_ticks
from dualmatfit.plotting.parameters import rats_ids, stress_dim

__all__ = [
    'stress_plot',
    'exp_test_plot',
]

# --- Plotting Configuration ---
DEFAULT_STYLE = 'seaborn-v0_8-whitegrid'
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (10, 6) # Default figure size for single plots
DEFAULT_MULTI_FIGSIZE = (12, 10) # Default for multi-panel plots

ticks_fontsize = 20
# label_fontsize = 25
label_fontsize = 20

# Instantiate a default helper
plot_helper = PlotHelper(use_latex=False)


def ese_plot(
        key_ese: list,
        xdisp: float,
        label_x: [str],
        list_xticks: [np.ndarray],
        pd_aorta_seq: pd.DataFrame,
):
    """

    **ARGS**
    :param key_ese:
    :param xdisp:
    :param label_x:
    :param list_xticks:
    :param pd_aorta_seq:

    :return:
    """

    fig, ax = plt.subplots(len(key_ese), figsize=(14, 10), sharex=True, dpi=120)
    fig.suptitle(f'Aorta Strain Energy Pattern (Inverse of Compliance) x-disp: {np.around(xdisp, 2)}')

    np_loc = np.where(pd_aorta_seq['sum'].values > 0.)[0]
    np_idx = pd_aorta_seq['idx'].values[np_loc]

    for w, key_w in enumerate(key_ese):
        np_ese_w = pd_aorta_seq[key_w].values[np_loc]

        ese_max_w = np.around(1.5 * np_ese_w.max(), 2)
        ese_max_w = max(ese_max_w, 0.01)

        ese_min_w = np.around(0.5 * np_ese_w.min(), 2)
        if ese_min_w < 0.05:
            ese_min_w = -0.1 * ese_max_w

        ymin_ticks = np.abs(ese_max_w / 10.)
        ymax_ticks = 4. * ymin_ticks

        ymajor_ticks_w = np.arange(0., ese_max_w, ymax_ticks)
        yminor_ticks_w = np.arange(0., ese_max_w, ymin_ticks)

        ax[w].set_xticks(list_xticks[1])
        ax[w].set_xticks(list_xticks[0], minor=True)
        ax[w].set_xlim([-0.5, np_idx.max() + 0.5])

        ax[w].set_yticks(ymajor_ticks_w)
        ax[w].set_yticks(yminor_ticks_w, minor=True)
        ax[w].set_ylim([ese_min_w, ese_max_w])

        ax[w].grid(which='minor', alpha=0.2)
        ax[w].grid(which='major', alpha=0.6)

        ax[w].plot(np_idx, np_ese_w, label=key_w)
        ax[w].plot(np_idx, np_ese_w, 'o', color='red')

        ax[w].legend()
        ax[w].set_ylabel(key_w)
        ax[w].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax[-1].set_xlabel('Aorta Section Idx')
    ax[-1].set_xticklabels(label_x)

    return fig


def mat_plot(opt_keys: list,
             label_x: [str],
             list_xticks: [np.ndarray],
             pd_aorta_seq: pd.DataFrame,
             ):
    """

    :param opt_keys:
    :param label_x:
    :param list_xticks:
    :param pd_aorta_seq:

    :return:
    """

    fig, ax = plt.subplots(len(opt_keys), figsize=(14, 10), sharex=True, dpi=120)
    fig.suptitle('Aorta Material Properties Pattern')

    np_loc = np.where(pd_aorta_seq['sum'].values > 0.)[0]
    np_idx = pd_aorta_seq['idx'].values[np_loc]

    np_aorta_seq = pd_aorta_seq['idx'].to_numpy(int)

    lst_lsq_spline = []

    for w, key_w in enumerate(opt_keys):
        np_mat_w = pd_aorta_seq[key_w].values[np_loc]

        if key_w == 'ka':
            mat_max_w = 0.35
            mat_min_w = 0.

        elif key_w == 'alpha':
            np_mat_w = np.rad2deg(np_mat_w)
            mat_max_w = np.rad2deg(np.pi / 2.)
            mat_min_w = 0.

        else:
            mat_max_w = np.around(1.5 * np_mat_w.max(), 4)
            mat_min_w = np.around(0.5 * np_mat_w.min(), 4)

        ymin_ticks = mat_max_w / 10.
        ymax_ticks = 4. * ymin_ticks

        ymajor_ticks_w = np.arange(0., mat_max_w, ymax_ticks)
        yminor_ticks_w = np.arange(0., mat_max_w, ymin_ticks)

        if np_mat_w.shape[0] > 4:
            spl_w = interpolate.UnivariateSpline(np_idx, np_mat_w, s=4)
            lsq_spl_w = interpolate.LSQUnivariateSpline(np_idx, np_mat_w, spl_w.get_knots()[1:-1])
            lst_lsq_spline.append(lsq_spl_w)

        ax[w].set_xticks(list_xticks[1])
        ax[w].set_xticks(list_xticks[0], minor=True)
        ax[w].set_xlim([-0.5, np_aorta_seq.max() + 0.5])

        ax[w].set_yticks(ymajor_ticks_w)
        ax[w].set_yticks(yminor_ticks_w, minor=True)
        ax[w].set_ylim([mat_min_w, mat_max_w])

        ax[w].grid(which='minor', alpha=0.2)
        ax[w].grid(which='major', alpha=0.6)

        ax[w].plot(np_idx, np_mat_w, label=key_w)
        ax[w].plot(np_idx, np_mat_w, 'ro', label='Observations')

        if np_mat_w.shape[0] > 4:
            ax[w].plot(list_xticks[0], spl_w(list_xticks[0]), label='LSQ - spline')

        ax[w].legend()

        if key_w == 'alpha':
            ax[w].set_ylabel('alpha (deg)')
        else:
            ax[w].set_ylabel(key_w)

        ax[w].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax[-1].set_xlabel('Aorta Section Idx')
    ax[-1].set_xticklabels(label_x)

    return fig


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


class PlotSolution2D:
    def __init__(self):
        """
        Initialize the PlotSolution2D class.

        Parameters:
        - title (str): The title of the plot.
        - ltype (dict): Dictionary mapping keys to line styles.
        - post_equations (dict, optional): Dictionary of equations to display.
        """

        # Labels for the plot
        self.ltx_energy = r"$\psi$"
        self.ltx_stress_x = r"$\sigma_x$"
        self.ltx_stress_y = r"$\sigma_y$"
        self.ltx_stress_z = r"$\sigma_z$"

        self.colors = {'iso': 'blue', 'vol': 'orange', 'ani': 'green', 'total': 'red'}

        # Initialize lists for legend handles and labels
        self.all_handles = []
        self.all_labels = []

        # Equation text increment position
        self.eq_inc = 4.5

    def _create_2d_plot(self, fontsize: int = 14):

        fig, ax = plt.subplots(2, 2, figsize=(24, 18), dpi=800)
        fl_ax = ax.ravel()

        ax[0, 0].set_xlabel(r'Stretch Ratio, $l_x$', fontsize=fontsize)
        ax[0, 0].set_ylabel(r'Strain Energy, $\psi (F)$', fontsize=fontsize)

        ax[0, 1].set_xlabel(r'Stretch Ratio, $l_x$', fontsize=fontsize)
        ax[0, 1].set_ylabel(f'Engineering Stress, {self.ltx_stress_x}', fontsize=fontsize)

        ax[1, 0].invert_xaxis()
        ax[1, 0].set_xlabel(r'Stretch Ratio, $l_y$', fontsize=fontsize)
        ax[1, 0].set_ylabel(f'Engineering Stress, {self.ltx_stress_y}', fontsize=fontsize)

        ax[1, 1].invert_xaxis()
        ax[1, 1].set_xlabel(r'Stretch Ratio, $l_z$', fontsize=fontsize)
        ax[1, 1].set_ylabel(f'Engineering Stress, {self.ltx_stress_z}', fontsize=fontsize)

        for ax_i in fl_ax:
            ax_i.axvline(x=1.0, color='k', linestyle=":")
            ax_i.axhline(y=0.0, color='r', linestyle=":")
            ax_i.grid(which='minor', alpha=0.2)
            ax_i.grid(which='major', alpha=0.5)

        dict_ax = {'ese': ax[0, 0], 'x': ax[0, 1], 'y': ax[1, 0], 'z': ax[1, 1]}

        return fig, ax, dict_ax

    @staticmethod
    def _create_force_plot(fontsize: int = 14):

        fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=400)

        ax.set_xlabel(r'Stretch Ratio, $l_x$', fontsize=fontsize)
        ax.set_ylabel(r'Force in $x$-axis', fontsize=fontsize)

        ax.axvline(x=1.0, color='k', linestyle=":")
        ax.axhline(y=0.0, color='r', linestyle=":")
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        return fig, ax

    def components_plot(self,
                        title: str,
                        results: dict,
                        ltype: dict,
                        fname: str,
                        post_equations: dict = None,
                        ):

        if post_equations is None:
            post_equations = {}

        fig, ax, dict_ax = self._create_2d_plot(fontsize=16)
        fig.suptitle(title, fontsize=20)

        eq_inc = 4.5
        all_handles, all_labels = [], []

        for i, (key_i, post_solu_i) in enumerate(results.items()):
            lx_i = post_solu_i.stretch[:, 0]
            ly_i = post_solu_i.stretch[:, 1]
            lz_i = post_solu_i.stretch[:, 2]
            ltype_i = ltype[key_i]

            for stype_k, color_k in zip(['iso', 'vol', 'ani'], ['blue', 'orange', 'green']):
                # Plot the strain energy contributions for iso, vol, ani in subplot 1
                kwargs_k = {'linestyle': ltype_i, 'color': color_k, 'alpha': 0.7, 'label': f'{key_i} ({stype_k})'}

                line_k, = dict_ax["ese"].plot(lx_i, post_solu_i.ese[stype_k], **kwargs_k)
                all_handles.append(line_k)
                all_labels.append(line_k.get_label())

                # Plot the engineering stress contributions in subplot 2 (sigma_x) and 3 (sigma_y)
                dict_ax["x"].plot(lx_i, post_solu_i.stress[stype_k][:, 0], **kwargs_k)
                dict_ax["y"].plot(ly_i, post_solu_i.stress[stype_k][:, 1], **kwargs_k)
                dict_ax["z"].plot(lz_i, post_solu_i.stress[stype_k][:, 2], **kwargs_k)

            # Add equations if provided
            if post_equations.get(key_i) is not None:
                ax[0].text(1.1, eq_inc, f'{self.ltx_energy} - {key_i} = ${post_equations.get(key_i)}$', fontsize=13,
                           ha='left')
                eq_inc -= 1.

        # Create a single legend outside the plots at the bottom
        fig.legend(handles=all_handles, labels=all_labels, loc="lower center", ncol=6, fontsize=12,
                   bbox_to_anchor=(0.5, 0.01))

        # Save figure
        fig.savefig(fname, dpi=300)
        plt.close(fig)

    def force_plot(self,
                    title: str,
                    results: OptimizeResult,
                    fname: str,
                    ):

        fig, ax = self._create_force_plot(fontsize=16)
        fig.suptitle(title, fontsize=20)

        np_lx = results.stretch[:, 0]
        np_fx_r = results.xforce
        np_fint = results.fint[:, 0]

        ax.plot(np_lx, np_fint, color="red", alpha=0.7, label='model')
        ax.plot(np_lx, np_fx_r, "o", color="black", alpha=0.7, label='experiment')

        # Create a single legend outside the plots at the bottom
        fig.legend(loc="lower center", fontsize=12, bbox_to_anchor=(0.5, 0.01))

        # Save figure
        fig.savefig(fname, dpi=300)
        plt.close(fig)

    def full_plot(self,
                  title: str,
                  results: dict,
                  ltype: dict,
                  fname: str,
                  post_equations: dict = None,
                  ):

        if post_equations is None:
            post_equations = {}

        fig, ax, dict_ax = self._create_2d_plot(fontsize=16)
        fig.suptitle(title, fontsize=20)

        eq_inc = 4.5
        all_handles, all_labels = [], []

        for i, (key_i, post_solu_i) in enumerate(results.items()):
            lx_i = post_solu_i.stretch[:, 0]
            ly_i = post_solu_i.stretch[:, 1]
            lz_i = post_solu_i.stretch[:, 2]
            ltype_i = ltype[key_i]

            for stype_k in ['iso', 'vol', 'ani']:

                if stype_k == 'ani':
                    kwargs_k = {'linestyle': ltype_i, 'color': "green", 'alpha': 0.7, 'label': f'{key_i} ({stype_k})'}

                    line_k, = dict_ax["ese"].plot(lx_i, post_solu_i.ese[stype_k], **kwargs_k)
                    dict_ax["x"].plot(lx_i, post_solu_i.stress[stype_k][:, 0], **kwargs_k)
                    dict_ax["y"].plot(ly_i, post_solu_i.stress[stype_k][:, 1], **kwargs_k)
                    dict_ax["z"].plot(ly_i, post_solu_i.stress[stype_k][:, 2], **kwargs_k)

                    all_handles.append(line_k)
                    all_labels.append(line_k.get_label())

            line_i, = dict_ax["ese"].plot(lx_i, post_solu_i.ese['total'], linestyle=ltype_i, color="red", label=f'{key_i}')

            all_handles.append(line_i)
            all_labels.append(line_i.get_label())

            # Plot the engineering stress contributions in subplot 2 (sigma_x) and 3 (sigma_y)
            kwargs_i = {'linestyle': ltype_i, 'color': "red", 'alpha': 0.7, 'label': f'{key_i}'}

            dict_ax["x"].plot(lx_i, post_solu_i.stress['full'][:, 0], **kwargs_i)
            dict_ax["y"].plot(ly_i, post_solu_i.stress['full'][:, 1], **kwargs_i)
            dict_ax["z"].plot(lz_i, post_solu_i.stress['full'][:, 2], **kwargs_i)

            # Add equations if provided
            if post_equations.get(key_i) is not None:
                ax[0].text(1.1, eq_inc, f'{self.ltx_energy} - {key_i} = ${post_equations.get(key_i)}$', fontsize=13,
                           ha='left')
                eq_inc -= 1.

        # Create a single legend outside the plots at the bottom
        fig.legend(handles=all_handles, labels=all_labels, loc="lower center", ncol=6, fontsize=12,
                   bbox_to_anchor=(0.5, 0.01))

        # Save figure
        fig.savefig(fname, dpi=300)
        plt.close(fig)