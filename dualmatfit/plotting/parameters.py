# -*- coding: utf-8 -*-
"""
Plotting parameters and constants.

This module provides shared constants, color palettes, and configuration
values for consistent styling across all plotting functions.
"""
import matplotlib.pyplot as plt

__all__ = [
    'COLORS',
    'NAME_SECTIONS',
    'SEGMENT_LINESTYLES',
    'RATS_STYLES',
    'DEFAULT_PLOT_LIMITS',
    'stress_dim',
    'rats_ids',
    'DEFAULT_STYLE',
    'DEFAULT_DPI',
    'DEFAULT_FIGSIZE',
    'DEFAULT_MULTI_FIGSIZE',
    'ticks_fontsize',
    'label_fontsize',
]

# Consistent color palette (example using seaborn's default)
# FIXME: Improve this line to avoid warnings
# COLORS = plt.cm.get_cmap('tab10').colors

# More info: https://matplotlib.org/stable/gallery/color/named_colors.html
cred = ['firebrick', 'red', 'tomato', 'crimson']
cgreen = ['green', 'olive', 'springgreen', 'forestgreen']
cblue = ['blue', 'royalblue', 'navy', 'cornflowerblue']
corange = ['bisque', 'darkorange', 'goldenrod', 'orange']
cblack = ['black', 'dimgray', 'darkgrey', 'slategray']
cpink = ['magenta', 'plum', 'deeppink', 'orchid']
cbrown = ['sienna', 'peru', 'chocolate', 'saddlebrown']

stress_dim = "MPa"

SEGMENT_LINESTYLES = {'A': '-', 'B': '--', 'C': '-.'}
STRESS_DIM_LABEL = "KPa"

NAME_SECTIONS = {'Ar': 'AoA', 'Tr': 'DTAo', 'Ab': 'DAAo'}

# Consistent color palette (example using seaborn's default)
# Ensure COLORS is defined or remove if not used directly by PlotHelper methods
try:
    COLORS = plt.colormaps['tab10'].colors
except (AttributeError, KeyError):
    # Fallback for older matplotlib versions
    COLORS = plt.cm.get_cmap('tab10').colors

rats_ids = {"rato_17": {'id': 1, 'color': cred[1]},
            "rato_23": {'id': 2, 'color': cpink[0]},
            # New rats
            "rato_wt_184085": {'id': 4, 'color': corange[3]},
            "rato_wt_184012": {'id': 3, 'color': cpink[3]},
            "rato_wt_183964": {'id': 10, 'color': cblue[3]},
            "rato_wt_183997": {'id': 4, 'color': cblue[3]},
            "rato_wt_183918": {'id': 11, 'color': cbrown[3]},
            }

RATS_STYLES = {
    'rato_17': {'color': cred[1], 'id_prefix': 'Rat-1'},
    'rato_23': {'color': cpink[0], 'id_prefix': 'Rat-2'},
    # New rats
    'rato_wt_184085': {'color': corange[3], 'id_prefix': 'Rat-4'},
    'rato_wt_184012': {'color': cpink[3], 'id_prefix': 'Rat-3'},
    'rato_wt_183997': {'color': cblue[3], 'id_prefix': 'Rat-5'},
}

# Default plot limits, can be overridden
DEFAULT_PLOT_LIMITS = {
    'lx': (0.95, 2.0), # Stretch
    'force': (-0.1, 1.5), # Force [N]
    'stress_total': (-0.2, 3.5), # Total Stress [KPa]
    'stress_iso': (-1.5, 1.0),   # Isotropic Stress [KPa]
    'stress_vol': (-0.5, 0.5),   # Volumetric Stress [KPa]
    'stress_ani': (-0.2, 4.2),   # Anisotropic Stress [KPa]
}

# --- Plotting Configuration from plot.py ---
DEFAULT_STYLE = 'seaborn-v0_8-whitegrid'
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (10, 6)  # Default figure size for single plots
DEFAULT_MULTI_FIGSIZE = (12, 10)  # Default for multi-panel plots

ticks_fontsize = 20
label_fontsize = 20