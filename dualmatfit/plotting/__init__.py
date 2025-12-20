# -*- coding: utf-8 -*-
"""
Plotting submodule for DualMatFit.

This submodule provides visualization functions for experimental data,
analytical model results, and material fitting diagnostics.
"""

from dualmatfit.plotting.experimental_visuals import (
    plot_raw_signals,
    plot_material_fit,
)
from dualmatfit.plotting.analytical_visuals import (
    plot_optimization_history,
    plot_segment_force_curves,
    plot_segment_stress_curves,
    plot_mean_stress_curves,
    plot_curves_from_xlsx,
)
from dualmatfit.plotting.plot_helpers import (
    PlotHelper,
    get_colors,
    get_x_stretch,
    get_y_stretch,
    plt_assign_sec,
    set_axis_labels,
    set_axis_ticks,
)
from dualmatfit.plotting.parameters import (
    COLORS,
    NAME_SECTIONS,
    SEGMENT_LINESTYLES,
    RATS_STYLES,
    DEFAULT_PLOT_LIMITS,
)

__all__ = [
    # Experimental visuals
    'plot_raw_signals',
    'plot_material_fit',
    # Analytical visuals
    'plot_optimization_history',
    'plot_segment_force_curves',
    'plot_segment_stress_curves',
    'plot_mean_stress_curves',
    'plot_curves_from_xlsx',
    # Helpers
    'PlotHelper',
    'get_colors',
    'get_x_stretch',
    'get_y_stretch',
    'plt_assign_sec',
    'set_axis_labels',
    'set_axis_ticks',
    # Parameters
    'COLORS',
    'NAME_SECTIONS',
    'SEGMENT_LINESTYLES',
    'RATS_STYLES',
    'DEFAULT_PLOT_LIMITS',
]
