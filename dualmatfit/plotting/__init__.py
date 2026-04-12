# -*- coding: utf-8 -*-
"""
Plotting submodule for DualMatFit.

This submodule provides visualization functions for experimental data,
analytical model results, and material fitting diagnostics.
"""

import importlib

__all__ = [
    # Experimental visuals
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
    # Analytical visuals
    'plot_optimization_history',
    'plot_segment_force_curves',
    'plot_segment_stress_curves',
    'plot_mean_stress_curves',
    'plot_curves_from_xlsx',
    'ese_plot',
    'mat_plot',
    # Solution visuals
    'PlotSolution2D',
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

_MODULE_MAP = {
    # Experimental visuals
    'plot_raw_signals': 'experimental_visuals',
    'plot_material_fit': 'experimental_visuals',
    'stress_plot': 'experimental_visuals',
    'exp_test_plot': 'experimental_visuals',
    'plot_time_extension': 'experimental_visuals',
    'plot_time_load': 'experimental_visuals',
    'plot_extension_load': 'experimental_visuals',
    'plot_reaction_force': 'experimental_visuals',
    'plot_volume_change': 'experimental_visuals',
    'plot_pk1_stress': 'experimental_visuals',
    # Analytical visuals
    'plot_optimization_history': 'analytical_visuals',
    'plot_segment_force_curves': 'analytical_visuals',
    'plot_segment_stress_curves': 'analytical_visuals',
    'plot_mean_stress_curves': 'analytical_visuals',
    'plot_curves_from_xlsx': 'analytical_visuals',
    'ese_plot': 'analytical_visuals',
    'mat_plot': 'analytical_visuals',
    # Solution visuals
    'PlotSolution2D': 'solution_visuals',
    # Helpers
    'PlotHelper': 'plot_helpers',
    'get_colors': 'plot_helpers',
    'get_x_stretch': 'plot_helpers',
    'get_y_stretch': 'plot_helpers',
    'plt_assign_sec': 'plot_helpers',
    'set_axis_labels': 'plot_helpers',
    'set_axis_ticks': 'plot_helpers',
    # Parameters
    'COLORS': 'parameters',
    'NAME_SECTIONS': 'parameters',
    'SEGMENT_LINESTYLES': 'parameters',
    'RATS_STYLES': 'parameters',
    'DEFAULT_PLOT_LIMITS': 'parameters',
}


def __getattr__(name):
    if name in _MODULE_MAP:
        module = importlib.import_module(f".{_MODULE_MAP[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")