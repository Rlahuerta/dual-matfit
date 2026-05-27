# -*- coding: utf-8 -*-
"""
Shared constants for the material fitting pipeline.

These constants are used by ``material_fit.py`` and its mixin modules
(``fit_optimization``, ``fit_persistence``, ``fit_visualization``,
``fit_covariance``).
"""

__all__ = [
    'DEFAULT_BULK_MODULUS',
    'DEFAULT_VOLUMETRIC_TYPE',
    'DEFAULT_NUM_CONTROL_POINTS',
    'DEFAULT_SIMPLIFY_TIMEOUT',
    'EXPERIMENTAL_DATA_COLUMNS_PER_SECTION',
    'SECTION_CODE_LENGTH',
    'POSITION_CODE_INDEX',
    'RAT_KEY_PREFIX_LENGTH',
    'UNSTRETCHED_STATE',
    'DEFAULT_FIGURE_SIZE',
    'DEFAULT_DPI',
    'GRID_ALPHA_MINOR',
    'GRID_ALPHA_MAJOR',
    'DEFAULT_PLOT_LIMITS',
    'ALTERNATIVE_PLOT_LIMITS',
    'PLOT_MIN_SCALE',
    'PLOT_MAX_SCALE',
    'PLOT_FALLBACK_SCALE',
    'PLOT_ENERGY_THRESHOLD',
    'PLOT_TICK_Y_DIVISOR',
    'PLOT_TICK_Y_MULTIPLIER',
    'PLOT_MIN_LIMIT',
    'TICK_MAJOR_DIVISOR',
    'TICK_MINOR_SUBDIVISIONS',
    'TICK_LABEL_TRIM',
    'DEFAULT_BASELINE_OPTIMIZATION_ITERATIONS',
    'DEFAULT_LOCAL_OPTIMIZATION_ITERATIONS',
    'DEFAULT_GLOBAL_ITERATIONS',
    'DEFAULT_SOLUTION_NCONTROL',
    'HIGH_RESOLUTION_NCONTROL',
]

# ============================================================================
# CONSTANTS AND CONFIGURATION

# Material Properties
# Reference: "On the Compressibility of Arterial Tissue"
# Bulk modulus range: 42.14-99.03 kPa
DEFAULT_BULK_MODULUS = 56.67 / 1000.0  # Median Value [MPa] = 0.05667 MPa

# Volumetric Strain Energy Types
# Options: 'bathe87', 'simo92', 'doll8' (works for fung model)
DEFAULT_VOLUMETRIC_TYPE = 'simo92'

# ============================================================================
# Fitting Configuration

DEFAULT_NUM_CONTROL_POINTS = 15  # Number of control points for material fitting
DEFAULT_SIMPLIFY_TIMEOUT = 1     # Symbolic simplification timeout (seconds)

# Data Structure Constants
EXPERIMENTAL_DATA_COLUMNS_PER_SECTION = 3  # Each section has 3 columns: force, stretch, position
SECTION_CODE_LENGTH = 2  # Length of section code (e.g., "Ar", "Tr", "Ab")
POSITION_CODE_INDEX = -1  # Index of position code (A, B, or C) in section key
RAT_KEY_PREFIX_LENGTH = 1  # Length of prefix to strip from HDF5 keys (leading "/")

# Physical Constants
UNSTRETCHED_STATE = 1.0  # Stretch ratio lambda = 1.0 (no deformation)

# ============================================================================
# Plot Configuration
DEFAULT_FIGURE_SIZE = (14, 10)  # Figure size (width, height) in inches
DEFAULT_DPI = 120  # Resolution for saved figures
GRID_ALPHA_MINOR = 0.2  # Transparency for minor grid lines
GRID_ALPHA_MAJOR = 0.6  # Transparency for major grid lines

# Plot Limits (MPa)
DEFAULT_PLOT_LIMITS = {
    'iso': [-1.5, 1.0],    # Isotropic stress limits
    'ani': [-0.2, 4.2],    # Anisotropic stress limits
    'sum': [-0.2, 3.5],    # Total stress limits
    'lx': [0.95, 2.0]      # Stretch ratio limits
}

# Alternative plot limits for different scenarios
ALTERNATIVE_PLOT_LIMITS = {
    'iso': [-1.2, 1.4],
    'ani': [-0.2, 3.6],
    'sum': [-0.2, 3.0],
    'lx': [0.9, 1.8]
}

# Plot Scaling Factors
PLOT_MIN_SCALE = 0.5            # Scale factor for minimum plot limits
PLOT_MAX_SCALE = 1.5            # Scale factor for maximum plot limits
PLOT_FALLBACK_SCALE = 0.1       # Fallback scale when minimum is near zero
PLOT_ENERGY_THRESHOLD = 0.05    # Threshold for minimum energy values
PLOT_TICK_Y_DIVISOR = 10.0      # Divisor for y-axis tick spacing
PLOT_TICK_Y_MULTIPLIER = 4.0    # Multiplier for minor tick spacing
PLOT_MIN_LIMIT = 0.01           # Minimum value for plot limits

# Tick Configuration
TICK_MAJOR_DIVISOR = 16.0       # Divisor for major tick spacing along aorta sections
TICK_MINOR_SUBDIVISIONS = 4     # Number of subdivisions between major ticks
TICK_LABEL_TRIM = -1            # Index to trim last label from tick labels

# Optimization Configuration
DEFAULT_BASELINE_OPTIMIZATION_ITERATIONS = 50           # Default iterations for baseline parameter optimization
DEFAULT_LOCAL_OPTIMIZATION_ITERATIONS = 100             # Default iterations for local parameter optimization
DEFAULT_GLOBAL_ITERATIONS = 1  # Default basin-hopping iterations
DEFAULT_SOLUTION_NCONTROL = 15  # Number of control points for solution evaluation
HIGH_RESOLUTION_NCONTROL = 100  # Number of control points for high-resolution plots