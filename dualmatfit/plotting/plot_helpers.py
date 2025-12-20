# -*- coding: utf-8 -*-
"""
Plotting helper utilities.

This module provides helper classes and functions for creating
consistent, publication-quality plots across the package.
"""
# import os
import colorsys
import numpy as np
import pandas as pd
# import sympy as sy

from pathlib import Path
# from scipy import interpolate
# from scipy.optimize import OptimizeResult
from typing import Optional, Union, Dict, List, Any, Tuple

# import matplotlib
import matplotlib.axes as axes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

from dualmatfit.logging_config import get_logger
from dualmatfit.numeric_utils import sanitize_array, has_nan
logger = get_logger('plotting')

__all__ = [
    'PlotHelper',
    'get_colors',
    'get_x_stretch',
    'get_y_stretch',
    'plt_assign_sec',
    'set_axis_labels',
    'set_axis_ticks',
]

# Use Agg backend for non-interactive plotting (suitable for saving files)
# matplotlib.use("Agg")

# --- Plotting Configuration ---
DEFAULT_STYLE = 'seaborn-v0_8-whitegrid'
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (10, 6) # Default figure size for single plots
DEFAULT_MULTI_FIGSIZE = (12, 10) # Default for multi-panel plots



def set_axis_labels(ax: plt.Axes, xlabel: str, ylabel: str):
    """
    Set labels and grid for an axis.

    Parameters:
    - ax (plt.Axes): Axis to set labels on.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)


def set_axis_ticks(ax: plt.Axes, xdata: np.ndarray, ydata: np.ndarray):
    """
    Set major and minor ticks for an axis based on data.

    Parameters:
    - ax (plt.Axes): Axis to set ticks on.
    - xdata (np.ndarray): Data for x-axis.
    - ydata (np.ndarray): Data for y-axis.
    """
    # Ensure data is not empty
    if xdata.size == 0 or ydata.size == 0:
        return

    # Handle cases with constant data to avoid tick errors
    if xdata.min() == xdata.max():
        x_major_ticks = np.array([xdata.min()])
        x_minor_ticks = np.array([xdata.min()])
    else:
        x_major_ticks = np.linspace(xdata.min(), xdata.max(), num=10)
        # Avoid creating minor ticks outside data range if min/max very close
        if xdata.max() - xdata.min() > 1e-9:
             x_minor_ticks = np.linspace(xdata.min(), xdata.max(), num=50)
        else:
             x_minor_ticks = x_major_ticks

    if ydata.min() == ydata.max():
         y_major_ticks = np.array([ydata.min()])
         y_minor_ticks = np.array([ydata.min()])
    else:
        y_major_ticks = np.linspace(ydata.min(), ydata.max(), num=10)
         # Avoid creating minor ticks outside data range if min/max very close
        if ydata.max() - ydata.min() > 1e-9:
             y_minor_ticks = np.linspace(ydata.min(), ydata.max(), num=50)
        else:
             y_minor_ticks = y_major_ticks

    ax.set_xticks(x_major_ticks)
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_yticks(y_major_ticks)
    ax.set_yticks(y_minor_ticks, minor=True)


def get_x_stretch(ly: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore'):
        return sanitize_array(1. / ly)


def get_y_stretch(lx: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore'):
        return sanitize_array(1. / lx)


def plt_assign_sec(ax, pos_x, pos_y, sec_name, color_in, size: float = 300):

    if isinstance(ax, axes.Axes):
        ax.scatter(pos_x, pos_y, s=size, marker="o", zorder=10, clip_on=False, linewidth=1, edgecolor=color_in,
                   facecolor="white")

        ax.text(pos_x, pos_y, sec_name, zorder=20, color=color_in, ha="center", va="center", size="medium",
                clip_on=False)


def color_distance(hsv1, hsv2):
    # Calculate the Euclidean distance between two HSV tuples
    return sum((a - b) ** 2 for a, b in zip(hsv1, hsv2)) ** 0.5


def get_colors():
    # Convert CSS4 colors from hex to HSV
    css4_hsvs = {name: colorsys.rgb_to_hsv(*mcolors.hex2color(hex)) for name, hex in mcolors.CSS4_COLORS.items()}

    # Sort colors by hue, saturation, and value to get a diverse starting point
    sorted_hsvs = sorted(css4_hsvs.items(), key=lambda item: (item[1][0], item[1][1], item[1][2]))

    # Initialize the list of distinct colors with a color with high saturation and value
    distinct_colors = [sorted_hsvs[-1]]  # Start with the color with the highest saturation and value

    # Set minimum perceptual distance we want between colors
    min_dist = 0.25  # Threshold for color distance

    for name, hsv in sorted_hsvs:
        # Check the color distance from the current color to all colors in the distinct list
        if all(color_distance(hsv, distinct_color[1]) > min_dist for distinct_color in distinct_colors):
            distinct_colors.append((name, hsv))

    # Convert the distinct HSV colors back to hex for matplotlib
    return [mcolors.rgb2hex(colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])) for name, hsv in distinct_colors]


def format_value(value, precision):
    """Formats a numerical value to a string with a given precision."""
    if pd.isna(value):
        return ""  # Returns an empty string for NaN values
    return f"{value:.{precision}f}"


class PlotHelper:
    """Handles common plotting tasks like setup, styling, and saving."""

    def __init__(self,
                 style: str = DEFAULT_STYLE,
                 dpi: int = DEFAULT_DPI,
                 use_latex: bool = False,
                 ):

        self.style = style
        self.dpi = dpi
        self.use_latex = use_latex
        try:
            plt.style.use(self.style)
        except OSError:
            logger.debug(f"Warning: Style '{self.style}' not found. Using default style.")
            plt.style.use('default')

        # Apply some global rcParams for consistency if needed
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 14
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.5
        plt.rcParams['grid.linestyle'] = '--'

        if self.use_latex:
            try:
                preamble = plt.rcParams.get('text.latex.preamble', '')
                if isinstance(preamble, str):
                    preamble = [line for line in preamble.split('\n') if line.strip()]
                amsmath_pkg = r'\usepackage{amsmath}'
                if amsmath_pkg not in preamble:
                    preamble.append(amsmath_pkg)
                    plt.rcParams['text.latex.preamble'] = "\n".join(preamble)
                plt.rcParams['text.usetex'] = True
                plt.rcParams['font.family'] = 'serif'
                plt.rcParams['font.serif'] = ["Latin Modern Roman"]
                logger.debug("Matplotlib configured to use LaTeX with amsmath.")
            except (RuntimeError, FileNotFoundError, OSError) as e:
                logger.debug(f"Warning: Could not configure LaTeX ({e}). Falling back to default text rendering.")
                plt.rcParams['text.usetex'] = False
                self.use_latex = False
        else:
            plt.rcParams['text.usetex'] = False

    @staticmethod
    def setup_figure(nrows: int = 1,
                     ncols: int = 1,
                     figsize: Optional[Tuple[float, float]] = None,
                     sharex: bool = False,
                     sharey: bool = False,
                     dpi: int = DEFAULT_DPI,
                     grid: bool = True,
                     **gridspec_kw,
                     ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """Creates a figure and axes with consistent sizing."""

        if figsize is None:
            figsize = DEFAULT_MULTI_FIGSIZE if nrows * ncols > 1 else DEFAULT_FIGSIZE

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey, dpi=dpi,
                                 gridspec_kw=gridspec_kw,
                                 )

        if isinstance(axes, list):
            axes = np.array(axes)

        if nrows == 1 and ncols == 1 and not isinstance(axes, np.ndarray):
             axes = np.array([axes]) # Ensure it's always an array
        elif (nrows == 1 or ncols == 1) and isinstance(axes, np.ndarray): # Ensure 1D array for single row/col > 1
             axes = axes.flatten()

        if grid:
            for ax in axes:
                ax.grid(which='minor', alpha=0.5)
                ax.grid(which='major', alpha=0.8)

        return fig, axes

    def set_labels_title(self,
                         ax: plt.Axes,
                         xlabel: str = None,
                         ylabel: str = None,
                         title: str = None,
                         xlabel_fontsize: int = 12,
                         ylabel_fontsize: int = 12,
                         title_fontsize: int = 14,
                         ):
        """Sets standardized labels and title."""
        if xlabel is not None:
            ax.set_xlabel(self._escape_latex(xlabel), fontsize=xlabel_fontsize)
        if ylabel is not None:
            ax.set_ylabel(self._escape_latex(ylabel), fontsize=ylabel_fontsize)
        if title is not None:
            ax.set_title(self._escape_latex(title), fontsize=title_fontsize)

    def _escape_latex(self, text: str) -> str:
        """Basic escaping for common LaTeX special characters."""
        # This is a simplified escape function. A more robust one might be needed.
        # Do not escape if already inside math mode $...$
        if not self.use_latex or '$' in text:
             return text

        chars = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
            '\\': r'\textbackslash{}',
        }

        return "".join([chars.get(c, c) for c in text])

    @staticmethod
    def set_limits_ticks(ax: plt.Axes,
                         xdata: Optional[np.ndarray] = None,
                         ydata: Optional[np.ndarray] = None,
                         xlims: Optional[Tuple]=None,
                         ylims: Optional[Tuple]=None,
                         x_major_n: int = 10,
                         y_major_n: int = 10,
                         ):
        """Sets limits and potentially ticks based on data or explicit limits."""
        if xlims:
            ax.set_xlim(xlims)
        elif xdata is not None and len(xdata) > 0:
            xmin, xmax = np.nanmin(xdata), np.nanmax(xdata)
            if not has_nan(np.array([xmin])) and not has_nan(np.array([xmax])):
                if xmin == xmax: # Handle constant data
                    ax.set_xlim(xmin - 0.5, xmax + 0.5) # Add some padding
                else:
                    ax.set_xlim(xmin, xmax)

        if ylims:
            ax.set_ylim(ylims)
        elif ydata is not None and len(ydata) > 0:
             ymin, ymax = np.nanmin(ydata), np.nanmax(ydata)
             if not has_nan(np.array([ymin])) and not has_nan(np.array([ymax])):
                if ymin == ymax: # Handle constant data
                    padding = abs(ymin) * 0.1 + 0.1 if ymin != 0 else 0.1
                    ax.set_ylim(ymin - padding, ymax + padding)
                else:
                    padding = (ymax - ymin) * 0.05
                    ax.set_ylim(ymin - padding, ymax + padding)

    @staticmethod
    def set_axis_ticks(ax: plt.Axes,
                       x_major: np.ndarray,
                       x_minor: np.ndarray,
                       y_major: np.ndarray,
                       y_minor: np.ndarray,
                       ):
        """
        Set major and minor ticks for an axis based on data.
        """
        ax.set_xticks(x_major)
        ax.set_xticks(x_minor, minor=True)
        ax.set_yticks(y_major)
        ax.set_yticks(y_minor, minor=True)
        ax.grid(which='minor', alpha=0.5)
        ax.grid(which='major', alpha=0.8)

    def save_plot(self,
                  fig: plt.Figure,
                  filename: str,
                  save_dir: str,
                  ):
        """Saves the plot to the specified directory."""

        if not filename:
            logger.warning(" No filename provided for saving plot.")
            plt.close(fig)
            return

        save_path = Path(save_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.debug(f"Plot saved to: {save_path}")
        except (IOError, PermissionError, OSError) as e:
            logger.debug(f"Error saving plot to {save_path}: {e}")
        finally:
            plt.close(fig) # Close figure to free memory

    def setup_multi_region_plot(self,
                                regions: List[str],
                                main_title: str,
                                y_labels_map: Dict[str, str],
                                x_label_common: str,
                                figsize: Optional[Tuple[float, float]] = None,
                                sharex: bool = True,
                                dpi: Optional[int] = None
                                ) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
        """
        Sets up a figure with multiple subplots, one for each specified region.

        Args:
            regions (List[str]): List of region keys (e.g., ['Ar', 'Tr', 'Ab']).
            main_title (str): The main title for the entire figure.
            y_labels_map (Dict[str, str]):  Dictionary mapping region keys to their y-axis labels.
            x_label_common (str):           The common x-axis label for the bottom-most plot.
            figsize (Optional[Tuple[float, float]]): Figure size.
            sharex (bool): Whether to share the x-axis among subplots.
            dpi (Optional[int]): DPI for the figure, overrides instance DPI if provided.

        Returns:
            Tuple[plt.Figure, Dict[str, plt.Axes]]: The figure object and a dictionary
                                                    mapping region keys to their respective Axes objects.
        """
        n_regions = len(regions)
        if n_regions == 0:
            raise ValueError("Regions list cannot be empty.")

        fig, list_axes = self.setup_figure(nrows=n_regions, ncols=1, figsize=figsize, sharex=sharex, dpi=dpi)

        if n_regions == 1:  # Ensure axes_list is always a list/array of Axes
            list_axes = [list_axes[0]]

        fig.suptitle(self._escape_latex(main_title), fontsize=20)

        axes_dict = {}
        for i, region_key in enumerate(regions):
            ax_i = list_axes[i]
            axes_dict[region_key] = ax_i
            self.set_labels_title(ax_i, ylabel=y_labels_map.get(region_key, ""))
            ax_i.axvline(x=1.0, color='k', linestyle=":", linewidth=0.8)  # Common vertical line at x=1
            ax_i.axhline(y=0.0, color='k', linestyle=":", linewidth=0.8)  # Common horizontal line at y=0

            # Setup Grid Plot
            ax_i.grid(True, which='major', linestyle='-', alpha=0.5)
            ax_i.grid(True, which='minor', linestyle=':', alpha=0.7)

            ax_i.tick_params(axis='x', labelsize=14)
            ax_i.tick_params(axis='y', labelsize=14)

        # Set common x-label only for the last subplot if sharex is True
        if sharex and n_regions > 0:
            self.set_labels_title(list_axes[-1], xlabel=x_label_common)
        elif not sharex and n_regions > 0:  # If not sharing x, set for all
            for ax_i in list_axes:
                self.set_labels_title(ax_i, xlabel=x_label_common)

        return fig, axes_dict

    def plot_series_on_ax(self,
                          ax: plt.Axes,
                          series_definitions: List[Dict[str, Any]],
                          data_context: Dict[str, Any]  # To pass rat_id, segment_detail etc. for styling
                          ) -> Tuple[List[plt.Line2D], List[str]]:
        """
        Plots multiple data series on a given axis.

        Args:
            ax (plt.Axes): The matplotlib Axes object to plot on.
            series_definitions (List[Dict[str, Any]]): A list of dictionaries,
                where each dictionary defines a data series with keys like:
                'data_source_key': Key to access the data object from `data_context`.
                'x_extractor': Callable (data_object) -> np.ndarray (x-data).
                'y_extractor': Callable (data_object) -> np.ndarray (y-data).
                'plot_kwargs': Dict for ax.plot() (e.g., color, linestyle, label).
                               The 'label' will be used for the legend.
            data_context (Dict[str, Any]): A dictionary providing context,
                e.g., {'experimental_data': instron_obj, 'model_data': model_solution_obj}.
                The 'data_source_key' in series_definitions will look up objects here.


        Returns:
            Tuple[List[plt.Line2D], List[str]]: Lists of legend handles and labels.
        """
        handles, labels = [], []
        for series_def in series_definitions:
            data_source = data_context.get(series_def['data_source_key'])
            if data_source is None:
                logger.debug(f"Warning: Data source key '{series_def['data_source_key']}' not found in data_context.")
                continue

            x_data = series_def['x_extractor'](data_source)
            y_data = series_def['y_extractor'](data_source)

            current_plot_kwargs = series_def.get('plot_kwargs', {}).copy()
            label = current_plot_kwargs.pop('label', None)

            if x_data is not None and y_data is not None and x_data.size > 0 and y_data.size > 0:
                # Ensure data is 1D for plotting
                if x_data.ndim > 1: x_data = x_data.flatten()
                if y_data.ndim > 1: y_data = y_data.flatten()

                if x_data.size != y_data.size:
                    logger.debug(
                        f"Warning: Mismatch in x_data ({x_data.size}) and y_data ({y_data.size}) sizes for label '{label}'. Skipping plot for this series.")
                    continue

                line, = ax.plot(x_data, y_data, **current_plot_kwargs)
                if label:
                    handles.append(line)
                    labels.append(self._escape_latex(label))
            else:
                logger.debug(f"Warning: Empty data for series with intended label '{label}'. Skipping.")

        return handles, labels

    def finalize_multi_region_plot(self,
                                   fig: plt.Figure,
                                   axes_dict: Dict[str, plt.Axes],
                                   common_x_limits: Optional[Tuple[float, float]],
                                   region_y_limits: Dict[str, Tuple[float, float]],
                                   legend_config: Optional[Dict[str, Any]] = None,
                                   save_dir: Union[str, Path] = None,
                                   filename: str = "multi_region_plot.png"
                                   ):
        """
        Finalizes a multi-region plot: sets limits, adds legends, adjusts layout, and saves.

        Args:
            fig (plt.Figure): The figure object.
            axes_dict (Dict[str, plt.Axes]): Dictionary mapping region keys to their Axes.
            common_x_limits (Optional[Tuple[float, float]]): Common x-axis limits for all subplots.
            region_y_limits (Dict[str, Tuple[float, float]]): Y-axis limits for each region's subplot.
            legend_config (Optional[Dict[str, Any]]): Configuration for the legend.
                Example: {'handles': [], 'labels': [], 'loc': 'best', 'ncol': 1, 'ax_key_for_legend': 'Ar'}
                If 'ax_key_for_legend' is provided, legend is placed on that specific axis.
                Otherwise, a figure-level legend can be attempted (more complex).
                If None, no legend is added by this function.
            save_dir (Union[str, Path], optional): Directory to save the plot. If None, plot is not saved by this function.
            filename (str): Filename for the saved plot.
        """
        for region_key, ax in axes_dict.items():
            if common_x_limits:
                ax.set_xlim(common_x_limits)
            if region_key in region_y_limits:
                ax.set_ylim(region_y_limits[region_key])

            # Apply tick formatting (example, can be customized)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        if legend_config and legend_config.get('handles') and legend_config.get('labels'):
            ax_key = legend_config.get('ax_key_for_legend')
            if ax_key and ax_key in axes_dict:
                axes_dict[ax_key].legend(
                    legend_config['handles'],
                    legend_config['labels'],
                    loc=legend_config.get('loc', 'best'),
                    ncol=legend_config.get('ncol', 1),
                    fontsize=legend_config.get('fontsize', plt.rcParams['legend.fontsize'])
                )
            elif len(axes_dict) > 0:  # Fallback to first axis if specific key not found or not provided
                logger.debug(
                    f"Warning: Legend axis key '{ax_key}' not found or not specified. Placing legend on first available axis.")
                first_ax = next(iter(axes_dict.values()))
                first_ax.legend(
                    legend_config['handles'],
                    legend_config['labels'],
                    loc=legend_config.get('loc', 'best'),
                    ncol=legend_config.get('ncol', 1),
                    fontsize=legend_config.get('fontsize', plt.rcParams['legend.fontsize'])
                )

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for main title

        if save_dir:
            self.save_plot(fig, filename, save_dir)
        else:
            plt.show()
