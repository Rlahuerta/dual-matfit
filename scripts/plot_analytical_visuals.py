# -*- coding: utf-8 -*-
"""
Utilities and CLI entrypoint for plotting analytical curves from fitted parameters.

This script rebuilds analytical force and stress curves from the shared
XLSX/HDF5 runtime inputs and writes the plot outputs to the configured plot
directory.

Typical CLI usage::

    conda run -n matfit1d python scripts/plot_analytical_visuals.py
    conda run -n matfit1d python scripts/plot_analytical_visuals.py --output-dir Results/M3-nh-ka-vol-glb/plots
    conda run -n matfit1d python scripts/plot_analytical_visuals.py --ncontrol 15

Typical library usage::

    plot_curves_from_xlsx(
        h5_path="path/to/final_data.h5",
        xlsx_path="Results/...xlsx",
        output_dir="Results/.../plots",
        ncontrol=15,
    )

The CLI defaults assume a repository checkout that already contains the
experimental HDF5 file and generated XLSX outputs.
"""
import argparse

from pathlib import Path
from typing import Any, Optional, Sequence

from dualmatfit.utils.logging_config import get_logger
from dualmatfit.plotting.analytical_visuals import (
    generate_plot_data_from_xlsx,
    plot_mean_stress_curves,
    plot_segment_force_curves,
    plot_segment_stress_curves,
)
try:
    from scripts._analytical_runtime import (
        DEFAULT_ANALYTICAL_NCONTROL,
        build_analytical_var_form_config,
        default_analytical_run_paths,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _analytical_runtime import (
        DEFAULT_ANALYTICAL_NCONTROL,
        build_analytical_var_form_config,
        default_analytical_run_paths,
    )


logger = get_logger('plotting')


def plot_curves_from_xlsx(
        h5_path: str | Path,
        xlsx_path: str | Path,
        output_dir: str | Path,
        ncontrol: int,
        config_name: str = 'global_params_plot',
        var_form_cfg: Optional[dict[str, Any]] = None,
) -> None:
    """
    Read fitted parameters, regenerate analytical curves, and save the plots.

    This is the main reusable API when the plotting workflow needs to be called
    from another Python module instead of from the command line.
    """
    vf_config = build_analytical_var_form_config(var_form_cfg)

    logger.debug(f"Generating plot data for '{config_name}' from: {xlsx_path}")
    plot_data = generate_plot_data_from_xlsx(
        h5_path=h5_path,
        xlsx_path=xlsx_path,
        var_form_config=vf_config,
        ncontrol=ncontrol,
        list_rats=None,
        rerun=False,
    )

    if not plot_data:
        logger.debug(f"No data generated for plotting from {xlsx_path}")
        return

    logger.debug(f"Plotting data for '{config_name}'")
    plot_segment_force_curves(
        data_by_region=plot_data,
        output_dir=output_dir,
        config_name=config_name,
    )
    plot_segment_stress_curves(
        data_by_region=plot_data,
        output_dir=output_dir,
        config_name=config_name,
    )
    plot_mean_stress_curves(
        data_by_region=plot_data,
        output_dir=output_dir,
        config_name=config_name,
    )


def _default_run_paths() -> tuple[Path, Path, Path]:
    """Return the shared default analytical input and output paths."""

    return default_analytical_run_paths()


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the analytical plotting entrypoint."""

    default_h5_file, default_xlsx_file, default_output_plot_dir = _default_run_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5-path', default=str(default_h5_file))
    parser.add_argument('--xlsx-path', default=str(default_xlsx_file))
    parser.add_argument('--output-dir', default=str(default_output_plot_dir))
    parser.add_argument('--ncontrol', type=int, default=DEFAULT_ANALYTICAL_NCONTROL)
    parser.add_argument(
        '--config-name',
        default='glb_ipopt_params_mix3_was_isosplit',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """
    Run the end-to-end analytical plotting workflow from the command line.

    Examples
    --------
    Generate plots with repository defaults::

        conda run -n matfit1d python scripts/plot_analytical_visuals.py

    Write plots to a custom directory::

        conda run -n matfit1d python scripts/plot_analytical_visuals.py --output-dir /tmp/analytical-plots
    """

    parser = _build_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_curves_from_xlsx(
        h5_path=args.h5_path,
        xlsx_path=args.xlsx_path,
        output_dir=output_dir,
        ncontrol=args.ncontrol,
        config_name=args.config_name,
        var_form_cfg=None,
    )


if __name__ == '__main__':
    main()