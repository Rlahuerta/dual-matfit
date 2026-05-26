"""
Shared analytical-runtime helpers for script entrypoints.

This module centralizes the XLSX/HDF5 reconstruction workflow shared by
``scripts/plot_analytical_visuals.py`` and
``scripts/analyze_beta_identifiability.py`` so both scripts use the same:

- default input paths
- variational-form configuration
- section reconstruction logic
- section-level ``CostFunction`` bootstrap

Typical usage for plotting::

    h5_path, xlsx_path, output_dir = default_analytical_run_paths()
    sections = load_analytical_sections(h5_path=h5_path, xlsx_path=xlsx_path)

Typical usage for one section analysis::

    sections = load_analytical_sections(
        h5_path="path/to/final_data.h5",
        xlsx_path="Results/...xlsx",
        rats=["rato_17"],
        sections=["Ar-A"],
    )
    section = filter_analytical_sections(sections, {"rato_17": {"Ar": ["A"]}})[0]
    cost_function = build_section_cost_function(section)

The repository-local defaults returned by ``default_analytical_run_paths`` are
intended for a source checkout that already contains the experimental HDF5 file
and generated results workbook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from dualmatfit.data.experimental import InstronData, MaterialSetup
from dualmatfit.utils.io_utils import load_excel_params, load_hdf5_data
from dualmatfit.optimization.cost import CostFunction
from dualmatfit.utils.logging_config import get_logger
from dualmatfit.plotting.analytical_visuals import DEFAULT_VAR_FORM_CONFIG, bulk_val
from dualmatfit.data.rato_info import excel_data
from dualmatfit.formulation.variational import VariationalFormulation, ring_geom

logger = get_logger("analytical-runtime")

DEFAULT_ANALYTICAL_RUN_VAR_FORM_CONFIG: dict[str, Any] = {
    "itype": "nh",
    "mix": 3,
    "kappa": True,
    "dvol": True,
    "iso_split": False,
    "vol_type": "simo92",
    "hv": False,
    "was": True,
}

DEFAULT_ANALYTICAL_NCONTROL = 15


@dataclass(slots=True)
class AnalyticalSection:
    """
    Reconstructed analytical inputs for one fitted section.

    Instances are produced by :func:`load_analytical_sections` and then consumed
    by higher-level scripts that either plot curves or run identifiability
    diagnostics on the reconstructed section.
    """

    rat_sheet_id: str
    rat_id: str
    region_id: str
    position_id: str
    section_id: str
    baseline_parameters: pd.Series
    fitted_parameters: pd.Series
    info_data: dict[str, Any]
    instron_data: InstronData
    var_form: VariationalFormulation
    var_form_config: dict[str, Any]


def default_analytical_run_paths(project_root: str | Path | None = None) -> tuple[Path, Path, Path]:
    """
    Return the default HDF5, XLSX, and plot-output paths for analytical scripts.

    Use this to keep script defaults synchronized across plotting and analysis
    entrypoints.
    """

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
    root = root.resolve()
    xlsx_file = root / "Results" / "M3-nh-ka-vol-glb" / "glb_opt_mat_param_ipopt_v01.xlsx"
    h5_file = root / "instron_data" / "final_data.h5"
    output_plot_dir = xlsx_file.parent / "plots"
    return h5_file.resolve(), xlsx_file.resolve(), output_plot_dir.resolve()


def build_analytical_var_form_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build the shared variational-form configuration used by analytical scripts.

    ``overrides`` can be used by callers that need to tweak one or two values
    while preserving the repository defaults.
    """

    config = DEFAULT_VAR_FORM_CONFIG.copy()
    config.update(DEFAULT_ANALYTICAL_RUN_VAR_FORM_CONFIG)
    if overrides:
        config.update(overrides)
    return config


def _normalize_sheet_rat_id(rat_id: str) -> str:
    """Normalize a rat identifier to the underscore sheet naming convention."""

    return rat_id.lstrip("/").replace("-", "_")


def _normalize_section_id(section_id: str) -> str:
    """Normalize a section identifier to the canonical ``Region-Position`` form."""

    return section_id.lstrip("/").replace("_", "-")


def _load_parameter_tables(xlsx_path: str | Path) -> dict[str, pd.DataFrame]:
    """Load fitted parameter tables from the analytical XLSX workbook."""

    parameter_tables = load_excel_params(xlsx_path)
    if parameter_tables is None:
        raise FileNotFoundError(f"Could not load analytical parameter workbook: {xlsx_path}")
    return parameter_tables


def _load_experimental_tables(h5_path: str | Path) -> dict[str, pd.DataFrame]:
    """Load experimental section data from the shared HDF5 file."""

    experimental_tables = load_hdf5_data(h5_path)
    if experimental_tables is None:
        raise FileNotFoundError(f"Could not load analytical experimental data: {h5_path}")
    return experimental_tables


def load_analytical_sections(
    *,
    h5_path: str | Path,
    xlsx_path: str | Path,
    var_form_config: dict[str, Any] | None = None,
    ncontrol: int = DEFAULT_ANALYTICAL_NCONTROL,
    rats: Sequence[str] | None = None,
    sections: Sequence[str] | None = None,
) -> list[AnalyticalSection]:
    """
    Reconstruct fitted analytical sections from the shared XLSX/HDF5 inputs.

    This is the main loader for script-level workflows:

    1. read fitted parameters from the XLSX workbook;
    2. read experimental traces from the HDF5 file;
    3. merge them with section metadata from :func:`dualmatfit.rato_info.excel_data`;
    4. build :class:`AnalyticalSection` objects ready for plotting or analysis.

    Parameters
    ----------
    h5_path, xlsx_path:
        Input files used by the analytical plotting and identifiability scripts.
    var_form_config:
        Optional overrides on top of the shared runtime defaults.
    ncontrol:
        Number of control points passed into :class:`InstronData`.
    rats:
        Optional subset of rats to load, using identifiers such as ``rato_17``
        or ``rato-17``.
    sections:
        Optional subset of section identifiers to load, using names such as
        ``Ar-A`` or ``Ar_A``. The filter is applied before reconstructing the
        per-section experimental and variational objects.
    """

    config = build_analytical_var_form_config(var_form_config)
    parameter_tables = _load_parameter_tables(xlsx_path)
    experimental_tables = _load_experimental_tables(h5_path)
    metadata = excel_data()

    requested_sheet_ids = ({_normalize_sheet_rat_id(rat_id) for rat_id in rats}
        if rats is not None
        else None
    )
    requested_section_ids = ({_normalize_section_id(section_id) for section_id in sections}
        if sections is not None
        else None
    )

    loaded_sections: list[AnalyticalSection] = []

    for rat_sheet_id, parameter_table in parameter_tables.items():
        if requested_sheet_ids is not None and rat_sheet_id not in requested_sheet_ids:
            continue
        if rat_sheet_id not in experimental_tables:
            logger.debug(f"Experimental data for '{rat_sheet_id}' is missing from {h5_path}")
            continue

        rat_id = rat_sheet_id.replace("_", "-")
        rat_metadata = metadata.get(rat_id)
        if rat_metadata is None:
            logger.debug(f"Metadata for '{rat_id}' is missing from rato_info.excel_data()")
            continue

        df_sample_data = experimental_tables[rat_sheet_id]
        sample_columns = list(df_sample_data.columns)
        parameter_table = parameter_table.rename(columns={"bulk": "D"})

        section_metadata = {
            region_id: rat_metadata.get(region_id)
            for region_id in ["Ar", "Tr", "Ab"]
            if rat_metadata.get(region_id) is not None
        }

        for region_id, region_info in section_metadata.items():
            region_sections = region_info.copy()
            for position_id in ["A", "B", "C"]:
                if region_info.get(position_id) is not None:
                    ring_geom(region_sections, position_id)
            region_sections.pop("dia", None)
            region_sections.pop("thick", None)

            for position_id, section_info in region_sections.items():
                section_id = f"{region_id}-{position_id}"
                if requested_section_ids is not None and section_id not in requested_section_ids:
                    continue
                if section_id not in parameter_table.index:
                    continue

                baseline_parameters = parameter_table.loc["baseline", :].copy()
                fitted_parameters = parameter_table.loc[section_id, :].copy()
                parameter_sum = pd.to_numeric(fitted_parameters, errors="coerce").fillna(0.0).sum()
                if parameter_sum <= 0.0:
                    continue

                column_mask = np.array([section_id in column_name for column_name in sample_columns], dtype=bool)
                if not column_mask.any():
                    logger.debug(f"Experimental columns for '{rat_sheet_id}:{section_id}' are missing")
                    continue

                info_data = section_info.copy()
                info_data.setdefault("sample_id", f"{rat_id}-{section_id}")
                instron_data = InstronData(
                    df_data=df_sample_data.iloc[:, column_mask],
                    info_data=info_data,
                    ncontrol=ncontrol,
                )

                var_form_kwargs = config.copy()
                var_form_kwargs["ds"] = info_data["ds"]
                var_form = VariationalFormulation(**var_form_kwargs)

                loaded_sections.append(
                    AnalyticalSection(
                        rat_sheet_id=rat_sheet_id,
                        rat_id=rat_id,
                        region_id=region_id,
                        position_id=position_id,
                        section_id=section_id,
                        baseline_parameters=baseline_parameters,
                        fitted_parameters=fitted_parameters,
                        info_data=info_data,
                        instron_data=instron_data,
                        var_form=var_form,
                        var_form_config=config.copy(),
                    )
                )

    loaded_sections.sort(key=lambda section: (section.rat_id, section.section_id))
    return loaded_sections


def filter_analytical_sections(
    sections: Sequence[AnalyticalSection],
    selection: dict[str, dict[str, list[str]]],
) -> list[AnalyticalSection]:
    """
    Filter reconstructed sections with a manuscript-style selection mapping.

    The ``selection`` format matches structures such as
    ``{"rato_17": {"Ar": ["A", "B"]}}``.
    """

    return [
        section
        for section in sections
        if section.rat_sheet_id in selection
        and section.region_id in selection[section.rat_sheet_id]
        and section.position_id in selection[section.rat_sheet_id][section.region_id]
    ]


def build_section_cost_function(
    section: AnalyticalSection,
    *,
    module: str = "jax",
    dtype: str = "adjoint",
    cache_size: int | None = 128,
) -> CostFunction:
    """
    Build a section-level :class:`CostFunction` from reconstructed fit inputs.

    This is the bridge from the shared reconstruction layer into numerical
    workflows such as the identifiability analysis. The fitted XLSX parameters
    are copied into the material variable table before the ``CostFunction`` is
    created.
    """

    if section.instron_data.np_tload_ref is None or section.instron_data.np_tstretch_ref is None:
        raise ValueError(f"Section '{section.section_id}' is missing reference load/stretch data")

    material_setup = MaterialSetup(
        itype=str(section.var_form_config["itype"]),
        bulk=bulk_val,
        dvol=bool(section.var_form_config["dvol"]),
        kappa=bool(section.var_form_config["kappa"]),
    )
    dsvars, _, _, _ = material_setup()
    dsvars = dsvars.copy()

    baseline_parameters = pd.to_numeric(
        section.baseline_parameters.rename(index={"bulk": "D"}),
        errors="coerce",
    )

    local_parameters = pd.to_numeric(
        section.fitted_parameters.rename(index={"bulk": "D"}),
        errors="coerce",
    )

    valid_parameters = local_parameters[local_parameters.notna()]
    common_keys = [key for key in dsvars.index if key in valid_parameters.index]

    if common_keys:
        dsvars.loc[common_keys, "ini"] = valid_parameters.loc[common_keys].to_numpy(dtype=float)
        dsvars.loc[common_keys, "values"] = valid_parameters.loc[common_keys].to_numpy(dtype=float)
        dsvars.loc[common_keys, "baseline"] = baseline_parameters.loc[common_keys].to_numpy(dtype=float)

    return CostFunction(
        var_form=section.var_form,
        load_ref=section.instron_data.np_tload_ref,
        stretch_x=section.instron_data.np_tstretch_ref,
        dsvars=dsvars,
        module=module,
        dtype=dtype,
        cache_size=cache_size,
    )