"""
Utilities and CLI entrypoint for Reviewer B Comment 5 identifiability analysis.

The workflow reconstructs section-level analytical models from the same XLSX/HDF5
inputs used by ``scripts/plot_analytical_visuals.py`` and then evaluates the local
conditioning between the fiber-angle parameter ``alpha`` (manuscript ``beta``) and
the stiffness parameter ``k_1``.

Typical CLI usage::

    conda run -n matfit1d python scripts/analyze_beta_identifiability.py
    conda run -n matfit1d python scripts/analyze_beta_identifiability.py --rat-id rato_17 --section Ar-A
    conda run -n matfit1d python scripts/analyze_beta_identifiability.py --csv-out reviews/metrics.csv
    conda run -n matfit1d python scripts/analyze_beta_identifiability.py --quarto

Quarto usage::
    quarto render identifiability_paper.qmd --to pdf 2>&1 | tail -30

Typical library usage::

    sections = load_sections(h5_path="path/to/final_data.h5", xlsx_path="Results/...xlsx")
    row = analyze_section(sections[0])
    write_reports([row], csv_out=Path("reviews/metrics.csv"), md_out=Path("reviews/report.md"))

The CLI defaults assume a repository checkout that already contains the
experimental HDF5 file and generated XLSX outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from scipy.stats import chi2
from typing import Any, Sequence, TypedDict, cast
from dualmatfit.fitting.covariance import robust_covariance_from_cost, CovarianceReport
# from dualmatfit.fitting.identifiability import ConditioningReport, analyze_cost_integrator
from dualmatfit.optimization.cost import CostIntegrator
from dualmatfit.plotting.parameters import RATS_STYLES

matplotlib.use("Agg")

try:
    from scripts._analytical_runtime import (
        DEFAULT_ANALYTICAL_NCONTROL,
        AnalyticalSection,
        build_section_cost_function,
        default_analytical_run_paths,
        filter_analytical_sections,
        load_analytical_sections,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _analytical_runtime import (
        DEFAULT_ANALYTICAL_NCONTROL,
        AnalyticalSection,
        build_section_cost_function,
        default_analytical_run_paths,
        filter_analytical_sections,
        load_analytical_sections,
    )


class IdentifiabilityCostConfig(TypedDict):
    """Configuration used to build the reviewer-oriented ``CostIntegrator``."""

    ftype: str
    c: float
    alpha: float
    beta: float
    epsilon: float
    rescale: str | None
    dvol: bool


MANUSCRIPT_SELECTION = {
    "rato_17": {"Ar": ["A", "B", "C"], "Tr": ["A", "B"], "Ab": ["A", "B", "C"]},
    "rato_23": {"Ar": ["A", "C"], "Tr": ["A", "B", "C"], "Ab": ["A", "B", "C"]},
    "rato_wt_184012": {"Ar": ["A", "B", "C"], "Tr": ["A", "B", "C"], "Ab": ["A", "B", "C"]},
    "rato_wt_184085": {"Ar": ["A", "B", "C"], "Tr": ["A", "B", "C"], "Ab": ["A", "B", "C"]},
    "rato_wt_183997": {"Ar": ["A", "B", "C"], "Tr": ["A", "B", "C"], "Ab": ["A", "B", "C"]},
}

IDENTIFIABILITY_COST_CONFIG: IdentifiabilityCostConfig = {
    "ftype": "cauchy_robust",
    "c": 40.0,
    "alpha": 0.001,
    "beta": 1.0,
    "epsilon": 1.0e-3,
    "rescale": None,
    "dvol": True,
}


def normalize_rat_id(rat_id: str) -> str:
    """Normalize a rat identifier to the hyphenated report format."""

    return rat_id.lstrip("/").replace("_", "-")


def normalize_section_id(section_id: str) -> str:
    """Normalize a section identifier so CLI and report formats match."""

    return section_id.replace("_", "-")


def selection_from_filters(
    rat_id: str | None = None,
    section_id: str | None = None,
) -> dict[str, dict[str, list[str]]]:
    """
    Build a manuscript-style selection mapping from optional CLI filters.

    Use this when narrowing the analysis to one rat or one section. With both
    arguments omitted, the full ``MANUSCRIPT_SELECTION`` is returned.
    """

    if rat_id is None:
        return MANUSCRIPT_SELECTION

    selection_rat_id = rat_id.lstrip("/").replace("-", "_")
    section_selection = MANUSCRIPT_SELECTION[selection_rat_id]

    if section_id is None:
        return {selection_rat_id: section_selection}

    normalized_section_id = normalize_section_id(section_id)
    section_name, position_code = normalized_section_id.split("-", maxsplit=1)
    if position_code not in section_selection[section_name]:
        raise KeyError(
            f"Section {normalized_section_id} is not part of the manuscript selection for {selection_rat_id}"
        )

    return {selection_rat_id: {section_name: [position_code]}}


# Anatomical order: Ar (AoA) → Tr (DTAo) → Ab (DAAo)
_REGION_ORDER: dict[str, int] = {"Ar": 0, "Tr": 1, "Ab": 2}


def selection_section_ids(selection: dict[str, dict[str, list[str]]]) -> list[str]:
    """Extract section identifiers in anatomical order (Ar → Tr → Ab).

    Returns sections ordered by region (proximal to distal) then by position (A, B, C).
    """

    section_ids = {
        f"{region_id}-{position_id}"
        for rat_selection in selection.values()
        for region_id, position_ids in rat_selection.items()
        for position_id in position_ids
    }
    # Sort by anatomical order, then by position (A, B, C)
    return sorted(
        section_ids,
        key=lambda s: (
            _REGION_ORDER.get(s.split("-")[0], 99),  # Region order (Ar=0, Tr=1, Ab=2)
            s.split("-")[1] if "-" in s else "",    # Position (A, B, C)
        ),
    )


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Display name mappings (code → manuscript notation)
# ---------------------------------------------------------------------------

_PARAM_DISPLAY: dict[str, str] = {
    "alpha": "\u03b2",   # β
    "mu": "\u03bc",      # μ
    "D": "D",
    "k_1": "k\u2081",   # k₁
    "k_2": "k\u2082",   # k₂
    "kappa": "\u03ba",   # κ
}

_REGION_DISPLAY: dict[str, str] = {"Ar": "AoA", "Tr": "DTAo", "Ab": "DAAo"}


def _get_rat_display_name(rat_id: str) -> str:
    """Get the display name for a rat from RATS_STYLES.

    Maps internal rat IDs (e.g., 'rato-17') to manuscript-style display names
    (e.g., 'Rat-1') used in plots and tables.

    Args:
        rat_id: The rat identifier (e.g., 'rato-17', 'rato-wt-184085')

    Returns:
        The display name from RATS_STYLES (e.g., 'Rat-1'), or the rat_id if not found.
    """
    # Convert hyphenated ID to underscore format for lookup
    rat_key = rat_id.replace("-", "_")
    if rat_key in RATS_STYLES:
        return RATS_STYLES[rat_key]['id_prefix']
    return rat_id


def _display_param(code_name: str) -> str:
    """Map code parameter name to manuscript symbol (e.g. ``'alpha'`` → ``'β'``)."""
    return _PARAM_DISPLAY.get(code_name, code_name)


def _display_section(section_id: str) -> str:
    """Map ``'Ar-A'`` → ``'AoA-A'``, ``'Tr-B'`` → ``'DTAo-B'``."""
    parts = section_id.split("-", maxsplit=1)
    if len(parts) == 2:
        return f"{_REGION_DISPLAY.get(parts[0], parts[0])}-{parts[1]}"
    return section_id


def _fmt_val(x: float, sig: int = 4) -> str:
    """Format a float with *sig* significant figures."""
    if not np.isfinite(x):
        return "\u2014"
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) > 1e4:
        return f"{x:.{sig - 1}e}"
    return f"{x:.{sig}g}"


def _fmt_ci(lower: float, upper: float) -> str:
    """Format a confidence interval as ``[lower, upper]``."""
    return f"[{_fmt_val(lower)}, {_fmt_val(upper)}]"


def report_to_row(
    rat_id: str,
    section_id: str,
    fit_kind: str,
    report: CovarianceReport,
    *,
    report_unreg: CovarianceReport | None = None,
) -> dict[str, object]:
    """Flatten a :class:`CovarianceReport` into one CSV/report row.
    """

    row: dict[str, object] = {
        "rat_id": rat_id,
        "section_id": section_id,
        "fit_kind": fit_kind,
        "param_names": ",".join(report.param_names),
        "hessian_matrix": report.hessian_matrix,
        "hessian_diagonal": report.hessian_diagonal,
        "condition_number": report.hessian_condition,
        "eigenvalues": report.eigenvalues,
        "covariance_matrix": report.covariance_matrix,
        "standard_errors": report.standard_errors,
        "confidence_intervals": report.confidence_interval,
        "confidence_level": report.confidence_level,
        "correlation_matrix": report.correlation_matrix,
        "method": report.method,
        "polished": report.polished,
        "calibrated": report.calibrated,
        "n_function_evals": report.n_function_evals,
    }

    if report_unreg is not None:
        row["condition_number_unreg"] = report_unreg.hessian_condition
        row["standard_errors_unreg"] = report_unreg.standard_errors
        row["confidence_intervals_unreg"] = report_unreg.confidence_interval
        row["covariance_matrix_unreg"] = report_unreg.covariance_matrix
        row["hessian_diagonal_unreg"] = report_unreg.hessian_diagonal
        row["hessian_matrix_unreg"] = report_unreg.hessian_matrix
        row["calibrated_unreg"] = report_unreg.calibrated

    return row


def render_markdown_summary(rows: list[dict[str, object]], rat_name: str | None = None) -> str:
    """Render the reviewer-facing Markdown summary from collected report rows.

    Produces a structured report with eight sections addressing Reviewer B
    Comment 5 on fiber-angle (β) identifiability.

    Parameters
    ----------
    rows : list[dict[str, object]]
        Analysis results from :func:`collect_rows`.
    rat_name : str, optional
        Display name for the rat (e.g., 'rato-17'). If provided, included
        in the report title and interpretation section.

    References
    ----------
    .. [1] White, H. (1980). A heteroskedasticity-consistent covariance
       matrix estimator and a direct test for heteroskedasticity.
       *Econometrica*, 48(4), 817–838.
    """

    lines: list[str] = []
    _append_header(lines, rat_name)
    _append_methodology(lines)
    _append_beta_ci_table(lines, rows)
    _append_full_ci_table(lines, rows)
    _append_correlation_table(lines, rows)
    _append_conditioning_table(lines, rows)
    _append_hessian_diagonal_table(lines, rows)
    _append_hessian_plots_section(lines, rows, rat_name)
    _append_interpretation(lines, rows, rat_name)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Private helpers for render_markdown_summary
# ---------------------------------------------------------------------------


def _append_header(lines: list[str], rat_name: str | None = None) -> None:
    title = "# Fiber Angle (\u03b2) Identifiability Analysis"
    if rat_name:
        title += f" — {rat_name}"
    lines.extend([
        title,
        "",
        "**Response to Reviewer B, Comment 5**",
        "",
        "> With only uniaxial extension data, the structural identifiability of the fiber angle",
        "> \u03b2 is mathematically compromised. The Hessian matrix of the cost function is likely",
        "> ill-conditioned with respect to \u03b2 and k\u2081. Did you calculate the covariance matrix or",
        "> confidence intervals for \u03b2? Without this, the reported regional variations in fiber",
        "> angle may just be numerical noise rather than true biological variation.",
        "",
    ])


def _append_methodology(lines: list[str]) -> None:
    lines.extend([
        "## 1. Methodology",
        "",
        "The covariance analysis pipeline is evaluated at the optimized parameter values",
        "for each specimen section:",
        "",
        "1. **Objective function**: Cauchy robust loss (c = 40) with volumetric",
        "   regularization. The Tikhonov weight \u03c9 = 0.001 is applied for the",
        "   primary analysis; a second analysis with \u03c9 = 0 isolates the data",
        "   contribution.",
        "2. **Hessian estimation**: Ridders\u2019 extrapolation with adaptive step sizes",
        "   (Baker, 2013). Eigenvalue polish via Richardson extrapolation with",
        "   `numpy.linalg.eigh` for symmetry.",
        "3. **NPD calibration**: Huang et al. (2017) \u03ba\u209c\u2090\u2093\u2091\u2091\u209c-bounded threshold",
        "   when the Hessian is not positive definite.",
        "4. **Sandwich estimator**: V = H\u207b\u00b9 B H\u207b\u00b9 where B is the",
        "   outer-product (meat) matrix (White, 1980; Huber & Ronchetti, 2009).",
        "5. **Confidence intervals**: Wald-type \u03be \u00b1 t_{\u03b1/2, dof} \u00d7 SE,",
        "   dof = n_obs \u2212 n_params, clipped to parameter bounds",
        "   (Seber & Wild, 2003 \u00a75.2).",
        "6. **Dual analysis**: both with \u03c9 = 0.001 and \u03c9 = 0 to separate",
        "   data-driven identifiability from regularization effects.",
        "",
    ])


def _append_beta_ci_table(lines: list[str], rows: list[dict[str, object]]) -> None:
    """Confidence intervals for both β and k₁ (the two parameters of interest)."""
    _FOCUS = [("alpha", "\u03b2"), ("k_1", "k\u2081")]
    # Build header (without Rat column since report is per-rat)
    hdr_parts = ["Section"]
    for _, sym in _FOCUS:
        hdr_parts.extend([f"{sym} (value)", "95% CI", f"SE({sym})", "Rel. SE (%)"])
    header = "| " + " | ".join(hdr_parts) + " |"
    sep = "|" + "|".join(["---"] * len(hdr_parts)) + "|"

    lines.extend([
        "## 2. Fiber Angle (\u03b2) and Stiffness (k\u2081) Confidence Intervals",
        "",
        header,
        sep,
    ])

    se_ratios: dict[str, list[float]] = {"alpha": [], "k_1": []}
    for row in rows:
        ci_df = row.get("confidence_intervals")
        if ci_df is None or not isinstance(ci_df, pd.DataFrame):
            continue
        se_arr = row.get("standard_errors")
        se_series = (
            pd.Series(se_arr, index=ci_df.index)
            if se_arr is not None and isinstance(se_arr, np.ndarray)
            else None
        )

        cells: list[str] = [
            _display_section(str(row["section_id"])),
        ]
        for code, _ in _FOCUS:
            if code not in ci_df.index:
                cells.extend(["\u2014"] * 4)
                continue
            val = float(ci_df.loc[code, "value"])
            lo = float(ci_df.loc[code, "lower"])
            hi = float(ci_df.loc[code, "upper"])
            se = float(se_series[code]) if se_series is not None else (hi - lo) / 3.92
            rel_se = (se / abs(val)) * 100 if abs(val) > 1e-12 else float("inf")
            se_ratios[code].append(rel_se)
            cells.extend([_fmt_val(val), _fmt_ci(lo, hi), _fmt_val(se), f"{rel_se:.1f}"])
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    for code, sym in _FOCUS:
        if se_ratios[code]:
            median_rel = float(np.median(se_ratios[code]))
            if median_rel < 10:
                quality = "well-determined"
            elif median_rel < 30:
                quality = "moderately determined"
            else:
                quality = "poorly determined"
            lines.append(
                f"**{sym}**: Across {len(se_ratios[code])} sections, median relative SE "
                f"= {median_rel:.1f}% \u2192 {quality}."
            )
    lines.append("")


def _append_full_ci_table(lines: list[str], rows: list[dict[str, object]]) -> None:
    lines.extend([
        "## 3. Full Parameter Confidence Intervals",
        "",
    ])
    if not rows:
        lines.append("*No data available.*\n")
        return

    # Determine parameter order from first row's CI DataFrame
    first_ci = rows[0].get("confidence_intervals")
    if first_ci is None or not isinstance(first_ci, pd.DataFrame):
        lines.append("*No confidence interval data available.*\n")
        return

    param_order = list(first_ci.index)
    header_params = " | ".join(_display_param(p) for p in param_order)
    lines.append(f"| Section | {header_params} |")
    lines.append("|---------|" + " | ".join(["---"] * len(param_order)) + " |")

    for row in rows:
        ci_df = row.get("confidence_intervals")
        if ci_df is None or not isinstance(ci_df, pd.DataFrame):
            continue
        cells = []
        for p in param_order:
            if p in ci_df.index:
                v = float(ci_df.loc[p, "value"])
                lo = float(ci_df.loc[p, "lower"])
                hi = float(ci_df.loc[p, "upper"])
                cells.append(f"{_fmt_val(v)} {_fmt_ci(lo, hi)}")
            else:
                cells.append("\u2014")
        lines.append(
            f"| {_display_section(str(row['section_id']))} | "
            + " | ".join(cells) + " |"
        )
    lines.append("")


def _append_correlation_table(lines: list[str], rows: list[dict[str, object]]) -> None:
    """Per-section square correlation matrices with subsections."""
    lines.extend([
        "## 4. Parameter Correlation Analysis",
        "",
        "Full correlation matrices are reported per section.  Off-diagonal entries",
        "close to ±1 indicate strong linear coupling between parameters.",
        "",
    ])
    if not rows:
        lines.append("*No data available.*\n")
        return

    rho_beta_k1: list[float] = []
    for row in rows:
        corr = row.get("correlation_matrix")
        ci_df = row.get("confidence_intervals")
        if corr is None or ci_df is None or not isinstance(ci_df, pd.DataFrame):
            continue
        params = list(ci_df.index)
        section_id = str(row["section_id"])
        rat_id = str(row["rat_id"])

        rat_display = _get_rat_display_name(rat_id)
        lines.append(f"### {rat_display} / {_display_section(section_id)}")
        lines.append("")

        # Header row
        header = "| |" + " | ".join(_display_param(p) for p in params) + " |"
        sep = "|---|" + " | ".join(["---"] * len(params)) + " |"
        lines.append(header)
        lines.append(sep)

        for i, pi in enumerate(params):
            cells = [f"**{_display_param(pi)}**"]
            for j, pj in enumerate(params):
                if isinstance(corr, pd.DataFrame):
                    rho = float(corr.loc[pi, pj])
                elif isinstance(corr, np.ndarray):
                    rho = float(corr[i, j])
                else:
                    cells.append("\u2014")
                    continue
                if i == j:
                    cells.append("1.000")
                else:
                    cells.append(f"{rho:.3f}")
                # Collect beta-k1
                if (pi, pj) == ("alpha", "k_1"):
                    rho_beta_k1.append(abs(rho))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    if rho_beta_k1:
        mean_abs_rho = float(np.mean(rho_beta_k1))
        if mean_abs_rho > 0.8:
            coupling = "strong"
        elif mean_abs_rho > 0.5:
            coupling = "moderate"
        else:
            coupling = "weak"
        lines.append(
            f"**Key finding**: The mean |\u03c1(\u03b2, k\u2081)| is {mean_abs_rho:.3f}, "
            f"indicating {coupling} coupling between fiber angle and stiffness."
        )
        lines.append("")


def _append_conditioning_table(lines: list[str], rows: list[dict[str, object]]) -> None:
    lines.extend([
        "## 5. Hessian Conditioning Diagnostics",
        "",
    ])
    has_unreg = any("condition_number_unreg" in row for row in rows)

    # -- Summary table --
    header = "| Section | \u03ba(H) reg. |"
    sep = "|---------|-----------|"
    if has_unreg:
        header += " \u03ba(H) unreg. |"
        sep += "-------------|"
    header += " \u03bb_min | \u03bb_max | Polished | Calibrated | Evals |"
    sep += "-------|-------|----------|-----------|-------|"
    lines.append(header)
    lines.append(sep)

    for row in rows:
        cond = row.get("condition_number")
        eig = row.get("eigenvalues")
        polished = row.get("polished", False)
        calibrated = row.get("calibrated", False)
        n_evals = row.get("n_function_evals", "\u2014")
        lmin = _fmt_val(float(np.min(eig))) if eig is not None and isinstance(eig, (np.ndarray, pd.Series)) else "\u2014"
        lmax = _fmt_val(float(np.max(eig))) if eig is not None and isinstance(eig, (np.ndarray, pd.Series)) else "\u2014"
        pol_mark = "\u2713" if polished else "\u2717"
        cal_mark = "\u2713" if calibrated else "\u2717"

        line = f"| {_display_section(str(row['section_id']))} | {_fmt_val(float(cond))} |"
        if has_unreg:
            cond_unreg = row.get("condition_number_unreg")
            line += f" {_fmt_val(float(cond_unreg)) if cond_unreg is not None else chr(0x2014)} |"
        line += f" {lmin} | {lmax} | {pol_mark} | {cal_mark} | {n_evals} |"
        lines.append(line)
    lines.append("")

    if has_unreg:
        lines.extend([
            "**Note**: The regularized \u03ba(H) reflects the Tikhonov-stabilized objective.",
            "The unregularized \u03ba(H) shows data-only conditioning. A large gap indicates",
            "that Tikhonov materially improves numerical stability.",
            "",
        ])

    # -- Per-section full Hessian square tables --
    _append_hessian_square_tables(lines, rows)


def _append_hessian_square_tables(lines: list[str], rows: list[dict[str, object]]) -> None:
    """Render the full Hessian matrix (regularized and unregularized) per section."""
    has_unreg = any("hessian_matrix_unreg" in row for row in rows)
    if not any(row.get("hessian_matrix") is not None for row in rows):
        return

    lines.extend([
        "### Full Hessian Matrices",
        "",
        "The tables below show the complete Hessian matrix at each section\u2019s",
        "fitted optimum for direct verification.",
        "",
    ])

    for row in rows:
        H_reg = row.get("hessian_matrix")
        ci_df = row.get("confidence_intervals")
        if H_reg is None or ci_df is None or not isinstance(ci_df, pd.DataFrame):
            continue
        params = list(ci_df.index)
        rat_display = _get_rat_display_name(str(row["rat_id"]))
        section_id = str(row["section_id"])

        lines.append(f"#### {rat_display} / {_display_section(section_id)}")
        lines.append("")

        # Regularized Hessian
        lines.append("**Regularized** (\u03c9 = 0.001)")
        lines.append("")
        _render_square_matrix(lines, np.asarray(H_reg), params)
        lines.append("")

        # Unregularized Hessian
        H_unreg = row.get("hessian_matrix_unreg")
        if has_unreg and H_unreg is not None:
            lines.append("**Unregularized** (\u03c9 = 0)")
            lines.append("")
            _render_square_matrix(lines, np.asarray(H_unreg), params)
            lines.append("")


def _render_square_matrix(
    lines: list[str],
    M: np.ndarray,
    param_names: list[str],
) -> None:
    """Render a square matrix as a Markdown table with parameter headers."""
    header = "| |" + " | ".join(_display_param(p) for p in param_names) + " |"
    sep = "|---|" + " | ".join(["---"] * len(param_names)) + " |"
    lines.append(header)
    lines.append(sep)
    for i, pi in enumerate(param_names):
        cells = [f"**{_display_param(pi)}**"]
        for j in range(len(param_names)):
            cells.append(_fmt_val(float(M[i, j])))
        lines.append("| " + " | ".join(cells) + " |")


def _append_hessian_diagonal_table(lines: list[str], rows: list[dict[str, object]]) -> None:
    """Regularized vs. unregularized Hessian diagonal for β and k₁."""
    has_unreg = any("hessian_diagonal_unreg" in row for row in rows)
    if not has_unreg:
        return

    lines.extend([
        "## 6. Hessian Diagonal Comparison (β, k₁): Regularized vs. Unregularized",
        "",
        "The table below compares the diagonal entries of the Hessian matrix for the",
        "two parameters of interest (β and k₁) under both regularized (ω = 0.001)",
        "and unregularized (ω = 0) settings. A substantially larger diagonal entry",
        "under regularization indicates that the Tikhonov term is adding curvature",
        "to that parameter direction.",
        "",
        "| Section "
        "| H(β,β) reg. | H(β,β) unreg. | Δ% "
        "| H(k₁,k₁) reg. | H(k₁,k₁) unreg. | Δ% |",
        "|---------|"
        "-------------|---------------|-----"
        "|----------------|------------------|-----|",
    ])

    for row in rows:
        diag_reg = row.get("hessian_diagonal")
        diag_unreg = row.get("hessian_diagonal_unreg")
        if diag_reg is None or diag_unreg is None:
            continue
        if not isinstance(diag_reg, pd.Series) or not isinstance(diag_unreg, pd.Series):
            continue

        cells = [_display_section(str(row["section_id"]))]
        for param in ("alpha", "k_1"):
            if param in diag_reg.index and param in diag_unreg.index:
                val_r = float(diag_reg[param])
                val_u = float(diag_unreg[param])
                delta_pct = ((val_r - val_u) / abs(val_u) * 100) if abs(val_u) > 1e-15 else float("inf")
                cells.extend([_fmt_val(val_r), _fmt_val(val_u), f"{delta_pct:+.1f}"])
            else:
                cells.extend(["\u2014", "\u2014", "\u2014"])
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "**Δ%** = relative change `(reg − unreg) / |unreg| × 100`. "
        "Positive values indicate Tikhonov adds curvature.",
        "",
    ])


def _append_hessian_plots_section(lines: list[str], rows: list[dict[str, object]], rat_name: str | None = None) -> None:
    if not rows:
        return
    # Derive the per-rat plot directory name from the display name
    if rat_name:
        rat_slug = rat_name.replace("/", "_").replace(" ", "_")
    else:
        rat_slug = str(rows[0]["rat_id"]).replace("/", "_")
    plot_dir = f"plots_{rat_slug}"

    # Use display name for plot links
    plot_name = rat_name if rat_name else str(rows[0]["rat_id"])

    lines.extend([
        "## 7. Hessian Landscape: (\u03b2, k\u2081) Subspace",
        "",
        "The plots below show the local quadratic approximation of the cost function",
        "in the (\u03b2, k\u2081) parameter subspace at each section\u2019s fitted optimum.",
        "Elongated ellipses indicate strong parameter coupling; narrow ellipses along",
        "one axis indicate weak identifiability in that direction.",
        "",
        "- **Solid ellipse**: with Tikhonov regularization (\u03c9 = 0.001)",
        "- **Dashed ellipse**: without regularization (\u03c9 = 0)",
        "",
        f"![Summary grid]({plot_dir}/hessian_grid_summary.png)",
        "",
        "### Individual section plots",
        "",
    ])
    for row in rows:
        section_id = str(row["section_id"])
        disp = _display_section(section_id)
        lines.append(f"- [{plot_name} / {disp}]({plot_dir}/hessian_{plot_name}_{section_id}.png)")
    lines.append("")


def _append_interpretation(lines: list[str], rows: list[dict[str, object]], rat_name: str | None = None) -> None:
    if rat_name:
        lines.extend([
            "## 8. Interpretation and Conclusion",
            "",
            f"**Specimen**: {rat_name}",
            "",
        ])
    else:
        lines.extend([
            "## 8. Interpretation and Conclusion",
            "",
        ])

    # -- Fiber angle identifiability --
    se_ratios: list[float] = []
    beta_by_region: dict[str, list[tuple[float, float, float]]] = {}
    for row in rows:
        ci_df = row.get("confidence_intervals")
        if ci_df is None or not isinstance(ci_df, pd.DataFrame) or "alpha" not in ci_df.index:
            continue
        val = float(ci_df.loc["alpha", "value"])
        lo = float(ci_df.loc["alpha", "lower"])
        hi = float(ci_df.loc["alpha", "upper"])
        se_arr = row.get("standard_errors")
        if se_arr is not None and isinstance(se_arr, np.ndarray):
            se = float(pd.Series(se_arr, index=ci_df.index)["alpha"])
        else:
            se = (hi - lo) / 3.92
        if abs(val) > 1e-12:
            se_ratios.append((se / abs(val)) * 100)
        sid = str(row["section_id"])
        region = sid.split("-")[0] if "-" in sid else sid
        beta_by_region.setdefault(region, []).append((val, lo, hi))

    if se_ratios:
        median_rel = float(np.median(se_ratios))
        if median_rel < 10:
            quality = "**well-determined**"
        elif median_rel < 30:
            quality = "**moderately determined**"
        else:
            quality = "**poorly determined**"
        lines.extend([
            f"**Fiber angle identifiability**: With a median relative standard error of "
            f"{median_rel:.1f}%, the fiber angle \u03b2 is {quality} from the uniaxial data.",
            "",
        ])

    # -- beta-k1 coupling --
    rho_k1_values: list[float] = []
    for row in rows:
        corr = row.get("correlation_matrix")
        ci_df = row.get("confidence_intervals")
        if corr is None or ci_df is None or not isinstance(ci_df, pd.DataFrame):
            continue
        params = list(ci_df.index)
        if "alpha" not in params or "k_1" not in params:
            continue
        if isinstance(corr, pd.DataFrame):
            rho_k1_values.append(abs(float(corr.loc["alpha", "k_1"])))
        elif isinstance(corr, np.ndarray):
            rho_k1_values.append(abs(float(corr[params.index("alpha"), params.index("k_1")])))
    if rho_k1_values:
        mean_abs = float(np.mean(rho_k1_values))
        if mean_abs > 0.8:
            coupling_word = "strong"
        elif mean_abs > 0.5:
            coupling_word = "moderate"
        else:
            coupling_word = "weak"
        lines.extend([
            f"**\u03b2\u2013k\u2081 coupling**: The mean |\u03c1(\u03b2, k\u2081)| = {mean_abs:.3f} "
            f"indicates {coupling_word} coupling. "
            "Note that coupling does not preclude identifiability when the standard "
            "error remains reasonable relative to the parameter magnitude.",
            "",
        ])

    # -- Regional variation --
    if len(beta_by_region) >= 2:
        sorted_regions = sorted(beta_by_region.keys())
        lines.append("**Regional variation significance**:")
        lines.append("")
        for i, r1 in enumerate(sorted_regions):
            for r2 in sorted_regions[i + 1:]:
                cis1 = beta_by_region[r1]
                cis2 = beta_by_region[r2]
                lo1_max = max(ci[1] for ci in cis1)
                hi1_min = min(ci[2] for ci in cis1)
                lo2_max = max(ci[1] for ci in cis2)
                hi2_min = min(ci[2] for ci in cis2)
                median_val1 = float(np.median([ci[0] for ci in cis1]))
                median_val2 = float(np.median([ci[0] for ci in cis2]))
                r1_disp = _REGION_DISPLAY.get(r1, r1)
                r2_disp = _REGION_DISPLAY.get(r2, r2)
                if hi1_min < lo2_max or hi2_min < lo1_max:
                    lines.append(
                        f"- {r1_disp} vs. {r2_disp}: "
                        f"median \u03b2 = {_fmt_val(median_val1)} vs. {_fmt_val(median_val2)} "
                        f"\u2014 confidence intervals show **partial overlap**, suggesting "
                        f"the difference is not fully distinguishable at the 95% level."
                    )
                else:
                    lines.append(
                        f"- {r1_disp} vs. {r2_disp}: "
                        f"median \u03b2 = {_fmt_val(median_val1)} vs. {_fmt_val(median_val2)} "
                        f"\u2014 confidence intervals are **non-overlapping**, confirming "
                        f"a statistically distinguishable difference at the 95% level."
                    )
        lines.append("")

    # -- Tikhonov effect --
    has_unreg = any("condition_number_unreg" in row for row in rows)
    if has_unreg:
        se_diffs: list[float] = []
        for row in rows:
            se_reg = row.get("standard_errors")
            se_unreg = row.get("standard_errors_unreg")
            ci_df = row.get("confidence_intervals")
            if se_reg is None or se_unreg is None or ci_df is None:
                continue
            if not isinstance(ci_df, pd.DataFrame) or "alpha" not in ci_df.index:
                continue
            idx = list(ci_df.index).index("alpha")
            if isinstance(se_reg, np.ndarray) and isinstance(se_unreg, np.ndarray):
                if idx < len(se_reg) and idx < len(se_unreg):
                    r = float(se_reg[idx])
                    u = float(se_unreg[idx])
                    if r > 1e-12:
                        se_diffs.append(abs(u - r) / r * 100)
        if se_diffs:
            mean_diff = float(np.mean(se_diffs))
            if mean_diff < 10:
                effect = ("the data alone provides sufficient curvature for "
                         "\u03b2 identification; Tikhonov has a **minor** stabilizing effect")
            elif mean_diff < 50:
                effect = ("Tikhonov has a **moderate** stabilizing effect on \u03b2 "
                         "standard errors")
            else:
                effect = ("Tikhonov **materially** stabilizes \u03b2 estimation; "
                         "the unregularized problem shows substantially larger uncertainties")
            lines.extend([
                f"**Effect of Tikhonov regularization**: The mean relative change in "
                f"SE(\u03b2) between regularized and unregularized analyses is "
                f"{mean_diff:.1f}%, indicating {effect}.",
                "",
            ])

    lines.extend([
        "**Methodology note**: The sandwich covariance estimator (White, 1980) provides",
        "heteroskedasticity-consistent standard errors without assuming a specific",
        "error distribution. Combined with the Cauchy robust loss function, the analysis",
        "is resistant to outlier observations in the experimental data.",
        "",
    ])



def build_local_integrator(cost_function: Any,
                           alpha: float | None = None,
                           epsilon: float | None = None,
                           ) -> CostIntegrator:
    """
    Wrap one section cost function in the reviewer identifiability integrator.

    This applies the regularization and robust-loss settings used to compute the
    local conditioning diagnostics reported for Comment 5.
    """

    if alpha is None:
        alpha = IDENTIFIABILITY_COST_CONFIG["alpha"]

    if epsilon is None:
        epsilon = IDENTIFIABILITY_COST_CONFIG["epsilon"]

    return CostIntegrator(
        [cost_function],
        ftype=str(IDENTIFIABILITY_COST_CONFIG["ftype"]),
        vol_reg=bool(IDENTIFIABILITY_COST_CONFIG["dvol"]),
        rescale=cast(Any, IDENTIFIABILITY_COST_CONFIG["rescale"]),
        c=IDENTIFIABILITY_COST_CONFIG["c"],
        alpha=alpha,
        beta=IDENTIFIABILITY_COST_CONFIG["beta"],
        epsilon=epsilon,
    )


def load_sections(
    *,
    h5_path: str | Path,
    xlsx_path: str | Path,
    ncontrol: int = DEFAULT_ANALYTICAL_NCONTROL,
    rat_id: str | None = None,
    section_id: str | None = None,
) -> list[AnalyticalSection]:
    """
    Reconstruct analytical sections from the shared XLSX/HDF5 runtime inputs.

    This is the main entrypoint when using the module programmatically:

    1. call :func:`load_sections` to get fitted sections;
    2. call :func:`analyze_section` or :func:`collect_rows`;
    3. call :func:`write_reports` to persist CSV/Markdown outputs.

    Parameters
    ----------
    h5_path, xlsx_path:
        Input files shared with ``scripts/plot_analytical_visuals.py``.
    ncontrol:
        Number of control points used to reconstruct the analytical test data.
    rat_id, section_id:
        Optional filters. ``section_id`` should use names such as ``Ar-A``.
    """

    animal_selection = selection_from_filters(rat_id, section_id)
    sections = load_analytical_sections(
        h5_path=h5_path,
        xlsx_path=xlsx_path,
        ncontrol=ncontrol,
        rats=list(animal_selection.keys()),
        sections=selection_section_ids(animal_selection),
    )
    selected_sections = filter_analytical_sections(sections, animal_selection)
    if not selected_sections:
        raise FileNotFoundError(
            "No analytical sections matched the requested filters in the configured XLSX/HDF5 inputs."
        )
    return selected_sections


def analyze_section(
    section: AnalyticalSection,
    *,
    fit_kind: str = "local",
) -> dict[str, object]:
    """
    Analyze one reconstructed section and return one report row.

    The ``section`` object should usually come from :func:`load_sections`. The
    returned mapping is ready for :func:`write_reports` or for direct conversion
    to a :class:`pandas.DataFrame`.
    """

    if fit_kind != "local":
        raise ValueError(f"Unsupported fit_kind: {fit_kind}")

    cost_function = build_section_cost_function(section)
    integrator_tik = build_local_integrator(cost_function, alpha=IDENTIFIABILITY_COST_CONFIG["alpha"])

    fitted_parameters = section.fitted_parameters.rename(index={"bulk": "D"})
    np_xi = fitted_parameters.loc[cost_function.inp_mat_keys].to_numpy(dtype=float)

    report_tik = robust_covariance_from_cost(integrator_tik, np_xi, param_names=["alpha", "k_1"])

    integrator_zero = build_local_integrator(cost_function, alpha=0.)
    report_zero = robust_covariance_from_cost(integrator_zero, np_xi, param_names=["alpha", "k_1"])

    return report_to_row(
        section.rat_id, section.section_id, fit_kind, report_tik,
        report_unreg=report_zero,
    )


def collect_rows(
    sections: Sequence[AnalyticalSection],
    *,
    fit_kind: str = "local",
) -> list[dict[str, object]]:
    """Analyze a sequence of sections and collect report rows for each one."""

    return [analyze_section(sec_i, fit_kind=fit_kind) for sec_i in sections]


def _generate_hessian_contour_plots(
    rows: list[dict[str, object]],
    output_dir: Path,
    rat_display_name: str | None = None,
) -> list[Path]:
    """Generate (β, k₁) Hessian contour plots for each section.

    For each row, extract the 2×2 sub-Hessian and sub-covariance for
    the ``(alpha, k_1)`` parameter pair.  Plot:

    1. Quadratic cost contours from the sub-Hessian.
    2. 95% confidence ellipse from the sub-covariance (solid).
    3. Fitted optimum (★).
    4. If unregularized data is present, overlay an unregularized
       confidence ellipse (dashed).

    Parameters
    ----------
    rows : list[dict[str, object]]
        Analysis results from :func:`collect_rows`.
    output_dir : Path
        Directory to save plots.
    rat_display_name : str, optional
        Display name for the rat (e.g., 'Rat-1'). If provided, used in
        filenames instead of internal rat_id.

    Returns
    -------
    list[Path]
        Paths to all generated PNG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # Collect subplot data for summary grid
    plot_data: list[dict[str, object]] = []

    with plt.style.context("seaborn-v0_8-whitegrid"):
        for row in rows:
            hessian = row.get("hessian_matrix")
            cov = row.get("covariance_matrix")
            ci_df = row.get("confidence_intervals")
            if hessian is None or cov is None or ci_df is None:
                continue
            if not isinstance(ci_df, pd.DataFrame):
                continue
            params = list(ci_df.index)
            if "alpha" not in params or "k_1" not in params:
                continue

            i_alpha = params.index("alpha")
            i_k1 = params.index("k_1")
            idx = [i_alpha, i_k1]

            H_sub = np.array(hessian)[np.ix_(idx, idx)]
            V_sub = np.array(cov)[np.ix_(idx, idx)]
            beta_star = float(ci_df.loc["alpha", "value"])
            k1_star = float(ci_df.loc["k_1", "value"])

            rat_id = str(row["rat_id"])
            section_id = str(row["section_id"])
            disp_section = _display_section(section_id)
            # Use display name for plot title and filename if provided
            plot_name = rat_display_name if rat_display_name else rat_id

            # Extract unreg covariance if available (full matrix, not diagonal approx).
            # covariance_matrix_unreg is stored as a pd.DataFrame, so accept both
            # ndarray and DataFrame to avoid silently dropping the dashed ellipse.
            V_sub_unreg = None
            cov_unreg = row.get("covariance_matrix_unreg")
            ci_unreg = row.get("confidence_intervals_unreg")
            if cov_unreg is not None and isinstance(ci_unreg, pd.DataFrame):
                params_u = list(ci_unreg.index)
                if "alpha" in params_u and "k_1" in params_u:
                    idx_u = [params_u.index("alpha"), params_u.index("k_1")]
                    V_sub_unreg = np.array(cov_unreg)[np.ix_(idx_u, idx_u)]

            pd_entry = {
                "H_sub": H_sub, "V_sub": V_sub, "V_sub_unreg": V_sub_unreg,
                "beta_star": beta_star, "k1_star": k1_star,
                "rat_id": rat_id, "section_id": section_id,
                "disp_section": disp_section,
                "plot_name": plot_name,
            }
            plot_data.append(pd_entry)

            # --- Individual plot ---
            fig, ax = plt.subplots(figsize=(4.5, 4.0))
            _draw_hessian_subplot(ax, H_sub, V_sub, V_sub_unreg, beta_star, k1_star,
                                  title=f"{plot_name} / {disp_section}")
            fig.tight_layout()
            fname = output_dir / f"hessian_{plot_name}_{section_id}.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            generated.append(fname)

        # --- Summary grid ---
        if plot_data:
            ncols = min(3, len(plot_data))
            nrows = (len(plot_data) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows),
                                     squeeze=False)
            for i, entry in enumerate(plot_data):
                r, c = divmod(i, ncols)
                _draw_hessian_subplot(
                    axes[r][c], entry["H_sub"], entry["V_sub"],
                    entry["V_sub_unreg"], entry["beta_star"], entry["k1_star"],
                    title=f"{entry['plot_name']} / {entry['disp_section']}",
                )
            # Hide unused axes
            for i in range(len(plot_data), nrows * ncols):
                r, c = divmod(i, ncols)
                axes[r][c].set_visible(False)
            fig.tight_layout()
            grid_path = output_dir / "hessian_grid_summary.png"
            fig.savefig(grid_path, dpi=150)
            plt.close(fig)
            generated.append(grid_path)

    return generated


def _draw_hessian_subplot(
    ax: plt.Axes,
    H_sub: np.ndarray,
    V_sub: np.ndarray,
    V_sub_unreg: np.ndarray | None,
    beta_star: float,
    k1_star: float,
    *,
    title: str = "",
) -> None:
    """Draw a single (β, k₁) Hessian contour with confidence ellipse(s)."""
    chi2_95 = float(chi2.ppf(0.95, df=2))

    # Standard deviations from covariance diagonal
    sd_beta = np.sqrt(max(V_sub[0, 0], 1e-15))
    sd_k1 = np.sqrt(max(V_sub[1, 1], 1e-15))

    # Grid: +/- 3 sigma around optimum
    n_grid = 100
    b_range = np.linspace(beta_star - 3 * sd_beta, beta_star + 3 * sd_beta, n_grid)
    k_range = np.linspace(k1_star - 3 * sd_k1, k1_star + 3 * sd_k1, n_grid)
    B, K = np.meshgrid(b_range, k_range)

    # Quadratic cost: z = 0.5 * [db, dk] @ H_sub @ [db, dk]^T
    dB = B - beta_star
    dK = K - k1_star
    Z = 0.5 * (H_sub[0, 0] * dB**2 + (H_sub[0, 1] + H_sub[1, 0]) * dB * dK + H_sub[1, 1] * dK**2)

    # Contour plot
    levels = np.linspace(0, float(np.percentile(Z[Z > 0], 95)) if np.any(Z > 0) else 1.0, 15)
    levels = levels[levels > 0]
    if len(levels) < 2:
        levels = np.linspace(0.01, 1.0, 15)
    ax.contourf(B, K, Z, levels=levels, cmap="RdBu_r", alpha=0.7)
    ax.contour(B, K, Z, levels=levels, colors="grey", linewidths=0.4, alpha=0.5)

    # Confidence ellipse (regularized)
    _add_confidence_ellipse(ax, V_sub, beta_star, k1_star, chi2_95,
                            edgecolor="black", linestyle="-", linewidth=1.5,
                            label="with Tikhonov")

    # Confidence ellipse (unregularized)
    if V_sub_unreg is not None:
        _add_confidence_ellipse(ax, V_sub_unreg, beta_star, k1_star, chi2_95,
                                edgecolor="blue", linestyle="--", linewidth=1.2,
                                label="no Tikhonov")

    # Optimum marker
    ax.plot(beta_star, k1_star, marker="*", color="gold", markersize=12,
            markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    ax.set_xlabel(r"$\beta$ [rad]", fontsize=10)
    ax.set_ylabel(r"$k_1$ [kPa]", fontsize=10)
    ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right")


def _add_confidence_ellipse(
    ax: plt.Axes,
    V_sub: np.ndarray,
    x_center: float,
    y_center: float,
    chi2_val: float,
    **ellipse_kwargs: object,
) -> None:
    """Add a 95% confidence ellipse from a 2×2 covariance matrix."""
    eigvals, eigvecs = np.linalg.eigh(V_sub)
    eigvals = np.maximum(eigvals, 1e-15)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width = 2 * np.sqrt(chi2_val * eigvals[1])
    height = 2 * np.sqrt(chi2_val * eigvals[0])
    label = ellipse_kwargs.pop("label", None)

    ell = Ellipse(xy=(x_center, y_center), width=width, height=height, angle=angle,
                  facecolor="none", **ellipse_kwargs,)
    ax.add_patch(ell)
    if label:
        ax.plot([], [], color=ellipse_kwargs.get("edgecolor", "black"),
                linestyle=ellipse_kwargs.get("linestyle", "-"), label=label)


def _to_json_serializable(obj: object) -> object:
    """Recursively convert numpy/pandas objects to JSON-safe Python types.

    ``numpy.nan`` and infinities are converted to ``None`` so the resulting
    JSON is valid and readable by Quarto / JavaScript.
    """
    # pandas types first (before generic list/dict handling)
    if isinstance(obj, pd.DataFrame):
        return _to_json_serializable(obj.to_dict(orient="list"))
    if isinstance(obj, pd.Series):
        return _to_json_serializable(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return _to_json_serializable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    return obj


def save_report_artifacts(
    rat_id: str,
    rat_rows: list[dict[str, object]],
    artifacts_root: Path,
    rat_plot_dir: Path | None = None,
) -> Path:
    """Save per-rat analysis data and plots as Quarto rendering artifacts.

    Creates the following layout::

        artifacts_root/
        └── Rat-N/
            ├── data.json          ← all row data (numpy → JSON-safe types)
            └── plots/             ← copied from rat_plot_dir (if provided)
                ├── hessian_*.png
                └── ...

    Parameters
    ----------
    rat_id : str
        Hyphenated rat ID (e.g. ``rato-17``).
    rat_rows : list[dict[str, object]]
        Per-section analysis rows for this rat.
    artifacts_root : Path
        Root directory for all rat artifacts (e.g. ``reviews/artifacts``).
    rat_plot_dir : Path, optional
        Directory containing the already-generated Hessian PNG files.

    Returns
    -------
    Path
        ``artifacts_root/Rat-N/`` directory (always created).
    """
    rat_display_name = _get_rat_display_name(rat_id)
    rat_slug = rat_display_name.replace("/", "_").replace(" ", "_")
    artifacts_dir = artifacts_root / rat_slug
    plots_dir = artifacts_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Serialize rows → JSON
    data = {
        "rat_id": rat_id,
        "rat_display_name": rat_display_name,
        "rows": [_to_json_serializable(row) for row in rat_rows],
    }
    data_json = artifacts_dir / "data.json"
    data_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved %d rows to %s", len(rat_rows), data_json)

    # Copy plot PNGs into artifacts directory
    if rat_plot_dir is not None and rat_plot_dir.is_dir():
        plots_dir.mkdir(parents=True, exist_ok=True)
        for png in sorted(rat_plot_dir.glob("*.png")):
            shutil.copy2(png, plots_dir / png.name)
        logger.info("Copied plots to %s", plots_dir)

    return artifacts_dir


def render_quarto_report(
    rat_id: str,
    artifacts_dir: Path,
    quarto_bin: Path,
    output_dir: Path,
) -> list[Path]:
    """Render the Quarto report template for a single rat in HTML and PDF.

    Invokes::

        quarto render report_template.qmd --to html --to pdf \\
            -P rat_id:<rat_id> \\
            -P artifacts_dir:<artifacts_dir> \\
            --output-dir <output_dir>

    from the ``scripts/quarto/`` directory.

    Parameters
    ----------
    rat_id : str
        Hyphenated rat ID (e.g. ``rato-17``).
    artifacts_dir : Path
        Directory produced by :func:`save_report_artifacts`.
    quarto_bin : Path
        Absolute path to the ``quarto`` executable.
    output_dir : Path
        Directory where the generated HTML and PDF are written.

    Returns
    -------
    list[Path]
        Paths to the generated HTML and PDF reports.
    """
    quarto_dir = Path(__file__).resolve().parent / "quarto"
    output_dir.mkdir(parents=True, exist_ok=True)
    rat_display_name = _get_rat_display_name(rat_id)
    rat_slug = rat_display_name.replace("/", "_").replace(" ", "_")

    result = subprocess.run(
        [
            str(quarto_bin),
            "render",
            "report_template.qmd",
            "--to", "html",
            "--to", "pdf",
            "--output-dir", str(output_dir.resolve()),
            "-P", f"rat_id:{rat_id}",
            "-P", f"rat_display_name:{rat_display_name}",
            "-P", f"artifacts_dir:{artifacts_dir.resolve()}",
        ],
        cwd=quarto_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        logger.debug("quarto stdout: %s", result.stdout[-500:])

    html_path = output_dir / f"comment5_identifiability_report_{rat_slug}.html"
    pdf_path = output_dir / f"comment5_identifiability_report_{rat_slug}.pdf"
    return [html_path, pdf_path]


def write_reports(
    rows: list[dict[str, object]],
    *,
    csv_out: Path,
    md_out: Path,
    quarto_bin: Path | None = None,
) -> list[Path]:
    """Persist the computed identifiability rows to CSV, Markdown, and optionally Quarto HTML/PDF.

    Use this after :func:`collect_rows` when you want the same artifacts
    generated by the CLI entrypoint.  Generates one Markdown report, one CSV,
    and one Hessian contour-plot directory **per rat**.

    Parameters
    ----------
    rows : list[dict[str, object]]
        Analysis results from :func:`collect_rows`.
    csv_out : Path
        Base path for CSV output (used as template for per-rat files).
    md_out : Path
        Base path for Markdown output (used as template for per-rat files).
    quarto_bin : Path, optional
        If provided, run :func:`save_report_artifacts` then :func:`render_quarto_report`
        for each rat.  Falls back to Markdown-only when ``None``. Default None.

    Returns
    -------
    list[Path]
        List of generated file paths (Markdown and optional Quarto HTML/PDF).

    Naming convention::

        reviews/comment5_identifiability_report_<display_name>.md
        reviews/artifacts/<display_name>/data.json                   (if quarto_bin set)
        reviews/quarto_reports/comment5_identifiability_report_<display_name>.html
        reviews/quarto_reports/comment5_identifiability_report_<display_name>.pdf
        reviews/plots_<display_name>/comment5_identifiability_metrics_<display_name>.csv
        reviews/plots_<display_name>/hessian_*.png
    """

    md_out.parent.mkdir(parents=True, exist_ok=True)

    # Group rows by rat_id
    rows_by_rat: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        rid = str(row["rat_id"])
        rows_by_rat.setdefault(rid, []).append(row)

    # Write per-rat reports
    stem = md_out.stem          # e.g. "comment5_identifiability_report"
    suffix = md_out.suffix      # e.g. ".md"
    out_dir = md_out.parent
    artifacts_root = out_dir / "artifacts"
    quarto_out_dir = out_dir / "quarto_reports"
    generated_files: list[Path] = []

    for rat_id, rat_rows in rows_by_rat.items():
        # Get display name for files/folders (e.g., "Rat-1" instead of "rato-17")
        rat_display_name = _get_rat_display_name(rat_id)
        # Create a filesystem-safe version of display name
        rat_slug = rat_display_name.replace("/", "_").replace(" ", "_")

        rat_md = out_dir / f"{stem}_{rat_slug}{suffix}"
        rat_plot_dir = out_dir / f"plots_{rat_slug}"

        # Hessian contour plots
        try:
            generated = _generate_hessian_contour_plots(rat_rows, rat_plot_dir, rat_display_name=rat_display_name)
            logger.info("Generated %d Hessian plots for %s", len(generated), rat_display_name)
        except Exception:
            logger.warning("Hessian plot generation failed for %s", rat_display_name, exc_info=True)

        # Write Markdown report
        md_content = render_markdown_summary(rat_rows, rat_name=rat_display_name)
        rat_md.write_text(md_content, encoding="utf-8")
        logger.info("Wrote report %s (%d sections)", rat_md.name, len(rat_rows))
        generated_files.append(rat_md)

        # Write per-rat CSV in plot directory
        rat_csv = rat_plot_dir / f"{stem.rsplit('_', 1)[0]}_metrics_{rat_slug}.csv"
        rat_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rat_rows).to_csv(rat_csv, index=False)
        logger.info("Wrote CSV %s", rat_csv.name)

        # Generate Quarto HTML/PDF if quarto_bin supplied
        if quarto_bin is not None:
            try:
                artifacts_dir = save_report_artifacts(
                    rat_id, rat_rows, artifacts_root, rat_plot_dir=rat_plot_dir
                )
                quarto_outputs = render_quarto_report(rat_id, artifacts_dir, quarto_bin, quarto_out_dir)
                generated_files.extend(quarto_outputs)
                for out_path in quarto_outputs:
                    logger.info("Quarto report written to %s", out_path)
            except Exception:
                logger.warning("Quarto render failed for %s", rat_display_name, exc_info=True)

    # Combined all-rats CSV
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    logger.info("Wrote combined CSV %s (%d rows)", csv_out.name, len(rows))

    return generated_files


def _default_review_output_paths() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "reviews" / "comment5_identifiability_metrics.csv"
    md_path = project_root / "reviews" / "comment5_identifiability_report.md"
    return csv_path.resolve(), md_path.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the standalone identifiability analysis script."""

    default_h5_file, default_xlsx_file, _ = default_analytical_run_paths()
    default_csv_out, default_md_out = _default_review_output_paths()

    parser = argparse.ArgumentParser(
        description="Analyze local beta/k_1 identifiability for Reviewer B Comment 5."
    )
    parser.add_argument("--h5-path", type=Path, default=default_h5_file)
    parser.add_argument("--xlsx-path", type=Path, default=default_xlsx_file)
    parser.add_argument("--ncontrol", type=int, default=DEFAULT_ANALYTICAL_NCONTROL)
    parser.add_argument("--rat-id", dest="rat_id", default=None,
        help="Restrict to one rat (e.g. rato_17). Default: all rats in MANUSCRIPT_SELECTION.",
    )

    parser.add_argument("--section", dest="section_id", default=None,
        help="Restrict to one section (e.g. Ar-A). Requires --rat-id.",
    )

    parser.add_argument("--fit-kind", default="local", choices=["local"])
    parser.add_argument("--csv-out", type=Path, default=default_csv_out)
    parser.add_argument("--md-out", type=Path, default=default_md_out)
    parser.add_argument("--quarto", action="store_true", default=False,
        help="Render Quarto HTML/PDF reports (auto-detects quarto binary via PATH).",
    )
    parser.add_argument("--quarto-bin", type=Path, default=None,
        help="Explicit path to quarto executable (overrides auto-detection).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the end-to-end identifiability workflow from the command line.

    With no arguments, all five rats from ``MANUSCRIPT_SELECTION`` are analyzed
    and one Markdown report is written per rat.

    Examples
    --------
    Analyze all rats (full manuscript selection)::

        conda run -n matfit1d python scripts/analyze_beta_identifiability.py

    Analyze one rat only::

        conda run -n matfit1d python scripts/analyze_beta_identifiability.py --rat-id rato_17

    Generate Quarto HTML/PDF reports (requires quarto in PATH or conda env)::

        conda run -n matfit1d python scripts/analyze_beta_identifiability.py --quarto
    """

    args = parse_args(argv)

    # Resolve quarto binary when --quarto or --quarto-bin is given
    quarto_bin: Path | None = None
    if args.quarto or args.quarto_bin:
        candidate = args.quarto_bin or Path(shutil.which("quarto") or "")
        if candidate.is_file():
            quarto_bin = candidate
        else:
            logger.warning(
                "--quarto requested but quarto binary not found at %r; skipping Quarto render.",
                str(candidate),
            )

    sections = load_sections(
        h5_path=args.h5_path,
        xlsx_path=args.xlsx_path,
        ncontrol=args.ncontrol,
        rat_id=args.rat_id,
        section_id=args.section_id,
    )

    rows = collect_rows(sections, fit_kind=args.fit_kind)
    generated = write_reports(
        rows,
        csv_out=args.csv_out,
        md_out=args.md_out,
        quarto_bin=quarto_bin,
    )

    logger.info("Generated %d output files", len(generated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
