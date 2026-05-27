from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

try:
    import scripts._analytical_runtime as analytical_runtime
    import scripts.analyze_beta_identifiability as analyze_beta_identifiability
    from scripts.analyze_beta_identifiability import (
        IDENTIFIABILITY_COST_CONFIG,
        _generate_hessian_contour_plots,
        analyze_section,
        load_sections,
        parse_args,
        render_markdown_summary,
        report_to_row,
        selection_from_filters,
    )
    from scripts._analytical_runtime import (
        AnalyticalSection,
        build_section_cost_function,
        default_analytical_run_paths,
        load_analytical_sections,
    )
    HAS_SCRIPTS = True
except ImportError:
    HAS_SCRIPTS = False

from dualmatfit.fitting.covariance import CovarianceReport
from dualmatfit.fitting.identifiability import ConditioningReport

# Skip entire module if scripts package is not available
pytestmark = pytest.mark.skipif(not HAS_SCRIPTS, reason="scripts package not available")


def _sample_report() -> ConditioningReport:
    return ConditioningReport(
        param_names=("alpha", "k_1", "mu"),
        cost=0.5,
        jacobian=np.eye(3),
        singular_values=np.array([10.0, 1.0, 0.01]),
        covariance_matrix=np.eye(3),
        standard_error=np.array([0.1, 0.2, 0.3]),
        hessian=np.eye(3),
        condition_number_jtj=1.0e6,
        beta_k1_condition_number=5.0e5,
        beta_k1_cosine_similarity=0.9999,
        smallest_singular_value=0.01,
        beta_variance_proxy=12.5,
        omega_tik=0.001,
    )


def _sample_covariance_report() -> CovarianceReport:
    """CovarianceReport stub for report_to_row tests."""
    V = np.diag([0.01, 0.04, 0.09])
    H = np.linalg.inv(V)
    se = np.sqrt(np.diag(V))
    names = ["alpha", "k_1", "mu"]
    return CovarianceReport(
        param_names=("alpha", "k_1", "mu"),
        param_idx=np.arange(3),
        covariance_matrix=V,
        standard_errors=se,
        correlation_matrix=np.eye(3),
        hessian_matrix=H,
        hessian_condition=float(np.max(np.diag(H)) / np.min(np.diag(H))),
        eigenvalues=np.linalg.eigvalsh(H),
        hessian_diagonal=pd.Series(np.diag(H), index=names),
        confidence_interval=pd.DataFrame({'lower': -se, 'value': np.zeros(3), 'upper': se}, index=names),
        confidence_level=0.95,
        n_function_evals=100,
        polished=False,
        calibrated=False,
        method="accurate",
    )


def test_report_to_row_contains_required_columns():
    row = report_to_row("rato-17", "Ar-A", "local", _sample_covariance_report())

    assert row["rat_id"] == "rato-17"
    assert row["section_id"] == "Ar-A"
    assert row["fit_kind"] == "local"
    assert row["condition_number"] > 0
    assert row["param_names"] == "alpha,k_1,mu"
    assert row["correlation_matrix"] is not None
    assert row["method"] == "accurate"
    assert isinstance(row["polished"], bool)
    assert isinstance(row["calibrated"], bool)
    assert row["n_function_evals"] == 100


def test_render_markdown_summary_contains_reviewer_report_structure():
    row = report_to_row("rato-17", "Ar-A", "local", _sample_covariance_report())
    markdown = render_markdown_summary([row])
    md_lower = markdown.lower()

    assert "sandwich" in md_lower
    assert "confidence interval" in md_lower
    assert "correlation" in md_lower
    assert "conditioning" in md_lower
    assert "reviewer b" in md_lower
    assert "hessian" in md_lower
    assert "no formal covariance" not in md_lower


def test_default_analytical_run_paths_match_plot_workflow():
    h5_path, xlsx_path, output_dir = default_analytical_run_paths()

    assert h5_path.name == "final_data.h5"
    assert xlsx_path.name == "glb_opt_mat_param_ipopt_v01.xlsx"
    assert output_dir.name == "plots"
    assert output_dir.parent == xlsx_path.parent


def test_load_analytical_sections_reconstructs_requested_section(tmp_path, monkeypatch):
    xlsx_path = tmp_path / "params.xlsx"
    pd.DataFrame(
        {
            "mu": [0.2, 0.2],
            "bulk": [0.05, 0.05],
            "k_1": [1.2, 1.2],
            "k_2": [2.4, 2.4],
            "alpha": [0.4, 0.4],
            "kappa": [0.1, 0.1],
        },
        index=["baseline", "Ar-A"],
    ).to_excel(xlsx_path, sheet_name="rato_17")

    h5_path = tmp_path / "final_data.h5"
    pd.DataFrame(
        {
            "(Ar-A) Time [s]": [0.0, 1.0, 2.0, 3.0],
            "(Ar-A) Extension [mm]": [0.0, 0.2, 0.5, 1.0],
            "(Ar-A) Load [N]": [0.0, 0.1, 0.4, 0.9],
        }
    ).to_hdf(h5_path, key="rato_17", mode="w")

    monkeypatch.setattr(
        analytical_runtime,
        "excel_data",
        lambda: {
            "rato-17": {
                "Ar": {
                    "A": {"len": 2.0, "dist": 4.0, "tcontrol": [0.0, 3.0]},
                    "dia": 4.0,
                    "thick": 0.5,
                }
            }
        },
    )

    class FakeVariationalFormulation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(analytical_runtime, "VariationalFormulation", FakeVariationalFormulation)

    sections = load_analytical_sections(
        h5_path=h5_path,
        xlsx_path=xlsx_path,
        rats=["rato-17"],
    )

    assert len(sections) == 1
    section = sections[0]
    assert section.rat_sheet_id == "rato_17"
    assert section.rat_id == "rato-17"
    assert section.section_id == "Ar-A"
    assert section.fitted_parameters["D"] == pytest.approx(0.05)
    assert section.var_form.kwargs["mix"] == analytical_runtime.DEFAULT_ANALYTICAL_RUN_VAR_FORM_CONFIG["mix"]
    assert section.info_data["ds"] == pytest.approx(1.0)


def test_load_analytical_sections_filters_by_section_ids(tmp_path, monkeypatch):
    xlsx_path = tmp_path / "params.xlsx"
    pd.DataFrame(
        {
            "mu": [0.1, 0.2, 0.3],
            "bulk": [0.01, 0.05, 0.07],
            "k_1": [1.0, 1.2, 1.4],
            "k_2": [2.0, 2.4, 2.8],
            "alpha": [0.3, 0.4, 0.6],
            "kappa": [0.05, 0.1, 0.2],
        },
        index=["baseline", "Ar-A", "Tr-B"],
    ).to_excel(xlsx_path, sheet_name="rato_17")

    h5_path = tmp_path / "final_data.h5"
    pd.DataFrame(
        {
            "(Ar-A) Time [s]": [0.0, 1.0, 2.0, 3.0],
            "(Ar-A) Extension [mm]": [0.0, 0.2, 0.5, 1.0],
            "(Ar-A) Load [N]": [0.0, 0.1, 0.4, 0.9],
            "(Tr-B) Time [s]": [0.0, 1.0, 2.0, 3.0],
            "(Tr-B) Extension [mm]": [0.0, 0.3, 0.6, 1.1],
            "(Tr-B) Load [N]": [0.0, 0.2, 0.5, 1.0],
        }
    ).to_hdf(h5_path, key="rato_17", mode="w")

    monkeypatch.setattr(
        analytical_runtime,
        "excel_data",
        lambda: {
            "rato-17": {
                "Ar": {
                    "A": {"len": 2.0, "dist": 4.0, "tcontrol": [0.0, 3.0]},
                    "dia": 4.0,
                    "thick": 0.5,
                },
                "Tr": {
                    "B": {"len": 2.5, "dist": 4.5, "tcontrol": [0.0, 3.0]},
                    "dia": 4.2,
                    "thick": 0.6,
                },
            }
        },
    )

    class FakeVariationalFormulation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(analytical_runtime, "VariationalFormulation", FakeVariationalFormulation)

    sections = load_analytical_sections(
        h5_path=h5_path,
        xlsx_path=xlsx_path,
        sections=["Tr_B"],
    )

    assert [section.section_id for section in sections] == ["Tr-B"]
    assert sections[0].fitted_parameters["D"] == pytest.approx(0.07)


def test_load_analytical_sections_combines_rat_and_section_filters(tmp_path, monkeypatch):
    xlsx_path = tmp_path / "params.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        pd.DataFrame(
            {
                "mu": [0.1, 0.2],
                "bulk": [0.01, 0.05],
                "k_1": [1.0, 1.2],
                "k_2": [2.0, 2.4],
                "alpha": [0.3, 0.4],
                "kappa": [0.05, 0.1],
            },
            index=["baseline", "Ar-A"],
        ).to_excel(writer, sheet_name="rato_17")
        pd.DataFrame(
            {
                "mu": [0.4, 0.5],
                "bulk": [0.08, 0.09],
                "k_1": [1.6, 1.8],
                "k_2": [3.0, 3.2],
                "alpha": [0.8, 0.9],
                "kappa": [0.25, 0.3],
            },
            index=["baseline", "Ar-A"],
        ).to_excel(writer, sheet_name="rato_23")

    h5_path = tmp_path / "final_data.h5"
    pd.DataFrame(
        {
            "(Ar-A) Time [s]": [0.0, 1.0, 2.0, 3.0],
            "(Ar-A) Extension [mm]": [0.0, 0.2, 0.5, 1.0],
            "(Ar-A) Load [N]": [0.0, 0.1, 0.4, 0.9],
        }
    ).to_hdf(h5_path, key="rato_17", mode="w")
    pd.DataFrame(
        {
            "(Ar-A) Time [s]": [0.0, 1.0, 2.0, 3.0],
            "(Ar-A) Extension [mm]": [0.0, 0.25, 0.55, 1.05],
            "(Ar-A) Load [N]": [0.0, 0.15, 0.45, 0.95],
        }
    ).to_hdf(h5_path, key="rato_23", mode="a")

    monkeypatch.setattr(
        analytical_runtime,
        "excel_data",
        lambda: {
            "rato-17": {
                "Ar": {
                    "A": {"len": 2.0, "dist": 4.0, "tcontrol": [0.0, 3.0]},
                    "dia": 4.0,
                    "thick": 0.5,
                }
            },
            "rato-23": {
                "Ar": {
                    "A": {"len": 2.2, "dist": 4.2, "tcontrol": [0.0, 3.0]},
                    "dia": 4.1,
                    "thick": 0.55,
                }
            },
        },
    )

    class FakeVariationalFormulation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(analytical_runtime, "VariationalFormulation", FakeVariationalFormulation)

    sections = load_analytical_sections(
        h5_path=h5_path,
        xlsx_path=xlsx_path,
        rats=["rato-17"],
        sections=["Ar_A"],
    )

    assert len(sections) == 1
    assert sections[0].rat_id == "rato-17"
    assert sections[0].section_id == "Ar-A"
    assert sections[0].fitted_parameters["D"] == pytest.approx(0.05)


def test_build_section_cost_function_uses_fitted_parameters(monkeypatch):
    calls: dict[str, object] = {}
    template_dsvars = pd.DataFrame(
        {
            "ini": [0.0] * 6,
            "values": [0.0] * 6,
            "lower": [0.0] * 6,
            "upper": [1.0] * 6,
            "limit": [0.1] * 6,
            "variable": [True] * 6,
            "baseline": [0.0] * 6,
        },
        index=["mu", "D", "k_1", "k_2", "alpha", "kappa"],
    )

    class FakeMaterialSetup:
        def __init__(self, *args, **kwargs):
            calls["setup_kwargs"] = kwargs

        def __call__(self):
            return template_dsvars.copy(), None, None, None

    def fake_cost_function(**kwargs):
        calls["cost_kwargs"] = kwargs
        return SimpleNamespace(inp_mat_keys=["alpha", "k_1", "mu"])

    monkeypatch.setattr(analytical_runtime, "MaterialSetup", FakeMaterialSetup)
    monkeypatch.setattr(analytical_runtime, "CostFunction", fake_cost_function)

    section = AnalyticalSection(
        rat_sheet_id="rato_17",
        rat_id="rato-17",
        region_id="Ar",
        position_id="A",
        section_id="Ar-A",
        baseline_parameters=pd.Series(
            {"mu": 0.5, "D": 1.0, "k_1": 1.5, "k_2": 2.0, "alpha": 2.5, "kappa": 0.05}
        ),
        fitted_parameters=pd.Series(
            {"mu": 1.0, "D": 2.0, "k_1": 3.0, "k_2": 4.0, "alpha": 5.0, "kappa": 0.1}
        ),
        info_data={"ds": 1.0},
        instron_data=SimpleNamespace(
            np_tload_ref=np.array([0.1, 0.2]),
            np_tstretch_ref=np.array([1.0, 1.1]),
        ),
        var_form=object(),
        var_form_config=analytical_runtime.build_analytical_var_form_config(),
    )

    build_section_cost_function(section)

    dsvars = calls["cost_kwargs"]["dsvars"]
    assert dsvars.loc["mu", "values"] == pytest.approx(1.0)
    assert dsvars.loc["D", "values"] == pytest.approx(2.0)
    assert dsvars.loc["alpha", "values"] == pytest.approx(5.0)
    assert calls["cost_kwargs"]["load_ref"] is section.instron_data.np_tload_ref
    assert calls["cost_kwargs"]["stretch_x"] is section.instron_data.np_tstretch_ref


def test_load_sections_filters_to_manuscript_selection(monkeypatch):
    requested: dict[str, object] = {}
    section = AnalyticalSection(
        rat_sheet_id="rato_17",
        rat_id="rato-17",
        region_id="Ar",
        position_id="A",
        section_id="Ar-A",
        baseline_parameters=pd.Series({"mu": 0.5}),
        fitted_parameters=pd.Series({"mu": 1.0}),
        info_data={},
        instron_data=SimpleNamespace(),
        var_form=object(),
        var_form_config={},
    )

    def fake_load_analytical_sections(**kwargs):
        requested.update(kwargs)
        return [section]

    monkeypatch.setattr(
        analyze_beta_identifiability,
        "load_analytical_sections",
        fake_load_analytical_sections,
    )
    monkeypatch.setattr(
        analyze_beta_identifiability,
        "filter_analytical_sections",
        lambda sections, selection: list(sections),
    )

    sections = load_sections(
        h5_path="final_data.h5",
        xlsx_path="params.xlsx",
        ncontrol=15,
        rat_id="rato_17",
        section_id="Ar-A",
    )

    assert sections == [section]
    assert requested["rats"] == ["rato_17"]
    assert requested["sections"] == ["Ar-A"]
    assert requested["ncontrol"] == 15


def test_selection_from_filters_restricts_to_requested_section():
    selection = selection_from_filters("rato_17", "Ar-A")

    assert selection == {"rato_17": {"Ar": ["A"]}}


def test_analyze_section_uses_shared_cost_builder(monkeypatch):
    """analyze_section builds two integrators (with/without Tikhonov) and
    calls robust_covariance_from_cost on each, returning the Tikhonov report
    converted to a row via report_to_row."""
    integrator_calls: list[dict[str, object]] = []
    covariance_calls: list[dict[str, object]] = []

    section = AnalyticalSection(
        rat_sheet_id="rato_17",
        rat_id="rato-17",
        region_id="Ar",
        position_id="A",
        section_id="Ar-A",
        baseline_parameters=pd.Series({"alpha": 0.5, "k_1": 1.0, "mu": 1.5}),
        fitted_parameters=pd.Series({"alpha": 1.0, "k_1": 2.0, "mu": 3.0}),
        info_data={},
        instron_data=SimpleNamespace(),
        var_form=object(),
        var_form_config={},
    )

    class FakeCostFunction:
        inp_mat_keys = ["alpha", "k_1", "mu"]

    class FakeIntegrator:
        def __init__(self, label):
            self.label = label
            self.inp_mat_keys = ["alpha", "k_1", "mu"]

    def fake_build_section_cost_function(section_input):
        return FakeCostFunction()

    def fake_build_local_integrator(cost_function, alpha=None, epsilon=None):
        call = {"cost_function": cost_function, "alpha": alpha, "epsilon": epsilon}
        integrator_calls.append(call)
        return FakeIntegrator(f"integrator_{len(integrator_calls)}")

    report = _sample_report()

    def fake_robust_covariance_from_cost(integrator, xi, **kwargs):
        covariance_calls.append({"integrator": integrator, "xi": xi, **kwargs})
        return report

    def fake_report_to_row(rat_id, section_id, fit_kind, rpt, *, report_unreg=None):
        return {"rat_id": rat_id, "section_id": section_id, "fit_kind": fit_kind}

    monkeypatch.setattr(
        analyze_beta_identifiability,
        "build_section_cost_function",
        fake_build_section_cost_function,
    )
    monkeypatch.setattr(
        analyze_beta_identifiability,
        "build_local_integrator",
        fake_build_local_integrator,
    )
    monkeypatch.setattr(
        analyze_beta_identifiability,
        "robust_covariance_from_cost",
        fake_robust_covariance_from_cost,
    )
    monkeypatch.setattr(
        analyze_beta_identifiability,
        "report_to_row",
        fake_report_to_row,
    )

    row = analyze_section(section, fit_kind="local")

    assert row["rat_id"] == "rato-17"
    assert row["section_id"] == "Ar-A"
    assert row["fit_kind"] == "local"

    # Two integrator calls: first with config alpha, second with alpha=0
    assert len(integrator_calls) == 2
    assert integrator_calls[0]["alpha"] == IDENTIFIABILITY_COST_CONFIG["alpha"]
    assert integrator_calls[1]["alpha"] == 0.0

    # Two covariance calls (tik and zero-regularization)
    assert len(covariance_calls) == 2


def test_parse_args_uses_shared_analytical_defaults():
    default_h5, default_xlsx, _ = default_analytical_run_paths()

    args = parse_args([])

    assert args.h5_path == default_h5
    assert args.xlsx_path == default_xlsx
    assert args.ncontrol == analytical_runtime.DEFAULT_ANALYTICAL_NCONTROL


def test_report_to_row_includes_unreg_comparison():
    report = _sample_covariance_report()
    report_unreg = _sample_covariance_report()
    row = report_to_row("rato-17", "Ar-A", "local", report, report_unreg=report_unreg)
    assert "condition_number_unreg" in row
    assert row["condition_number_unreg"] > 0
    assert "standard_errors_unreg" in row
    assert "confidence_intervals_unreg" in row
    assert "covariance_matrix_unreg" in row
    assert "hessian_diagonal_unreg" in row
    assert "hessian_matrix_unreg" in row


def test_report_to_row_omits_unreg_when_not_provided():
    row = report_to_row("rato-17", "Ar-A", "local", _sample_covariance_report())
    assert "condition_number_unreg" not in row


def test_generate_hessian_contour_plots_creates_files(tmp_path):
    """Verify plot files are created using stub data (no real fitting)."""
    row = report_to_row("rato-17", "Ar-A", "local", _sample_covariance_report())
    paths = _generate_hessian_contour_plots([row], tmp_path)
    assert (tmp_path / "hessian_rato-17_Ar-A.png").exists()
    assert (tmp_path / "hessian_grid_summary.png").exists()
    assert len(paths) == 2

def test_render_markdown_with_unreg_includes_hessian_diagonal():
    """Report with unreg data should contain the Hessian diagonal comparison table."""
    report = _sample_covariance_report()
    report_unreg = _sample_covariance_report()
    row = report_to_row("rato-17", "Ar-A", "local", report, report_unreg=report_unreg)
    markdown = render_markdown_summary([row])
    assert "hessian diagonal comparison" in markdown.lower()
    assert "unreg" in markdown.lower()


def test_render_markdown_correlation_includes_k1():
    """Correlation table should show k₁ as both row and column in the square matrix."""
    row = report_to_row("rato-17", "Ar-A", "local", _sample_covariance_report())
    markdown = render_markdown_summary([row])
    assert "k\u2081" in markdown           # k₁ appears as row/column header
    assert "\u03b2" in markdown            # β appears as row/column header
    assert "correlation" in markdown.lower()


def test_write_reports_creates_per_rat_files(tmp_path):
    """write_reports should create one MD per rat_id, not a single file."""
    report1 = _sample_covariance_report()
    report2 = _sample_covariance_report()
    rows = [
        report_to_row("rato-17", "Ar-A", "local", report1),
        report_to_row("rato-23", "Tr-B", "local", report2),
    ]
    from scripts.analyze_beta_identifiability import write_reports

    csv_out = tmp_path / "metrics.csv"
    md_out = tmp_path / "report.md"
    write_reports(rows, csv_out=csv_out, md_out=md_out)

    # Combined CSV is written
    assert csv_out.exists()
    # Per-rat Markdown uses display names from RATS_STYLES (Rat-1, Rat-2)
    assert (tmp_path / "report_Rat-1.md").exists()
    assert (tmp_path / "report_Rat-2.md").exists()
    # The template path itself is NOT written
    assert not md_out.exists()


def test_render_markdown_includes_square_hessian_tables():
    """With unreg data, the report should include full Hessian square tables."""
    report = _sample_covariance_report()
    report_unreg = _sample_covariance_report()
    row = report_to_row("rato-17", "Ar-A", "local", report, report_unreg=report_unreg)
    markdown = render_markdown_summary([row])
    assert "full hessian matrices" in markdown.lower()
    assert "regularized" in markdown.lower()
    assert "unregularized" in markdown.lower()
