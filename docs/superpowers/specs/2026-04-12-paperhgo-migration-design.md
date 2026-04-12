# PaperHGOMatFit Migration Design

## Overview

Migrate improvements from `~/NAS/Repositories/PaperHGOMatFit` into `dual-matfit`. The PaperHGOMatFit repo has a restructured package layout, new modules (covariance estimation, identifiability analysis), solver upgrades, and a decomposed monolithic class. This design covers a full migration with a clean break on imports and a 3-tier test restructure.

**Strategy**: Incremental migration in 4 phases, each leaving the codebase in a working state.

**Conda environment**: `matfit1d`

---

## Phase 1 — Package Restructuring + Conda Migration

### 1.1 Replace Poetry with Conda

Replace Poetry dependency management with Conda using environment name `matfit1d`.

**Create `environment.yml`:**
- `channels:` — defaults + conda-forge (for `ipyopt`, `jax`, `jaxlib`)
- `dependencies:` — version-pinned packages from current `pyproject.toml`
- `pip:` subsection for packages not on conda-forge (e.g., `pathlib`, `latex`)

**Update `pyproject.toml`:**
- Strip `[tool.poetry.dependencies]`, `[tool.poetry.group.*.dependencies]`, `[tool.poetry.source]`
- Keep `[tool.ruff]`, `[tool.mypy]`, `[build-system]`, project metadata
- Keep dev/lint dependencies listed but managed via pip/conda, not Poetry groups

**New dev commands:**
```bash
conda env create -f environment.yml       # create matfit1d env
conda activate matfit1d                    # activate
pip install -e .                           # editable install
pytest                                     # run tests
ruff check dualmatfit/                     # lint
mypy dualmatfit/                           # type check
```

### 1.2 Restructure flat package into subpackages

Move all files from the flat `dualmatfit/` layout into 7 subpackages:

```
dualmatfit/
├── __init__.py
├── formulation/
│   ├── __init__.py
│   ├── material_law.py        ← material_law.py
│   ├── variational.py         ← variational_form.py
│   ├── tensor.py               ← tensor.py
│   ├── simplify.py             ← simplify.py
│   └── lambdify.py             ← lambdify_builder.py
├── solvers/
│   ├── __init__.py
│   ├── solution.py             ← solution.py
│   ├── extension.py            ← extension_solution.py
│   ├── derivative.py           ← derivative.py
│   └── barrier.py              ← barrier.py
├── optimization/
│   ├── __init__.py
│   ├── loss.py                 ← loss functions from cost_functions.py
│   ├── cost.py                 ← CostFunction, CostIntegrator from least_square.py
│   ├── cache.py                ← cost_cache.py
│   ├── regularization.py       ← regularization.py
│   ├── core.py                 ← optimization.py
│   ├── drivers.py              ← drivers.py
│   ├── basinhopping.py         ← basinhopping.py
│   └── ipopt.py                ← ipopt.py
├── fitting/
│   ├── __init__.py
│   ├── core.py                 ← AnisoModelSolve from material_fit.py
│   ├── optimization.py         ← FitOptimizationMixin from material_fit.py
│   ├── persistence.py          ← FitPersistenceMixin from material_fit.py
│   ├── visualization.py        ← FitVisualizationMixin from material_fit.py
│   └── constants.py            ← extracted inline constants from material_fit.py
├── data/
│   ├── __init__.py
│   ├── experimental.py          ← experimental.py
│   └── rato_info.py             ← rato_info.py
├── plotting/
│   ├── __init__.py
│   ├── analytical_visuals.py    ← from plot.py (ese_plot, mat_plot)
│   ├── experimental_visuals.py ← from plot.py (exp_test_plot, stress_plot, plot_*)
│   ├── solution_visuals.py      ← from plot.py (PlotSolution2D)
│   ├── parameters.py            ← extracted constants from plot.py
│   └── plot_helpers.py          ← plotting/plot_helpers.py (stays)
└── utils/
    ├── __init__.py
    ├── numeric.py               ← numeric_utils.py
    ├── io_utils.py              ← io_utils.py
    ├── logging_config.py        ← logging_config.py
    ├── path_manager.py          ← path_manager.py
    ├── log_contexts.py          ← log_contexts.py
    └── latex_post.py            ← latex_post.py
```

### 1.3 Update all import paths

Every internal import changes from flat to subpackage path:

| Old | New |
|---|---|
| `dualmatfit.material_law` | `dualmatfit.formulation.material_law` |
| `dualmatfit.variational_form` | `dualmatfit.formulation.variational` |
| `dualmatfit.tensor` | `dualmatfit.formulation.tensor` |
| `dualmatfit.simplify` | `dualmatfit.formulation.simplify` |
| `dualmatfit.lambdify_builder` | `dualmatfit.formulation.lambdify` |
| `dualmatfit.solution` | `dualmatfit.solvers.solution` |
| `dualmatfit.extension_solution` | `dualmatfit.solvers.extension` |
| `dualmatfit.derivative` | `dualmatfit.solvers.derivative` |
| `dualmatfit.barrier` | `dualmatfit.solvers.barrier` |
| `dualmatfit.cost_functions` | `dualmatfit.optimization.loss` |
| `dualmatfit.least_square` | `dualmatfit.optimization.cost` |
| `dualmatfit.cost_cache` | `dualmatfit.optimization.cache` |
| `dualmatfit.regularization` | `dualmatfit.optimization.regularization` |
| `dualmatfit.optimization` | `dualmatfit.optimization.core` |
| `dualmatfit.drivers` | `dualmatfit.optimization.drivers` |
| `dualmatfit.basinhopping` | `dualmatfit.optimization.basinhopping` |
| `dualmatfit.ipopt` | `dualmatfit.optimization.ipopt` |
| `dualmatfit.material_fit` | `dualmatfit.fitting.core` |
| `dualmatfit.experimental` | `dualmatfit.data.experimental` |
| `dualmatfit.rato_info` | `dualmatfit.data.rato_info` |
| `dualmatfit.numeric_utils` | `dualmatfit.utils.numeric` |
| `dualmatfit.io_utils` | `dualmatfit.utils.io_utils` |
| `dualmatfit.logging_config` | `dualmatfit.utils.logging_config` |
| `dualmatfit.path_manager` | `dualmatfit.utils.path_manager` |
| `dualmatfit.log_contexts` | `dualmatfit.utils.log_contexts` |
| `dualmatfit.latex_post` | `dualmatfit.utils.latex_post` |

Each `__init__.py` re-exports the public API so `from dualmatfit.formulation import VariationalFormulation` works.

### 1.4 `cost_functions.py` → `loss.py` + `cost.py` split

`cost_functions.py` currently contains both standalone loss functions and the `CostFunction`/`CostIntegrator` classes. These split into:
- `optimization/loss.py` — standalone loss functions (`lsq_fval`, `cauchy_fval`, `huber_fval`, etc.)
- `optimization/cost.py` — `LSQFit`, `CostFunction`, `CostIntegrator` classes (imports from `loss.py`)

`least_square.py` is absorbed entirely into `cost.py`.

### 1.5 `material_fit.py` decomposition

The monolithic `material_fit.py` decomposes into:
- `fitting/core.py` — `AnisoModelSolve` base class
- `fitting/optimization.py` — `FitOptimizationMixin` (optimization methods)
- `fitting/persistence.py` — `FitPersistenceMixin` (save/load)
- `fitting/visualization.py` — `FitVisualizationMixin` (plotting)
- `fitting/constants.py` — extracted inline constants

`AnisoMaterialFit` composes via multiple inheritance:
```python
class AnisoMaterialFit(AnisoModelSolve, FitOptimizationMixin,
                       FitPersistenceMixin, FitVisualizationMixin): ...
```

### 1.6 `plot.py` decomposition

The monolithic `plot.py` decomposes into:
- `plotting/analytical_visuals.py` — `ese_plot`, `mat_plot`
- `plotting/experimental_visuals.py` — `stress_plot`, `exp_test_plot`, `plot_time_extension`, `plot_time_load`, `plot_extension_load`, `plot_reaction_force`, `plot_volume_change`, `plot_pk1_stress`
- `plotting/solution_visuals.py` — `PlotSolution2D` class
- `plotting/parameters.py` — extracted constants (`DEFAULT_FIGURE_SIZE`, `DEFAULT_DPI`, grid alphas, tick params, etc.)

---

## Phase 2 — Existing Module Upgrades

Functional changes to modules that already exist in dual-matfit. No new files — modifications only.

| Module | Change | Detail |
|---|---|---|
| `solvers/solution.py` | New method | `Root._accept_small_residual_root_result()` — accepts root results where residual is small but not strictly zero. Called from `Root.solve()`. |
| `solvers/extension.py` | Moved function + new attribute | `check_dsvars()` moved from `utils.py` to module-level here, added to `__all__`. Adds `self.xi_ref = self.dsvars["values"].values.astype(float)` to `_init_design_variables`. |
| `solvers/derivative.py` | Logic change | `_auto_generate_bounds()` adds handling for negative params (`xi_val * 1000` lower, `min(-1e-6, xi_val * 0.001)` upper) and zero params (`[-1e3, 1e3]`). |
| `optimization/drivers.py` | New function | `_sanitize_lbfgsb_options()` removes deprecated SciPy options (`disp`, `iprint`). Called in `_run_lbfgs()`. |
| `optimization/cost.py` | Signature change | `CostIntegrator._build_regularization(self, vol_reg: bool, epsilon: float)` — explicit parameters instead of implicit access. All call sites updated. |
| `optimization/loss.py` | Return type change | All loss functions (`cauchy_fval`, `huber_fval`, `logcosh_fval`, `ln_fval`) return `np.ndarray` instead of `float` — enables vectorized/batched computation. Imports `safe_divide` from `utils.numeric`. |
| `optimization/core.py` | New import | `from dualmatfit.utils.ks import min_ks, max_ks` — KS aggregation functions for constraint aggregation. `utils/ks.py` created in this phase. |
| `utils/path_manager.py` | New methods | `get_rat_solution_dir()`, `get_section_dir()`, `validate_file_exists()`, `remove_file()`, `get_output_path()`. |
| `utils/latex_post.py` | New function | `sympy2latex()` — SymPy expression to LaTeX string conversion. |
| `utils/ks.py` | New file | KS aggregation functions `min_ks`, `max_ks` — smooth min/max approximations for constraint aggregation. Required by `optimization/core.py`. |

**Highest-impact change**: The `loss.py` return type change from `float` to `np.ndarray` affects all callers. Must verify all call sites handle arrays correctly.

---

## Phase 3 — New Modules

Entirely new modules from PaperHGOMatFit that don't exist in dual-matfit.

### `fitting/constants.py`

Extracts inline magic numbers and defaults from `material_fit.py`:

- Material defaults: `DEFAULT_BULK_MODULUS`, `DEFAULT_VOLUMETRIC_TYPE`, `DEFAULT_NUM_CONTROL_POINTS`, `DEFAULT_SIMPLIFY_TIMEOUT`
- Experimental data constants: `EXPERIMENTAL_DATA_COLUMNS_PER_SECTION`, `SECTION_CODE_LENGTH`, `POSITION_CODE_INDEX`, `RAT_KEY_PREFIX_LENGTH`, `UNSTRETCHED_STATE`
- Plotting constants: `DEFAULT_FIGURE_SIZE`, `DEFAULT_DPI`, `GRID_ALPHA_MINOR`, `GRID_ALPHA_MAJOR`, `DEFAULT_PLOT_LIMITS`, `ALTERNATIVE_PLOT_LIMITS`, `PLOT_MIN_SCALE`, `PLOT_MAX_SCALE`, `PLOT_FALLBACK_SCALE`, `PLOT_ENERGY_THRESHOLD`, `PLOT_TICK_Y_DIVISOR`, `PLOT_TICK_Y_MULTIPLIER`, `PLOT_MIN_LIMIT`, `TICK_MAJOR_DIVISOR`, `TICK_MINOR_SUBDIVISIONS`, `TICK_LABEL_TRIM`
- Optimization defaults: `DEFAULT_BASELINE_OPTIMIZATION_ITERATIONS`, `DEFAULT_LOCAL_OPTIMIZATION_ITERATIONS`, `DEFAULT_GLOBAL_ITERATIONS`, `DEFAULT_SOLUTION_NCONTROL`, `HIGH_RESOLUTION_NCONTROL`

### `fitting/covariance.py`

MLE covariance estimation and Hessian computation:

- `CovarianceReport` dataclass — param names, covariance matrix, standard errors, correlation matrix, Hessian, condition number, eigenvalues, confidence intervals, calibration flags
- Ridders' method: `find_initial_step`, `ridders_curvature`, `central_diff_cross`, `richardson_off_diagonal`, `accurate_hessian`
- Polish/calibration: `eigenvalue_polish` (NPD repair), `huang_calibration` (Huang 2017)
- Distance metrics: `frobenius_distance`, `correlation_distance`, `g_metric`
- Covariance estimators: `robust_covariance_from_cost`, `mle_covariance` (sandwich), `covariance_from_gauss_newton`
- I/O: `save_covariance_report`, `load_covariance_report` (NPZ format)
- Formatting: `format_params_with_uncertainty`, `build_covariance_summary_table`

Dependencies: numpy, pandas, scipy, `fitting.identifiability`, `utils.logging_config`

### `fitting/identifiability.py`

Local identifiability and conditioning diagnostics:

- `ConditioningReport` dataclass — param names, cost, Jacobian shape, singular values, condition number, effective rank, beta variance proxy, cosine similarity for collinearity
- Functions: `as_2d_jacobian`, `as_1d_residual`, `cosine_similarity`, `beta_variance_proxy`, `analyze_cost_integrator`
- Lazy-imports `optimization.cost` to avoid circular dependency

### `fitting/persistence.py`

`FitPersistenceMixin` — extracted from `AnisoMaterialFit`:

- `_save_data`, `save_data` — HDF5/Parquet/Excel output
- `_save_covariance_for_rat` — persists `CovarianceReport` objects
- `load_results` — load saved results

### `fitting/optimization.py`

`FitOptimizationMixin` — extracted from `AnisoMaterialFit`:

- `find_baseline_parameters`, `_eval_problem`, `_configure_optimization_parameters`
- `_get_local_solution_path`, `_prepare_cost_and_data`
- `_optimize_single_position`, `_optimize_section_positions`
- `_save_rat_optimization_results`, `_generate_aggregate_plots`
- `find_optimal_parameters`

Imports constants from `fitting.constants`.

### `fitting/visualization.py`

`FitVisualizationMixin` — extracted from `AnisoMaterialFit`:

- `_setup_plot_fit`, `_plot_stress`, `plot_fit`, `correlation`

---

## Phase 4 — Test Restructure + New Tests

### Directory restructure

```
tests/
├── __init__.py
├── conftest.py                    ← shared fixtures (unchanged)
├── unit/
│   ├── __init__.py
│   ├── test_barrier.py
│   ├── test_cost_cache.py
│   ├── test_cost_functions.py
│   ├── test_derivative.py
│   ├── test_extension_solution.py
│   ├── test_functions.py
│   ├── test_io_utils.py
│   ├── test_latex_post.py
│   ├── test_logging.py
│   ├── test_numeric_utils.py
│   ├── test_path_manager.py
│   ├── test_regularization.py
│   ├── test_solution.py
│   ├── test_strain_energy.py
│   ├── test_utils.py
│   ├── test_variational_form.py
│   ├── test_variational_form_derivative.py
│   ├── test_covariance.py                  ← NEW
│   ├── test_hessian_edge_cases.py           ← NEW
│   ├── test_identifiability.py             ← NEW
│   └── test_analyze_beta_identifiability.py ← NEW
├── integration/
│   ├── __init__.py
│   ├── test_drivers.py
│   ├── test_experimental.py
│   ├── test_integrated_cost_function.py
│   ├── test_jax_integration.py
│   └── test_plotting.py
└── performance/
    ├── __init__.py
    └── test_lambdify_performance.py
```

### `pytest.ini` additions

```ini
[pytest]
testpaths = tests/unit tests/integration tests/performance
markers =
    slow: slow-running tests
    integration: multi-module integration tests
    requires_data: tests requiring experimental data files
    fast: fast unit tests
filterwarnings =
    ignore::DeprecationWarning:mpmath
```

### New test files

| File | What it tests |
|---|---|
| `test_covariance.py` | `CovarianceReport` construction, Ridders/Richardson/Hessian phases, MLE covariance, `CostIntegrator` integration |
| `test_hessian_edge_cases.py` | Numerical Hessian edge cases (Rosenbrock, Zakharov) |
| `test_identifiability.py` | `ConditioningReport`, condition numbers, singular values |
| `test_analyze_beta_identifiability.py` | `report_to_row`, `render_markdown_summary`, `load_sections`, CLI arg parsing |

### Existing tests

All import paths update from flat (`dualmatfit.X`) to subpackage (`dualmatfit.formulation.X`, `dualmatfit.solvers.X`, etc.). No logic changes.