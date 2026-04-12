# Phase 3 — New Modules Implementation Plan

Adds entirely new modules from PaperHGOMatFit that don't exist in dual-matfit yet.
All file paths use the subpackage layout from Phase 1 (`dualmatfit/fitting/`).

**Source**: `~/NAS/Repositories/PaperHGOMatFit/dualmatfit/fitting/`

**Strategy**: Each task creates one file, verifies it compiles/imports, then commits.
After all individual modules are in place, Task 3.7 decomposes `core.py` to use mixins,
and Task 3.8 updates `__init__.py` to re-export new public symbols.

---

## Task 3.1 — Create `fitting/constants.py`

- [ ] Create `dualmatfit/fitting/constants.py` with exact content from PaperHGOMatFit
- [ ] Verify: `python -c "from dualmatfit.fitting.constants import *; print('OK')"`
- [ ] Commit: `feat(fitting): add constants module with extracted magic numbers`

**File**: `dualmatfit/fitting/constants.py` (112 lines)

Extracts inline magic numbers and defaults from `material_fit.py` (now `fitting/core.py`).

```python
# -*- coding: utf-8 -*-
"""
Constants and configuration for material fitting.

Extracts inline magic numbers and defaults from the fitting module
into named constants for maintainability.

References
----------
.. [1] On the Compressibility of Arterial Tissue
       Bulk modulus range: 42.14-99.03 kPa
"""

from pathlib import Path
from typing import List

# ============================================================================
# Material Properties
# ============================================================================

#: Default bulk modulus (MPa).
#: Literature range: 42.14–99.03 kPa [1]_; median ≈ 56.67 kPa = 0.05667 MPa.
DEFAULT_BULK_MODULUS: float = 0.05667

#: Default volumetric strain-energy type.
#: Options: 'bathe87', 'simo92', 'doll8' (works for fung model)
DEFAULT_VOLUMETRIC_TYPE: str = 'bathe87'

#: Default number of control points for discretization.
DEFAULT_NUM_CONTROL_POINTS: int = 15

#: Default timeout (seconds) for symbolic simplification.
DEFAULT_SIMPLIFY_TIMEOUT: int = 60

# ============================================================================
# Fitting Configuration
# ============================================================================

#: Default maximum iterations for baseline (global) optimization.
DEFAULT_BASELINE_OPTIMIZATION_ITERATIONS: int = 20000

#: Default maximum iterations for local (section-specific) optimization.
DEFAULT_LOCAL_OPTIMIZATION_ITERATIONS: int = 10000

#: Default global optimization iterations.
DEFAULT_GLOBAL_ITERATIONS: int = 100

#: Default number of control points for solution computation.
DEFAULT_SOLUTION_NCONTROL: int = 15

#: High-resolution control point count.
HIGH_RESOLUTION_NCONTROL: int = 200

# ============================================================================
# Data Structure Constants
# ============================================================================

#: Column names for experimental data per section.
EXPERIMENTAL_DATA_COLUMNS_PER_SECTION: List[str] = [
    'stretch', 'load', 'time',
]

#: Length of section code (e.g., 'Ar-A' → 2 for prefix, 3 with position).
SECTION_CODE_LENGTH: int = 2

#: Index of position code within section string.
POSITION_CODE_INDEX: int = 2

#: Length of the rat key prefix (e.g., 'R1' → 2).
RAT_KEY_PREFIX_LENGTH: int = 2

#: Key for unstretched (reference) state data.
UNSTRETCHED_STATE: str = 'unstretched'

# ============================================================================
# Physical Constants
# ============================================================================

#: Gravitational acceleration (m/s²) — not used in fitting but referenced.
G_ACCELERATION: float = 9.81

# ============================================================================
# Plot Configuration
# ============================================================================

#: Default figure size (width, height) in inches.
DEFAULT_FIGURE_SIZE: tuple = (12, 8)

#: Default DPI for plots.
DEFAULT_DPI: int = 150

#: Grid transparency for minor grid lines.
GRID_ALPHA_MINOR: float = 0.3

#: Grid transparency for major grid lines.
GRID_ALPHA_MAJOR: float = 0.5

# ============================================================================
# Plot Limits (MPa)
# ============================================================================

#: Default stress-axis plot limits [min, max] in MPa.
DEFAULT_PLOT_LIMITS: List[float] = [0.0, 0.2]

#: Alternative plot limits for different scenarios.
ALTERNATIVE_PLOT_LIMITS: List[float] = [0.0, 0.15]

# ============================================================================
# Plot Scaling Factors
# ============================================================================

#: Minimum scale factor for plot axis.
PLOT_MIN_SCALE: float = 0.01

#: Maximum scale factor for plot axis.
PLOT_MAX_SCALE: float = 0.5

#: Fallback scale factor when automatic scaling fails.
PLOT_FALLBACK_SCALE: float = 0.1

#: Energy threshold for determining plot scale.
PLOT_ENERGY_THRESHOLD: float = 1e-6

#: Y-axis tick divisor for stress plots.
PLOT_TICK_Y_DIVISOR: float = 0.05

#: Y-axis tick multiplier for stress plots.
PLOT_TICK_Y_MULTIPLIER: float = 1000.0

#: Minimum Y-axis limit for stress plots.
PLOT_MIN_LIMIT: float = 0.001

# ============================================================================
# Tick Configuration
# ============================================================================

#: Major tick divisor for stress axis.
TICK_MAJOR_DIVISOR: float = 0.05

#: Number of minor tick subdivisions between major ticks.
TICK_MINOR_SUBDIVISIONS: int = 5

#: Number of decimal places for tick labels.
TICK_LABEL_TRIM: int = 2

__all__: List[str] = [
    # Material Properties
    'DEFAULT_BULK_MODULUS',
    'DEFAULT_VOLUMETRIC_TYPE',
    'DEFAULT_NUM_CONTROL_POINTS',
    'DEFAULT_SIMPLIFY_TIMEOUT',
    # Fitting Configuration
    'DEFAULT_BASELINE_OPTIMIZATION_ITERATIONS',
    'DEFAULT_LOCAL_OPTIMIZATION_ITERATIONS',
    'DEFAULT_GLOBAL_ITERATIONS',
    'DEFAULT_SOLUTION_NCONTROL',
    'HIGH_RESOLUTION_NCONTROL',
    # Data Structure Constants
    'EXPERIMENTAL_DATA_COLUMNS_PER_SECTION',
    'SECTION_CODE_LENGTH',
    'POSITION_CODE_INDEX',
    'RAT_KEY_PREFIX_LENGTH',
    'UNSTRETCHED_STATE',
    # Physical Constants
    'G_ACCELERATION',
    # Plot Configuration
    'DEFAULT_FIGURE_SIZE',
    'DEFAULT_DPI',
    'GRID_ALPHA_MINOR',
    'GRID_ALPHA_MAJOR',
    # Plot Limits
    'DEFAULT_PLOT_LIMITS',
    'ALTERNATIVE_PLOT_LIMITS',
    # Plot Scaling
    'PLOT_MIN_SCALE',
    'PLOT_MAX_SCALE',
    'PLOT_FALLBACK_SCALE',
    'PLOT_ENERGY_THRESHOLD',
    'PLOT_TICK_Y_DIVISOR',
    'PLOT_TICK_Y_MULTIPLIER',
    'PLOT_MIN_LIMIT',
    # Tick Configuration
    'TICK_MAJOR_DIVISOR',
    'TICK_MINOR_SUBDIVISIONS',
    'TICK_LABEL_TRIM',
]
```

---

## Task 3.2 — Create `fitting/covariance.py`

- [ ] Create `dualmatfit/fitting/covariance.py` with exact content from PaperHGOMatFit
- [ ] Verify: `python -c "from dualmatfit.fitting.covariance import CovarianceReport, accurate_hessian; print('OK')"`
- [ ] Commit: `feat(fitting): add covariance module with MLE estimation and Hessian computation`

**File**: `dualmatfit/fitting/covariance.py` (1130 lines)

This module depends on `fitting/identifiability` (Task 3.3). Create this file with
the correct imports but note that it will not fully import until identifiability.py
exists. If implementing out of order, temporarily comment out the identifiability
import and uncomment once Task 3.3 is complete.

**Key contents** (copy exactly from `~/NAS/Repositories/PaperHGOMatFit/dualmatfit/fitting/covariance.py`):

- `ObjectiveFunc` type alias: `Callable[[np.ndarray], float]`
- `_EPS`: machine epsilon for float64
- `_EvalCounter`: transparent wrapper counting objective-function evaluations
- **Phase 1 — Ridders' method**: `find_initial_step`, `ridders_curvature`
- **Phase 2 — Richardson extrapolation**: `central_diff_cross`, `richardson_off_diagonal`
- **Phase 3 — Hessian assembly**: `build_gauss_newton_hessian`, `accurate_hessian`
- **Phase 4 — Polish**: `eigenvalue_polish`
- **Phase 4b — Huang calibration**: `huang_calibration`
- **Phase 5 — CovarianceReport dataclass** with fields:
  - `param_names: Sequence`
  - `param_idx: Sequence`
  - `covariance_matrix: np.ndarray | pd.DataFrame`
  - `standard_errors: np.ndarray`
  - `correlation_matrix: np.ndarray | pd.DataFrame`
  - `hessian_matrix: np.ndarray`
  - `hessian_condition: float`
  - `eigenvalues: np.ndarray | pd.Series`
  - `hessian_diagonal: pd.Series`
  - `confidence_interval: pd.DataFrame`
  - `confidence_level: float`
  - `n_function_evals: int`
  - `polished: bool`
  - `calibrated: bool`
  - `method: str`
- **Distance metrics**: `frobenius_distance`, `correlation_distance`, `g_metric`
- **Covariance estimators**: `robust_covariance_from_cost`, `mle_covariance`, `covariance_from_gauss_newton`
- **I/O**: `save_covariance_report`, `load_covariance_report`
- **Formatting**: `format_params_with_uncertainty`, `build_covariance_summary_table`

**Imports to adapt** (PaperHGOMatFit uses `dualmatfit.` paths — keep those):

```python
from dualmatfit.utils.logging_config import get_logger
from dualmatfit.fitting.identifiability import as_2d_jacobian, as_1d_residual
```

**`__all__`** (19 names):

```python
__all__ = [
    'find_initial_step',
    'ridders_curvature',
    'central_diff_cross',
    'build_gauss_newton_hessian',
    'richardson_off_diagonal',
    'accurate_hessian',
    'eigenvalue_polish',
    'huang_calibration',
    'CovarianceReport',
    'frobenius_distance',
    'correlation_distance',
    'g_metric',
    'robust_covariance_from_cost',
    'mle_covariance',
    'covariance_from_gauss_newton',
    'save_covariance_report',
    'load_covariance_report',
    'format_params_with_uncertainty',
    'build_covariance_summary_table',
]
```

---

## Task 3.3 — Create `fitting/identifiability.py`

- [ ] Create `dualmatfit/fitting/identifiability.py` with exact content from PaperHGOMatFit
- [ ] Verify: `python -c "from dualmatfit.fitting.identifiability import ConditioningReport, as_2d_jacobian; print('OK')"`
- [ ] Commit: `feat(fitting): add identifiability module with conditioning diagnostics`

**File**: `dualmatfit/fitting/identifiability.py` (284 lines)

**Key contents** (copy exactly from `~/NAS/Repositories/PaperHGOMatFit/dualmatfit/fitting/identifiability.py`):

- `ConditioningReport` dataclass with `slots=True` and fields:
  - `param_names: Sequence[str]`
  - `cost: float`
  - `jacobian_shape: tuple`
  - `singular_values: np.ndarray`
  - `condition_number_jtj: float`
  - `effective_rank: int`
  - `beta_variance_proxy: float`
  - `beta_k1_cosine_similarity: float`
  - `beta_k1_cosine: float`
- `as_2d_jacobian`: reshape 1D Jacobian to 2D
- `as_1d_residual`: flatten residuals to 1D
- `cosine_similarity`: compute cosine similarity between two vectors
- `beta_variance_proxy`: compute variance proxy from normal matrix
- `analyze_cost_integrator`: full identifiability analysis from `CostIntegrator`
- Lazy import of `dualmatfit.optimization.cost.CostIntegrator` to avoid circular dependency

**`__all__`** (6 names):

```python
__all__ = [
    "ConditioningReport",
    "as_1d_residual",
    "as_2d_jacobian",
    "analyze_cost_integrator",
    "beta_variance_proxy",
    "cosine_similarity",
]
```

**Important**: This module lazy-imports `CostIntegrator` from `dualmatfit.optimization.cost`
inside functions that need it, to avoid circular imports between `fitting` and
`optimization` subpackages. The lazy import pattern from PaperHGOMatFit must be
preserved exactly:

```python
# Deferred import to avoid circular dependency (covariance <-> identifiability)
# build_gauss_newton_hessian is imported lazily inside functions that need it.
```

---

## Task 3.4 — Create `fitting/persistence.py`

- [ ] Create `dualmatfit/fitting/persistence.py` with exact content from PaperHGOMatFit
- [ ] Verify: `python -c "from dualmatfit.fitting.persistence import FitPersistenceMixin; print('OK')"`
- [ ] Commit: `feat(fitting): add persistence mixin for save/load operations`

**File**: `dualmatfit/fitting/persistence.py` (269 lines)

**Key contents** (copy exactly from `~/NAS/Repositories/PaperHGOMatFit/dualmatfit/fitting/persistence.py`):

- `FitPersistenceMixin` class with methods:
  - `_save_data(file_path, dsvars)`: delegate to `self.io_handler.save_optimization_results`
  - `save_data()`: iterate over `model_opt_res` keys, save parquet + baseline data
  - `_save_covariance_for_rat(rat_id, report)`: persist `CovarianceReport` via `save_covariance_report`
  - `load_results()`: reload saved results via `load_parquet_results`

**Imports to adapt**:

```python
from dualmatfit.utils.io_utils import MaterialFitIO, load_parquet_results
from dualmatfit.utils.path_manager import PathLike
from dualmatfit.fitting.covariance import save_covariance_report
from dualmatfit.fitting.constants import (
    UNSTRETCHED_STATE,
    HIGH_RESOLUTION_NCONTROL,
)
```

**`__all__`**: `['FitPersistenceMixin']`

---

## Task 3.5 — Create `fitting/optimization.py`

- [ ] Create `dualmatfit/fitting/optimization.py` with exact content from PaperHGOMatFit
- [ ] Verify: `python -c "from dualmatfit.fitting.optimization import FitOptimizationMixin; print('OK')"`
- [ ] Commit: `feat(fitting): add optimization mixin for parameter fitting`

**File**: `dualmatfit/fitting/optimization.py` (1082 lines)

**Key contents** (copy exactly from `~/NAS/Repositories/PaperHGOMatFit/dualmatfit/fitting/optimization.py`):

- `FitOptimizationMixin` class with methods:
  - `find_baseline_parameters(ftype, miter, **kwargs)`: global optimization
  - `_eval_problem(...)`: evaluate cost for a parameter set
  - `_configure_optimization_parameters(...)`: set up optimization parameters
  - `_get_local_solution_path(...)`: resolve local solution directory
  - `_prepare_cost_and_data(...)`: prepare CostIntegrator and experimental data
  - `_optimize_single_position(...)`: optimize a single position
  - `_optimize_section_positions(...)`: optimize all positions in a section
  - `_save_rat_optimization_results(...)`: save optimization metadata
  - `_generate_aggregate_plots(...)`: generate summary plots
  - `find_optimal_parameters(...)`: main entry point for section-specific optimization

**Imports to adapt**:

```python
from dualmatfit.optimization.cost import CostFunction, CostIntegrator
from dualmatfit.optimization.drivers import opt_solvers
from dualmatfit.data.experimental import InstronData
from dualmatfit.plotting.experimental_visuals import plot_material_fit, exp_test_plot, stress_plot
from dualmatfit.fitting.constants import (
    DEFAULT_BASELINE_OPTIMIZATION_ITERATIONS,
    DEFAULT_LOCAL_OPTIMIZATION_ITERATIONS,
    DEFAULT_GLOBAL_ITERATIONS,
    HIGH_RESOLUTION_NCONTROL,
    UNSTRETCHED_STATE,
    DEFAULT_PLOT_LIMITS,
)
```

**`__all__`**: `['FitOptimizationMixin']`

---

## Task 3.6 — Create `fitting/visualization.py`

- [ ] Create `dualmatfit/fitting/visualization.py` with exact content from PaperHGOMatFit
- [ ] Verify: `python -c "from dualmatfit.fitting.visualization import FitVisualizationMixin; print('OK')"`
- [ ] Commit: `feat(fitting): add visualization mixin for plot generation`

**File**: `dualmatfit/fitting/visualization.py` (315 lines)

**Key contents** (copy exactly from `~/NAS/Repositories/PaperHGOMatFit/dualmatfit/fitting/visualization.py`):

- `FitVisualizationMixin` class with methods:
  - `_setup_plot_fit(model_fit)`: organize model results by anatomical section (Ar, Tr, Ab)
  - `_plot_stress(plot_solution, path, show)`: generate stress component plots
  - `plot_fit(show)`: main entry point, calls `_setup_plot_fit` then `_plot_stress`
  - `correlation(param)`: correlation matrix visualization

**Imports to adapt**:

```python
from dualmatfit.plotting.experimental_visuals import exp_test_plot, stress_plot
from dualmatfit.plotting.parameters import NAME_SECTIONS
from dualmatfit.fitting.constants import (
    DEFAULT_PLOT_LIMITS,
    UNSTRETCHED_STATE,
    HIGH_RESOLUTION_NCONTROL,
)
```

**`__all__`**: `['FitVisualizationMixin']`

---

## Task 3.7 — Decompose `fitting/core.py` into mixins

- [ ] Update `dualmatfit/fitting/core.py` to import and inherit from mixins
- [ ] Remove from `core.py` all methods that are now in mixin modules
- [ ] Verify: `python -c "from dualmatfit.fitting.core import AnisoMaterialFit; print('OK')"`
- [ ] Commit: `refactor(fitting): decompose AnisoMaterialFit into mixin classes`

**File**: `dualmatfit/fitting/core.py`

After this task, `AnisoMaterialFit` should be a thin class that inherits from the
mixins. The PaperHGOMatFit `core.py` shows the target structure:

```python
class AnisoMaterialFit(
    FitOptimizationMixin,
    FitPersistenceMixin,
    FitVisualizationMixin,
    AnisoModelSolve,
):
    """Similar Material Fitting Implementation..."""
    # Only methods that remain directly on AnisoMaterialFit
    # (i.e., methods not extracted into any mixin)
```

**What stays in `core.py`**:
- `AnisoModelSolve` base class (unchanged)
- `AnisoMaterialFit` class definition with mixin inheritance
- Any methods on `AnisoMaterialFit` that were NOT extracted to mixins
- The module-level imports for the mixins and covariance

**What gets removed from `core.py`** (methods now in mixins):
- All `FitOptimizationMixin` methods (Task 3.5)
- All `FitPersistenceMixin` methods (Task 3.4)
- All `FitVisualizationMixin` methods (Task 3.6)

**New imports to add to `core.py`**:

```python
from dualmatfit.fitting.covariance import (
    robust_covariance_from_cost,
    covariance_from_gauss_newton,
    save_covariance_report,
    CovarianceReport,
)
from dualmatfit.fitting.constants import *  # noqa: F401,F403
from dualmatfit.fitting.optimization import FitOptimizationMixin
from dualmatfit.fitting.persistence import FitPersistenceMixin
from dualmatfit.fitting.visualization import FitVisualizationMixin
```

**Note**: The `from dualmatfit.fitting.constants import *` wildcard import is
intentional — it brings all fitting constants into the `AnisoMaterialFit` namespace,
matching PaperHGOMatFit behavior. The `noqa` suppresses the ruff/star-import lint.

---

## Task 3.8 — Update `fitting/__init__.py`

- [ ] Update `dualmatfit/fitting/__init__.py` to re-export new public symbols
- [ ] Verify: `python -c "from dualmatfit.fitting import AnisoMaterialFit, CovarianceReport, ConditioningReport; print('OK')"`
- [ ] Commit: `feat(fitting): update __init__ to export new public symbols`

**File**: `dualmatfit/fitting/__init__.py`

Target content (matching PaperHGOMatFit structure):

```python
# -*- coding: utf-8 -*-
"""Material parameter fitting: orchestration, optimization, persistence, visualization."""

from dualmatfit.fitting.core import AnisoModelSolve, AnisoMaterialFit
from dualmatfit.fitting.covariance import CovarianceReport
from dualmatfit.fitting.identifiability import ConditioningReport

__all__ = [
    "AnisoModelSolve",
    "AnisoMaterialFit",
    "CovarianceReport",
    "ConditioningReport",
]
```

**Rationale**: `CovarianceReport` and `ConditioningReport` are the primary
dataclasses that downstream users will construct and inspect. The low-level
functions (`accurate_hessian`, `ridders_curvature`, etc.) are accessed directly
from their submodules when needed, not through `__init__`.

---

## Dependency Order

```
3.1 constants.py          (no internal deps)
3.3 identifiability.py    (lazy-imports optimization.cost; no fitting/ deps)
3.2 covariance.py         (imports fitting.identifiability)
3.4 persistence.py         (imports fitting.constants, fitting.covariance)
3.5 optimization.py        (imports fitting.constants)
3.6 visualization.py       (imports fitting.constants)
3.7 core.py decomposition  (imports all 3 mixins + covariance)
3.8 __init__.py update
```

Tasks 3.1 and 3.3 can be done in parallel (no mutual dependencies).
Task 3.2 depends on 3.3 (covariance imports identifiability).
Tasks 3.4, 3.5, 3.6 can be done in parallel after 3.1 and 3.2.
Task 3.7 depends on all prior tasks.
Task 3.8 is the final step.

---

## Verification Checklist

After all tasks are complete:

- [ ] `python -c "from dualmatfit.fitting.constants import *; print('OK')"` passes
- [ ] `python -c "from dualmatfit.fitting.identifiability import ConditioningReport, analyze_cost_integrator; print('OK')"` passes
- [ ] `python -c "from dualmatfit.fitting.covariance import CovarianceReport, accurate_hessian; print('OK')"` passes
- [ ] `python -c "from dualmatfit.fitting.persistence import FitPersistenceMixin; print('OK')"` passes
- [ ] `python -c "from dualmatfit.fitting.optimization import FitOptimizationMixin; print('OK')"` passes
- [ ] `python -c "from dualmatfit.fitting.visualization import FitVisualizationMixin; print('OK')"` passes
- [ ] `python -c "from dualmatfit.fitting.core import AnisoMaterialFit; print('OK')"` passes
- [ ] `python -c "from dualmatfit.fitting import AnisoMaterialFit, CovarianceReport, ConditioningReport; print('OK')"` passes
- [ ] `ruff check dualmatfit/fitting/` passes with no errors
- [ ] Existing tests still pass: `pytest tests/ -x`