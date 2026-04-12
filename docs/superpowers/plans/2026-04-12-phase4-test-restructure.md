# Phase 4: Test Suite Restructure

**Date:** 2026-04-12
**Phase:** 4 of N (follows Phase 3: module restructuring)
**Goal:** Restructure the flat test suite into a 3-tier hierarchy (unit/, integration/, performance/) and add new test files for Phase 3 modules (covariance, identifiability, Hessian edge cases, analyze_beta_identifiability).

---

## Overview

The current flat `tests/` directory has 22 test files + `conftest.py` + `find_stuck.py` + `__init__.py`. This phase:
1. Creates the 3-tier directory structure
2. Moves existing tests into their correct tier
3. Updates all import paths from flat `dualmatfit.X` to the new subpackage paths
4. Adds 4 new test files from PaperHGOMatFit with adapted imports
5. Updates `pytest.ini` for the new testpaths and markers

---

## Task 1: Create Directory Structure

- [ ] Create `tests/unit/` directory
- [ ] Create `tests/integration/` directory
- [ ] Create `tests/performance/` directory
- [ ] Create `tests/unit/__init__.py` (empty)
- [ ] Create `tests/integration/__init__.py` (empty)
- [ ] Create `tests/performance/__init__.py` (empty)

Commands:
```bash
mkdir -p tests/unit tests/integration tests/performance
touch tests/unit/__init__.py tests/integration/__init__.py tests/performance/__init__.py
```

---

## Task 2: Move Existing Tests into Tier Directories

### 2a. Move unit tests (17 files)

- [ ] `tests/test_barrier.py` -> `tests/unit/test_barrier.py`
- [ ] `tests/test_cost_cache.py` -> `tests/unit/test_cost_cache.py`
- [ ] `tests/test_cost_functions.py` -> `tests/unit/test_cost_functions.py`
- [ ] `tests/test_derivative.py` -> `tests/unit/test_derivative.py`
- [ ] `tests/test_extension_solution.py` -> `tests/unit/test_extension_solution.py`
- [ ] `tests/test_functions.py` -> `tests/unit/test_functions.py`
- [ ] `tests/test_io_utils.py` -> `tests/unit/test_io_utils.py`
- [ ] `tests/test_latex_post.py` -> `tests/unit/test_latex_post.py`
- [ ] `tests/test_logging.py` -> `tests/unit/test_logging.py`
- [ ] `tests/test_numeric_utils.py` -> `tests/unit/test_numeric_utils.py`
- [ ] `tests/test_path_manager.py` -> `tests/unit/test_path_manager.py`
- [ ] `tests/test_regularization.py` -> `tests/unit/test_regularization.py`
- [ ] `tests/test_solution.py` -> `tests/unit/test_solution.py`
- [ ] `tests/test_strain_energy.py` -> `tests/unit/test_strain_energy.py`
- [ ] `tests/test_utils.py` -> `tests/unit/test_utils.py`
- [ ] `tests/test_variational_form.py` -> `tests/unit/test_variational_form.py`
- [ ] `tests/test_variational_form_derivative.py` -> `tests/unit/test_variational_form_derivative.py`

Commands:
```bash
cd tests
git mv test_barrier.py unit/
git mv test_cost_cache.py unit/
git mv test_cost_functions.py unit/
git mv test_derivative.py unit/
git mv test_extension_solution.py unit/
git mv test_functions.py unit/
git mv test_io_utils.py unit/
git mv test_latex_post.py unit/
git mv test_logging.py unit/
git mv test_numeric_utils.py unit/
git mv test_path_manager.py unit/
git mv test_regularization.py unit/
git mv test_solution.py unit/
git mv test_strain_energy.py unit/
git mv test_utils.py unit/
git mv test_variational_form.py unit/
git mv test_variational_form_derivative.py unit/
```

### 2b. Move integration tests (5 files)

- [ ] `tests/test_drivers.py` -> `tests/integration/test_drivers.py`
- [ ] `tests/test_experimental.py` -> `tests/integration/test_experimental.py`
- [ ] `tests/test_integrated_cost_function.py` -> `tests/integration/test_integrated_cost_function.py`
- [ ] `tests/test_jax_integration.py` -> `tests/integration/test_jax_integration.py`
- [ ] `tests/test_plotting.py` -> `tests/integration/test_plotting.py`

Commands:
```bash
cd tests
git mv test_drivers.py integration/
git mv test_experimental.py integration/
git mv test_integrated_cost_function.py integration/
git mv test_jax_integration.py integration/
git mv test_plotting.py integration/
```

### 2c. Move performance tests (1 file)

- [ ] `tests/test_lambdify_performance.py` -> `tests/performance/test_lambdify_performance.py`

Commands:
```bash
cd tests
git mv test_lambdify_performance.py performance/
```

---

## Task 3: Update Import Paths in conftest.py

The file `tests/conftest.py` stays at its current location (shared by all tiers). Update its import.

- [ ] Update `tests/conftest.py`: change `from dualmatfit.variational_form import VariationalFormulation` to `from dualmatfit.formulation.variational import VariationalFormulation`

Exact change in `tests/conftest.py`:
```python
# Line 87 — OLD:
from dualmatfit.variational_form import VariationalFormulation

# Line 87 — NEW:
from dualmatfit.formulation.variational import VariationalFormulation
```

---

## Task 4: Update Import Paths in All Moved Test Files

Each file needs its `dualmatfit.*` imports remapped according to the Phase 3 module restructuring. The full mapping:

| Old import path | New import path |
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
| `dualmatfit.latex_post` | `dualmatfit.utils.latext_post` |
| `dualmatfit.plot` | `dualmatfit.plotting.core` |
| `dualmatfit.plotting.experimental_visuals` | `dualmatfit.plotting.experimental_visuals` |
| `dualmatfit.plotting.analytical_visuals` | `dualmatfit.plotting.analytical_visuals` |
| `dualmatfit.plotting.plot_helpers` | `dualmatfit.plotting.plot_helpers` |
| `dualmatfit.utils` | `dualmatfit.utils.core` |

### 4a. Unit tests

#### `tests/unit/test_barrier.py`
- [ ] `from dualmatfit.solution import ...` -> `from dualmatfit.solvers.solution import ...`

#### `tests/unit/test_cost_cache.py`
- [ ] `from dualmatfit.cost_cache import ...` -> `from dualmatfit.optimization.cache import ...`

#### `tests/unit/test_cost_functions.py`
- [ ] `from dualmatfit.least_square import ...` -> `from dualmatfit.optimization.cost import ...`
- [ ] `from dualmatfit.variational_form import ...` -> `from dualmatfit.formulation.variational import ...`
- [ ] `from dualmatfit.lambdify_builder import ...` -> `from dualmatfit.formulation.lambdify import ...`
- [ ] `from dualmatfit.cost_functions import ...` -> `from dualmatfit.optimization.loss import ...`

#### `tests/unit/test_derivative.py`
- [ ] `from dualmatfit.drivers import ...` -> `from dualmatfit.optimization.drivers import ...`
- [ ] `# from dualmatfit.solution import Root` -> `# from dualmatfit.solvers.solution import Root`
- [ ] `from dualmatfit.experimental import ...` -> `from dualmatfit.data.experimental import ...`
- [ ] `from dualmatfit.least_square import ...` -> `from dualmatfit.optimization.cost import ...`
- [ ] `from dualmatfit.material_law import ...` -> `from dualmatfit.formulation.material_law import ...`
- [ ] `from dualmatfit.variational_form import ...` -> `from dualmatfit.formulation.variational import ...`
- [ ] `from dualmatfit.extension_solution import ...` -> `from dualmatfit.solvers.extension import ...`
- [ ] `from dualmatfit.plot import ...` -> `from dualmatfit.plotting.core import ...`
- [ ] `from dualmatfit.plotting.experimental_visuals import ...` -> `from dualmatfit.plotting.experimental_visuals import ...` (unchanged)

#### `tests/unit/test_extension_solution.py`
- [ ] `from dualmatfit.extension_solution import ...` -> `from dualmatfit.solvers.extension import ...`
- [ ] `from dualmatfit.lambdify_builder import ...` -> `from dualmatfit.formulation.lambdify import ...`
- [ ] `from dualmatfit.variational_form import ...` -> `from dualmatfit.formulation.variational import ...`

#### `tests/unit/test_functions.py`
- [ ] `from dualmatfit.material_law import ...` -> `from dualmatfit.formulation.material_law import ...`
- [ ] `from dualmatfit.tensor import ...` -> `from dualmatfit.formulation.tensor import ...`

#### `tests/unit/test_io_utils.py`
- [ ] `from dualmatfit.io_utils import ...` -> `from dualmatfit.utils.io_utils import ...`

#### `tests/unit/test_latex_post.py`
- [ ] `from dualmatfit.latex_post import ...` -> `from dualmatfit.utils.latext_post import ...`

#### `tests/unit/test_logging.py`
- [ ] `from dualmatfit.logging_config import ...` -> `from dualmatfit.utils.logging_config import ...`
- [ ] `from dualmatfit.log_contexts import ...` -> `from dualmatfit.utils.log_contexts import ...`

#### `tests/unit/test_numeric_utils.py`
- [ ] `from dualmatfit.numeric_utils import ...` -> `from dualmatfit.utils.numeric import ...`

#### `tests/unit/test_path_manager.py`
- [ ] `from dualmatfit.path_manager import ...` -> `from dualmatfit.utils.path_manager import ...`

#### `tests/unit/test_regularization.py`
- [ ] `from dualmatfit.cost_cache import ...` -> `from dualmatfit.optimization.cache import ...`
- [ ] `from dualmatfit.regularization import ...` -> `from dualmatfit.optimization.regularization import ...`

#### `tests/unit/test_solution.py`
- [ ] `from dualmatfit.extension_solution import ...` -> `from dualmatfit.solvers.extension import ...`
- [ ] `from dualmatfit.plot import ...` -> `from dualmatfit.plotting.core import ...`
- [ ] `from dualmatfit.solution import ...` -> `from dualmatfit.solvers.solution import ...`
- [ ] `from dualmatfit.variational_form import ...` -> `from dualmatfit.formulation.variational import ...`

#### `tests/unit/test_strain_energy.py`
- [ ] `from dualmatfit.lambdify_builder import ...` -> `from dualmatfit.formulation.lambdify import ...`
- [ ] `from dualmatfit.material_law import ...` -> `from dualmatfit.formulation.material_law import ...`
- [ ] `from dualmatfit.plotting.analytical_visuals import ...` -> `from dualmatfit.plotting.analytical_visuals import ...` (unchanged)
- [ ] `from dualmatfit.plotting.plot_helpers import ...` -> `from dualmatfit.plotting.plot_helpers import ...` (unchanged)
- [ ] `from dualmatfit.utils import ...` -> `from dualmatfit.utils.core import ...`
- [ ] `from dualmatfit.variational_form import ...` -> `from dualmatfit.formulation.variational import ...`

#### `tests/unit/test_utils.py`
- [ ] `from dualmatfit.tensor import ...` -> `from dualmatfit.formulation.tensor import ...`
- [ ] `from dualmatfit.tensor import logger as tensor_logger` -> `from dualmatfit.formulation.tensor import logger as tensor_logger` (both occurrences)

#### `tests/unit/test_variational_form.py`
- [ ] `from dualmatfit.variational_form import ...` -> `from dualmatfit.formulation.variational import ...`
- [ ] `from dualmatfit.extension_solution import ...` -> `from dualmatfit.solvers.extension import ...`
- [ ] `from dualmatfit.least_square import ...` -> `from dualmatfit.optimization.cost import ...`
- [ ] `from dualmatfit.material_law import ...` -> `from dualmatfit.formulation.material_law import ...`
- [ ] `from dualmatfit.tensor import ...` -> `from dualmatfit.formulation.tensor import ...`

#### `tests/unit/test_variational_form_derivative.py`
- [ ] `from dualmatfit.extension_solution import ...` -> `from dualmatfit.solvers.extension import ...`
- [ ] `from dualmatfit.material_law import ...` -> `from dualmatfit.formulation.material_law import ...`
- [ ] `from dualmatfit.tensor import ...` -> `from dualmatfit.formulation.tensor import ...`
- [ ] `from dualmatfit.variational_form import ...` -> `from dualmatfit.formulation.variational import ...`

### 4b. Integration tests

#### `tests/integration/test_drivers.py`
- [ ] `from dualmatfit.drivers import ...` -> `from dualmatfit.optimization.drivers import ...`
- [ ] `from dualmatfit.ipopt import ...` -> `from dualmatfit.optimization.ipopt import ...`
- [ ] `from dualmatfit.basinhopping import ...` -> `from dualmatfit.optimization.basinhopping import ...`
- [ ] `from dualmatfit.least_square import ...` -> `from dualmatfit.optimization.cost import ...`
- [ ] All inline `from dualmatfit.basinhopping import ...` -> `from dualmatfit.optimization.basinhopping import ...`
- [ ] All inline `from dualmatfit.ipopt import ...` -> `from dualmatfit.optimization.ipopt import ...`

#### `tests/integration/test_experimental.py`
- [ ] `from dualmatfit.experimental import ...` -> `from dualmatfit.data.experimental import ...`
- [ ] `from dualmatfit.material_fit import ...` -> `from dualmatfit.fitting.core import ...`
- [ ] `from dualmatfit.plotting.experimental_visuals import ...` -> `from dualmatfit.plotting.experimental_visuals import ...` (unchanged)

#### `tests/integration/test_integrated_cost_function.py`
- [ ] `from dualmatfit.cost_functions import ...` -> `from dualmatfit.optimization.loss import ...`
- [ ] `from dualmatfit.least_square import ...` -> `from dualmatfit.optimization.cost import ...`
- [ ] `from dualmatfit.variational_form import ...` -> `from dualmatfit.formulation.variational import ...`

#### `tests/integration/test_jax_integration.py`
- [ ] `from dualmatfit.extension_solution import ...` -> `from dualmatfit.solvers.extension import ...`
- [ ] `from dualmatfit.lambdify_builder import ...` -> `from dualmatfit.formulation.lambdify import ...`
- [ ] `from dualmatfit.variational_form import ...` -> `from dualmatfit.formulation.variational import ...`

#### `tests/integration/test_plotting.py`
- [ ] `from dualmatfit.experimental import ...` -> `from dualmatfit.data.experimental import ...`
- [ ] `from dualmatfit.plotting.experimental_visuals import ...` -> `from dualmatfit.plotting.experimental_visuals import ...` (unchanged)

### 4c. Performance tests

#### `tests/performance/test_lambdify_performance.py`
- [ ] `from dualmatfit.lambdify_builder import ...` -> `from dualmatfit.formulation.lambdify import ...`

---

## Task 5: Add New Test Files

Copy 4 test files from `~/NAS/Repositories/PaperHGOMatFit/tests/unit/` into `tests/unit/`, adapting their imports to match the Phase 3 module structure.

### 5a. `tests/unit/test_covariance.py`

- [ ] Copy `/home/hephaestus/NAS/Repositories/PaperHGOMatFit/tests/unit/test_covariance.py` to `tests/unit/test_covariance.py`
- [ ] Update imports: `from dualmatfit.fitting.covariance import ...` (already uses Phase 3 paths — verify no old paths remain)
- [ ] Verify all imports resolve against the Phase 3 package structure

Key imports to verify/adapt:
```python
from dualmatfit.fitting.covariance import (
    find_initial_step, ridders_curvature, richardson_off_diagonal,
    central_diff_cross, accurate_hessian, eigenvalue_polish,
    mle_covariance, CovarianceReport, frobenius_distance,
    correlation_distance, g_metric,
)
```

### 5b. `tests/unit/test_hessian_edge_cases.py`

- [ ] Copy `/home/hephaestus/NAS/Repositories/PaperHGOMatFit/tests/unit/test_hessian_edge_cases.py` to `tests/unit/test_hessian_edge_cases.py`
- [ ] Update imports: `from dualmatfit.solvers.derivative import _hessian_fd, _fdm` (already uses Phase 3 paths — verify)

Key imports to verify/adapt:
```python
from dualmatfit.solvers.derivative import _hessian_fd, _fdm
```

### 5c. `tests/unit/test_identifiability.py`

- [ ] Copy `/home/hephaestus/NAS/Repositories/PaperHGOMatFit/tests/unit/test_identifiability.py` to `tests/unit/test_identifiability.py`
- [ ] Update imports to Phase 3 paths:
  - `from dualmatfit.fitting.identifiability import ...` (verify module exists)
  - `from dualmatfit.optimization.cost import CostIntegrator, LSQFit` (was `dualmatfit.least_square`)

Key imports to verify/adapt:
```python
from dualmatfit.fitting.identifiability import (
    analyze_cost_integrator, as_2d_jacobian, beta_variance_proxy,
)
from dualmatfit.optimization.cost import CostIntegrator, LSQFit
```

### 5d. `tests/unit/test_analyze_beta_identifiability.py`

- [ ] Copy `/home/hephaestus/NAS/Repositories/PaperHGOMatFit/tests/unit/test_analyze_beta_identifiability.py` to `tests/unit/test_analyze_beta_identifiability.py`
- [ ] This file imports from `scripts._analytical_runtime` and `scripts.analyze_beta_identifiability` — these are external script modules, not part of the `dualmatfit` package. Evaluate whether these imports should be:
  - Kept as-is (if `scripts/` is available on `sys.path`)
  - Commented out or marked `@pytest.mark.skip` if the scripts module is not yet migrated
  - Adapted to a new location if the scripts are part of the project
- [ ] Update `dualmatfit` package imports:
  - `from dualmatfit.fitting.covariance import CovarianceReport` (verify)
  - `from dualmatfit.fitting.identifiability import ConditioningReport` (verify)

Key imports to verify/adapt:
```python
from dualmatfit.fitting.covariance import CovarianceReport
from dualmatfit.fitting.identifiability import ConditioningReport
# scripts.* imports — evaluate separately
```

---

## Task 6: Update pytest.ini

- [ ] Update `pytest.ini` at project root with the new testpaths, markers, and filterwarnings.

New `pytest.ini` content:
```ini
[pytest]
testpaths =
    tests/unit
    tests/integration
    tests/performance
addopts = --cov=dualmatfit --cov-report=term-missing --cov-report=html
markers =
    slow: slow-running tests
    integration: multi-module integration tests
    requires_data: tests requiring experimental data files
    fast: fast unit tests
filterwarnings =
    ignore::DeprecationWarning:scipy.optimize
    ignore:.*multi-threaded.*fork.*deadlock.*:DeprecationWarning
    ignore:.*os.fork.*multithreaded.*:RuntimeWarning
    ignore:bitcount function is deprecated:DeprecationWarning:mpmath.libmp.libintmath
```

Changes from current `pytest.ini`:
- Add `testpaths` to restrict discovery to the 3 tier directories
- Add `markers` section for slow, integration, requires_data, fast
- Add `ignore:bitcount function is deprecated:DeprecationWarning:mpmath.libmp.libintmath` filterwarning

---

## Task 7: Handle `find_stuck.py`

The utility script `tests/find_stuck.py` is not a test file. Decide disposition:

- [ ] Option A: Move to `tests/unit/find_stuck.py` alongside unit tests (simplest)
- [ ] Option B: Move to a `scripts/` or `tools/` directory at project root (cleaner separation)
- [ ] Verify `find_stuck.py` does not contain `from dualmatfit.*` imports that need updating (it likely does not, but verify)
- [ ] If Option B chosen: update any references to `find_stuck.py` in CI or developer docs

**Recommendation:** Option A (keep in `tests/`) — move to `tests/unit/find_stuck.py` for simplicity, or leave at `tests/find_stuck.py` since `testpaths` excludes non-test files anyway.

---

## Task 8: Verify and Validate

After all moves and import updates are complete:

- [ ] Run the full unit test suite:
  ```bash
  python -m pytest tests/unit/ -v --tb=short 2>&1 | head -100
  ```
- [ ] Run the integration test suite:
  ```bash
  python -m pytest tests/integration/ -v --tb=short 2>&1 | head -100
  ```
- [ ] Run the performance test suite:
  ```bash
  python -m pytest tests/performance/ -v --tb=short 2>&1 | head -100
  ```
- [ ] Verify `pytest --collect-only` discovers all tests in the 3 tiers
  ```bash
  python -m pytest --collect-only 2>&1 | tail -20
  ```
- [ ] Check for any remaining old-style imports:
  ```bash
  grep -rn 'from dualmatfit\.\(material_law\|variational_form\|tensor\|simplify\|lambdify_builder\|solution\|extension_solution\|derivative\|barrier\|cost_functions\|least_square\|cost_cache\|regularization\|optimization\b\|drivers\|basinhopping\|ipopt\|material_fit\|experimental\|rato_info\|numeric_utils\|io_utils\|logging_config\|path_manager\|log_contexts\|latex_post\|plot\b\)' tests/
  ```
  This should return zero matches. If any remain, update those files.
- [ ] Verify no test files remain in the flat `tests/` directory (except `conftest.py`, `__init__.py`, and optionally `find_stuck.py`):
  ```bash
  ls tests/test_*.py 2>&1
  ```
  Should return "No such file or directory" or empty.

---

## Task 9: Add `@pytest.mark.integration` Decorators

Mark integration and performance test files with appropriate pytest markers for selective test runs.

- [ ] Add `@pytest.mark.integration` to `tests/integration/test_drivers.py`
- [ ] Add `@pytest.mark.integration` to `tests/integration/test_experimental.py`
- [ ] Add `@pytest.mark.integration` to `tests/integration/test_integrated_cost_function.py`
- [ ] Add `@pytest.mark.integration` to `tests/integration/test_jax_integration.py`
- [ ] Add `@pytest.mark.integration` to `tests/integration/test_plotting.py`
- [ ] Add `@pytest.mark.slow` to `tests/performance/test_lambdify_performance.py`

Pattern (add at top of each file, after imports):
```python
import pytest

pytestmark = pytest.mark.integration  # or pytest.mark.slow for performance
```

---

## Task 10: Commit

- [ ] Stage all changes:
  ```bash
  git add tests/ pytest.ini
  ```
- [ ] Verify staging:
  ```bash
  git status
  git diff --cached --stat
  ```
- [ ] Commit:
  ```bash
  git commit -m "Phase 4: Restructure test suite into unit/integration/performance tiers

- Move 17 unit tests to tests/unit/
- Move 5 integration tests to tests/integration/
- Move 1 performance test to tests/performance/
- Update all dualmatfit imports to Phase 3 subpackage paths
- Add test_covariance, test_hessian_edge_cases, test_identifiability,
  test_analyze_beta_identifiability from PaperHGOMatFit
- Update pytest.ini with testpaths, markers, and mpmath filterwarning
- Add integration/slow markers to appropriate test files"
  ```

---

## Summary of File Movements

| Source | Destination |
|---|---|
| `tests/test_barrier.py` | `tests/unit/test_barrier.py` |
| `tests/test_cost_cache.py` | `tests/unit/test_cost_cache.py` |
| `tests/test_cost_functions.py` | `tests/unit/test_cost_functions.py` |
| `tests/test_derivative.py` | `tests/unit/test_derivative.py` |
| `tests/test_extension_solution.py` | `tests/unit/test_extension_solution.py` |
| `tests/test_functions.py` | `tests/unit/test_functions.py` |
| `tests/test_io_utils.py` | `tests/unit/test_io_utils.py` |
| `tests/test_latex_post.py` | `tests/unit/test_latex_post.py` |
| `tests/test_logging.py` | `tests/unit/test_logging.py` |
| `tests/test_numeric_utils.py` | `tests/unit/test_numeric_utils.py` |
| `tests/test_path_manager.py` | `tests/unit/test_path_manager.py` |
| `tests/test_regularization.py` | `tests/unit/test_regularization.py` |
| `tests/test_solution.py` | `tests/unit/test_solution.py` |
| `tests/test_strain_energy.py` | `tests/unit/test_strain_energy.py` |
| `tests/test_utils.py` | `tests/unit/test_utils.py` |
| `tests/test_variational_form.py` | `tests/unit/test_variational_form.py` |
| `tests/test_variational_form_derivative.py` | `tests/unit/test_variational_form_derivative.py` |
| `tests/test_drivers.py` | `tests/integration/test_drivers.py` |
| `tests/test_experimental.py` | `tests/integration/test_experimental.py` |
| `tests/test_integrated_cost_function.py` | `tests/integration/test_integrated_cost_function.py` |
| `tests/test_jax_integration.py` | `tests/integration/test_jax_integration.py` |
| `tests/test_plotting.py` | `tests/integration/test_plotting.py` |
| `tests/test_lambdify_performance.py` | `tests/performance/test_lambdify_performance.py` |

## New Files to Create

| Source | Destination |
|---|---|
| `~/NAS/Repositories/PaperHGOMatFit/tests/unit/test_covariance.py` | `tests/unit/test_covariance.py` |
| `~/NAS/Repositories/PaperHGOMatFit/tests/unit/test_hessian_edge_cases.py` | `tests/unit/test_hessian_edge_cases.py` |
| `~/NAS/Repositories/PaperHGOMatFit/tests/unit/test_identifiability.py` | `tests/unit/test_identifiability.py` |
| `~/NAS/Repositories/PaperHGOMatFit/tests/unit/test_analyze_beta_identifiability.py` | `tests/unit/test_analyze_beta_identifiability.py` |

## Files Remaining in `tests/` Root

- `tests/__init__.py` (unchanged)
- `tests/conftest.py` (import paths updated)
- `tests/find_stuck.py` (disposition decided in Task 7)

---

## Risk Assessment

1. **Import path completeness:** The grep check in Task 8 catches any missed renames. The mapping table covers all 25 old module paths.
2. **New test compatibility:** `test_analyze_beta_identifiability.py` imports from `scripts.*` which may not exist in this repo. Needs evaluation (Task 5d).
3. **Test discovery:** The `testpaths` setting in `pytest.ini` ensures pytest only discovers tests in the 3 tier directories, not the root `tests/` directory.
4. **conftest.py sharing:** A single `tests/conftest.py` is automatically discovered by pytest for all subdirectories. No `conftest.py` duplication needed in subdirectories unless tier-specific fixtures are added later.
5. **`find_stuck.py`**: This utility script is not a test and won't be collected by pytest. Keeping it at `tests/find_stuck.py` is safe.