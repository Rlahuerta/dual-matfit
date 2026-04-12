# Phase 1: Package Restructuring + Conda Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the flat dualmatfit package into subpackages, replace Poetry with Conda (env name: matfit1d), and update all import paths.

**Architecture:** Move 29 flat .py files into 7 subpackages (formulation/, solvers/, optimization/, fitting/, data/, plotting/, utils/). Split cost_functions.py + least_square.py into loss.py + cost.py. Decompose plot.py into plotting/ files. Keep material_fit.py as fitting/core.py (monolithic for now — mixin decomposition is Phase 3). Replace Poetry dependency management with Conda environment.yml.

**Tech Stack:** Python 3.11-3.13, Conda/Micromamba, pytest, ruff, mypy

---

## Task 1: Create environment.yml and update pyproject.toml

**Files:**
- Create: `environment.yml`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create environment.yml**

```yaml
name: matfit1d
channels:
  - defaults
  - conda-forge
dependencies:
  - python>=3.11,<3.14
  - numpy>=2.2.0
  - scipy>=1.16.1
  - pandas>=2.3.1
  - matplotlib>=3.8.2
  - sympy>=1.14.0
  - seaborn>=0.12.0
  - ipyopt>=0.12.10
  - pytables>=3.10.2
  - openpyxl>=3.1.5
  - scikit-learn>=1.1.3
  - pyarrow>=21.0.0
  - fastparquet>=2023.4.0
  - numexpr>=2.11.0
  - numba>=0.61.2
  - cython>=3.1.2
  - joblib>=1.4.2
  - dill>=0.3.8
  - pathos>=0.3.2
  - setuptools>=74.1.2
  - ipython>=9.5.0
  - jax>=0.7.0
  - jaxlib>=0.7.0
  - multiprocess>=0.70.17
  - pip:
    - pathlib>=1.0.1
    - latex>=0.7.0
    - jaxopt>=0.8.5
    - timeout-decorator>=0.5.0
    - -e .
```

- [ ] **Step 2: Strip Poetry-specific sections from pyproject.toml**

Remove `[tool.poetry.dependencies]`, `[tool.poetry.group.dev.dependencies]`, `[tool.poetry.group.lint.dependencies]`, `[tool.poetry.source]`, and the Poetry build-system requirement. Keep `[tool.ruff]`, `[tool.mypy]`, and add a setuptools build-system. The result should look like:

```toml
[build-system]
requires = ["setuptools>=74.1.2"]
build-backend = "setuptools.build_meta"

[project]
name = "dualmatfit"
version = "0.1.0"
description = "Dual Material Fitting for 1D"
requires-python = ">=3.11,<3.14"
license = {text = "Other/Proprietary License"}
classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
]

# Dependencies are managed via environment.yml (conda)
# Dev/lint dependencies are managed via pip install in the conda env

[tool.mypy]
warn_unused_configs = true
exclude = ["tests/"]

[[tool.mypy.overrides]]
module = [
    "antlr4.*", "Cython.*", "flint.*", "gmpy.*", "gmpy2.*",
    "IPython.*", "lxml.*", "matchpy.*", "matplotlib.*", "mpmath.*",
    "numpy.*", "PIL.*", "pycosat.*", "pyglet.*", "pymc.*", "pymc3.*",
    "python-sat.*", "pytest.*", "_pytest.*", "sage.*", "scipy.*",
    "symengine.*", "aesara.*", "xml.*", "pydy.*", "theano.*",
    "multiset.*", "pysat.*", "bpython.*",
]
ignore_missing_imports = true

[tool.ruff]
lint.select = [
    "B015", "C4", "E", "F", "FURB", "LOG", "PIE810", "PLE",
    "PLR1736", "PLW0602", "SIM101", "SLOT", "TRY002",
]
lint.ignore = [
    "E401", "E402", "E501", "E701", "E702", "E711", "E712",
    "E713", "E714", "E721", "E731", "E741", "E743",
    "FURB161", "FURB187",
]
exclude = ["tests/"]
line-length = 88
lint.dummy-variable-rgx = "^(_+|_[a-zA-Z0-9_]*)$"
```

- [ ] **Step 3: Verify environment can be created**

Run: `conda env create -f environment.yml --dry-run`
Expected: Success (or resolves dependencies)

---

## Task 2: Create subpackage directories and __init__.py files

**Files:**
- Create: `dualmatfit/formulation/__init__.py`
- Create: `dualmatfit/solvers/__init__.py`
- Create: `dualmatfit/optimization/__init__.py`
- Create: `dualmatfit/fitting/__init__.py`
- Create: `dualmatfit/data/__init__.py`
- Create: `dualmatfit/utils/__init__.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p dualmatfit/formulation dualmatfit/solvers dualmatfit/optimization dualmatfit/fitting dualmatfit/data dualmatfit/utils
```

- [ ] **Step 2: Create formulation/__init__.py**

```python
# -*- coding: utf-8 -*-
"""Symbolic formulation: material laws, variational forms, tensor algebra, lambdify."""

from dualmatfit.formulation.variational import VariationalFormulation
from dualmatfit.formulation.lambdify import LambdifyBuilder
from dualmatfit.formulation.tensor import TensorManager

__all__ = ["VariationalFormulation", "LambdifyBuilder", "TensorManager"]
```

- [ ] **Step 3: Create solvers/__init__.py**

```python
# -*- coding: utf-8 -*-
"""Numerical solvers: root-finding, extension solution, derivatives, barrier methods."""

from dualmatfit.solvers.solution import Root
from dualmatfit.solvers.extension import ExtensionSolution, DesignVariablesMixin, check_dsvars

__all__ = ["Root", "ExtensionSolution", "DesignVariablesMixin", "check_dsvars"]
```

- [ ] **Step 4: Create optimization/__init__.py**

```python
# -*- coding: utf-8 -*-
"""Optimization: cost functions, loss functions, regularization, drivers."""

from dualmatfit.optimization.cost import CostFunction, CostIntegrator, LSQFit
from dualmatfit.optimization.cache import CostCache, LimitedOrderedDict
from dualmatfit.optimization.regularization import (
    RegularizationStrategy,
    L2Regularization,
    VolumeRegularization,
    CompositeRegularization,
)

__all__ = [
    "CostFunction", "CostIntegrator", "LSQFit",
    "CostCache", "LimitedOrderedDict",
    "RegularizationStrategy", "L2Regularization",
    "VolumeRegularization", "CompositeRegularization",
]
```

- [ ] **Step 5: Create fitting/__init__.py**

```python
# -*- coding: utf-8 -*-
"""Material parameter fitting: orchestration, optimization, persistence, visualization."""

from dualmatfit.fitting.core import AnisoModelSolve, AnisoMaterialFit

__all__ = ["AnisoModelSolve", "AnisoMaterialFit"]
```

- [ ] **Step 6: Create data/__init__.py**

```python
# -*- coding: utf-8 -*-
"""Experimental data loading and processing."""

from dualmatfit.data.experimental import InstronData

__all__ = ["InstronData"]
```

- [ ] **Step 7: Create utils/__init__.py**

```python
# -*- coding: utf-8 -*-
"""Cross-cutting utilities: logging, numeric helpers, I/O, path management."""

from dualmatfit.utils.logging_config import get_logger, setup_logging
from dualmatfit.utils.numeric import (
    sanitize_array,
    sanitize_gradient,
    has_nan,
    has_inf,
    is_finite,
    has_non_finite,
    safe_divide,
)
from dualmatfit.utils.path_manager import PathConfiguration, PathManager

__all__ = [
    "get_logger", "setup_logging",
    "sanitize_array", "sanitize_gradient", "has_nan", "has_inf",
    "is_finite", "has_non_finite", "safe_divide",
    "PathConfiguration", "PathManager",
]
```

---

## Task 3: Move formulation files

**Files:**
- Move: `dualmatfit/material_law.py` → `dualmatfit/formulation/material_law.py`
- Move + rename: `dualmatfit/variational_form.py` → `dualmatfit/formulation/variational.py`
- Move: `dualmatfit/tensor.py` → `dualmatfit/formulation/tensor.py`
- Move: `dualmatfit/simplify.py` → `dualmatfit/formulation/simplify.py`
- Move + rename: `dualmatfit/lambdify_builder.py` → `dualmatfit/formulation/lambdify.py`

- [ ] **Step 1: Move files**

```bash
git mv dualmatfit/material_law.py dualmatfit/formulation/material_law.py
git mv dualmatfit/variational_form.py dualmatfit/formulation/variational.py
git mv dualmatfit/tensor.py dualmatfit/formulation/tensor.py
git mv dualmatfit/simplify.py dualmatfit/formulation/simplify.py
git mv dualmatfit/lambdify_builder.py dualmatfit/formulation/lambdify.py
```

- [ ] **Step 2: Update imports in each formulation file**

In each file, replace internal imports using this mapping:

| Old import | New import |
|---|---|
| `from dualmatfit.material_law` | `from dualmatfit.formulation.material_law` |
| `from dualmatfit.variational_form` | `from dualmatfit.formulation.variational` |
| `from dualmatfit.tensor` | `from dualmatfit.formulation.tensor` |
| `from dualmatfit.simplify` | `from dualmatfit.formulation.simplify` |
| `from dualmatfit.lambdify_builder` | `from dualmatfit.formulation.lambdify` |
| `from dualmatfit.logging_config` | `from dualmatfit.utils.logging_config` |
| `from dualmatfit.numeric_utils` | `from dualmatfit.utils.numeric` |

Use ruff or sed for the replacements. Example for formulation/material_law.py:
```bash
sed -i 's/from dualmatfit\.simplify/from dualmatfit.formulation.simplify/g' dualmatfit/formulation/material_law.py
sed -i 's/from dualmatfit\.tensor/from dualmatfit.formulation.tensor/g' dualmatfit/formulation/material_law.py
sed -i 's/from dualmatfit\.logging_config/from dualmatfit.utils.logging_config/g' dualmatfit/formulation/material_law.py
```

Repeat for each file in the subpackage.

- [ ] **Step 3: Verify formulation imports resolve**

Run: `python -c "from dualmatfit.formulation import VariationalFormulation, LambdifyBuilder, TensorManager; print('OK')"`
Expected: `OK`

---

## Task 4: Move solvers files

**Files:**
- Move: `dualmatfit/solution.py` → `dualmatfit/solvers/solution.py`
- Move + rename: `dualmatfit/extension_solution.py` → `dualmatfit/solvers/extension.py`
- Move: `dualmatfit/derivative.py` → `dualmatfit/solvers/derivative.py`
- Move: `dualmatfit/barrier.py` → `dualmatfit/solvers/barrier.py`

- [ ] **Step 1: Move files**

```bash
git mv dualmatfit/solution.py dualmatfit/solvers/solution.py
git mv dualmatfit/extension_solution.py dualmatfit/solvers/extension.py
git mv dualmatfit/derivative.py dualmatfit/solvers/derivative.py
git mv dualmatfit/barrier.py dualmatfit/solvers/barrier.py
```

- [ ] **Step 2: Update imports in each solvers file**

Apply the full import mapping (see Task 3 table plus):
| Old import | New import |
|---|---|
| `from dualmatfit.solution` | `from dualmatfit.solvers.solution` |
| `from dualmatfit.extension_solution` | `from dualmatfit.solvers.extension` |
| `from dualmatfit.derivative` | `from dualmatfit.solvers.derivative` |
| `from dualmatfit.barrier` | `from dualmatfit.solvers.barrier` |
| `from dualmatfit.numeric_utils` | `from dualmatfit.utils.numeric` |
| `from dualmatfit.logging_config` | `from dualmatfit.utils.logging_config` |

Plus any cross-references to formulation, optimization, data, utils modules.

- [ ] **Step 3: Verify solvers imports resolve**

Run: `python -c "from dualmatfit.solvers import Root, ExtensionSolution, DesignVariablesMixin; print('OK')"`
Expected: `OK`

---

## Task 5: Split cost_functions.py + least_square.py → loss.py + cost.py

**Files:**
- Create: `dualmatfit/optimization/loss.py` (standalone loss functions from cost_functions.py)
- Create: `dualmatfit/optimization/cost.py` (LSQFit, CostFunction, CostIntegrator from least_square.py)
- Delete: `dualmatfit/cost_functions.py`, `dualmatfit/least_square.py`

This is a split, not just a move. Read the current files to identify which functions/classes go where:

**loss.py** gets from cost_functions.py:
- `_ensure_2d_residuum`, `_ensure_3d_jacobian`
- `lsq_fval`, `lsq_dfval`, `lsq_wise_fval`, `lsq_wise_dfval`
- `cauchy_fval`, `cauchy_dfval`
- `huber_fval`, `huber_dfval`
- `logcosh_fval`, `logcosh_dfval`
- `ln_fval`, `ln_dfval`
- `sum_lsq_fun`, `sum_lsq_fun_diff`, `ln_lsq_fun`, `ln_lsq_fun_diff`

**cost.py** gets from least_square.py:
- `LSQFit` class
- `CostFunction` class
- `CostIntegrator` class
- Imports from `dualmatfit.optimization.loss`

- [ ] **Step 1: Move least_square.py to optimization/cost.py**

```bash
git mv dualmatfit/least_square.py dualmatfit/optimization/cost.py
```

- [ ] **Step 2: Create optimization/loss.py from cost_functions.py**

Copy cost_functions.py to optimization/loss.py, then edit cost.py to import from loss.py instead of cost_functions.py. Remove the loss function definitions from cost.py (they now live in loss.py).

```bash
cp dualmatfit/cost_functions.py dualmatfit/optimization/loss.py
```

Then edit `dualmatfit/optimization/loss.py` to update its imports to use new paths.

Then edit `dualmatfit/optimization/cost.py` to:
- Change `from dualmatfit.cost_functions import ...` to `from dualmatfit.optimization.loss import ...`
- Remove any loss function definitions that now live in loss.py

- [ ] **Step 3: Delete original files**

```bash
git rm dualmatfit/cost_functions.py
```

- [ ] **Step 4: Update imports in loss.py and cost.py**

Apply the full import mapping to both files.

- [ ] **Step 5: Verify optimization imports resolve**

Run: `python -c "from dualmatfit.optimization import CostFunction, CostIntegrator, LSQFit, CostCache; print('OK')"`
Expected: `OK`

---

## Task 6: Move remaining optimization files

**Files:**
- Move + rename: `dualmatfit/cost_cache.py` → `dualmatfit/optimization/cache.py`
- Move: `dualmatfit/regularization.py` → `dualmatfit/optimization/regularization.py`
- Move + rename: `dualmatfit/optimization.py` → `dualmatfit/optimization/core.py`
- Move: `dualmatfit/drivers.py` → `dualmatfit/optimization/drivers.py`
- Move: `dualmatfit/basinhopping.py` → `dualmatfit/optimization/basinhopping.py`
- Move: `dualmatfit/ipopt.py` → `dualmatfit/optimization/ipopt.py`

- [ ] **Step 1: Move files**

```bash
git mv dualmatfit/cost_cache.py dualmatfit/optimization/cache.py
git mv dualmatfit/regularization.py dualmatfit/optimization/regularization.py
git mv dualmatfit/optimization.py dualmatfit/optimization/core.py
git mv dualmatfit/drivers.py dualmatfit/optimization/drivers.py
git mv dualmatfit/basinhopping.py dualmatfit/optimization/basinhopping.py
git mv dualmatfit/ipopt.py dualmatfit/optimization/ipopt.py
```

- [ ] **Step 2: Update imports in each file**

Apply the full import mapping. Key changes:
- `from dualmatfit.cost_functions` → `from dualmatfit.optimization.loss`
- `from dualmatfit.least_square` → `from dualmatfit.optimization.cost`
- `from dualmatfit.cost_cache` → `from dualmatfit.optimization.cache`
- `from dualmatfit.regularization` → `from dualmatfit.optimization.regularization`
- `from dualmatfit.optimization` → `from dualmatfit.optimization.core`
- `from dualmatfit.drivers` → `from dualmatfit.optimization.drivers`
- `from dualmatfit.basinhopping` → `from dualmatfit.optimization.basinhopping`
- `from dualmatfit.ipopt` → `from dualmatfit.optimization.ipopt`
- Plus all other subpackage paths

---

## Task 7: Move material_fit.py to fitting/core.py

**Files:**
- Move + rename: `dualmatfit/material_fit.py` → `dualmatfit/fitting/core.py`

In Phase 1, we move it as-is (no mixin decomposition). The file stays monolithic.

- [ ] **Step 1: Move file**

```bash
git mv dualmatfit/material_fit.py dualmatfit/fitting/core.py
```

- [ ] **Step 2: Update imports in fitting/core.py**

Apply the full import mapping. Key changes:
- `from dualmatfit.variational_form` → `from dualmatfit.formulation.variational`
- `from dualmatfit.material_law` → `from dualmatfit.formulation.material_law`
- `from dualmatfit.extension_solution` → `from dualmatfit.solvers.extension`
- `from dualmatfit.solution` → `from dualmatfit.solvers.solution`
- `from dualmatfit.cost_functions` → `from dualmatfit.optimization.loss`
- `from dualmatfit.least_square` → `from dualmatfit.optimization.cost`
- `from dualmatfit.drivers` → `from dualmatfit.optimization.drivers`
- `from dualmatfit.experimental` → `from dualmatfit.data.experimental`
- `from dualmatfit.rato_info` → `from dualmatfit.data.rato_info`
- `from dualmatfit.numeric_utils` → `from dualmatfit.utils.numeric`
- `from dualmatfit.logging_config` → `from dualmatfit.utils.logging_config`
- `from dualmatfit.path_manager` → `from dualmatfit.utils.path_manager`
- `from dualmatfit.latex_post` → `from dualmatfit.utils.latex_post`
- `from dualmatfit.io_utils` → `from dualmatfit.utils.io_utils`

---

## Task 8: Move data files

**Files:**
- Move: `dualmatfit/experimental.py` → `dualmatfit/data/experimental.py`
- Move: `dualmatfit/rato_info.py` → `dualmatfit/data/rato_info.py`

- [ ] **Step 1: Move files**

```bash
git mv dualmatfit/experimental.py dualmatfit/data/experimental.py
git mv dualmatfit/rato_info.py dualmatfit/data/rato_info.py
```

- [ ] **Step 2: Update imports**

In experimental.py: `from dualmatfit.logging_config` → `from dualmatfit.utils.logging_config`, etc.
In rato_info.py: update any internal imports.

---

## Task 9: Move utils files

**Files:**
- Move + rename: `dualmatfit/numeric_utils.py` → `dualmatfit/utils/numeric.py`
- Move: `dualmatfit/io_utils.py` → `dualmatfit/utils/io_utils.py`
- Move: `dualmatfit/logging_config.py` → `dualmatfit/utils/logging_config.py`
- Move: `dualmatfit/path_manager.py` → `dualmatfit/utils/path_manager.py`
- Move: `dualmatfit/log_contexts.py` → `dualmatfit/utils/log_contexts.py`
- Move: `dualmatfit/latex_post.py` → `dualmatfit/utils/latex_post.py`

- [ ] **Step 1: Move files**

```bash
git mv dualmatfit/numeric_utils.py dualmatfit/utils/numeric.py
git mv dualmatfit/io_utils.py dualmatfit/utils/io_utils.py
git mv dualmatfit/logging_config.py dualmatfit/utils/logging_config.py
git mv dualmatfit/path_manager.py dualmatfit/utils/path_manager.py
git mv dualmatfit/log_contexts.py dualmatfit/utils/log_contexts.py
git mv dualmatfit/latex_post.py dualmatfit/utils/latex_post.py
```

- [ ] **Step 2: Update imports in each utils file**

Apply the full import mapping within each file.

---

## Task 10: Decompose plot.py into plotting/ files

**Files:**
- Modify: `dualmatfit/plotting/analytical_visuals.py` (add functions from plot.py)
- Modify: `dualmatfit/plotting/experimental_visuals.py` (add functions from plot.py)
- Create: `dualmatfit/plotting/solution_visuals.py` (PlotSolution2D class)
- Modify: `dualmatfit/plotting/parameters.py` (add constants from plot.py)
- Delete: `dualmatfit/plot.py`

The current `dualmatfit/plot.py` contains:
- `ese_plot`, `mat_plot` → move to `plotting/analytical_visuals.py`
- `stress_plot`, `exp_test_plot`, `plot_time_extension`, `plot_time_load`, `plot_extension_load`, `plot_reaction_force`, `plot_volume_change`, `plot_pk1_stress` → move to `plotting/experimental_visuals.py`
- `PlotSolution2D` → move to `plotting/solution_visuals.py`
- Inline constants (figure sizes, DPI, limits, etc.) → extract to `plotting/parameters.py`

- [ ] **Step 1: Read current plot.py to identify exact function assignments**

Read `dualmatfit/plot.py` and categorize every function/class/constant.

- [ ] **Step 2: Move functions to appropriate plotting subpackage files**

Move each function to the correct file. Update imports in each file to use new subpackage paths.

- [ ] **Step 3: Update plotting/__init__.py**

Add `PlotSolution2D`, `stress_plot`, `exp_test_plot`, etc. to the `__all__` and import them.

- [ ] **Step 4: Delete plot.py**

```bash
git rm dualmatfit/plot.py
```

---

## Task 11: Remove old flat files

**Files:**
- Delete: `dualmatfit/utils.py` (check_dsvars moves to solvers/extension.py; deprecated utilities removed)

- [ ] **Step 1: Verify check_dsvars is available from solvers/extension.py**

After the move in Task 4, `check_dsvars` should be importable from `dualmatfit.solvers.extension`. Verify no other file imports it from `dualmatfit.utils`.

- [ ] **Step 2: Delete utils.py**

```bash
git rm dualmatfit/utils.py
```

---

## Task 12: Update top-level dualmatfit/__init__.py

**Files:**
- Modify: `dualmatfit/__init__.py`

- [ ] **Step 1: Update __init__.py to import from subpackages**

```python
# -*- coding: utf-8 -*-
"""Dual Material Fitting for 1D."""

__version__ = "0.1.0"

from dualmatfit.utils.logging_config import setup_logging

# Initialize logging on import
setup_logging()

from dualmatfit.optimization.cache import CostCache, LimitedOrderedDict
from dualmatfit.optimization.regularization import (
    RegularizationStrategy,
    L2Regularization,
    VolumeRegularization,
    CompositeRegularization,
)
from dualmatfit.utils.numeric import (
    sanitize_array,
    sanitize_gradient,
    has_nan,
    has_inf,
    is_finite,
    has_non_finite,
    safe_divide,
)

__all__ = [
    "__version__",
    "CostCache",
    "LimitedOrderedDict",
    "RegularizationStrategy",
    "L2Regularization",
    "VolumeRegularization",
    "CompositeRegularization",
    "sanitize_array",
    "sanitize_gradient",
    "has_nan",
    "has_inf",
    "is_finite",
    "has_non_finite",
    "safe_divide",
]
```

- [ ] **Step 2: Verify top-level import works**

Run: `python -c "import dualmatfit; print(dualmatfit.__version__); print('OK')"`
Expected: `0.1.0` then `OK`

---

## Task 13: Update test imports

**Files:**
- Modify: all files in `tests/`

- [ ] **Step 1: Update all test file imports**

In every test file, apply the full import mapping. Key changes:
- `from dualmatfit.material_law` → `from dualmatfit.formulation.material_law`
- `from dualmatfit.variational_form` → `from dualmatfit.formulation.variational`
- `from dualmatfit.solution` → `from dualmatfit.solvers.solution`
- `from dualmatfit.extension_solution` → `from dualmatfit.solvers.extension`
- `from dualmatfit.cost_functions` → `from dualmatfit.optimization.loss`
- `from dualmatfit.least_square` → `from dualmatfit.optimization.cost`
- `from dualmatfit.material_fit` → `from dualmatfit.fitting.core`
- etc.

- [ ] **Step 2: Update conftest.py imports**

Same mapping applied to `tests/conftest.py`.

---

## Task 14: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Build & Development Commands section**

Replace Poetry commands with Conda commands:
```bash
# Install (Conda with matfit1d environment)
conda env create -f environment.yml
conda activate matfit1d
pip install -e .

# Run all tests
pytest

# Run a single test file
pytest tests/test_cost_functions.py

# Run a single test by name
pytest tests/test_cost_functions.py::test_specific_name

# Run tests in parallel
pytest -n auto

# Run with coverage
pytest --cov=dualmatfit

# Lint
ruff check dualmatfit/

# Type check (tests excluded)
mypy dualmatfit/

# Format check
ruff format --check dualmatfit/
```

- [ ] **Step 2: Update the Architecture & Data Flow section**

Replace all flat module references with subpackage references (formulation/, solvers/, optimization/, fitting/, data/, utils/).

- [ ] **Step 3: Update the Key Module Map table**

Replace with:
| Area | Modules |
|---|---|
| Symbolic mechanics | `formulation/material_law.py`, `formulation/tensor.py`, `formulation/variational.py`, `formulation/simplify.py` |
| SymPy-to-JAX bridge | `formulation/lambdify.py` |
| Numeric differentiation | `solvers/derivative.py` |
| Forward solve | `solvers/solution.py` (Root) |
| Cost functions | `optimization/loss.py`, `optimization/cost.py`, `optimization/cache.py` |
| Optimization | `optimization/core.py`, `optimization/drivers.py`, `optimization/ipopt.py`, `optimization/basinhopping.py`, `solvers/barrier.py`, `optimization/regularization.py` |
| Data I/O | `data/experimental.py`, `utils/io_utils.py`, `utils/path_manager.py` |
| Plotting | `plotting/` subpackage |
| Utilities | `utils/numeric.py`, `utils/log_contexts.py`, `utils/logging_config.py`, `utils/latex_post.py` |
| LaTeX output | `utils/latex_post.py`, `formulation/lambdify.py` |

---

## Task 15: Run full test suite and verify

- [ ] **Step 1: Activate conda environment and install**

```bash
conda env create -f environment.yml
conda activate matfit1d
pip install -e .
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests pass (or same failures as before migration, just with new import paths).

- [ ] **Step 3: Run ruff check**

```bash
ruff check dualmatfit/
```

Expected: No new errors.

- [ ] **Step 4: Run mypy**

```bash
mypy dualmatfit/
```

Expected: No new errors (existing type errors may remain).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: restructure flat package into subpackages, migrate to Conda

Phase 1 of PaperHGOMatFit migration:
- Replace Poetry with Conda (matfit1d environment)
- Move files into formulation/, solvers/, optimization/, fitting/, data/, utils/ subpackages
- Split cost_functions.py + least_square.py into optimization/loss.py + optimization/cost.py
- Decompose plot.py into plotting/ subpackage files
- Update all import paths across source and tests
- Update CLAUDE.md with new structure and commands

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```