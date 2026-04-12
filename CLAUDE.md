# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dual Material Fitting (dualmatfit) fits hyperelastic constitutive models to arterial tissue experimental data from Instron uniaxial extension tests. "Dual" refers to the decomposition into isotropic (elastin, Neo-Hookean or Fung) and anisotropic (collagen fibers, Holzapfel-Gasser-Ogden style with Heaviside-switched fiber engagement) strain energy contributions. Tissue is tested across anatomical sections (Adventitia Ar, Media Tr, Intima Ab) at multiple positions.

## Build & Development Commands

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

## Architecture & Data Flow

The pipeline flows through these stages:

1. **Experimental data loading** (`data/experimental.py`): `InstronData` parses Instron test DataFrames (Time, Extension, Load), interpolates to control points, and scales data.

2. **Symbolic formulation** (`formulation/`): `VariationalFormulation` builds strain energy symbolically from material law components. SymPy handles all symbolic tensor algebra (deformation gradient, right Cauchy-Green, isochoric split) and automatic differentiation to derive PK1 stress and stiffness. Supports three formulation types: displacement-only (mtype=1), displacement-pressure (mtype=2), displacement-pressure+Lagrange multiplier (mtype=3).

3. **Numeric conversion** (`formulation/lambdify.py`): Bridges SymPy to JAX by converting symbolic expressions into JAX-callable numeric functions via `sy.lambdify` with JAX backend.

4. **Forward solve** (`solvers/solution.py`): `AnisoModelSolve` uses `Root` (Newton-Raphson) to solve the nonlinear variational problem at each stretch level, yielding model stress predictions.

5. **Cost computation** (`optimization/loss.py` + `optimization/cost.py`): `CostFunction` computes residuals between model and experiment; `CostIntegrator` aggregates across sections.

6. **Optimization** (`optimization/drivers.py` + `optimization/core.py` + `optimization/ipopt.py`): `opt_solvers` dispatches to scipy (SLSQP, L-BFGS-B, TNC), IPOPT (via ipyopt wrapper), or global solvers (basinhopping, differential_evolution, SHGO). Design variables are managed as DataFrames with initial values, bounds, and active flags.

7. **Fitting orchestration** (`fitting/core.py`): `AnisoMaterialFit` extends `AnisoModelSolve` and runs the full pipeline: baseline fitting, local optimization per section, result saving/plotting.

### Key Module Map

| Area | Modules |
|---|---|
| Symbolic mechanics | `formulation/material_law.py`, `formulation/tensor.py`, `formulation/variational.py`, `formulation/simplify.py` |
| SymPy-to-JAX bridge | `formulation/lambdify.py` |
| Numeric differentiation | `solvers/derivative.py` (JAX autodiff) |
| Forward solve | `solvers/solution.py` (Root) |
| Cost functions | `optimization/loss.py`, `optimization/cost.py`, `optimization/cache.py` |
| Optimization | `optimization/core.py`, `optimization/drivers.py`, `optimization/ipopt.py`, `optimization/basinhopping.py`, `solvers/barrier.py`, `optimization/regularization.py` |
| Data I/O | `data/experimental.py`, `utils/io_utils.py`, `utils/path_manager.py` |
| Plotting | `plotting/` subpackage |
| Utilities | `utils/numeric.py`, `utils/log_contexts.py`, `utils/logging_config.py`, `utils/latex_post.py` |
| LaTeX output | `utils/latex_post.py`, `formulation/lambdify.py` |

### Import Path Convention

All imports use subpackage paths:
- `from dualmatfit.formulation.material_law import ...`
- `from dualmatfit.solvers.solution import ...`
- `from dualmatfit.optimization.cost import ...`
- `from dualmatfit.fitting.core import ...`
- `from dualmatfit.data.experimental import ...`
- `from dualmatfit.utils.numeric import ...`

## Key Technical Details

- **SymPy is the backbone**: All constitutive models and variational forms are built symbolically. Do not bypass SymPy when working with material laws.
- **JAX is used targetedly**: Numeric evaluation and autodiff (`jax.jacobian`). Some JAX imports in `solvers/solution.py` are commented out (migration in progress).
- **No CLI or entry points**: The package is used programmatically by instantiating `AnisoMaterialFit`.
- **Volumetric models**: bathe87, simo92, doll8 (configured via material law parameters).
- **Python**: 3.11–3.13 only.

## Linting & Formatting

- **Ruff** is the primary linter/formatter (line-length 88). Tests directory is excluded from ruff.
- **mypy** for type checking (tests excluded). Most third-party modules have `ignore_missing_imports`.
- **isort** and **flake8** (with bandit, bugbear, builtins plugins) are also available.