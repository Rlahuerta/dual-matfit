# Phase 2: Module Upgrades

This phase covers functional upgrades to existing modules that were restructured in Phase 1.
All file paths use the new subpackage layout.

---

## 2.1 solvers/solution.py — Add `Root._accept_small_residual_root_result()`

Add a new method to the `Root` class that accepts root results where the residual is small but not strictly zero. This method wraps `_least_squares_solve()` and `_scipy_root_solve()` return values to recover from false-negative convergence failures.

### Source

`PaperHGOMatFit/dualmatfit/solvers/solution.py` lines 428-447

### Changes

- [ ] Add the `_accept_small_residual_root_result` method to `Root` class in `dualmatfit/solvers/solution.py`

**Insert after `_least_squares_solve` method and before `_newton_solve` method:**

```python
def _accept_small_residual_root_result(self, result: optimize.OptimizeResult) -> optimize.OptimizeResult:
    """Accept root results whose final residual already satisfies the solver tolerance."""
    if result.success or not hasattr(result, "fun"):
        return result

    residual = sanitize_array(np.atleast_1d(np.asarray(result.fun, dtype=float)))
    residual_norm = np.linalg.norm(residual)
    if np.isfinite(residual_norm) and residual_norm <= max(self.tol, 1e-8):
        result.success = True
        result.message = (
            f"{result.message} Residual norm {residual_norm:.3e} is within tolerance; "
            "accepting solution."
        )

    return result
```

- [ ] Update `_least_squares_solve` to return through `_accept_small_residual_root_result`:

Change:
```python
        return result

    def _scipy_root_solve(self, x0: np.ndarray) -> optimize.OptimizeResult:
```

To:
```python
        return self._accept_small_residual_root_result(result)

    def _scipy_root_solve(self, x0: np.ndarray) -> optimize.OptimizeResult:
```

- [ ] Update `_scipy_root_solve` to return through `_accept_small_residual_root_result`:

Change:
```python
        result = optimize.root(fun_flat, x0_flat, jac=jac_flat, method='hybr', tol=self.tol)
        return result
```

To:
```python
        result = optimize.root(fun_flat, x0_flat, jac=jac_flat, method='hybr', tol=self.tol)
        return self._accept_small_residual_root_result(result)
```

- [ ] Verify `sanitize_array` is imported at the top of `solution.py`. It should already be imported from `dualmatfit.utils.numeric` (or `dualmatfit.numeric_utils` in the current flat layout). Confirm the import exists and add if missing.

---

## 2.2 solvers/extension.py — Move `check_dsvars` and add `xi_ref` baseline

### 2.2.1 Move `check_dsvars()` from `utils.py` to `solvers/extension.py`

The `check_dsvars` function currently lives in `dualmatfit/utils.py`. Move it to `solvers/extension.py` as a module-level function.

- [ ] Add `check_dsvars` to `solvers/extension.py`. Insert the following function at module level (after imports, before `DesignVariablesMixin` class):

```python
def check_dsvars(
        var_form: VariationalFormulation,
        dsvars: Union[pd.DataFrame, pd.Series],
) -> Union[tuple[list[str], pd.DataFrame], tuple[list[str], pd.Series], None]:
    """
    Validate and extract design variables from a DataFrame or Series.

    Parameters
    ----------
    var_form : VariationalFormulation
        The variational formulation containing required material variables.
    dsvars : pd.DataFrame or pd.Series
        Design variables data indexed by variable names.

    Returns
    -------
    tuple[list[str], pd.DataFrame | pd.Series] or None
        Tuple of (variable_keys, filtered_dsvars) if successful, None otherwise.

    Raises
    ------
    TypeError
        If dsvars is not a DataFrame or Series.
    ValueError
        If required design variables are missing from dsvars.
    """
    if not isinstance(dsvars, (pd.DataFrame, pd.Series)):
        raise TypeError("dsvars must be a pandas DataFrame or Series.")

    missing_dsvars, dsvars_keys = [], []
    for var_i in var_form.dict_mat_vars.keys():
        if var_i not in dsvars.index:
            missing_dsvars.append(var_i)
        else:
            dsvars_keys.append(var_i)

    if 'lambda_' in missing_dsvars:
        missing_dsvars.remove('lambda_')

    if len(missing_dsvars) > 0:
        raise ValueError(
            f"Missing design variables in dsvars DataFrame: {missing_dsvars}. "
            f"Required variables: {list(var_form.mat_vars)}. "
            f"Available variables in dsvars index: {list(dsvars.index)}"
        )

    if isinstance(dsvars, pd.DataFrame):
        return dsvars_keys, dsvars.loc[dsvars_keys, :]

    elif isinstance(dsvars, pd.Series):
        return dsvars_keys, dsvars[dsvars_keys]

    return None
```

- [ ] Add `check_dsvars` to the `__all__` list in `solvers/extension.py`:

```python
__all__ = [
    'ExtensionSolution',
    'DesignVariablesMixin',
    'check_dsvars',
]
```

- [ ] Update the import in `extension_solution.py` (or `solvers/extension.py` in the new layout). Change:
```python
from dualmatfit.utils import check_dsvars
```
To:
```python
from dualmatfit.solvers.extension import check_dsvars
```

- [ ] Remove `check_dsvars` from `dualmatfit/utils.py` (and from its `__all__` list if present).

### 2.2.2 Add `self.xi_ref` baseline column

- [ ] In `DesignVariablesMixin._init_design_variables()`, add baseline column handling and change `xi_ref` assignment. After the `self.inp_mat_keys = mat_vars_keys` line, add:

```python
        # Ensure 'baseline' column exists; default to 'lower' if absent
        if isinstance(self.dsvars, pd.DataFrame) and "baseline" not in self.dsvars.columns:
            self.dsvars = self.dsvars.copy()
            self.dsvars["baseline"] = self.dsvars["lower"]
```

- [ ] Change the `xi_ref` assignment from:
```python
        self.xi_ref = self.dsvars["values"].values.astype(float)
```
To:
```python
        self.xi_ref = self.dsvars["baseline"].values.astype(float)
```

---

## 2.3 solvers/derivative.py — `_auto_generate_bounds` negative/zero handling

Add handling for negative and zero parameter values in `_auto_generate_bounds()`.

### Source

`PaperHGOMatFit/dualmatfit/solvers/derivative.py` lines ~50-80

### Changes

- [ ] In `_auto_generate_bounds()`, change the `else` branch (generic parameter case) to handle negative and zero values. Currently:

```python
        else:  # Generic positive parameter
            if xi_val <= 0:
                bounds.append([1e-6, 1e3])
            else:
                bounds.append([max(1e-6, xi_val * 0.001), xi_val * 1000])
```

Replace with:

```python
        else:  # Generic parameter
            if xi_val < 0:
                # For negative values, lower bound is more negative, upper bound is less negative
                bounds.append([xi_val * 1000, min(-1e-6, xi_val * 0.001)])
            elif xi_val == 0:
                # For zero, use symmetric bounds around zero
                bounds.append([-1e3, 1e3])
            else:
                # For positive values
                bounds.append([max(1e-6, xi_val * 0.001), xi_val * 1000])
```

This splits the `xi_val <= 0` case into two separate branches:
- **Negative values**: `[xi_val * 1000, min(-1e-6, xi_val * 0.001)]` — lower bound is more negative, upper bound is slightly below zero.
- **Zero values**: `[-1e3, 1e3]` — symmetric bounds around zero.
- **Positive values**: unchanged — `[max(1e-6, xi_val * 0.001), xi_val * 1000]`.

---

## 2.4 optimization/drivers.py — Add `_sanitize_lbfgsb_options()`

Add a new function that removes deprecated SciPy options (`disp`, `iprint`) from L-BFGS-B option dicts.

### Source

`PaperHGOMatFit/dualmatfit/optimization/drivers.py` lines 52-58

### Changes

- [ ] Add the `_sanitize_lbfgsb_options` function. Insert after `update_opt_parameters` and before `_run_slsqp`:

```python
def _sanitize_lbfgsb_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Remove L-BFGS-B options that are deprecated in SciPy."""
    sanitized_options = dict(options or {})
    for deprecated_key in ("disp", "iprint"):
        sanitized_options.pop(deprecated_key, None)
    return sanitized_options
```

- [ ] Ensure the import of `Optional`, `Dict`, `Any` from `typing` is present. Add to the imports at the top of the file:
```python
from typing import Optional, Dict, Any
```

- [ ] Call `_sanitize_lbfgsb_options` in `_run_lbfgs()`. After the line that updates `lbfgsb_options` from `solver_options`, add:

Change:
```python
    lbfgsb_options = {'maxls': 80, 'maxcor': 15, 'maxiter': miter}
    if solver_options and 'lbfgsb_options' in solver_options:
        lbfgsb_options.update(solver_options['lbfgsb_options'])
```

To:
```python
    lbfgsb_options = {'maxls': 80, 'maxcor': 15, 'maxiter': miter}
    if solver_options and 'lbfgsb_options' in solver_options:
        lbfgsb_options.update(solver_options['lbfgsb_options'])
    lbfgsb_options = _sanitize_lbfgsb_options(lbfgsb_options)
```

---

## 2.5 optimization/cost.py — `_build_regularization` signature change

Change `CostIntegrator._build_regularization(self)` to `CostIntegrator._build_regularization(self, vol_reg: bool, epsilon: float)` with explicit parameters instead of implicit access to `self._dvol` and `self._epsilon`.

### Source

`PaperHGOMatFit/dualmatfit/optimization/cost.py`

### Changes

- [ ] Change the method signature. In the `CostIntegrator` class, change:

```python
    def _build_regularization(self) -> CompositeRegularization:
        """Build the composite regularization strategy."""
```

To:

```python
    def _build_regularization(self, vol_reg: bool, epsilon: float) -> CompositeRegularization:
        """Build the composite regularization strategy."""
```

- [ ] Change the volume regularization conditional inside `_build_regularization`. Change:

```python
        # Add volume regularization if enabled
        if self._dvol and self._epsilon > 0:
            vol_reg = VolumeRegularization(
                cost_functions=self.cost_function,
                epsilon=self._epsilon,
                xi_bounds=self.xi_bounds,
                cache=self._cache,
            )
            regularization.add_strategy(vol_reg)
```

To:

```python
        # Add volume regularization if enabled
        if vol_reg and epsilon > 0.:
            vol_strategy = VolumeRegularization(
                cost_functions=self.cost_functions,
                epsilon=epsilon,
                xi_bounds=self.xi_bounds,
                cache=self._cache,
            )
            regularization.add_strategy(vol_strategy)
```

- [ ] Update the call site in `CostIntegrator.__init__`. Change:

```python
        self._regularization = self._build_regularization()
```

To:

```python
        epsilon = kwargs.get("epsilon", 0.) or 0.
        self._regularization = self._build_regularization(vol_reg, epsilon)
```

Note: `vol_reg` is already a parameter in `__init__`; `epsilon` needs to be extracted from `kwargs` before the call.

- [ ] Update any other call sites to `_build_regularization()` (e.g., in `update_variables` or other methods) to pass `vol_reg` and `epsilon` explicitly. Search for all occurrences with:

```bash
grep -rn "_build_regularization" dualmatfit/
```

---

## 2.6 optimization/loss.py — Return type change (`float` -> `np.ndarray`)

Change all loss functions (`cauchy_fval`, `huber_fval`, `logcosh_fval`, `ln_fval`) to return `np.ndarray` instead of `float`. Import `safe_divide` from `dualmatfit.utils.numeric`.

### Source

`PaperHGOMatFit/dualmatfit/optimization/loss.py`

### Changes

- [ ] Change `cauchy_fval` return type and implementation. Change:

```python
def cauchy_fval(residuum: np.ndarray, c: float, **kwargs) -> float:
    """
    Compute the Cauchy loss function value.

    Args:
        residuum (np.ndarray): Residual vector.
        c (float): Scaling parameter.

    Returns:
        float: Function value.
    """
    residuum_in = _ensure_2d_residuum(residuum)

    list_fvals = []
    for i in range(residuum_in.shape[0]):
        residuum2_i = (residuum_in[i, :] / c) ** 2
        fval_i = 0.5 * c ** 2 * np.sum(np.log1p(residuum2_i))
        list_fvals.append(fval_i)

    return sum(list_fvals)
```

To:

```python
def cauchy_fval(residuum: np.ndarray, c: float, **kwargs) -> np.ndarray:
    """
    Compute the Cauchy loss function value.

    Args:
        residuum (np.ndarray): Residual vector.
        c (float): Scaling parameter.

    Returns:
        float: Function value.
    """
    residuum_in = _ensure_2d_residuum(residuum)

    np_residuum2 = (residuum_in / c) ** 2
    np_fval = 0.5 * c ** 2 * np.log1p(np_residuum2)

    return np_fval.sum(axis=0)
```

- [ ] Change `huber_fval` return type and implementation. Change:

```python
def huber_fval(residuum: np.ndarray, delta: float = 1., **kwargs) -> float:
    ...
    return sum(list_fvals)
```

To:

```python
def huber_fval(residuum: np.ndarray, delta: float = 1., **kwargs) -> np.ndarray:
    """
    Compute the Huber loss function value.

    Args:
        residuum (np.ndarray): Residual vector.
        delta (float): Threshold parameter.

    Returns:
        float: Function value.
    """
    residuum_in = _ensure_2d_residuum(residuum)

    list_fvals = []
    for i in range(residuum_in.shape[0]):
        abs_res_i = np.abs(residuum_in[i, :])
        resid2_i = 0.5 * residuum_in[i, :] ** 2
        linear_i = delta * (abs_res_i - 0.5 * delta)
        fval_i = np.where(abs_res_i <= delta, resid2_i, linear_i)
        list_fvals.append(np.sum(fval_i))

    return np.array(list_fvals, dtype=float)
```

- [ ] Change `logcosh_fval` return type and implementation. Change:

```python
def logcosh_fval(residuum: np.ndarray, **kwargs) -> float:
    ...
    return sum(list_fvals)
```

To:

```python
def logcosh_fval(residuum: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute the Log-Cosh function value.

    Args:
        residuum (np.ndarray): Residual vector, shape (n_rows, n_residuals) or (n_residuals,).

    Returns:
        np.ndarray: Shape (n_rows,) — sum of log-cosh per row.
    """
    residuum_in = _ensure_2d_residuum(residuum)

    return np.array([np.sum(np.log(np.cosh(residuum_in[i, :])))
                     for i in range(residuum_in.shape[0])])
```

- [ ] Change `ln_fval` return type and implementation. Change:

```python
def ln_fval(residuum: np.ndarray, **kwargs) -> float:
    ...
    return sum(list_fvals)
```

To:

```python
def ln_fval(residuum: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute the logarithm of the sum of squares function value.

    Args:
        residuum (np.ndarray): Residual vector, shape (n_rows, n_residuals) or (n_residuals,).

    Returns:
        np.ndarray: Shape (n_rows,) — one value per row.
    """
    residuum_in = _ensure_2d_residuum(residuum)

    list_fvals = []
    for i in range(residuum_in.shape[0]):
        residuum2_i = np.dot(residuum_in[i, :], residuum_in[i, :])
        fval_i = np.log1p(residuum2_i)
        list_fvals.append(fval_i)

    return np.array(list_fvals)
```

- [ ] Also update `lsq_fval` return type from `float` to `np.ndarray` for consistency. Change:

```python
def lsq_fval(residuum: np.ndarray, **kwargs) -> float:
```

To:

```python
def lsq_fval(residuum: np.ndarray, **kwargs) -> np.ndarray:
```

(The implementation already returns `np.linalg.norm(...)` which returns `np.ndarray`.)

- [ ] Add `safe_divide` import to the loss module. Add to imports:

```python
from dualmatfit.utils.numeric import safe_divide
```

(In the current flat layout, this would be `from dualmatfit.numeric_utils import safe_divide`.)

- [ ] Update `lsq_dfval` to use `safe_divide` instead of manual division. The PaperHGOMatFit version already uses it. Confirm the current code uses it; if not, change:

```python
    # Normalize by the norm of each residual vector (guard against zero norms)
    return safe_divide(dfvals, norms[:, np.newaxis], default=0.0)
```

(This should already be present in the current code.)

---

## 2.7 optimization/core.py — Add `ks` import

Add `from dualmatfit.utils.ks import min_ks, max_ks` to `optimization/core.py`.

This requires creating `utils/ks.py` with `min_ks` and `max_ks` KS aggregation functions.

### Source

The `min_ks` and `max_ks` functions are referenced in the old `dualmatfit/optimization.py` (line 19: `from dualmatfit.ks import min_ks, max_ks`) but the `ks.py` module does not yet exist. The `ConstraintAggregation` class in `optimization.py` uses these functions.

### Changes

- [ ] Create `dualmatfit/utils/ks.py` (or `dualmatfit/ks.py` in the current flat layout) with `min_ks` and `max_ks` KS aggregation functions:

```python
# -*- coding: utf-8 -*-
"""
Kreisselmeier-Steinhauser (KS) aggregation functions for constraint handling.

Provides smooth approximations of min/max operations for gradient-based
optimization of constrained problems.
"""
import numpy as np

__all__ = [
    'min_ks',
    'max_ks',
]


def min_ks(g: np.ndarray, rho: float = 10.0) -> float:
    """
    Kreisselmeier-Steinhauser (KS) smooth approximation of the minimum.

    Parameters
    ----------
    g : np.ndarray
        Array of constraint values.
    rho : float, optional
        Aggregation parameter (default 10.0). Higher values give tighter
        approximation to the true minimum.

    Returns
    -------
    float
        KS approximation of min(g).
    """
    g_min = np.min(g)
    return g_min - (1.0 / rho) * np.log(np.sum(np.exp(-rho * (g - g_min))))


def max_ks(g: np.ndarray, rho: float = 10.0) -> float:
    """
    Kreisselmeier-Steinhauser (KS) smooth approximation of the maximum.

    Parameters
    ----------
    g : np.ndarray
        Array of constraint values.
    rho : float, optional
        Aggregation parameter (default 10.0). Higher values give tighter
        approximation to the true maximum.

    Returns
    -------
    float
        KS approximation of max(g).
    """
    g_max = np.max(g)
    return g_max + (1.0 / rho) * np.log(np.sum(np.exp(rho * (g - g_max))))
```

- [ ] In `optimization/core.py`, add the import:

```python
from dualmatfit.utils.ks import min_ks, max_ks
```

(In the current flat layout, this would be `from dualmatfit.ks import min_ks, max_ks`.)

- [ ] Update `utils/__init__.py` (or `dualmatfit/__init__.py`) to export `min_ks` and `max_ks` if desired.

---

## 2.8 utils/path_manager.py — New methods

Add five new methods to `PathManager`.

### Source

`PaperHGOMatFit/dualmatfit/utils/path_manager.py` lines 332-490

### Changes

- [ ] Add `get_rat_solution_dir` static method:

```python
    @staticmethod
    def get_rat_solution_dir(results_dir: PathLike, rat_id: str) -> Path:
        """
        Get solution directory for a specific rat/specimen.

        Parameters
        ----------
        results_dir : PathLike
            Base results directory
        rat_id : str
            Rat/specimen identifier

        Returns
        -------
        Path
            Path to rat-specific solution directory

        Notes
        -----
        Handles rat_id with or without leading slash.
        """
        results_dir = _normalize_path(results_dir)
        clean_id = rat_id.lstrip('/')
        return results_dir / clean_id
```

- [ ] Add `get_section_dir` static method:

```python
    @staticmethod
    def get_section_dir(rat_dir: PathLike, section: str) -> Path:
        """
        Get directory for a specific section within rat results.

        Parameters
        ----------
        rat_dir : PathLike
            Rat solution directory
        section : str
            Section identifier (e.g., 'Ar', 'Tr', 'Ab')

        Returns
        -------
        Path
            Path to section-specific directory
        """
        rat_dir = _normalize_path(rat_dir)
        return rat_dir / section
```

- [ ] Add `validate_file_exists` instance method:

```python
    def validate_file_exists(self, file_path: PathLike) -> Path:
        """
        Validate that a file exists and return its absolute path.

        Parameters
        ----------
        file_path : PathLike
            Path to file to validate (str or Path)

        Returns
        -------
        Path
            Absolute path to the validated file

        Raises
        ------
        FileNotFoundError
            If file does not exist
        """
        file_path = _normalize_path(file_path)
        abs_path = self._resolve_path(file_path)
        if not abs_path.is_file():
            raise FileNotFoundError(f"File not found: {abs_path}")
        return abs_path
```

- [ ] Add `remove_file` instance method:

```python
    def remove_file(self, file_path: PathLike) -> bool:
        """
        Remove a file if it exists.

        Parameters
        ----------
        file_path : PathLike
            Path to file to remove (str or Path)

        Returns
        -------
        bool
            True if file was removed, False if it didn't exist
        """
        file_path = _normalize_path(file_path)
        abs_path = self._resolve_path(file_path)

        if abs_path.exists():
            try:
                abs_path.unlink()
                logger.info(f"Successfully removed: {abs_path}")
                return True
            except OSError as e:
                logger.error(f"Error removing {abs_path}: {e}")
                return False
        else:
            logger.info(f"File does not exist: {abs_path}")
            return False
```

- [ ] Add `get_output_path` instance method:

```python
    def get_output_path(self, filename: str, subdir: Optional[str] = None) -> Path:
        """
        Get an output path for saving files.

        Constructs a path relative to results directory, optionally
        within a subdirectory. Ensures parent directory exists.

        Parameters
        ----------
        filename : str
            Name of the output file
        subdir : str, optional
            Subdirectory within results directory

        Returns
        -------
        Path
            Absolute path for the output file
        """
        if subdir:
            output_dir = self.config.results_base_dir / subdir
        else:
            output_dir = self.config.results_base_dir

        abs_dir = self.ensure_dir(output_dir)
        return abs_dir / filename
```

- [ ] Ensure `_normalize_path` helper function exists in the module (it is used by the new methods). The PaperHGOMatFit version defines it as a module-level function. Check if it exists in the current `path_manager.py` and add if missing:

```python
def _normalize_path(path: PathLike) -> Path:
    """Normalize a path-like object to a Path instance."""
    if isinstance(path, str):
        return Path(path)
    return path
```

---

## 2.9 utils/latex_post.py — Add `sympy2latex()`

Add a `sympy2latex()` function for SymPy expression to LaTeX string conversion and PDF compilation.

### Source

`PaperHGOMatFit/dualmatfit/utils/latex_post.py` lines 414-465

### Changes

- [ ] Add `sympy2latex` to the module's `__all__` list.

- [ ] Add the `sympy2latex` function to `utils/latex_post.py`:

```python
def sympy2latex(latex_code: Union[str, list], fname: str, wpath: Union[str, Path] = "") -> Path:
    """
    Convert LaTeX code to PDF using pdflatex.

    Parameters
    ----------
    latex_code : str or list
        LaTeX code to include in the document body.
        If list, elements are concatenated.
    fname : str
        Output filename (should end with .tex)
    wpath : str or Path, optional
        Working directory path. Defaults to current directory.

    Returns
    -------
    Path
        Path to the output LaTeX file
    """
    latex_document = '\\documentclass{article}\n'
    latex_document += '\\usepackage{amsmath, amssymb}\n'
    latex_document += '\\usepackage{breqn}\n'
    latex_document += '\\usepackage{graphicx}\n'
    latex_document += '\\begin{document}\n'

    if isinstance(latex_code, str):
        latex_document += latex_code
    elif isinstance(latex_code, list):
        for latex_code_i in latex_code:
            latex_document += latex_code_i
    else:
        raise NotImplementedError("latex_code must be str or list")

    latex_document += '\\end{document}\n'

    work_path = Path(wpath) if wpath else Path.cwd()
    output_path = work_path / fname

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as file:
        file.write(latex_document)

    result = subprocess.run(
        ['pdflatex', output_path.name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=output_path.parent
    )

    if result.returncode != 0:
        logger.info("An error occurred during compilation:")
        logger.info(result.stderr.decode())
    else:
        logger.info("Compilation successful! PDF created.")

    return output_path
```

- [ ] Ensure `subprocess` is imported at the top of the file. Add if missing:
```python
import subprocess
```

- [ ] Ensure `Path` is imported from `pathlib` and `Union` from `typing`. Add if missing:
```python
from pathlib import Path
from typing import Union
```

---

## 2.10 plotting/ — Split `PlotSolution2D` into `plotting/solution_visuals.py`

Extract the `PlotSolution2D` class from `plot.py` into `plotting/solution_visuals.py` and update `plotting/__init__.py` to re-export it.

### Source

`PaperHGOMatFit/dualmatfit/plotting/solution_visuals.py`

### Changes

- [ ] Create `dualmatfit/plotting/solution_visuals.py` with the `PlotSolution2D` class. The class currently lives in `dualmatfit/plot.py` at line 860. Extract it into the new file:

```python
# -*- coding: utf-8 -*-
"""
Solution visualization functions.

This module provides the PlotSolution2D class for visualizing
2D solution results including strain energy, stress components,
and force plots.
"""
import matplotlib.pyplot as plt

from scipy.optimize import OptimizeResult

__all__ = [
    'PlotSolution2D',
]

class PlotSolution2D:
    def __init__(self):
        """
        Initialize the PlotSolution2D class.

        Parameters:
        - title (str): The title of the plot.
        - ltype (dict): Dictionary mapping keys to line styles.
        - post_equations (dict, optional): Dictionary of equations to display.
        """

        # Labels for the plot
        self.ltx_energy = r"$\psi$"
        # ... (copy the full __init__ from the current plot.py class)
```

The full class implementation (all methods: `__init__`, `_create_2d_plot`, `_create_force_plot`, `components_plot`, `force_plot`, `full_plot`) should be copied verbatim from the current `dualmatfit/plot.py` lines 860-1062.

- [ ] Update `dualmatfit/plotting/__init__.py` to import and re-export `PlotSolution2D`:

```python
from dualmatfit.plotting.solution_visuals import (
    PlotSolution2D,
)
```

And add `'PlotSolution2D'` to the `__all__` list.

- [ ] Remove the `PlotSolution2D` class from `dualmatfit/plot.py` (lines 860-1062).

- [ ] Update any imports that reference `PlotSolution2D` from `dualmatfit.plot` to instead import from `dualmatfit.plotting.solution_visuals` or `dualmatfit.plotting`.

- [ ] Search for all `PlotSolution2D` usages across the codebase:
```bash
grep -rn "PlotSolution2D" dualmatfit/
```

---

## Execution Order

The recommended order for implementing these changes is:

1. **2.7** — Create `utils/ks.py` (no dependencies on other changes)
2. **2.2** — Move `check_dsvars` and add `xi_ref` (solvability dependency)
3. **2.1** — Add `_accept_small_residual_root_result` (depends on `sanitize_array` import)
4. **2.3** — Update `_auto_generate_bounds` (standalone change)
5. **2.4** — Add `_sanitize_lbfgsb_options` (standalone change)
6. **2.5** — Change `_build_regularization` signature (requires updating all call sites)
7. **2.6** — Change loss function return types (affects downstream callers)
8. **2.8** — Add `PathManager` methods (standalone addition)
9. **2.9** — Add `sympy2latex` (standalone addition)
10. **2.10** — Extract `PlotSolution2D` (visual module reorganization)

---

## Testing Checklist

After all changes are implemented:

- [ ] Run existing unit tests to confirm no regressions
- [ ] Verify `Root.solve()` correctly accepts small-residual results
- [ ] Verify `_auto_generate_bounds` produces correct bounds for negative and zero parameters
- [ ] Verify `_sanitize_lbfgsb_options` strips `disp` and `iprint` from option dicts
- [ ] Verify `_build_regularization(vol_reg=True, epsilon=0.1)` adds volume regularization
- [ ] Verify loss functions return `np.ndarray` and callers handle the type correctly
- [ ] Verify `min_ks` and `max_ks` produce correct KS aggregation values
- [ ] Verify new `PathManager` methods work as expected
- [ ] Verify `sympy2latex` generates valid LaTeX and compiles to PDF
- [ ] Verify `PlotSolution2D` is importable from `dualmatfit.plotting`