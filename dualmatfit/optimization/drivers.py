# -*- coding: utf-8 -*-
"""
Optimization driver functions for material fitting.

This module provides high-level interfaces to various optimization algorithms
(SLSQP, L-BFGS-B, TNC, IPOPT, Differential Evolution, SHGO) for fitting
material parameters.
"""
import gc
import numpy as np
import pandas as pd

from typing import Dict, Any, Union, Optional
from scipy.optimize import minimize, basinhopping, differential_evolution, shgo, OptimizeResult

from dualmatfit.optimization.basinhopping import ipopt_basinhopping
from dualmatfit.optimization.cost import CostFunction, CostIntegrator

from dualmatfit.utils.logging_config import get_logger
logger = get_logger('drivers')

__all__ = [
    'opt_solvers',
    'update_opt_parameters',
]


# Set precision to 4 decimal places, suppress scientific notation
np.set_printoptions(precision=4, suppress=True)

_IpyoptMinimizer = None
_IPYOPT_IMPORT_ERROR: Optional[ImportError] = None


def _get_ipyopt_minimizer_class():
    """Import IPOPT lazily so non-IPOPT installs can still use other solvers."""
    global _IpyoptMinimizer, _IPYOPT_IMPORT_ERROR

    if _IpyoptMinimizer is None and _IPYOPT_IMPORT_ERROR is None:
        try:
            from dualmatfit.optimization.ipopt import IpyoptMinimizer as imported_minimizer
        except ImportError as exc:
            _IPYOPT_IMPORT_ERROR = exc
        else:
            _IpyoptMinimizer = imported_minimizer

    if _IpyoptMinimizer is None:
        raise ImportError(
            "IPOPT support requires the optional ipopt dependencies. "
            "Install them with `pip install -e .[ipopt]` or use a different "
            "optimizer such as 'L-BFGS-B'."
        ) from _IPYOPT_IMPORT_ERROR

    return _IpyoptMinimizer


def update_opt_parameters(local_res: OptimizeResult, global_res: OptimizeResult):
    """
    Updates attributes of a global OptimizeResult object with values from a local one.

    This function iterates through a predefined list of common optimization result attributes
    (fun, x, status, success, nfev, njev, nhev). If an attribute exists in the
    `local_res` object, its value is copied to the corresponding attribute in the
    `global_res` object. This is typically used to ensure the main result object
    reflects the outcome of the latest or most successful optimization step.

    Args:
        local_res (OptimizeResult): The optimization result object from which to copy attributes.
        global_res (OptimizeResult): The optimization result object to be updated.
    """
    for attr in ["fun", "x", "status", "success", "nfev", "njev", "nhev", "message", "nit"]:
        if hasattr(local_res, attr):
            setattr(global_res, attr, getattr(local_res, attr))


def _sanitize_lbfgsb_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Remove L-BFGS-B options that are deprecated in SciPy."""
    sanitized_options = dict(options or {})
    for deprecated_key in ("disp", "iprint"):
        sanitized_options.pop(deprecated_key, None)
    return sanitized_options


def _run_slsqp(cost_function, np_ini_dsvars, list_dsvars_bounds, glb, giter, solver_options, np_xi_lwr, np_xi_upp,
               np_xi_dt):
    if glb:
        default_bh_options = {'disp': True, 'niter': giter}
        if solver_options and 'basinhopping_options' in solver_options:
            default_bh_options.update(solver_options['basinhopping_options'])

        default_minimize_options = {'disp': True}
        if solver_options and 'minimize_options' in solver_options:
            default_minimize_options.update(solver_options['minimize_options'])

        min_kwargs = {"method": "SLSQP", "jac": cost_function.derivative,
                      "bounds": list_dsvars_bounds, "options": default_minimize_options}
        opt_res = basinhopping(cost_function, np_ini_dsvars, minimizer_kwargs=min_kwargs, **default_bh_options)
        opt_method = 'Basin Hopping [slsqp]'
    else:
        list_fval, list_res = [], []
        perturbation_factor = solver_options.get('perturbation_factor', 0.05) if solver_options else 0.05
        minimize_options_slsqp = {'disp': True}
        if solver_options and 'minimize_options' in solver_options:
            minimize_options_slsqp.update(solver_options['minimize_options'])

        current_ini_dsvars = np_ini_dsvars.copy()
        nvars = len(np_ini_dsvars)
        for i in range(giter):
            if i > 0:
                np_rand_i = np.random.uniform(-1., 1., nvars)
                np_pert_i = perturbation_factor * np_rand_i
                perturbed_vars = current_ini_dsvars + np_pert_i * np_xi_dt
                np_chk_i = (np_xi_lwr <= perturbed_vars) & (perturbed_vars < np_xi_upp)
                current_ini_dsvars[np_chk_i] = perturbed_vars[np_chk_i]

            opt_res_i = minimize(cost_function, current_ini_dsvars, method='SLSQP', jac=cost_function.derivative,
                                   bounds=list_dsvars_bounds, options=minimize_options_slsqp)
            list_fval.append(opt_res_i.fun)
            list_res.append(opt_res_i)
            gc.collect()
            current_ini_dsvars = np.array(opt_res_i.x, dtype=float)

        opt_res = list_res[np.argmin(list_fval)]
        opt_method = 'Random Uniform trials [slsqp]'
    return opt_res, opt_method


def _run_mpowell(cost_function, np_ini_dsvars, solver_options):
    powell_options = {'disp': True}
    if solver_options and 'powell_options' in solver_options:
        powell_options.update(solver_options['powell_options'])
    opt_res = minimize(cost_function, np_ini_dsvars, method='Powell', options=powell_options)
    opt_method = 'Single run [mpowell]'
    return opt_res, opt_method


def _run_lbfgsb(cost_function, np_ini_dsvars, list_dsvars_bounds, glb, giter, miter, solver_options, **kwargs):
    lbfgsb_options = {'maxls': 80, 'maxcor': 15, 'maxiter': miter}
    if solver_options and 'lbfgsb_options' in solver_options:
        lbfgsb_options.update(solver_options['lbfgsb_options'])
    lbfgsb_options = _sanitize_lbfgsb_options(lbfgsb_options)

    def print_callback(xk):
        logger.info(f"Current parameters: {xk}")

    if glb:
        bh_options = {'niter': giter, 'stepsize': 0.2, 'niter_success': 10}
        if solver_options and 'basinhopping_options' in solver_options:
            bh_options.update(solver_options['basinhopping_options'])

        min_kwargs = {"method": "L-BFGS-B", "jac": cost_function.derivative,
                          "bounds": list_dsvars_bounds, "options": lbfgsb_options}
        opt_res = basinhopping(cost_function, np_ini_dsvars, minimizer_kwargs=min_kwargs, **bh_options)
        opt_method = 'Basin Hopping [lbfgsb]'
    else:
        current_ini_dsvars = np_ini_dsvars.copy()
        opt_res = None
        for i in range(giter):
            opt_res = minimize(cost_function, current_ini_dsvars, method='L-BFGS-B', jac=cost_function.derivative,
                                 bounds=list_dsvars_bounds, options=lbfgsb_options, callback=print_callback)
            gc.collect()
            current_ini_dsvars = np.array(opt_res.x, dtype=float)
        opt_method = 'Warming Start [lbfgsb]' if giter > 1 else 'Single run [lbfgsb]'
    return opt_res, opt_method


def _run_tnc(cost_function, np_ini_dsvars, list_dsvars_bounds, glb, giter, miter, solver_options, np_xi_lwr,
             np_xi_upp, np_xi_dt, **kwargs):
    if glb:
        default_tnc_minimizer_options = {'maxfun': 200, 'eta': 0.2, 'disp': False}
        if solver_options and 'tnc_minimizer_options' in solver_options:
            default_tnc_minimizer_options.update(solver_options['tnc_minimizer_options'])

        default_bh_options = {'niter': giter, 'disp': True, 'T': 10.0}
        if solver_options and 'basinhopping_options' in solver_options:
            default_bh_options.update(solver_options['basinhopping_options'])

        min_kwargs = {"method": "TNC", "jac": cost_function.derivative,
                          "bounds": list_dsvars_bounds, "options": default_tnc_minimizer_options}
        opt_res = basinhopping(cost_function, np_ini_dsvars, minimizer_kwargs=min_kwargs, **default_bh_options)
        opt_method = 'Basin Hopping [tnc]'
    else:
        default_tnc_options = {'disp': True, 'maxfun': 1000, 'eta': 0.2, 'maxCGit': miter if miter else -1}
        if solver_options and 'tnc_options' in solver_options:
            default_tnc_options.update(solver_options['tnc_options'])

        perturbation_factor = solver_options.get('perturbation_factor', 0.01) if solver_options else 0.01
        list_fval, list_res = [], []
        current_dsvars = np_ini_dsvars.copy()
        nvars = len(np_ini_dsvars)
        for i in range(giter):
            if i > 0:
                np_pert_i = perturbation_factor * np.random.uniform(-1., 1., nvars)
                perturbed_vars = current_dsvars + np_pert_i * np_xi_dt
                np_chk_i = (np_xi_lwr <= perturbed_vars) & (perturbed_vars < np_xi_upp)
                current_dsvars[np_chk_i] = perturbed_vars[np_chk_i]

            opt_res_i = minimize(cost_function, current_dsvars, method='TNC', jac=cost_function.derivative,
                                 bounds=list_dsvars_bounds, options=default_tnc_options)
            list_fval.append(opt_res_i.fun)
            list_res.append(opt_res_i)
            current_dsvars = np.array(opt_res_i.x, dtype=float)

        opt_res = list_res[np.argmin(list_fval)]
        opt_method = 'Random Uniform trials [tnc]'
    return opt_res, opt_method


def _run_diffevol(cost_function, list_dsvars_bounds, miter, seed, solver_options):
    de_options = {'strategy': 'randtobest1bin', 'maxiter': miter, 'popsize': 15, 'tol': 1.e-4,
                  'mutation': (0.5, 1), 'recombination': 0.7, 'seed': seed, 'disp': True, 'polish': True,
                  'updating': 'immediate', 'workers': 1}
    if solver_options:
        de_options.update(solver_options)
    opt_res = differential_evolution(cost_function, list_dsvars_bounds, **de_options)
    opt_method = f'Differential Evolution [{de_options["strategy"]}]'
    return opt_res, opt_method


def _run_shgo(cost_function, list_dsvars_bounds, miter, solver_options):
    default_shgo_options = {'disp': True, 'infty_constraints': False, 'minimize_every_iter': True, 'iters': miter}
    if solver_options and 'shgo_main_options' in solver_options:
        default_shgo_options.update(solver_options['shgo_main_options'])

    default_shgo_minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': cost_function.derivative}
    if solver_options and 'shgo_minimizer_kwargs' in solver_options:
        default_shgo_minimizer_kwargs.update(solver_options['shgo_minimizer_kwargs'])

    sampling_method = solver_options.get('shgo_sampling_method', 'sobol') if solver_options else 'sobol'
    opt_res = shgo(cost_function, list_dsvars_bounds, sampling_method=sampling_method,
                   minimizer_kwargs=default_shgo_minimizer_kwargs, options=default_shgo_options)
    opt_method = f'SHGO [{sampling_method}]'
    return opt_res, opt_method


def _run_ipopt(cost_function,
               np_ini_dsvars,
               np_xi_lwr,
               np_xi_upp,
               miter,
               glb,
               giter,
               seed,
               constraint,
               solver_options,
               **kwargs,
               ):

    bh_step = kwargs.get("bh_step", "random_displacement")
    default_ipyopt_kwargs_set = {
        'max_iter': int(miter), 'expect_infeasible_problem': 'yes', 'limited_memory_max_history': 25,
        'least_square_init_duals': 'yes', 'alpha_for_y': 'min', 'required_infeasibility_reduction': 0.25,
        'bound_relax_factor': 1.e-7, 'constr_viol_tol': 1.e-4, 'warm_start_init_point': "yes",
        'warm_start_bound_push': 1.e-8, 'warm_start_slack_bound_push': 1.e-8,
        'warm_start_mult_bound_push': 1.e-8, 'print_level': 5,
    }
    if solver_options and 'ipyopt_options' in solver_options:
        default_ipyopt_kwargs_set.update(solver_options['ipyopt_options'])

    default_basinhopping_kwargs = {
        'niter': giter, 'T': 10., 'stepsize': 0.5, 'interval': 5, 'step_taking_method': bh_step,
        'pareto_alpha': 2., 'disp': True, 'rng': seed,
    }
    if solver_options and 'basinhopping_options' in solver_options:
        default_basinhopping_kwargs.update(solver_options['basinhopping_options'])

    if glb:
        global_ipyopt_mods = {'bound_relax_factor': 1.e-6, 'constr_viol_tol': 1.e-3}
        if solver_options and 'ipyopt_glb_options' in solver_options:
            global_ipyopt_mods.update(solver_options['ipyopt_glb_options'])
        default_ipyopt_kwargs_set.update(global_ipyopt_mods)

        global_bh_mods = {'T': 100., 'stepsize': 0.8}
        if solver_options and 'basinhopping_glb_options' in solver_options:
            global_bh_mods.update(solver_options['basinhopping_glb_options'])
        default_basinhopping_kwargs.update(global_bh_mods)

    ipopt_min_kwargs = {'ipyopt_options': default_ipyopt_kwargs_set}
    if constraint is not None:
        ipopt_min_cst_kwargs = {
            'ncon': constraint.cst_num, 'g_l': constraint.bounds['lower'], 'g_u': constraint.bounds['upper'],
            'eval_jac_g_sparsity_indices': constraint.sparsity_indices(), 'eval_g': constraint,
            'eval_jac_g': constraint.derivative,
        }
        ipopt_min_kwargs.update(ipopt_min_cst_kwargs)

    ipyopt_minimizer_cls = _get_ipyopt_minimizer_class()
    min_ipopt = ipyopt_minimizer_cls(np_xi_lwr, np_xi_upp, cost_function, **ipopt_min_kwargs)

    if default_basinhopping_kwargs['niter'] == 0:
        opt_res = min_ipopt(np_ini_dsvars)
        opt_method = 'Single run [ipopt]'

    else:
        opt_res = ipopt_basinhopping(min_ipopt,
                                     np_ini_dsvars,
                                     x_l = np_xi_lwr,
                                     x_u = np_xi_upp,
                                     **default_basinhopping_kwargs,
                                     )
        opt_method = 'Basin Hopping [ipopt]'
    return opt_res, opt_method


def opt_solvers(otype: str,
                cost_fun: Union[CostFunction, CostIntegrator],
                dsvars: pd.DataFrame,
                miter: int = 200,
                glb: bool = False,
                giter: int = 1,
                seed: int = 0,
                empty: bool = False,
                solver_options: Optional[Dict[str, Any]] = None,
                **kwargs: Any,
                ) -> OptimizeResult:
    """
    Optimizes a given cost function using various local and global optimization algorithms.

    Args:
        otype (str): The type of optimizer to use (e.g., 'slsqp', 'lbfgsb', 'ipopt').
        cost_fun (Union[CostFunction, CostIntegrator]): The cost function object to be minimized.
            It should have methods for evaluation and derivative calculation.
        dsvars (pd.DataFrame): DataFrame containing design variable information, including
            initial values, bounds, and which variables are active for optimization.
        miter (int, optional): Maximum number of iterations for local optimizers or specific
            global optimizers like Differential Evolution and SHGO. Defaults to 200.
        glb (bool, optional): If True, attempts to use a global optimization strategy,
            often involving basinhopping with a local optimizer. Defaults to False.
        giter (int, optional): Number of global iterations, typically for basinhopping
            or random restart strategies. Defaults to 1.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        empty (bool, optional): If True, skips optimization and returns an initial result
            structure. Defaults to False.
        solver_options (Optional[Dict[str, Any]], optional): Dictionary to pass
            optimizer-specific options. The structure depends on `otype`.
            Examples:
            - SLSQP: `{'perturbation_factor': 0.01, 'minimize_options': {'disp': False}}`
            - L-BFGS-B: `{'lbfgsb_options': {'maxls': 60}, 'basinhopping_options': {'stepsize': 0.1}}`
            - IPOPT:
                - To override general IPOPT settings for `IpyoptMinimizer` and `ipopt_basinhopping`:
                  `{'ipyopt_options': {'max_iter': 500, 'tol': 1e-7}, 'basinhopping_options': {'T': 5.0}}`
                - To specifically override options when `glb=True` (these apply *on top* of
                  general defaults and user-supplied general options):
                  `{'ipyopt_glb_options': {'constr_viol_tol': 1e-5}, 'basinhopping_glb_options': {'stepsize': 0.75}}`
                These can be combined. The layering for IPOPT options is:
                1. Internal hardcoded defaults.
                2. User-supplied general options (`ipyopt_options`, `basinhopping_options`).
                3. If `glb=True`, internal hardcoded global modifications are applied.
                4. If `glb=True`, user-supplied global-specific options (`ipyopt_glb_options`,
                   `basinhopping_glb_options`) are applied.
            Defaults to None, in which case default options are used.
        **kwargs (Any): Additional keyword arguments, primarily `bh_step` for basinhopping
            step-taking method.

    Returns:
        OptimizeResult: An object (typically from `scipy.optimize.OptimizeResult` or a
            custom extension) containing the optimization results, including the optimal
            design variables, function value, and other solver-specific information.

    Raises:
        NotImplementedError: If the specified `cost_fun` type is not supported.

    """
    np.random.seed(seed)

    run_dsvars = dsvars[dsvars["variable"] == True]
    np_ini_dsvars = run_dsvars["values"].values
    list_dsvars_bounds = run_dsvars[["lower", "upper"]].values.tolist()

    if not isinstance(cost_fun, (CostFunction, CostIntegrator)):
        raise NotImplementedError("Cost function type not implemented or not supported!")

    np_xi_lwr = run_dsvars["lower"].values.astype(float)
    np_xi_upp = run_dsvars["upper"].values.astype(float)
    np_xi_dt = np_xi_upp - np_xi_lwr

    optimal_res = OptimizeResult(x0=np_ini_dsvars,
                                 bounds={"lower": np_xi_lwr, "upper": np_xi_upp},
                                 global_iter=giter,
                                 local_iter=miter,
                                 seed=seed,
                                 series=pd.Series([]),
                                 method="",
                                 )

    if empty:
        optimal_res.method = otype
        optimal_res.x = np_ini_dsvars
        optimal_res.fun = None
        optimal_res.series = pd.Series(data=np_ini_dsvars, index=run_dsvars["values"].keys())
        return optimal_res

    opt_res = None
    opt_method = ""
    otype_lower = otype.lower()

    if otype_lower == 'slsqp':
        opt_res, opt_method = _run_slsqp(cost_fun, np_ini_dsvars, list_dsvars_bounds, glb, giter, solver_options,
                                         np_xi_lwr, np_xi_upp, np_xi_dt)
    elif otype_lower == 'mpowell':
        opt_res, opt_method = _run_mpowell(cost_fun, np_ini_dsvars, solver_options)
    elif otype_lower == 'lbfgsb':
        opt_res, opt_method = _run_lbfgsb(cost_fun, np_ini_dsvars, list_dsvars_bounds, glb, giter, miter,
                                          solver_options, **kwargs)
    elif otype_lower == 'tnc':
        opt_res, opt_method = _run_tnc(cost_fun, np_ini_dsvars, list_dsvars_bounds, glb, giter, miter,
                                       solver_options, np_xi_lwr, np_xi_upp, np_xi_dt, **kwargs)
    elif otype_lower == 'diffevol':
        opt_res, opt_method = _run_diffevol(cost_fun, list_dsvars_bounds, miter, seed, solver_options)
    elif otype_lower == 'shgo':
        opt_res, opt_method = _run_shgo(cost_fun, list_dsvars_bounds, miter, solver_options)
    elif otype_lower == 'ipopt':
        opt_res, opt_method = _run_ipopt(cost_fun, np_ini_dsvars, np_xi_lwr, np_xi_upp, miter, glb, giter, seed,
                                         None, solver_options, **kwargs)

    # Validate optimizer type - raise error for unknown optimizer
    if opt_res is None and not empty:
        supported_optimizers = ['slsqp', 'mpowell', 'lbfgsb', 'tnc', 'diffevol', 'shgo', 'ipopt']
        raise ValueError(
            f"Unknown optimizer type '{otype}'. "
            f"Supported optimizers: {', '.join(supported_optimizers)}"
        )

    if opt_res is not None:
        optimal_res.method = opt_method
        optimal_res.series = pd.Series(data=opt_res['x'], index=run_dsvars["values"].keys())
        logger.info(f"\n Final Optimal: fval {opt_res['fun']:.4f},  DS Vars: {opt_res['x'].round(4)} \n")
        logger.info(f"\n DS Vars: \n {optimal_res.series} \n")
        update_opt_parameters(opt_res, optimal_res)
        gc.collect()

    else:
        optimal_res.method = otype
        optimal_res.x = np_ini_dsvars
        optimal_res.fun = None
        optimal_res.series = pd.Series(data=np_ini_dsvars, index=run_dsvars["values"].keys())

    return optimal_res
