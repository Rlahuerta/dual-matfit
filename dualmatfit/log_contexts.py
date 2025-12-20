# -*- coding: utf-8 -*-
"""
Context managers for diagnostic logging in DualMatFit.

This module provides context managers that automatically track and log
diagnostic information for numerical solvers, optimization algorithms,
and performance-critical operations.

Examples
--------
Track solver convergence:

>>> from dualmatfit.logging_config import get_logger
>>> from dualmatfit.log_contexts import log_solver_diagnostics
>>> 
>>> logger = get_logger('solvers')
>>> with log_solver_diagnostics(logger, 'Newton-Raphson',
...                              initial_residual=1e-2,
...                              tolerance=1e-6):
...     # Solver iterations
...     pass

Track optimization iterations:

>>> with log_optimization_iteration(logger, iteration=5,
...                                  objective_value=123.45) as result:
...     # Optimization step
...     result['objective_value'] = 120.30
...     result['converged'] = False

Time performance-critical operations:

>>> with log_performance(logger, 'tensor_contraction'):
...     # Expensive operation
...     pass
"""

import logging
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional

__all__ = [
    'log_solver_diagnostics',
    'log_optimization_iteration',
    'log_performance',
]

# Try to import psutil for memory tracking (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@contextmanager
def log_solver_diagnostics(
    logger: logging.Logger,
    solver_name: str,
    initial_residual: Optional[float] = None,
    tolerance: Optional[float] = None,
    max_iterations: Optional[int] = None,
    **kwargs: Any,
):
    """
    Context manager for numerical solver diagnostics.
    
    Logs entry with initial conditions, and exit with convergence information.
    Automatically tracks execution time and handles exceptions.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use for output
    solver_name : str
        Name of the solver (e.g., 'Newton-Raphson', 'GMRES')
    initial_residual : float, optional
        Initial residual norm
    tolerance : float, optional
        Convergence tolerance
    max_iterations : int, optional
        Maximum number of iterations
    **kwargs : dict
        Additional solver-specific parameters to log
        
    Yields
    ------
    dict
        Dictionary to store results. Set 'converged', 'iterations', 
        'final_residual' etc. to log on exit.
        
    Examples
    --------
    Basic solver tracking:
    
    >>> with log_solver_diagnostics(logger, 'Newton-Raphson',
    ...                              initial_residual=0.01,
    ...                              tolerance=1e-6) as result:
    ...     # Solver iterations
    ...     result['converged'] = True
    ...     result['iterations'] = 5
    ...     result['final_residual'] = 1e-7
    
    With custom parameters:
    
    >>> with log_solver_diagnostics(logger, 'GMRES',
    ...                              restart=30,
    ...                              preconditioner='ILU') as result:
    ...     # Iterative solve
    ...     pass
    """
    start_time = time.perf_counter()
    
    # Build entry message
    entry_extra = {
        'solver': solver_name,
        'initial_residual': initial_residual,
        'tolerance': tolerance,
        'max_iterations': max_iterations,
    }
    entry_extra.update(kwargs)
    
    logger.info(f"Starting {solver_name} solver", extra=entry_extra)
    
    # Create result dictionary for user to populate
    result = {}
    
    try:
        yield result
        
        # Log successful completion
        duration = time.perf_counter() - start_time
        
        exit_extra = {
            'solver': solver_name,
            'duration_s': f"{duration:.6f}",
            'converged': result.get('converged', True),
        }
        
        # Add user-provided results
        for key in ['iterations', 'final_residual', 'error_estimate']:
            if key in result:
                exit_extra[key] = result[key]
        
        if result.get('converged', True):
            logger.info(f"{solver_name} solver converged", extra=exit_extra)
        else:
            logger.warning(f"{solver_name} solver did not converge", extra=exit_extra)
            
    except Exception as e:
        # Log failure
        duration = time.perf_counter() - start_time
        
        error_extra = {
            'solver': solver_name,
            'duration_s': f"{duration:.6f}",
            'error': str(e),
            'error_type': type(e).__name__,
        }
        
        logger.error(
            f"{solver_name} solver failed with exception: {type(e).__name__}: {str(e)}", 
            extra=error_extra
        )
        raise


@contextmanager
def log_optimization_iteration(
    logger: logging.Logger,
    iteration: int,
    objective_value: Optional[float] = None,
    **kwargs: Any,
):
    """
    Context manager for optimization iteration tracking.
    
    Logs entry and exit for each optimization iteration with timing
    and objective function values.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    iteration : int
        Current iteration number
    objective_value : float, optional
        Objective function value at iteration start
    **kwargs : dict
        Additional iteration-specific data (gradient_norm, step_size, etc.)
        
    Yields
    ------
    dict
        Dictionary to store iteration results
        
    Examples
    --------
    Track optimization iteration:
    
    >>> with log_optimization_iteration(logger, iteration=10,
    ...                                  objective_value=145.2,
    ...                                  gradient_norm=1e-3) as result:
    ...     # Optimization step
    ...     result['objective_value'] = 142.8
    ...     result['step_accepted'] = True
    ...     result['alpha'] = 0.5
    """
    start_time = time.perf_counter()
    
    entry_extra = {
        'iteration': iteration,
        'objective_value_pre': objective_value,
    }
    entry_extra.update(kwargs)
    
    logger.debug(f"Optimization iteration {iteration} starting", extra=entry_extra)
    
    result = {}
    
    try:
        yield result
        
        duration = time.perf_counter() - start_time
        
        exit_extra = {
            'iteration': iteration,
            'duration_s': f"{duration:.6f}",
            'objective_value_post': result.get('objective_value'),
        }
        
        # Add common optimization metrics
        for key in ['gradient_norm', 'step_size', 'alpha', 'step_accepted', 
                    'converged', 'constraint_violation']:
            if key in result:
                exit_extra[key] = result[key]
        
        logger.info(f"Optimization iteration {iteration} complete", extra=exit_extra)
        
    except Exception as e:
        duration = time.perf_counter() - start_time
        
        error_extra = {
            'iteration': iteration,
            'duration_s': f"{duration:.6f}",
            'error': str(e),
            'error_type': type(e).__name__,
        }
        
        logger.error(f"Optimization iteration {iteration} failed", extra=error_extra)
        raise


@contextmanager
def log_performance(
    logger: logging.Logger,
    operation_name: str,
    log_level: int = logging.DEBUG,
    track_memory: bool = True,
):
    """
    Context manager for performance timing and memory tracking.
    
    Automatically times operations and optionally tracks memory usage.
    Logs on entry and exit with timing information.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    operation_name : str
        Name of the operation being timed
    log_level : int, default=logging.DEBUG
        Logging level to use (DEBUG, INFO, etc.)
    track_memory : bool, default=True
        Track memory usage if psutil is available
        
    Examples
    --------
    Time a function call:
    
    >>> with log_performance(logger, 'matrix_factorization'):
    ...     A_factor = compute_factorization(A)
    
    Time with INFO level logging:
    
    >>> with log_performance(logger, 'parameter_optimization', 
    ...                       log_level=logging.INFO):
    ...     optimal_params = fit_parameters(data)
    """
    start_time = time.perf_counter()
    start_memory = None
    
    # Try to get memory usage
    if track_memory and PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except Exception:
            pass
    
    logger.log(log_level, f"Starting: {operation_name}")
    
    try:
        yield
        
        duration = time.perf_counter() - start_time
        
        extra = {
            'operation': operation_name,
            'duration_s': f"{duration:.6f}",
        }
        
        # Add memory information if available
        if start_memory is not None:
            try:
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                extra['memory_mb'] = f"{end_memory:.2f}"
                extra['memory_delta_mb'] = f"{end_memory - start_memory:+.2f}"
            except Exception:
                pass
        
        logger.log(log_level, f"Completed: {operation_name} in {duration:.4f}s", 
                   extra=extra)
        
    except Exception as e:
        duration = time.perf_counter() - start_time
        
        error_extra = {
            'operation': operation_name,
            'duration_s': f"{duration:.6f}",
            'error': str(e),
            'error_type': type(e).__name__,
        }
        
        logger.error(f"Failed: {operation_name}", extra=error_extra)
        raise


@contextmanager
def log_convergence_tracking(
    logger: logging.Logger,
    tolerance: float,
    max_iterations: int,
    algorithm_name: str = "iterative_solver",
):
    """
    Context manager for convergence monitoring.
    
    Tracks convergence of iterative algorithms and logs warnings
    if convergence is not achieved.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    tolerance : float
        Convergence tolerance
    max_iterations : int
        Maximum number of iterations
    algorithm_name : str, default='iterative_solver'
        Name of the algorithm for logging
        
    Yields
    ------
    dict
        Dictionary to populate with convergence information
        
    Examples
    --------
    Track iterative solver:
    
    >>> with log_convergence_tracking(logger, tolerance=1e-6,
    ...                                max_iterations=100,
    ...                                algorithm_name='CG') as conv:
    ...     for i in range(max_iterations):
    ...         residual = compute_residual()
    ...         if residual < tolerance:
    ...             conv['converged'] = True
    ...             conv['iterations'] = i
    ...             conv['final_residual'] = residual
    ...             break
    """
    start_time = time.perf_counter()
    
    entry_extra = {
        'algorithm': algorithm_name,
        'tolerance': tolerance,
        'max_iterations': max_iterations,
    }
    
    logger.debug(f"Starting convergence tracking for {algorithm_name}", 
                 extra=entry_extra)
    
    convergence_info = {
        'converged': False,
        'iterations': 0,
        'final_residual': None,
    }
    
    try:
        yield convergence_info
        
        duration = time.perf_counter() - start_time
        
        exit_extra = {
            'algorithm': algorithm_name,
            'converged': convergence_info['converged'],
            'iterations': convergence_info['iterations'],
            'final_residual': convergence_info.get('final_residual'),
            'duration_s': f"{duration:.6f}",
        }
        
        if convergence_info['converged']:
            logger.info(f"{algorithm_name} converged", extra=exit_extra)
        else:
            logger.warning(
                f"{algorithm_name} did not converge within {max_iterations} iterations",
                extra=exit_extra
            )
            
    except Exception as e:
        duration = time.perf_counter() - start_time
        
        error_extra = {
            'algorithm': algorithm_name,
            'duration_s': f"{duration:.6f}",
            'error': str(e),
            'error_type': type(e).__name__,
        }
        
        logger.error(f"{algorithm_name} failed", extra=error_extra)
        raise


@contextmanager
def log_function_call(
    logger: logging.Logger,
    function_name: str,
    log_args: bool = False,
    log_result: bool = False,
    *args,
    **kwargs,
):
    """
    Context manager for function call logging.
    
    Logs function entry and exit with optional argument and result logging.
    Useful for debugging complex call chains.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    function_name : str
        Name of the function being called
    log_args : bool, default=False
        Log function arguments
    log_result : bool, default=False
        Log function result
    *args
        Function arguments (if log_args=True)
    **kwargs
        Function keyword arguments (if log_args=True)
        
    Yields
    ------
    dict
        Dictionary to store result (if log_result=True, set result['value'])
        
    Examples
    --------
    Log function calls:
    
    >>> with log_function_call(logger, 'compute_jacobian',
    ...                         log_args=True, x=x0) as result:
    ...     J = compute_jacobian(x0)
    ...     result['value'] = J
    """
    entry_extra = {'function': function_name}
    
    if log_args:
        entry_extra['args'] = str(args)[:100]  # Truncate for safety
        entry_extra['kwargs'] = {k: str(v)[:50] for k, v in kwargs.items()}
    
    logger.debug(f"Calling: {function_name}", extra=entry_extra)
    
    result = {}
    
    try:
        yield result
        
        exit_extra = {'function': function_name}
        
        if log_result and 'value' in result:
            exit_extra['result'] = str(result['value'])[:100]
        
        logger.debug(f"Returned: {function_name}", extra=exit_extra)
        
    except Exception as e:
        error_extra = {
            'function': function_name,
            'error': str(e),
            'error_type': type(e).__name__,
        }
        
        logger.error(f"Exception in: {function_name}", extra=error_extra)
        raise
