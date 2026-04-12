# -*- coding: utf-8 -*-
"""
Safe simplification utilities with timeout protection.

This module provides utilities for safely simplifying SymPy expressions
with timeout protection to prevent hanging on complex expressions.
"""

import warnings
import sympy as sy
from multiprocessing import Process, Queue
from queue import Empty
from typing import Any, Optional

from dualmatfit.utils.logging_config import get_logger

logger = get_logger('simplify')

__all__ = [
    'safe_simplify',
]


def safe_simplify(expr: Any, timeout: float = 5.0) -> Any:
    """
    Safely simplify a SymPy expression with a timeout.
    
    Uses a separate process to perform simplification, allowing the operation
    to be terminated if it exceeds the specified timeout. This prevents
    hanging on complex expressions that can take excessive time to simplify.

    Parameters
    ----------
    expr : Any
        SymPy expression to simplify. Can be any SymPy object that
        supports the simplify() function.
    timeout : float, optional
        Maximum time (in seconds) allowed for simplification.
        Defaults to 5.0 seconds.

    Returns
    -------
    Any
        Simplified expression if successful within timeout,
        otherwise the original expression.
        
    Warns
    -----
    UserWarning
        If simplification times out or fails due to process errors.

    Examples
    --------
    >>> import sympy as sy
    >>> x = sy.Symbol('x')
    >>> expr = sy.sin(x)**2 + sy.cos(x)**2
    >>> safe_simplify(expr)
    1
    
    >>> # Complex expression that might timeout
    >>> safe_simplify(complex_expr, timeout=10.0)
    
    Notes
    -----
    - The function spawns a separate process, which has overhead.
      For simple expressions, direct sy.simplify() may be faster.
    - If the worker process crashes, the original expression is returned.
    - Resources are properly cleaned up even on timeout or failure.
    """
    def worker(q: Queue, expression: Any) -> None:
        """Worker function that runs simplification in separate process."""
        try:
            result = sy.simplify(expression)
            q.put(('success', result))
        except Exception as e:
            q.put(('error', str(e)))

    q: Queue = Queue()
    p: Optional[Process] = None
    
    try:
        p = Process(target=worker, args=(q, expr))
        p.start()
        p.join(timeout=timeout)

        if p.is_alive():
            # Process is still running - terminate it
            p.terminate()
            p.join(timeout=1.0)  # Give it a moment to terminate gracefully
            
            # If still alive after terminate, force kill
            if p.is_alive():
                p.kill()
                p.join(timeout=1.0)
            
            warnings.warn(
                f"Simplification timed out after {timeout} seconds. "
                "Returning original expression.",
                UserWarning
            )
            return expr
        
        # Process completed - try to get result with a short timeout
        # to avoid blocking if queue is empty (worker crashed before putting)
        try:
            status, result = q.get(timeout=0.1)
            if status == 'success':
                return result
            else:
                # Worker reported an error
                logger.debug(f"Simplification failed with error: {result}")
                warnings.warn(
                    f"Simplification failed: {result}. Returning original expression.",
                    UserWarning
                )
                return expr
        except Empty:
            # Queue is empty - worker crashed before putting result
            logger.debug("Worker process terminated without producing result")
            warnings.warn(
                "Simplification process terminated unexpectedly. "
                "Returning original expression.",
                UserWarning
            )
            return expr
            
    except (OSError, ProcessLookupError, TimeoutError, ValueError) as e:
        # Handle errors in process management (process creation, termination, queue operations)
        logger.debug(f"Error during simplification: {e}")
        warnings.warn(
            f"Error during simplification: {e}. Returning original expression.",
            UserWarning
        )
        return expr
        
    finally:
        # Ensure process resources are cleaned up
        if p is not None:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=1.0)
            p.close()
        
        # Close the queue to release resources
        try:
            q.close()
            q.join_thread()
        except Exception:
            pass  # Queue cleanup is best-effort
