# -*- coding: utf-8 -*-
"""
Utility functions for DualMatFit.

Note: Path-related functions in this module are deprecated.
Use PathManager from dualmatfit.path_manager instead.
"""
import os
import errno
import subprocess
import warnings
import functools

import pandas as pd

from pathlib import Path
from typing import Any, Callable, TypeVar, Union

from dualmatfit.variational_form import VariationalFormulation

from dualmatfit.logging_config import get_logger
logger = get_logger('utils')

__all__ = [
    'check_dsvars',
]

# TypeVar for generic decorator typing
F = TypeVar('F', bound=Callable[..., Any])


def _deprecated(replacement: str) -> Callable[[F], F]:
    """
    Decorator to mark functions as deprecated.
    
    Parameters
    ----------
    replacement : str
        The recommended replacement function/method to use instead.
        
    Returns
    -------
    Callable[[F], F]
        Decorator function that wraps the original function with deprecation warning.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in a future version. "
                f"Use {replacement} instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


@_deprecated("PathManager.remove_file() from dualmatfit.path_manager")
def remove_file(filepath: Union[str, Path]) -> None:
    """
    Removes a file if it exists.
    
    .. deprecated::
        Use :meth:`PathManager.remove_file` from :mod:`dualmatfit.path_manager` instead.
    """
    logger.info(f"--- Cleanup function called: Attempting to remove {filepath} ---")
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            logger.info(f"--- Cleanup function: Successfully removed {filepath} ---")
        except OSError as e:
            logger.info(f"--- Cleanup function: Error removing {filepath}: {e} ---")
    else:
        logger.info(f"--- Cleanup function: File {filepath} did not exist. ---")


@_deprecated("PathManager.ensure_dir() or PathManager.ensure_parent_dir() from dualmatfit.path_manager")
def ensure_directory_exists(file_path: str) -> bool:
    """
    Ensures that the directory for the given file_path exists.
    If it doesn't exist, it attempts to create it.
    
    .. deprecated::
        Use :meth:`PathManager.ensure_dir` or :meth:`PathManager.ensure_parent_dir` 
        from :mod:`dualmatfit.path_manager` instead.

    Args:
        file_path (str): The full path to the file for which the directory
                         needs to be ensured.

    Returns:
        bool: True if the directory exists or was successfully created,
              False otherwise.
    """
    if not file_path:
        logger.error(" file_path cannot be empty.")
        return False

    # Extract the directory part of the file_path
    directory = os.path.dirname(file_path)

    # If directory is an empty string, it means the file is intended for the
    # current working directory, which always exists.
    if directory == "":
        logger.info(f"File '{file_path}' will be in the current working directory.")
        return True

    # Check if the directory exists
    if not os.path.exists(directory):
        logger.info(f"Directory '{directory}' does not exist. Attempting to create it.")
        try:
            # Create the directory, including any necessary parent directories.
            # exist_ok=True means it won't raise an error if the directory already exists.
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Successfully created directory: '{directory}'")
            return True

        except OSError as e:
            # Handle potential errors during directory creation
            # e.g., permission denied, invalid path component
            logger.info(f"Error creating directory '{directory}': {e}")
            if e.errno == errno.EACCES:
                logger.info("Permission denied to create directory.")
            elif e.errno == errno.ENAMETOOLONG:
                logger.info("The path name is too long.")
            # Add more specific error handling if needed
            return False
        except (PermissionError, FileExistsError, FileNotFoundError) as e:
            # Catch other file system errors not covered by specific errno checks
            logger.info(f"File system error while creating directory '{directory}': {e}")
            return False
    else:
        logger.info(f"Directory '{directory}' already exists.")
        return True


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

    # Use Path for all path operations
    work_path = Path(wpath) if wpath else Path.cwd()
    output_path = work_path / fname
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as file:
        file.write(latex_document)

    # Run pdflatex in the file's directory
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
