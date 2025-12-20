# -*- coding: utf-8 -*-
"""
I/O utility functions for DualMatFit.

This module provides standardized file loading functions for Excel and HDF5 files,
consolidating duplicated patterns across the codebase.

Functions
---------
load_excel_params
    Load material parameters from an Excel file with multiple sheets.
load_hdf5_data
    Load experimental data from an HDF5 file.
save_dataframe
    Save a DataFrame to multiple formats (Excel, Parquet).
load_parquet_results
    Load optimization results from Parquet file.

Classes
-------
MaterialFitIO
    Handler class for material fitting I/O operations.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple

from dualmatfit.logging_config import get_logger

__all__ = [
    'MaterialFitIO',
    'load_excel_params',
    'load_hdf5_data',
    'save_dataframe',
    'load_parquet_results',
]

logger = get_logger('io_utils')


def load_excel_params(
    xlsx_path: Union[str, Path],
    decimal: str = ',',
    index_col: int = 0,
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Load material parameters from an Excel file with multiple sheets.
    
    This function reads an Excel file where each sheet contains parameter data
    for a different experimental sample (e.g., different rats). It handles
    common error cases gracefully and returns None on failure.
    
    Parameters
    ----------
    xlsx_path : str or Path
        Path to the Excel file (.xlsx).
    decimal : str, default=','
        Character to recognize as decimal point (handles European formats).
    index_col : int, default=0
        Column to use as row labels.
        
    Returns
    -------
    Dict[str, pd.DataFrame] or None
        Dictionary mapping sheet names to DataFrames, or None if loading fails.
        
    Examples
    --------
    >>> params = load_excel_params('results/material_params.xlsx')
    >>> if params is not None:
    ...     for sheet_name, df in params.items():
    ...         print(f"Sheet: {sheet_name}, Shape: {df.shape}")
    
    Notes
    -----
    The function handles the following error conditions:
    - File not found
    - Permission denied
    - Invalid Excel format
    - Empty data
    """
    xlsx_path = Path(xlsx_path)
    
    if not xlsx_path.is_file():
        logger.debug(f"Excel file not found at {xlsx_path}")
        return None
    
    try:
        df_params = pd.read_excel(
            str(xlsx_path),
            decimal=decimal,
            sheet_name=None,
            index_col=index_col,
        )
        return df_params
        
    except (FileNotFoundError, PermissionError, ValueError) as e:
        # FileNotFoundError: File doesn't exist (race condition)
        # PermissionError: Cannot read file
        # ValueError: Invalid Excel format or empty data
        logger.debug(f"Error reading Excel file {xlsx_path}: {e}")
        return None


def load_hdf5_data(
    h5_path: Union[str, Path],
    key_prefix: str = '/rato',
    mode: str = 'r',
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Load experimental data from an HDF5 file.
    
    This function reads an HDF5 file and extracts DataFrames stored under keys
    that match a specified prefix. It uses a context manager to ensure proper
    resource cleanup.
    
    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file (.h5).
    key_prefix : str, default='/rato'
        Prefix to filter keys in the HDF5 store. Only keys starting with this
        prefix (case-insensitive) will be loaded.
    mode : str, default='r'
        File mode for opening the HDF5 store ('r' for read-only).
        
    Returns
    -------
    Dict[str, pd.DataFrame] or None
        Dictionary mapping key names (without leading slash) to DataFrames,
        or None if loading fails.
        
    Examples
    --------
    >>> data = load_hdf5_data('instron_data/final_data.h5')
    >>> if data is not None:
    ...     for key, df in data.items():
    ...         print(f"Key: {key}, Shape: {df.shape}")
    
    >>> # Load with custom prefix
    >>> data = load_hdf5_data('data.h5', key_prefix='/experiment')
    
    Notes
    -----
    The function handles the following error conditions:
    - File not found
    - Permission denied
    - Invalid key access
    - HDF5 file corruption
    
    The function uses a context manager to ensure the HDF5 store is properly
    closed even if an error occurs during reading.
    """
    h5_path = Path(h5_path)
    
    if not h5_path.is_file():
        logger.debug(f"HDF5 file not found at {h5_path}")
        return None
    
    try:
        with pd.HDFStore(str(h5_path), mode=mode) as h5_store:
            # Filter keys by prefix (case-insensitive)
            h5_keys = [
                key for key in h5_store.keys()
                if key.lower().startswith(key_prefix.lower())
            ]
            # Build dict with key names stripped of leading slash
            h5_data = {key.lstrip('/'): h5_store[key] for key in h5_keys}
        
        return h5_data
        
    except (FileNotFoundError, PermissionError, KeyError, OSError) as e:
        # FileNotFoundError: File doesn't exist (race condition)
        # PermissionError: Cannot read file
        # KeyError: Invalid key in HDF5 store
        # OSError: HDF5 file corruption or format issues
        logger.debug(f"Error reading HDF5 file {h5_path}: {e}")
        return None


def save_dataframe(
    df: pd.DataFrame,
    base_path: Union[str, Path],
    filename: str,
    formats: Optional[List[str]] = None,
    compression: str = 'gzip',
) -> Dict[str, Path]:
    """
    Save a DataFrame to multiple formats.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    base_path : str or Path
        Directory where files will be saved.
    filename : str
        Base filename without extension.
    formats : List[str], optional
        List of formats to save ('excel', 'parquet'). Default is both.
    compression : str, default='gzip'
        Compression type for parquet files.
        
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping format names to saved file paths.
        
    Examples
    --------
    >>> saved = save_dataframe(df, 'results/', 'opt_params')
    >>> print(saved['excel'])  # Path to Excel file
    >>> print(saved['parquet'])  # Path to Parquet file
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    if formats is None:
        formats = ['excel', 'parquet']
    
    saved_paths: Dict[str, Path] = {}
    
    if 'excel' in formats:
        excel_path = base_path / f'{filename}.xlsx'
        try:
            df.to_excel(str(excel_path))
            saved_paths['excel'] = excel_path
            logger.debug(f"Saved Excel file: {excel_path}")
        except (PermissionError, OSError) as e:
            logger.warning(f"Failed to save Excel file {excel_path}: {e}")
    
    if 'parquet' in formats:
        parquet_path = base_path / f'{filename}.parquet.gzip'
        try:
            df.to_parquet(str(parquet_path), compression=compression)
            saved_paths['parquet'] = parquet_path
            logger.debug(f"Saved Parquet file: {parquet_path}")
        except (PermissionError, OSError) as e:
            logger.warning(f"Failed to save Parquet file {parquet_path}: {e}")
    
    return saved_paths


def load_parquet_results(
    parquet_path: Union[str, Path],
    index_cols: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Load optimization results from a Parquet file.
    
    Parameters
    ----------
    parquet_path : str or Path
        Path to the Parquet file.
    index_cols : List[str], optional
        Columns to filter/validate in loaded DataFrame.
        
    Returns
    -------
    pd.DataFrame or None
        Loaded DataFrame or None if loading fails.
        
    Examples
    --------
    >>> df = load_parquet_results('results/opt_params.parquet.gzip')
    >>> if df is not None:
    ...     print(df.columns)
    """
    parquet_path = Path(parquet_path)
    
    if not parquet_path.is_file():
        logger.debug(f"Parquet file not found at {parquet_path}")
        return None
    
    try:
        df = pd.read_parquet(str(parquet_path))
        
        if index_cols is not None:
            # Filter to only requested columns if they exist
            available_cols = df.columns.intersection(index_cols)
            if len(available_cols) < len(index_cols):
                missing = set(index_cols) - set(available_cols)
                logger.debug(f"Some columns not found in parquet: {missing}")
        
        return df
        
    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.debug(f"Error reading Parquet file {parquet_path}: {e}")
        return None


class MaterialFitIO:
    """
    Handler class for material fitting I/O operations.
    
    Encapsulates all I/O functionality for the AnisoMaterialFit class,
    following the Single Responsibility Principle.
    
    Parameters
    ----------
    path_manager : PathManager
        Path manager instance for directory handling.
    opt_type : str
        Optimization type identifier for file naming.
        
    Attributes
    ----------
    path_manager : PathManager
        Path manager for directory operations.
    opt_type : str
        Optimization type for file naming.
        
    Examples
    --------
    >>> from dualmatfit.path_manager import PathManager
    >>> pm = PathManager()
    >>> io_handler = MaterialFitIO(pm, 'slsqp')
    >>> data = io_handler.load_experimental_data('data.h5')
    """
    
    def __init__(self, path_manager, opt_type: str = 'slsqp'):
        """Initialize the I/O handler."""
        self.path_manager = path_manager
        self.opt_type = opt_type
    
    def load_experimental_data(
        self,
        h5_path: Union[str, Path],
    ) -> Dict[str, pd.DataFrame]:
        """
        Load experimental data from HDF5 file.
        
        Parameters
        ----------
        h5_path : str or Path
            Path to HDF5 file containing experimental data.
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping keys to experimental DataFrames.
            
        Raises
        ------
        FileNotFoundError
            If the HDF5 file does not exist.
        """
        h5_path = Path(h5_path)
        validated_path = self.path_manager.validate_file_exists(h5_path)
        
        with pd.HDFStore(str(validated_path), mode='r') as h5_store:
            h5_keys = h5_store.keys()
            exp_data = {key: h5_store[key] for key in h5_keys}
        
        return exp_data
    
    def save_optimization_results(
        self,
        dsvars: pd.DataFrame,
        output_dir: Union[str, Path],
        filename_prefix: str = 'opt_mat_param',
    ) -> Dict[str, Path]:
        """
        Save optimization results to disk.
        
        Saves results in both Excel and Parquet formats for compatibility
        and efficiency.
        
        Parameters
        ----------
        dsvars : pd.DataFrame
            DataFrame containing optimized design variables.
        output_dir : str or Path
            Directory to save results.
        filename_prefix : str, default='opt_mat_param'
            Prefix for output filenames.
            
        Returns
        -------
        Dict[str, Path]
            Dictionary mapping format names to saved file paths.
        """
        output_dir = Path(output_dir)
        self.path_manager.ensure_dir(output_dir)
        
        filename = f'{filename_prefix}_{self.opt_type}'
        return save_dataframe(dsvars, output_dir, filename)
    
    def load_optimization_results(
        self,
        results_dir: Union[str, Path],
        index_ds_vars: Optional[List[str]] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Load previously saved optimization results.
        
        Parameters
        ----------
        results_dir : str or Path
            Directory containing saved results.
        index_ds_vars : List[str], optional
            List of design variable names to extract baseline from.
            
        Returns
        -------
        Tuple[Optional[pd.DataFrame], Optional[pd.Series]]
            Tuple of (full results DataFrame, baseline Series).
            Returns (None, None) if loading fails.
        """
        results_dir = Path(results_dir)
        parquet_file = results_dir / f"opt_mat_param_{self.opt_type}.parquet.gzip"
        
        df_optimal = load_parquet_results(parquet_file)
        
        if df_optimal is None:
            return None, None
        
        # Extract baseline parameters
        baseline = None
        if index_ds_vars is not None:
            if 'mean' in df_optimal.index:
                baseline = df_optimal.loc["mean", index_ds_vars]
                baseline.name = "baseline"
            elif 'baseline' in df_optimal.index:
                baseline = df_optimal.loc["baseline", index_ds_vars]
        
        return df_optimal, baseline
