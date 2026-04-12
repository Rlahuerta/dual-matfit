# -*- coding: utf-8 -*-
"""
Path Management Module

Centralized path handling for material fitting workflow.
Provides consistent, cross-platform path operations and directory management.

This module provides the canonical path handling utilities for the DualMatFit
package. All path operations should use these classes instead of os.path
or direct Path manipulation.

Type Alias
----------
PathLike : Union[str, Path]
    Accepts both string and Path objects for flexibility.
"""

from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
import warnings

from dualmatfit.utils.logging_config import get_logger

logger = get_logger('path_manager')

__all__ = [
    'PathManager',
    'PathLike',
]

# Type alias for path-like objects
PathLike = Union[str, Path]


def _normalize_path(path: PathLike) -> Path:
    """
    Normalize a path-like object to a Path.
    
    Parameters
    ----------
    path : PathLike
        String or Path object to normalize
        
    Returns
    -------
    Path
        Normalized Path object
    """
    if isinstance(path, str):
        return Path(path)
    return path


@dataclass
class PathConfiguration:
    """
    Configuration for all paths used in material fitting.
    
    Attributes
    ----------
    data_dir : Path
        Base directory for experimental data files
    results_base_dir : Path
        Base directory for all results
    h5_filename : str
        Default HDF5 data file name
    """
    
    data_dir: Path = Path('instron_data')
    results_base_dir: Path = Path('Results')
    h5_filename: str = 'final_data.h5'
    
    def get_h5_path(self, custom_path: Optional[Path] = None) -> Path:
        """
        Get HDF5 data file path.
        
        Parameters
        ----------
        custom_path : Path, optional
            Custom path to HDF5 file. If None, uses default location.
            
        Returns
        -------
        Path
            Absolute or relative path to HDF5 file
            
        Examples
        --------
        >>> config = PathConfiguration()
        >>> config.get_h5_path()
        Path('instron_data/final_data.h5')
        >>> config.get_h5_path(Path('/data/custom.h5'))
        Path('/data/custom.h5')
        """
        return custom_path if custom_path else self.data_dir / self.h5_filename
    
    def get_results_dir(self, mtype: int, itype: str, kappa: bool, 
                       dvol: bool, opt_glb: bool) -> Path:
        """
        Build standardized results directory path.
        
        Parameters
        ----------
        mtype : int
            Mixed formulation type (1, 2, or 3)
        itype : str
            Isotropic model type ('nh', 'fung', etc.)
        kappa : bool
            Include fiber dispersion in path name
        dvol : bool
            Include volumetric effects in path name
        opt_glb : bool
            Use global optimization (True) or local (False)
            
        Returns
        -------
        Path
            Standardized results directory path
            
        Examples
        --------
        >>> config = PathConfiguration()
        >>> config.get_results_dir(1, 'nh', True, True, True)
        Path('Results/M1-nh-ka-vol-glb')
        >>> config.get_results_dir(3, 'fung', False, False, False)
        Path('Results/M3-fung-lc')
        """
        dir_name = f'M{mtype}-{itype}'
        if kappa:
            dir_name += '-ka'
        if dvol:
            dir_name += '-vol'
        dir_name += '-glb' if opt_glb else '-lc'
        return self.results_base_dir / dir_name


class PathManager:
    """
    Manages all file system paths for material fitting workflow.
    
    Provides centralized path operations including:
    - Path resolution (relative to absolute)
    - Directory creation with validation
    - Path validation and existence checks
    - Standardized directory structure
    
    Attributes
    ----------
    config : PathConfiguration
        Path configuration settings
    base_path : Path
        Base directory for resolving relative paths (always absolute)
        
    Examples
    --------
    Create with default configuration:
    
    >>> manager = PathManager()
    >>> manager.base_path
    Path('/current/working/directory')
    
    Create with custom base path:
    
    >>> manager = PathManager(base_path=Path('/data'))
    >>> manager.base_path
    Path('/data')
    
    Ensure directory exists:
    
    >>> output_dir = manager.ensure_dir(Path('Results/test'))
    >>> output_dir.exists()
    True
    """
    
    def __init__(self, config: Optional[PathConfiguration] = None,
                 base_path: Optional[PathLike] = None):
        """
        Initialize path manager.
        
        Parameters
        ----------
        config : PathConfiguration, optional
            Path configuration. If None, uses default configuration.
        base_path : PathLike, optional
            Base directory for all relative paths. Accepts str or Path.
            If None, uses current working directory.
            Will be converted to absolute path.
        """
        self.config = config or PathConfiguration()
        
        # Set and ensure base_path is absolute
        if base_path is None:
            self.base_path = Path.cwd()
        else:
            self.base_path = _normalize_path(base_path)
            if not self.base_path.is_absolute():
                self.base_path = Path.cwd() / self.base_path
    
    @staticmethod
    def project_root() -> Path:
        """
        Get the project root directory.
        
        Returns the parent directory of the dualmatfit package,
        which is typically the repository root.
        
        Returns
        -------
        Path
            Absolute path to the project root directory
            
        Examples
        --------
        >>> PathManager.project_root()
        Path('/home/user/DualMatFit')
        """
        return Path(__file__).resolve().parent.parent
    
    def ensure_dir(self, path: PathLike) -> Path:
        """
        Create directory if it doesn't exist and return absolute path.
        
        Creates all necessary parent directories. Safe to call multiple times.
        
        Parameters
        ----------
        path : PathLike
            Directory path to create (can be relative or absolute, str or Path)
            
        Returns
        -------
        Path
            Absolute path to the created/existing directory
            
        Examples
        --------
        >>> manager = PathManager()
        >>> output_dir = manager.ensure_dir(Path('Results/test/subsection'))
        >>> output_dir.exists()
        True
        >>> output_dir.is_dir()
        True
        >>> output_dir = manager.ensure_dir('Results/test')  # Also works with str
        """
        path = _normalize_path(path)
        abs_path = self._resolve_path(path)
        abs_path.mkdir(parents=True, exist_ok=True)
        return abs_path
    
    def ensure_parent_dir(self, file_path: PathLike) -> Path:
        """
        Create parent directory for a file path if it doesn't exist.
        
        Useful when you need to ensure a file's parent directory exists
        before writing to it.
        
        Parameters
        ----------
        file_path : PathLike
            Path to a file (can be relative or absolute, str or Path)
            
        Returns
        -------
        Path
            Absolute path to the parent directory
            
        Examples
        --------
        >>> manager = PathManager()
        >>> parent = manager.ensure_parent_dir('Results/plots/figure.png')
        >>> parent.exists()
        True
        """
        file_path = _normalize_path(file_path)
        abs_path = self._resolve_path(file_path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        return abs_path.parent
    
    def _resolve_path(self, path: PathLike) -> Path:
        """
        Resolve path relative to base_path if not absolute.
        
        Parameters
        ----------
        path : PathLike
            Path to resolve (can be relative or absolute, str or Path)
            
        Returns
        -------
        Path
            Absolute path
            
        Examples
        --------
        >>> manager = PathManager(base_path=Path('/home/user'))
        >>> manager._resolve_path(Path('Results'))
        Path('/home/user/Results')
        >>> manager._resolve_path(Path('/absolute/path'))
        Path('/absolute/path')
        >>> manager._resolve_path('Results')  # Also works with str
        Path('/home/user/Results')
        """
        path = _normalize_path(path)
        if path.is_absolute():
            return path
        return self.base_path / path
    
    def resolve_path(self, path: PathLike) -> Path:
        """
        Public method to resolve path relative to base_path.
        
        Parameters
        ----------
        path : PathLike
            Path to resolve (can be relative or absolute, str or Path)
            
        Returns
        -------
        Path
            Absolute path
            
        Examples
        --------
        >>> manager = PathManager(base_path='/home/user')
        >>> manager.resolve_path('Results')
        Path('/home/user/Results')
        """
        return self._resolve_path(path)
    
    @staticmethod
    def get_rat_solution_dir(results_dir: PathLike, rat_id: str) -> Path:
        """
        Get solution directory for a specific rat/specimen.
        
        Parameters
        ----------
        results_dir : PathLike
            Base results directory
        rat_id : str
            Rat/specimen identifier (e.g., '/rato_17')
            
        Returns
        -------
        Path
            Path to rat-specific solution directory
            
        Notes
        -----
        Handles rat_id with or without leading slash.
        
        Examples
        --------
        >>> manager = PathManager()
        >>> results = Path('Results/M1-nh-glb')
        >>> manager.get_rat_solution_dir(results, '/rato_17')
        Path('Results/M1-nh-glb/rato_17')
        >>> manager.get_rat_solution_dir(results, 'rato_17')
        Path('Results/M1-nh-glb/rato_17')
        """
        results_dir = _normalize_path(results_dir)
        clean_id = rat_id.lstrip('/')
        return results_dir / clean_id
    
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
            
        Examples
        --------
        >>> manager = PathManager()
        >>> rat_dir = Path('Results/M1-nh-glb/rato_17')
        >>> manager.get_section_dir(rat_dir, 'Ar')
        Path('Results/M1-nh-glb/rato_17/Ar')
        """
        rat_dir = _normalize_path(rat_dir)
        return rat_dir / section
    
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
            
        Examples
        --------
        >>> manager = PathManager()
        >>> manager.validate_file_exists(Path('instron_data/final_data.h5'))
        Path('/absolute/path/to/instron_data/final_data.h5')
        >>> manager.validate_file_exists('instron_data/final_data.h5')  # str works too
        """
        file_path = _normalize_path(file_path)
        abs_path = self._resolve_path(file_path)
        if not abs_path.is_file():
            raise FileNotFoundError(f"File not found: {abs_path}")
        return abs_path
    
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
            
        Examples
        --------
        >>> manager = PathManager()
        >>> manager.remove_file('temp/output.txt')
        True
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
            
        Examples
        --------
        >>> manager = PathManager()
        >>> manager.get_output_path('figure.png', 'plots')
        Path('/path/to/Results/plots/figure.png')
        """
        if subdir:
            output_dir = self.config.results_base_dir / subdir
        else:
            output_dir = self.config.results_base_dir
        
        abs_dir = self.ensure_dir(output_dir)
        return abs_dir / filename
