# -*- coding: utf-8 -*-
"""
Logging configuration for DualMatFit.

This module provides centralized logging configuration with support for:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Multiple handlers (console, file, rotating file)
- Structured formatters with timestamps and context
- Environment variable configuration
- Module-specific log levels
- Thread-safe initialization

Environment Variables
---------------------
DUALMATFIT_LOG_LEVEL : str
    Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    Default: INFO
DUALMATFIT_LOG_FILE : str
    Path to log file
    Default: dualmatfit.log
DUALMATFIT_LOG_FORMAT : str
    Log format style ('simple', 'detailed', 'json')
    Default: detailed
DUALMATFIT_LOG_TO_FILE : str
    Enable file logging ('true' or 'false')
    Default: false
DUALMATFIT_DEBUG_MODULES : str
    Comma-separated list of modules to set to DEBUG level
    Example: "optimization,solvers"
    
Examples
--------
Configure logging with default settings:

>>> from dualmatfit.utils.logging_config import setup_logging
>>> setup_logging()

Configure with DEBUG level and file logging:

>>> setup_logging(level='DEBUG', enable_file_logging=True)

Set specific modules to DEBUG:

>>> from dualmatfit.utils.logging_config import set_module_log_level
>>> set_module_log_level('optimization', 'DEBUG')
"""

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Union
from logging.handlers import RotatingFileHandler

__all__ = [
    'setup_logging',
    'get_logger',
    'set_module_log_level',
]

# Package root logger name
ROOT_LOGGER_NAME = "dualmatfit"

# Default configuration
DEFAULT_LOG_LEVEL = "INFO"  # INFO by default (users can enable DEBUG when needed)
DEFAULT_LOG_FILE = "dualmatfit.log"

# Format templates
SIMPLE_FORMAT = "%(levelname)s: %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s"
MINIMAL_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _LoggingState:
    """
    Thread-safe singleton for tracking logging initialization state.
    
    This class provides a thread-safe way to track whether logging has been
    initialized, avoiding issues with global mutable state in multi-threaded
    environments and improving test isolation.
    """
    
    _instance: Optional['_LoggingState'] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> '_LoggingState':
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    @property
    def initialized(self) -> bool:
        """Check if logging has been initialized."""
        return self._initialized
    
    @initialized.setter
    def initialized(self, value: bool) -> None:
        """Set initialization state (thread-safe)."""
        with self._lock:
            self._initialized = value
    
    def reset(self) -> None:
        """
        Reset logging state for testing purposes.
        
        This method is primarily intended for use in test fixtures to ensure
        clean logging state between tests.
        """
        with self._lock:
            self._initialized = False


# Module-level singleton instance
_logging_state = _LoggingState()


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    value = os.environ.get(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    return default


def _get_log_format(format_style: str) -> str:
    """Get format string based on style."""
    format_map = {
        "simple": SIMPLE_FORMAT,
        "detailed": DETAILED_FORMAT,
        "minimal": MINIMAL_FORMAT,
    }
    return format_map.get(format_style.lower(), DETAILED_FORMAT)


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    format_style: str = "detailed",
    enable_file_logging: bool = False,
    enable_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 3,
) -> logging.Logger:
    """
    Configure logging for DualMatFit.
    
    This function sets up the logging system with console and optional file handlers.
    It should be called once at module initialization. Subsequent calls will reconfigure
    the existing loggers.
    
    Parameters
    ----------
    level : str, optional
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        If None, reads from DUALMATFIT_LOG_LEVEL environment variable.
        Default: INFO
    log_file : str or Path, optional
        Path to log file. If None, reads from DUALMATFIT_LOG_FILE environment variable.
        Default: dualmatfit.log
    format_style : str, default='detailed'
        Format style: 'simple', 'detailed', or 'minimal'
        Can be overridden with DUALMATFIT_LOG_FORMAT environment variable
    enable_file_logging : bool, default=False
        Enable logging to file. Can be overridden with DUALMATFIT_LOG_TO_FILE 
        environment variable
    enable_console : bool, default=True
        Enable logging to console
    max_bytes : int, default=10485760
        Maximum size of log file before rotation (bytes)
    backup_count : int, default=3
        Number of backup log files to keep
        
    Returns
    -------
    logging.Logger
        Configured root logger for dualmatfit
        
    Examples
    --------
    Basic setup with defaults:
    
    >>> logger = setup_logging()
    >>> logger.info("Application started")
    
    Setup with DEBUG level and file logging:
    
    >>> logger = setup_logging(level='DEBUG', enable_file_logging=True)
    
    Setup with custom log file:
    
    >>> logger = setup_logging(log_file='/tmp/my_analysis.log', 
    ...                         enable_file_logging=True)
    """
    # Read from environment variables
    level = level or os.environ.get("DUALMATFIT_LOG_LEVEL", DEFAULT_LOG_LEVEL)
    log_file = log_file or os.environ.get("DUALMATFIT_LOG_FILE", DEFAULT_LOG_FILE)
    format_style = os.environ.get("DUALMATFIT_LOG_FORMAT", format_style)
    enable_file_logging = enable_file_logging or _get_env_bool("DUALMATFIT_LOG_TO_FILE")
    
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Get root logger for dualmatfit
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates (thread-safe check)
    if _logging_state.initialized:
        logger.handlers.clear()
    
    # Create formatter
    log_format = _get_log_format(format_style)
    formatter = logging.Formatter(log_format, datefmt=DATE_FORMAT)
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file_logging:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Allow propagation to root logger for testing frameworks like pytest
    # Propagation must be True for pytest's caplog to capture logs
    logger.propagate = True
    
    # Process module-specific debug settings
    debug_modules = os.environ.get("DUALMATFIT_DEBUG_MODULES", "")
    if debug_modules:
        for module_name in debug_modules.split(","):
            module_name = module_name.strip()
            if module_name:
                set_module_log_level(module_name, "DEBUG")
    
    _logging_state.initialized = True
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    This function returns a child logger under the dualmatfit namespace.
    The logger inherits settings from the root dualmatfit logger.
    
    Parameters
    ----------
    name : str
        Module name (without 'dualmatfit.' prefix).
        Can be hierarchical using dots, e.g., 'optimization.basinhopping'
        
    Returns
    -------
    logging.Logger
        Logger instance for the specified module
        
    Examples
    --------
    Get logger for a module:
    
    >>> logger = get_logger('optimization')
    >>> logger.info("Starting optimization")
    
    Get logger with hierarchical name:
    
    >>> logger = get_logger('optimization.basinhopping')
    >>> logger.debug("Basin hopping iteration %d", iteration)
    """
    # Remove 'dualmatfit.' prefix if present
    if name.startswith(f"{ROOT_LOGGER_NAME}."):
        name = name[len(ROOT_LOGGER_NAME) + 1:]
    
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def set_module_log_level(module: str, level: str) -> None:
    """
    Set log level for a specific module.
    
    This allows fine-grained control over logging verbosity for different
    parts of the codebase.
    
    Parameters
    ----------
    module : str
        Module name (without 'dualmatfit.' prefix)
    level : str
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Examples
    --------
    Enable DEBUG logging for optimization:
    
    >>> set_module_log_level('optimization', 'DEBUG')
    
    Reduce verbosity for plotting:
    
    >>> set_module_log_level('plotting', 'WARNING')
    """
    logger = get_logger(module)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)


def disable_logging() -> None:
    """
    Disable all logging output.
    
    This sets the root logger level to CRITICAL+1, effectively
    silencing all log messages.
    
    Examples
    --------
    >>> disable_logging()
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(logging.CRITICAL + 1)


def enable_logging(level: str = "INFO") -> None:
    """
    Re-enable logging after it has been disabled.
    
    Parameters
    ----------
    level : str, default='INFO'
        Log level to restore
        
    Examples
    --------
    >>> enable_logging('DEBUG')
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)


def get_log_level() -> str:
    """
    Get the current log level of the root logger.
    
    Returns
    -------
    str
        Current log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Examples
    --------
    >>> level = get_log_level()
    >>> print(f"Current log level: {level}")
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    level_name = logging.getLevelName(logger.level)
    return level_name


# Initialize logging on module import with defaults
# This ensures logging works even if setup_logging() is never called explicitly
if not _logging_state.initialized:
    setup_logging()
