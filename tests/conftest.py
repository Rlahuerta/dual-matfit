# Test configuration for DualMatFit
# Suppress noisy SyntaxWarnings originating from plotting modules due to
# LaTeX-style escape sequences (e.g. '\\lambda') inside normal Python strings.
# These don't affect runtime behavior but clutter test output.
import warnings

# Ignore all SyntaxWarning about invalid escape sequences from our plotting package
warnings.filterwarnings(
    "ignore",
    message=r"invalid escape sequence",
    category=SyntaxWarning,
    module=r"dualmatfit\.plotting.*",
)

# (Optional) If additional specific modules produce the same pattern, extend here.

# --- Begin: Global PyDev debugger warning filter ----------------------------------
# Centralized suppression of noisy non-warning stderr lines emitted by PyDev debugger
# when combined with coverage (e.g. "PYDEV DEBUGGER WARNING: sys.settrace() should not be used").
# This was previously duplicated in individual test modules; consolidated here.
import sys
import re
from io import TextIOBase

_ORIGINAL_STDERR = sys.stderr

class _PyDevWarningFilter(TextIOBase):
    _pattern = re.compile(r"PYDEV DEBUGGER WARNING:|sys\.settrace\(\) should not be used|Call Location:|coverage/collector\.py")

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, s):  # type: ignore[override]
        if self._pattern.search(s):
            return len(s)
        return self._wrapped.write(s)

    def flush(self):  # type: ignore[override]
        return self._wrapped.flush()

    def __getattr__(self, item):  # Delegate everything else (encoding, fileno, etc.)
        return getattr(self._wrapped, item)

# Apply filter immediately upon test session import
sys.stderr = _PyDevWarningFilter(sys.stderr)

def pytest_sessionfinish(session, exitstatus):  # noqa: D401 - test infra hook
    """Restore the original stderr at end of the pytest session."""
    sys.stderr = _ORIGINAL_STDERR
# --- End: Global PyDev debugger warning filter ------------------------------------


# --- Begin: VariationalFormulation test fixtures ----------------------------------
import pytest

@pytest.fixture
def fast_variational_form_args():
    """
    Default arguments for fast VariationalFormulation creation in tests.
    
    These settings prioritize initialization speed over expression simplification,
    which is typically unnecessary for numerical tests.
    """
    return {
        'simplify_tensors': False,  # Skip expensive symbolic simplification
        'simplify_timeout': 1,       # Short timeout if simplification is enabled
    }


@pytest.fixture
def variational_form_factory(fast_variational_form_args):
    """
    Factory fixture for creating VariationalFormulation instances in tests.
    
    Usage:
        def test_something(variational_form_factory):
            vf = variational_form_factory(
                ds=0.5, itype='nh', mix=1, 
                kappa=False, dvol=False, 
                bulk=1.0, hv=False, 
                vol_type='simo92', was=True
            )
            # Test with vf...
    
    The factory automatically applies fast mode settings unless overridden.
    """
    from dualmatfit.variational_form import VariationalFormulation
    
    def _factory(**override_args):
        # Merge fast defaults with user overrides
        args = {**fast_variational_form_args, **override_args}
        return VariationalFormulation(**args)
    
    return _factory
# --- End: VariationalFormulation test fixtures ------------------------------------


# --- Begin: Temporary directory fixtures ------------------------------------------
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for tests with automatic cleanup.
    
    This fixture creates a fresh temporary directory before each test and
    automatically removes it after the test completes (including on failure).
    
    Usage:
        def test_something(temp_dir):
            file_path = temp_dir / 'test_file.txt'
            file_path.write_text('test content')
            # Directory and contents are cleaned up after test
    
    Yields:
        Path: Path object pointing to the temporary directory.
    """
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir):
    """
    Create a temporary file within a temporary directory.
    
    Usage:
        def test_something(temp_file):
            temp_file.write_text('test content')
            # File and directory are cleaned up after test
    
    Args:
        temp_dir: The temp_dir fixture (automatically injected).
    
    Yields:
        Path: Path object pointing to a temporary file.
    """
    file_path = temp_dir / 'test_file.tmp'
    file_path.touch()
    yield file_path
# --- End: Temporary directory fixtures --------------------------------------------


# --- Begin: Sample data fixtures --------------------------------------------------
import numpy as np
import pandas as pd


@pytest.fixture
def sample_stretch_data():
    """
    Sample stretch data for testing.
    
    Returns:
        np.ndarray: Array of stretch values [1.0, 1.05, 1.1, 1.15, 1.2].
    """
    return np.array([1.0, 1.05, 1.1, 1.15, 1.2])


@pytest.fixture
def sample_load_data():
    """
    Sample load/force data corresponding to stretch data.
    
    Returns:
        np.ndarray: Array of load values.
    """
    return np.array([0.0, 0.5, 1.2, 2.1, 3.5])


@pytest.fixture
def sample_dsvars_dataframe(variational_form_factory):
    """
    Factory fixture for creating sample design variables DataFrame.
    
    Usage:
        def test_something(sample_dsvars_dataframe):
            vf = VariationalFormulation(ds=1., itype='nh', mix=1, kappa=False, dvol=False)
            dsvars = sample_dsvars_dataframe(vf)
    
    Returns:
        Callable: Factory function that creates dsvars DataFrame for a VariationalFormulation.
    """
    def _factory(var_form, values=None, variable=True, lower=0.1, upper=10.0):
        mat_vars_keys = [str(s) for s in var_form.mat_vars]
        n_vars = len(mat_vars_keys)
        
        if values is None:
            values = [1.0] * n_vars
        
        if isinstance(variable, bool):
            variable = [variable] * n_vars
        if isinstance(lower, (int, float)):
            lower = [lower] * n_vars
        if isinstance(upper, (int, float)):
            upper = [upper] * n_vars
            
        return pd.DataFrame({
            'values': values,
            'variable': variable,
            'lower': lower,
            'upper': upper
        }, index=mat_vars_keys)
    
    return _factory
# --- End: Sample data fixtures ----------------------------------------------------

