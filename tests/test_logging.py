# -*- coding: utf-8 -*-
"""
Tests for logging configuration and functionality.
"""

import logging
# import os
# import sys
import tempfile
from pathlib import Path

import pytest

from dualmatfit.utils.logging_config import (
    setup_logging,
    get_logger,
    set_module_log_level,
    disable_logging,
    enable_logging,
    get_log_level,
)
from dualmatfit.utils.log_contexts import (
    log_solver_diagnostics,
    log_optimization_iteration,
    log_performance,
    log_convergence_tracking,
)


class TestLoggingConfiguration:
    """Test logging configuration functions."""
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        logger = setup_logging()
        assert logger is not None
        assert logger.level == logging.INFO
    
    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level."""
        logger = setup_logging(level='DEBUG')
        assert logger.level == logging.DEBUG
        
        logger = setup_logging(level='WARNING')
        assert logger.level == logging.WARNING
    
    def test_get_logger(self):
        """Test getting module-specific loggers."""
        logger1 = get_logger('optimization')
        logger2 = get_logger('solvers')
        
        assert logger1.name == 'dualmatfit.optimization'
        assert logger2.name == 'dualmatfit.solvers'
        assert logger1 is not logger2
    
    def test_set_module_log_level(self):
        """Test setting log level for specific modules."""
        logger = get_logger('test_module')
        set_module_log_level('test_module', 'DEBUG')
        assert logger.level == logging.DEBUG
        
        set_module_log_level('test_module', 'ERROR')
        assert logger.level == logging.ERROR
    
    def test_disable_enable_logging(self):
        """Test disabling and re-enabling logging."""
        disable_logging()
        level = get_log_level()
        assert level == 'Level 51'  # CRITICAL (50) + 1
        
        enable_logging('INFO')
        level = get_log_level()
        assert level == 'INFO'
    
    def test_file_logging(self):
        """Test logging to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'test.log'
            
            logger = setup_logging(
                level='INFO',
                log_file=str(log_file),
                enable_file_logging=True
            )
            
            test_logger = get_logger('test')
            test_logger.info("Test message")
            
            # Force flush
            for handler in logger.handlers:
                handler.flush()
            
            assert log_file.exists()
            content = log_file.read_text()
            assert 'Test message' in content


class TestContextManagers:
    """Test diagnostic context managers."""
    
    def test_log_solver_diagnostics_success(self, caplog):
        """Test solver diagnostics context manager with successful solve."""
        logger = get_logger('test')
        
        with caplog.at_level(logging.INFO):
            with log_solver_diagnostics(
                logger,
                'Newton-Raphson',
                initial_residual=1e-2,
                tolerance=1e-6
            ) as result:
                result['converged'] = True
                result['iterations'] = 5
                result['final_residual'] = 1e-7
        
        # Check log messages
        assert 'Starting Newton-Raphson solver' in caplog.text
        assert 'Newton-Raphson solver converged' in caplog.text
    
    def test_log_solver_diagnostics_failure(self, caplog):
        """Test solver diagnostics with non-convergence."""
        logger = get_logger('test')
        
        with caplog.at_level(logging.WARNING):
            with log_solver_diagnostics(logger, 'GMRES') as result:
                result['converged'] = False
                result['iterations'] = 100
        
        assert 'did not converge' in caplog.text
    
    def test_log_solver_diagnostics_exception(self, caplog):
        """Test solver diagnostics with exception."""
        logger = get_logger('test')
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                with log_solver_diagnostics(logger, 'Test') as result:
                    raise ValueError("Test error")
        
        assert 'failed with exception' in caplog.text
        assert 'Test error' in caplog.text
    
    def test_log_optimization_iteration(self, caplog):
        """Test optimization iteration tracking."""
        logger = get_logger('test')
        
        with caplog.at_level(logging.INFO):
            with log_optimization_iteration(
                logger,
                iteration=5,
                objective_value=123.45
            ) as result:
                result['objective_value'] = 120.30
                result['converged'] = False
                result['gradient_norm'] = 1e-3
        
        assert 'Optimization iteration 5 complete' in caplog.text
    
    def test_log_performance(self, caplog):
        """Test performance timing context."""
        logger = get_logger('test')
        logger.setLevel(logging.DEBUG)  # Ensure logger accepts DEBUG messages
        
        with caplog.at_level(logging.DEBUG):
            with log_performance(logger, 'test_operation'):
                pass  # Quick operation
        
        assert 'Starting: test_operation' in caplog.text
        assert 'Completed: test_operation' in caplog.text
        # Check that timing information is present (in format "in X.XXXXs")
        assert ' in ' in caplog.text and 's' in caplog.text
    
    def test_log_convergence_tracking_converged(self, caplog):
        """Test convergence tracking with convergence."""
        logger = get_logger('test')
        
        with caplog.at_level(logging.INFO):
            with log_convergence_tracking(
                logger,
                tolerance=1e-6,
                max_iterations=100,
                algorithm_name='CG'
            ) as conv:
                conv['converged'] = True
                conv['iterations'] = 10
                conv['final_residual'] = 1e-7
        
        assert 'CG converged' in caplog.text
    
    def test_log_convergence_tracking_not_converged(self, caplog):
        """Test convergence tracking without convergence."""
        logger = get_logger('test')
        
        with caplog.at_level(logging.WARNING):
            with log_convergence_tracking(
                logger,
                tolerance=1e-6,
                max_iterations=10,
                algorithm_name='Test'
            ) as conv:
                conv['converged'] = False
                conv['iterations'] = 10
        
        assert 'did not converge' in caplog.text


class TestEnvironmentConfiguration:
    """Test environment variable configuration."""
    
    def test_env_log_level(self, monkeypatch):
        """Test setting log level via environment variable."""
        monkeypatch.setenv('DUALMATFIT_LOG_LEVEL', 'DEBUG')
        logger = setup_logging()
        assert logger.level == logging.DEBUG
    
    def test_env_debug_modules(self, monkeypatch):
        """Test setting debug modules via environment variable."""
        monkeypatch.setenv('DUALMATFIT_DEBUG_MODULES', 'optimization,solvers')
        setup_logging()
        
        opt_logger = get_logger('optimization')
        solvers_logger = get_logger('solvers')
        
        assert opt_logger.level == logging.DEBUG
        assert solvers_logger.level == logging.DEBUG


class TestLoggingIntegration:
    """Integration tests for logging across modules."""
    
    def test_multiple_modules(self, caplog):
        """Test logging from multiple modules."""
        with caplog.at_level(logging.INFO):
            logger1 = get_logger('optimization')
            logger2 = get_logger('solvers')
            
            logger1.info("Optimization message")
            logger2.info("Solver message")
        
        assert 'Optimization message' in caplog.text
        assert 'Solver message' in caplog.text
        assert 'optimization' in caplog.text
        assert 'solvers' in caplog.text
    
    def test_hierarchical_logging(self, caplog):
        """Test hierarchical logger structure."""
        with caplog.at_level(logging.DEBUG):
            parent_logger = get_logger('optimization')
            child_logger = get_logger('optimization.basinhopping')
            
            parent_logger.debug("Parent message")
            child_logger.debug("Child message")
        
        assert 'Parent message' in caplog.text
        assert 'Child message' in caplog.text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
