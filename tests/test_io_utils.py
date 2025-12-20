# -*- coding: utf-8 -*-
"""
Unit tests for I/O utility functions.

Tests for:
- load_excel_params
- load_hdf5_data
- save_dataframe
- load_parquet_results
- MaterialFitIO class

Note: Uses shared `tmp_path` fixture from pytest (built-in) for temporary directory handling.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from dualmatfit.io_utils import (
    load_excel_params,
    load_hdf5_data,
    save_dataframe,
    load_parquet_results,
    MaterialFitIO,
)


class TestLoadExcelParams:
    """Tests for load_excel_params function."""
    
    def test_load_nonexistent_file(self):
        """Should return None for non-existent file."""
        result = load_excel_params('/nonexistent/path/file.xlsx')
        assert result is None
    
    def test_load_valid_excel(self, tmp_path):
        """Should load valid Excel file with multiple sheets."""
        # Create test Excel file
        excel_path = tmp_path / 'test.xlsx'
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df2 = pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})
        
        with pd.ExcelWriter(str(excel_path)) as writer:
            df1.to_excel(writer, sheet_name='Sheet1')
            df2.to_excel(writer, sheet_name='Sheet2')
        
        result = load_excel_params(excel_path, decimal='.')
        
        assert result is not None
        assert 'Sheet1' in result
        assert 'Sheet2' in result
    
    def test_load_with_string_path(self, tmp_path):
        """Should accept string path."""
        excel_path = tmp_path / 'test.xlsx'
        df = pd.DataFrame({'A': [1, 2, 3]})
        df.to_excel(str(excel_path))
        
        result = load_excel_params(str(excel_path), decimal='.')
        
        assert result is not None


class TestLoadHdf5Data:
    """Tests for load_hdf5_data function."""
    
    def test_load_nonexistent_file(self):
        """Should return None for non-existent file."""
        result = load_hdf5_data('/nonexistent/path/file.h5')
        assert result is None
    
    def test_load_valid_hdf5(self, tmp_path):
        """Should load valid HDF5 file."""
        h5_path = tmp_path / 'test.h5'
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        with pd.HDFStore(str(h5_path), mode='w') as store:
            store['/rato_1'] = df
            store['/rato_2'] = df.copy()
        
        result = load_hdf5_data(h5_path, key_prefix='/rato')
        
        assert result is not None
        assert len(result) == 2
    
    def test_load_with_custom_prefix(self, tmp_path):
        """Should filter by key prefix."""
        h5_path = tmp_path / 'test.h5'
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        with pd.HDFStore(str(h5_path), mode='w') as store:
            store['/rato_1'] = df
            store['/other_key'] = df.copy()
        
        result = load_hdf5_data(h5_path, key_prefix='/rato')
        
        assert result is not None
        assert len(result) == 1
        assert 'rato_1' in result


class TestSaveDataFrame:
    """Tests for save_dataframe function."""
    
    def test_save_to_excel_and_parquet(self, tmp_path):
        """Should save to both formats."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        result = save_dataframe(df, tmp_path, 'test_file')
        
        assert 'excel' in result
        assert 'parquet' in result
        assert result['excel'].exists()
        assert result['parquet'].exists()
    
    def test_save_only_parquet(self, tmp_path):
        """Should save only parquet when specified."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        result = save_dataframe(df, tmp_path, 'test_file', formats=['parquet'])
        
        assert 'parquet' in result
        assert 'excel' not in result
    
    def test_save_creates_directory(self, tmp_path):
        """Should create directory if it doesn't exist."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        new_dir = tmp_path / 'new_dir' / 'nested'
        
        result = save_dataframe(df, new_dir, 'test_file')
        
        assert new_dir.exists()
        assert 'excel' in result


class TestLoadParquetResults:
    """Tests for load_parquet_results function."""
    
    def test_load_nonexistent_file(self):
        """Should return None for non-existent file."""
        result = load_parquet_results('/nonexistent/path/file.parquet')
        assert result is None
    
    def test_load_valid_parquet(self, tmp_path):
        """Should load valid parquet file."""
        parquet_path = tmp_path / 'test.parquet.gzip'
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df.to_parquet(str(parquet_path), compression='gzip')
        
        result = load_parquet_results(parquet_path)
        
        assert result is not None
        assert list(result.columns) == ['A', 'B']
    
    def test_load_with_index_cols(self, tmp_path):
        """Should handle index_cols parameter."""
        parquet_path = tmp_path / 'test.parquet'
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        df.to_parquet(str(parquet_path))
        
        result = load_parquet_results(parquet_path, index_cols=['A', 'B'])
        
        assert result is not None


class TestMaterialFitIO:
    """Tests for MaterialFitIO class."""
    
    @pytest.fixture
    def mock_path_manager(self, tmp_path):
        """Create a mock PathManager."""
        pm = MagicMock()
        pm.validate_file_exists = MagicMock(side_effect=lambda p: p)
        pm.ensure_dir = MagicMock(side_effect=lambda p: Path(p).mkdir(parents=True, exist_ok=True))
        return pm
    
    def test_init(self, mock_path_manager):
        """Should initialize correctly."""
        io_handler = MaterialFitIO(mock_path_manager, 'slsqp')
        
        assert io_handler.path_manager == mock_path_manager
        assert io_handler.opt_type == 'slsqp'
    
    def test_load_experimental_data(self, mock_path_manager, tmp_path):
        """Should load experimental data from HDF5."""
        # Create test HDF5 file
        h5_path = tmp_path / 'test.h5'
        df = pd.DataFrame({'force': [1, 2, 3], 'stretch': [1.0, 1.1, 1.2]})
        
        with pd.HDFStore(str(h5_path), mode='w') as store:
            store['/rato_1'] = df
        
        io_handler = MaterialFitIO(mock_path_manager, 'slsqp')
        result = io_handler.load_experimental_data(h5_path)
        
        assert '/rato_1' in result
    
    def test_save_optimization_results(self, mock_path_manager, tmp_path):
        """Should save optimization results to disk."""
        io_handler = MaterialFitIO(mock_path_manager, 'slsqp')
        df = pd.DataFrame({'mu': [0.01], 'k1': [0.02]}, index=['baseline'])
        
        result = io_handler.save_optimization_results(df, tmp_path)
        
        assert 'excel' in result
        assert 'parquet' in result
    
    def test_load_optimization_results_not_found(self, mock_path_manager, tmp_path):
        """Should return (None, None) when file doesn't exist."""
        io_handler = MaterialFitIO(mock_path_manager, 'slsqp')
        
        df, baseline = io_handler.load_optimization_results(tmp_path)
        
        assert df is None
        assert baseline is None
    
    def test_load_optimization_results_with_baseline(self, mock_path_manager, tmp_path):
        """Should extract baseline from loaded results."""
        # Create test parquet file
        parquet_path = tmp_path / 'opt_mat_param_slsqp.parquet.gzip'
        df = pd.DataFrame({
            'mu': [0.01, 0.02],
            'k1': [0.03, 0.04]
        }, index=['baseline', 'Ar-A'])
        df.to_parquet(str(parquet_path), compression='gzip')
        
        io_handler = MaterialFitIO(mock_path_manager, 'slsqp')
        loaded_df, baseline = io_handler.load_optimization_results(
            tmp_path, 
            index_ds_vars=['mu', 'k1']
        )
        
        assert loaded_df is not None
        assert baseline is not None
        assert baseline.name == 'baseline'


class TestIntegration:
    """Integration tests for I/O utilities."""
    
    def test_save_and_load_roundtrip(self, tmp_path):
        """Should be able to save and reload data correctly."""
        # Create and save
        original_df = pd.DataFrame({
            'mu': [0.01, 0.02, 0.03],
            'k1': [0.1, 0.2, 0.3],
            'k2': [1.0, 2.0, 3.0]
        }, index=['baseline', 'Ar-A', 'Tr-B'])
        
        save_dataframe(original_df, tmp_path, 'test_params', formats=['parquet'])
        
        # Reload
        loaded_df = load_parquet_results(tmp_path / 'test_params.parquet.gzip')
        
        assert loaded_df is not None
        pd.testing.assert_frame_equal(original_df, loaded_df)
