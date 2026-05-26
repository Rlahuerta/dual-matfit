# -*- coding: utf-8 -*-
"""
Unit tests for PathManager module

Note: Uses shared `temp_dir` fixture from conftest.py for temporary directory handling.
"""

import pytest
from pathlib import Path
from dualmatfit.utils.path_manager import PathConfiguration, PathManager, PathLike, _normalize_path


class TestNormalizePath:
    """Tests for _normalize_path helper function."""
    
    def test_normalize_string(self):
        """Test normalizing a string to Path."""
        result = _normalize_path('some/path')
        assert isinstance(result, Path)
        assert result == Path('some/path')
    
    def test_normalize_path_object(self):
        """Test normalizing a Path object (should return unchanged)."""
        path = Path('some/path')
        result = _normalize_path(path)
        assert result is path


class TestPathConfiguration:
    """Tests for PathConfiguration dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = PathConfiguration()
        assert config.data_dir == Path('instron_data')
        assert config.results_base_dir == Path('Results')
        assert config.h5_filename == 'final_data.h5'
    
    def test_get_h5_path_default(self):
        """Test default H5 path construction."""
        config = PathConfiguration()
        expected = Path('instron_data/final_data.h5')
        assert config.get_h5_path() == expected
    
    def test_get_h5_path_custom(self):
        """Test custom H5 path."""
        config = PathConfiguration()
        custom = Path('/custom/path/data.h5')
        assert config.get_h5_path(custom) == custom
    
    def test_get_results_dir_all_options(self):
        """Test results directory with all options enabled."""
        config = PathConfiguration()
        result = config.get_results_dir(1, 'nh', True, True, True)
        assert result == Path('Results/M1-nh-ka-vol-glb')
    
    def test_get_results_dir_minimal(self):
        """Test results directory with minimal options."""
        config = PathConfiguration()
        result = config.get_results_dir(3, 'fung', False, False, False)
        assert result == Path('Results/M3-fung-lc')
    
    def test_get_results_dir_mixed(self):
        """Test results directory with mixed options."""
        config = PathConfiguration()
        result = config.get_results_dir(2, 'nh', True, False, True)
        assert result == Path('Results/M2-nh-ka-glb')


class TestPathManager:
    """Tests for PathManager class."""
    
    def test_init_default(self):
        """Test initialization with defaults."""
        manager = PathManager()
        assert manager.config is not None
        assert manager.base_path == Path.cwd()
        assert manager.base_path.is_absolute()
    
    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = PathConfiguration(data_dir=Path('custom_data'))
        manager = PathManager(config=config)
        assert manager.config.data_dir == Path('custom_data')
    
    def test_init_custom_base_path_absolute(self, temp_dir):
        """Test initialization with absolute base path."""
        manager = PathManager(base_path=temp_dir)
        assert manager.base_path == temp_dir
        assert manager.base_path.is_absolute()
    
    def test_init_custom_base_path_relative(self):
        """Test initialization with relative base path."""
        manager = PathManager(base_path=Path('relative/path'))
        assert manager.base_path.is_absolute()
        assert manager.base_path == Path.cwd() / 'relative/path'
    
    def test_resolve_path_absolute(self, temp_dir):
        """Test path resolution with absolute path."""
        manager = PathManager(base_path=temp_dir)
        abs_path = Path('/absolute/path')
        result = manager._resolve_path(abs_path)
        assert result == abs_path
    
    def test_resolve_path_relative(self, temp_dir):
        """Test path resolution with relative path."""
        manager = PathManager(base_path=temp_dir)
        rel_path = Path('relative/path')
        result = manager._resolve_path(rel_path)
        assert result == temp_dir / rel_path
        assert result.is_absolute()
    
    def test_ensure_dir_creates_directory(self, temp_dir):
        """Test directory creation."""
        manager = PathManager(base_path=temp_dir)
        new_dir = Path('test_dir/sub_dir')
        result = manager.ensure_dir(new_dir)
        
        assert result.exists()
        assert result.is_dir()
        assert result == temp_dir / new_dir
    
    def test_ensure_dir_idempotent(self, temp_dir):
        """Test ensure_dir can be called multiple times."""
        manager = PathManager(base_path=temp_dir)
        test_dir = Path('test_dir')
        
        result1 = manager.ensure_dir(test_dir)
        result2 = manager.ensure_dir(test_dir)
        
        assert result1 == result2
        assert result1.exists()
    
    def test_ensure_dir_absolute_path(self, temp_dir):
        """Test ensure_dir with absolute path."""
        manager = PathManager()
        abs_dir = temp_dir / 'absolute_test'
        result = manager.ensure_dir(abs_dir)
        
        assert result.exists()
        assert result == abs_dir
    
    def test_get_rat_solution_dir(self):
        """Test rat solution directory path construction."""
        manager = PathManager()
        results_dir = Path('Results/M1-nh-glb')
        
        result1 = manager.get_rat_solution_dir(results_dir, '/rato_17')
        result2 = manager.get_rat_solution_dir(results_dir, 'rato_17')
        
        expected = Path('Results/M1-nh-glb/rato_17')
        assert result1 == expected
        assert result2 == expected
    
    def test_get_section_dir(self):
        """Test section directory path construction."""
        manager = PathManager()
        rat_dir = Path('Results/M1-nh-glb/rato_17')
        
        result = manager.get_section_dir(rat_dir, 'Ar')
        expected = Path('Results/M1-nh-glb/rato_17/Ar')
        assert result == expected
    
    def test_validate_file_exists_success(self, temp_dir):
        """Test file validation with existing file."""
        manager = PathManager(base_path=temp_dir)
        test_file = temp_dir / 'test.txt'
        test_file.write_text('test content')
        
        result = manager.validate_file_exists(Path('test.txt'))
        assert result == test_file
        assert result.is_absolute()
    
    def test_validate_file_exists_failure(self, temp_dir):
        """Test file validation with non-existent file."""
        manager = PathManager(base_path=temp_dir)
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            manager.validate_file_exists(Path('nonexistent.txt'))
    
    def test_validate_file_exists_absolute(self, temp_dir):
        """Test file validation with absolute path."""
        manager = PathManager()
        test_file = temp_dir / 'absolute_test.txt'
        test_file.write_text('test')
        
        result = manager.validate_file_exists(test_file)
        assert result == test_file
    
    def test_validate_file_exists_with_string(self, temp_dir):
        """Test file validation with string path."""
        manager = PathManager(base_path=temp_dir)
        test_file = temp_dir / 'test.txt'
        test_file.write_text('test content')
        
        result = manager.validate_file_exists('test.txt')
        assert result == test_file
        assert result.is_absolute()

    def test_resolve_h5_data_path_default(self, temp_dir):
        """Test resolving the default HDF5 path when the file exists."""
        manager = PathManager(base_path=temp_dir)
        h5_file = temp_dir / 'instron_data' / 'final_data.h5'
        h5_file.parent.mkdir()
        h5_file.write_text('test')

        result = manager.resolve_h5_data_path()

        assert result == h5_file
        assert result.is_absolute()

    def test_resolve_h5_data_path_directory_input(self, temp_dir):
        """Test resolving a directory input to its HDF5 file."""
        manager = PathManager(base_path=temp_dir)
        h5_dir = temp_dir / 'custom-data'
        h5_dir.mkdir()
        h5_file = h5_dir / 'final_data.h5'
        h5_file.write_text('test')

        result = manager.resolve_h5_data_path(h5_dir)

        assert result == h5_file

    def test_resolve_h5_data_path_default_missing_has_public_hint(self, temp_dir):
        """Test the public-facing error when the default dataset is unavailable."""
        manager = PathManager(base_path=temp_dir)

        with pytest.raises(FileNotFoundError, match='do not bundle'):
            manager.resolve_h5_data_path()


class TestPathManagerNewMethods:
    """Tests for new PathManager methods added in refactoring."""
    
    def test_project_root(self):
        """Test project_root static method."""
        root = PathManager.project_root()
        assert root.is_absolute()
        # Should contain dualmatfit package
        assert (root / 'dualmatfit').exists() or root.name == 'dual-matfit'
    
    def test_ensure_parent_dir(self, temp_dir):
        """Test ensure_parent_dir method."""
        manager = PathManager(base_path=temp_dir)
        file_path = 'subdir/nested/file.txt'
        
        parent = manager.ensure_parent_dir(file_path)
        assert parent.exists()
        assert parent.is_dir()
        assert parent == temp_dir / 'subdir' / 'nested'
    
    def test_ensure_parent_dir_with_string(self, temp_dir):
        """Test ensure_parent_dir with string path."""
        manager = PathManager(base_path=temp_dir)
        
        parent = manager.ensure_parent_dir('plots/output.png')
        assert parent.exists()
        assert parent == temp_dir / 'plots'
    
    def test_resolve_path_public_method(self, temp_dir):
        """Test public resolve_path method."""
        manager = PathManager(base_path=temp_dir)
        
        result = manager.resolve_path('Results/test')
        assert result.is_absolute()
        assert result == temp_dir / 'Results' / 'test'
    
    def test_resolve_path_with_string(self, temp_dir):
        """Test resolve_path with string input."""
        manager = PathManager(base_path=temp_dir)
        
        result = manager.resolve_path('some/path')
        assert isinstance(result, Path)
        assert result.is_absolute()
    
    def test_remove_file_existing(self, temp_dir):
        """Test remove_file with existing file."""
        manager = PathManager(base_path=temp_dir)
        test_file = temp_dir / 'to_delete.txt'
        test_file.write_text('delete me')
        
        result = manager.remove_file('to_delete.txt')
        assert result is True
        assert not test_file.exists()
    
    def test_remove_file_nonexistent(self, temp_dir):
        """Test remove_file with non-existent file."""
        manager = PathManager(base_path=temp_dir)
        
        result = manager.remove_file('nonexistent.txt')
        assert result is False
    
    def test_get_output_path(self, temp_dir):
        """Test get_output_path method."""
        manager = PathManager(base_path=temp_dir)
        
        result = manager.get_output_path('figure.png', 'plots')
        assert result.is_absolute()
        assert result.name == 'figure.png'
        assert result.parent.exists()
    
    def test_get_output_path_no_subdir(self, temp_dir):
        """Test get_output_path without subdirectory."""
        manager = PathManager(base_path=temp_dir)
        
        result = manager.get_output_path('output.csv')
        assert result.name == 'output.csv'
        assert result.parent.exists()
    
    def test_ensure_dir_with_string(self, temp_dir):
        """Test ensure_dir with string path."""
        manager = PathManager(base_path=temp_dir)
        
        result = manager.ensure_dir('string_test_dir')
        assert result.exists()
        assert result.is_dir()
    
    def test_init_with_string_base_path(self, temp_dir):
        """Test initialization with string base_path."""
        manager = PathManager(base_path=str(temp_dir))
        assert manager.base_path == temp_dir
        assert manager.base_path.is_absolute()


class TestPathManagerIntegration:
    """Integration tests for PathManager."""
    
    def test_complete_workflow(self, temp_dir):
        """Test complete path management workflow."""
        # Initialize
        config = PathConfiguration()
        manager = PathManager(config=config, base_path=temp_dir)
        
        # Create results directory
        results_dir = config.get_results_dir(1, 'nh', True, True, True)
        results_path = manager.ensure_dir(results_dir)
        assert results_path.exists()
        
        # Create rat directory
        rat_dir = manager.get_rat_solution_dir(results_path, 'rato_17')
        rat_path = manager.ensure_dir(rat_dir)
        assert rat_path.exists()
        assert 'rato_17' in str(rat_path)
        
        # Create section directory
        section_dir = manager.get_section_dir(rat_path, 'Ar')
        section_path = manager.ensure_dir(section_dir)
        assert section_path.exists()
        assert 'Ar' in str(section_path)
        
        # Verify hierarchy
        assert section_path.parent == rat_path
        assert rat_path.parent == results_path
    
    def test_pathlike_type_flexibility(self, temp_dir):
        """Test that methods accept both str and Path inputs."""
        manager = PathManager(base_path=temp_dir)
        
        # Test with string
        dir1 = manager.ensure_dir('test_str')
        assert dir1.exists()
        
        # Test with Path
        dir2 = manager.ensure_dir(Path('test_path'))
        assert dir2.exists()
        
        # Test get_rat_solution_dir with both
        rat1 = manager.get_rat_solution_dir('Results', 'rato_1')
        rat2 = manager.get_rat_solution_dir(Path('Results'), 'rato_2')
        assert isinstance(rat1, Path)
        assert isinstance(rat2, Path)
