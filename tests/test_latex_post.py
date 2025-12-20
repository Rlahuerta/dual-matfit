"""Unit tests for latex_post.py module.

Note: Uses shared `temp_dir` fixture from conftest.py for temporary directory handling.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from dualmatfit.latex_post import (
    generate_latex_material_props_table,
    generate_latex_dimensions_table,
    generate_latex_dim2_table,
    sections_keys,
)


@pytest.fixture
def sample_mat_param():
    """Create sample material parameters DataFrame."""
    return {
        "rat_01": pd.DataFrame({
            "mu": [0.1234, 0.2345, 0.3456],
            "bulk": [1.0, 1.1, 1.2],
            "k_1": [0.5, 0.6, 0.7],
            "k_2": [2.0, 2.1, 2.2],
            "alpha": [45.0, 50.0, 55.0],
            "kappa": [0.1, 0.15, 0.2],
        }, index=["baseline", "Ar-A", "Ar-B"]),
        "rat_02": pd.DataFrame({
            "mu": [0.4567, 0.5678, 0.0],  # Include zero value to test conditional
            "bulk": [1.3, 1.4, 1.5],
            "k_1": [0.8, 0.9, 1.0],
            "k_2": [2.3, 2.4, 2.5],
            "alpha": [60.0, 65.0, 70.0],
            "kappa": [0.25, 0.3, 0.35],
        }, index=["baseline", "Tr-A", "Ab-B"]),
    }


@pytest.fixture
def sample_dim_data():
    """Create sample dimension data."""
    return {
        "rat-01": {
            "Ar": {
                "thick": 0.1234,
                "dia": 2.5678,
                "A": {"len": 1.0},
                "B": {"len": 1.5},
                "C": {"len": 2.0},
            },
            "Tr": {
                "thick": 0.2345,
                "dia": 3.4567,
                "A": {"len": 1.1},
                "B": {"len": 1.6},
                "C": None,
            },
            "Ab": {
                "thick": 0.3456,
                "dia": 4.5678,
                "A": {"len": 1.2},
                "B": {"len": 1.7},
                "C": {"len": 2.2},
            },
        },
        "rat-02": {
            "Ar": {
                "thick": 0.4567,
                "dia": 5.6789,
                "A": {"len": 1.3},
                "B": {"len": 1.8},
                "C": {"len": 2.3},
            },
        },
    }


class TestLatexPost:
    """Test cases for latex_post.py functions."""

    def test_sections_keys_mapping(self):
        """Test that sections_keys mapping is correct."""
        assert sections_keys["Ar"] == "aoa"
        assert sections_keys["Tr"] == "dtao"
        assert sections_keys["Ab"] == "daao"

    def test_generate_latex_material_props_table_creates_file(self, temp_dir, sample_mat_param):
        """Test that generate_latex_material_props_table creates an output file."""
        output_file = temp_dir / "material_table.tex"
        
        generate_latex_material_props_table(
            sample_mat_param,
            str(output_file),
            "Test Caption",
            "tab:test",
        )
        
        assert output_file.exists()

    def test_generate_latex_material_props_table_content(self, temp_dir, sample_mat_param):
        """Test that the generated LaTeX table contains expected content."""
        output_file = temp_dir / "material_table.tex"
        
        generate_latex_material_props_table(
            sample_mat_param,
            str(output_file),
            "Test Caption",
            "tab:test",
        )
        
        content = output_file.read_text()
        
        # Check for LaTeX structure
        assert r"\begin{table}" in content
        assert r"\end{table}" in content
        assert r"\caption{Test Caption}" in content
        assert r"\label{tab:test}" in content
        assert "baseline" in content
        assert r"\rowcolor{gray!30}" in content

    def test_generate_latex_material_props_table_handles_zero_mu(self, temp_dir, sample_mat_param):
        """Test that zero mu values are handled with dashes."""
        output_file = temp_dir / "material_table.tex"
        
        generate_latex_material_props_table(
            sample_mat_param,
            str(output_file),
            "Test Caption",
            "tab:test",
        )
        
        content = output_file.read_text()
        
        # Check that dashes are used for zero mu values
        assert "{-}" in content

    def test_generate_latex_dimensions_table_creates_file(self, temp_dir, sample_dim_data):
        """Test that generate_latex_dimensions_table creates an output file."""
        output_file = temp_dir / "dimensions_table.tex"
        
        generate_latex_dimensions_table(
            sample_dim_data,
            str(output_file),
            "Dimensions Caption",
            "tab:dim",
        )
        
        assert output_file.exists()

    def test_generate_latex_dimensions_table_content(self, temp_dir, sample_dim_data):
        """Test that the generated dimensions table contains expected content."""
        output_file = temp_dir / "dimensions_table.tex"
        
        generate_latex_dimensions_table(
            sample_dim_data,
            str(output_file),
            "Dimensions Caption",
            "tab:dim",
        )
        
        content = output_file.read_text()
        
        # Check for LaTeX structure
        assert r"\begin{table}" in content
        assert r"\end{table}" in content
        assert r"\caption{Dimensions Caption}" in content
        assert r"\label{tab:dim}" in content
        assert r"\acrshort{aoa}" in content  # Ar -> aoa

    def test_generate_latex_dimensions_table_rats_ids_populated(self, temp_dir, sample_dim_data):
        """Test that rats_ids dictionary is correctly populated (regression test for key_i bug)."""
        output_file = temp_dir / "dimensions_table.tex"
        
        # This should not raise NameError for undefined 'key_i'
        generate_latex_dimensions_table(
            sample_dim_data,
            str(output_file),
            "Test Caption",
            "tab:test",
        )
        
        # If we get here without NameError, the bug is fixed
        assert output_file.exists()

    def test_generate_latex_dim2_table_creates_file(self, temp_dir, sample_dim_data):
        """Test that generate_latex_dim2_table creates an output file."""
        output_file = temp_dir / "dim2_table.tex"
        
        generate_latex_dim2_table(
            sample_dim_data,
            str(output_file),
            "Dim2 Caption",
            "tab:dim2",
        )
        
        assert output_file.exists()

    def test_generate_latex_dim2_table_content(self, temp_dir, sample_dim_data):
        """Test that the generated dim2 table contains expected content."""
        output_file = temp_dir / "dim2_table.tex"
        
        generate_latex_dim2_table(
            sample_dim_data,
            str(output_file),
            "Dim2 Caption",
            "tab:dim2",
        )
        
        content = output_file.read_text()
        
        # Check for LaTeX structure
        assert r"\begin{table}" in content
        assert r"\end{table}" in content
        assert r"\caption{Dim2 Caption}" in content
        assert r"\label{tab:dim2}" in content

    def test_generate_latex_dim2_table_rats_ids_populated(self, temp_dir, sample_dim_data):
        """Test that rats_ids dictionary is correctly populated (regression test for key_i bug)."""
        output_file = temp_dir / "dim2_table.tex"
        
        # This should not raise NameError for undefined 'key_i'
        generate_latex_dim2_table(
            sample_dim_data,
            str(output_file),
            "Test Caption",
            "tab:test",
        )
        
        # If we get here without NameError, the bug is fixed
        assert output_file.exists()

    def test_generate_latex_dim2_table_dtao_row_color(self, temp_dir):
        """Test that dtao sections get row color applied when data structure is correct."""
        output_file = temp_dir / "dim2_table.tex"
        
        # Create data structure with segment dicts first (as expected by the function)
        dim2_data = {
            "rat-01": {
                "Tr": {  # Tr -> dtao, should get row color
                    "A": {"len": 1.0},
                    "B": {"len": 1.5},
                    "thick": 0.1234,
                    "dia": 2.5678,
                },
            },
        }
        
        generate_latex_dim2_table(
            dim2_data,
            str(output_file),
            "Test Caption",
            "tab:test",
        )
        
        content = output_file.read_text()
        
        # Check that row color is applied for dtao sections
        assert r"\rowcolor{gray!10}" in content

    def test_generate_latex_material_props_table_io_error_handling(self, sample_mat_param):
        """Test that IOError is handled gracefully."""
        # Try to write to an invalid path
        invalid_path = "/nonexistent_dir/material_table.tex"
        
        # Should not raise, just log error
        generate_latex_material_props_table(
            sample_mat_param,
            invalid_path,
            "Test Caption",
            "tab:test",
        )

    def test_generate_latex_dimensions_table_io_error_handling(self, sample_dim_data):
        """Test that IOError is handled gracefully."""
        invalid_path = "/nonexistent_dir/dimensions_table.tex"
        
        # Should not raise, just log error
        generate_latex_dimensions_table(
            sample_dim_data,
            invalid_path,
            "Test Caption",
            "tab:test",
        )

    def test_generate_latex_dim2_table_io_error_handling(self, sample_dim_data):
        """Test that IOError is handled gracefully."""
        invalid_path = "/nonexistent_dir/dim2_table.tex"
        
        # Should not raise, just log error
        generate_latex_dim2_table(
            sample_dim_data,
            invalid_path,
            "Test Caption",
            "tab:test",
        )

    def test_empty_mat_param_dict(self, temp_dir):
        """Test handling of empty material parameters dictionary."""
        output_file = temp_dir / "empty_table.tex"
        
        generate_latex_material_props_table(
            {},
            str(output_file),
            "Empty Caption",
            "tab:empty",
        )
        
        assert output_file.exists()

    def test_empty_dim_data_dict(self, temp_dir):
        """Test handling of empty dimension data dictionary."""
        output_file = temp_dir / "empty_dim_table.tex"
        
        generate_latex_dimensions_table(
            {},
            str(output_file),
            "Empty Caption",
            "tab:empty",
        )
        
        assert output_file.exists()
