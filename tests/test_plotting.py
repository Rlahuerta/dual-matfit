# import os
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import sympy as sy
from scipy.optimize import OptimizeResult
from unittest.mock import patch, MagicMock
from pathlib import Path

from dualmatfit.plotting.experimental_visuals import plot_material_fit, plot_raw_signals
from dualmatfit.data.experimental import InstronData

# --- Test Data Setup ---

# Create sample info_data dictionary
info_data = {
    'tcontrol': (0., 10.),
    'dp': 20.,
    'ds': 2.,
    'sample_id': 'TestRat-Ar-A'
}

# Create dummy experimental data for testing
def create_dummy_instron_data(n_points=100, n_control=10):

    rng = np.random.default_rng()
    np_noise = 0.02 * rng.normal(size=n_points)
    np_time = np.linspace(0., 13., n_points)

    # Create sample pd_data DataFrame
    data = {
        'Time': np_time,
        'Extension': np.linspace(0., 4., n_points),
        'Load': (np.linspace(0., 2.5, n_points) ** 3) + np_noise
    }
    df_data = pd.DataFrame(data)

    return InstronData(df_data, info_data, ncontrol=n_control)

# Create dummy optimization history
def create_dummy_history(n_iter=20, n_params=3):
    history = []
    params = np.random.rand(n_params) * 5
    fun_val = 100.0
    for _ in range(n_iter):
        params = params * (0.95 + np.random.rand(n_params)*0.1) # Simulate parameter changes
        fun_val *= (0.9 + np.random.rand()*0.1) # Simulate function value decrease
        history.append({'x': params.copy(), 'fun': fun_val})

    return history

# Create dummy solution data
def create_dummy_solution(n_points=100):

    np_stretch_x = np.linspace(1.0, 1.5, n_points)
    np_stretch = np.zeros((n_points, 3), dtype=float)

    np_stretch[:, 0] = np_stretch_x
    np_stretch[:, 1] = 1.0 / np.sqrt(np_stretch_x) # Simple incompressible assumption
    np_stretch[:, 2] = (1.0 / (np_stretch[:, 0] * np_stretch[:, 1])) ** 1.5

    np_detJ = np.prod(np_stretch, axis=1)

    np_fint_x = ((np_stretch_x ** 5 - 1.) + np.log(np_detJ) ** 2)
    np_fint = np.zeros_like(np_stretch)
    np_fint[:, 0] = np_fint_x

    np_stress_x = np_fint_x / info_data['ds']
    np_stress_iso_x = (0.1 * np_stress_x.max()) * (np_stretch[:, 0] ** 1.2 - 1.)
    np_stress_vol_x = - np.log(np_detJ) ** 2 / info_data['ds']
    np_stress_ani_x = np_stress_x - np_stress_iso_x - np_stress_vol_x

    np_stress_total = np.zeros_like(np_stretch)
    np_stress_total[:, 0] = np_stress_x

    np_stress_iso = np.zeros_like(np_stretch)
    np_stress_iso[:, 0] = np_stress_iso_x

    np_stress_vol = np.zeros_like(np_stretch)
    np_stress_vol[:, 0] = np_stress_vol_x

    np_stress_ani = np.zeros_like(np_stretch)
    np_stress_ani[:, 0] = np_stress_ani_x

    stress = {"total": np_stress_total,
              "iso": np_stress_iso,
              "vol": np_stress_vol,
              "ani": np_stress_ani,
              }

    # mu, bulk, k1, k2, ang, disp
    xmat_vars = {'mu': 0.1, 'D': 0.05, 'k_1': 0.1, 'k_2': 0.025, 'alpha': 0.3 * np.pi, 'kappa': 0.15}

    solution_res = OptimizeResult(stretch=np_stretch,
                                  detJ=np_detJ,
                                  fint=np_fint,
                                  stress=stress,
                                  x_mat=xmat_vars,
                                  )

    return solution_res

# --- Test Case ---
class TestPlottingFunctions(unittest.TestCase):

    def setUp(self):
        """Set up test data and environment."""
        self.dummy_instron = create_dummy_instron_data()
        # self.dummy_history = create_dummy_history() # Not used in test_plot_raw_signals
        # self.dummy_solution = create_dummy_solution() # Not used in test_plot_raw_signals
        self.test_save_dir = Path("./test_plots_output")
        self.test_save_dir.mkdir(exist_ok=True)

    @patch('dualmatfit.plotting.experimental_visuals.plt.show')  # Patch where show is called
    @patch('dualmatfit.plotting.experimental_visuals.plt.savefig')  # Patch where savefig is called
    @patch('dualmatfit.plotting.experimental_visuals.plt.subplots')  # Patch where subplots is called
    def test_plot_raw_signals(self, mock_subplots, mock_savefig, mock_show):
        """Test plot_raw_signals function."""
        # Arrange: Configure the mock for subplots to return a mock figure and axes
        mock_fig = MagicMock(spec=plt.Figure)
        # The plot_raw_signals function expects 3 axes if plot_cfg is default
        mock_axes_list = [MagicMock(spec=plt.Axes) for _ in range(3)]
        # Ensure mock_subplots returns a tuple (figure, array_of_axes)
        mock_subplots.return_value = (mock_fig, np.array(mock_axes_list))

        # Act
        # plot_raw_signals is imported from experimental_visuals, so its internal PlotHelper
        # will use the updated set_labels_title with fontsize.
        plot_raw_signals(self.dummy_instron, str(self.test_save_dir))

        # Assert
        mock_subplots.assert_called_once()  # Check if subplots was called

        # Default font sizes from PlotHelper.set_labels_title
        xlabel_fontsize = 12
        ylabel_fontsize = 12
        title_fontsize = 14  # As per PlotHelper defaults for titles (though not explicitly checked in original error)

        # Axes 0: Time vs Extension
        self.assertEqual(mock_axes_list[0].plot.call_count, 2)  # Signal + Control points (potentially)
        mock_axes_list[0].set_xlabel.assert_called_with('Time [s]', fontsize=xlabel_fontsize)
        mock_axes_list[0].set_ylabel.assert_called_with('Extension [mm]', fontsize=ylabel_fontsize)
        mock_axes_list[0].set_title.assert_called_with('Time vs Extension', fontsize=title_fontsize)
        mock_axes_list[0].legend.assert_called_once()

        # Axes 1: Time vs Load
        self.assertEqual(mock_axes_list[1].plot.call_count, 2)
        mock_axes_list[1].set_xlabel.assert_called_with('Time [s]', fontsize=xlabel_fontsize)
        mock_axes_list[1].set_ylabel.assert_called_with('Load [N]', fontsize=ylabel_fontsize)
        mock_axes_list[1].set_title.assert_called_with('Time vs Load', fontsize=title_fontsize)
        mock_axes_list[1].legend.assert_called_once()

        # Axes 2: Extension vs Load
        self.assertEqual(mock_axes_list[2].plot.call_count, 2)
        mock_axes_list[2].set_xlabel.assert_called_with('Extension [mm]', fontsize=xlabel_fontsize)
        mock_axes_list[2].set_ylabel.assert_called_with('Load [N]', fontsize=ylabel_fontsize)
        mock_axes_list[2].set_title.assert_called_with('Extension vs Load', fontsize=title_fontsize)
        mock_axes_list[2].legend.assert_called_once()

        mock_fig.savefig.assert_called_once()  # Check if savefig was called on the figure object

        # Check if the filename contains the sample ID
        args, kwargs = mock_fig.savefig.call_args
        save_path = Path(args[0])
        # The filename generated by plot_raw_signals includes the sample_id
        expected_filename_part = self.dummy_instron.get_sample_id().replace('-', '_')
        self.assertTrue(expected_filename_part in save_path.name)
        self.assertTrue("raw_signals.png" in save_path.name)  # Check for suffix

        # Ensure it's saving to the correct directory (resolve paths for robust comparison)
        self.assertEqual(save_path.parent.resolve(), self.test_save_dir.resolve())
        mock_show.assert_not_called()  # Ensure plt.show() wasn't called


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
