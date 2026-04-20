# -*- coding: utf-8 -*-
import importlib
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import dualmatfit.plotting as plotting


plot_analytical_visuals = importlib.import_module('scripts.plot_analytical_visuals')
plot_experimental_visuals = importlib.import_module('scripts.plot_experimental_visuals')


def test_plotting_package_exports_core_visuals_only() -> None:
    assert 'plot_raw_signals' in plotting.__all__
    assert 'plot_material_fit' in plotting.__all__
    assert 'plot_segment_force_curves' in plotting.__all__
    assert 'plot_curves_from_xlsx' not in plotting.__all__
    assert not hasattr(plotting, 'plot_curves_from_xlsx')


def test_analytical_script_runner_calls_core_helpers(tmp_path, monkeypatch) -> None:
    plot_data = {'Ar': {'sample-a': {'experimental': object(), 'model': object()}}, 'baseline': {}}
    generate_plot_data = Mock(return_value=plot_data)
    plot_force_curves = Mock()
    plot_stress_curves = Mock()
    plot_mean_curves = Mock()

    monkeypatch.setattr(plot_analytical_visuals, 'generate_plot_data_from_xlsx', generate_plot_data)
    monkeypatch.setattr(plot_analytical_visuals, 'plot_segment_force_curves', plot_force_curves)
    monkeypatch.setattr(plot_analytical_visuals, 'plot_segment_stress_curves', plot_stress_curves)
    monkeypatch.setattr(plot_analytical_visuals, 'plot_mean_stress_curves', plot_mean_curves)

    plot_analytical_visuals.plot_curves_from_xlsx(
        h5_path='final_data.h5',
        xlsx_path='params.xlsx',
        output_dir=tmp_path,
        ncontrol=15,
        config_name='cfg-name',
        var_form_cfg={'mix': 3, 'was': True},
    )

    generate_kwargs = generate_plot_data.call_args.kwargs
    assert generate_kwargs['h5_path'] == 'final_data.h5'
    assert generate_kwargs['xlsx_path'] == 'params.xlsx'
    assert generate_kwargs['ncontrol'] == 15
    assert generate_kwargs['list_rats'] is None
    assert generate_kwargs['rerun'] is False
    assert generate_kwargs['var_form_config']['mix'] == 3
    assert generate_kwargs['var_form_config']['was'] is True

    plot_force_curves.assert_called_once_with(
        data_by_region=plot_data,
        output_dir=tmp_path,
        config_name='cfg-name',
    )
    plot_stress_curves.assert_called_once_with(
        data_by_region=plot_data,
        output_dir=tmp_path,
        config_name='cfg-name',
    )
    plot_mean_curves.assert_called_once_with(
        data_by_region=plot_data,
        output_dir=tmp_path,
        config_name='cfg-name',
    )


def test_analytical_script_runner_skips_plotting_without_data(tmp_path, monkeypatch) -> None:
    generate_plot_data = Mock(return_value={})
    plot_force_curves = Mock()
    plot_stress_curves = Mock()
    plot_mean_curves = Mock()

    monkeypatch.setattr(plot_analytical_visuals, 'generate_plot_data_from_xlsx', generate_plot_data)
    monkeypatch.setattr(plot_analytical_visuals, 'plot_segment_force_curves', plot_force_curves)
    monkeypatch.setattr(plot_analytical_visuals, 'plot_segment_stress_curves', plot_stress_curves)
    monkeypatch.setattr(plot_analytical_visuals, 'plot_mean_stress_curves', plot_mean_curves)

    plot_analytical_visuals.plot_curves_from_xlsx(
        h5_path='final_data.h5',
        xlsx_path='params.xlsx',
        output_dir=tmp_path,
        ncontrol=15,
    )

    plot_force_curves.assert_not_called()
    plot_stress_curves.assert_not_called()
    plot_mean_curves.assert_not_called()


def test_experimental_script_article_post_plots_selected_samples(tmp_path, monkeypatch) -> None:
    h5_path = tmp_path / 'final_data.h5'
    df = pd.DataFrame({
        '(Ar-A) Time [s]': [0.0, 1.0, 2.0],
        '(Ar-A) Extension [mm]': [0.0, 0.5, 1.0],
        '(Ar-A) Load [N]': [0.0, 0.4, 0.9],
    })
    df.to_hdf(h5_path, key='rato_demo', mode='w')

    metadata = {
        'rato-demo': {
            'Ar': {
                'A': {'len': 2.0},
                'thick': 0.5,
                'dia': 4.0,
            },
        },
    }
    captured: dict[str, object] = {}
    plot_raw_signals = Mock()

    class FakeInstronData:
        def __init__(self, df_data, info_data, ncontrol):
            captured['df_data'] = df_data.copy()
            captured['info_data'] = info_data.copy()
            captured['ncontrol'] = ncontrol
            self.np_extn = np.array([0.0, 1.0])
            self.np_time = np.array([0.0, 2.0])

    monkeypatch.setattr(plot_experimental_visuals, 'excel_data', lambda: metadata)
    monkeypatch.setattr(plot_experimental_visuals, 'InstronData', FakeInstronData)
    monkeypatch.setattr(plot_experimental_visuals, 'plot_raw_signals', plot_raw_signals)

    output_dir = tmp_path / 'plots'
    plot_experimental_visuals.article_post(
        h5_input_path=h5_path,
        plot_output_root_dir=output_dir,
        rats_ids_to_process=['rato_demo'],
    )

    assert captured['ncontrol'] == 3
    info_data = captured['info_data']
    assert info_data['sample_id'] == 'rato-demo-Ar-A'
    assert info_data['ds'] == pytest.approx(1.0)
    assert info_data['dp'] == pytest.approx(np.pi * 3.75)

    plot_kwargs = plot_raw_signals.call_args.kwargs
    assert plot_kwargs['save_dir'] == str((output_dir / 'rato_demo').resolve())
    assert plot_kwargs['filename_prefix'] == 'rato_demo_Ar_A'
    assert plot_kwargs['xlim_time'] == pytest.approx((0.0, 2.1))
    assert plot_kwargs['xlim_ext'] == pytest.approx((-0.05, 1.05))
