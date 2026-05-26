# -*- coding: utf-8 -*-
"""
Script entrypoint for generating raw experimental signal plots.
"""
import argparse
import re

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from pathlib import Path
from typing import Optional, Sequence

from dualmatfit.data.experimental import InstronData
from dualmatfit.utils.logging_config import get_logger
from dualmatfit.plotting.experimental_visuals import plot_raw_signals
from dualmatfit.data.rato_info import excel_data
from dualmatfit.utils.path_manager import PathManager


logger = get_logger('plotting')

_COLUMN_NAME_PATTERN = re.compile(
    r"\(([A-Za-z]{2})-([A-Za-z0-9])\)\s+(?:Time|Extension|Load)(?:\s*\[.*\])?"
)
_DEFAULT_H5_FILE_NAME = 'final_data.h5'
_DEFAULT_DIRECT_RUN_RATS = ['rato_wt_184041']


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_h5_file_path(h5_input_path: str | Path | None) -> Path | None:
    path_manager = PathManager(base_path=_project_root())
    try:
        return path_manager.resolve_h5_data_path(h5_input_path)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return None


def _resolve_plot_output_dir(plot_output_root_dir: str | Path | None) -> Path:
    if plot_output_root_dir is None:
        return (_project_root() / 'Results' / 'instron_plots').resolve()
    return Path(plot_output_root_dir).resolve()


def article_post(
        h5_input_path: str | Path | None = None,
        plot_output_root_dir: str | Path | None = None,
        rats_ids_to_process: Optional[list[str]] = None,
) -> None:
    """
    Process an HDF5 file and generate raw signal plots for each selected sample.
    """
    h5_file_path = _resolve_h5_file_path(h5_input_path)
    if h5_file_path is None:
        return

    plot_output_dir_base = _resolve_plot_output_dir(plot_output_root_dir)
    plot_output_dir_base.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Plot output base directory: {plot_output_dir_base}")

    dict_info_data = excel_data()
    logger.debug(f"Processing HDF5 file: {h5_file_path}")

    try:
        with pd.HDFStore(str(h5_file_path), mode='r') as h5_store:
            if rats_ids_to_process is None:
                h5_keys_to_process = [key for key in h5_store.keys() if key.lower().startswith('/rato')]
            else:
                h5_keys_to_process = [
                    f"/{rat_id.lower()}"
                    for rat_id in rats_ids_to_process
                    if f"/{rat_id.lower()}" in h5_store.keys()
                ]

            if not h5_keys_to_process:
                logger.debug(
                    f"No keys starting with '/rato' found in {h5_file_path.name}. "
                    "Nothing to process."
                )
                return

            for h5_key_i in h5_keys_to_process:
                logger.debug(f"\n  Processing HDF5 key: {h5_key_i}")
                rat_group_key_i = h5_key_i.lstrip('/').replace('_', '-')

                pd_exp_data_group_i = h5_store[h5_key_i]
                all_columns_in_group_i = pd_exp_data_group_i.columns
                num_columns_i = len(all_columns_in_group_i)

                logger.debug(f"    Output directory for this rat group: {plot_output_dir_base}")

                for col_group_start_k in range(0, num_columns_i, 3):
                    current_col_group_names_k = all_columns_in_group_i[
                        col_group_start_k: col_group_start_k + 3
                    ]
                    if len(current_col_group_names_k) < 3:
                        logger.debug(
                            "    Skipping incomplete column group: "
                            f"{current_col_group_names_k.tolist()}"
                        )
                        continue

                    pd_sample_data_k = pd_exp_data_group_i[current_col_group_names_k].copy()
                    pd_sample_data_k.dropna(inplace=True)

                    if pd_sample_data_k.empty:
                        logger.debug(
                            "    Skipping empty data for column group: "
                            f"{current_col_group_names_k.tolist()}"
                        )
                        continue

                    first_col_name_k = current_col_group_names_k[0]
                    name_parts_match_k = _COLUMN_NAME_PATTERN.match(first_col_name_k)
                    if not name_parts_match_k:
                        logger.debug(
                            "    Warning: Could not parse section/position from column name "
                            f"'{first_col_name_k}' for rat group '{rat_group_key_i}'. "
                            "Skipping this sample."
                        )
                        continue

                    section_key_from_col_k = name_parts_match_k.group(1)
                    position_key_from_col_k = name_parts_match_k.group(2)
                    sample_id_k = (
                        f"{rat_group_key_i}-{section_key_from_col_k}-{position_key_from_col_k}"
                    )
                    logger.debug(f"      Processing sample: {sample_id_k}")

                    if rat_group_key_i not in dict_info_data:
                        logger.debug(
                            "      Warning: Metadata key "
                            f"'{rat_group_key_i}' not found in dict_info_data. "
                            f"Skipping sample {sample_id_k}."
                        )
                        continue

                    current_rat_specific_info_k = dict_info_data[rat_group_key_i]
                    if section_key_from_col_k not in current_rat_specific_info_k:
                        logger.debug(
                            "      Warning: Metadata key "
                            f"'{section_key_from_col_k}' not found for rat "
                            f"'{rat_group_key_i}'. Skipping sample {sample_id_k}."
                        )
                        continue

                    current_section_info_k = current_rat_specific_info_k[section_key_from_col_k]
                    if position_key_from_col_k not in current_section_info_k:
                        logger.debug(
                            "      Warning: Metadata key "
                            f"'{position_key_from_col_k}' not found for section "
                            f"'{section_key_from_col_k}' of rat '{rat_group_key_i}'. "
                            f"Skipping sample {sample_id_k}."
                        )
                        continue

                    current_sample_info_k = current_section_info_k[position_key_from_col_k].copy()
                    current_sample_info_k['sample_id'] = sample_id_k

                    try:
                        length_k = current_sample_info_k['len']
                        thickness_k = current_section_info_k['thick']
                        diameter_k = current_section_info_k['dia']

                        current_sample_info_k['ds'] = length_k * thickness_k
                        current_sample_info_k['dp'] = np.pi * (
                            diameter_k - thickness_k / 2.0
                        )
                    except KeyError as exc:
                        logger.debug(
                            "      Warning: Missing geometric key "
                            f"({exc}) for sample {sample_id_k}. Cannot calculate ds/dp. "
                            "Skipping."
                        )
                        continue
                    except TypeError as exc:
                        logger.debug(
                            "      Warning: TypeError during geometric calculation for "
                            f"sample {sample_id_k} (key: {exc}). Check data types. Skipping."
                        )
                        continue

                    instron_obj_k = InstronData(
                        df_data=pd_sample_data_k,
                        info_data=current_sample_info_k,
                        ncontrol=3,
                    )

                    dt_ext_k = instron_obj_k.np_extn[-1] * 0.05
                    xlim_time_k = (0.0, instron_obj_k.np_time[-1] * 1.05)
                    xlim_ext_k = (
                        instron_obj_k.np_extn[0] - dt_ext_k,
                        instron_obj_k.np_extn[-1] + dt_ext_k,
                    )
                    save_dir_i = (plot_output_dir_base / h5_key_i[1:]).resolve()
                    filename_prefix_k = sample_id_k.replace('-', '_')

                    plot_raw_signals(
                        instron_data=instron_obj_k,
                        xlim_time=xlim_time_k,
                        xlim_ext=xlim_ext_k,
                        save_dir=str(save_dir_i),
                        filename_prefix=filename_prefix_k,
                    )
    except FileNotFoundError:
        logger.debug(f"Error: HDF5 data file not found after checks. Path was: {h5_file_path}")
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug(f"An error occurred while processing HDF5 data: {exc}")

    logger.debug("\narticle_post processing complete.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--h5-input-path',
        help=(
            "Path to the HDF5 file or a directory containing "
            f"'{_DEFAULT_H5_FILE_NAME}'. If omitted, the script uses the "
            "repository-local dataset only when it is present."
        ),
    )
    parser.add_argument(
        '--plot-output-root-dir',
        help='Root directory used to store the generated raw signal plots.',
    )
    parser.add_argument(
        '--all-rats',
        action='store_true',
        help='Process every /rato* group in the HDF5 file.',
    )
    parser.add_argument(
        '--rats-ids-to-process',
        nargs='*',
        help='Specific rat ids to process (for example: rato_1 rato_2).',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    rats_to_process = None if args.all_rats else (
        args.rats_ids_to_process or _DEFAULT_DIRECT_RUN_RATS
    )
    article_post(
        h5_input_path=args.h5_input_path,
        plot_output_root_dir=args.plot_output_root_dir,
        rats_ids_to_process=rats_to_process,
    )


if __name__ == '__main__':
    main()