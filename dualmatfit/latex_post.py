# -*- coding: utf-8 -*-
"""
LaTeX table generation for material fitting results.

This module provides functions for generating LaTeX tables from
material fitting results for scientific publications.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict, List, Any, Tuple
from dualmatfit.rato_info import excel_data
from dualmatfit.io_utils import load_excel_params, load_hdf5_data

from dualmatfit.logging_config import get_logger
logger = get_logger('latex')

__all__ = [
    'generate_latex_material_props_table',
    'generate_latex_dimensions_table',
    'generate_latex_dim2_table',
]

sections_keys = dict(Ar="aoa", Tr="dtao", Ab="daao")


def generate_latex_material_props_table(
        mat_param: Dict[str, pd.DataFrame],
        output_filename: str,
        caption: str,
        label: str,
):
    """
    Generates a custom LaTeX table from a pandas DataFrame with specific formatting.

    This function is tailored to create a table with features like:
    - Grouping by a primary column ('Rat N.').
    - A special 'baseline' row for each group.
    - Alternating row colors for data entries.
    - Specific header and column formatting as per the example.

    Args:
        mat_param: Dict[str, pd.DataFrame]: Input DataFrame. It is crucial that the first row for each
                           'Rat N.' group is the 'baseline' data. The DataFrame should
                           contain the columns in the order they appear in the table.
        output_filename (str): The name of the .tex file to be created (e.g., 'material_parameters.tex').
        caption (str):      The caption for the LaTeX table.
        label (str):        The label for cross-referencing the table (e.g., 'tab:mat_fit').
    """

    # LaTeX Preamble for the table ---
    latex_preamble = r"""\begin{table}[htb!]
    \centering
    \caption{""" + caption + r"""}
    \label{""" + label + r"""}
    \resizebox{\textwidth}{!}{
        \begin{tabular}{c|cc|cccccccc}
            \toprule
    """

    # Header Row ---
    header = r"""            \rowcolor{gray!40}
            \textbf{Rat N.} & \textbf{Section} & \textbf{Segment} & $\mu$ [KPa] & $D$ [KPa]& $k_1$ [KPa] & $k_2$ [-] & $\beta$ [rad] & $\beta$ [deg] & $\kappa$ [-] & $\kappa$ [\%] \\
            \midrule \hline
    """

    # --- 3. Table Body ---
    latex_body = ""

    # Group data by 'Rat N.' to handle each block separately
    rats_ids = dict()
    for rat_i, (key_i, mat_params_i) in enumerate(mat_param.items()):
        rat_num_i = rat_i + 1
        rats_ids[key_i] = rat_i + 1

        if rat_i > 0:
            # Add a horizontal line to separate data from different rats
            latex_body += "\t\t\t\\hline\n"

        latex_body += f"\t\t\t% Data for Rat N. {rat_num_i}: {key_i}\n"

        mat_params_i.loc[:, 'alpha_deg'] = np.radians(mat_params_i['alpha'])
        mat_params_i.loc[:, 'kappa_perc'] = mat_params_i['kappa'] * 300.

        # The first row in each group is assumed to be the 'baseline'
        baseline_row_i = mat_params_i.iloc[0, :]

        # Format the baseline row with a dark gray background
        latex_body += "\t\t\t\\rowcolor{gray!30} \n"

        b_values = [
            f"{rat_num_i}", "baseline", "-",
            f"${baseline_row_i['mu']:.4f}$",
            f"${baseline_row_i['bulk']:.4f}$",
            f"${baseline_row_i['k_1']:.4f}$",
            f"${baseline_row_i['k_2']:.4f}$",
            f"${baseline_row_i['alpha']:.4f}$",
            f"${baseline_row_i['alpha_deg']:.4f}$",
            f"${baseline_row_i['kappa']:.4f}$",
            f"${baseline_row_i['kappa_perc']:.4f}$"
        ]
        latex_body += "\t\t\t" + " & ".join(b_values) + " \\\\\n"

        # Add a midrule after the baseline row
        latex_body += "\t\t\t\\midrule\n"
        data_rows_i = mat_params_i.iloc[1:, :]

        # Format the subsequent data rows with alternating light gray color
        for k, (key_k, row_k) in enumerate(data_rows_i.iterrows()):
            # Rows at index 0, 2, 4... (i.e., the 1st, 3rd, 5th data row) get colored
            sp_key_k = key_k.split('-')
            section_k = sections_keys.get(sp_key_k[0], "None")

            if k % 2 == 0:
                latex_body += "\t\t\t\\rowcolor{gray!10}\n"

            if row_k['mu'].item() > 0.:
                d_values = [
                    f"{rat_num_i}", f"\\acrshort{{{section_k}}}", sp_key_k[-1],
                    f"${row_k['mu']:.4f}$",
                    f"${row_k['bulk']:.4f}$",
                    f"${row_k['k_1']:.4f}$",
                    f"${row_k['k_2']:.4f}$",
                    f"${row_k['alpha']:.4f}$",
                    f"${row_k['alpha_deg']:.4f}$",
                    f"${row_k['kappa']:.4f}$",
                    f"${row_k['kappa_perc']:.4f}$"
                ]
            else:
                d_values = [
                    f"{rat_num_i}", f"\\acrshort{{{section_k}}}", sp_key_k[-1],
                    "{-}", "{-}", "{-}", "{-}", "{-}", "{-}", "{-}", "{-}",
                ]

            latex_body += "\t\t\t" + " & ".join(d_values) + " \\\\\n"

        latex_body += r"""			\bottomrule""" + "\n"

    # Table Footer
    latex_footer = r"""
        \end{tabular}
    }
\end{table}"""

    # Combine all parts and write to the output file ---
    full_latex_code = latex_preamble + header + latex_body + latex_footer
    try:
        with open(output_filename, 'w') as f:
            f.write(full_latex_code)
        logger.debug(f"Success: LaTeX table was generated and saved to '{output_filename}'")
    except IOError as e:
        logger.debug(f"Error: Could not write to file '{output_filename}'. Reason: {e}")


def generate_latex_dimensions_table(dim_data: Dict[str, Any],
                                    output_filename: str,
                                    caption: str,
                                    label: str,
                                    ):
    """
    Generates a LaTeX table for animal aortic dimensions from a Dictionary.

    This function applies specific formatting rules:
    - Groups data by rat.
    - Uses custom LaTeX commands for rat identifiers provided via a map.
    - Conditionally applies a background color to rows where the 'Section' is 'dtao'.
    - Formats numbers for use with the siunitx 'S' column type.

    Args:
        dim_data: Input DataFrame containing the dimension data.
        output_filename (str): The name of the .tex file to be created.
        caption (str): The caption for the LaTeX table.
        label (str): The label for cross-referencing the table.

    """
    # LaTeX Preamble for the table
    # The tabular format uses the `siunitx` S column for decimal alignment.
    latex_string = r"""\begin{table}[htb!]
	\centering
	\caption{""" + caption + r"""}
	\label{""" + label + r"""}
	\begin{tabular}{ c c S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] }
		\toprule
"""

    # Header Row
    header = r"""		\rowcolor{gray!40}
		\textbf{Rat} & \textbf{Section} & \textbf{${\tau}^r$} & \textbf{${\phi}^r$} & \multicolumn{3}{c}{\textbf{Segments} width $\lyr$ [mm]} \\
		% \cmidrule(lr){5-7} % A light rule spanning only columns 5 through 7
		% Row 2: Units and sub-titles.
		\rowcolor{gray!40}
		N. & & {[mm]} & {[mm]} & {A} & {B} & {C} \\
		\midrule \hline
"""
    latex_string += header

    # Table Body
    is_first_group = True

    # Group data by 'Rat N.' to handle each block separately
    rats_ids = dict()

    for i, (rat_id_i, rat_data_i) in enumerate(dim_data.items()):
        rat_num_i = i + 1
        rats_ids[rat_id_i] = i + 1

        if not is_first_group:
            # Add a midrule to separate data from different rats
            latex_string += "\t\t\\midrule\n"

        is_first_group = False
        # latex_string += f"\t\t% rato - {rat_id_i}\n"

        # Iterate through each row in the current rat's data group
        list_rows_i = [
            f"\t\t% Data for Rat N. {rat_num_i}: {rat_id_i}\n",
            "\t\t\\multirow{3}{*}" + f"{{{rat_num_i}}} \n"]

        for sec_k, info_k in rat_data_i.items():
            sec_rnm_k = sections_keys.get(sec_k, None)

            if sec_rnm_k is None:
                break

            list_rows_k = [f"& \\acrshort{{{sec_rnm_k}}}",
                           f"{info_k['thick']:.4f}",
                           f"{info_k['dia']:.4f}",
                           ]

            for seg_j in ['A', 'B', 'C']:
                data_j = info_k.get(seg_j, None)
                len_j = " {-} "

                if isinstance(data_j, dict):
                    if data_j.get('len') is not None:
                        len_j = f"{data_j['len']:.4f}"

                list_rows_k.append(len_j)

            # Join the values with '&' and append the LaTeX line ending
            list_rows_i.append("\t\t" + " & ".join(list_rows_k) + " \\\\\n")

        row_str_i = "".join(list_rows_i)

        latex_string += row_str_i

    # --- 4. Table Footer ---
    footer = r"""		\bottomrule
	\end{tabular}
\end{table}
"""
    latex_string += footer

    # --- 5. Write the complete string to the output file ---
    try:
        with open(output_filename, 'w') as f:
            f.write(latex_string)
        logger.debug(f"Success: LaTeX table was generated and saved to '{output_filename}'")
    except IOError as e:
        logger.debug(f"Error: Could not write to file '{output_filename}'. Reason: {e}")


def generate_latex_dim2_table(dim_data: Dict[str, Any],
                            output_filename: str,
                            caption: str,
                            label: str,
                            ):
    """
    Generates a LaTeX table for animal aortic dimensions from a Dictionary.

    This function applies specific formatting rules:
    - Groups data by rat.
    - Uses custom LaTeX commands for rat identifiers provided via a map.
    - Conditionally applies a background color to rows where the 'Section' is 'dtao'.
    - Formats numbers for use with the siunitx 'S' column type.

    Args:
        dim_data: Input DataFrame containing the dimension data.
        output_filename (str): The name of the .tex file to be created.
        caption (str): The caption for the LaTeX table.
        label (str): The label for cross-referencing the table.

    """
    # LaTeX Preamble for the table
    # The tabular format uses the `siunitx` S column for decimal alignment.
    latex_string = r"""\begin{table}[htb!]
	\centering
	\caption{""" + caption + r"""}
	\label{""" + label + r"""}
	\begin{tabular}{c|c|c|S[table-format=1.4] S[table-format=1.4] S[table-format=1.4]}
		\toprule
"""

    # Header Row
    header = r"""		\rowcolor{gray!30}
		\textbf{Rat N.} & \textbf{Section} & \textbf{Segment} & \textbf{${\tau}^r$ [mm]} & \textbf{${\phi}^r$ [mm]} & \textbf{$\lyr$ [mm]} \\
		\midrule
"""
    latex_string += header

    # Table Body
    is_first_group = True

    # Group data by 'Rat N.' to handle each block separately
    rats_ids = dict()

    for i, (rat_id_i, rat_data_i) in enumerate(dim_data.items()):
        rat_num_i = i + 1
        rats_ids[rat_id_i] = i + 1

        if not is_first_group:
            # Add a midrule to separate data from different rats
            latex_string += "\t\t\\midrule\n"

        is_first_group = False
        latex_string += f"\t\t% rat {rat_num_i} - {rat_id_i}\n"

        # Iterate through each row in the current rat's data group
        list_rows_i = []
        for sec_k, info_k in rat_data_i.items():
            sec_rnm_k = sections_keys.get(sec_k, None)

            if sec_rnm_k is None:
                break

            for seg_j, data_j in info_k.items():
                if not isinstance(data_j, dict):
                    break

                # Conditionally add row color if the Section is 'dtao'
                if sec_rnm_k.lower() == 'dtao':
                    list_rows_i.append("\t\t\\rowcolor{gray!10}\n")

                # Format the data for each column into a list of strings
                values_k = [
                    str(rat_num_i),
                    f"\\acrshort{{{sec_rnm_k}}}", seg_j,
                    f"{info_k['thick']:.4f}",
                    f"{info_k['dia']:.4f}",
                    f"{data_j.get('len', None):.4f}"
                ]

                # Join the values with '&' and append the LaTeX line ending
                list_rows_i.append("\t\t" + " & ".join(values_k) + " \\\\\n")

        row_str_i = "".join(list_rows_i)

        latex_string += row_str_i

    # --- 4. Table Footer ---
    footer = r"""		\bottomrule
	\end{tabular}
\end{table}
"""
    latex_string += footer

    # --- 5. Write the complete string to the output file ---
    try:
        with open(output_filename, 'w') as f:
            f.write(latex_string)
        logger.debug(f"Success: LaTeX table was generated and saved to '{output_filename}'")
    except IOError as e:
        logger.debug(f"Error: Could not write to file '{output_filename}'. Reason: {e}")


if __name__ == "__main__":

    script_dir = Path(__file__).resolve().parent.parent
    xlsx_file = (script_dir / "Results/M3-nh-ka-vol-glb/glb_opt_mat_param_ipopt_v01.xlsx").resolve()
    h5_file = (script_dir / 'instron_data' / 'final_data.h5').resolve()

    output_plot_dir = (xlsx_file.parent / "plots").resolve()
    output_plot_dir.mkdir(parents=True, exist_ok=True)

    # ##################################################################################
    # Load Excel and HDF5 files
    df_mat_params = load_excel_params(xlsx_file)
    h5_data = load_hdf5_data(h5_file)

    # ##################################################################################
    table_fpath = (output_plot_dir / "material_table.tex").resolve()
    table_caption = "Estimated material parameters for different segments of the aorta from different rats"
    table_label = "tab:mat_fit"
    
    generate_latex_material_props_table(df_mat_params,
                                        str(table_fpath),
                                        table_caption,
                                        table_label,
                                        )

    # ##################################################################################
    table_fpath = (output_plot_dir / "config_table.tex").resolve()
    table_caption = "List of animals studied, the section of the aorta divided into aortic arch and ascending aorta"
    table_label = "tab:mat_dim"

    info_data = excel_data()

    info_data_slc = {}
    for key_i in df_mat_params.keys():
        key_rp_i = key_i.replace('_', '-')
        info_data_slc[key_rp_i] = info_data[key_rp_i]

    generate_latex_dimensions_table(info_data_slc,
                                    str(table_fpath),
                                    table_caption,
                                    table_label,
                                    )
    test = 1.