import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pathlib import Path
from typing import Union, List
from scipy.stats import f_oneway, ttest_ind

from dualmatfit.plotting.plot_helpers import format_value
from dualmatfit.plotting.analytical_visuals import get_plot_style_for_sample
from dualmatfit.plotting.parameters import NAME_SECTIONS
from dualmatfit.utils.logging_config import get_logger

logger = get_logger('gen_lab_plots')


# --- Data Loading Functions ---
def load_protein_data(filename):
    """Loads data from the specific CSV-like format, handling potential issues."""
    try:
        # Read skipping initial comment/empty lines, use first row as data potentially
        # Need to infer columns based on structure - seems like alternating region/value
        df = pd.read_excel(filename)
        df = df.fillna(0)

        # Assuming the structure is consistent: Row 1 labels regions, Row 2 has data
        # Extract relevant data (adjust indices if format differs slightly)
        arc_cols = [i for i, col in enumerate(df.columns) if "arc" in col.lower()]
        tor_cols = [i for i, col in enumerate(df.columns) if 'toracic' in col.lower()]
        abd_cols = [i for i, col in enumerate(df.columns) if 'abdominal' in col.lower()]

        arc_data = df.iloc[:, arc_cols].values[0, :]
        tor_data = df.iloc[:, tor_cols].values[0, :]
        abd_data = df.iloc[:, abd_cols].values[0, :]

        dict_res = {}
        for key_i, data_i in zip(['AoA', 'DTAo', 'DAAo'], [arc_data, tor_data, abd_data]):
            dict_res[key_i] = data_i[data_i > 0.]

        return dict_res

    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        # Return empty dict or raise error if preferred
        return {'AoA': np.array([]), 'DTAo': np.array([]), 'DAAo': np.array([])}


def perform_statistical_analysis(protein_data, regions):
    """
    Performs ANOVA and post-hoc pairwise t-tests with Bonferroni correction.

    Args:
        protein_data (dict): A dictionary with region names as keys and data arrays as values.
        regions (list): A list of region names to compare.

    Returns:
        tuple: A tuple containing:
            - float: The p-value from the one-way ANOVA.
            - list: A list of tuples for significant pairs, where each tuple is
                    (group1, group2, p_value_uncorrected).
    """
    # 1. Perform one-way ANOVA
    f_stat, p_value_anova = f_oneway(*[protein_data[r] for r in regions])

    logger.info(f"ANOVA p-value: {p_value_anova:.4f}")

    significant_pairs = []

    # 2. If ANOVA is significant, perform pairwise t-tests with Bonferroni correction
    if p_value_anova < 0.05:
        region_pairs = list(itertools.combinations(regions, 2))
        num_comparisons = len(region_pairs)
        bonferroni_alpha = 0.05 / num_comparisons

        logger.info(f"Performing {num_comparisons} pairwise t-tests with Bonferroni correction (alpha = {bonferroni_alpha:.4f})")

        for group1, group2 in region_pairs:
            data1 = protein_data[group1]
            data2 = protein_data[group2]

            # Perform independent t-test (Welch's t-test is default when equal_var=False)
            t_stat, p_value_ttest = ttest_ind(data1, data2, equal_var=False)

            logger.info(f"  {group1} vs {group2}: p-value = {p_value_ttest:.4f}")

            # Check for significance against the corrected alpha
            if p_value_ttest < bonferroni_alpha:
                significant_pairs.append((group1, group2, p_value_ttest))

    return p_value_anova, significant_pairs


def plot_significance_bars(ax, significant_pairs, region_x_map, means_i, stds_i):
    """
    Plots significance bars and asterisks on a given axes object.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        significant_pairs (list): List of tuples for significant pairs.
        region_x_map (dict): Mapping from region name to x-coordinate.
        means_i (list): List of mean values for the bars.
        stds_i (list): List of standard deviation values for the bars.
    """
    if not significant_pairs:
        return

    # 3. Add significance bars to the plot
    y_max = max(np.array(means_i) + np.array(stds_i))
    y_step = y_max * 0.1  # Increment for spacing out significance bars

    for group1, group2, p_value_ttest in significant_pairs:
        # Determine significance level for asterisks based on uncorrected p-value
        if p_value_ttest < 0.001:
            sig_symbol = '***'
        elif p_value_ttest < 0.01:
            sig_symbol = '**'
        else:
            sig_symbol = '*'

        x1, x2 = region_x_map[group1], region_x_map[group2]
        bar_y = y_max + y_step

        ax.plot([x1, x2], [bar_y, bar_y], lw=1.5, c='black')
        ax.text((x1 + x2) / 2, bar_y, sig_symbol, ha='center', va='bottom', color='black', fontsize=14)

        y_max = bar_y  # Update y_max to stack the next bar on top


def plots_protein_quantification():
    """
    Main Script Logic
    """

    file_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(file_path), 'lab_data')
    results_path = os.path.join(os.path.dirname(file_path), 'Results')

    if not os.path.exists(results_path):
        os.makedirs(results_path) # Create Results directory if it doesn't exist

    # Define file paths (assuming they are in the same directory as the script)
    file_elastin = os.path.join(data_path, 'elastina.xlsx')     # Adjusted filename
    file_col1 = os.path.join(data_path, 'col1.xlsx')            # Adjusted filename
    file_col3 = os.path.join(data_path, 'col3.xlsx')            # Adjusted filename

    # Check if files exist
    if not all(os.path.exists(f) for f in [file_elastin, file_col1, file_col3]):
        logger.error("One or more data files not found. Please check filenames/paths.")

    # Prepare data for plotting (using data from summary file for consistency)
    proteins = ['Elastin', 'Collagen I', 'Collagen III']
    regions = ['AoA', 'DTAo', 'DAAo']

    data_elastin = load_protein_data(file_elastin)
    data_col1 = load_protein_data(file_col1)
    data_col3 = load_protein_data(file_col3)

    all_data = {
        'Elastin': data_elastin,
        'Collagen I': data_col1,
        'Collagen III': data_col3
    }

    means = {p: [np.mean(all_data[p][r]) for r in regions] for p in proteins}
    median = {p: [np.median(all_data[p][r]) for r in regions] for p in proteins}
    stds = {p: [np.std(all_data[p][r]) for r in regions] for p in proteins}
    ns_ = {p: [all_data[p][r].size for r in regions] for p in proteins}

    # Data from 'Análise proteínas' summary
    means_summary = {
        'Elastin': means['Elastin'],
        'Collagen I': means['Collagen I'],
        'Collagen III': means['Collagen III'],
    }

    

    stds_summary = {
        'Elastin': stds['Elastin'],
        'Collagen I': stds['Collagen I'],
        'Collagen III': stds['Collagen III'],
    }
    ns_summary = {
        'Elastin': ns_['Elastin'],
        'Collagen I':  ns_['Collagen I'],
        'Collagen III':  ns_['Collagen III'],
    }

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')  # Using a clean style
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False, dpi=700)
    bar_width = 0.6
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Default blue, orange, green
    colors = ['#80b1d3','#fdb462','#b3de69'] # Lighter pastel colors

    # Pre-calculate the maximum y-value across all data points for consistent axis scaling
    max_y_overall = 0
    for p in proteins:
        max_y_overall = max(max_y_overall, max(np.array(means_summary[p]) + np.array(stds_summary[p])))

    dict_p_val = {}

    for i, protein_i in enumerate(proteins):
        ax_i = axes[i]
        x_pos_i = np.arange(len(regions))

        means_i = means_summary[protein_i]
        stds_i = stds_summary[protein_i]
        ns_i = ns_summary[protein_i]

        # Plot bars with error bars (standard deviation)
        ax_i.bar(x_pos_i, means_summary[protein_i], bar_width,
               yerr=stds_summary[protein_i],
               capsize=6, color=colors[i], alpha=0.9,
               edgecolor='black', linewidth=0.5,                        # Add slight edge for definition
               error_kw={"ecolor": 'gray', "lw": 1.5, "capthick": 1.5})      # Customize error bars

        text_offset = max_y_overall * 0.02
        for j, x_j in enumerate(x_pos_i):
            # Position the text above the top of the error bar
            y_position = means_i[j] + stds_i[j] + text_offset
            n_value = ns_i[j]
            ax_i.text(x_j, y_position, f'n = {n_value}',
                    ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')

        # --- STATISTICAL ANALYSIS AND VISUALIZATION (SciPy-only) ---
        logger.info(f"--- {protein_i} ---")
        p_anova_i, significant_pairs = perform_statistical_analysis(all_data[protein_i], regions)
        dict_p_val[protein_i] = p_anova_i

        if significant_pairs:
            region_x_map = {region: i for i, region in enumerate(regions)}
            plot_significance_bars(ax_i, significant_pairs, region_x_map, means_i, stds_i)

        ax_i.set_title(protein_i, fontsize=18, pad=15)
        ax_i.set_ylabel('Relative Protein Amount\n(Arbitrary Units)', fontsize=14, labelpad=10)
        ax_i.set_xticks(x_pos_i)
        ax_i.set_xticklabels(regions, fontsize=14)
        ax_i.tick_params(axis='y', labelsize=12) # Adjust tick label size
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['right'].set_visible(False)
        ax_i.spines['left'].set_linewidth(0.5)
        ax_i.spines['bottom'].set_linewidth(0.5)
        ax_i.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f')) # Format y-ticks
        ax_i.grid(axis='y', linestyle='--', which='major', color='grey', alpha=0.5, linewidth=0.5)
        ax_i.set_axisbelow(True) # Ensure grid is behind bars

    # Adjust y-axis limits individually or set a common one if appropriate
    max_y_overall = 0
    for p in proteins:
        max_y_overall = max(max_y_overall, max(np.array(means_summary[p]) + np.array(stds_summary[p])))

    for ax_i in axes:
         ax_i.set_ylim(bottom=0, top=max_y_overall * 1.1) # Set common limit based on max overall

    plt.suptitle('Relative Protein Distribution Across Aortic Segments', fontsize=22, y=1.05)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout

    # Save the plot
    output_filename = os.path.join(results_path, 'protein_distribution.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved as {output_filename}")


def dataframe2latex_table(
        xlsx_path: Union[str, Path],
        list_rats: List[str] = None,
):
    """
    Generates a LaTeX table from a Pandas DataFrame.
    """

    if isinstance(xlsx_path, (str, Path)):
        if xlsx_path.is_file() is False:
            raise FileNotFoundError(f"File {xlsx_path} does not exist")

        results_path = (xlsx_path.parent / "plots").resolve()
    else:
        raise ValueError('xlsx_path must be str or Path')

    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    # Handling xlsx file
    df_params = pd.read_excel(str(xlsx_path), decimal=',', sheet_name=None, index_col=0)

    if list_rats is None:
        list_rats = list(df_params.keys())

    # --- Table Info ---

    table_caption = "Estimated material parameters for different segments of the aorta from different rats"
    table_label = "tab:mat_fit_py_generated"

    column_precisions = {'default': 4}
    default_precision = column_precisions.get('default', 4)

    # Define data columns and their precisions
    data_cols_info = {
        'mu': column_precisions.get('mu', default_precision),
        'D': column_precisions.get('D', default_precision),
        'k1': column_precisions.get('k1', default_precision),
        'k2': column_precisions.get('k2', default_precision),
        'beta_rad': column_precisions.get('beta_rad', default_precision),
        'beta_deg': column_precisions.get('beta_deg', default_precision),
        'kappa_val': column_precisions.get('kappa_val', default_precision),
        'kappa_pct': column_precisions.get('kappa_pct', default_precision),
    }

    latex_parts = []

    # Table preamble
    latex_parts.append("\\begin{table}[htb!]")
    latex_parts.append("\t\\centering")
    latex_parts.append(f"\t\\caption{{{table_caption}}}")
    latex_parts.append(f"\t\\label{{{table_label}}}")
    latex_parts.append("\t\\resizebox{\\textwidth}{!}{")
    latex_parts.append("\t\t\\begin{tabular}{c|cc|cccccccc}")

    # Table header
    latex_parts.append("\t\t\t\\toprule")
    latex_parts.append("\t\t\t\\rowcolor{gray!40}")
    header_line = ("\t\t\t\\textbf{Rat N.} & \\textbf{Section} & \\textbf{Segment} & "
                   "$\\mu$ [KPa] & $D$ [KPa]& $k_1$ [KPa] & $k_2$ [-] & "
                   "$\\beta$ [rad] & $\\beta$ [deg] & $\\kappa$ [-] & $\\kappa$ [\\%] \\\\")  # Use \\% for the % sign
    latex_parts.append(header_line)
    latex_parts.append("\t\t\t\\midrule \\hline")  # Double line after header

    last_rat_key_processed = None
    for i, (rat_i, df_param_i) in enumerate(df_params.items()):
        if rat_i in list_rats:
            if last_rat_key_processed is not None:
                latex_parts.append("\t\t\t\\hline")

            rat_rs_i = rat_i.replace('_', '-')
            non_baseline_row_color_idx = 0
            style_info_i = get_plot_style_for_sample(rat_rs_i)
            idx_i = style_info_i['label'].split('-', 2)

            for sec_k, mat_params_k in df_param_i.iterrows():
                if mat_params_k.sum() > 0.:
                    if sec_k == "mean":
                        section_disp_k = "baseline"
                        segment_disp_k = "-"
                    else:
                        sec_idx_k = sec_k.split("-", 2)
                        section_disp_k = "\\acrshort{" + NAME_SECTIONS[sec_idx_k[0]].lower() + "}: "
                        segment_disp_k = sec_idx_k[-1]

                    mu_val_k = mat_params_k.get('mu', pd.NA)
                    d_val_k = mat_params_k.get('bulk', pd.NA)  # 'bulk' maps to 'D'
                    k1_val_k = mat_params_k.get('k_1', pd.NA)  # Input col is 'k_1'
                    k2_val_k = mat_params_k.get('k_2', pd.NA)  # Input col is 'k_2'

                    alpha_val_k = mat_params_k.get('alpha', pd.NA)  # 'alpha' maps to 'beta_rad'
                    beta_rad_val_k = alpha_val_k
                    beta_deg_val_k = np.rad2deg(beta_rad_val_k)

                    kappa_single_val_k = mat_params_k.get('kappa', pd.NA)  # 'kappa' maps to 'kappa_val'
                    kappa_val_val_k = kappa_single_val_k
                    kappa_pct_val_k = float(kappa_single_val_k) * 100     # Assumption: kappa is a fraction

                    # Store values for formatting, using output column names as keys
                    current_row_values_k = {
                        'mu': mu_val_k, 'D': d_val_k, 'k1': k1_val_k, 'k2': k2_val_k,
                        'beta_rad': beta_rad_val_k, 'beta_deg': beta_deg_val_k,
                        'kappa_val': kappa_val_val_k, 'kappa_pct': kappa_pct_val_k
                    }

                    # Format numerical values
                    formatted_data_values = []
                    for col_key in ['mu', 'D', 'k1', 'k2', 'beta_rad', 'beta_deg', 'kappa_val', 'kappa_pct']:
                        precision = data_cols_info[col_key]
                        formatted_data_values.append(f"${format_value(current_row_values_k[col_key], precision)}$")

                    data_string = " & ".join(formatted_data_values)

                    # Determine row color
                    row_color_prefix = ""
                    if sec_k == "mean":
                        row_color_prefix = "\t\t\t\\rowcolor{gray!30} "
                        non_baseline_row_color_idx = 0  # Reset for new section group
                    else:
                        if non_baseline_row_color_idx % 2 == 0:
                            row_color_prefix = "\t\t\t\\rowcolor{gray!10} "
                        non_baseline_row_color_idx += 1

                    latex_parts.append(row_color_prefix)
                    line_i = f"\t\t\t{idx_i[-1]} & {section_disp_k} & {segment_disp_k} & {data_string} \\\\"
                    latex_parts.append(line_i)

                    if sec_k == "mean":
                        latex_parts.append("\t\t\t\\midrule")

            last_rat_key_processed = rat_i

    # Table footer
    latex_parts.append("\t\t\t\\bottomrule")
    latex_parts.append("\t\t\\end{tabular}")
    latex_parts.append("\t}")
    latex_parts.append("\\end{table}")

    latex_table = "\n".join(latex_parts)

    output_filename = (results_path / 'material_parameters.tex').resolve()

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        logger.info(f"LaTeX code successfully saved to {output_filename}")
    except IOError:
        logger.error(f"Could not write to file {output_filename}")


def plots_material_params(
        xlsx_path: Union[str, Path],
        list_rats: List[str] = None,
):

    if isinstance(xlsx_path, (str, Path)):
        if xlsx_path.is_file() is False:
            raise FileNotFoundError(f"File {xlsx_path} does not exist")

        results_path = (xlsx_path.parent / "plots").resolve()
    else:
        raise ValueError('xlsx_path must be str or Path')

    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    output_filename = (results_path / 'material_parameters.png').resolve()

    # Handling xlsx file
    df_params = pd.read_excel(str(xlsx_path), decimal=',', sheet_name=None, index_col=0)

    if list_rats is None:
        list_rats = list(df_params.keys())

    # --- Data Extraction from Table 3 ---
    # Consistent order of segments
    all_segments = [
        'AoA-A', 'AoA-B', 'AoA-C',
        'DTAo-A', 'DTAo-B', 'DTAo-C',
        'DAAo-A', 'DAAo-B', 'DAAo-C'
    ]
    n_segments = len(all_segments)
    x_centers = np.arange(n_segments)  # Numerical positions for segments

    params_by_rats = {}
    param_names = ['μ [KPa]', 'D [KPa]', 'k₁ [KPa]', 'k₂ [-]', 'β [degrees]', 'κ [-]']

    # --- Data Extraction from Excel file ---
    for i, (rat_i, df_param_i) in enumerate(df_params.items()):
        if rat_i in list_rats:
            rat_rs_i = rat_i.replace('_', '-')
            df_param_rs_i = df_param_i.iloc[1:, :]

            style_info_i = get_plot_style_for_sample(rat_rs_i)
            mat_params_i = {}

            for k, (mat_key_k, mat_param_k) in enumerate(df_param_rs_i.items()):
                if mat_key_k == 'alpha':
                    np_mat_k = np.rad2deg(mat_param_k.values)
                else:
                    np_mat_k = mat_param_k.values
                mat_params_i[param_names[k]] = np_mat_k.tolist()

            params_by_rats[style_info_i['label']] = mat_params_i

    # Data organized by Rat, then Parameter
    rat_names = list(params_by_rats.keys())

    # --- Create DataFrame ---
    df_data = {}
    for param in param_names:
        for rat in rat_names:
            # Ensure the list has the correct length, padding with NaN if necessary
            param_data = params_by_rats[rat][param]
            if len(param_data) < n_segments:
                param_data.extend([np.nan] * (n_segments - len(param_data)))
            elif len(param_data) > n_segments:
                param_data = param_data[:n_segments]  # Truncate if too long
            df_data[(param, rat)] = param_data

    df = pd.DataFrame(df_data, index=all_segments)

    # --- Calculate Mean and Std Deviation for each parameter across rats ---
    mean_section = []
    std_section = []
    for key_i, val_i in df.iterrows():
        mean_section_i = {}
        std_section_i = {}
        for mat_name_j in param_names:
            sr_mat_val_j = val_i.loc[mat_name_j, :]
            np_chk_val_j = sr_mat_val_j.values > 0.

            mean_section_i[mat_name_j] = sr_mat_val_j[np_chk_val_j].mean()
            std_section_i[mat_name_j] = sr_mat_val_j[np_chk_val_j].std()

        mean_section.append(mean_section_i)
        std_section.append(std_section_i)

    pd_mean_section = pd.DataFrame(mean_section, index=df.index)

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True)
    axes = axes.ravel()
    mean_color = 'black'

    # --- Create Line Plots ---
    for i, param_name_i in enumerate(param_names):
        ax_i = axes[i]

        for k, sec_k in enumerate(df.index):
            sr_mat_val_k = df.loc[sec_k, param_name_i]
            chk_value_k = sr_mat_val_k.values > 0.
            np_mat_val_k = sr_mat_val_k[chk_value_k].values

            try:
                ax_i.violinplot(np_mat_val_k, positions=[x_centers[k]], showmeans=True, showmedians=True)
            except Exception as e:
                logger.warning(f"Mat param {param_name_i}: {e}")

        # Plot Mean Line
        ax_i.plot(x_centers, pd_mean_section[param_name_i].values,
                marker='o', markersize=6, linestyle='-', linewidth=2,
                color=mean_color, alpha=0.25, label='Mean')

        # Labels and Formatting
        ax_i.set_ylabel(param_name_i, fontsize=14, labelpad=10)
        ax_i.tick_params(axis='y', labelsize=11)
        ax_i.yaxis.grid(True, linestyle=':', which='major', color='grey', alpha=0.6, linewidth=0.5)
        ax_i.set_axisbelow(True)

        # Y-axis formatting
        if 'degrees' in param_name_i:
            ax_i.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°'))
        elif '[KPa]' in param_name_i and ('μ' in param_name_i or 'κ' in param_name_i):  # Micro and Kappa likely smaller
            ax_i.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
        elif '[KPa]' in param_name_i:  # D, k1
            ax_i.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        else:  # k2
            ax_i.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        # Spines
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['right'].set_visible(False)
        ax_i.spines['left'].set_linewidth(0.5)
        ax_i.spines['bottom'].set_linewidth(0.5)

    # --- Set common X-axis labels ---
    for ax_i in axes[-3:]:
        ax_i.set_xticks(x_centers)
        ax_i.set_xticklabels(all_segments, rotation=55, ha='right', fontsize=12)
        ax_i.set_xlabel('Aortic Segment', fontsize=14, labelpad=15)

    # Add legend outside the plots
    # fig.legend(rat_names, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=n_rats, fontsize=12, title_fontsize=13)

    # --- Final Adjustments ---
    plt.suptitle('Material Parameter Trend Along the Aorta', fontsize=22, y=1.03)
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])  # Adjust layout

    # --- Save the plot ---
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    logger.info(f"Grouped bar plot saved as {output_filename}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve()
    # xlsx_file = (script_dir.parent / "Results/M3-nh-ka-vol-glb/glb_opt_mat_param_ipopt_v01.xlsx").resolve()
    xlsx_file = (script_dir.parent / "Results/M3-nh-ka-vol-glb/opt_mat_param_ipopt_ko.xlsx").resolve()

    list_rats2plot = None
    # list_rats2plot = []
    # list_rats2plot += ["rato_17", "rato_18", "rato_20", "rato_21", "rato_22", "rato_23"]   # wt
    # list_rats2plot += ["rato_idoso_8", "rato_idoso_9", "rato_idoso_11", "rato_idoso_12"]
    # list_rats2plot += ["rato_wt_184085", "rato_wt_184012", "rato_wt_183964", "rato_wt_183918"]
    plots_material_params(xlsx_path=xlsx_file, list_rats=list_rats2plot)

    # plots_protein_quantification()

    # --- Generate the LaTeX code with material properties ---
    # latex_code = dataframe2latex_table(xlsx_file, list_rats2plot)

