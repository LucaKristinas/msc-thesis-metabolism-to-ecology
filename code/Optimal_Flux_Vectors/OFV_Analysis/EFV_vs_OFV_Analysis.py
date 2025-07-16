# ════════════════════════════════════════════════════════════════
# 1. Import Packages
# ════════════════════════════════════════════════════════════════

# Standard Libraries
import sys
from pathlib import Path

# Data Handling
import pandas as pd
import numpy as np

# COBRApy (Constraint-Based Reconstruction and Analysis)
import cobra
from cobra import Reaction
from cobra.io import read_sbml_model

# Plotting and Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from matplotlib_venn import venn2, venn3

# Add Repo Root to sys.path 
project_root = Path(__file__).resolve().parents[3]  
sys.path.append(str(project_root))

# Import Helper Functions
from src.utils import (read_rfile, read_mfile)

# ════════════════════════════════════════════════════════════════
# 2. Import & Export Paths
# ════════════════════════════════════════════════════════════════

# Processed Data 
processed_data_path = project_root / "data" / "processed"
csv_processed_path = processed_data_path / "csv_files"

# Output
output_path = processed_data_path / "figures"
txt_path = processed_data_path / "txt_files"

# ════════════════════════════════════════════════════════════════
# 3. Data Import
# ════════════════════════════════════════════════════════════════

print("Importing data...")

# Define filenames and corresponding variable names
csv_files = {
    "EFV_paths_full": "EFVs_paths.csv",
    "EFV_int_S": "EFVs_intS.csv",
    "EFV_ext_S": "EFVs_extS.csv",
    
    "OFV_paths_full": "OFVs_paths.csv",
    "OFV_int_S": "OFVs_intS.csv",
    "OFV_ext_S": "OFVs_extS.csv",
    
    "EFMs_paths_M": "Martino_EFMs_paths.csv",
    "EFVs_paths_M": "Martino_EFVs_paths.csv",
    
    "EFMs_paths_MN": "Martino_EFMs_paths_scaled.csv",
    "EFVs_paths_MN": "Martino_EFVs_paths_scaled.csv"
}

# Load all CSVs into a dictionary
csv_data = {
    name: pd.read_csv(csv_processed_path / filename, index_col=0)
    for name, filename in csv_files.items()
}

# Unpack into variables
EFV_paths_full = csv_data["EFV_paths_full"]
EFV_int_S = csv_data["EFV_int_S"]
EFV_ext_S = csv_data["EFV_ext_S"]
OFV_paths_full = csv_data["OFV_paths_full"]
OFV_int_S = csv_data["OFV_int_S"]
OFV_ext_S = csv_data["OFV_ext_S"]
EFMs_paths_M = csv_data["EFMs_paths_M"]
EFVs_paths_M = csv_data["EFVs_paths_M"]
EFMs_paths_MN = csv_data["EFMs_paths_MN"]
EFVs_paths_MN = csv_data["EFVs_paths_MN"]

print(" Done! ✅")

# ════════════════════════════════════════════════════════════════
# 4. Analysis of EFVs vs EFMs for Ideal Microbes Benchmarking
# ════════════════════════════════════════════════════════════════

print("Analysis of EFVs vs EFMs for Ideal Microbes Benchmarking...")

#--------------------------------------------
# Define Data Basics
#--------------------------------------------

# Columns of interest
glc_col = "EX_glc(e)"
biomass_col = "Biomass_Ecoli_core"

# Labels and titles
df_labels = ['EFMs', 'EFVs']
plot_titles = ["EFMs", "EFVs"]

# dataframe lists
martino_df_list = [EFMs_paths_M,EFVs_paths_M] # Raw Data (not normalised)

martino_df_list_2 = [] # Data normalised to have max. EX_glc(e) == 1

for i, df in enumerate(martino_df_list):
    max_glc = df[glc_col].max()           # max of glucose import
    max_biomass = df[biomass_col].max()   # max of biomass before normalization

    # Normalize the full DataFrame by glc_col max
    normalized_df = df / max_glc
    martino_df_list_2.append(normalized_df)

martino_df_list_3 = [EFMs_paths_MN, EFVs_paths_MN] # Data normalised to have all EX_glc(e) == 1

#--------------------------------------------
# Biomass Flux Distribution 
#--------------------------------------------

print(" Generating Biomass flux distribution...")

# Combine all sets of DataFrames into one list
all_martino_lists = [
    martino_df_list,     # Panel 1
    martino_df_list_2,   # Panel 2
    [EFMs_paths_MN, EFVs_paths_MN]  # Panel 3
]

# Flatten the list and track subplot titles
flattened_df_list = [df for sublist in all_martino_lists for df in sublist]
panel_titles = [
    "Raw Flux Data (Unscaled)",
    "Globally Scaled Data (Max Glc Import = 1)",
    "Row-Wise Scaled Data (Glc Import per Path = 1)"
]

# Create 3 rows (one per panel), 2 columns (EFMs and EFVs)
fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharey=True)

# Plot each histogram
for idx, (ax, df) in enumerate(zip(axs.flat, flattened_df_list)):
    data = df[biomass_col].dropna()
    
    # Define bins
    bins = np.linspace(data.min(), data.max(), 100)
    weights = np.ones_like(data) * 100 / len(data)

    # Plot
    ax.hist(
        data, bins=bins, weights=weights,
        edgecolor='white', facecolor='black', linewidth=0.8
    )

    # Titles
    row = idx // 2
    if idx % 2 == 0:  # Left column: EFMs
        ax.set_ylabel("Percentage of Entries (%)", fontsize=10)
        ax.set_title(f"{panel_titles[row]} – EFMs", fontsize=10)
    else:  # Right column: EFVs
        ax.set_title(f"{panel_titles[row]} – EFVs", fontsize=10)

    ax.set_xlabel("Biomass Flux (Biomass_Ecoli_core)", fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    # Styling
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)

# Final layout adjustment
plt.subplots_adjust(hspace=0.4)

# Save full figure
fig.savefig(output_path / "EFV_EFM_Glc_biomass_histograms.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "EFV_EFM_Glc_biomass_histograms.svg", bbox_inches='tight')

# Save each row as its own figure
for i in range(3):
    fig_row, ax_row = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
    
    for j in range(2):
        df = all_martino_lists[i][j]
        data = df[biomass_col].dropna()
        bins = np.linspace(data.min(), data.max(), 100)
        weights = np.ones_like(data) * 100 / len(data)

        ax = ax_row[j]
        ax.hist(
            data, bins=bins, weights=weights,
            edgecolor='white', facecolor='black', linewidth=0.8
        )

        # Titles and labels
        ax.set_title(f"{panel_titles[i]} – {'EFMs' if j == 0 else 'EFVs'}", fontsize=10)
        ax.set_xlabel("Biomass Flux (Biomass_Ecoli_core)", fontsize=10)
        if j == 0:
            ax.set_ylabel("Percentage of Entries (%)", fontsize=10)

        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(0.8)
        ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)

    fig_row.tight_layout()
    fig_row.savefig(output_path / f"EFV_EFM_Glc_biomass_histograms_{i+1}.png", dpi=300, bbox_inches='tight')
    fig_row.savefig(output_path / f"EFV_EFM_Glc_biomass_histograms_{i+1}.svg", bbox_inches='tight')
    plt.close(fig_row)  # Close to avoid overlapping plots

# Show full plot
plt.close()

#--------------------------------------------
# Glucose Import Distribution 
#--------------------------------------------

print(" Generating Glucose Import distribution...")

# Define panel titles
panel_titles = [
    "Raw Flux Data (Unscaled)",
    "Globally Scaled Data (Max Glc Import = 1)"
]

# Define the two sets of DataFrames (EFMs and EFVs for each panel)
all_glc_dfs = [
    martino_df_list,      # Panel 1
    martino_df_list_2     # Panel 2
]

# Create a combined figure: 2 rows (panels), 2 columns (EFMs, EFVs)
fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharey='row')

# Plot all panels
for panel_idx, (row_axes, df_pair) in enumerate(zip(axs, all_glc_dfs)):
    for col_idx, (ax, df) in enumerate(zip(row_axes, df_pair)):
        data = df[glc_col].dropna()
        weights_col = np.ones(len(data)) * 100 / len(data)

        # Choose bin strategy
        if panel_idx == 0 and col_idx == 0:  # Raw EFMs – use log scale
            min_val = data[data > 0].min()
            max_val = data.max()
            bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)
            ax.set_xscale('log')
        elif panel_idx == 1:  # Scaled data
            bins = np.linspace(0, 1, 101)
        else:
            bins = np.linspace(0, 10, 100)

        # Plot
        ax.hist(
            data, bins=bins, weights=weights_col,
            edgecolor='white', facecolor='black', linewidth=0.8
        )

        # Titles and labels
        type_label = "EFMs" if col_idx == 0 else "EFVs"
        ax.set_title(f"{panel_titles[panel_idx]} – {type_label}", fontsize=10)
        ax.set_xlabel("Glucose Import (EX_glc(e))", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("Percentage of Entries (%)", fontsize=10)

        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(0.8)
        ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)

# Final layout
plt.subplots_adjust(hspace=0.4)

# Save full figure
fig.savefig(output_path / "EFV_EFM_Glc_import_histograms_combined.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "EFV_EFM_Glc_import_histograms_combined.svg", bbox_inches='tight')

# Save each row (panel) separately
for i, df_pair in enumerate(all_glc_dfs):
    fig_row, axs_row = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    
    for j, (ax, df) in enumerate(zip(axs_row, df_pair)):
        data = df[glc_col].dropna()
        weights_col = np.ones(len(data)) * 100 / len(data)

        # Choose bin strategy
        if i == 0 and j == 0:  # Raw EFMs
            min_val = data[data > 0].min()
            max_val = data.max()
            bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)
            ax.set_xscale('log')
        elif i == 1:
            bins = np.linspace(0, 1, 101)
        else:
            bins = np.linspace(0, 10, 100)

        ax.hist(
            data, bins=bins, weights=weights_col,
            edgecolor='white', facecolor='black', linewidth=0.8
        )

        # Labels and titles
        type_label = "EFMs" if j == 0 else "EFVs"
        ax.set_title(f"{panel_titles[i]} – {type_label}", fontsize=10)
        ax.set_xlabel("Glucose Import (EX_glc(e))", fontsize=10)
        if j == 0:
            ax.set_ylabel("Percentage of Entries (%)", fontsize=10)

        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(0.8)
        ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)

    fig_row.tight_layout()
    fig_row.savefig(output_path / f"EFV_EFM_Glc_import_histograms_panel_{i+1}.png", dpi=300, bbox_inches='tight')
    fig_row.savefig(output_path / f"EFV_EFM_Glc_import_histograms_panel_{i+1}.svg", bbox_inches='tight')
    plt.close(fig_row)

# Show combined plot
plt.close()

#--------------------------------------------
# Glucose Import vs Biomass Distribution 
#--------------------------------------------

print(" Generating Glucose Import vs Biomass flux distribution...")

# Panel metadata
panel_titles = [
    "Raw Flux Data (Unscaled)",
    "Globally Scaled Data (Max Glc Import = 1)"
]

all_data_sources = [
    martino_df_list,     # Panel 1
    martino_df_list_2    # Panel 2
]

col_x = biomass_col
col_y = glc_col

# Combined figure: 2 rows, 2 columns
fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

scatters = []

for panel_idx, (row_axes, data_sources) in enumerate(zip(axs, all_data_sources)):
    for idx, (ax, df) in enumerate(zip(row_axes, data_sources)):
        rounded = df[[col_x, col_y]].dropna().round(2)
        counts = rounded.groupby([col_x, col_y]).size().reset_index(name="count")

        scatter = ax.scatter(
            counts[col_x],
            counts[col_y],
            c=counts["count"],
            cmap="rocket_r",
            s=20,
            edgecolor="white",
            linewidth=0.5
        )
        scatters.append(scatter)

        # Axis configuration
        if panel_idx == 0:  # Raw data
            if idx == 0:  # EFMs
                ax.set_yscale("log")
                ax.set_ylim(1, 1e9)
                ax.set_ylabel("Glucose Import", fontsize=10)
                ax.margins(x=0.05)
            else:  # EFVs
                ax.set_ylim(0, 10.5)
                ax.set_ylabel("")
                ax.set_xlim(-0.1, 1.0)
        else:  # Scaled data
            ax.set_ylim(0, 1.1)
            ax.set_xlim((-0.001, 0.03) if idx == 0 else (-0.01, 0.1))
            ax.set_ylabel("Glucose Import", fontsize=10)

        ax.set_xlabel("Biomass Flux", fontsize=10)
        type_label = "EFMs" if idx == 0 else "EFVs"
        ax.set_title(f"{panel_titles[panel_idx]} – {type_label}", fontsize=10)

        sns.despine(ax=ax)
        ax.tick_params(width=0.8, length=4, direction="out", labelsize=10)

# Add shared colorbar to the right
cbar = fig.colorbar(scatters[1], ax=axs, location="right", shrink=0.85, pad=0.03)
cbar.set_label("Frequency", fontsize=10)
cbar.ax.tick_params(labelsize=10)

# Save full plot
fig.savefig(output_path / "EFV_EFM_Glc_vs_biomass_scatter_combined.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "EFV_EFM_Glc_vs_biomass_scatter_combined.svg", bbox_inches='tight')

# Save each panel (row) separately
for i, data_sources in enumerate(all_data_sources):
    fig_row, axs_row = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    scatters_row = []

    for j, (ax, df) in enumerate(zip(axs_row, data_sources)):
        rounded = df[[col_x, col_y]].dropna().round(2)
        counts = rounded.groupby([col_x, col_y]).size().reset_index(name="count")

        scatter = ax.scatter(
            counts[col_x],
            counts[col_y],
            c=counts["count"],
            cmap="rocket_r",
            s=20,
            edgecolor="white",
            linewidth=0.5
        )
        scatters_row.append(scatter)

        if i == 0:
            if j == 0:
                ax.set_yscale("log")
                ax.set_ylim(1, 1e9)
                ax.set_ylabel("Glucose Import", fontsize=10)
                ax.margins(x=0.05)
            else:
                ax.set_ylim(0, 10.5)
                ax.set_ylabel("")
                ax.set_xlim(-0.1, 1.0)
        else:
            ax.set_ylim(0, 1.1)
            ax.set_xlim((-0.001, 0.03) if j == 0 else (-0.01, 0.1))
            ax.set_ylabel("Glucose Import", fontsize=10)

        ax.set_xlabel("Biomass Flux", fontsize=10)
        type_label = "EFMs" if j == 0 else "EFVs"
        ax.set_title(f"{panel_titles[i]} – {type_label}", fontsize=10)

        sns.despine(ax=ax)
        ax.tick_params(width=0.8, length=4, direction="out", labelsize=10)

    # Shared colorbar for row
    cbar = fig_row.colorbar(scatters_row[1], ax=axs_row, location="right", shrink=0.85, pad=0.03)
    cbar.set_label("Frequency", fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    # Save individual panel
    fig_row.savefig(output_path / f"EFV_EFM_Glc_vs_biomass_scatter_panel_{i+1}.png", dpi=300, bbox_inches='tight')
    fig_row.savefig(output_path / f"EFV_EFM_Glc_vs_biomass_scatter_panel_{i+1}.svg", bbox_inches='tight')
    plt.close(fig_row)

# Close full plot
plt.close()

print(" Done! ✅")

# ══════════════════════════════════════════════════════════════════
# 5. Analysis of EFVs vs OFVs for E. coli core simulations (Glc/Frc)
# ══════════════════════════════════════════════════════════════════

print("Analysis of EFVs vs OFVs for E. coli core simulations (Glc/Frc)...")

#--------------------------------------------
# Define Data Basics
#--------------------------------------------

# Define the columns of interest
biomass_col = "BIOMASS_Ecoli_core_w_GAM"
fru_col = "EX_fru_e_rev"
glc_col = "EX_glc__D_e_rev"
columns = [fru_col,glc_col]

# Create copies with last row removed (protein reaction)
EFV_paths = EFV_paths_full.iloc[:-1].copy()
OFV_paths = OFV_paths_full.iloc[:-1].copy()

# Combine into list
df_list = [OFV_paths, EFV_paths]
df_labels = ['OFVs', 'EFVs']

# Set Seaborn style
sns.set_style("white")

#--------------------------------------------------------------
# Glucose Import vs Fructose Import vs Biomass Distribution 
#--------------------------------------------------------------

print(" Generating Glucose Import vs Fructose Import vs Biomass Distribution...")

### generate data table capturing the distribution ###

# Define bins: fine near 0, coarser near 1
small_bins = np.logspace(-6, -2, num=5).tolist()
coarse_bins = list(np.round(np.linspace(0.1, 1.0, 10), 2))
value_bins = [0] + small_bins + coarse_bins

# Collect output text
output_lines = []

# Binned Value Counts + Sanity Checks
for col in columns:
    output_lines.append(f"\n--- Binned Value Counts for '{col}' ---")
    
    for i, df in enumerate(df_list):
        label = df_labels[i]
        output_lines.append(f"\n{label}:")

        series = df[col].dropna()
        binned = pd.cut(series, bins=value_bins, include_lowest=True, right=True)
        bin_counts = series.groupby(binned, observed=True).count()

        for interval, count in bin_counts.items():
            output_lines.append(f"{str(interval).ljust(20)} : {count}")

        # Sanity Check 1
        total_in_bins = bin_counts.sum()
        assert total_in_bins == len(series), (
            f"Sanity check failed for {col} in {label}: "
            f"sum of bins = {total_in_bins}, but non-NaN rows = {len(series)}"
        )
        output_lines.append(f"> Sanity check passed: {total_in_bins} values binned (out of {len(series)})")

        # Sanity Check 2
        count_exact_1 = (df[col] == 1.0).sum()
        output_lines.append(f"> Entries exactly equal to 1.0: {count_exact_1}")

        # Sanity Check 3
        count_exact_0 = (df[col] == 0.0).sum()
        output_lines.append(f"> Entries exactly equal to 0.0: {count_exact_0}")

# Co-Occurrence Analysis 
for i, df in enumerate(df_list, start=1):
    output_lines.append(f"\n--- Co-occurrence Analysis for df{i} ---")

    condition = pd.DataFrame({
        'fru_pos': df[fru_col] > 0,
        'glc_pos': df[glc_col] > 0
    })

    counts = condition.value_counts().rename_axis(['fru>0', 'glc>0']).reset_index(name='count')

    # Convert DataFrame rows to lines
    for _, row in counts.iterrows():
        line = f"fru>0: {row['fru>0']} | glc>0: {row['glc>0']} → count: {row['count']}"
        output_lines.append(line)

# Save all collected output to a text file 
with open(txt_path / "EFV_OFV_Glc_Frc_import_distribution_summary.txt", "w") as f:
    f.write("\n".join(output_lines))

### plot distribution of frc and glc import as histogram for EFVs ###

efv_df = df_list[1]
fine_bins = np.linspace(0, 1, 101)
plot_titles = ["Fructose Import", "Glucose Import"]

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for idx, (ax, col) in enumerate(zip(axs, columns)):
    data = efv_df[col].dropna()
    weights_col = np.ones(len(data)) * 100 / len(data)

    ax.hist(
        data,
        bins=fine_bins,
        weights=weights_col,
        edgecolor='white',
        facecolor='black',
        linewidth=0.8
    )

    ax.set_title(plot_titles[idx], fontsize=10)
    ax.set_xlabel("", fontsize=10)
    if idx == 0:
        ax.set_ylabel('Percentage of Entries (%)', fontsize=10)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)

plt.tight_layout()

# Save plots
fig.savefig(output_path / "EFV_Glc_Frc_import_distribution.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "EFV_Glc_Frc_import_distribution.svg", bbox_inches='tight')
plt.close()

### plot distribution of frc; glc and biomass as heatmap for EFVs ###

# Define plot configs: (x_col, y_col, xlabel, ylabel, xlim, ylim)
plot_configs = [
    (fru_col, glc_col, "Fructose Import", "Glucose Import", (-0.1, 1.1), (-0.1, 1.1)),       # Panel A
    (fru_col, biomass_col, "Fructose Import", "Biomass Production", (-0.1, 1.1), (-0.01, 0.1)), # Panel B
    (glc_col, biomass_col, "Glucose Import", "Biomass Production", (-0.1, 1.1), (-0.01, 0.1))  # Panel C
]

fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

for i, (ax, (x_col, y_col, xlabel, ylabel, xlim, ylim)) in enumerate(zip(axs, plot_configs)):
    # Round and count frequencies
    rounded = EFV_paths[[x_col, y_col]].dropna().copy()

    # Special case for panel A (index 0): round both to 1 decimal
    if i == 0:
        rounded[x_col] = rounded[x_col].round(1)
        rounded[y_col] = rounded[y_col].round(1)
    else:
        rounded[x_col] = rounded[x_col].round(1)
        rounded[y_col] = rounded[y_col].round(2)

    counts = rounded.groupby([x_col, y_col]).size().reset_index(name='count')

    # Pivot into 2D grid for heatmap
    pivot = counts.pivot(index=y_col, columns=x_col, values='count').fillna(0)

    # Plot heatmap
    sns.heatmap(
        pivot,
        ax=ax,
        cmap='rocket_r',
        cbar=False,  # Disable individual colorbars
        linewidths=0.1
    )
    ax.invert_yaxis()  # Flip it so y increases upward
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)
    #sns.despine(ax=ax)
    # Fully despine all four sides
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)


# Add a single shared colorbar
mappable = axs[-1].collections[0]
cbar = fig.colorbar(mappable, ax=axs, location='right', shrink=0.8, pad=0.02)
cbar.set_label("Frequency", fontsize=10)
cbar.ax.tick_params(labelsize=10)

# save plot
fig.savefig(output_path / "EFV_Glc_Frc_Biomass_Heatmap.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "EFV_Glc_Frc_Biomass_Heatmap.svg", bbox_inches='tight')
plt.close()

#--------------------------------------------------------------
# Biomass Statistics (Glc/ Frc)
#--------------------------------------------------------------

print(" Generating Biomass Statistics (Glc/ Frc)...")

# Settings
sns.set(style="white")
bins = np.arange(0, 0.0901, 0.0025)  # Bins from 0 to 0.09 in 0.0025 steps

### generate data table capturing the distribution ###

# Store all summary text lines
summary_lines = []

# Loop through each DataFrame
for idx, df in enumerate(df_list):
    biomass_values = df[biomass_col].dropna().values  # Drop NaNs

    # Zero and max stats
    num_zero = np.sum(biomass_values == 0)
    max_val = np.max(biomass_values)
    num_max = np.sum(biomass_values == max_val)

    # Frequency table using fixed bins
    binned = pd.cut(biomass_values, bins, include_lowest=True)
    table = binned.value_counts().sort_index()

    # Build output
    summary_lines.append(f"\n=== DataFrame {idx + 1} ===")
    summary_lines.append("Biomass Value Distribution Table:")
    for interval, count in table.items():
        summary_lines.append(f"{str(interval).ljust(20)} : {count}")
    summary_lines.append(f"\nNumber of zeros: {num_zero}")
    summary_lines.append(f"Max biomass value: {max_val}")
    summary_lines.append(f"Number of values equal to max: {num_max}")

### plot distribution of biomass as histogram for EFVs ###

    # Plot only for DataFrame 2 (index 1)
    if idx == 1:
        summary_lines.append(f"\nPlotting DF {idx + 1}: Min = {biomass_values.min()}, Max = {max_val}, #Bins = {len(bins)}")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            biomass_values,
            stat="percent",
            bins=bins,
            edgecolor='white',
            facecolor='black',
            linewidth=0.8,
            ax=ax
        )

        # Style and labels
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(0.8)
        ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)

        ax.set_title("", fontsize=10)
        ax.set_xlabel("Biomass Value", fontsize=10)
        ax.set_ylabel("Percentage", fontsize=10)

        plt.tight_layout()

        # Save plot
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / "EFV_Glc_Frc_biomass_distribution.png", dpi=300, bbox_inches='tight')
        fig.savefig(output_path / "EFV_Glc_Frc_biomass_distribution.svg", bbox_inches='tight')
        plt.close(fig)

# Write all summary text to file
txt_path.mkdir(parents=True, exist_ok=True)
with open(txt_path / "EFV_Glc_Frc_biomass_distribution_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

#--------------------------------------------------------------
# Path Length Statistics (Glc/ Frc)
#--------------------------------------------------------------

print(" Path Length Statistics (Glc/ Frc)...")

# Considered zero threshold
threshold = 1e-3

# Store output text
summary_lines = []

# Loop through each DataFrame
for idx, df in enumerate(df_list):
    lengths_raw = []
    lengths_thresh = []

    for i in range(len(df)):
        row = df.iloc[i]
        length_raw = (row > 0).sum()
        length_thresh = (row > threshold).sum()
        lengths_raw.append(length_raw)
        lengths_thresh.append(length_thresh)

    # Store statistics 
    summary_lines.append(f"\n=== DataFrame {idx + 1} ===")
    summary_lines.append(f"Average pathway length (raw): {np.mean(lengths_raw):.2f}")
    summary_lines.append(f"Average pathway length (threshold > {threshold}): {np.mean(lengths_thresh):.2f}")

    summary_lines.append("\nPathway Length Table (raw):")
    raw_counts = pd.Series(lengths_raw).value_counts().sort_index()
    for val, count in raw_counts.items():
        summary_lines.append(f"  Length {val}: {count}")

    summary_lines.append(f"\nPathway Length Table (threshold > {threshold}):")
    thresh_counts = pd.Series(lengths_thresh).value_counts().sort_index()
    for val, count in thresh_counts.items():
        summary_lines.append(f"  Length {val}: {count}")

    # Plot only for DataFrame 2 (index 1) 
    if idx == 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            lengths_thresh,
            bins=range(min(lengths_thresh), max(lengths_thresh) + 2),
            edgecolor='white',
            facecolor='black',
            linewidth=0.8,
            stat="percent",
            ax=ax
        )

        # Style
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(0.8)
        ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)

        ax.set_title("", fontsize=10)
        ax.set_xlabel("EFV Length", fontsize=10)
        ax.set_ylabel("Percentage", fontsize=10)

        plt.tight_layout()

        # Save plot
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / "EFV_Glc_Frc_pathlength_distribution.png", dpi=300, bbox_inches='tight')
        fig.savefig(output_path / "EFV_Glc_Frc_pathlength_distribution.svg", bbox_inches='tight')
        plt.close(fig)

# Save text summary 
with open(txt_path / "EFV_Glc_Frc_pathlength_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

#--------------------------------------------------------------
# Optimal Pathways Statistics (Glc/ Frc)
#--------------------------------------------------------------

print(" Generating Optimal Pathways Statistics (Glc/ Frc)...")

# Initialize output lines
output_lines = []
threshold = 1e-3

### General Optimality Stats ###

for idx, df in enumerate(df_list):
    output_lines.append(f"\n=== DataFrame {idx + 1} ===")

    max_biomass = df[biomass_col].max()
    max_rows = df[df[biomass_col] == max_biomass]

    output_lines.append(f"Max biomass value: {max_biomass}")
    output_lines.append(f"Number of rows with max biomass: {len(max_rows)}")

    for row_idx, (_, row) in enumerate(max_rows.iterrows()):
        output_lines.append(f"\n-- Pathway {row_idx + 1} (Index: {row.name}) --")
        reactions = row.drop(labels=[biomass_col])
        non_zero_reactions = reactions[reactions > threshold]
        output_lines.append("Non-zero reactions:")
        output_lines.append(", ".join(non_zero_reactions.index.tolist()))

### Optimality: EFV vs OFV most optimal paths ###

df2 = df_list[1]
max_biomass_df2 = df2[biomass_col].max()
df2_max_row = df2[df2[biomass_col] == max_biomass_df2].iloc[0]
df2_max_active = df2_max_row.drop(labels=[biomass_col])
set_df2_max = set(df2_max_active[df2_max_active > threshold].index)

df1 = df_list[0]
df1_rows = [df1.iloc[0], df1.iloc[1]]
comparison_labels = ["OFV (Fructose)", "OFV (Glucose)"]

# Define rocket-style Venn colors (distinct)
venn_colors = ['#35193e', '#a8325c', '#f2b134']  # Left, Right, Overlap

for i, ref_row in enumerate(df1_rows):
    label = comparison_labels[i]
    ref_flux = ref_row.drop(labels=[biomass_col])
    ref_set = set(ref_flux[ref_flux > threshold].index)

    intersection = ref_set & set_df2_max
    only_df1 = ref_set - set_df2_max
    only_df2 = set_df2_max - ref_set

    # Venn plot
    plt.figure(figsize=(5, 4))
    v = venn2([ref_set, set_df2_max], set_labels=(label, "EFV (Max. biomass)"))
    if v.get_patch_by_id('10'): v.get_patch_by_id('10').set_color(venn_colors[0])
    if v.get_patch_by_id('01'): v.get_patch_by_id('01').set_color(venn_colors[1])
    if v.get_patch_by_id('11'): v.get_patch_by_id('11').set_color(venn_colors[2])
    plt.title(f"{label} vs EFV (Max. biomass)")
    plt.tight_layout()

    venn_base = f"EFV_OFV_Glc_Frc_OPs_{label.replace(' ', '_')}_venn3"
    plt.savefig(output_path / f"{venn_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / f"{venn_base}.svg", bbox_inches='tight')
    plt.close()

    # Write comparison to text
    output_lines.append(f"\n=== {label} vs EFV (Max. biomass) ===")
    output_lines.append(f"🟢 Shared Reactions: {len(intersection)}")
    output_lines.append(", ".join(sorted(intersection)))

    output_lines.append(f"\n🔵 Only in {label}: {len(only_df1)}")
    for rxn in sorted(only_df1):
        output_lines.append(f"  {rxn}: {ref_flux[rxn]}")

    output_lines.append(f"\n🔴 Only in EFV (Max. biomass): {len(only_df2)}")
    for rxn in sorted(only_df2):
        output_lines.append(f"  {rxn}: {df2_max_active[rxn]}")

    output_lines.append("\n📊 Summary:")
    output_lines.append(f"  {label} active reactions:   {len(ref_set)}")
    output_lines.append(f"  EFV (Max. biomass) active reactions:   {len(set_df2_max)}")
    output_lines.append(f"  Shared:                     {len(intersection)}")
    output_lines.append(f"  Only in {label}:            {len(only_df1)}")
    output_lines.append(f"  Only in EFV (Max. biomass):    {len(only_df2)}")
    output_lines.append("-" * 50)

### Optimality: EFV Top 10 ###

output_lines.append("\n🔟 Top 10 biomass-producing EFVs:")
df2 = df_list[1]  # EFVs
top10 = df2.sort_values(by=biomass_col, ascending=False).head(10)

non_zero_sets = []
row_indices = []

for i, row in top10.iterrows():
    biomass_val = row[biomass_col]
    output_lines.append(f"Row {i}: biomass_col = {biomass_val:.10f}")

    flux_values = row.drop(labels=[biomass_col])
    active_reactions = set(flux_values[flux_values > threshold].index)
    non_zero_sets.append(active_reactions)
    row_indices.append(i)

# Shared Reactions
shared_reactions = set.intersection(*non_zero_sets)
output_lines.append(f"\n🧬 Shared active reactions (flux > {threshold}) in top 10 EFVs:")
output_lines.append(f"Count: {len(shared_reactions)}")
output_lines.append(", ".join(sorted(shared_reactions)))

# Unique Reactions Per EFV 
output_lines.append("\n🧷 Unique reactions per EFV (not shared by all):")
for i, row in zip(row_indices, top10.itertuples(index=False)):
    flux_values = pd.Series(row._asdict()).drop(labels=[biomass_col])
    active = set(flux_values[flux_values > threshold].index)
    unique_reactions = active - shared_reactions

    output_lines.append(f"\nRow {i} unique reactions (n = {len(unique_reactions)}):")
    for rxn in sorted(unique_reactions):
        output_lines.append(f"  {rxn}: {flux_values[rxn]:.6f}")

# Venn Diagram for Top 3 EFVs 
set1, set2, set3 = non_zero_sets[:3]
venn_labels = [f"Row {row_indices[0]}", f"Row {row_indices[1]}", f"Row {row_indices[2]}"]

plt.figure(figsize=(6, 5))
v = venn3([set1, set2, set3], set_labels=venn_labels)

# Customize color (use rocket-like palette)
colors = ['#35193e', '#a8325c', '#f2b134']
patch_ids = ['100', '010', '001', '110', '101', '011', '111']
for pid, color in zip(patch_ids, colors * 3):  # cycle if needed
    if v.get_patch_by_id(pid):
        v.get_patch_by_id(pid).set_color(color)

plt.title("Overlap of Active Reactions in Top 3 EFV Pathways")
plt.tight_layout()

# Save Venn diagram
venn_path_base = output_path / "EFV_Glc_Frc_OPs_venn3"
plt.savefig(f"{venn_path_base}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{venn_path_base}.svg", bbox_inches='tight')
plt.close()

### Save txt file ###

txt_path.mkdir(parents=True, exist_ok=True)
with open(txt_path / "EFV_OFV_Glc_Frc_OPs.txt", "w") as f:
    f.write("\n".join(output_lines))

print(" Done! ✅")

print('Everything complete! 🎉')



