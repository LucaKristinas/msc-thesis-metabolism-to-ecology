# ════════════════════════════════════════════════════════════════
# 1. Import Packages
# ════════════════════════════════════════════════════════════════

# Core libraries
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# COBRApy for metabolic modeling
from cobra import Reaction, Metabolite
from cobra.io import read_sbml_model, save_matlab_model, write_sbml_model

# Statistics
from sklearn.metrics import mean_squared_error

# Add Repo Root to sys.path 
project_root = Path(__file__).resolve().parents[2]  
sys.path.append(str(project_root))

# Import Helper Functions
from src.utils import (clean_values, get_weighted_lambdas)

# ════════════════════════════════════════════════════════════════
# 2. Import & Export Paths
# ════════════════════════════════════════════════════════════════

# Raw Data 
raw_data_path = project_root / "data" / "raw"
csv_raw_path = raw_data_path / "csv_files"

# Processed Data 
processed_data_path = project_root / "data" / "processed"
sbml_processed_path = processed_data_path / "sbml_files"
csv_processed_path = processed_data_path / "csv_files"
model_path = sbml_processed_path / "efmlrs" / "output_pre" / "klamt_martino_model_mplrs.xml"

# Output
output_path = processed_data_path / "figures"

# ════════════════════════════════════════════════════════════════
# 3. Data Import
# ════════════════════════════════════════════════════════════════

print('📦 Importing Data...')

EFMs_df = pd.read_csv(csv_processed_path / "EFMs.csv")
EFVs_df = pd.read_csv(csv_processed_path / "EFVs.csv")
MSE_EFMs_df = pd.read_csv(csv_processed_path / "MSE_EFMs_df.csv")
MSE_EFVs_df = pd.read_csv(csv_processed_path / "MSE_EFVs_df.csv")
FBA_fluxes = pd.read_csv(csv_processed_path / "Fba_fluxes.csv", index_col=0)
FBA_fluxes_std_bounds = pd.read_csv(csv_processed_path / "Fba_fluxes_std_bounds.csv", index_col=0)
FBA_MSE = pd.read_csv(csv_processed_path / "fba_mse_values_GR01_GR02.csv", index_col=0)
Flux_Measurements_df = pd.read_csv(csv_processed_path / "Flux_Measurements_Normalised.csv", index_col=0)
panel_b_lines = pd.read_csv(csv_processed_path / "Martino_GR02_MSE.csv")
flux_std_GR02_df = pd.read_csv(csv_raw_path / "Martino_Std_GR02.csv", sep=',')
flux_std_GR01_df = pd.read_csv(csv_raw_path / "Martino_Std_GR01.csv", sep=',')

print(' ✅ Done!')

# ════════════════════════════════════════════════════════════════
# 4. Data Preparation
# ════════════════════════════════════════════════════════════════

print('Pre-Processing of data for plotting...')

### GENERAL ###

# Choose color palette & theme
rocket_colors = sns.color_palette("rocket", 10)
sns.set_theme(style="white")

### PANEL A ###

# Define target reactions for x axis 
target_reactions = [col for col in Flux_Measurements_df.columns if col != "T"]

# Get rows with minimum MSE for each GR condition and method
min_mse_df = pd.DataFrame({
    "EFMs (min MSE for GR01)": MSE_EFMs_df.loc[MSE_EFMs_df["MSE_GR_01"].idxmin(), target_reactions],
    "EFMs (min MSE for GR02)": MSE_EFMs_df.loc[MSE_EFMs_df["MSE_GR_02"].idxmin(), target_reactions],
    "EFVs (min MSE for GR01)": MSE_EFVs_df.loc[MSE_EFVs_df["MSE_GR_01"].idxmin(), target_reactions],
    "EFVs (min MSE for GR02)": MSE_EFVs_df.loc[MSE_EFVs_df["MSE_GR_02"].idxmin(), target_reactions]
}).T

# Extract EFMs and EFVs predictions at min and max T
efms_min_T = MSE_EFMs_df.loc[MSE_EFMs_df["T"].idxmin(), target_reactions]
efms_max_T = MSE_EFMs_df.loc[MSE_EFMs_df["T"].idxmax(), target_reactions]
efvs_min_T = MSE_EFVs_df.loc[MSE_EFVs_df["T"].idxmin(), target_reactions]
efvs_max_T = MSE_EFVs_df.loc[MSE_EFVs_df["T"].idxmax(), target_reactions]


# Define growth rate labels and match to experimental flux rows
gr_conditions = {
    "GR=0.2": {
        "exp_row": "Flux_Average_GR_02",
        "label_suffix": "GR02"
    },
    "GR=0.1": {
        "exp_row": "Flux_Average_GR_01",
        "label_suffix": "GR01"
    }
}

# Define shared model-predicted profiles (same for all GRs)
shared_profiles = {
    "EFMs (min T)": efms_min_T,
    "EFVs (min T)": efvs_min_T,
    "EFMs (max T)": efms_max_T,
    "EFVs (max T)": efvs_max_T,
    "FBA (model bounds)": FBA_fluxes.loc[target_reactions].squeeze(),
    "FBA (standard bounds)": FBA_fluxes_std_bounds.loc[target_reactions].squeeze()
}


# Build one plot dataframe per GR condition
plot_data_dict = {}

for gr_label, info in gr_conditions.items():
    exp_row = info["exp_row"]
    suffix = info["label_suffix"]

    plot_data_dict[gr_label] = pd.DataFrame({
        f"Experimental Data ({gr_label})": Flux_Measurements_df.loc[exp_row, target_reactions].astype(float),
        "EFMs (best)": min_mse_df.loc[f"EFMs (min MSE for {suffix})"],
        "EFVs (best)": min_mse_df.loc[f"EFVs (min MSE for {suffix})"],
        **shared_profiles
    }).T

plot_data_02 = plot_data_dict["GR=0.2"]
plot_data_01 = plot_data_dict["GR=0.1"]


# Clean values and reassign column headers
for df in [plot_data_01, plot_data_02]:
    df[:] = clean_values(df)
    df.columns = target_reactions


# Define plotting subsets and reaction order
manual_reaction_order = [
    "GLCpts", "G6PDH2r", "GND", "PGI", "PFK", "TKT1", "TKT2", "TALA",
    "GAPD", "ENO", "PYK", "PDH", "CS", "ICDHyr", "AKGDH", "FUM",
    "MDH", "ME1", "ICL"
]

labels_EFVs_02 = [
    "Experimental Data (GR=0.2)", "FBA (model bounds)",
    "EFVs (min T)", "EFVs (best)", "EFVs (max T)"
]

labels_EFMs_02 = [
    "Experimental Data (GR=0.2)", "FBA (standard bounds)",
    "EFMs (min T)", "EFMs (best)", "EFMs (max T)"
]

labels_EFVs_01 = [
    "Experimental Data (GR=0.1)", "FBA (model bounds)",
    "EFVs (min T)", "EFVs (best)", "EFVs (max T)"
]

labels_EFMs_01 = [
    "Experimental Data (GR=0.1)", "FBA (standard bounds)",
    "EFMs (min T)", "EFMs (best)", "EFMs (max T)"
]

# Extract subsets for plotting
plot_subset_EFVs_02 = plot_data_02.loc[labels_EFVs_02]
plot_subset_EFMs_02 = plot_data_02.loc[labels_EFMs_02]
plot_subset_EFVs_01 = plot_data_01.loc[labels_EFVs_01]
plot_subset_EFMs_01 = plot_data_01.loc[labels_EFMs_01]

# Define x-axis base
target_reactions_sorted = manual_reaction_order
x_base = np.arange(len(target_reactions_sorted))


# Load and align experimental standard deviations (GR01 + GR02)
flux_std_GR02_df = flux_std_GR02_df.rename(columns={"# Std": "Std"})
flux_std_GR02_df["Std"] = flux_std_GR02_df["Std"] * 0.01
flux_std_GR02_df = flux_std_GR02_df.set_index("Reaction")
std_aligned_GR02 = flux_std_GR02_df.reindex(manual_reaction_order)["Std"].values

flux_std_GR01_df = flux_std_GR01_df.rename(columns={"# Std": "Std"})
flux_std_GR01_df["Std"] = flux_std_GR01_df["Std"] * 0.01
flux_std_GR01_df = flux_std_GR01_df.set_index("Reaction")
std_aligned_GR01 = flux_std_GR01_df.reindex(manual_reaction_order)["Std"].values

### PANEL B ###

# Optimal and boundary T values
T_opt_EFV = MSE_EFVs_df.loc[MSE_EFVs_df["MSE_GR_02"].idxmin(), "T"]
T_opt_EFM = MSE_EFMs_df.loc[MSE_EFMs_df["MSE_GR_02"].idxmin(), "T"]
T_min = MSE_EFVs_df["T"].min()  # For COBRApy dot placement

# Labels and MSE types used for curves
curve_labels = ['EPs (Normalized Glc Uptake to 10)', 'EFVs (Normalized Glc Uptake to 10)']
mse_types = ['MSE_GR_02', 'MSE_GR_01']

# 3. Reference lines from De Martino et al. (2018)
ref_lines = {
    'UNIFORM': panel_b_lines.loc[0, 'Line_UNIFORM'],
    'MAXENT':  panel_b_lines.loc[0, 'Line_MAXENT'],
    'FBA':     panel_b_lines.loc[0, 'Line_FBA']
}

# COBRApy FBA MSE points (shown as dots on plots)
mse_dots = {
    'FBA (Model Bounds) 01': {
        'value': FBA_MSE.loc['FBA_flux_bounds_01', 'MSE'],
        'color': rocket_colors[3]
    },
    'FBA (Standard Bounds) 01': {
        'value': FBA_MSE.loc['FBA_std_bounds_01', 'MSE'],
        'color': rocket_colors[3]
    },
    'FBA (Model Bounds) 02': {
        'value': FBA_MSE.loc['FBA_flux_bounds_02', 'MSE'],
        'color': rocket_colors[3]
    },
    'FBA (Standard Bounds) 02': {
        'value': FBA_MSE.loc['FBA_std_bounds_02', 'MSE'],
        'color': rocket_colors[3]
    }
}

# λ annotation target and corresponding MSE values
target_x = 0.00007 
x_idx = np.argmin(np.abs(MSE_EFVs_df["T"].values - target_x))
y_GR02 = MSE_EFVs_df["MSE_GR_02"].values[x_idx]
y_GR01 = MSE_EFVs_df["MSE_GR_01"].values[x_idx]

### PANEL C ### 

# Setup for both EFVs and EFMs 
scaled_fluxes = {}
best_T = {}
min_T = {}
max_T = {}
lambdas = {}
means = {}
dfs = {}
avg_lambda_proxies = {}

for label, df, mse_df in [
    ('EFV', EFVs_df, MSE_EFVs_df),
    ('EFM', EFMs_df, MSE_EFMs_df)
]:
    # Scale to unit glucose uptake
    scaling_factors = 1 / df['EX_glc(e)']
    scaled_fluxes[label] = df.multiply(scaling_factors, axis=0)

    # T values
    best_T[label] = mse_df.loc[mse_df['MSE_GR_02'].idxmin(), 'T']
    min_T[label] = mse_df['T'].min()
    max_T[label] = mse_df['T'].max()

    # Weighted λ distributions
    lambdas[label] = {
        'best': get_weighted_lambdas(scaled_fluxes[label], best_T[label]),
        'min':  get_weighted_lambdas(scaled_fluxes[label], min_T[label]),
        'max':  get_weighted_lambdas(scaled_fluxes[label], max_T[label])
    }

    # Weighted λ means
    means[label] = {
        k: np.sum(l * w)
        for k, (l, w) in lambdas[label].items()
    }

    # DataFrames for KDE plotting
    dfs[label] = {
        k: pd.DataFrame({'λ': l, 'w': w})
        for k, (l, w) in lambdas[label].items()
    }

    # Custom legend proxy
    avg_lambda_proxies[label] = Line2D(
        [0], [0], color='lightgrey', linestyle='--', linewidth=1.5, label=f'average λ ({label})'
    )

print(' ✅ Done!')

# ════════════════════════════════════════════════════════════════
# 5. Plot Design 
# ════════════════════════════════════════════════════════════════

print('Plot Style configurations and theme setup...')

### PANEL A ###

# Style configurations
color_indices = [0, 3, 3, 7, 9]
marker_styles = ['o', 'o', 'o', '*', 'd']
fillstyles = ['full', 'none', 'full', 'full', 'full']
markersizes = [8, 8, 4, 8, 5]

# Offsets to spread markers across each reaction group
offsets = [-0.3, -0.1, -0.1, 0.1, 0.3]

### PANEL B ### 

# Matplotlib theme setup
plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "in",
    "ytick.direction": "in"
})

# Color for the background vertical reference line
background_color = rocket_colors[7]

# Curve styles: color, line type, width for EFV and EFM curves
curve_colors = ['lightgrey', 'lightgrey', 'black', 'black']
curve_linestyles = ['-', '--', '-', '--']
curve_linewidths = [1.5, 1, 1.5, 1]

# Horizontal reference line styles
line_colors = {
    'UNIFORM': rocket_colors[0],
    'MAXENT': rocket_colors[0],
    'FBA': rocket_colors[0]
}
line_style = 'dotted'
ref_linewidth = 0.8

# Text config for reference labels
x_ref = 1
text_offset = 0.001
fontsize_ref = 7

# Legend setup
custom_legend_labels2 = [
    "EFV",  
    "EFM / EP",
    "COBRApy FBA"
]

custom_handles2 = [
    Line2D([0], [0], color=curve_colors[2], linestyle='-', linewidth=2),
    Line2D([0], [0], color=curve_colors[0], linestyle='-', linewidth=2),
    Line2D([0], [0], marker='o', color=rocket_colors[3], linestyle='None', markersize=6)
]

# Configuration list for EFV / EFM subplots
curve_configs = [
    {
        "df": MSE_EFMs_df,
        "label": "EFM / EP",
        "color": curve_colors[0],
        "dot_keys": ["FBA (Standard Bounds) 01", "FBA (Standard Bounds) 02"],
        "T_opt_local": T_opt_EFM
    },
    {
        "df": MSE_EFVs_df,
        "label": "EFV",
        "color": curve_colors[2],
        "dot_keys": ["FBA (Model Bounds) 01", "FBA (Model Bounds) 02"],
        "T_opt_local": T_opt_EFV
    }
]

# Shared legend elements for both subplots
shared_legend_handles = [
    Line2D([0], [0], color="gray", linestyle='-', linewidth=2, label=r'$\lambda = 0.1\,h^{-1}$'),
    Line2D([0], [0], color="black", linestyle='-', linewidth=2, label=r'$\lambda = 0.2\,h^{-1}$'),
    Line2D([0], [0], marker='o', color=rocket_colors[3], linestyle='None', markersize=6, label='COBRApy FBA')
]

### PANEL C ###

# Color Choice
color_min_T = rocket_colors[3]
color_best_T = rocket_colors[7]
color_max_T = rocket_colors[9]

# Legend
handles_panelc = [
    Patch(facecolor=color_max_T, edgecolor='black', linewidth=0.5, label='Unregulated (max. T)'),
    Patch(facecolor=color_min_T, edgecolor='black', linewidth=0.5, label='Regulated (min. T)'),
    Patch(facecolor=color_best_T, edgecolor='black', linewidth=0.5, label='Least Biased'),
    Line2D([0], [0], color='black', linestyle=':', linewidth=1, label='average λ')
]

print(' ✅ Done!')

# ════════════════════════════════════════════════════════════════
# 6. Plots
# ════════════════════════════════════════════════════════════════

print('Plot Generation...')

### PANEL A ###

### Single Plots

# Define all plotting configs
plot_configs = [
    ("IMB_Panel_A_EFVs_01", labels_EFVs_01, plot_subset_EFVs_01, std_aligned_GR01, "0.1"),
    ("IMB_Panel_A_EFVs_02", labels_EFVs_02, plot_subset_EFVs_02, std_aligned_GR02, "0.2"),
    ("IMB_Panel_A_EFMs_01", labels_EFMs_01, plot_subset_EFMs_01, std_aligned_GR01, "0.1"),
    ("IMB_Panel_A_EFMs_02", labels_EFMs_02, plot_subset_EFMs_02, std_aligned_GR02, "0.2"),
]

for name, selected_labels, selected_subset, selected_std, gr_value in plot_configs:
    print(f"### Processing: {name} ###")

    pathway_type = "EFVs" if any("EFVs" in label for label in selected_labels) else "EFMs"

    fig, ax = plt.subplots(figsize=(10, 4))

    # Background shading
    for i in range(len(target_reactions_sorted)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color='lightgrey', alpha=0.3, zorder=0)

    # Plotting loop
    for i, label in enumerate(selected_labels):
        y = selected_subset.loc[label, target_reactions_sorted].values
        x_shifted = x_base + offsets[i]

        color = rocket_colors[color_indices[i]]
        marker = marker_styles[i]
        fill = fillstyles[i]
        markersize = markersizes[i]

        if i == 0:  # Experimental with error bars
            ax.errorbar(
                x_shifted, y,
                yerr=selected_std,
                fmt=marker,
                color=color,
                markerfacecolor='none' if fill == 'none' else color,
                markersize=markersize,
                linestyle='None',
                capsize=3,
                elinewidth=1,
                ecolor=color
            )
        else:
            ax.plot(
                x_shifted, y,
                linestyle='None',
                marker=marker,
                color=color,
                markerfacecolor='none' if fill == 'none' else color,
                markersize=markersize
            )

    # Legend
    custom_legend_labels_panelA = [
        fr"Flux Data ($\lambda = {gr_value}\,h^{{-1}}$)",  
        "COBRApy FBA",
        "Regulated (min. T)", 
        "Least-biased", 
        "Unregulated (max. T)"
    ]

    custom_handles_panelA = [
        Line2D([0], [0], marker='o', color=rocket_colors[0], markersize=8, linestyle='None'),
        Line2D([0], [0], marker='o', color=rocket_colors[3], markerfacecolor='none', markersize=8, linestyle='None'),
        Line2D([0], [0], marker='o', color=rocket_colors[3], markersize=4, linestyle='None'),
        Line2D([0], [0], marker='*', color=rocket_colors[7], markersize=8, linestyle='None'),
        Line2D([0], [0], marker='d', color=rocket_colors[9], markersize=5, linestyle='None')
    ]

    legend = ax.legend(
        handles=custom_handles_panelA,
        labels=custom_legend_labels_panelA,
        loc='upper right',
        frameon=True,
        fontsize=9
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)

    # Title and formatting
    ax.set_title(fr"Flux Prediction using {pathway_type} for $\lambda = {gr_value}\,h^{{-1}}$", fontsize=11)
    ax.set_xticks(x_base)
    ax.set_xticklabels(target_reactions_sorted, rotation=90)
    ax.set_ylabel("Flux relative to GLC uptake")
    ax.set_ylim(-0.06, 2)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlim(-0.5 + min(offsets), len(target_reactions_sorted) - 0.5 + max(offsets))
    plt.tight_layout(pad=0.1)

    # Save PNG and SVG
    fig.savefig(output_path / f"{name}.png", dpi=300)
    fig.savefig(output_path / f"{name}.svg")
    plt.close()

    print(f"✓ Saved: {name}.png and {name}.svg to {output_path}")

### Overview Plot

# Panel Configurations
panel_configs = [
    ("EFMs 01", labels_EFMs_01, plot_subset_EFMs_01),
    ("EFMs 02", labels_EFMs_02, plot_subset_EFMs_02),
    ("EFVs 01", labels_EFVs_01, plot_subset_EFVs_01),
    ("EFVs 02", labels_EFVs_02, plot_subset_EFVs_02)
]

# Define individual subplot titles
subplot_titles = [
    r"Extreme Pathways / Elementary Flux Modes ($\lambda = 0.1\,h^{-1}$)",
    r"Extreme Pathways / Elementary Flux Modes ($\lambda = 0.2\,h^{-1}$)",
    r"Elementary Flux Vectors ($\lambda = 0.1\,h^{-1}$)",
    r"Elementary Flux Vectors ($\lambda = 0.2\,h^{-1}$)"
]

# Set shared parameters
gr_value = "0.1"
selected_std = std_aligned_GR01

# 2x2 Panel Creation
fig, axs = plt.subplots(4, 1, figsize=(13, 14), sharex=True, sharey=True)
axs = axs.flatten()

# Plot each panel
for idx, (label_str, selected_labels, selected_subset) in enumerate(panel_configs):
    ax = axs[idx]

    # Shaded background columns
    for i in range(len(target_reactions_sorted)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color='lightgrey', alpha=0.3, zorder=0)

    # Plot each data series
    for i, label in enumerate(selected_labels):
        y = selected_subset.loc[label, target_reactions_sorted].values
        x_shifted = x_base + offsets[i]

        color = rocket_colors[color_indices[i]]
        marker = marker_styles[i]
        fill = fillstyles[i]
        markersize = markersizes[i]

        if i == 0:  # Experimental
            ax.errorbar(
                x_shifted, y, yerr=selected_std,
                fmt=marker, color=color,
                markerfacecolor='none' if fill == 'none' else color,
                markersize=markersize,
                linestyle='None', capsize=3, elinewidth=1, ecolor=color
            )
        else:
            ax.plot(
                x_shifted, y,
                linestyle='None', marker=marker,
                color=color,
                markerfacecolor='none' if fill == 'none' else color,
                markersize=markersize
            )

    # Set axis properties
    ax.set_xticks(x_base)
    ax.set_xticklabels(target_reactions_sorted, rotation=90)
    ax.set_ylim(-0.06, 2)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlim(-0.5 + min(offsets), len(target_reactions_sorted) - 0.5 + max(offsets))

    # Set Y-label only on left plots
    if idx % 2 == 0:
        ax.set_ylabel("Flux relative to GLC uptake", fontsize=9)

    # Set individual title
    ax.set_title(subplot_titles[idx], fontsize=10, pad=8)

#Legend
custom_legend_labels = [
    fr"Flux Measurement Data",  
    "COBRApy FBA",
    "Regulated (min. T)", 
    "Least-biased", 
    "Unregulated (max. T)"
]

custom_handles = [
    Line2D([0], [0], marker='o', color=rocket_colors[0], markersize=8, linestyle='None'),
    Line2D([0], [0], marker='o', color=rocket_colors[3], markerfacecolor='none', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='o', color=rocket_colors[3], markersize=4, linestyle='None'),
    Line2D([0], [0], marker='*', color=rocket_colors[7], markersize=8, linestyle='None'),
    Line2D([0], [0], marker='d', color=rocket_colors[9], markersize=5, linestyle='None')
]

legend = axs[0].legend(
    handles=custom_handles,
    labels=custom_legend_labels,
    fontsize=9,
    frameon=True,
    fancybox=True,
    facecolor='white',
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0)
)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.8)

#Final Layout

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.subplots_adjust(hspace=0.15)
# Save PNG and SVG
fig.savefig(output_path / "IMB_Panel_A_Overview.png", dpi=300)
fig.savefig(output_path / "IMB_Panel_A_Overview.svg")
plt.close()

print(f"✓ Saved: IMB_Panel_A_Overview.png and .svg to {output_path}")

### PANEL B ###

### Single Plot (Overview)

# Setup Plot
fig, ax = plt.subplots(figsize=(6.5, 5))

# Add Static Reference Lines
for name, y_val in ref_lines.items():
    ax.axhline(y=y_val, color=line_colors[name], linestyle=line_style, linewidth=ref_linewidth)

ax.axvline(x=T_opt_EFV, color=background_color, linestyle='-', linewidth=1.5, alpha=0.4, zorder=0)

ax.text(
    T_opt_EFV, 0.23,
    fr'$T = {T_opt_EFV:.5f}$',
    fontsize=9,
    color=background_color,
    ha='right',
    va='top'
)

# Plot EFV/EFM MSE Curves
color_idx = 0
for df, label_prefix in zip([MSE_EFMs_df, MSE_EFVs_df], curve_labels):
    T_vals = df["T"].values
    for mse_col in mse_types:
        mse_vals = df[mse_col].values
        ax.plot(
            T_vals, mse_vals,
            label=f'{label_prefix} ({mse_col})',
            color=curve_colors[color_idx],
            linestyle=curve_linestyles[color_idx],
            linewidth=curve_linewidths[color_idx]
        )
        color_idx += 1


# COBRApy FBA Dot Markers
for label, dot in mse_dots.items():
    ax.plot(T_min, dot['value'], 'o', color=dot['color'])

# Annotate Growth Rate Points (λ annotations)
ax.text(target_x, y_GR02 + 0.0025, r'$\lambda = 0.2\,h^{-1}$', fontsize=9, color=curve_colors[2])
ax.text(target_x, y_GR01 + 0.0025, r'$\lambda = 0.1\,h^{-1}$', fontsize=9, color=curve_colors[3])


# Label Horizontal Reference Lines
label_x_pos = 1.02 

ax.text(label_x_pos, ref_lines['UNIFORM'], 'UNIFORM',
        fontsize=fontsize_ref, color=line_colors['UNIFORM'],
        va='center', ha='left', transform=ax.get_yaxis_transform())

ax.text(label_x_pos, ref_lines['MAXENT'], 'MAXENT',
        fontsize=fontsize_ref, color=line_colors['MAXENT'],
        va='center', ha='left', transform=ax.get_yaxis_transform())

ax.text(label_x_pos, ref_lines['FBA'], 'FBA',
        fontsize=fontsize_ref, color=line_colors['FBA'],
        va='center', ha='left', transform=ax.get_yaxis_transform())

# Axis Styling──
ax.set_xscale('log')
ax.set_xlabel(r'$T$', fontsize=10)
ax.set_ylabel('MSE', fontsize=10)
ax.set_yticks(np.arange(0.05, 0.275, 0.025))
ax.grid(False)

# Custom Legend
legend = ax.legend(
    handles=custom_handles2,
    labels=custom_legend_labels2,
    loc='upper left',               
    bbox_to_anchor=(1.02, 1.0),     
    frameon=True,
    fontsize=9,
    handletextpad=0.5,
    borderpad=0.5
)

legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_alpha(1.0)

# Save plot
plt.tight_layout()
fig.savefig(output_path / "IMB_Panel_B_Overview.png", dpi=300)
fig.savefig(output_path / "IMB_Panel_B_Overview.svg")
print("✓ Saved: IMB_Panel_B_Overview.png and .svg")

### Split EFM / EFV Version of Plot (v1: Infobox)

fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

for config, ax in zip(curve_configs, axs):
    df = config["df"]
    label = config["label"]
    T_opt_local = config["T_opt_local"]

    # Plot main curve here (assuming you already do this)

    # Plot all associated dots
    for dot_key in config["dot_keys"]:
        dot_info = mse_dots[dot_key]
        ax.scatter(
            T_min,
            dot_info["value"],
            color=dot_info["color"],
            label=dot_key,
            zorder=5
        )

    ax.axvline(x=T_opt_local, color=background_color, linestyle='-', linewidth=1.5, alpha=0.4, zorder=0)
    x_offset = T_opt_local * 0.25
    ax.text(T_opt_local - x_offset, 0.23, fr'$T = {T_opt_local:.5f}$',
            fontsize=9, color=background_color, ha='right', va='top')

    ax.plot(df["T"].values, df["MSE_GR_01"].values, color="gray", linestyle='-', linewidth=2)
    ax.plot(df["T"].values, df["MSE_GR_02"].values, color="black", linestyle='-', linewidth=2)

    T_min = np.min(df["T"].values)

    ax.set_xscale('log')
    ax.set_xlabel(r'$T$', fontsize=10)
    ax.set_yticks(np.arange(0.05, 0.275, 0.025))
    ax.grid(False)
    ax.set_title(f"MSE Curves for {label}", fontsize=11)

# Shared y-label
axs[0].set_ylabel('MSE', fontsize=10)

# Shared legend
legend = axs[1].legend(
    handles=shared_legend_handles,
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    frameon=True,
    fontsize=9
)
legend.get_frame().set_edgecolor('black')

# Add shared info box to second subplot

info_title = r"$\mathbf{MSE\ (De\ Martino\ et\ al.,\ 2018)}$"
info_lines = [fr"$\it{{{key}}}$: {value:.5f}" for key, value in ref_lines.items()]
info_text = info_title + "\n" + "\n".join(info_lines)

info_box = AnchoredText(
    info_text,
    loc='lower left',           # You can adjust this (e.g., 'upper right', 'center left', etc.)
    bbox_to_anchor=(1.03, 0.6), # (x, y) position in axes fraction coordinates
    bbox_transform=axs[1].transAxes,
    prop=dict(size=8),
    frameon=True
)

info_box.patch.set_boxstyle("round,pad=0.4")
info_box.patch.set_edgecolor('black')
info_box.patch.set_facecolor("#fceef6")  # Light grey

axs[1].add_artist(info_box)

# Save plot
plt.tight_layout()
plt.subplots_adjust(top=0.88, right=0.85)
fig.savefig(output_path / "IMB_Panel_B_EFV_EFM_v1.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "IMB_Panel_B_EFV_EFM_v1.svg", bbox_inches='tight')
print("✓ Saved: IMB_Panel_B_EFV_EFM_v1.png and .svg")

### Split EFM / EFV Version of Plot (v2: 'X' Desing)

# Setup subplot canvas
fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

# Add EFV/EFM data to each subplot
for config, ax in zip(curve_configs, axs):
    df = config["df"]
    label = config["label"]
    T_opt_local = config["T_opt_local"]

    # Plot main curve here (assuming you already do this)

    # Plot all associated dots
    for dot_key in config["dot_keys"]:
        dot_info = mse_dots[dot_key]
        ax.scatter(
            T_min,
            dot_info["value"],
            color=dot_info["color"],
            label=dot_key,
            zorder=5
        )
    
    # Add white horizontal reference lines (in background)
    if ax is axs[1]:  # Only for the right subplot
        for name, y_val in ref_lines.items():
            ax.axhline(y=y_val, color='white', linestyle='--', linewidth=1.0, zorder=0)

        for name, y_val in ref_lines.items():
                # Add a vertical tick-like marker
                ax.plot(
                    [1], [y_val],  # X just outside plot
                    marker='x', color='black', markersize=8,
                    transform=ax.get_yaxis_transform(), clip_on=False
                )

        # Add external labels for those lines
        label_x_pos = 1.02  # Outside the right edge
        for name, y_val in ref_lines.items():
            ax.text(
                label_x_pos, y_val, name,
                fontsize=9,
                color='black',
                va='center',
                ha='left',
                transform=ax.get_yaxis_transform()
            )

    # Vertical T_opt line
    ax.axvline(x=T_opt_local, color=background_color, linestyle='-', linewidth=1.5, alpha=0.4, zorder=0)
    x_offset = T_opt_local * 0.25
    ax.text(T_opt_local - x_offset, 0.23, fr'$T = {T_opt_local:.5f}$',
            fontsize=9, color=background_color, ha='right', va='top')
    
    # Plot MSE curves
    ax.plot(df["T"].values, df["MSE_GR_01"].values, color="gray", linestyle='-', linewidth=2)
    ax.plot(df["T"].values, df["MSE_GR_02"].values, color="black", linestyle='-', linewidth=2)

    # Add COBRApy FBA dot
    T_min = np.min(df["T"].values)

    # Axis styling
    ax.set_xscale('log')
    ax.set_xlabel(r'$T$', fontsize=10)
    ax.set_yticks(np.arange(0.05, 0.275, 0.025))
    ax.grid(False)
    ax.set_title(f"MSE Curves for {label}", fontsize=11)

# Shared y-axis
axs[0].set_ylabel('MSE', fontsize=10)

# Add shared legend to second subplot
legend = axs[1].legend(
    handles=shared_legend_handles,
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    frameon=True,
    fontsize=9
)
legend.get_frame().set_edgecolor('black')

# Final layout
# Save plot
plt.tight_layout()
plt.subplots_adjust(top=0.88, right=0.85)
fig.savefig(output_path / "IMB_Panel_B_EFV_EFM_v2.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "IMB_Panel_B_EFV_EFM_v2.svg", bbox_inches='tight')
print("✓ Saved: IMB_Panel_B_EFV_EFM_v2.png and .svg")

### PANEL C ###

## version 1

# Set-Up
fig, axs = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)

# λ DISTRIBUTIONS FOR EFMs & EFVs

titles_c = {
    'EFM': "Extreme Pathways / Elementary Flux Modes",
    'EFV': "Elementary Flux Vectors"
}

# Set number of bins for histograms
bins = 100

for i, label in enumerate(['EFM', 'EFV']):
    ax = axs[i]
    ax.set_title(titles_c[label], fontsize=12, pad=10)

    # Plot histograms
    ax.hist(lambdas[label]['max'][0], bins=bins, weights=lambdas[label]['max'][1], alpha=0.4, color=color_max_T)
    ax.hist(lambdas[label]['min'][0], bins=bins, weights=lambdas[label]['min'][1], alpha=0.4, color=color_min_T)
    ax.hist(lambdas[label]['best'][0], bins=bins, weights=lambdas[label]['best'][1], alpha=0.6, color=color_best_T)

    # KDE curves
    sns.kdeplot(data=dfs[label]['max'], x='λ', weights='w', color=color_max_T, fill=True, alpha=0.25, linewidth=2, ax=ax)
    sns.kdeplot(data=dfs[label]['min'], x='λ', weights='w', color=color_min_T, fill=True, alpha=0.25, linewidth=2, ax=ax)
    sns.kdeplot(data=dfs[label]['best'], x='λ', weights='w', color=color_best_T, fill=True, alpha=0.3, linewidth=2, ax=ax)

    # Vertical λ mean lines
    ax.axvline(means[label]['max'], color=color_max_T, linestyle=':', linewidth=1.5)
    ax.axvline(means[label]['min'], color=color_min_T, linestyle=':', linewidth=1.5)
    ax.axvline(means[label]['best'], color=color_best_T, linestyle=':', linewidth=1.5)

    # Annotate mean λ values
    ax.text(means[label]['max'] - 0.005, 450, f"{means[label]['max']:.3f}", color=color_max_T, ha='right', va='bottom', fontsize=9)
    ax.text(means[label]['min'] + 0.005, 450, f"{means[label]['min']:.3f}", color=color_min_T, ha='left', va='bottom', fontsize=9)
    ax.text(means[label]['best'] - 0.005, 450, f"{means[label]['best']:.3f}", color=color_best_T, ha='right', va='bottom', fontsize=9)

    # Normalize y-axis
    yticks = ax.get_yticks()
    max_y = yticks[-1] if yticks[-1] != 0 else 1
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y / max_y:.1f}" for y in yticks])

    # Labels
    if label == 'EFV':
        ax.set_xlabel(r'Growth Rate $\lambda\, \mathrm{in}\ \mathrm{h}^{-1}$ (Glucose Uptake 2.83 mmol/gDWh)', fontsize=10)
    ax.set_ylabel("Frequency (Cell Population)", fontsize=10)

# Legend Set-Up

legend = axs[0].legend(
    handles=handles_panelc,
    fontsize=9,
    frameon=True,
    fancybox=True,
    facecolor='white',
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0)
)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.0)

# Final Layout

plt.tight_layout()
plt.subplots_adjust(right=0.78)  # extra space for legend
fig.savefig(output_path / "IMB_Panel_C_v1.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "IMB_Panel_C_v1.svg", bbox_inches='tight')
print("✓ Saved: IMB_Panel_C_v1.png and .svg")

## version 2

# Set-Up
fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

# λ DISTRIBUTIONS FOR EFMs & EFVs

# Set number of bins for histograms
bins = 100

for i, label in enumerate(['EFM', 'EFV']):
    ax = axs[i]
    ax.set_title(titles_c[label], fontsize=12, pad=10)

    # Plot histograms
    ax.hist(lambdas[label]['max'][0], bins=bins, weights=lambdas[label]['max'][1], alpha=0.4, color=color_max_T)
    ax.hist(lambdas[label]['min'][0], bins=bins, weights=lambdas[label]['min'][1], alpha=0.4, color=color_min_T)
    ax.hist(lambdas[label]['best'][0], bins=bins, weights=lambdas[label]['best'][1], alpha=0.6, color=color_best_T)

    # KDE curves
    sns.kdeplot(data=dfs[label]['max'], x='λ', weights='w', color=color_max_T, fill=True, alpha=0.25, linewidth=2, ax=ax)
    sns.kdeplot(data=dfs[label]['min'], x='λ', weights='w', color=color_min_T, fill=True, alpha=0.25, linewidth=2, ax=ax)
    sns.kdeplot(data=dfs[label]['best'], x='λ', weights='w', color=color_best_T, fill=True, alpha=0.3, linewidth=2, ax=ax)

    # Vertical λ mean lines
    ax.axvline(means[label]['max'], color=color_max_T, linestyle=':', linewidth=1.5)
    ax.axvline(means[label]['min'], color=color_min_T, linestyle=':', linewidth=1.5)
    ax.axvline(means[label]['best'], color=color_best_T, linestyle=':', linewidth=1.5)

    # Annotate mean λ values
    ax.text(means[label]['max'] - 0.005, 450, f"{means[label]['max']:.3f}", color=color_max_T, ha='right', va='bottom', fontsize=9)
    ax.text(means[label]['min'] + 0.005, 450, f"{means[label]['min']:.3f}", color=color_min_T, ha='left', va='bottom', fontsize=9)
    ax.text(means[label]['best'] - 0.005, 450, f"{means[label]['best']:.3f}", color=color_best_T, ha='right', va='bottom', fontsize=9)

    # Normalize y-axis
    yticks = ax.get_yticks()
    max_y = yticks[-1] if yticks[-1] != 0 else 1
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y / max_y:.1f}" for y in yticks])

    # Labels
    if label == 'EFV':
        ax.set_xlabel(r'Growth Rate $\lambda\, \mathrm{in}\ \mathrm{h}^{-1}$ (Glucose Uptake 2.83 mmol/gDWh)', fontsize=10)
    ax.set_ylabel("Frequency (Cell Population)", fontsize=10)

# Legend Set-Up

legend = axs[1].legend(
    handles=handles_panelc,
    fontsize=9,
    frameon=True,
    fancybox=True,
    facecolor='white',
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0)
)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.0)

# Final Layout

plt.tight_layout()
plt.subplots_adjust(right=0.78)  # extra space for legend
fig.savefig(output_path / "IMB_Panel_C_v2.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "IMB_Panel_C_v2.svg", bbox_inches='tight')
print("✓ Saved: IMB_Panel_C_v2.png and .svg")

### Overall figure ###

# === Overall figure set-up ===
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np

fig = plt.figure(figsize=(20, 14))  # Overall figure size

# OUTER LAYOUT: split figure into left (Panel A) and right (Panels B+C)
outer_gs = GridSpec(1, 2, width_ratios=[2, 1.5], figure=fig)  # width_ratios controls left/right panel width

# === PANEL A ===
gs_a = GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_gs[0], hspace=0.4)  # hspace controls vertical space between A subplots
axs_a = [fig.add_subplot(gs_a[i]) for i in range(4)]

# --- Panel A Plotting ---
panel_configs = [
    ("EFMs 01", labels_EFMs_01, plot_subset_EFMs_01),
    ("EFMs 02", labels_EFMs_02, plot_subset_EFMs_02),
    ("EFVs 01", labels_EFVs_01, plot_subset_EFVs_01),
    ("EFVs 02", labels_EFVs_02, plot_subset_EFVs_02)
]

subplot_titles = [
    r"Extreme Pathways / Elementary Flux Modes ($\lambda = 0.1\,h^{-1}$)",
    r"Extreme Pathways / Elementary Flux Modes ($\lambda = 0.2\,h^{-1}$)",
    r"Elementary Flux Vectors ($\lambda = 0.1\,h^{-1}$)",
    r"Elementary Flux Vectors ($\lambda = 0.2\,h^{-1}$)"
]

for idx, (label_str, selected_labels, selected_subset) in enumerate(panel_configs):
    ax = axs_a[idx]
    for i in range(len(target_reactions_sorted)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color='lightgrey', alpha=0.3, zorder=0)
    for i, label in enumerate(selected_labels):
        y = selected_subset.loc[label, target_reactions_sorted].values
        x_shifted = x_base + offsets[i]
        color = rocket_colors[color_indices[i]]
        marker = marker_styles[i]
        fill = fillstyles[i]
        markersize = markersizes[i]
        if i == 0:
            ax.errorbar(x_shifted, y, yerr=selected_std, fmt=marker, color=color,
                        markerfacecolor='none' if fill == 'none' else color,
                        markersize=markersize, linestyle='None', capsize=3, ecolor=color)
        else:
            ax.plot(x_shifted, y, marker=marker, linestyle='None',
                    color=color, markerfacecolor='none' if fill == 'none' else color,
                    markersize=markersize)
    ax.set_xticks(x_base)
    ax.set_xticklabels(target_reactions_sorted, rotation=90)
    ax.set_ylim(-0.06, 2)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlim(-0.5 + min(offsets), len(target_reactions_sorted) - 0.5 + max(offsets))
    if idx % 2 == 0:
        ax.set_ylabel("Flux relative to GLC uptake", fontsize=9)
    ax.set_title(subplot_titles[idx], fontsize=11, pad=8)  # Set consistent title size

# Legend in Panel A
legend_a = axs_a[0].legend(
    handles=[
        Line2D([0], [0], marker='o', color=rocket_colors[0], markersize=8, linestyle='None'),
        Line2D([0], [0], marker='o', color=rocket_colors[3], markerfacecolor='none', markersize=8, linestyle='None'),
        Line2D([0], [0], marker='o', color=rocket_colors[3], markersize=4, linestyle='None'),
        Line2D([0], [0], marker='*', color=rocket_colors[7], markersize=8, linestyle='None'),
        Line2D([0], [0], marker='d', color=rocket_colors[9], markersize=5, linestyle='None')
    ],
    labels=[
        "Flux Measurement Data", "COBRApy FBA", "Regulated (min. T)",
        "Least-biased", "Unregulated (max. T)"
    ],
    fontsize=9, frameon=True, fancybox=True, facecolor='white', loc='upper left'
)
legend_a.get_frame().set_edgecolor('black')
legend_a.get_frame().set_linewidth(0.8)

fig.text(-0.02, 0.965, "A", fontsize=18, fontweight='bold', va='top', ha='left')  # Panel A label (adjust x to move left/right)

# === PANEL B + C combined block ===
gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[1], height_ratios=[1, 2], hspace=0.15)

# --- Panel B ---
gs_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_right[0], wspace=0.3)
axs_b = [fig.add_subplot(gs_b[i]) for i in range(2)]

for config, ax in zip(curve_configs, axs_b):
    df = config["df"]
    label = config["label"]
    T_opt_local = config["T_opt_local"]
    T_min = np.min(df["T"].values)
    for dot_key in config["dot_keys"]:
        dot_info = mse_dots[dot_key]
        ax.scatter(T_min, dot_info["value"], color=dot_info["color"], label=dot_key, zorder=5)
    if ax is axs_b[1]:
        for name, y_val in ref_lines.items():
            ax.axhline(y=y_val, color='white', linestyle='--', linewidth=1.0, zorder=0)
            ax.plot([1], [y_val], marker='x', color='black', markersize=8,
                    transform=ax.get_yaxis_transform(), clip_on=False)
            ax.text(1.02, y_val, name, fontsize=9, color='black',
                    va='center', ha='left', transform=ax.get_yaxis_transform())
    ax.axvline(x=T_opt_local, color=background_color, linestyle='-', linewidth=1.5, alpha=0.4, zorder=0)
    ax.text(T_opt_local - T_opt_local * 0.25, 0.23, fr'$T = {T_opt_local:.5f}$',
            fontsize=9, color=background_color, ha='right', va='top')
    ax.plot(df["T"].values, df["MSE_GR_01"].values, color="gray", linestyle='-', linewidth=2)
    ax.plot(df["T"].values, df["MSE_GR_02"].values, color="black", linestyle='-', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel(r'$T$', fontsize=10)
    ax.set_yticks(np.arange(0.05, 0.275, 0.025))
    ax.set_title(f"MSE Curves for {label}", fontsize=11)  # Set consistent title size

axs_b[0].set_ylabel("MSE", fontsize=10)
legend_b = axs_b[1].legend(handles=shared_legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, fontsize=9)
legend_b.get_frame().set_edgecolor('black')

fig.text(0.51, 0.965, "B", fontsize=18, fontweight='bold', va='top', ha='left')  # Panel B label (adjust x to control proximity)

# --- Panel C ---
gs_c = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_right[1], hspace=0.3)
axs_c = [fig.add_subplot(gs_c[i]) for i in range(2)]

for i, label in enumerate(['EFM', 'EFV']):
    ax = axs_c[i]
    ax.set_title(titles_c[label], fontsize=11, pad=10)  # Set consistent title size
    ax.hist(lambdas[label]['max'][0], bins=100, weights=lambdas[label]['max'][1], alpha=0.4, color=color_max_T)
    ax.hist(lambdas[label]['min'][0], bins=100, weights=lambdas[label]['min'][1], alpha=0.4, color=color_min_T)
    ax.hist(lambdas[label]['best'][0], bins=100, weights=lambdas[label]['best'][1], alpha=0.6, color=color_best_T)
    sns.kdeplot(data=dfs[label]['max'], x='\u03bb', weights='w', color=color_max_T, fill=True, alpha=0.25, linewidth=2, ax=ax)
    sns.kdeplot(data=dfs[label]['min'], x='\u03bb', weights='w', color=color_min_T, fill=True, alpha=0.25, linewidth=2, ax=ax)
    sns.kdeplot(data=dfs[label]['best'], x='\u03bb', weights='w', color=color_best_T, fill=True, alpha=0.3, linewidth=2, ax=ax)
    ax.axvline(means[label]['max'], color=color_max_T, linestyle=':', linewidth=1.5)
    ax.axvline(means[label]['min'], color=color_min_T, linestyle=':', linewidth=1.5)
    ax.axvline(means[label]['best'], color=color_best_T, linestyle=':', linewidth=1.5)
    ax.set_ylabel("Frequency (Cell Population)", fontsize=10)
    if label == 'EFV':
        ax.set_xlabel(r'Growth Rate $\lambda\, \mathrm{in}\ \mathrm{h}^{-1}$ (Glucose Uptake 2.83 mmol/gDWh)', fontsize=10)
    yticks = ax.get_yticks()
    max_y = yticks[-1] if yticks[-1] != 0 else 1
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y / max_y:.1f}" for y in yticks])

legend_c = axs_c[0].legend(handles=handles_panelc, fontsize=9, frameon=True, fancybox=True,
                           facecolor='white', loc='upper left', bbox_to_anchor=(1.02, 1.0))
legend_c.get_frame().set_edgecolor('black')

fig.text(0.51, 0.63, "C", fontsize=18, fontweight='bold', va='top', ha='left')  # Panel C label

# === Final Layout Settings ===
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.2)  # wspace: gap between A and B/C, hspace: vertical spacing inside B/C
fig.savefig(output_path / "IMB_Combined_Tightened.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "IMB_Combined_Tightened.svg", bbox_inches='tight')
print("\u2713 Saved: IMB_Combined_Tightened.png and .svg")

print(' ✅ Done!')

print('Everything complete! 🎉')
