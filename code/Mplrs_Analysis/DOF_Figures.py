# ════════════════════════════════════════════════════════════════
# 1. Import Packages
# ════════════════════════════════════════════════════════════════

# Standard Library 
from pathlib import Path
import sys

# Core Scientific Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scientific Tools 
from scipy.optimize import curve_fit
from scipy.interpolate import griddata

# COBRApy 
from cobra.io import read_sbml_model
from cobra.util.array import create_stoichiometric_matrix

# ════════════════════════════════════════════════════════════════
# 2. Import & Export Paths
# ════════════════════════════════════════════════════════════════

# Set the project root path
notebook_dir = Path(__file__).parent if '__file__' in globals() else Path().resolve()
project_root = notebook_dir.parent.parent  

# define paths to data directories
raw_sbml_path = project_root / "data" / "raw" / "sbml_files"
temp_sbml_path = raw_sbml_path / "temporary_files"

processed_csv_path = project_root / "data" / "processed" / "csv_files"
processed_sbml_path = project_root / "data" / "processed" / "sbml_files"
martino_path = processed_sbml_path / "klamt_martino_model_mplrs.xml"
figures_path = project_root / "data" / "processed" / "figures"

# ════════════════════════════════════════════════════════════════
# 3. Data Import
# ════════════════════════════════════════════════════════════════

print('📦 Importing Data...')

run_time_list = [22, 31, 66, 263, 712, 1855] # from mplrs log files
EFVs_list = [78644, 78645, 230615, 1106576, 4963589, 14612038] # from mplrs log files
runtime_df = pd.read_csv(processed_csv_path / 'mplrs_analysis_core_data.csv')

# Paths and data
model_versions = ["v_Martino"] + [f"v{i}" for i in range(5)]
model_paths = [martino_path] + [temp_sbml_path / f"model_v{i}_quickcheck3.xml" for i in range(5)]

# ════════════════════════════════════════════════════════════════
# 4. Data Preparation
# ════════════════════════════════════════════════════════════════

### load data from all different DOF models ###

# List to store results 
dof_data = []

# Process each model
for version, path, runtime, efvs in zip(model_versions, model_paths, run_time_list, EFVs_list):
    model = read_sbml_model(path)
    stoich = create_stoichiometric_matrix(model)
    rank = np.linalg.matrix_rank(stoich)
    num_rxns = len(model.reactions)
    dof = num_rxns - rank

    dof_data.append({
        "model_version": version,
        "num_reactions": num_rxns,
        "rank": rank,
        "degrees_of_freedom": dof,
        "run_time": runtime,
        "num_EFVs": efvs
    })

# Convert to DataFrame
dof_df = pd.DataFrame(dof_data)

### data for panel a ###

time_thresholds = {
    '1 hour': 3600,
    '1 day': 86400,
    '1 week': 604800,
    '1 month': 30.44 * 86400
}

# Define cutoff: 13.79 billion years in seconds
universe_age_seconds = 13.79e9 * 365.25 * 86400  # ≈ 4.35e17

# Gather the data
x_dof = dof_df['degrees_of_freedom'].to_numpy()
y_runtime = dof_df['run_time'].to_numpy()
y_efvs = dof_df['num_EFVs'].to_numpy()

# Filter to valid data
mask = (y_runtime > 0) & (y_efvs > 0)
x_dof = x_dof[mask]
y_runtime = y_runtime[mask]
y_efvs = y_efvs[mask]

# Fit exponential models (log space)
p_rt = np.polyfit(x_dof, np.log(y_runtime), 1)
a_rt = np.exp(p_rt[1])
b_rt = p_rt[0]

p_efv = np.polyfit(x_dof, np.log(y_efvs), 1)
a_efv = np.exp(p_efv[1])
b_efv = p_efv[0]

# Calculate cutoff x where runtime = age of universe
cutoff_x = (np.log(universe_age_seconds) - np.log(a_rt)) / b_rt

# X range for fitted curves (stop at cutoff)
x_fitted = np.linspace(20, cutoff_x, 300)
y_rt_fit = a_rt * np.exp(b_rt * x_fitted)
y_efv_fit = a_efv * np.exp(b_efv * x_fitted)

### data for panel b ###

# Only replace Version where it matches "ecoli_vM"
runtime_df['Version'] = runtime_df['Version'].apply(lambda x: 'Martino' if 'ecoli_vM' in x else x)

# Extract cores only if pattern matches, leave other values untouched
runtime_df['Cores'] = runtime_df['Cores'].where(~runtime_df['Cores'].str.contains(r'ecoli_vM_\d+\.log'),  # keep if it doesn't match
    runtime_df['Cores'].str.extract(r'_(\d+)\.log$')[0])

# Convert to numeric only where values are not missing
runtime_df['Cores'] = pd.to_numeric(runtime_df['Cores'], errors='coerce')

# Compute Threads
runtime_df['Threads'] = runtime_df['Cores'] * 48

### data for panel c ###

data = {
    "EColiCentral": {
        "x": [5.911116759090997, 11.803572476129743, 17.89846453165723, 23.66945340595023, 
              29.84416535759842, 35.57086896928722, 41.893072886798016, 47.76904931075511, 
              55.539497038486076, 317.5165594041926],
        "y": [1412.9377004028654, 584.9981092263812, 391.1046883657927, 293.81023563308713, 
              238.48880606937985, 200.0338344088342, 169.64580080998428, 147.4606241103923, 
              128.51002179208814, 27.3255433386538]
    },
    "iPS189": {
        "x": [5.99645834114054, 11.97706275115881, 17.984343487884765, 23.922459556003748,
              29.945683443735813, 35.91016793057282, 42.0330656106426, 47.82809272517059,
              56.0001704458186, 199.23788413352992],
        "y": [8958.75777228873, 3696.0884514208633, 2461.6191275532005, 1851.1592778783213,
              1497.0813300301274, 1237.043290935754, 1064.8883914323333, 929.4714299928731,
              964.2548881677122, 207.25583190451817]
    },
    "EColiCore2": {
        "x": [31.616687221449475, 47.51796612469675, 63.18476766556022, 126.4723100695787, 666.1131422891385],
        "y": [118408.77747950132, 60987.311247331265, 42414.60966961601, 30518.325349775278, 5528.709691762037]
    }
}

print(' ✅ Done!')

# ════════════════════════════════════════════════════════════════
# 5. Plot Design 
# ════════════════════════════════════════════════════════════════

print('Set style and plot theme...')

### general ###

# Set Seaborn style
sns.set_style("white")

# define colors

rocket_colors = sns.color_palette("rocket", 10)

### panel a ###

color_rt = rocket_colors[1]
color_efv = rocket_colors[5]
color_lines = "black"

### panel b ###

version_color_map = {    
    '2': rocket_colors[0],
    '1': rocket_colors[3],
    '0': rocket_colors[6],
    'Martino': rocket_colors[9]
}

legend_labels = {
    'Martino': '23 DoF',
    '0': '24 DoF',
    '1': '25 DoF',
    '2': '26 DoF'
}

print(' ✅ Done!')

# ════════════════════════════════════════════════════════════════
# 6. Plots
# ════════════════════════════════════════════════════════════════

print('Generating Plots...')

### panel a single ###

print(' Panel A...')

# Create plot
fig, ax = plt.subplots(figsize=(8, 4))

# Plot data (scatter with no legend label)
ax.scatter(x_dof, y_runtime, color=color_rt, s=20, label=None, zorder=2)
ax.scatter(x_dof, y_efvs, color=color_efv, s=20, label=None, zorder=2)

# Plot fits
ax.plot(x_fitted, y_rt_fit, color=color_rt, label='Run Time [s]', zorder=3)
ax.plot(x_fitted, y_efv_fit, color=color_efv, label='Enumerated EFVs', zorder=3)

# Log scale and limits
ax.set_yscale('log')
ax.set_ylim(1e0, 1e27)  # Updated max y-limit
ax.set_xlim(20, cutoff_x)

# Labels and title
ax.set_xlabel('Degrees of Freedom', fontsize=10)
ax.set_ylabel('Run Time / EFV Count', fontsize=10)
ax.set_title('', fontsize=9)
ax.tick_params(axis='both', labelsize=9)

# Add subtle custom grid
ax.grid(True, which='both', linewidth=0.4, color='gray', alpha=0.3)

# Legend styling
legend = ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0.77, 0.18), frameon=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_alpha(1.0)

# Axes border styling
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)

# Plot vertical lines for each time threshold
for label, seconds in time_thresholds.items():
    x_threshold = (np.log(seconds) - np.log(a_rt)) / b_rt
    if 20 <= x_threshold <= cutoff_x:
        ax.axvline(x=x_threshold, linestyle=':', color=color_lines, linewidth=1.2, zorder=1)
        ax.text(
            x_threshold + 0.5,   # Slight right offset
            1e15,              # New fixed y-position
            label,
            rotation=90,
            verticalalignment='bottom',
            fontsize=8,
            color=color_lines
        )

# Final layout
plt.tight_layout(pad=0.1)
plt.savefig(figures_path / "Mplrs_analyis_panel_a.png", dpi=300, bbox_inches='tight')
plt.savefig(figures_path / "Mplrs_analyis_panel_a.svg", bbox_inches='tight')

### panel b single ###

print(' Panel B...')

# Create plot
fig, ax = plt.subplots(figsize=(8, 4))

# Plot curves for each version
for version_key, color in version_color_map.items():
    df_sub = runtime_df[runtime_df['Version'] == version_key]
    
    x = df_sub['Threads'].to_numpy()
    y = df_sub['Runtime'].to_numpy()
    
    # Sort for smooth line
    sorted_idx = np.argsort(x)
    x = x[sorted_idx]
    y = y[sorted_idx]
    
    # Scatter and line
    ax.scatter(x, y, color=color, s=20)
    ax.plot(x, y, color=color, label=legend_labels[version_key])

# Labels and limits
ax.set_xlabel('CPU cores', fontsize=10)
ax.set_ylabel('Runtime [s]', fontsize=10)
ax.set_title('', fontsize=11)
ax.tick_params(axis='both', labelsize=9)
#ax.set_yscale('log')

# Add subtle custom grid
ax.grid(True, which='both', linewidth=0.4, color='gray', alpha=0.3)

# Legend styling
legend = ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(0.982, 0.97), frameon=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.8)

# Border styling
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)

# Final layout
plt.tight_layout(pad=0.1)
plt.savefig(figures_path / "Mplrs_analyis_panel_b.png", dpi=300, bbox_inches='tight')
plt.savefig(figures_path / "Mplrs_analyis_panel_b.svg", bbox_inches='tight')

### panel a and b combined ###

print(' Panel A and B combined...')

# Create the figure with 2 vertical subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

# ───── Panel A ─────

# Plot EFVs and runtime
ax1.scatter(x_dof, y_runtime, color=color_rt, s=20, label='Run Time [s]')
ax1.scatter(x_dof, y_efvs, color=color_efv, s=20, label='Enumerated EFVs')
ax1.plot(x_fitted, y_rt_fit, color=color_rt)
ax1.plot(x_fitted, y_efv_fit, color=color_efv)

# Axis config
ax1.set_yscale('log')
ax1.set_ylim(1e0, 1e27)
ax1.set_xlim(20, cutoff_x)
ax1.set_ylabel('Run Time / EFV Count', fontsize=10)
ax1.tick_params(axis='both', labelsize=9)
ax1.grid(True, which='both', linewidth=0.4, color='gray', alpha=0.3)
for spine in ax1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)

# Vertical threshold lines
for label, seconds in time_thresholds.items():
    x_threshold = (np.log(seconds) - np.log(a_rt)) / b_rt
    if 20 <= x_threshold <= cutoff_x:
        ax1.axvline(x=x_threshold, linestyle=':', color=color_lines, linewidth=1.2)
        ax1.text(x_threshold + 0.5, 1e15, label, rotation=90, verticalalignment='bottom', fontsize=8, color=color_lines)

# Subplot label "A"
ax1.text(-0.1, 1.03, 'A', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# Create legend and store it in a variable
legend = ax1.legend(loc='upper left', bbox_to_anchor=(0.76, 0.18), fontsize=9, frameon=True)

# Customize the frame
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_alpha(1.0)  # Optional: make it fully opaque


# ───── Panel B ─────

for version_key, color in version_color_map.items():
    df_sub = runtime_df[runtime_df['Version'] == version_key]
    x = df_sub['Threads'].to_numpy()
    y = df_sub['Runtime'].to_numpy()
    sorted_idx = np.argsort(x)
    x = x[sorted_idx]
    y = y[sorted_idx]
    ax2.scatter(x, y, color=color, s=20)
    ax2.plot(x, y, color=color, label=legend_labels[version_key])

# Axis config
ax2.set_xlabel('CPU cores', fontsize=10)
ax2.set_ylabel('Runtime [s]', fontsize=10)
ax2.tick_params(axis='both', labelsize=9)
ax2.grid(True, which='both', linewidth=0.4, color='gray', alpha=0.3)
for spine in ax2.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)

# Subplot label "B"
ax2.text(-0.1, 1.03, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# Create legend and store it in a variable
legend = ax2.legend(loc='upper left', bbox_to_anchor=(0.8, 0.97), fontsize=9, frameon=True)

# Customize the frame
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_alpha(1.0)  # Optional: make it fully opaque

# Save figure
plt.savefig(figures_path / "Mplrs_analyis_panel_ab_combined.png", dpi=300, bbox_inches='tight')
plt.savefig(figures_path / "Mplrs_analyis_panel_ab_combined.svg", bbox_inches='tight')

### panel c ###

print(' Panel C...')

# Set style
sns.set(style="whitegrid")

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)

# Plot each model
for ax, (model, values) in zip(axes, data.items()):
    ax.plot(values["x"], values["y"], marker="o", color="dimgray")
    ax.set_title(model)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Wall Time / [s]")
    ax.grid(True)

# Apply tight layout
plt.tight_layout()
plt.savefig(figures_path / "Mplrs_analyis_panel_c.png", dpi=300, bbox_inches='tight')
plt.savefig(figures_path / "Mplrs_analyis_panel_c.svg", bbox_inches='tight')

print(' ✅ Done!')

# ════════════════════════════════════════════════════════════════
# 6. Export data
# ════════════════════════════════════════════════════════════════

dof_df.to_csv(processed_csv_path / 'mplrs_analysis_metadata.csv')

print('Everything complete! 🎉')