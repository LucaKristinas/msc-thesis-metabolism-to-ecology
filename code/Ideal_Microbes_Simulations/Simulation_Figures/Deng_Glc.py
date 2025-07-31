# ════════════════════════════════════════════════════════════════
# 1. Import Packages
# ════════════════════════════════════════════════════════════════

# Core Libraries
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Add Repo Root to sys.path 
project_root = Path(__file__).resolve().parents[3]  
sys.path.append(str(project_root))

# Import Helper Functions
from src.utils import (smooth_line2)

# ════════════════════════════════════════════════════════════════
# 2. Import & Export Paths
# ════════════════════════════════════════════════════════════════


# Basic Paths
raw_data_path = project_root / "data" / "raw"
processed_data_path = project_root / "data" / "processed"

# Raw Data
csv_raw_path = raw_data_path / "csv_files"

# Processed Data 
csv_processed_path = processed_data_path / "csv_files"
export_path = processed_data_path / "figures"

# ════════════════════════════════════════════════════════════════
# 3. Data Import
# ════════════════════════════════════════════════════════════════

# Manually rmoved point of glc measurement at 480min, because glc increases there again which is not possible! (clean version)
Deng_df = pd.read_csv(csv_raw_path / "Deng_glc_biomass_clean.csv")

OFV_df = pd.read_csv(csv_processed_path / "E_coli_core_sim_OFVs_Deng.csv", sep=';')
EFV_df = pd.read_csv(csv_processed_path / "E_coli_core_sim_EFVs_Deng.csv", sep=';')

# ════════════════════════════════════════════════════════════════
# 4. Data Processing
# ════════════════════════════════════════════════════════════════

#--------------------------------------------------------------------------------------------------------
# Deng Data (Fig. 6?) 
#--------------------------------------------------------------------------------------------------------

# Store raw and smoothed data
raw_all = {}
smoothed_all = {}

# Define and convert times to hours
Deng_df['biomass_t_hr'] = Deng_df['biomass_t_min'] / 60
Deng_df['glucose_t_hr'] = Deng_df['glucose_t_min'] / 60

# Biomass Data 
raw_all['biomass'] = {
    'time': Deng_df['biomass_t_hr'].values,
    'value': Deng_df['biomass_mean_mg_l'].values,
    'error': Deng_df['biomass_SD_mg_l'].values
}

time_biomass_smooth, biomass_smooth = smooth_line2(Deng_df['biomass_t_hr'].values, Deng_df['biomass_mean_mg_l'].values)
_, biomass_SD_smooth = smooth_line2(Deng_df['biomass_t_hr'].values, Deng_df['biomass_SD_mg_l'].values)

smoothed_all['biomass'] = {
    'time': time_biomass_smooth,
    'value': biomass_smooth,
    'error': biomass_SD_smooth
}

# Glucose Data 
raw_all['glucose'] = {
    'time': Deng_df['glucose_t_hr'].values,
    'value': Deng_df['glucose_mean_mM'].values,
    'error': Deng_df['glucose_SD_mM'].values
}

time_glucose_smooth, glucose_smooth = smooth_line2(Deng_df['glucose_t_hr'].values, Deng_df['glucose_mean_mM'].values)
_, glucose_SD_smooth = smooth_line2(Deng_df['glucose_t_hr'].values, Deng_df['glucose_SD_mM'].values)

smoothed_all['glucose'] = {
    'time': time_glucose_smooth,
    'value': glucose_smooth,
    'error': glucose_SD_smooth
}

# Get smoothed time and biomass (CDW) values
time_smooth = smoothed_all["biomass"]["time"]
cdw_smooth = smoothed_all["biomass"]["value"]

# Calculate specific growth rate µ = dX/dt / X
dcdw_dt = np.gradient(cdw_smooth, time_smooth)
specific_growth_rate = dcdw_dt / cdw_smooth

# Filter valid values (avoid nan, inf, div0)
valid = np.isfinite(specific_growth_rate)
mu_valid = specific_growth_rate[valid]
time_valid = time_smooth[valid]

# Find maximum specific growth rate (µmax)
mu_max = np.max(mu_valid)

# Define exponential phase as range within ±5% of µmax
tolerance = 0.05 * mu_max
exp_mask = np.abs(mu_valid - mu_max) < tolerance
mu_peak_times = time_valid[exp_mask]

exp_phase_start = mu_peak_times.min()
exp_phase_end = mu_peak_times.max()

# Find the index of the last time point before exponential phase start
idx_biomass = np.where(smoothed_all["biomass"]["time"] < exp_phase_start)[0][-1]
idx_glucose = np.where(smoothed_all["glucose"]["time"] < exp_phase_start)[0][-1]

# Extract time and value before exponential phase for biomass and glucose
t_biomass = smoothed_all["biomass"]["time"][idx_biomass]
v_biomass = smoothed_all["biomass"]["value"][idx_biomass]

t_glucose = smoothed_all["glucose"]["time"][idx_glucose]
v_glucose = smoothed_all["glucose"]["value"][idx_glucose]

# Print summary
print(f"μmax = {mu_max:.3f} 1/h during exponential phase from {exp_phase_start:.2f}h to {exp_phase_end:.2f}h")
print("→ Smoothed values just before exponential phase:")
print(f"   Biomass @ {t_biomass:.2f}h = {v_biomass:.4f} mg/L")
print(f"   Glucose @ {t_glucose:.2f}h = {v_glucose:.4f} mM")

### EFV and OFV results

# Store Raw Fit and smooth fit

variables = ["glc", "frc", "cdw"]
column_map = {
    "glc": "glc",
    "frc": "frc",
    "cdw": "mic"
}

sim_data = {
    "ofv": {
        "df": OFV_df,
        "time": OFV_df["time_h"].values
    },
    "efv": {
        "df": EFV_df,
        "time": EFV_df["time_h"].values
    }
}

for sim_type in sim_data:
    df = sim_data[sim_type]["df"]
    time = sim_data[sim_type]["time"]
    sim_data[sim_type]["raw"] = {}
    sim_data[sim_type]["smooth"] = {}
    
    for var in variables:
        values = df[column_map[var]].values
        sim_data[sim_type]["raw"][var] = values
        t_smooth, y_smooth = smooth_line2(time, values)
        sim_data[sim_type]["smooth"][var] = {
            "time": t_smooth,
            "value": y_smooth
        }

# ════════════════════════════════════════════════════════════════
# 6. Plots
# ════════════════════════════════════════════════════════════════

#-----------------------------------------------------------------
# General Settings
#-----------------------------------------------------------------

# Define colors
colors = sns.color_palette("rocket", 10)
color_cdw = colors[0]
color_glc = colors[4]
color_frc = colors[8]
color_gal = colors[9]

#-----------------------------------------------------------------
# Glucose (Fig 6) Panel 
#-----------------------------------------------------------------

# Create 3-panel plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=False)

# Titles
titles = [
    "Empirical Data",
    "EFV Simulation",
    "OFV Simulation"
]

# 1. Empirical Data (Deng)
# Molar mass of glucose in g/mM
GLUCOSE_MOLAR_MASS = 180.16 / 1000 

# Axes
ax1 = axes[0]

# Plot glucose in mg/L
ax1.errorbar(
    raw_all['glucose']['time'],
    raw_all['glucose']['value'] * GLUCOSE_MOLAR_MASS,
    yerr=raw_all['glucose']['error'] * GLUCOSE_MOLAR_MASS,
    fmt='o',
    color=color_glc,
    capsize=3
)

ax1.plot(
    smoothed_all['glucose']['time'],
    smoothed_all['glucose']['value'] * GLUCOSE_MOLAR_MASS,
    color=color_glc
)

ax1.fill_between(
    smoothed_all['glucose']['time'],
    (smoothed_all['glucose']['value'] - smoothed_all['glucose']['error']) * GLUCOSE_MOLAR_MASS,
    (smoothed_all['glucose']['value'] + smoothed_all['glucose']['error']) * GLUCOSE_MOLAR_MASS,
    color=color_glc,
    alpha=0.3
)

# Axis settings
ax1.set_ylabel("Glucose [g/L]", fontsize=10)  # <-- updated unit label
ax1.set_xlabel("Time [h]", fontsize=10)
ax1.set_xlim(0, 12)
ax1.set_ylim(0,2.5)
ax1.tick_params(axis='both', labelsize=10)
ax1.set_title(titles[0], fontsize=11)

# Background shading
fig.canvas.draw()  # Ensure limits are calculated
ax1.axvspan(ax1.get_xlim()[0], 6.5, color="gray", zorder=0, alpha=0.1)
ax1.axvspan(6.5, 9, color='darkred', alpha=0.1, zorder=0)

ax1r = ax1.twinx()
ax1r.errorbar(
    raw_all['biomass']['time'],
    raw_all['biomass']['value'] / 1000,
    yerr=raw_all['biomass']['error'] / 1000,
    fmt='o',
    color=color_cdw,
    capsize=3
)

ax1r.plot(
    smoothed_all['biomass']['time'],
    smoothed_all['biomass']['value'] / 1000,
    color=color_cdw
)

ax1r.fill_between(
    smoothed_all['biomass']['time'],
    (smoothed_all['biomass']['value'] - smoothed_all['biomass']['error']) / 1000,
    (smoothed_all['biomass']['value'] + smoothed_all['biomass']['error']) / 1000,
    color=color_cdw,
    alpha=0.3
)

#ax1r.set_ylabel("Biomass [gCDW/L]", fontsize=10)  # updated unit label
ax1r.tick_params(axis='y', labelsize=10)
ax1r.set_yticklabels([])  # optional: hides tick labels
ax1r.set_ylim(0,1)

# 2. OFV Simulation
ax2 = axes[1]

ax2.plot(sim_data["efv"]["smooth"]["glc"]["time"], (sim_data["efv"]["smooth"]["glc"]["value"] * GLUCOSE_MOLAR_MASS),
         color=color_glc, linewidth=3)
ax2.set_xlabel("Time [h]", fontsize=10)
ax2.set_ylim(0,2.5)
ax2.tick_params(axis='both', labelsize=10)
ax2.set_title(titles[1], fontsize=11)
ax2.set_yticklabels([])
ax2.set_xlim(0,3)
ax2.axvspan(ax2.get_xlim()[0], 2.525, color="darkred", zorder=0, alpha=0.1)

ax2r = ax2.twinx()
ax2r.plot(sim_data["efv"]["smooth"]["cdw"]["time"], sim_data["efv"]["smooth"]["cdw"]["value"],
          color=color_cdw, linewidth=3)
#ax2r.set_ylim()
ax2r.tick_params(axis='y', labelsize=10)
ax2r.set_yticklabels([])
ax2r.set_ylim(0,1)

# 3. EFV Simulation
ax3 = axes[2]

ax3.plot(sim_data["ofv"]["smooth"]["glc"]["time"], (sim_data["ofv"]["smooth"]["glc"]["value"] * GLUCOSE_MOLAR_MASS),
         color=color_glc, linewidth=3)
ax3.set_xlabel("Time [h]", fontsize=10)
ax3.set_ylim(0,2.5)
ax3.tick_params(axis='both', labelsize=10)
ax3.set_title(titles[2], fontsize=11)
ax3.set_yticklabels([])
ax3.set_xlim(0,3)
ax3.axvspan(ax3.get_xlim()[0], 2.34, color="darkred", zorder=0, alpha=0.1)

ax3r = ax3.twinx()
ax3r.plot(sim_data["ofv"]["smooth"]["cdw"]["time"], sim_data["ofv"]["smooth"]["cdw"]["value"],
          color=color_cdw, linewidth=3)
ax3r.set_ylabel("Biomass [gCDW/L]", fontsize=10)
#ax3r.set_ylim()
ax3r.tick_params(axis='y', labelsize=10)
ax3r.set_ylim(0,1)
# Legend

# Create legend handles with colored dots

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Glucose', markerfacecolor=color_glc, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Biomass', markerfacecolor=color_cdw, markersize=8)
]

legend = ax1.legend(
    handles=legend_elements,
    loc='upper left',
    #bbox_to_anchor=(0.03, 0.83),
    frameon=True,
    framealpha=1,
    edgecolor='black',
    fancybox=True,
    fontsize=9
)

# Make the frame thinner
legend.get_frame().set_linewidth(0.8)  # e.g. 0.8 for thin, default is ~1.5

# Save and show
plt.tight_layout()
plt.savefig(export_path / "Deng_Sim_Glc_Fig_6.png", dpi=300, bbox_inches="tight")
plt.savefig(export_path / "Deng_Sim_Glc_Fig_6.svg", format="svg", bbox_inches="tight")
plt.show()
plt.close()

#-----------------------------------------------------------------
# Glucose (Fig 6) Ind Versions
#-----------------------------------------------------------------

# ────────────────────────────────────────────────────────────────
# Save each subplot individually (5x4 size)
# ────────────────────────────────────────────────────────────────

# 1. Empirical Only
fig_emp, ax1 = plt.subplots(figsize=(5, 4))
ax1.errorbar(
    raw_all['glucose']['time'],
    raw_all['glucose']['value'] * GLUCOSE_MOLAR_MASS,
    yerr=raw_all['glucose']['error'] * GLUCOSE_MOLAR_MASS,
    fmt='o',
    color=color_glc,
    capsize=3
)
ax1.plot(
    smoothed_all['glucose']['time'],
    smoothed_all['glucose']['value'] * GLUCOSE_MOLAR_MASS,
    color=color_glc
)
ax1.fill_between(
    smoothed_all['glucose']['time'],
    (smoothed_all['glucose']['value'] - smoothed_all['glucose']['error']) * GLUCOSE_MOLAR_MASS,
    (smoothed_all['glucose']['value'] + smoothed_all['glucose']['error']) * GLUCOSE_MOLAR_MASS,
    color=color_glc,
    alpha=0.3
)
ax1.set_ylabel("Glucose [g/L]", fontsize=10)
ax1.set_xlabel("Time [h]", fontsize=10)
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 2.5)
ax1.tick_params(axis='both', labelsize=10)
ax1.set_title("Empirical Data", fontsize=11)
fig_emp.canvas.draw()
ax1.axvspan(ax1.get_xlim()[0], 6.5, color="gray", zorder=0, alpha=0.1)
ax1.axvspan(6.5, 9, color='darkred', alpha=0.1, zorder=0)
ax1r = ax1.twinx()
ax1r.errorbar(
    raw_all['biomass']['time'],
    raw_all['biomass']['value'] / 1000,
    yerr=raw_all['biomass']['error'] / 1000,
    fmt='o',
    color=color_cdw,
    capsize=3
)
ax1r.plot(
    smoothed_all['biomass']['time'],
    smoothed_all['biomass']['value'] / 1000,
    color=color_cdw
)
ax1r.fill_between(
    smoothed_all['biomass']['time'],
    (smoothed_all['biomass']['value'] - smoothed_all['biomass']['error']) / 1000,
    (smoothed_all['biomass']['value'] + smoothed_all['biomass']['error']) / 1000,
    color=color_cdw,
    alpha=0.3
)
ax1r.set_ylim(0, 1)
ax1r.set_ylabel("Biomass [gCDW/L]", fontsize=10)

plt.tight_layout()
fig_emp.savefig(export_path / "Deng_Empirical_Glc_Fig6.png", dpi=300, bbox_inches="tight")
fig_emp.savefig(export_path / "Deng_Empirical_Glc_Fig6.svg", format="svg", bbox_inches="tight")
plt.close(fig_emp)

# 2. EFV Only
fig_efv, ax2 = plt.subplots(figsize=(5, 4))
ax2.plot(sim_data["efv"]["smooth"]["glc"]["time"], sim_data["efv"]["smooth"]["glc"]["value"] * GLUCOSE_MOLAR_MASS,
         color=color_glc, linewidth=3)
ax2.set_xlabel("Time [h]", fontsize=10)
ax2.set_ylabel("Glucose [g/L]", fontsize=10)
ax2.set_xlim(0, 3)
ax2.set_ylim(0, 2.5)
ax2.set_title("EFV Simulation", fontsize=11)
ax2.axvspan(ax2.get_xlim()[0], 2.525, color="darkred", zorder=0, alpha=0.1)
ax2.tick_params(axis='both', labelsize=10)
ax2r = ax2.twinx()
ax2r.plot(sim_data["efv"]["smooth"]["cdw"]["time"], sim_data["efv"]["smooth"]["cdw"]["value"],
          color=color_cdw, linewidth=3)
ax2r.set_ylim(0, 1)
ax2r.tick_params(axis='y', labelsize=10)
ax2r.set_ylabel("Biomass [gCDW/L]", fontsize=10)

plt.tight_layout()
fig_efv.savefig(export_path / "Deng_EFV_Glc_Fig6.png", dpi=300, bbox_inches="tight")
fig_efv.savefig(export_path / "Deng_EFV_Glc_Fig6.svg", format="svg", bbox_inches="tight")
plt.close(fig_efv)

# 3. OFV Only
fig_ofv, ax3 = plt.subplots(figsize=(5, 4))
ax3.plot(sim_data["ofv"]["smooth"]["glc"]["time"], sim_data["ofv"]["smooth"]["glc"]["value"] * GLUCOSE_MOLAR_MASS,
         color=color_glc, linewidth=3)
ax3.set_xlabel("Time [h]", fontsize=10)
ax3.set_ylabel("Glucose [g/L]", fontsize=10)
ax3.set_xlim(0, 3)
ax3.set_ylim(0, 2.5)
ax3.set_title("OFV Simulation", fontsize=11)
ax3.axvspan(ax3.get_xlim()[0], 2.34, color="darkred", zorder=0, alpha=0.1)
ax3.tick_params(axis='both', labelsize=10)
ax3r = ax3.twinx()
ax3r.plot(sim_data["ofv"]["smooth"]["cdw"]["time"], sim_data["ofv"]["smooth"]["cdw"]["value"],
          color=color_cdw, linewidth=3)
ax3r.set_ylim(0, 1)
ax3r.tick_params(axis='y', labelsize=10)
ax3r.set_ylabel("Biomass [gCDW/L]", fontsize=10)

plt.tight_layout()
fig_ofv.savefig(export_path / "Deng_OFV_Glc_Fig6.png", dpi=300, bbox_inches="tight")
fig_ofv.savefig(export_path / "Deng_OFV_Glc_Fig6.svg", format="svg", bbox_inches="tight")
plt.close(fig_ofv)
