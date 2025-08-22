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
from src.utils import (smooth_line)

# ════════════════════════════════════════════════════════════════
# 3. Import & Export Paths
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
# 4. Data Import
# ════════════════════════════════════════════════════════════════

print("📦 Importing data...")

# Ideal Microbes (mimick Lendenmann after 2.1h (lag phase ca done): 0.015 mM Glc and 0.012 mM Frc and 0.0098 g/L cells)

sim_glcfrc_OFV = pd.read_csv(csv_processed_path / "E_coli_core_sim_OFVs_GlcFrc.csv", sep=";")
sim_glcfrc_EFV = pd.read_csv(csv_processed_path / "E_coli_core_sim_EFVs_GlcFrc.csv", sep=";")
sim_glcgal_iJO1366 = pd.read_csv(csv_processed_path / "iJO1366_sim_GlcGal.csv", sep=";")

# LendenMann Data for Glucose / Fructose (Fig 9.2) and Glucose / Galactose (Fig 9.1)

files_info = {
    "glcfrc_frc.csv": ["time_h","frc_mg_per_l"],
    "glcfrc_glc.csv": ["time_h","glc_mg_per_l"],
    "glcfrc_OD546.csv": ["time_h","OD546"],
    "glcgal_gal_values.csv": ["time_h","gal_mg_per_l"],
    "glcgal_glc_values.csv": ["time_h","glc_mg_per_l"],
    "glcgal_OD546.csv": ["time_h","OD546"]
}

# Load and sort all DataFrames into a dictionary
dataframes = {
    name.replace(".csv", ""): pd.read_csv(csv_raw_path/ name, names=cols).sort_values("time_h")
    for name, cols in files_info.items()
}

print(' ✅ Done!')

# ════════════════════════════════════════════════════════════════
# 5. Data Processing
# ════════════════════════════════════════════════════════════════

print("Processing data...")

#---


# === Settings ===
MW_glucose = 180.16
MW_fructose = 180.16
MW_galactose = 180.16
OD_to_mg_L = 0.05
threshold_mgL = 0.001  # equivalent to 0.001 g/L

# === Functions ===
def report_empirical(df, label):
    df = df.copy()
    if "OD546" in df.columns:
        df["biomass_mg_L"] = df["OD546"] * OD_to_mg_L * 1000
        col_to_report = "biomass_mg_L"
    else:
        col_to_report = [c for c in df.columns if c.endswith("_mg_per_l")][0]

    print(f"\n===== Empirical: {label} =====")
    print("First row:")
    print(df[[col_to_report]].head(1).to_string(index=False))
    print(f"\nMax value [mg/L]: {df[col_to_report].max():.2f}")
    if (df[col_to_report] < threshold_mgL).any():
        first_time = df.loc[df[col_to_report] < threshold_mgL, "time_h"].iloc[0]
        print(f"First time_h where value < {threshold_mgL} mg/L: {first_time}")

def report_simulation(df, label, sugar_cols):
    df = df.copy()
    for col in sugar_cols:
        df[f"{col}_g_L"] = df[col] * MW_glucose / 1000
    print(f"\n===== Simulation: {label} =====")
    print("First row (g/L):")
    print(df[[f"{col}_g_L" for col in sugar_cols] + ["mic"]].head(1).to_string(index=False))
    print(f"\nMax mic value: {df['mic'].max():.6f}")
    for col in sugar_cols:
        if (df[f"{col}_g_L"] < threshold_mgL / 1000).any():
            first_time = df.loc[df[f"{col}_g_L"] < threshold_mgL / 1000, "time_h"].iloc[0]
            print(f"First time_h where {col} < {threshold_mgL} mg/L: {first_time}")

# === Reports ===
# Empirical Glc/Frc
report_empirical(dataframes["glcfrc_glc"], "Glucose (empirical, Glc/Frc)")
report_empirical(dataframes["glcfrc_frc"], "Fructose (empirical, Glc/Frc)")
report_empirical(dataframes["glcfrc_OD546"], "Biomass from OD (empirical, Glc/Frc)")

# Empirical Glc/Gal
report_empirical(dataframes["glcgal_glc_values"], "Glucose (empirical, Glc/Gal)")
report_empirical(dataframes["glcgal_gal_values"], "Galactose (empirical, Glc/Gal)")
report_empirical(dataframes["glcgal_OD546"], "Biomass from OD (empirical, Glc/Gal)")

# Simulation Glc/Frc
report_simulation(sim_glcfrc_OFV, "Glc/Frc OFV", ["glc", "frc"])
report_simulation(sim_glcfrc_EFV, "Glc/Frc EFV", ["glc", "frc"])

# Simulation Glc/Gal
report_simulation(sim_glcgal_iJO1366, "Glc/Gal iJO1366", ["glc", "gal"])

print("\n✅ Done!")



#---

#--------------------------------------------------------------------------------------------------------
# Lendenmann Data for Glucose / Fructose (Fig 9.2) and Glucose / Galactose (Fig 9.1) 
#--------------------------------------------------------------------------------------------------------

# Convert OD546 → OD600 → gCDW/L (OD600 ≈ OD546 × 0.91, and OD600 × 0.33 = gCDW/L (Sauer et al., 1999))
# alternative used here: it was inoculated with ca 1 mg/L biomass 

# conversion_factor = 0.91 * 0.33  # 0.3003
conversion_factor = round(0.001 / dataframes["glcgal_OD546"].iloc[0, 1],2)
print(conversion_factor)

for key in ["glcfrc_OD546", "glcgal_OD546"]:
    df = dataframes[key]
    df["cdw_g_per_l"] = df["OD546"] * conversion_factor

# Convert OD546 to gCDW/L
for key in ["glcfrc_OD546", "glcgal_OD546"]:
    df = dataframes[key]
    df["cdw_g_per_l"] = df["OD546"] * conversion_factor

# Extract raw data and smooth it

# Create raw and smoothed dictionaries for all files

raw_all = {}
smoothed_all = {}

for name, df in dataframes.items():
    # Identify value column (assume it's the second column)
    time_col = "time_h"
    value_col = [col for col in df.columns if col != time_col][0]

    # Handle CDW conversion for OD546 data
    if value_col == "OD546":
        df = df.copy()
        df["cdw_g_per_l"] = df["OD546"] * conversion_factor
        value_col = "cdw_g_per_l"

    # Store raw values
    raw_all[name] = {
        "time": df[time_col].values,
        "value": df[value_col].values
    }

    # Smooth the data
    time_smooth, y_smooth = smooth_line(df[time_col].values, df[value_col].values)
    smoothed_all[name] = {
        "time": time_smooth,
        "value": y_smooth
    }

# extract exponential window + all the corresponding data

# Define datasets and their corresponding metabolite keys in smoothed_all
datasets = {
    "Glucose/Fructose": {
        "cdw": "glcfrc_OD546",
        "glc": "glcfrc_glc",
        "frc_or_gal": "glcfrc_frc"
    },
    "Glucose/Galactose": {
        "cdw": "glcgal_OD546",
        "glc": "glcgal_glc_values",
        "frc_or_gal": "glcgal_gal_values"
    }
}

# Loop through datasets
for label, keys in datasets.items():
    # Get smoothed time and CDW values
    time_smooth = smoothed_all[keys["cdw"]]["time"]
    cdw_smooth = smoothed_all[keys["cdw"]]["value"]

    # Calculate specific growth rate
    dcdw_dt = np.gradient(cdw_smooth, time_smooth)
    specific_growth_rate = dcdw_dt / cdw_smooth

    # Filter valid points
    valid = np.isfinite(specific_growth_rate)
    mu_valid = specific_growth_rate[valid]
    time_valid = time_smooth[valid]

    # Find exponential phase window
    mu_max = np.max(mu_valid)
    tolerance = 0.05
    mu_peak_times = time_valid[np.abs(mu_valid - mu_max) < tolerance]
    exp_phase_start = mu_peak_times.min()
    exp_phase_end = mu_peak_times.max()

    # Get closest previous smoothed timepoint < exp_phase_start for each variable
    idx_cdw = np.where(smoothed_all[keys["cdw"]]["time"] < exp_phase_start)[0][-1]
    idx_glc = np.where(smoothed_all[keys["glc"]]["time"] < exp_phase_start)[0][-1]
    idx_frcgal = np.where(smoothed_all[keys["frc_or_gal"]]["time"] < exp_phase_start)[0][-1]

    # Extract times and values
    t_cdw = smoothed_all[keys["cdw"]]["time"][idx_cdw]
    v_cdw = smoothed_all[keys["cdw"]]["value"][idx_cdw]

    t_glc = smoothed_all[keys["glc"]]["time"][idx_glc]
    v_glc = smoothed_all[keys["glc"]]["value"][idx_glc]

    t_frcgal = smoothed_all[keys["frc_or_gal"]]["time"][idx_frcgal]
    v_frcgal = smoothed_all[keys["frc_or_gal"]]["value"][idx_frcgal]

    # Print report
    print(f"    {label}: μmax = {mu_max:.2f} 1/h during exponential phase from {exp_phase_start:.1f}h to {exp_phase_end:.1f}h")
    print(f"    → Smoothed values just before exponential phase:")
    print(f"         Glucose @ {t_glc:.2f}h = {v_glc:.4f} mg/L")
    print(f"         {'Fructose' if 'frc' in keys['frc_or_gal'] else 'Galactose'} @ {t_frcgal:.2f}h = {v_frcgal:.4f} mg/L")
    print(f"         Biomass @ {t_cdw:.2f}h = {v_cdw:.4f} gCDW/L\n")
#-----------------------------------------------------------------------------------------------------------------------
# Ideal Microbes (mimick Lendenmann 9.2 after 2.1h (lag phase ca done): 0.015 mM Glc and 0.012 mM Frc and 0.0098 g/L cells) 
#-----------------------------------------------------------------------------------------------------------------------

# Convert mM into mg/L

MW = 180.16  # g/mol

# List of simulation DataFrames
sim_dfs = [sim_glcfrc_OFV, sim_glcfrc_EFV, sim_glcgal_iJO1366]

# Apply molar mass conversion
for df in sim_dfs:
    if "glc" in df.columns:
        df["glc_mg"] = df["glc"] * MW
    if "frc" in df.columns:
        df["frc_mg"] = df["frc"] * MW
    if "gal" in df.columns:
        df["gal_mg"] = df["gal"] * MW


# Store Raw Fit and smooth fit

variables = ["glc92", "frc", "cdw"]
column_map = {
    "glc92": "glc_mg",
    "frc": "frc_mg",
    "cdw": "mic"
}

sim_data = {
    "ofv": {
        "df": sim_glcfrc_OFV,
        "time": sim_glcfrc_OFV["time_h"].values
    },
    "efv": {
        "df": sim_glcfrc_EFV,
        "time": sim_glcfrc_EFV["time_h"].values
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
        t_smooth, y_smooth = smooth_line(time, values)
        sim_data[sim_type]["smooth"][var] = {
            "time": t_smooth,
            "value": y_smooth
        }

# Create a new dictionary for glcgal fits
sim_data["glcgal_glc"] = {}
sim_data["glcgal_gal"] = {}
sim_data["glcgal_cdw"] = {}

df_gal = sim_glcgal_iJO1366
time_gal = df_gal["time_h"].values

# Glucose (glc_mg)
sim_data["glcgal_glc"]["raw"] = df_gal["glc_mg"].values
t_smooth, y_smooth = smooth_line(time_gal, df_gal["glc_mg"].values)
sim_data["glcgal_glc"]["smooth"] = {"time": t_smooth, "value": y_smooth}

# Galactose (gal_mg)
sim_data["glcgal_gal"]["raw"] = df_gal["gal_mg"].values
t_smooth, y_smooth = smooth_line(time_gal, df_gal["gal_mg"].values)
sim_data["glcgal_gal"]["smooth"] = {"time": t_smooth, "value": y_smooth}

# Biomass (mic)
sim_data["glcgal_cdw"]["raw"] = df_gal["mic"].values
t_smooth, y_smooth = smooth_line(time_gal, df_gal["mic"].values)
sim_data["glcgal_cdw"]["smooth"] = {"time": t_smooth, "value": y_smooth}

print(' ✅ Done!')

# ════════════════════════════════════════════════════════════════
# 6. Plots
# ════════════════════════════════════════════════════════════════

#-----------------------------------------------------------------
# General Settings
#-----------------------------------------------------------------

# Define colors
colors = sns.color_palette("rocket", 10)
colors2 = sns.color_palette("mako", 10)
color_cdw = "black"
color_glc = colors[5]
color_frc = colors2[6]
color_gal = colors2[3]

#-----------------------------------------------------------------
# Glucose Galactose (Fig 9.1) Panel 
#-----------------------------------------------------------------

print("Plotting Data...")

print(" Generating Glucose/Galactose Empirical + OFV Simulation panel...")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=False)

# Titles for each subplot
titles = [
    "Empirical Data",
    "OFV Simulation"
]

# Plot 1: Empirical Data
ax1 = axes[0]
ax1.scatter(raw_all["glcgal_glc_values"]["time"], raw_all["glcgal_glc_values"]["value"], color=color_glc, s=30)
ax1.plot(smoothed_all["glcgal_glc_values"]["time"], smoothed_all["glcgal_glc_values"]["value"], color=color_glc)

ax1.scatter(raw_all["glcgal_gal_values"]["time"], raw_all["glcgal_gal_values"]["value"], color=color_gal, s=30)
ax1.plot(smoothed_all["glcgal_gal_values"]["time"], smoothed_all["glcgal_gal_values"]["value"], color=color_gal)

ax1.set_ylabel("Carbon Source [mg/L]", fontsize=10)
ax1.set_xlabel("Time [h]", fontsize=10)
ax1.set_ylim(0, 4.5)
ax1.tick_params(axis='both', labelsize=10)
ax1.set_title(titles[0], fontsize=11)
ax1.set_facecolor("#f2f2f2")

# Right y-axis: Biomass
ax1_right = ax1.twinx()
ax1_right.scatter(raw_all["glcgal_OD546"]["time"], (raw_all["glcgal_OD546"]["value"]) * 1000, color=color_cdw, s=30)
ax1_right.plot(smoothed_all["glcgal_OD546"]["time"], (smoothed_all["glcgal_OD546"]["value"]) * 1000, color=color_cdw)
ax1_right.set_ylim(1, 6)
ax1_right.tick_params(axis='y', labelsize=10)
ax1_right.set_yticklabels([])  # Hide right y-ticks

# Plot 2: OFV Simulation
ax2 = axes[1]
ax2.plot(sim_data["glcgal_glc"]["smooth"]["time"], sim_data["glcgal_glc"]["smooth"]["value"], color=color_glc, linewidth=3)
ax2.plot(sim_data["glcgal_gal"]["smooth"]["time"], sim_data["glcgal_gal"]["smooth"]["value"], color=color_gal, linewidth=3)

ax2.set_xlabel("Time [h]", fontsize=10)
ax2.set_ylim(0, 4.5)
ax2.tick_params(axis='both', labelsize=10)
ax2.set_title(titles[1], fontsize=11)
ax2.set_yticklabels([])

# Right y-axis: Biomass
ax2_right = ax2.twinx()
ax2_right.plot(sim_data["glcgal_cdw"]["smooth"]["time"], (sim_data["glcgal_cdw"]["smooth"]["value"]) * 1000, color=color_cdw, linewidth=3)
ax2_right.set_ylabel("Biomass [mgCDW/L]", fontsize=10)
ax2_right.set_ylim(1, 6)
ax2_right.tick_params(axis='y', labelsize=10)

# Create legend handles with colored dots
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Glucose', markerfacecolor=color_glc, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Galactose', markerfacecolor=color_gal, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Biomass', markerfacecolor=color_cdw, markersize=8)
]

legend = ax1.legend(
    handles=legend_elements,
    loc='upper right',
    bbox_to_anchor=(1.0, 1.0),
    frameon=True,
    framealpha=1,
    edgecolor='black',
    fancybox=True,
    fontsize=9
)

# Layout and display
plt.tight_layout()
plt.savefig(export_path / "LM_Sim_GlcGal_Fig_9.1.png", dpi=300, bbox_inches="tight")
plt.savefig(export_path / "LM_Sim_GlcGal_Fig_9.1.svg", format="svg", bbox_inches="tight")
plt.show()
plt.close()


print(" Saved: LM_Sim_GlcGal_Fig_9.1.png and LM_Sim_GlcGal_Fig_9.1.svg")

# ────────────────────────────────────────────────────────────────
# Save individual plots for Glc/Gal Empirical and OFV Simulation
# ────────────────────────────────────────────────────────────────
# 1. Empirical Data (5x4) with axis labels
fig1, ax1 = plt.subplots(figsize=(5, 4))
ax1.scatter(raw_all["glcgal_glc_values"]["time"], raw_all["glcgal_glc_values"]["value"], color=color_glc, s=30)
ax1.plot(smoothed_all["glcgal_glc_values"]["time"], smoothed_all["glcgal_glc_values"]["value"], color=color_glc)

ax1.scatter(raw_all["glcgal_gal_values"]["time"], raw_all["glcgal_gal_values"]["value"], color=color_gal, s=30)
ax1.plot(smoothed_all["glcgal_gal_values"]["time"], smoothed_all["glcgal_gal_values"]["value"], color=color_gal)

ax1.set_ylabel("Carbon Source [mg/L]", fontsize=10)
ax1.set_xlabel("Time [h]", fontsize=10)
ax1.set_ylim(0, 4.5)
ax1.set_xlim(0, 4)
ax1.tick_params(axis='both', labelsize=10)
ax1.set_title("Empirical Data", fontsize=11)

# Right y-axis
ax1_right = ax1.twinx()
ax1_right.scatter(raw_all["glcgal_OD546"]["time"], raw_all["glcgal_OD546"]["value"] * 1000, color=color_cdw, s=30)
ax1_right.plot(smoothed_all["glcgal_OD546"]["time"], smoothed_all["glcgal_OD546"]["value"] * 1000, color=color_cdw)
ax1_right.set_ylim(1, 6)
ax1_right.tick_params(axis='y', labelsize=10)
ax1_right.set_ylabel("Biomass [mgCDW/L]", fontsize=10)
ax1_right.axvspan(ax1.get_xlim()[0], 1.3, color="gray", zorder=0, alpha=0.1)
ax1_right.axvspan(1.3, 3.7, color="darkred", zorder=0, alpha=0.1)

plt.tight_layout()
fig1.savefig(export_path / "LM_GlcGal_Empirical.png", dpi=300, bbox_inches="tight")
fig1.savefig(export_path / "LM_GlcGal_Empirical.svg", format="svg", bbox_inches="tight")
plt.close(fig1)

# 2. OFV Simulation (5x4) with axis labels
fig2, ax2 = plt.subplots(figsize=(5, 4))
ax2.plot(sim_data["glcgal_glc"]["smooth"]["time"], sim_data["glcgal_glc"]["smooth"]["value"], color=color_glc, linewidth=3)
ax2.plot(sim_data["glcgal_gal"]["smooth"]["time"], sim_data["glcgal_gal"]["smooth"]["value"], color=color_gal, linewidth=3)

ax2.set_xlabel("Time [h]", fontsize=10)
ax2.set_ylabel("Carbon Source [mg/L]", fontsize=10)
ax2.set_ylim(0, 4.5)
ax2.set_xlim(0, 8)
ax2.tick_params(axis='both', labelsize=10)
ax2.set_title("OFV Simulation", fontsize=11)

# Right y-axis
ax2_right = ax2.twinx()
ax2_right.plot(sim_data["glcgal_cdw"]["smooth"]["time"], sim_data["glcgal_cdw"]["smooth"]["value"] * 1000, color=color_cdw, linewidth=3)
ax2_right.set_ylabel("Biomass [mgCDW/L]", fontsize=10)
ax2_right.set_ylim(1, 6)
ax2_right.tick_params(axis='y', labelsize=10)
ax2_right.axvspan(0,6, color="darkred", zorder=0, alpha=0.1)

plt.tight_layout()
fig2.savefig(export_path / "LM_GlcGal_OFV.png", dpi=300, bbox_inches="tight")
fig2.savefig(export_path / "LM_GlcGal_OFV.svg", format="svg", bbox_inches="tight")
plt.close(fig2)

# Define legend elements using your colors
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Glucose', markerfacecolor=color_glc, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Galactose', markerfacecolor=color_gal, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Biomass', markerfacecolor=color_cdw, markersize=8)
]

# Create a separate figure for the legend
fig_leg, ax_leg = plt.subplots(figsize=(3, 1.5))
ax_leg.axis('off')  # Hide axes

# Create the legend in the new figure
legend = ax_leg.legend(
    handles=legend_elements,
    loc='center',
    frameon=True,
    framealpha=1,
    edgecolor='black',
    fancybox=True,
    fontsize=9
)

# Save the legend as standalone image
fig_leg.savefig(export_path / "LM_GlcGal_Legend.png", dpi=300, bbox_inches="tight",transparent=True)
fig_leg.savefig(export_path / "LM_GlcGal_Legend.svg", format="svg", bbox_inches="tight")
plt.close(fig_leg)


#-----------------------------------------------------------------
# Glucose Fructose (Fig 9.2) Panel 
#-----------------------------------------------------------------

print(" Generating Glucose/Fructose Empirical + OFV/ EFV Simulation panels...")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=False)

# Titles for each subplot
titles = [
    "Empirical Data",
    "OFV Simulation",
    "EFV Simulation"
]

# 1. Batch Culture Data (Plot 1)
ax1 = axes[0]
ax1.scatter(raw_all["glcfrc_glc"]["time"], raw_all["glcfrc_glc"]["value"], color=color_glc, s=30)
ax1.plot(smoothed_all["glcfrc_glc"]["time"], smoothed_all["glcfrc_glc"]["value"], color=color_glc)  # default line

ax1.scatter(raw_all["glcfrc_frc"]["time"], raw_all["glcfrc_frc"]["value"], color=color_frc, s=30)
ax1.plot(smoothed_all["glcfrc_frc"]["time"], smoothed_all["glcfrc_frc"]["value"], color=color_frc)  # default line

ax1.set_ylabel("Carbon Source [mg/L]", fontsize=10)
ax1.set_xlabel("Time [h]", fontsize=10)
ax1.set_ylim(0, 5)
ax1.tick_params(axis='both', labelsize=10)
ax1.set_title(titles[0], fontsize=11)
ax1.set_facecolor("#f2f2f2")

# Right y-axis for plot 1
ax1_right = ax1.twinx()
ax1_right.scatter(raw_all["glcfrc_OD546"]["time"], (raw_all["glcfrc_OD546"]["value"])* 1000, color=color_cdw, s=30)
ax1_right.plot(smoothed_all["glcfrc_OD546"]["time"], (smoothed_all["glcfrc_OD546"]["value"]) * 1000, color=color_cdw)  # default line
ax1_right.set_ylim(1, 4.5)
ax1_right.tick_params(axis='y', labelsize=10)
ax1_right.set_yticklabels([])  # Remove right y-axis labels

# 2. OFV Simulation (Plot 2)
ax2 = axes[1]
ax2.plot(sim_data["ofv"]["smooth"]["glc92"]["time"], sim_data["ofv"]["smooth"]["glc92"]["value"], color=color_glc, linewidth=3)
ax2.plot(sim_data["ofv"]["smooth"]["frc"]["time"], sim_data["ofv"]["smooth"]["frc"]["value"], color=color_frc, linewidth=3)

ax2.set_xlabel("Time [h]", fontsize=10)
ax2.set_ylim(0, 5)
ax2.tick_params(axis='both', labelsize=10)
ax2.set_title(titles[1], fontsize=11)
ax2.set_yticklabels([])

ax2_right = ax2.twinx()
ax2_right.plot(sim_data["ofv"]["smooth"]["cdw"]["time"], (sim_data["ofv"]["smooth"]["cdw"]["value"]) * 1000, color=color_cdw, linewidth=3)
ax2_right.set_ylim(1, 4.5)
ax2_right.tick_params(axis='y', labelsize=10)
ax2_right.set_yticklabels([])

# 3. EFV Simulation (Plot 3)
ax3 = axes[2]
ax3.plot(sim_data["efv"]["smooth"]["glc92"]["time"], sim_data["efv"]["smooth"]["glc92"]["value"], color=color_glc, linewidth=3)
ax3.plot(sim_data["efv"]["smooth"]["frc"]["time"], sim_data["efv"]["smooth"]["frc"]["value"], color=color_frc, linewidth=3)

ax3.set_xlabel("Time [h]", fontsize=10)
ax3.set_ylim(0, 5)
ax3.tick_params(axis='both', labelsize=10)
ax3.set_title(titles[2], fontsize=11)
ax3.set_yticklabels([])

ax3_right = ax3.twinx()
ax3_right.plot(sim_data["efv"]["smooth"]["cdw"]["time"], (sim_data["efv"]["smooth"]["cdw"]["value"]) * 1000, color=color_cdw, linewidth=3)
ax3_right.set_ylabel("Biomass [mgCDW/L]", fontsize=10)
ax3_right.set_ylim(1, 4.5)
ax3_right.tick_params(axis='y', labelsize=10)

# Create legend handles with colored dots
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Glucose', markerfacecolor=color_glc, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Fructose', markerfacecolor=color_frc, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Biomass', markerfacecolor=color_cdw, markersize=8)
]

legend = ax1.legend(
    handles=legend_elements,
    loc='upper right',
    bbox_to_anchor=(1.0, 1.0),
    frameon=True,
    framealpha=1,
    edgecolor='black',
    fancybox=True,
    fontsize=9
)

# Make the frame thinner
legend.get_frame().set_linewidth(0.8)  # e.g. 0.8 for thin, default is ~1.5


# Layout
plt.tight_layout()
plt.savefig(export_path / "LM_Sim_GlcFrc_Fig_9.1.png", dpi=300, bbox_inches="tight")
plt.savefig(export_path / "LM_Sim_GlcFrc_Fig_9.1.svg", format="svg", bbox_inches="tight")
plt.show()
plt.close()

# ────────────────────────────────────────────────────────────────
# Save individual panels (5x4) with prefix "LM_"
# ────────────────────────────────────────────────────────────────

# 1. Empirical Data
fig1, ax1 = plt.subplots(figsize=(5, 4))
ax1.scatter(raw_all["glcfrc_glc"]["time"], raw_all["glcfrc_glc"]["value"], color=color_glc, s=30)
ax1.plot(smoothed_all["glcfrc_glc"]["time"], smoothed_all["glcfrc_glc"]["value"], color=color_glc)

ax1.scatter(raw_all["glcfrc_frc"]["time"], raw_all["glcfrc_frc"]["value"], color=color_frc, s=30)
ax1.plot(smoothed_all["glcfrc_frc"]["time"], smoothed_all["glcfrc_frc"]["value"], color=color_frc)

ax1.set_ylabel("Carbon Source [mg/L]", fontsize=10)
ax1.set_xlabel("Time [h]", fontsize=10)
ax1.set_ylim(0, 5)
ax1.set_xlim(0, 4)
ax1.set_title("Empirical Data", fontsize=11)
ax1.tick_params(axis='both', labelsize=10)

ax1r = ax1.twinx()
ax1r.scatter(raw_all["glcfrc_OD546"]["time"], raw_all["glcfrc_OD546"]["value"] * 1000, color=color_cdw, s=30)
ax1r.plot(smoothed_all["glcfrc_OD546"]["time"], smoothed_all["glcfrc_OD546"]["value"] * 1000, color=color_cdw)
ax1r.set_ylim(1, 4.5)
ax1r.tick_params(axis='y', labelsize=10)
ax1r.set_ylabel("Biomass [mgCDW/L]", fontsize=10)
ax1r.axvspan(ax1.get_xlim()[0], 2.1, color="gray", zorder=0, alpha=0.1)
ax1r.axvspan(2.1, 3.5, color="darkred", zorder=0, alpha=0.1)

plt.tight_layout()
fig1.savefig(export_path / "LM_GlcFrc_Empirical.png", dpi=300, bbox_inches="tight")
fig1.savefig(export_path / "LM_GlcFrc_Empirical.svg", format="svg", bbox_inches="tight")
plt.close(fig1)

# 2. OFV Simulation
fig2, ax2 = plt.subplots(figsize=(5, 4))
ax2.plot(sim_data["ofv"]["smooth"]["glc92"]["time"], sim_data["ofv"]["smooth"]["glc92"]["value"], color=color_glc, linewidth=3)
ax2.plot(sim_data["ofv"]["smooth"]["frc"]["time"], sim_data["ofv"]["smooth"]["frc"]["value"], color=color_frc, linewidth=3)

ax2.set_xlabel("Time [h]", fontsize=10)
ax2.set_ylabel("Carbon Source [mg/L]", fontsize=10)
ax2.set_ylim(0, 5)
ax2.set_xlim(0, 8)
ax2.set_title("OFV Simulation", fontsize=11)
ax2.tick_params(axis='both', labelsize=10)
ax2.set_ylabel("Carbon Source [mg/L]", fontsize=10)

ax2r = ax2.twinx()
ax2r.plot(sim_data["ofv"]["smooth"]["cdw"]["time"], sim_data["ofv"]["smooth"]["cdw"]["value"] * 1000, color=color_cdw, linewidth=3)
ax2r.set_ylim(1, 4)
ax2r.tick_params(axis='y', labelsize=10)
ax2r.set_ylabel("Biomass [mgCDW/L]", fontsize=10)
ax2r.axvspan(ax2.get_xlim()[0], 4, color="darkred", zorder=0, alpha=0.1)

plt.tight_layout()
fig2.savefig(export_path / "LM_GlcFrc_OFV.png", dpi=300, bbox_inches="tight")
fig2.savefig(export_path / "LM_GlcFrc_OFV.svg", format="svg", bbox_inches="tight")
plt.close(fig2)

# 3. EFV Simulation
fig3, ax3 = plt.subplots(figsize=(5, 4))
ax3.plot(sim_data["efv"]["smooth"]["glc92"]["time"], sim_data["efv"]["smooth"]["glc92"]["value"], color=color_glc, linewidth=3)
ax3.plot(sim_data["efv"]["smooth"]["frc"]["time"], sim_data["efv"]["smooth"]["frc"]["value"], color=color_frc, linewidth=3)

ax3.set_xlabel("Time [h]", fontsize=10)
ax3.set_ylabel("Carbon Source [mg/L]", fontsize=10)
ax3.set_ylim(0, 5)
ax3.set_xlim(0, 8)
ax3.set_title("EFV Simulation", fontsize=11)
ax3.tick_params(axis='both', labelsize=10)
ax3.set_ylabel("Carbon Source [mg/L]", fontsize=10)

ax3r = ax3.twinx()
ax3r.plot(sim_data["efv"]["smooth"]["cdw"]["time"], sim_data["efv"]["smooth"]["cdw"]["value"] * 1000, color=color_cdw, linewidth=3)
ax3r.set_ylabel("Biomass [mgCDW/L]", fontsize=10)
ax3r.set_ylim(1, 4.5)
ax3r.tick_params(axis='y', labelsize=10)
ax3.axvspan(ax3.get_xlim()[0], 4.5, color="darkred", zorder=0, alpha=0.1)

plt.tight_layout()
fig3.savefig(export_path / "LM_GlcFrc_EFV.png", dpi=300, bbox_inches="tight")
fig3.savefig(export_path / "LM_GlcFrc_EFV.svg", format="svg", bbox_inches="tight")
plt.close(fig3)

# Define legend handles with colored dots for Glucose, Fructose, Biomass
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Glucose', markerfacecolor=color_glc, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Fructose', markerfacecolor=color_frc, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Biomass', markerfacecolor=color_cdw, markersize=8)
]

# Create a new figure for the standalone legend
fig_leg2, ax_leg2 = plt.subplots(figsize=(3, 1.5))  # Adjust size if needed
ax_leg2.axis('off')  # Hide the axes

# Add the legend to the figure
legend = ax_leg2.legend(
    handles=legend_elements,
    loc='center',
    frameon=True,
    framealpha=1,
    edgecolor='black',
    fancybox=True,
    fontsize=9
)

# Set thinner frame line width
legend.get_frame().set_linewidth(0.8)

# Save the legend as PNG and SVG
fig_leg2.savefig(export_path / "LM_GlcFrc_Legend.png", dpi=300, bbox_inches="tight",transparent=True)
fig_leg2.savefig(export_path / "LM_GlcFrc_Legend.svg", format="svg", bbox_inches="tight")
plt.close(fig_leg2)

print(" Saved: LM_Sim_GlcFrc_Fig_9.1.png and LM_Sim_GlcFrc_Fig_9.1.svg")

print(' ✅ Done!')

print('Everything complete! 🎉')