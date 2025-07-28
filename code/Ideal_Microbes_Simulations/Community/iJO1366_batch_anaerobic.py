# Using the E. coli iJO1366 metabolic model under anaerobic conditions in batch culture
# ???

# ════════════════════════════════════════════════════════════════
# Import Packages
# ════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import IdealMicrobe as im
from pathlib import Path  
import matplotlib.pyplot as plt
import seaborn as sns

# ════════════════════════════════════════════════════════════════
# File Paths
# ════════════════════════════════════════════════════════════════

# Add Repo Root 
project_root = Path(__file__).resolve().parents[3] 

# processed path
processed_path = project_root / "data" / "processed" 
processed_csv_path = processed_path / "csv_files"
output_dir = processed_path / "figures"

# ════════════════════════════════════════════════════════════════
# Import Data
# ════════════════════════════════════════════════════════════════

print("Loading metabolic structure data...")

# Import the external, internal stochiometric matrix and the OFVs
ext_S_df = pd.read_csv(processed_csv_path / "iJO1366_ext_S.csv", index_col=0)
int_S_df = pd.read_csv(processed_csv_path / "iJO1366_int_S.csv", index_col=0)
ofv_df = pd.read_csv(processed_csv_path / "iJO1366_OFVs.csv", index_col=0)

# Define the columns of interest
biomass_col = "BIOMASS_Ec_iJO1366_core_53p95M" # careful: BIOMASS_Ec_iJO1366_WT_53p95M
fru_col = "EX_fru_e_rev"
glc_col = "EX_glc__D_e_rev"
gal_col = "EX_gal_e_rev"
mal_col = "EX_malt_e_rev"
lac_col = "EX_lcts_e_rev"
suc_col = "EX_sucr_e_rev"

substrate_names = {
    0: "Fructose",
    1: "Galactose",
    2: "Glucose",
    3: "Lactose",
    4: "Maltose",
    5: "Sucrose"
}

cols_of_interest = [fru_col, glc_col, gal_col, mal_col, lac_col, suc_col, biomass_col]

print(" Done! ✅")

# ════════════════════════════════════════════════════════════════
# Data preparation: Internal/ External S and OFVs
# ════════════════════════════════════════════════════════════════

# Reorder columns in ofv_df to match ext_S_df
ofv_df = ofv_df[ext_S_df.columns]

# Normalisation by Carbon Source
ofv_df_norm = ofv_df / 10

# Add dummy reaction/ pathway for proteins not related to metabolism

extreme_final_df = ofv_df_norm.copy()
ext_S_final_df = ext_S_df.copy()
int_S_final_df = int_S_df.copy()

if 'Protein' not in extreme_final_df.columns:
    # Only add 'Protein' column if it doesn't exist
    extreme_final_df['Protein'] = 0.0

pseudo_path = pd.Series(0, index=extreme_final_df.columns)
pseudo_path['Protein'] = 1219

if not ((extreme_final_df == pseudo_path.values).all(axis=1)).any():
    extreme_final_df = pd.concat([extreme_final_df, pseudo_path.to_frame().T], ignore_index=True)

for df in [ext_S_final_df, int_S_final_df]:
    if 'Protein' not in df.columns:
        df.loc[:, 'Protein'] = 0.0  

# Transform dfs into np.arrays

extreme_path = extreme_final_df.to_numpy(dtype=float) 
stoich_int = int_S_final_df.to_numpy(dtype=float)
stoich_ext = ext_S_final_df.to_numpy(dtype=float)

# ════════════════════════════════════════════════════════════════
# Data preparation: Adjusting to correct units & parametrization
# ════════════════════════════════════════════════════════════════

# Basis values
µmax = 0.76 # literature value
mmol_in_liter = 55510 # H2O only
Km_batch = 0.028/mmol_in_liter # transporter Km
Km_const = 0.001/mmol_in_liter # transporter Km
Vmax = µmax / extreme_final_df[biomass_col].max() 

# adjust for maximum reaction rate
extreme_path = extreme_path * Vmax
num_reactions = stoich_int.shape[1]
num_paths = extreme_path.shape[0]

# Create the stoich_biomass array
reaction_names = list(extreme_final_df.columns)
biomass_index = reaction_names.index(biomass_col)
stoich_biomass = np.zeros(len(reaction_names))
stoich_biomass[biomass_index] = 1.0 

# Define distinct Ks values
Ks_batch = 7160/ 180160
Ks_cont = 73/ 180160

print("\n📈 Monod constants for Batch and Continuous Culture:\n")
print(f"K: {Ks_batch} (mmol/L)")
print(f"K: {Ks_cont} (mmol/L)")

# Define other required arrays with sensible defaults
react_rate = np.full(num_reactions, Vmax)
react_rate[biomass_index] = 1.1 # division 1 time/ hour
met_noise = 0.00238 * Vmax
mich_ment = np.full(num_reactions, Km_batch) # Km_batch or Km_const
met_ext_total = np.full(6, mmol_in_liter) 
pot_batch = -0.556
pot_const = -0.91
exp_pot = np.array([0.76, 0.67304348, 0.76, 0.71652174, 0.76, 0.76, pot_batch]) # pot_batch or pot_const

# ════════════════════════════════════════════════════════════════
# Initialise Ideal Microbe
# ════════════════════════════════════════════════════════════════

print("Object Initialisation...")

# Build the Microbe instance
microbe = im.Microbe(
    stoich_int=stoich_int,
    stoich_ext=stoich_ext,
    extreme_path=extreme_path,
    stoich_biomass=stoich_biomass,
    met_noise=met_noise,
    react_rate=react_rate,
    mich_ment=mich_ment,
    exp_pot = exp_pot,
    fba_approach=True #i.e., biomass is pseudo-reaction in different unit
)

print(microbe.exp_pot)

print(" Initialised Microbe! 🦠")

# Initialize Culture Object
culture = im.Culture([microbe],0,np.asarray(np.zeros(6)), met_ext_total)

print(" Initialised Culture! 🧫")

# optional break

exit()  # Script stops here

# ════════════════════════════════════════════════════════════════
# Generate Plots
# ════════════════════════════════════════════════════════════════

# -------------------------------
# Growth Plane
# -------------------------------

print("Constructing Growth Planes...")

substrate_pairs = [
    (0, 2, "Fructose", "Glucose"),
    (1, 2, "Galactose", "Glucose"),
    (0, 1, "Fructose", "Galactose")
]

# Loop over each substrate pair
for idx1, idx2, name1, name2 in substrate_pairs:
    print(f"  → Plotting: {name1} vs {name2}")

    # Access correct K values
    K1 = monod_params[name1]["K"]
    K2 = monod_params[name2]["K"]

    # Setup
    met_ext = np.full(6, 0.001)  # baseline metabolite levels (length = 6)
    met_ext_index = [idx1, idx2]
    met_ext_max = [K1, K2]

    # Create plot
    fig, ax = plt.subplots(figsize=(5, 4))
    heatmap, contours = microbe.plot_growth_plane(
        met_ext=met_ext,
        met_ext_index=met_ext_index,
        met_ext_max=met_ext_max,
        ax=ax,
        contours=True,
        prod_cons=False,
        met_ext_total=met_ext_total,
        cmap='rocket_r'
    )

    # Final formatting
    plt.colorbar(heatmap, ax=ax, label="Growth Rate")
    ax.set_xlabel(f"{name1} [mM]", fontsize=10)
    ax.set_ylabel(f"{name2} [mM]", fontsize=10)
    ax.set_title("Growth Rate", fontsize=12)
    plt.tight_layout()

    # Save plot
    filename_base = f"iJO1366_{name2.lower()}_{name1.lower()}_growthplane"
    plt.savefig(output_dir / f"{filename_base}_batch.png", dpi=300)
    plt.savefig(output_dir / f"{filename_base}_batch.svg")
    plt.close()
    print(f"     ✔ Saved {filename_base}")

# -------------------------------
# Microbe and Nutrient levels
# -------------------------------

# Define simulation pairs 
substrate_pairs = [
    (0, 2, "Fructose", "Glucose"),
    (1, 2, "Galactose", "Glucose"),
    (0, 1, "Fructose", "Galactose")
]

t_span = (0, 6)
mic_level0 = np.array([0.008])
colors = sns.color_palette("rocket", 10)

for idx1, idx2, name1, name2 in substrate_pairs:
    print(f"Running simulation for {name1} + {name2}...")

    # Set up external metabolites: 6 total, all 0 except selected ones
    met_ext0 = np.zeros(6)
    met_ext0[idx1] = 0.02
    met_ext0[idx2] = 0.02

    # Run simulation
    solution = culture.cr_model_ode(t_span, mic_level0, met_ext0, atol=1e-8, rtol=1e-8)
    mic_level, met_ext = culture.slice_cr_solution(solution)
    times = solution.t

    ### Plot Growth ###
    plt.figure(figsize=(8, 4))
    plt.plot(times, mic_level[0], marker='o', color='black')
    plt.title("E. coli Growth", fontsize=12)
    plt.xlabel("Time [h]")
    plt.ylabel("Growth Level [g/L]")
    plt.grid(True, which='major', linewidth=0.5, color='gray', alpha=0.3)
    ax = plt.gca()
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / f"iJO1366_{name2.lower()}_{name1.lower()}_microbegrowth_batch.png", dpi=300)
    plt.savefig(output_dir / f"iJO1366_{name2.lower()}_{name1.lower()}_microbegrowth_batch.svg")
    plt.close()
    print(" Saved Microbe Growth Plot! 💾")

    ### Plot Nutrient Concentrations ###
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    axes[0].plot(times, met_ext[idx1], marker='s', color=colors[2])
    axes[0].set_title(name1, fontsize=12)
    axes[0].set_xlabel("Time [h]")
    axes[0].set_ylabel("Concentration [mM]")
    axes[0].grid(True, linewidth=0.5, alpha=0.3)

    axes[1].plot(times, met_ext[idx2], marker='s', color=colors[7])
    axes[1].set_title(name2, fontsize=12)
    axes[1].set_xlabel("Time [h]")
    axes[1].grid(True, linewidth=0.5, alpha=0.3)

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(0.8)
        ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / f"iJO1366_{name2.lower()}_{name1.lower()}_nutrientlevel_batch.png", dpi=300)
    plt.savefig(output_dir / f"iJO1366_{name2.lower()}_{name1.lower()}_nutrientlevel_batch.svg")
    plt.close()
    print(" Saved Nutrient Utilisation Plot! 💾")

print('Everything complete! 🎉')


