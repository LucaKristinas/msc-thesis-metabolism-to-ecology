# Using the E. coli iJO1366 metabolic model under aerobic conditions in continuous culture
# 0.0004 mM Ks / 0.92 µmax / 0.001 Km (transporter)

# ════════════════════════════════════════════════════════════════
# Import Packages
# ════════════════════════════════════════════════════════════════

import MetaGrowth as im
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ════════════════════════════════════════════════════════════════
# Import Data
# ════════════════════════════════════════════════════════════════

print("Loading metabolic structure data...")

# Import the external, internal stochiometric matrix and the OFVs
ext_S_df = pd.read_csv("SourceData/iJO1366_ext_S_O2.csv", index_col=0)
int_S_df = pd.read_csv("SourceData/iJO1366_int_S_O2.csv", index_col=0)
ofv_df = pd.read_csv("SourceData/iJO1366_OFVs_O2.csv", index_col=0)
Lendenmann_df = pd.read_csv("SourceData/LM_Glc_Gal_Growthplane.csv")

print(" Done! ✅")

# ════════════════════════════════════════════════════════════════
# Data preparation: Internal/ External S and OFVs
# ════════════════════════════════════════════════════════════════

# Define the columns of interest
biomass_col = "BIOMASS_Ec_iJO1366_core_53p95M" # careful: BIOMASS_Ec_iJO1366_WT_53p95M
fru_col = "EX_fru_e_rev"
glc_col = "EX_glc__D_e_rev"
gal_col = "EX_gal_e_rev"
mal_col = "EX_malt_e_rev"
lac_col = "EX_lcts_e_rev"
suc_col = "EX_sucr_e_rev"

substrate_names = {0: "Fructose",1: "Galactose",2: "Glucose",3: "Lactose",4: "Maltose",5: "Sucrose"}

cols_of_interest = [fru_col, glc_col, gal_col, mal_col, lac_col, suc_col, biomass_col]

# Correct units for Lendenmann
MW = 180.16  # g/mol → 180160 µg/mmol
Lendenmann_df["Glc_mM"] = Lendenmann_df["Glc_res"] / (MW * 1000)
Lendenmann_df["Gal_mM"] = Lendenmann_df["Gal_res"] / (MW * 1000)

# Normalisation by Carbon Source
ofv_df = ofv_df[ext_S_df.columns]
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
µmax_continuous_aerobic = 0.92
mmol_in_liter = 55510 # H2O only
Km_const = 0.001/mmol_in_liter # transporter Km
Vmax = µmax_continuous_aerobic / extreme_final_df[biomass_col].max() 

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
react_rate[biomass_index] = µmax_continuous_aerobic
met_noise = 0.00238 * Vmax
mich_ment = np.full(num_reactions, Km_const) # Km_batch or Km_const
met_ext_total = np.full(6, mmol_in_liter) 
exp_pot_def = np.array([0.92, 0.91028921, 0.92, 0.91514461, 0.92, 0.92, 0])

pseudo_exp_pot = 0.436 + met_noise * np.log(1 + (1 / Km_const))
exp_pot_fit = np.array([0.92, 0.91028921, 0.92, 0.91514461, 0.92, 0.92, pseudo_exp_pot])

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
    exp_pot = exp_pot_fit,
    fba_approach=True 
)

print(" Initialised Microbe! 🦠")

print(f"exp_pot{microbe.exp_pot}")

# Initialize Culture Object
culture = im.Culture([microbe],0,np.asarray(np.zeros(6)), met_ext_total)

print(" Initialised Culture! 🧫")

# ════════════════════════════════════════════════════════════════
# Store important data
# ════════════════════════════════════════════════════════════════

print("\n📈 Monod parameters (r and K) for each carbon source:\n")

monod_params = {} # store for later

for i in range(len(substrate_names)):
    with np.errstate(invalid='ignore', divide='ignore'):
        result = microbe.infer_monod_parameters(np.zeros(6), i, met_ext_total)
    r, _, _, _, K = result
    substrate = substrate_names[i]
    monod_params[substrate] = {"r": float(r), "K": float(K)}
    print(f"{substrate} → r: {float(r):.4f} hr⁻¹, K: {float(K):.6f} (mmol/L)")

# ════════════════════════════════════════════════════════════════
# Generate Panel Fig.S11D
# ════════════════════════════════════════════════════════════════

# Access correct K values
K1 = monod_params["Glucose"]["K"]
K2 = monod_params["Galactose"]["K"]

# Setup
met_ext = np.full(6, 0.0)  # baseline metabolite levels (length = 6)
met_ext_index = [2, 1]
met_ext_max = [0.00056, 0.00056]

fig, ax = plt.subplots(figsize=(5, 4))
result = microbe.plot_growth_plane(
    met_ext=met_ext,
    met_ext_index=met_ext_index,
    met_ext_max=met_ext_max,
    ax=ax,
    contours=True,
    prod_cons=False,
    met_ext_total=met_ext_total,
    cmap='viridis_r'
)

# Extract just the heatmap for colorbar 
heatmap = result[0]  

# Manually add one colorbar
cbar = plt.colorbar(heatmap, ax=ax)
cbar.set_label("Growth Rate [h⁻¹]", fontsize=14)
cbar.ax.tick_params(labelsize=14)

# Axes labels
ax.set_title("OFV Growth Plane", fontsize=16)

# Define tick positions (in mM)
tick_step = 0.0001
tick_max = 0.0006  # Slightly beyond your max of 0.00056
xticks = np.arange(0, tick_max, tick_step)
yticks = np.arange(0, tick_max, tick_step)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# Convert to µg/L for labels using MW of glucose/galactose
MW = 180.16  # g/mol → 180160 µg/mmol
xtick_labels = [f"{tick * MW * 1000:.0f}" for tick in xticks]
ytick_labels = [f"{tick * MW * 1000:.0f}" for tick in yticks]

# Apply to plot
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels(xtick_labels, fontsize=14)
ax.set_yticklabels(ytick_labels, fontsize=14)

# Update axis labels if desired
ax.set_xlabel("Glucose [µg/L]", fontsize=14)
ax.set_ylabel("Galactose [µg/L]", fontsize=14)

# Create the same normalization as the heatmap
vmin = heatmap.get_clim()[0]
vmax = heatmap.get_clim()[1]
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap('viridis_r')  # Use same colormap as heatmap

# Get RGBA color for each point based on its growth rate
colors = cmap(norm(Lendenmann_df["Dilution"]))  

# Plot colored bubbles
ax.scatter(
    Lendenmann_df["Glc_mM"],
    Lendenmann_df["Gal_mM"],
    s=140,
    facecolors=colors,
    edgecolors='white',
    linewidths=1,
    zorder=5,
    clip_on=False
)

plt.tight_layout()
plt.savefig("Fig_S11D.svg", dpi=300, bbox_inches='tight')
plt.savefig("Fig_S11D.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Calculate R2
y_experimental = Lendenmann_df["Dilution"].values
y_predicted = []

for _, row in Lendenmann_df.iterrows():
    temp_met_ext = np.zeros(6)
    temp_met_ext[2] = row["Glc_mM"]
    temp_met_ext[1] = row["Gal_mM"]
    
    with np.errstate(invalid='ignore', divide='ignore'):
        r, _, _, _, _ = microbe.infer_monod_parameters(temp_met_ext, 0, met_ext_total)
        y_predicted.append(r)

y_predicted = np.array(y_predicted)
r2 = r2_score(y_experimental, y_predicted)

print(f"Model Fit Analysis:")
print(f"\n R² Score: {r2:.4f}")

print("Done 🥳")