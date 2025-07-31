
# Using the E. coli core metabolic model (Orth et al.) under aerobic conditions (EFVs)
# 0.02 mM Ks / 0.75 µmax / 0.028 Km (transporter)

# -------------------------------
# Imports
# -------------------------------

import numpy as np
import pandas as pd
import IdealMicrobe as im
from pathlib import Path  
import matplotlib.pyplot as plt
import seaborn as sns
import sys

import cobra 
from cobra import Reaction
from cobra.io import read_sbml_model
from cobra.util.array import create_stoichiometric_matrix

# Add Repo Root to sys.path 
project_root = Path(__file__).resolve().parents[3]  
sys.path.append(str(project_root))

# Import Helper Functions
from src.utils import (read_rfile, read_mfile)

# -------------------------------
# File Paths
# -------------------------------

# raw path
raw_path = project_root / "data" / "raw"
mplrs_path = raw_path / "mplrs_data"
sbml_path = raw_path / "sbml_files"
npy_path = raw_path / "txt_files"

# processed path
processed_path = project_root / "data" / "processed" 
processed_csv_path = processed_path / "csv_files"
output_dir = processed_path / "figures"

# -------------------------------
# Data
# -------------------------------

print("Loading metabolic structure data...")

EFVs_S_df = pd.read_csv(mplrs_path / "ecolicore_model_v1_frc10.sfile", header=None, sep=r'\s+') 
EFVs_df = pd.read_csv(mplrs_path / "ecolicore_model_v1_frc10.evfs", header=None, sep=r'\s+') 
reaction_list_2 = read_rfile(mplrs_path / "ecolicore_model_v1_frc10.rfile", remove_lambda=False) 
metabolites_list_2 = read_mfile(mplrs_path / "ecolicore_model_v1_frc10.mfile")

model = read_sbml_model(sbml_path / "ecoli_core_simulation.xml")

exp_pot = np.load(npy_path / 'EFV_default_exp_pot.npy') # default Exp_pot 

# Define the columns of interest
biomass_col = "BIOMASS_Ecoli_core_w_GAM"
fru_col = "EX_fru_e_rev"
glc_col = "EX_glc__D_e_rev"

print(" Done! ✅")

# -------------------------------
# Prepare EFVs as input files
# -------------------------------

print("Preparing the metabolic structure data for simulations...")

# Pre-Process mplrs data 

reaction_list_2_efvs = reaction_list_2[0:89] # chop off "artificial" columns 
EFVs_S_df.columns = reaction_list_2
EFVs_S_df.index = metabolites_list_2
EFVs_df.columns = reaction_list_2_efvs

# Remove internal cycles

inactive_mask = (
    (EFVs_df[biomass_col] == 0) &
    (EFVs_df[fru_col] == 0) &
    (EFVs_df[glc_col] == 0)
)

EFVs_df_filtered = EFVs_df[~inactive_mask].reset_index(drop=True)

# Normalise Carbon Source 

df_normalized = EFVs_df_filtered.copy()

only_fru = 0
only_glc = 0
both = 0
neither = 0

for idx, row in EFVs_df_filtered.iterrows():
    fru = row[fru_col]
    glc = row[glc_col]

    if fru > 0 and glc == 0:
        df_normalized.loc[idx] = row / fru
        only_fru += 1

    elif glc > 0 and fru == 0:
        df_normalized.loc[idx] = row / glc
        only_glc += 1

    elif glc > 0 and fru > 0:
        x = 1 / (fru + glc)
        df_normalized.loc[idx] = row * x
        both += 1

    else:
        # both are 0
        neither += 1

# Split reversible reactions into fwd and rev (non-negative flux)

only_neg_or_zero = []
neg_and_pos = []

for col in df_normalized.columns:
    col_vals = df_normalized[col]
    has_neg = (col_vals < 0).any()
    has_pos = (col_vals > 0).any()

    if has_neg:
        if has_pos:
            neg_and_pos.append(col)
        else:
            only_neg_or_zero.append(col)

df_transformed = df_normalized.copy()

for col in neg_and_pos:
    rev_col = f"{col}_rev"
    df_transformed[rev_col] = df_transformed[col]
    df_transformed[col] = df_transformed[col].clip(lower=0)
    df_transformed[rev_col] = df_transformed[rev_col].clip(upper=0).abs()

# Generate internal and external S

stoich_dense = create_stoichiometric_matrix(model)

stoich_df = pd.DataFrame(
    stoich_dense,
    index=[met.id for met in model.metabolites],
    columns=[rxn.id for rxn in model.reactions]
)

stoich_reactions = set(stoich_df.columns)
transformed_reactions = set(df_transformed.columns)

only_in_stoich = stoich_reactions - transformed_reactions
only_in_transformed = transformed_reactions - stoich_reactions

only_in_stoich = sorted(list(only_in_stoich))
only_in_transformed = sorted(list(only_in_transformed))

only_in_transformed_rev = [col for col in only_in_transformed if col.endswith("_rev")]

base_reactions = [col[:-4] for col in only_in_transformed_rev]

found_base_reactions = [rxn for rxn in base_reactions if rxn in stoich_df.columns]
missing_base_reactions = [rxn for rxn in base_reactions if rxn not in stoich_df.columns]

for rxn in found_base_reactions:
    rev_col_name = f"{rxn}_rev"
    stoich_df[rev_col_name] = -stoich_df[rxn]  # Flip the signs

df_transformed_sorted = df_transformed[sorted(df_transformed.columns)]
stoich_df_sorted = stoich_df[sorted(stoich_df.columns)]

columns_match = list(df_transformed_sorted.columns) == list(stoich_df_sorted.columns)

stoich_custom = pd.DataFrame(
    data=0,
    index=["Fructose_ext", "Glucose_ext"],
    columns=stoich_df_sorted.columns
)

stoich_custom.loc["Fructose_ext", fru_col] = -1
stoich_custom.loc["Glucose_ext", glc_col] = -1

# Add Protein Pathway

extreme_final_df = df_transformed.copy()
ext_S_final_df = stoich_custom.copy()
int_S_final_df = stoich_df_sorted.copy()

if 'Protein' not in extreme_final_df.columns:
    extreme_final_df['Protein'] = 0.0

pseudo_path = pd.Series(0, index=extreme_final_df.columns)
pseudo_path['Protein'] = 1219

if not ((extreme_final_df == pseudo_path.values).all(axis=1)).any():
    extreme_final_df = pd.concat([extreme_final_df, pseudo_path.to_frame().T], ignore_index=True)

for df in [ext_S_final_df, int_S_final_df]:
    if 'Protein' not in df.columns:
        df.loc[:, 'Protein'] = 0.0  

# Reorder columns 
extreme_final_df = extreme_final_df[int_S_final_df.columns]

print(" Done! ✅")

# -------------------------------
# Data Preparation
# -------------------------------

print("Adjusting to correct units & define other parameters...")

# Correct Formatting
extreme_path = extreme_final_df.to_numpy(dtype=float) 
stoich_int = int_S_final_df.to_numpy(dtype=float)
stoich_ext = ext_S_final_df.to_numpy(dtype=float)

# Unit Basis
µmax = 0.75 # literature value
mmol_in_liter = 55510 # H2O
Km = 0.028/mmol_in_liter # global Km (batch culture)
Vmax = µmax / extreme_final_df['BIOMASS_Ecoli_core_w_GAM'].max() 
print(f"Vmax:{Vmax}")

# adjust for maximum reaction rate
extreme_path = extreme_path * Vmax
num_reactions = stoich_int.shape[1]
num_paths = extreme_path.shape[0]


# Create the stoich_biomass array
reaction_names = list(extreme_final_df.columns)
biomass_index = reaction_names.index(biomass_col)
stoich_biomass = np.zeros(len(reaction_names))
stoich_biomass[biomass_index] = 1.0  # set 1 at biomass reaction 

# Define other required arrays with sensible defaults

react_rate = np.full(num_reactions, Vmax)
react_rate[biomass_index] = µmax
met_noise = 0.00238 * Vmax
mich_ment = np.full(num_reactions, Km)
met_ext_total = np.full(2, mmol_in_liter) # mmol of water molecules in 1 liter of water
exp_pot[-1] = 0.4475 # set protein pathway 

print(" Done! ✅")

# -------------------------------
# Initialise Ideal Microbe
# -------------------------------

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

# Basic expression potential
#np.save(npy_path /'EFV_default_exp_pot.npy', microbe.exp_pot)

print(" Initialised Microbe! 🦠")

substrate_names = {0: "Fructose", 1: "Glucose"}

for i in [0, 1]:
    with np.errstate(invalid='ignore', divide='ignore'):
        result = microbe.infer_monod_parameters(np.zeros(2), i, met_ext_total)
    r, m, p, p0, K = result
    print(f"{substrate_names[i]} Monod parameters → r: {r:.4f} (per hour), K: {K:.6f} (mmol/L)")

# Initialize Culture Object
culture = im.Culture([microbe],0,np.asarray([0, 0]), met_ext_total)

print(" Initialised Culture! 🧫")


# optinal break here for fitting

#exit()

# -------------------------------
# Generate Plots
# -------------------------------

print("Constructing Growth Plane...")

### Growth Plane

# Define baseline and setup
met_ext = np.array([0.001, 0.001])
met_ext_index = [0, 1]
met_ext_max = [K, K]

# Create the actual plot
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
ax.set_xlabel("Fructose [mM]", fontsize=10)
ax.set_ylabel("Glucose [mM]", fontsize=10)
ax.set_title("Growth Rate", fontsize=12)

plt.tight_layout()
plt.savefig(output_dir / "E_coli_core_growthplane_EFVs.png", dpi=300)  
plt.savefig(output_dir / "E_coli_core_growthplane_EFVs.svg")  
plt.show()

print(" Saved Growth Plane! 💾")

### Microbe and Nutrient levels

print("Running Growth and Nutrient Consumption Simulations...")

# Choose meaningful nutrient levels 
# mimick Lendenmann after 2.1h (lag phase ca done): 0.015 mM Glc and 0.012 mM Frc and 0.0098 g/L cells
t_span = (0, 8)
mic_level0 = np.array([0.0016])
met_ext0 = np.array([2.1152/180 , 2.7585/180])

# Run the simulation
solution = culture.cr_model_ode(t_span, mic_level0, met_ext0, atol=1e-8, rtol=1e-8)
mic_level, met_ext = culture.slice_cr_solution(solution)

# Extract time points from the solution
times = solution.t  

# Set Seaborn whitegrid style
plt.rcParams.update({'font.size': 10})

# Plot Microbe Growth
plt.figure(figsize=(8, 4))
plt.plot(times, mic_level[0], marker='o', color='black')

# Titles and labels
plt.title("E. coli Growth", fontsize=12)
plt.xlabel("Time [h]")
plt.ylabel("Growth Level [g/L]")
plt.grid(
    True,               
    which='major',     
    linewidth=0.5,    
    color='gray',      
    alpha=0.3           
)
plt.tight_layout()

# Styling the plot frame (thinner, cleaner)
ax = plt.gca()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(0.8)
ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)

# Save and show
plt.savefig(output_dir / "E_coli_core_microbegrowth_EFVs.png", dpi=300)  
plt.savefig(output_dir / "E_coli_core_microbegrowth_EFVs.svg")  
plt.show()

print(" Saved Microbe Growth Plot! 💾")

# Generate 5 colors from the 'rocket' palette
colors = sns.color_palette("rocket", 10)

# Use the first (index 0) and fourth (index 3) colors
color1 = colors[2]
color2 = colors[7]

# Create figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# Plot Metabolite 1 with first rocket color
axes[0].plot(times, met_ext[0], marker='s', color=color1)
axes[0].set_title("Fructose",fontsize=12)
axes[0].set_xlabel("Time [h]")
axes[0].set_ylabel("Concentration [mM]")
axes[0].grid(True, linewidth=0.5, alpha=0.3)

# Plot Metabolite 2 with fourth rocket color
axes[1].plot(times, met_ext[1], marker='s', color=color2)
axes[1].set_title("Glucose",fontsize=12)
axes[1].set_xlabel("Time [h]")
axes[1].grid(True, linewidth=0.5, alpha=0.3)

# Styling the plot frames (apply to both axes)
for ax in axes:
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)

# Final formatting
plt.tight_layout()
plt.savefig(output_dir / "E_coli_core_nutrientlevel_EFVs.png", dpi=300)  
plt.savefig(output_dir / "E_coli_core_nutrientlevel_EFVs.svg")  
plt.show()

print(" Saved Nutrient Utilisation Plot! 💾")

# Create a DataFrame with glucose, fructose, and microbe levels
# Create a DataFrame with time, glucose, fructose, and microbe levels
df_out = pd.DataFrame({
    "time_h": times,
    "glc": met_ext[1],
    "frc": met_ext[0],
    "mic": mic_level[0]
})

# Save to CSV
output_file = processed_csv_path / "E_coli_core_sim_EFVs.csv"
df_out.to_csv(output_file, sep=';', index=False)

print(f"Saved simulation data to {output_file} 📁")

print('Everything complete! 🎉')

