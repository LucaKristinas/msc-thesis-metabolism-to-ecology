# Using the E. coli core metabolic model (Orth et al.) under aerobic conditions (OFVs)
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

# -------------------------------
# File Paths
# -------------------------------

# Add Repo Root 
project_root = Path(__file__).resolve().parents[3] 

# processed path
processed_path = project_root / "data" / "processed" 
processed_csv_path = processed_path / "csv_files"
output_dir = processed_path / "figures"

# -------------------------------
# Data
# -------------------------------

print("Loading metabolic structure data...")

# Import the external, internal stochiometric matrix and the OFVs
ext_S_df = pd.read_csv(processed_csv_path / "ecoli_sim_e_coli_core_ext_S.csv", index_col=0)
int_S_df = pd.read_csv(processed_csv_path / "ecoli_sim_e_coli_core_int_S.csv", index_col=0)
ofv_df = pd.read_csv(processed_csv_path / "ecoli_sim_e_coli_core_OFVs.csv", index_col=0)

# Define the columns of interest
biomass_col = "BIOMASS_Ecoli_core_w_GAM"
fru_col = "EX_fru_e_rev"
glc_col = "EX_glc__D_e_rev"

print(" Done! ✅")

# -------------------------------
# Data Preparation
# -------------------------------

print("Preparing the metabolic structure data for simulations...")

# Sort columns alphabetically
ext_S_df_sorted = ext_S_df[sorted(ext_S_df.columns)]
int_S_df_sorted = int_S_df[sorted(int_S_df.columns)]
ofv_df_sorted   = ofv_df[sorted(ofv_df.columns)]

# Normalisation by carbon source
df_normalized = ofv_df_sorted.copy()

only_fru = 0
only_glc = 0
both = 0
neither = 0

for idx, row in ofv_df_sorted.iterrows():
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

# Add dummy reaction/ pathway for proteins not related to metabolism

extreme_final_df = df_normalized.copy()
ext_S_final_df = ext_S_df_sorted.copy()
int_S_final_df = int_S_df_sorted.copy()

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

print(" Done! ✅")

print("Adjusting to correct units & define other parameters...")

# Basis values
µmax = 0.75 # literature value
mmol_in_liter = 55510 # H2O only
Km = 0.028/mmol_in_liter # global Michaelis-Menten constant 
Vmax = µmax / extreme_final_df['BIOMASS_Ecoli_core_w_GAM'].max() 

# adjust for maximum reaction rate
extreme_path = extreme_path * Vmax
num_reactions = stoich_int.shape[1]
num_paths = extreme_path.shape[0]

# Create the stoich_biomass array
reaction_names = list(extreme_final_df.columns)
biomass_index = reaction_names.index(biomass_col)
stoich_biomass = np.zeros(len(reaction_names))
stoich_biomass[biomass_index] = 1.0 

# converting wanted Ks
Ks = 3400/ 180160
print(f'Ks: {Ks}')

# Define other required arrays with sensible defaults
react_rate = np.full(num_reactions, Vmax)
react_rate[biomass_index] = µmax
met_noise = 0.00238 * Vmax
mich_ment = np.full(num_reactions, Km)
met_ext_total = np.full(2, mmol_in_liter) 
exp_pot = np.array([0.75,0.75,0.382])

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

# optional break

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
plt.savefig(output_dir / "E_coli_core_growthplane_OFVs.png", dpi=300)  
plt.savefig(output_dir / "E_coli_core_growthplane_OFVs.svg")  
plt.show()

print(" Saved Growth Plane! 💾")

### Microbe and Nutrient levels

print("Running Growth and Nutrient Consumption Simulations...")

# Choose meaningful nutrient levels 

# mimick Lendenmann after 2.1h (lag phase ca done): 0.015 mM Glc and 0.012 mM Frc and 0.0098 g/L cells
t_span = (0, 8)
mic_level0 = np.array([0.0016])
met_ext0 = np.array([2.1152/180 , 2.7585/180])

# mimick Deng after 6.5h (lag phase ca done): 8.1417 mM Glucose and 0.1518122 g/L cells 
t_span = (0, 3)
mic_level0 = np.array([0.1518])
met_ext0 = np.array([0 , 8.1417])


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
plt.savefig(output_dir / "E_coli_core_microbegrowth_OFVs.png", dpi=300)  
plt.savefig(output_dir / "E_coli_core_microbegrowth_OFVs.svg")  
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
plt.savefig(output_dir / "E_coli_core_nutrientlevel_OFVs.png", dpi=300)  
plt.savefig(output_dir / "E_coli_core_nutrientlevel_OFVs.svg")  
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
output_file = processed_csv_path / "E_coli_core_sim_OFVs.csv"
df_out.to_csv(output_file, sep=';', index=False)

print(f"Saved simulation data to {output_file} 📁")

print('Everything complete! 🎉')

