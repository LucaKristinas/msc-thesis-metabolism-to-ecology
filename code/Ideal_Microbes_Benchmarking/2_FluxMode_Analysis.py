# ════════════════════════════════════════════════════════════════
# 1. Import Packages
# ════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import sys
from cobra import Reaction, Metabolite
from cobra.io import read_sbml_model
from sklearn.metrics import mean_squared_error
from pathlib import Path

# Add Repo Root to sys.path 
project_root = Path(__file__).resolve().parents[2]  
sys.path.append(str(project_root))

# Import Helper Functions
from src.utils import (read_rfile, read_mfile, check_steady_state_modes, softmax_from_reaction, flux_ratio, clean_values)

# ════════════════════════════════════════════════════════════════
# 2. Import & Export Paths
# ════════════════════════════════════════════════════════════════

# Raw Data 
raw_data_path = project_root / "data" / "raw"
csv_raw_path = raw_data_path / "csv_files"
mplrs_path = raw_data_path / "mplrs_data"

# Processed Data 
processed_data_path = project_root / "data" / "processed"
sbml_processed_path = processed_data_path / "sbml_files"
csv_processed_path = processed_data_path / "csv_files"

# MPLRS Files 
rfile_path = mplrs_path / "kmm_model_NO_bounds.rfile"
mfile_path = mplrs_path / "kmm_model_NO_bounds.mfile"
rfile_path_2 = mplrs_path / "kmm_model_with_bounds.rfile"
mfile_path_2 = mplrs_path / "kmm_model_with_bounds.mfile"
EFMs_S_file  = mplrs_path / "kmm_model_NO_bounds.sfile"
EFMs_file    = mplrs_path / "kmm_model_NO_bounds.efms"
EFVs_S_file  = mplrs_path / "kmm_model_with_bounds.sfile"
EFVs_file    = mplrs_path / "kmm_model_with_bounds.efvs"

# Flux Measurements 
flux_measurements_file = csv_raw_path / "MartinoPaper_Flux_Measurements.csv"

# ════════════════════════════════════════════════════════════════
# 3. Data Import 
# ════════════════════════════════════════════════════════════════

# Load model and data
print("📦 Loading model and raw data...")
model = read_sbml_model(str(sbml_processed_path / "klamt_martino_model_mplrs.xml"))

flux_measurement_df = pd.read_csv(flux_measurements_file)

reaction_list = read_rfile(rfile_path, remove_lambda=False)
metabolites_list = read_mfile(mfile_path)
EFMs_S_df = pd.read_csv(EFMs_S_file, header=None, sep=r'\s+')
EFMs_df = pd.read_csv(EFMs_file, header=None, sep=r'\s+')
EFMs_S_df.columns = reaction_list
EFMs_S_df.index = metabolites_list
EFMs_df.columns = reaction_list

reaction_list_2 = read_rfile(rfile_path_2, remove_lambda=False)
metabolites_list_2 = read_mfile(mfile_path_2)
reaction_list_2_efvs = reaction_list_2[:86]
EFVs_S_df = pd.read_csv(EFVs_S_file, header=None, sep=r'\s+')
EFVs_df = pd.read_csv(EFVs_file, header=None, sep=r'\s+')
EFVs_S_df.columns = reaction_list_2
EFVs_S_df.index = metabolites_list_2
EFVs_df.columns = reaction_list_2_efvs

print(' ✅ Done!')

### Sanity check
print("🔍 Steady State Sanity Check")
check_steady_state_modes(EFMs_df, model, Mode="EFM")
check_steady_state_modes(EFVs_df, model, Mode="EFV")

# ════════════════════════════════════════════════════════════════
# 4. Normalisation 
# ════════════════════════════════════════════════════════════════

print("Normalizing EFMs and EFVs by glucose uptake...")

### Normalisation
Carbon_Source = "EX_glc(e)"
# print("🔍 Checking for EFMs and EFVs without glucose uptake...")
# print(f"EFVs w/o {Carbon_Source}: {(EFVs_df[Carbon_Source] == 0).sum()}")
# print(f"EFMs w/o {Carbon_Source}: {(EFMs_df[Carbon_Source] == 0).sum()}")

EFVs_scaled_df = EFVs_df.div(EFVs_df[Carbon_Source], axis=0)
EFMs_scaled_df = EFMs_df.div(EFMs_df[Carbon_Source], axis=0)
print(" ✅ Complete!")

# ════════════════════════════════════════════════════════════════
# 5. Molecular Noise Computation
# ════════════════════════════════════════════════════════════════

print("📊 Computing MSEs across molecular noise range...")

### Compute MSEs over T range

measured_reactions = flux_measurement_df['Reaction'].tolist()
missing = [r for r in measured_reactions if r not in EFMs_scaled_df.columns]
# print(f"🔎 Missing reactions from metabolic model: {missing}")

exclude_reactions = ["PPCK-PPC"]
MSE_df = (
    flux_measurement_df[~flux_measurement_df["Reaction"].isin(exclude_reactions)]
    .assign(Reaction_labels=lambda df: df["Reaction"])
    .pipe(lambda df: df.set_index("Reaction_labels").T)
    .iloc[1:3]
)
MSE_df_scaled = MSE_df * 0.01

# ════════════════════════════════════════════════════════════════
# 5. MSE Computation
# ════════════════════════════════════════════════════════════════

T_values = np.logspace(-5, 0, 1000)
input_dfs_list = [EFMs_scaled_df, EFVs_scaled_df]
output_dfs_with_mse = []

for i, df in enumerate(input_dfs_list):
    mode_name = "EFMs" if i == 0 else "EFVs"
    results = []
    for T in T_values:
        try:
            q = softmax_from_reaction(df, "Biomass_Ecoli_core", T)
            flux = flux_ratio(df, "EX_glc(e)", q)
            flux["T"] = T
            results.append(flux)
        except Exception as e:
            print(f"T={T:.5f} failed for {mode_name}: {e}")
    output_df = pd.concat(results, ignore_index=True)
    
    target_reactions = MSE_df_scaled.columns.tolist()
    subset = output_df[target_reactions + ["T"]].copy()

    target_flux_01 = MSE_df_scaled.loc["Flux_Average_GR_01"].astype(float).values
    target_flux_02 = MSE_df_scaled.loc["Flux_Average_GR_02"].astype(float).values

    mse_01_list = [mean_squared_error(target_flux_01, row[target_reactions].astype(float)) for _, row in subset.iterrows()]
    mse_02_list = [mean_squared_error(target_flux_02, row[target_reactions].astype(float)) for _, row in subset.iterrows()]

    subset["MSE_GR_01"] = mse_01_list
    subset["MSE_GR_02"] = mse_02_list
    output_dfs_with_mse.append(subset)

print(" ✅ Complete!")

# ════════════════════════════════════════════════════════════════
# 5. Data Export 
# ════════════════════════════════════════════════════════════════

### Export data
print("💾 Saving output files to disk...")

EFMs_df.to_csv(csv_processed_path / 'EFMs.csv', index=False)
EFVs_df.to_csv(csv_processed_path / 'EFVs.csv', index=False)
MSE_df_scaled.to_csv(csv_processed_path / 'Flux_Measurements_Normalised.csv')
output_dfs_with_mse[0].to_csv(csv_processed_path / 'MSE_EFMs_df.csv', index=False)
output_dfs_with_mse[1].to_csv(csv_processed_path / 'MSE_EFVs_df.csv', index=False)

print(" ✅ Done!")

print('Everything complete! 🎉')