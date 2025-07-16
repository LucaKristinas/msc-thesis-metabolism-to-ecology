# ════════════════════════════════════════════════════════════════
# 1. Import Packages
# ════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import copy
import sys
from pathlib import Path
from cobra.io import read_sbml_model
from sklearn.metrics import mean_squared_error

# ════════════════════════════════════════════════════════════════
# 2. Import & Export Paths
# ════════════════════════════════════════════════════════════════

# Root of the project 
project_root = Path(__file__).resolve().parents[2]

# Raw Data 
raw_data_path = project_root / "data" / "raw"
csv_raw_path = raw_data_path / "csv_files"

# Processed Data 
processed_data_path = project_root / "data" / "processed"
sbml_processed_path = processed_data_path / "sbml_files"
csv_processed_path = processed_data_path / "csv_files"

# ════════════════════════════════════════════════════════════════
# 3. Import Data
# ════════════════════════════════════════════════════════════════

print("📦 Loading model and data...")

MSE_df_scaled = pd.read_csv(csv_processed_path / 'Flux_Measurements_Normalised.csv', index_col=0)

model = read_sbml_model(str(sbml_processed_path / "klamt_martino_model_mplrs.xml"))

methods = ['MAXENT', 'UNIFORM', 'FBA']
flux_dfs = []
for method in methods:
    file_path = csv_raw_path / f'martino_{method}_fluxes.csv'
    df = pd.read_csv(file_path, header=None)
    df.columns = ['Reaction_labels', 'Flux_Average_GR_02']
    df = df.set_index("Reaction_labels").T
    df["Type"] = method
    df = df.drop(columns=["PPCK-PPC"])
    flux_dfs.append(df)

martino_flux_df = pd.concat(flux_dfs).set_index("Type")

maxent_curve_GR02_df = pd.read_csv(csv_raw_path / 'martino_curve_GR02.csv', header=None)
maxent_curve_GR02_df.columns = ["beta", "MSE"]

print(" Done! ✅")

# ════════════════════════════════════════════════════════════════
# 4. FBA Calculations
# ════════════════════════════════════════════════════════════════

# "EFV-relevant" version

print("Running FBA (flux-bounded)...")
solution = model.optimize()
fba_fluxes = solution.fluxes.copy()
current_glucose_flux = abs(fba_fluxes['EX_glc(e)'])
scale_factor = 1 / current_glucose_flux
fba_fluxes_rescaled = fba_fluxes * scale_factor

if "ICD" in fba_fluxes_rescaled.index:
    fba_fluxes_rescaled.rename(index={"ICD": "ICDHyr"}, inplace=True)

common_reactions = fba_fluxes_rescaled.index.intersection(MSE_df_scaled.columns)
y_true_GR01 = MSE_df_scaled.loc["Flux_Average_GR_01", common_reactions].values.astype(float)
y_true_GR02 = MSE_df_scaled.loc["Flux_Average_GR_02", common_reactions].values.astype(float)
y_pred = fba_fluxes_rescaled.loc[common_reactions].values.astype(float)
COBRA_FBA_mse_GR02 = mean_squared_error(y_true_GR02, y_pred)
COBRA_FBA_mse_GR01 = mean_squared_error(y_true_GR01, y_pred)

print(" Done! ✅")

# "EFM-relevant" version

print("Running FBA (standard COBRA bounds)...")
no_bounds_model = copy.deepcopy(model)
for rxn_id in ['ATPM']:
    if rxn_id in no_bounds_model.reactions:
        no_bounds_model.reactions.get_by_id(rxn_id).bounds = (0, 1000)

solution_2 = no_bounds_model.optimize()
fba_fluxes_2 = solution_2.fluxes.copy()
current_glucose_flux_2 = abs(fba_fluxes_2['EX_glc(e)'])
scale_factor_2 = 1 / current_glucose_flux_2
fba_fluxes_rescaled_2 = fba_fluxes_2 * scale_factor_2

if "ICD" in fba_fluxes_rescaled_2.index:
    fba_fluxes_rescaled_2.rename(index={"ICD": "ICDHyr"}, inplace=True)

common_reactions_2 = fba_fluxes_rescaled_2.index.intersection(MSE_df_scaled.columns)
y_pred_2 = fba_fluxes_rescaled_2.loc[common_reactions_2].values.astype(float)
COBRA_FBA_mse_std_flux_bounds_GR02 = mean_squared_error(y_true_GR02, y_pred_2)
COBRA_FBA_mse_std_flux_bounds_GR01 = mean_squared_error(y_true_GR01, y_pred_2)

print(" Done! ✅")

# ════════════════════════════════════════════════════════════════
# 4. MSE Calculations (De Martino et al., 2018)
# ════════════════════════════════════════════════════════════════

print("Calculating MSEs for Martino Model Predictions...")

maxent_mse = maxent_curve_GR02_df["MSE"].min()
uniform_mse = maxent_curve_GR02_df.loc[maxent_curve_GR02_df["beta"].idxmin(), "MSE"]
FBA_mse = maxent_curve_GR02_df.loc[maxent_curve_GR02_df["beta"].idxmax(), "MSE"]

mse_results = {}
for method in martino_flux_df.index:
    print(f"  ↳ Computing MSE for {method}")
    common_reactions = martino_flux_df.columns.intersection(MSE_df_scaled.columns)
    y_true_GR02 = MSE_df_scaled.loc["Flux_Average_GR_02", common_reactions].values.astype(float)
    y_pred = martino_flux_df.loc[method, common_reactions].values.astype(float)
    mse_results[method] = mean_squared_error(y_true_GR02, y_pred)

martino_flux_mse_df = pd.DataFrame.from_dict(mse_results, orient='index', columns=['MSE'])

# Compute scaling factors
martino_mse_values = martino_flux_mse_df["MSE"].to_dict()
uniform_mse_scale = uniform_mse / martino_mse_values["UNIFORM"]
maxent_mse_scale = maxent_mse / martino_mse_values["MAXENT"]
fba_mse_scale = FBA_mse / martino_mse_values["FBA"]

print(" Scaling factors to match Panel B MSE values:")
print(f"        UNIFORM → {round(uniform_mse_scale,2)}x")
print(f"        MAXENT  → {round(maxent_mse_scale,2)}x")
print(f"        FBA     → {round(fba_mse_scale,2)}x")

print(" Done! ✅")

# ════════════════════════════════════════════════════════════════
# 5. Data Export
# ════════════════════════════════════════════════════════════════

print("💾 Exporting results...")

# Create scaled lines
line_UNIFORM = uniform_mse / uniform_mse_scale
line_MAXENT = maxent_mse / maxent_mse_scale
line_FBA = FBA_mse / fba_mse_scale

# Save MSE lines
line_df = pd.DataFrame({
    'Line_UNIFORM': [line_UNIFORM],
    'Line_MAXENT': [line_MAXENT],
    'Line_FBA': [line_FBA]
})
line_df.to_csv(csv_processed_path / 'Martino_GR02_MSE.csv', index=False)

# Save FBA flux vectors
fba_fluxes_rescaled.to_csv(csv_processed_path / 'Fba_fluxes.csv')
fba_fluxes_rescaled_2.to_csv(csv_processed_path / 'Fba_fluxes_std_bounds.csv')

# Save FBA MSE values
fba_mse_df = pd.DataFrame({
    "MSE": [COBRA_FBA_mse_GR01, COBRA_FBA_mse_GR02, COBRA_FBA_mse_std_flux_bounds_GR01, COBRA_FBA_mse_std_flux_bounds_GR02]
}, index=["FBA_flux_bounds_01", "FBA_flux_bounds_02","FBA_std_bounds_01", "FBA_std_bounds_02"])
fba_mse_df.to_csv(csv_processed_path / 'fba_mse_values_GR01_GR02.csv')

print(" Done! ✅")

print('Everything complete! 🎉')