# -------------------------------
# Imports
# -------------------------------

import pandas as pd  # type: ignore
from pathlib import Path  # type: ignore
from cobra.io import read_sbml_model, write_sbml_model  # type: ignore
from cobra import Model  # type: ignore

# -------------------------------
# File Paths
# -------------------------------

# Add Repo Root 
project_root = Path(__file__).resolve().parents[2] 

# raw data
raw_path = project_root / "data" / "raw" 
raw_sbml_path = raw_path / "sbml_files"
temp_sbml_path = raw_sbml_path / "temporary_files"
raw_csv_path = raw_path / "csv_files"

# -------------------------------
# Model Definitions
# -------------------------------

print("Loading SBML models...")

model_files = {
    "e_coli_core": "e_coli_core.xml",
    "iJO1366": "iJO1366.xml",
    "iMS520": "iMS520.xml",
    "iKS1119": "iKS1119.xml"
}

# Load models
models = {name: read_sbml_model(raw_sbml_path / file) for name, file in model_files.items()}

print(" Done! ✅")

# -------------------------------
# Initial FBA
# -------------------------------

print("Running initial FBA on each model...")

# Store original FBA results
fba_results = []

for name, model in models.items():
    solution = model.optimize()
    obj_val = solution.objective_value
    fba_results.append({
        "model": name,
        "stage": "before_modification",
        "objective_value": obj_val
    })
    print(f"        [{name}] Initial FBA objective value: {obj_val}")

print(" Done! ✅")

# -------------------------------
# Apply Minimal Environments
# -------------------------------

print("Applying minimal environment conditions...")

modifications = {
    "iKS1119": {
        "EX_his_L_e": 0.0,
        "EX_glc_e": -10.0,
        "EX_so4_e": -1000.0
    },
    "iJO1366": {
        "EX_cbl1_e": 0.0,
        "EX_glc__D_e": -10.0,
        "EX_o2_e": 0.0,
        "EX_cys__L_e": -1000.0
    },
    "e_coli_core": {
        "EX_glc__D_e": -10.0,
        "EX_o2_e": 0.0
    },
    "iMS520": {
        "EX_tungs(e)": 0.0,
        "EX_ch4s(e)": 0.0,
        "EX_ptrc(e)": 0.0,
        "EX_spmd(e)": 0.0,
        "EX_4abz(e)": 0.0,
        "EX_ncam(e)": 0.0,
        "EX_nac(e)": 0.0,
        "EX_so4(e)": -1000.0
    }
}

for name, changes in modifications.items():
    print(f"Modifying model: {name}")
    model = models[name]
    for rxn_id, lb in changes.items():
        model.reactions.get_by_id(rxn_id).lower_bound = lb
        print(f"  Set {rxn_id} lower bound to {lb}")
    write_sbml_model(model, temp_sbml_path / f"{name}_neutral.xml")
    print(f"  Saved modified model: {name}_neutral.xml")

print(" Done! ✅")

# -------------------------------
# FBA After Modification
# -------------------------------

print("Running FBA after environment modification...")

for name, model in models.items():
    solution = model.optimize()
    obj_val = solution.objective_value
    fba_results.append({
        "model": name,
        "stage": "after_modification",
        "objective_value": obj_val
    })
    print(f"        [{name}] FBA after modification objective value: {obj_val}")

print(" Done! ✅")

# -------------------------------
# Export FBA Results
# -------------------------------

print("Exporting FBA results to CSV...")

fba_df = pd.DataFrame(fba_results)
fba_df.to_csv(raw_csv_path / "FBA_results_minimal_environment.csv", index=False)

print(" Done! ✅")

# -------------------------------
# End of Script
# -------------------------------

print('Everything complete! 🎉')
