# ---------------------------------------
# Imports and Path Setup
# ---------------------------------------

import pandas as pd # type: ignore
import numpy as np # type: ignore
import pickle
from pathlib import Path
from cobra import Reaction # type: ignore
from cobra.io import read_sbml_model, write_sbml_model # type: ignore
from cobra.util.array import create_stoichiometric_matrix # type: ignore

# ---------------------------------------
# File Paths
# ---------------------------------------

# Add Repo Root 
project_root = Path(__file__).resolve().parents[3] 

# processed data
processed_path = project_root / "data" / "processed"
processed_csv_path = processed_path / "csv_files"

# raw data
raw_path = project_root / "data" / "raw" 
raw_sbml_path = raw_path / "sbml_files"
temp_sbml_path = raw_sbml_path / "temporary_files"
raw_csv_path = raw_path / "csv_files"

# ---------------------------------------
# Load Models
# ---------------------------------------

print("Data Import ...")

model_names = ['e_coli_core', 'iJO1366', 'iMS520', 'iKS1119']
models = {name: read_sbml_model(temp_sbml_path / f"{name}_neutral.xml") for name in model_names}

# ---------------------------------------
# Load DataFrames
# ---------------------------------------

# Load activity DataFrames
activity_dfs = {
    name: pd.read_csv(raw_csv_path / f"{name}_activity.csv")
    for name in model_names
}

# Load full flux distribution DataFrames
flux_df_dict = {
    name: pd.read_csv(raw_csv_path / f"{name}_fluxes.csv", index_col=0)
    for name in model_names
}

# Load dataframes with information on exchange fluxes
exchange_dfs = {
    name: pd.read_csv(raw_csv_path / f"{name}_edf.csv")
    for name in model_names
}

print(" Done! ✅")

# ---------------------------------------
# Modify Models Based on Reaction Activity
# ---------------------------------------

print("Modifying metabolic models for OFV compatibility ...")

missing_reactions = []

for model_id, model in models.items():
    df = activity_dfs[model_id]

    for _, row in df.iterrows():
        rxn_id = row['ID']
        activity = row['Activity']

        if activity in [2, 3, 5, 6]:
            try:
                original_rxn = model.reactions.get_by_id(rxn_id)

                # Create reversed reaction
                reversed_rxn = Reaction(id=rxn_id + '_rev')
                reversed_rxn.name = original_rxn.name + ' reversed'
                reversed_rxn.lower_bound = 0
                reversed_rxn.upper_bound = abs(original_rxn.lower_bound)
                reversed_rxn.add_metabolites({met: -coef for met, coef in original_rxn.metabolites.items()})

                model.add_reactions([reversed_rxn])

                if activity in [2, 5]:
                    model.reactions.remove(original_rxn)

            except KeyError:
                missing_reactions.append((model_id, rxn_id))

print("     ✅ Model modification complete.")
if missing_reactions:
    print("     ⚠️ Reactions not found in models:")
    for model_id, rxn_id in missing_reactions:
        print(f" - {rxn_id} in {model_id}")
else:
    print("     ✅ All reactions found and handled.")

print(" Done! ✅")

# ---------------------------------------
# Export Modified Models as SBML
# ---------------------------------------

print("Saving modified models ...")

for model_id, model in models.items():
    write_sbml_model(model, temp_sbml_path / f"{model_id}_rev.xml")

print(f"✅ Modified models exported to: {temp_sbml_path}")

# ---------------------------------------
# Transform Flux DataFrames for OFVs
# ---------------------------------------

print("Generating OFVs ...")

modified_flux_df_dict = {}

for name in model_names:
    df = flux_df_dict[name].copy()
    new_cols_to_add = {}
    renamed_count = 0
    split_count = 0

    for col in df.columns:
        col_data = df[col]
        negatives = col_data < 0
        positives = col_data > 0

        if col_data.eq(0).all():
            continue
        elif negatives.all() or (negatives | (col_data == 0)).all():
            new_col = col + '_rev'
            df.rename(columns={col: new_col}, inplace=True)
            df[new_col] = df[new_col].abs()
            renamed_count += 1
        elif negatives.any() and positives.any():
            rev_col = col + '_rev'
            rev_flux = col_data.where(col_data < 0, 0).abs()
            df[col] = col_data.where(col_data >= 0, 0)
            new_cols_to_add[rev_col] = rev_flux
            split_count += 1

    if new_cols_to_add:
        df = pd.concat([df, pd.DataFrame(new_cols_to_add, index=df.index)], axis=1)

    modified_flux_df_dict[name] = df
    print(f"🔄 {name}: {renamed_count} reactions renamed, {split_count} reactions split")

# Validation
for name, df in modified_flux_df_dict.items():
    if (df >= 0).all().all():
        print(f"✅ {name}: All fluxes are non-negative")
    else:
        print(f"❌ {name}: Contains negative fluxes")


print(" Done! ✅")

# ---------------------------------------
# Stoichiometric Matrix Analysis
# ---------------------------------------

print("Generating internal and exernal matrices ...")

stoich_matrices = {}
stoich_ranks = {}
stoich_dofs = {}

for name, model in models.items():
    S_dense = create_stoichiometric_matrix(model)
    rank = np.linalg.matrix_rank(S_dense)
    num_rxns = len(model.reactions)
    dof = num_rxns - rank

    met_ids = [met.id for met in model.metabolites]
    rxn_ids = [rxn.id for rxn in model.reactions]
    S_df = pd.DataFrame(S_dense, index=met_ids, columns=rxn_ids)

    stoich_matrices[name] = S_df
    stoich_ranks[name] = rank
    stoich_dofs[name] = dof

    print(f"\n📊 {name}")
    print(f"   Rank = {rank}, Reactions = {num_rxns}, DOF = {dof}")

# ---------------------------------------
# Construct External Reaction Matrices
# ---------------------------------------

# Define row labels (external sugar conditions)
e_coli_core_index = ['Fructose_ext', 'Glucose_ext']
general_index = ['Fructose_ext', 'Galactose_ext', 'Glucose_ext', 'Lactose_ext', 'Maltose_ext', 'Sucrose_ext']

# Create empty reaction-only matrices: copy of stoichiometric matrix (same columns = reactions), but no rows
reaction_only_matrices = {
    name: stoich_matrices[name].iloc[0:0, :].copy()
    for name in model_names
}

# Extract all reversed exchange reaction IDs (from exchange_dfs)
# Each entry in the dictionary maps model name to a list of reversed exchange reactions (e.g., 'EX_glc__D_e_rev')
rev_exchange_ids = {
    name: [f"{rxn_id}_rev" for rxn_id in df["ID"]]
    for name, df in exchange_dfs.items()
}

# Fill each external matrix with -1 one-hot encoded rows for reversed exchange reactions
for name in model_names:
    # Select the appropriate empty matrix and reversed reaction list
    S = reaction_only_matrices[name]
    rev_ids = rev_exchange_ids[name]

    # Filter rev_ids to include only those actually present in the matrix (i.e., model reactions)
    valid_rev_ids = [r for r in rev_ids if r in S.columns]

    # Choose the correct row index (i.e., sugar condition labels)
    row_index = e_coli_core_index if name == 'e_coli_core' else general_index

    # Truncate or extend row index to match number of valid reversed reactions
    if len(valid_rev_ids) > len(row_index):
        # Extend row_index by appending suffixes (e.g., 'Fructose_ext_0', 'Fructose_ext_1', etc.)
        row_index += [f"{row_index[-1]}_{i}" for i in range(len(valid_rev_ids) - len(row_index))]
    else:
        # Truncate if too long
        row_index = row_index[:len(valid_rev_ids)]

    # Construct new rows with -1 at the column of each valid reversed reaction
    new_rows = []
    for rxn_id in valid_rev_ids:
        # Create a zero-filled row
        row = pd.Series(0, index=S.columns)
        # Set -1 at the column corresponding to the reversed exchange reaction
        row[rxn_id] = -1
        new_rows.append(row)

    # Combine the new rows into a DataFrame with the appropriate index
    new_df = pd.DataFrame(new_rows, index=row_index)

    # Append these rows to the initially empty external matrix
    reaction_only_matrices[name] = pd.concat([S, new_df], axis=0)

    # Log progress
    print(f"➕ {name}: {len(new_rows)} external rows added to matrix with row labels: {row_index}")


print(" Done! ✅")

# ---------------------------------------
# Export All Final Outputs as CSV
# ---------------------------------------

print("Data Export...")

for name in model_names:
    # Export modified OFVs
    modified_flux_df_dict[name].to_csv(processed_csv_path / f"{name}_OFVs.csv")

    # Export internal stoichiometric matrix
    stoich_matrices[name].to_csv(processed_csv_path / f"{name}_int_S.csv")

    # Export external matrix
    reaction_only_matrices[name].to_csv(processed_csv_path / f"{name}_ext_S.csv")

print(" Done! ✅")

print('Everything complete! 🎉')
