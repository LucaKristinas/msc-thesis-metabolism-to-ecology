# ════════════════════════════════════════════════════════════════
# 1. Import Libraries
# ════════════════════════════════════════════════════════════════

print("📦 Importing libraries...")

# Core
import pandas as pd
import numpy as np
import copy
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from itertools import chain, combinations


# Graphs
import networkx as nx

# COBRApy
import cobra
from cobra import Model, Reaction
from cobra.core import Group
from cobra.flux_analysis.fastcc import fastcc
from cobra.io import read_sbml_model, save_matlab_model, write_sbml_model
from cobra.util.array import create_stoichiometric_matrix
from cobra.util.solver import linear_reaction_coefficients

print("✅ Libraries imported.")

# ════════════════════════════════════════════════════════════════
# 2. Configuration & Helper Functions
# ════════════════════════════════════════════════════════════════

# Decide whether to run full single lumping (true) or only subset for smaller tests / debugging
RUN_FULL_DATAFRAME = True  # Set to True to run for all entries

# Add the repo root to sys.path so Python can find "src" & import helper functions
project_root = Path(__file__).resolve().parents[2]  
sys.path.append(str(project_root))
from src.utils import (lump_reaction, are_reactions_interconnected)

# ════════════════════════════════════════════════════════════════
# 3. Import & Export Paths
# ════════════════════════════════════════════════════════════════

# Raw data paths
raw_data_path = project_root / "data" / "raw"
csv_raw_path = raw_data_path / "csv_files"
sbml_raw_path = raw_data_path / "sbml_files"
mat_raw_path = raw_data_path / "matlab_files"
temporary_sbml_path = sbml_raw_path / "temporary_files"

# Processed data paths
processed_data_path = project_root / "data" / "processed"
csv_processed_path = processed_data_path / "csv_files"
sbml_processed_path = processed_data_path / "sbml_files"

# model
model_path = sbml_raw_path / "iMS520.xml"


# ════════════════════════════════════════════════════════════════
# 4. Data Import
# ════════════════════════════════════════════════════════════════ 

print("📄 Reading Data...")

# reading B. longum model
model = read_sbml_model(str(model_path))

# reading F2C2 data
Coupled_Pairs_df = pd.read_csv(csv_raw_path / "fctable_iMS520.csv", header=None)
F2C2_Blocked_Reactions_df = pd.read_csv(csv_raw_path / "blocked_reactions_iMS520.csv", header=None)

# ════════════════════════════════════════════════════════════════════════════════
# 5. Model Pre-Processing (Preparation of model for matlab and fastcc processing)
# ════════════════════════════════════════════════════════════════════════════════

print("📋 Preparation of model for matlab and fastcc processing...")

# Create a deep copy of the model to avoid modifying the original
model_copy = copy.deepcopy(model)

# allow flux in and out for all exchange reactions
for rxn in model_copy.exchanges:
    rxn.lower_bound = -10
    rxn.upper_bound = 1000

# Export as .mat for matlab applications
save_matlab_model(model_copy, mat_raw_path / "iMS520.mat")

# ═════════════════════════════════════════════════════════
# 6. Model Processing (FASTCC and F2C2 Data Preparation)
# ═════════════════════════════════════════════════════════

print("⚙️ Model Processing (FASTCC and F2C2 Data Preparation...")

# Reduce model with COBRApy FASTCC and export
consistent_generic_model = fastcc(model_copy)
write_sbml_model(consistent_generic_model, temporary_sbml_path / "iMS520_consistent_generic.xml")

# Add reaction annotations to match F2C2 blocked reactions
model_rxns = [rxn.id for rxn in model.reactions]
F2C2_Blocked_Reactions_df.loc[len(F2C2_Blocked_Reactions_df)] = model_rxns

# Identify unblocked reactions from the second row (index 1)
unblocked_mask = F2C2_Blocked_Reactions_df.loc[0] == 0
unblocked_reactions = F2C2_Blocked_Reactions_df.loc[1][unblocked_mask].tolist()

# Update Coupled_Pairs_df index and columns with unblocked reactions
Coupled_Pairs_df.index = unblocked_reactions
Coupled_Pairs_df.columns = unblocked_reactions

# Build a graph of fully coupled reactions
G = nx.Graph()
for row in Coupled_Pairs_df.index:
    for col in Coupled_Pairs_df.columns:
        if Coupled_Pairs_df.loc[row, col] == 1 and row != col:
            G.add_edge(row, col)

# Find connected components (fully coupled groups)
coupled_groups = list(nx.connected_components(G))

# Create a mapping from reaction ID to COBRA group name
reaction_to_group = {}
for group in model.groups:
    group_name = group.name
    for member in group.members:
        if hasattr(member, 'id'):
            reaction_to_group[member.id] = group_name

# Combine coupled groups with their annotated groups in the COBRA model 
# and obtain in- and output of coupled groups)
combined_data = []
for group in coupled_groups:
    group_reactions = list(group)
    
    group_names = set()
    for rxn in group_reactions:
        if rxn in reaction_to_group:
            group_names.add(reaction_to_group[rxn])
    
    stoich = defaultdict(float)
    for rxn_id in group_reactions:
        rxn = model.reactions.get_by_id(rxn_id)
        for met, coeff in rxn.metabolites.items():
            stoich[met] += coeff

    inputs = [met.id for met, coeff in stoich.items() if coeff < 0]
    outputs = [met.id for met, coeff in stoich.items() if coeff > 0]

    combined_data.append({
        "Coupled_Reactions": group_reactions,
        "COBRA_Groups": list(group_names) if group_names else ["Unassigned"],
        "Num_Inputs": len(inputs),
        "Num_Outputs": len(outputs),
        "Input_Metabolites": inputs,
        "Output_Metabolites": outputs
    })

Fully_Coupled_df = pd.DataFrame(combined_data)

# Check whether the reactions per group form a single 
# connected component (i.e., fully connected cluster)
Fully_Coupled_df['Cluster'] = Fully_Coupled_df['Coupled_Reactions'].apply(
    lambda rxn_list: are_reactions_interconnected(consistent_generic_model, rxn_list)
)

# Remove Entries that are not fully connected
Fully_Coupled_df = Fully_Coupled_df[Fully_Coupled_df['Cluster'] != False]

# Remove objective function (if in df)
objective_id = "R_biomass_BIF"
Fully_Coupled_df['Reactions'] = Fully_Coupled_df['Coupled_Reactions'].apply(lambda rxns: [r for r in rxns if r != objective_id])

# ═════════════════════════════════════════════════════════════════════════
# 7. Model 'Single Reaction' Lumping, export, re-import & FBA Sanity Check
# ═════════════════════════════════════════════════════════════════════════

print("🧬 Beginning model lumping process...")
all_models = {}
all_logs = []

if RUN_FULL_DATAFRAME:
    df_to_process = Fully_Coupled_df
    print("🚀 Running for the entire DataFrame...")
else:
    df_to_process = Fully_Coupled_df.head(5)
    print("🧪 Running for first 5 entries only (debug mode)...")

for idx, row in df_to_process.iterrows():

    print(f"\n🔁 Processing group {idx}...")
    subset_df = pd.DataFrame([{
        "Coupled_Reactions": row['Reactions'],
        "COBRA_Groups": row['COBRA_Groups']
    }])
    
    lumped_model, lumping_log = lump_reaction(
        consistent_generic_model,
        subset_df,
        verbose=True,
        label_type="Group"
    )
    
    file_path = sbml_raw_path / "temporary_files" / f"lumped_model_{idx}.xml"
    print(f"💾 Saving lumped model to: {file_path}")
    write_sbml_model(lumped_model, file_path)

    print("📥 Re-importing lumped model...")
    lumped_model_reimported = read_sbml_model(file_path)

    model_name = f"lumped_model_{idx}"
    all_models[model_name] = lumped_model            # Original (before export)
    all_models[model_name + "_imported"] = lumped_model_reimported  # Re-imported


    lumping_log['Model_Name'] = model_name
    lumping_log['Coupled_Reactions'] = [row['Reactions']]
    all_logs.append(lumping_log)

# Add base models
print("\n🗂️ Adding base models to model dictionary...")
all_models.update({
    "original_model": model,
    "model_copy": model_copy,
    "consistent_generic_model": consistent_generic_model
})

print("📈 Running FBA on all models and matching imported versions...")

# Run FBA and store results
fba_results = {}

for name, mod in all_models.items():
    print(f"⚙️ Optimizing model: {name}")
    try:
        sol = mod.optimize()
        value = round(sol.objective_value, 4) if sol.objective_value is not None else None
        print(f"    Objective value: {value}")
        fba_results[name] = value
    except Exception as e:
        print(f"    Optimization failed for {name}: {e}")
        fba_results[name] = None

# Create a comparison DataFrame for lumped models
original_vs_imported = []

for name, value in fba_results.items():
    if name.startswith("lumped_model_") and not name.endswith("_imported"):
        imported_name = name + "_imported"
        imported_value = fba_results.get(imported_name, None)
        original_vs_imported.append({
            "Model": name,
            "Original_Model_Objective": value,
            "Reimported_Model_Objective": imported_value
        })

comparison_df = pd.DataFrame(original_vs_imported)

print("📦 Compiling final summary table...")
log_df = pd.concat(all_logs, ignore_index=True)
log_df = log_df[['Model_Name', 'Pseudo_Reaction_ID', 'Original_Reactions', 'COBRA_Groups']]
log_df = log_df.rename(columns={"Model_Name": "Model"})
final_df = log_df.merge(comparison_df, on="Model", how="left")

# ═════════════════════════════════════
# 8. FBA pre-/ post-import analysis
# ═════════════════════════════════════

print("📊 Starting FBA pre-/ post-import analysis...")

# Create a clean summary dataframe
final_df_v2 = final_df.drop(columns=['Model', 'Original_Reactions', 'COBRA_Groups'])
FBA_df = pd.concat([Fully_Coupled_df.reset_index(drop=True), final_df_v2.reset_index(drop=True)], axis=1)
FBA_df = FBA_df.drop(columns=['Cluster'])

# Define subsets depending on re-imported FBA objective & report results
epsilon = 1e-6
FBA_df_positive = FBA_df[FBA_df['Reimported_Model_Objective'] > epsilon]
FBA_df_filtered_out = FBA_df[FBA_df['Reimported_Model_Objective'] <= epsilon]

print(f"Total number of coupled reactions: {len(FBA_df)}")
print(f"    Models valid after single reaction lumping: {len(FBA_df_positive)}")
print(f"    Models broken after single reaction lumping: {len(FBA_df_filtered_out)}")

print("\n🔄 Performing full lumping on positive subset...")

### Complete Lumping Test ###

# Based on 'positive' models, perform complete lumping

lumped_model, lumping_log = lump_reaction(consistent_generic_model, FBA_df_positive, verbose=False)
write_sbml_model(lumped_model, temporary_sbml_path / "iMS520_lumped_model_full.xml")
lumped_model_reimported = read_sbml_model(temporary_sbml_path / "iMS520_lumped_model_full.xml")

# Run FBA
sol1 = lumped_model.optimize()
sol2 = lumped_model_reimported.optimize()

val1 = round(sol1.objective_value, 4) if sol1.objective_value is not None else None
val2 = round(sol2.objective_value, 4) if sol2.objective_value is not None else None

print(f"    📈 Original lumped model FBA objective: {val1}")
print(f"    📈 Reimported lumped model FBA objective: {val2}")

### Selective Lumping Test after consistency check ###

# run fastcc on lumped_model and report inconsistent reactions
model_consistent = fastcc(lumped_model) 

all_rxns = {rxn.id for rxn in lumped_model.reactions}
consistent_rxns = {rxn.id for rxn in model_consistent.reactions}
inconsistent_rxns = list(all_rxns - consistent_rxns)

inconsistent_indices = []

for rxn_id in inconsistent_rxns:
    rxn = lumped_model.reactions.get_by_id(rxn_id)
    print(f" - {rxn.id}: {rxn.name}")
    
    # If reaction ID matches "Pseudo_Group_<number>", extract the number
    match = re.search(r"Pseudo_Group_(\d+)", rxn.id)
    if match:
        inconsistent_indices.append(int(match.group(1)))

# create a subset of FBA_df_positive removing the row where rxn_id in inconsistent_rxns
Consistent_FBA_df = FBA_df.drop(index=inconsistent_indices).reset_index(drop=True)
Consistent_FBA_df = Consistent_FBA_df[Consistent_FBA_df['Reimported_Model_Objective'] > epsilon]

print(f"    \n🧹 {len(Consistent_FBA_df)} entries remain after removing inconsistent reactions")

# Run Lumping Again with Consistent Subset

print("🔁 Running second round of lumping on consistent subset...")
lumped_model2, lumping_log2 = lump_reaction(consistent_generic_model, Consistent_FBA_df, verbose=False)
write_sbml_model(lumped_model2, temporary_sbml_path / "iMS520_lumped_model_selective.xml")
lumped_model2_reimported = read_sbml_model(temporary_sbml_path / "iMS520_lumped_model_selective.xml")

# FBA sanity check
sol3 = lumped_model2.optimize()
sol4 = lumped_model2_reimported.optimize()

val3 = round(sol3.objective_value, 4) if sol3.objective_value is not None else None
val4 = round(sol4.objective_value, 4) if sol4.objective_value is not None else None

print(f"    ✅ Consistent lumped model objective: {val3}")
print(f"    ✅ Reimported consistent lumped model objective: {val4}")

# ══════════════════════════════════
# 9. Data Export
# ══════════════════════════════════

print(f"💾 Saving the data...")

final_df.to_csv(csv_raw_path / "Cobrapy_Test_Single_Reactions_Lump.csv", index=False)

print("🎉 All done!")

# To check valid comparisons of all relevant models, check the memote reports in the data rep