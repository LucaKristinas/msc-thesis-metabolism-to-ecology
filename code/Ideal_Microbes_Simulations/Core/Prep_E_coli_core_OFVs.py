# -------------------------------
# Imports
# -------------------------------

import pickle
from pathlib import Path  # type: ignore

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

from cobra import Reaction  # type: ignore
from cobra.io import read_sbml_model, write_sbml_model  # type: ignore
from cobra.util.array import create_stoichiometric_matrix  # type: ignore

# -------------------------------
# File Paths
# -------------------------------

# Project Root
project_root = Path(__file__).resolve().parents[3]  

# raw path
raw_path = project_root / "data" / "raw"
csv_path = raw_path / "csv_files"
sbml_path = raw_path / "sbml_files"
temp_path = sbml_path / "temporary_files"

# processed
processed_path = project_root / "data" / "processed" 
processed_csv_path = processed_path / "csv_files"
output_dir = processed_path / "figures"

# -------------------------------
# Data Import 
# -------------------------------

print("Loading E. coli core and iJO1366...")

# model data
model_names = ['e_coli_core', 'iJO1366']
models = {name: read_sbml_model(sbml_path / f"{name}.xml") for name in model_names}

# exchange reaction data
glc_id = "EX_glc__D_e"
gal_id = "EX_gal_e"
frc_id = "EX_fru_e"

# Sugar uptake conditions
conditions = [
    ("glc_1", 10.0, 0.0, 0.0),
    ("gal_1", 0.0, 10.0, 0.0),
    ("frc_1", 0.0, 0.0, 10.0),
]

# reactions
reaction_lists = {
    'e_coli_core': ['EX_fru_e', 'EX_glc__D_e'],
    'iJO1366': ['EX_fru_e', 'EX_gal_e', 'EX_glc__D_e']
}

print(" Done! ✅")

# -------------------------------
# Run FBA
# -------------------------------

print("Running FBAs...")

all_objectives = {}
all_fluxes = {}

for model_name, model in models.items():
    print(f"\n🔍 Running FBA for model: {model_name}")

    # Model-specific conditions
    if model_name == "iJO1366":
        model_conditions = conditions  # includes gal_1
    else:
        model_conditions = [c for c in conditions if c[0] != "gal_1"]

    # Identify available exchange reactions
    available_exchanges = {
        rxn_id: model.reactions.get_by_id(rxn_id)
        for rxn_id in [glc_id, gal_id, frc_id]
        if rxn_id in model.reactions
    }

    objectives = []
    fluxes = {}

    for name, glc, gal, frc in model_conditions:
        # Reset bounds
        for rxn in available_exchanges.values():
            rxn.lower_bound = 0.0

        # Set bounds for current condition
        if glc_id in available_exchanges:
            available_exchanges[glc_id].lower_bound = -glc
        if gal_id in available_exchanges:
            available_exchanges[gal_id].lower_bound = -gal
        if frc_id in available_exchanges:
            available_exchanges[frc_id].lower_bound = -frc

        # Run FBA
        sol = model.optimize()

        # Store results
        objectives.append({"condition": name, "objective": sol.objective_value})
        fluxes[name] = sol.fluxes.to_dict()

    # Store results for export
    all_objectives[model_name] = objectives
    all_fluxes[model_name] = fluxes

# Convert collected results into DataFrames and store
objective_dfs = {
    model_name: pd.DataFrame(obj_list)
    for model_name, obj_list in all_objectives.items()
}

# Store flux dictionaries directly
flux_distributions = {
    model_name: flux_dict
    for model_name, flux_dict in all_fluxes.items()
}

print(" Done! ✅")

# -------------------------------
# Reaction activity Analysis
# -------------------------------

print("Analysing Reaction activity...")

# Convert Flux Distributions to DataFrames

flux_df_dict = {}

for name, model in models.items():
    reaction_ids = [rxn.id for rxn in model.reactions]
    rows = []
    index_labels = []

    for combo, fluxes in flux_distributions[name].items():
        row = {rid: 0.0 for rid in reaction_ids}
        row.update(fluxes)
        rows.append(row)
        index_labels.append(str(combo))

    flux_df_dict[name] = pd.DataFrame(rows, index=index_labels)

# Subset Flux DataFrames and Add Row Sum

subset_flux_df_with_sum = {}

for name in model_names:
    df = flux_df_dict[name]
    target_rxns = reaction_lists[name]
    subset = df[target_rxns].fillna(0)
    subset['total_flux'] = subset.sum(axis=1)
    subset_flux_df_with_sum[name] = subset

# Analyze Reaction Activity

reaction_activity_info = {}
reaction_usage_stats = {}
activity_summary_tables = {}

for name, df in flux_df_dict.items():
    non_zero_counts = (df != 0).sum(axis=1)
    reaction_usage_stats[name] = non_zero_counts

    activity_map = {}
    num_combinations = len(df)

    for rxn in df.columns:
        v = df[rxn]
        pos, neg, zero = (v > 0).sum(), (v < 0).sum(), (v == 0).sum()

        if zero == num_combinations:
            activity = 0
        elif pos == num_combinations:
            activity = 1
        elif neg == num_combinations:
            activity = 2
        elif zero == 0 and pos and neg:
            activity = 3
        elif zero and pos and not neg:
            activity = 4
        elif zero and neg and not pos:
            activity = 5
        elif zero and pos and neg:
            activity = 6
        else:
            activity = -1

        activity_map[rxn] = activity

    activity_df = pd.DataFrame({'ID': activity_map.keys(), 'Activity': activity_map.values()})
    reaction_activity_info[name] = activity_df
    activity_summary_tables[name] = activity_df['Activity'].value_counts().sort_index()

normalized_activity_percentages = {
    name: (summary / len(reaction_activity_info[name]) * 100).round(2)
    for name, summary in activity_summary_tables.items()
}

# Plot: Reaction Activity by Model

sns.set_style("ticks")

# Prepare the DataFrame
comparison_df = pd.DataFrame(normalized_activity_percentages).fillna(0)
comparison_df.index.name = 'Activity'
comparison_df.reset_index(inplace=True)

long_df = comparison_df.melt(id_vars='Activity', var_name='Model', value_name='Percentage')
model_order = ['e_coli_core','iJO1366']
long_df['Model'] = pd.Categorical(long_df['Model'], categories=model_order, ordered=True)

# Custom colors and labels
palette = sns.color_palette("rocket", 10)
custom_palette = [palette[i] for i in [6, 2]]
label_map = {
    'e_coli_core': 'E. coli Core',
    'iJO1366': 'E. coli'
}
long_df['ModelLabel'] = long_df['Model'].map(label_map)

# Plot
plt.figure(figsize=(8, 4))

# Add reference lines
for y in [1, 10, 50, 80]:
    plt.axhline(y, color='black', linestyle=':', linewidth=0.8, zorder=0)
    plt.text(x=6.6, y=y * 0.85, s=f"{y}%", fontsize=8, ha='left')

# Bar plot
sns.barplot(
    data=long_df,
    x='Activity',
    y='Percentage',
    hue='ModelLabel',
    palette=custom_palette,
    zorder=2
)

# Axis settings
plt.yscale('log')
plt.ylim(1e-1, 100)
plt.xlabel("Activity Code", fontsize=10)
plt.ylabel("Percentage of Reactions (%)", fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Correct legend creation and styling
legend = plt.legend(
    loc='upper right',
    fontsize=9,
    frameon=True,
    labelspacing=0.4,
    handlelength=1.5,
    handletextpad=0.5,
    borderpad=0.5
)

# Customize the legend frame (AFTER getting the object)
frame = legend.get_frame()
frame.set_linewidth(0.5)
frame.set_edgecolor('black')
frame.set_facecolor('white')
frame.set_alpha(1.0)

# Save high-resolution image
plt.tight_layout()
plt.savefig(output_dir / "E_coli_OFVs_Activtiy.png", dpi=300)  
plt.savefig(output_dir / "E_coli_OFVs_Activtiy.svg")  
plt.show()

print(" Done! ✅")

# --------------------------------------------
# Model modification (split/rev reactions)
# --------------------------------------------

print("Adjusting Model reversibilities...")

# Modify Models Based on Reaction Activity

missing_reactions = []

for model_id, model in models.items():
    df = reaction_activity_info[model_id]

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

print("\n   ✅ Model modification complete.")
if missing_reactions:
    print(" ⚠️ Reactions not found in models:")
    for model_id, rxn_id in missing_reactions:
        print(f" - {rxn_id} in {model_id}")
else:
    print(" ✅ All reactions found and handled.")

print(" Done! ✅")

# --------------------------------------------
# Generate Ideal Microbes Input
# --------------------------------------------

print("Generating Ideal Microbe Input...")

# Transform Flux DataFrames for OFVs

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
        print(f"    ✅ {name}: All fluxes are non-negative")
    else:
        print(f"    ❌ {name}: Contains negative fluxes")

# Stoichiometric Matrix Analysis

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

    print(f"\n  📊 {name}")
    print(f"    Rank = {rank}, Reactions = {num_rxns}, DOF = {dof}")

# Construct External Reaction Matrices

# Define row labels (external sugar conditions)
e_coli_core_index = ['Fructose_ext', 'Glucose_ext']
general_index = ['Fructose_ext', 'Galactose_ext', 'Glucose_ext']

# Create empty reaction-only matrices: copy of stoichiometric matrix (same columns = reactions), but no rows
reaction_only_matrices = {
    name: stoich_matrices[name].iloc[0:0, :].copy()
    for name in model_names
}

# Each entry in the dictionary maps model name to a list of reversed exchange reactions (e.g., 'EX_glc__D_e_rev')
rev_exchange_ids = {
    name: [f"{rxn_id}_rev" for rxn_id in rxn_list]
    for name, rxn_list in reaction_lists.items()
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

# --------------------------------------------
# Data Export
# --------------------------------------------

print("Saving the data...")

# FBA Data

for model_name in all_objectives:
    obj_df = pd.DataFrame(all_objectives[model_name])
    obj_df.to_csv(csv_path / f"ecoli_sim_fba_objectives_{model_name}.csv", index=False)
    pd.to_pickle(all_fluxes[model_name], csv_path / f"fba_fluxes_{model_name}.pkl")

# Activity DataFrames

for model_name, df in reaction_activity_info.items():
    activity_path = csv_path / f'ecoli_sim_{model_name}_activity.csv'
    df.to_csv(activity_path, index=False)


# Full Flux Distribution DataFrames

for model_name, df in flux_df_dict.items():
    flux_path = csv_path / f'ecoli_sim_{model_name}_fluxes.csv'
    df.to_csv(flux_path)

# Export Modified Models as SBML

for model_id, model in models.items():
    write_sbml_model(model, temp_path / f"ecoli_sim_{model_id}_rev.xml")

# final results

for name in model_names:
    # Export modified OFVs
    modified_flux_df_dict[name].to_csv(processed_csv_path / f"ecoli_sim_{name}_OFVs.csv")

    # Export internal stoichiometric matrix
    stoich_matrices[name].to_csv(processed_csv_path / f"ecoli_sim_{name}_int_S.csv")

    # Export external matrix
    reaction_only_matrices[name].to_csv(processed_csv_path / f"ecoli_sim_{name}_ext_S.csv")

print(" Done! ✅")

print('Everything complete! 🎉')