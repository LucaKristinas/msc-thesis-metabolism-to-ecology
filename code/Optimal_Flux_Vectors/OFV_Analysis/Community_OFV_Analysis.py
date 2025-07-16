# ════════════════════════════════════════════════════════════════
# 1. Import Packages
# ════════════════════════════════════════════════════════════════

# Standard Libraries
import sys
from pathlib import Path
import pickle
from collections import Counter

# Data Handling
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from pprint import pprint

# COBRApy (Constraint-Based Reconstruction and Analysis)
import cobra
from cobra.io import read_sbml_model

# Plotting and Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from matplotlib_venn import venn2, venn3

# Add Repo Root to sys.path 
project_root = Path(__file__).resolve().parents[3] 
sys.path.append(str(project_root))

# Import Helper Functions
from src.utils import (match_reactions_to_bigg, print_summary_to_file)

# ════════════════════════════════════════════════════════════════
# 2. Import & Export Paths
# ════════════════════════════════════════════════════════════════

# Raw Data 
raw_data_path = project_root / "data" / "raw"
raw_csv_path = raw_data_path / "csv_files"
sbml_path = raw_data_path / "sbml_files" / "temporary_files"
pkl_path = raw_data_path / "pkl_files"
txt_raw_path = raw_data_path / "txt_files"

# Processed Data 
processed_data_path = project_root / "data" / "processed"
processed_csv_path = processed_data_path / "csv_files"

# Output
output_path = processed_data_path / "figures"
txt_path = processed_data_path / "txt_files"

# ════════════════════════════════════════════════════════════════
# 3. Data Import
# ════════════════════════════════════════════════════════════════

print("Importing data...")

# import models in correct conditions

model_names = ['e_coli_core', 'iJO1366', 'iMS520', 'iKS1119']
models = {name: read_sbml_model(sbml_path / f"{name}_neutral.xml") for name in model_names}
models_list = models.copy()

# Import objectives, fva results and fba flux distributions

objective_dfs = {}
fva_dfs = {}
flux_distributions = {}

for name in model_names:
    obj_path = raw_csv_path / f'FBA_Objectives_{name}.csv'
    fva_path = raw_csv_path / f'FVA_{name}.csv'
    flux_path = pkl_path / f'FBA_Distributions_{name}.pkl'

    objective_dfs[name] = pd.read_csv(obj_path)
    fva_dfs[name] = pd.read_csv(fva_path)
    with open(flux_path, 'rb') as f:
        flux_distributions[name] = pickle.load(f)

# define important reactions

reaction_lists = {
    'e_coli_core': ['EX_fru_e', 'EX_glc__D_e'],
    'iJO1366': ['EX_fru_e', 'EX_gal_e', 'EX_glc__D_e', 'EX_sucr_e', 'EX_malt_e', 'EX_lcts_e'],
    'iKS1119': ['EX_fru_e', 'EX_gal_e', 'EX_glc_e', 'EX_sucr_e', 'EX_lcts_e', 'EX_malt_e'],
    'iMS520': ['EX_fru(e)', 'EX_gal(e)', 'EX_glc-D(e)', 'EX_lcts(e)', 'EX_malt(e)', 'EX_sucr(e)']
}

# BIGG Data

reaction_df = pd.read_csv(txt_raw_path / "bigg_models_reactions.txt", sep="\t")
metabolite_df = pd.read_csv(txt_raw_path / "bigg_models_metabolites.txt", sep="\t")

print(" Done! ✅")

# ════════════════════════════════════════════════════════════════
# 4. Data Preparation for Analysis
# ════════════════════════════════════════════════════════════════


print("Data Preparation for Analysis...")

print(" Preparing FBA Data...")

### FBA DATA ###

#------------------------------------------------------------------------------
# Generate dataframe listing objective values for each model in each condition
#------------------------------------------------------------------------------

for model_name, df in objective_dfs.items():
    df["Objective"] = df["Objective"].round(2)
    df["Model"] = model_name  # Add model name as a column

combined_df = pd.concat(objective_dfs.values(), ignore_index=True)
combined_df = combined_df[["Model", "Combination", "Objective"]]

#------------------------------------------------------------------------------
# Mean and Variance of Active Reactions per Model (FBA)
#------------------------------------------------------------------------------

model_stats = {}

for model_name, condition_dict in flux_distributions.items():
    active_counts = []
    
    for condition, flux_dict in condition_dict.items():
        active = sum(abs(v) > 1e-6 for v in flux_dict.values())
        active_counts.append(active)
    
    mean_active = np.mean(active_counts)
    var_active = np.var(active_counts)
    
    model_stats[model_name] = {
        "mean_active": mean_active,
        "var_active": var_active,
        "n_conditions": len(active_counts)
    }

#------------------------------------------------------------------------------
# Data Preparation for Reaction Activity Analysis
#------------------------------------------------------------------------------

# Extract reactions and respective rounded flux values per model
all_flux_data = []

for model_name, cond_dict in flux_distributions.items():
    for condition, flux_dict in cond_dict.items():
        for reaction, flux in flux_dict.items():
            all_flux_data.append({
                "model": model_name,
                "reaction": reaction,
                "flux": round(flux, 1)
            })

# Create initial DataFrame
df = pd.DataFrame(all_flux_data)

# Group by model and reaction, collect flux values into a list
grouped_df = (
    df.groupby(["model", "reaction"])["flux"]
    .apply(list)
    .reset_index()
)

# Add missing reactions from the model with flux = 0
for model_name, model in models.items():
    all_model_reactions = {rxn.id for rxn in model.reactions}
    existing_reactions = set(grouped_df[grouped_df["model"] == model_name]["reaction"])
    missing_reactions = all_model_reactions - existing_reactions

    for reaction in missing_reactions:
        grouped_df = pd.concat([
            grouped_df,
            pd.DataFrame([{
                "model": model_name,
                "reaction": reaction,
                "flux": [0.0]
            }])
        ], ignore_index=True)

# Assign a category column based on flux value types
def categorize_flux(flux_list):
    signs = set()
    for v in flux_list:
        if v > 0:
            signs.add("pos")
        elif v < 0:
            signs.add("neg")
        else:
            signs.add("zero")

    if signs == {"zero"}:
        return 0
    elif signs == {"pos"}:
        return 1
    elif signs == {"neg"}:
        return 2
    elif signs == {"pos", "neg"}:
        return 3
    elif signs == {"zero", "pos"}:
        return 4
    elif signs == {"zero", "neg"}:
        return 5
    elif signs == {"zero", "pos", "neg"}:
        return 6

grouped_df["category"] = grouped_df["flux"].apply(categorize_flux)

print(" Preparing FVA Data...")

### FVA DATA ###

#------------------------------------------------------------------------------
# Clean FVA raw data and collect information on variable reactions 
#------------------------------------------------------------------------------

# Round all fva results to 1 decimal (avoid numerical noise)
for key, df in fva_dfs.items():
    if "minimum" in df.columns:
        df["minimum"] = pd.to_numeric(df["minimum"], errors="coerce").round(1)
    if "maximum" in df.columns:
        df["maximum"] = pd.to_numeric(df["maximum"], errors="coerce").round(1)

fva_dfs_ao = {}

# Generate "ungrouped" dfs which contain information on the nature of each variable reaction

for key, df in fva_dfs.items():
    if "minimum" in df.columns and "maximum" in df.columns:
        # Filter for reactions with alternate optima
        filtered_df = df[df["minimum"] != df["maximum"]].copy()

        # Define sign function
        def get_sign(val):
            if val > 0:
                return "+"
            elif val < 0:
                return "-"
            else:
                return "0"

        # Create the "Sign" column
        filtered_df["Sign"] = list(zip(
            filtered_df["minimum"].apply(get_sign),
            filtered_df["maximum"].apply(get_sign)
        ))

        fva_dfs_ao[key] = filtered_df

# Generate Grouped dfs which contain information on the nature of each variable reaction

fva_reaction_variability = []

for key, df in fva_dfs_ao.items():
    reaction_info = {}

    for idx, row in df.iterrows():
        rxn = row["index"]
        comb = row["Combination"]
        min_val = row["minimum"]
        max_val = row["maximum"]

        if rxn not in reaction_info:
            reaction_info[rxn] = {
                "Combinations": [],
                "Ranges": [],
                "Signs": []
            }

        reaction_info[rxn]["Combinations"].append(comb)
        reaction_info[rxn]["Ranges"].append((min_val, max_val))

        if min_val > 0:
            min_sign = "+"
        elif min_val < 0:
            min_sign = "-"
        else:
            min_sign = "0"

        if max_val > 0:
            max_sign = "+"
        elif max_val < 0:
            max_sign = "-"
        else:
            max_sign = "0"

        reaction_info[rxn]["Signs"].append((min_sign, max_sign))

    records = []
    for rxn, data in reaction_info.items():
        combinations = data["Combinations"]
        ranges = data["Ranges"]
        signs = data["Signs"]
        n_combinations = len(combinations)
        combination_variability = len(set(ranges)) > 1
        same_signs = len(set(signs)) == 1

        records.append({
            "Reaction": rxn,
            "Combinations": combinations,
            "n_Combinations": n_combinations,
            "Combination_variability": combination_variability,
            "Ranges": ranges,
            "Same_signs": same_signs,
            "Signs": signs
        })

    summary_df = pd.DataFrame(records)
    fva_reaction_variability.append(summary_df)

fva_reaction_variability_dict = {
    key: summary_df for key, summary_df in zip(fva_dfs_ao.keys(), fva_reaction_variability)
}

print(" Preparing BIGG Analysis...")

### FBA & FVA DATA ###

#------------------------------------------------------------------------------
# BIGG Data Matching 
#------------------------------------------------------------------------------

# === Run Matching for All Types ===
BIGG_results = {
    "all": {},
    "fba": {},
    "fva": {},
}

models_to_check = ["iMS520", "iKS1119", "iJO1366"]

# 1. All reactions in GEMs
for model_name in models_to_check:
    model = models[model_name]
    all_rxns = [rxn.id for rxn in model.reactions]
    BIGG_results["all"][model_name] = match_reactions_to_bigg(all_rxns, reaction_df)

# 2. FBA-active reactions
for model_name in models_to_check:
    active_rxns = set()
    for flux_dict in flux_distributions[model_name].values():
        active_rxns.update(rxn for rxn, val in flux_dict.items() if abs(val) > 1e-6)
    BIGG_results["fba"][model_name] = match_reactions_to_bigg(active_rxns, reaction_df)

# 3. Variable reactions from FVA
for model_name in models_to_check:
    fva_df = fva_reaction_variability_dict.get(model_name)
    if fva_df is None:
        continue
    variable_rxns = fva_df["Reaction"].unique()
    BIGG_results["fva"][model_name] = match_reactions_to_bigg(variable_rxns, reaction_df)

print(" Done! ✅")

# ════════════════════════════════════════════════════════════════
# FBA Analysis
# ════════════════════════════════════════════════════════════════


print("Starting FBA Analysis...")

#------------------------------------------------------------------------------
# Fluxes between models and conditions: Basic Statistics and Consrvation
#------------------------------------------------------------------------------

txt_file = txt_path / "COM_OFV_FBA_Flux_Conservation_Summary.txt"

with open(txt_file, "w") as f:
    
    # Basic Model Stats
    for model, stats in model_stats.items():
        print(
            f"🔍 {model}: Mean active = {stats['mean_active']:.1f}, "
            f"Variance = {stats['var_active']:.1f}, "
            f"Conditions = {stats['n_conditions']}",
            file=f
        )

    # Conservation of reactions active in FBA solutions per model across carbon sources
    model_conservation_scores = {}

    for model_name, condition_dict in flux_distributions.items():
        all_sets = [
            set(rxn for rxn, v in fluxes.items() if abs(v) > 1e-6)
            for fluxes in condition_dict.values()
        ]
        if not all_sets:
            continue
        union = set.union(*all_sets)
        intersection = set.intersection(*all_sets)

        conservation = len(intersection) / len(union) if union else 0
        model_conservation_scores[model_name] = conservation

    print("\n🧬 Conservation Score (Intersection / Union of Active Reactions):", file=f)
    for model, score in model_conservation_scores.items():
        print(f"{model}: {score:.2%}", file=f)

    # Conservation of reactions active in FBA solutions per carbon source across models
    models_to_compare = ["iJO1366", "iKS1119", "iMS520"]
    intersection_per_condition = []

    if all(model in flux_distributions for model in models_to_compare):
        conditions = list(flux_distributions[models_to_compare[0]].keys())[:6]

        for cond in conditions:
            sets = []

            for model in models_to_compare:
                flux_dict = flux_distributions[model].get(cond, {})
                active_rxns = {rxn for rxn, val in flux_dict.items() if abs(val) > 1e-6}
                sets.append(active_rxns)

            # Compute intersection for this condition
            common_reactions = set.intersection(*sets)
            intersection_per_condition.append(common_reactions)

            print(
                f"🔗 Condition {cond[0]} — Intersection of Active Reactions "
                f"({len(common_reactions)} total):",
                file=f
            )
            print("", file=f)

        final_common_reactions = set.intersection(*intersection_per_condition)

        print("🧬 Final Intersection Across All Conditions:", file=f)
        print(f"• Number of reactions: {len(final_common_reactions)}", file=f)

        percent = (
            100 * len(final_common_reactions) / len(set.union(*intersection_per_condition))
            if intersection_per_condition else 0
        )
        print(f"• Percent (of union of all condition intersections): {percent:.2f}%", file=f)

# Use a categorical palette
palette = sns.color_palette("rocket", 10)
custom_palette = [palette[i] for i in [6, 3, 0]]

models_to_plot = ['iMS520', 'iKS1119', 'iJO1366']
model_label_map = {
    'iMS520': 'B. longum',
    'iKS1119': 'B. theta',
    'iJO1366': 'E. coli'
}
model_color_map = dict(zip(models_to_plot, custom_palette))

if all(model in flux_distributions for model in models_to_compare):

    # Get the first 6 conditions
    conditions = list(flux_distributions[models_to_compare[0]].keys())[:6]

    # Split into two chunks of 3
    for group_idx in range(2):
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # 3 plots per row
        condition_chunk = conditions[group_idx * 3: (group_idx + 1) * 3]

        for idx, cond in enumerate(condition_chunk):
            sets = []
            labels = []

            for model in models_to_compare:
                flux_dict = flux_distributions[model].get(cond, {})
                active_rxns = {rxn for rxn, val in flux_dict.items() if abs(val) > 1e-6}
                sets.append(active_rxns)
                labels.append(model_label_map[model])

            ax = axes[idx]
            v = venn3(sets, set_labels=labels, ax=ax)

            # Color patches
            patch_ids = ['100', '010', '001']
            for patch_id, model in zip(patch_ids, models_to_compare):
                patch = v.get_patch_by_id(patch_id)
                if patch:
                    patch.set_color(model_color_map[model])
                    patch.set_alpha(0.8)

            # Label styles
            if v.set_labels:
                for label in v.set_labels:
                    if label:
                        label.set_fontsize(8)
                        #label.set_fontweight('bold')
            if v.subset_labels:
                for label in v.subset_labels:
                    if label:
                        label.set_fontsize(10)

            ax.set_title(f"{cond[0]}", fontsize=12)

        # Save uniquely for each panel group
        fig.savefig(output_path / f"COM_OFV_FBA_Flux_Venn_panel_{group_idx + 1}.png", dpi=300, bbox_inches='tight')
        fig.savefig(output_path / f"COM_OFV_FBA_Flux_Venn_panel_{group_idx + 1}.svg", bbox_inches='tight')
        plt.close()

#------------------------------------------------------------------------------------------
# Fuzzy: Conservation of reactions active in FBA solutions per carbon source across models
#------------------------------------------------------------------------------------------

models_to_compare = ["iJO1366", "iKS1119", "iMS520"]
similarity_thresholds = list(range(100, 69, -6))
results = []

def fuzzy_intersection(sets, threshold):
    """Compute fuzzy intersection across multiple sets."""
    base_set = sets[0]
    for other_set in sets[1:]:
        matched = set()
        for item in base_set:
            # Check if there's a similar string in the other set above threshold
            best_match, score, _ = process.extractOne(item, other_set, scorer=fuzz.ratio)
            if score >= threshold:
                matched.add(item)
        base_set = matched
    return base_set

if all(model in flux_distributions for model in models_to_compare):
    conditions = list(flux_distributions[models_to_compare[0]].keys())[:6]

    for thresh in similarity_thresholds:
        intersection_per_condition = []

        for cond in conditions:
            sets = []
            for model in models_to_compare:
                flux_dict = flux_distributions[model].get(cond, {})
                active_rxns = {rxn for rxn, val in flux_dict.items() if abs(val) > 1e-6}
                sets.append(active_rxns)

            common_reactions = fuzzy_intersection(sets, threshold=thresh)
            intersection_per_condition.append(common_reactions)

        final_common = set.intersection(*intersection_per_condition)
        union_all = set.union(*[set.union(*sets) for sets in [intersection_per_condition]])

        percent = 100 * len(final_common) / len(union_all) if union_all else 0

        results.append({
            "Threshold (%)": thresh,
            "Final Intersection Size": len(final_common),
            "Union Size": len(union_all),
            "Percent Conserved": round(percent, 2)
        })

# Convert to DataFrame
df_fuzzy_results = pd.DataFrame(results)

#------------------------------------------------------------------------------------------
# Correlation of Model Size, Number of Active Reactions and Variable Reactions across GEMs
#------------------------------------------------------------------------------------------

output_file = txt_path / "COM_OFV_cor_size_react_var.txt"

with open(output_file, "w") as f:
    for model_name, model in models.items():

        total_reactions = len(model.reactions)

        # Get FBA solutions
        fba = flux_distributions.get(model_name, {})
        active_counts = [sum(abs(v) > 1e-6 for v in fluxes.values()) for fluxes in fba.values()]
        avg_active = np.mean(active_counts) if active_counts else 0

        # Variable reactions
        variable_rxns = len(fva_reaction_variability_dict.get(model_name, []))

        # Compute percentages
        pct_active_total = 100 * avg_active / total_reactions if total_reactions else 0
        pct_variable_total = 100 * variable_rxns / total_reactions if total_reactions else 0
        pct_variable_active = 100 * variable_rxns / avg_active if avg_active else 0

        print(f"Model: {model_name}", file=f)
        print(f"  Total reactions: {total_reactions}", file=f)
        print(f"  Avg. active in FBA: {avg_active:.1f} ({pct_active_total:.1f}%)", file=f)
        print(f"  Variable reactions (FVA): {variable_rxns} ({pct_variable_total:.1f}%)", file=f)
        print(f"  Variable reactions as % of FBA-active: {pct_variable_active:.1f}%", file=f)
        print("-" * 50, file=f)

# Match Seaborn style
sns.set(style="white", font_scale=1.1, rc={"axes.edgecolor": "black"})

model_names = []
model_sizes = []
avg_fba_active = []
variable_reactions = []

for model_name, model in models_list.items():
    if model_name == "e_coli_core":
        continue  # Skip e_coli_core

    # 1. Model size
    model_size = len(model.reactions)

    # 2. Average FBA-active reactions across conditions
    fba = flux_distributions.get(model_name, {})
    active_counts = [sum(abs(v) > 1e-6 for v in fluxes.values()) for fluxes in fba.values()]
    avg_active = np.mean(active_counts) if active_counts else 0

    # 3. Unique variable reactions
    variable_rxns = len(fva_reaction_variability_dict.get(model_name, []))

    # Collect
    model_names.append(model_name)
    model_sizes.append(model_size)
    avg_fba_active.append(avg_active)
    variable_reactions.append(variable_rxns)


# Plotting
x = np.arange(len(model_names))
width = 0.25

# Custom blue-grey shades
colors = ['#000000', '#7f7f7f', '#d3d3d3']

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width, model_sizes, width, label='Model size', color=colors[0])
ax.bar(x, avg_fba_active, width, label='Avg. FBA-active', color=colors[1])
ax.bar(x + width, variable_reactions, width, label='Variable (FVA)', color=colors[2])

# Labels & Styling
ax.set_ylabel('Reaction Count', fontsize=10)
ax.set_title('', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='y', labelsize=10, width=1, length=4, direction='out')
ax.tick_params(axis='x', width=1, length=4, direction='out')

# Legend: external right-aligned, frameless
legend = ax.legend(
    loc='center left',
    bbox_to_anchor=(0.82, 0.9),
    frameon=False,
    borderaxespad=0.5,
    labelspacing=0.4
)
legend.set_title(None)
plt.setp(legend.get_texts(), fontsize=10)

# Styling the plot frame (thinner, cleaner)
ax = plt.gca()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(0.8)
ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)

# Final layout
plt.tight_layout()
fig.savefig(output_path / "COM_OFV_cor_size_react_var_Barplot.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "COM_OFV_cor_size_react_var_Barplot.svg", bbox_inches='tight')
plt.close()

print(" Done! ✅")

# ════════════════════════════════════════════════════════════════
# FVA data analysis
# ════════════════════════════════════════════════════════════════

print("Starting FVA Analysis...")

#------------------------------------------------------------------
# Distribution of Sign Patters
#------------------------------------------------------------------

output_file = txt_path / "COM_OFV_FVA_sign_patterns.txt"

with open(output_file, "w") as f:
    for model_name, df in fva_reaction_variability_dict.items():
        if model_name == "e_coli_core":
            continue  # Skip this one

        print(f"\n================ {model_name} ================\n", file=f)

        total_reactions = len(df)

        # 1. n_Combinations distribution
        n_comb_counts = df["n_Combinations"].value_counts().sort_index()
        print("1) n_Combinations Distribution (non-zero only):", file=f)
        for k, v in n_comb_counts.items():
            pct = 100 * v / total_reactions
            print(f"   {k} combinations: {v} reactions ({pct:.1f}%)", file=f)

        # 2. Combination_variability True/False
        combo_counts = df["Combination_variability"].value_counts()
        print("\n2) Combination Variability (True/False):", file=f)
        for k, v in combo_counts.items():
            pct = 100 * v / total_reactions
            print(f"   {k}: {v} reactions ({pct:.1f}%)", file=f)

        # 3. Same_signs True/False within variable reactions only
        subset = df[df["Combination_variability"] == True]
        total_subset = len(subset)
        sign_counts = subset["Same_signs"].value_counts()
        print("\n3) Same Signs (within reactions with combination variability):", file=f)
        for k, v in sign_counts.items():
            pct = 100 * v / total_subset if total_subset else 0
            print(f"   {k}: {v} reactions ({pct:.1f}%)", file=f)

        # 4. Unique sign patterns for rows where signs differ
        sign_subset = subset[subset["Same_signs"] == False]
        sign_sets = [frozenset(signs) for signs in sign_subset["Signs"]]
        sign_set_counts = Counter(sign_sets)
        top_sign_sets = sign_set_counts.most_common(10)

        print("\n4) Unique Sign Sets (where signs differ):", file=f)
        for sign_set, count in top_sign_sets:
            label = str(list(sign_set))
            pct = 100 * count / total_subset if total_subset else 0
            print(f"   {label}: {count} reactions ({pct:.1f}%)", file=f)

#------------------------------------------------------------------
# FVA Plots
#------------------------------------------------------------------

### SIGN PATTERNS ###

sns.set(style="white", font_scale=1.1)

# Custom color palette
palette = sns.color_palette("rocket", 10)
custom_palette = [palette[i] for i in [6, 3, 0]]
models_to_plot = ['iMS520', 'iKS1119', 'iJO1366']
model_label_map = {
    'iMS520': 'B. longum',
    'iKS1119': 'B. thetaiotaomicron',
    'iJO1366': 'E. coli'
}

# Process sign set data
all_sign_sets = []

for model_name, df in fva_reaction_variability_dict.items():
    if model_name not in models_to_plot:
        continue

    total = len(df)
    sign_sets = [frozenset(s) for s in df["Signs"]]
    sign_set_counts = Counter(sign_sets)

    for sign_set, count in sign_set_counts.items():
        percent = (count / total) * 100
        all_sign_sets.append({
            "Model": model_label_map[model_name],
            "Sign_Set": str(sorted(sign_set)),  # readable & consistent
            "Percentage": percent
        })

# Create DataFrame
sign_sets_df = pd.DataFrame(all_sign_sets)

# Sort by total percentage across models
total_percent = sign_sets_df.groupby("Sign_Set")["Percentage"].sum()
sorted_sign_sets = total_percent.sort_values(ascending=False).index.tolist()
sign_sets_df["Sign_Set"] = pd.Categorical(sign_sets_df["Sign_Set"], categories=sorted_sign_sets, ordered=True)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=sign_sets_df,
    x="Sign_Set",
    y="Percentage",
    hue="Model",
    palette=custom_palette,
    ax=ax
)

# Axis and tick styling
ax.set_xlabel("", fontsize=10)
ax.set_ylabel("Percentage of Reactions", fontsize=10)


# Define shorthand labels
sign_map = {
    '-': 'neg.',
    '0': '0',
    '+': 'pos.'
}

# Extract original tick labels
raw_labels = [label.get_text() for label in ax.get_xticklabels()]

# Convert each label
formatted_labels = []
for label in raw_labels:
    try:
        # Convert string representation of list of tuples into actual Python object
        tuple_list = eval(label)

        # Ensure it's a list
        if not isinstance(tuple_list, list):
            tuple_list = [tuple_list]

        # Build readable label
        converted = [f"{sign_map.get(a, a)} / {sign_map.get(b, b)}" for a, b in tuple_list]
        formatted_labels.append("\n".join(converted))

    except Exception:
        # Fallback if parsing fails
        formatted_labels.append(label)

# Apply formatted labels
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(formatted_labels, ha="center", fontsize=10)
ax.tick_params(axis='y', labelsize=9)

# Styling the plot frame
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(0.8)
ax.tick_params(width=0.8, length=4, direction='out')


# Legend styling (clean and styled box)
legend = ax.legend(
    loc='center left',
    bbox_to_anchor=(0.8, 0.95),
    frameon=True
)
legend.set_title(None)
plt.setp(legend.get_texts(), fontsize=9)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_boxstyle('round', pad=0.4)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(1.0)

# Final layout
plt.tight_layout()
fig.savefig(output_path / "COM_OFV_FVA_signpatterns_barplot.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "COM_OFV_FVA_signpatterns_barplot.svg", bbox_inches='tight')
plt.close()

### VARIABILITY DISTRIBUTION OVER COND ###

# Style
sns.set(style="ticks", font_scale=1.1)

# Custom color palette
palette = sns.color_palette("rocket", 10)
custom_palette = [palette[i] for i in [6, 3, 0]]

# Models and pretty labels
models_to_plot = ['iMS520', 'iKS1119', 'iJO1366']
model_label_map = {
    'iMS520': 'B. longum',
    'iKS1119': 'B. thetaiotaomicron',
    'iJO1366': 'E. coli'
}

# Plot setup
bins = np.arange(1, 65)
fig = plt.figure(figsize=(10, 5)) 

# Plot each model
for model_name, color in zip(models_to_plot, custom_palette):
    df = fva_reaction_variability_dict.get(model_name)
    if df is None:
        continue

    values = df["n_Combinations"]
    counts, _ = np.histogram(values, bins=bins)
    percentages = counts / counts.sum() * 100
    x_vals = bins[:-1]

    # Restrict data to x range 1–6
    mask_range = (x_vals >= 1) & (x_vals <= 6)
    x_vals_filtered = x_vals[mask_range]
    percentages_filtered = percentages[mask_range]

    # Plot line and points
    plt.plot(x_vals_filtered, percentages_filtered, color=color, linewidth=2)
    mask = percentages_filtered > 0
    plt.scatter(x_vals_filtered[mask], percentages_filtered[mask], s=50, color=color)

# Axes and styling
plt.xlabel("Carbon Source Combinations in which Reaction is Variable", fontsize=10)
plt.ylabel("Percentage of Variable Reactions", fontsize=10)
plt.xticks(ticks=range(1, 7), fontsize=10)
plt.xlim(0.5, 6.5)
plt.yticks(fontsize=10)
plt.title("")

# Styling the plot frame (thinner, cleaner)
ax = plt.gca()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(0.8)
ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)

plt.tight_layout()
fig.savefig(output_path / "COM_OFV_FVA_vardist_histogram.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "COM_OFV_FVA_vardist_histogram.svg", bbox_inches='tight')
plt.close()

### TRUE FALSE VAR STATS ###

# Plot style (Seaborn base + font/tick clean-up)
sns.set(style="white", font_scale=1.1, rc={"axes.edgecolor": "black"})

custom_titles = {
    "Combination_variability": "Consistency of Variation Across Carbon Sources",
    "Same_signs": "Consistency of Flux Category Across Carbon Sources"
}

# Prepare tidy dataframe
plot_data = []

for model_name, df in fva_reaction_variability_dict.items():
    if model_name == "e_coli_core":
        continue

    total_cv = len(df)
    cv_counts = df["Combination_variability"].value_counts()
    for val, count in cv_counts.items():
        pct = (count / total_cv) * 100
        plot_data.append({
            "Model": model_name,
            "Metric": "Combination_variability",
            "Value": str(val),
            "Percentage": pct
        })

    subset = df[df["Combination_variability"] == True]
    total_ss = len(subset)
    if total_ss > 0:
        ss_counts = subset["Same_signs"].value_counts()
        for val, count in ss_counts.items():
            pct = (count / total_ss) * 100
            plot_data.append({
                "Model": model_name,
                "Metric": "Same_signs",
                "Value": str(val),
                "Percentage": pct
            })

plot_df = pd.DataFrame(plot_data)
avg_lines = plot_df[plot_df["Value"] == "True"].groupby("Metric")["Percentage"].mean().to_dict()

# Custom colors: black and light grey
custom_palette = {
    "True": "#000000",
    "False": "#d3d3d3"
}

# Create the plot
g = sns.catplot(
    data=plot_df,
    kind="bar",
    x="Model",
    y="Percentage",
    hue="Value",
    col="Metric",
    palette=custom_palette,
    height=5,
    aspect=1.3,
    sharey=True
)

# Set overall figure size
g.fig.set_size_inches(10, 5)

# Remove default titles and set Y label
g.set_titles("")
g.set_axis_labels("", "Percentage of Reactions")


# Tick formatting, axis cleanup, and frame styling
for ax in g.axes.flat:
    # Tick settings
    ax.tick_params(axis='both', labelsize=10, direction='out', length=4, width=0.8)

    # Axis labels
    ax.set_xlabel(ax.get_xlabel(), fontsize=10)
    ax.set_ylabel(ax.get_ylabel(), fontsize=10)

    # Frame (spines)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)

# Adjust X-tick labels
g.set_xticklabels(rotation=45, ha='right')

# Legend styling: clean and external
g._legend.set_title(None)
plt.setp(g._legend.get_texts(), fontsize=10)
g._legend.set_bbox_to_anchor((1.08, 0.85))
g._legend.set_frame_on(False)

# Add average lines and labels + custom subplot titles
for i, (ax, metric) in enumerate(zip(g.axes.flat, avg_lines)):

    # Subplot title
    ax.set_title(custom_titles.get(metric, ""), fontsize=10)

# Final layout
plt.tight_layout()
g.fig.savefig(output_path / "COM_OFV_FVA_vardist_barplot.png", dpi=300, bbox_inches='tight')
g.fig.savefig(output_path / "COM_OFV_FVA_vardist_barplot.svg", bbox_inches='tight')
plt.close()

print(" Done! ✅")

# ════════════════════════════════════════════════════════════════
# Reaction Activity Analysis
# ════════════════════════════════════════════════════════════════

print("Starting Reaction Activity Analysis...")

# --------------------------
# Label map and model order
# --------------------------
label_map = {
    'e_coli_core': 'E. coli Core',
    'iJO1366': 'E. coli',
    'iMS520': 'B. longum',
    'iKS1119': 'B. thetaiotaomicron'
}
model_order = ['e_coli_core', 'iMS520', 'iKS1119', 'iJO1366']  # match color order

# --------------------------
# Rebuild relative frequency matrix (safe for re-runs)
# --------------------------
category_counts = grouped_df.groupby(["model", "category"]).size().unstack(fill_value=0)
relative_freq = category_counts.div(category_counts.sum(axis=1), axis=0) * 100
relative_freq_T = relative_freq.T

# Only reorder and relabel if raw model names are still present
if all(model in relative_freq_T.columns for model in model_order):
    relative_freq_T = relative_freq_T[model_order]
    relative_freq_T.columns = [label_map[model] for model in relative_freq_T.columns]

# --------------------------
# Plotting
# --------------------------
# Set seaborn style
sns.set_style("white")

# Custom rocket palette
palette = sns.color_palette("rocket", 10)
custom_palette = [palette[i] for i in [9, 6, 3, 0]]

# Create plot
fig, ax = plt.subplots(figsize=(8, 4))

# Gridlines and percent labels
y_vals = [80, 50, 10, 1, 0.1]
offset = 0.04

for y_val in y_vals:
    ax.axhline(y=y_val, color='black', linestyle=':', linewidth=0.8, zorder=1)
    label_y = y_val * (1 + offset)
    ax.text(len(relative_freq_T.index) - 0.4, label_y, f'{y_val}%',
            va='bottom', ha='right', fontsize=8, color='black', zorder=3)

# Plot bar chart
relative_freq_T.plot(
    kind='bar',
    width=0.85,
    color=custom_palette,
    edgecolor='white',
    ax=ax,
    zorder=2
)

# Log scale and ticks
ax.set_yscale('log')
ax.set_yticks(y_vals)
ax.set_yticklabels([f"{y}%" for y in y_vals], fontsize=10)

# Labels
ax.set_xlabel("Activity Code", fontsize=10)
ax.set_ylabel("Percentage of Reactions (%)", fontsize=10)
ax.tick_params(axis='x', labelsize=10, rotation=0)
ax.tick_params(axis='y', labelsize=10)

# Remove top/right spines
sns.despine(ax=ax)

# Legend
legend = ax.legend(
    loc='upper right',
    bbox_to_anchor=(1.1, 1),
    frameon=True
)
legend.set_title(None)
plt.setp(legend.get_texts(), fontsize=9)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_boxstyle('round', pad=0.4)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(1.0)

# Styling the plot frame (thinner, cleaner)
ax = plt.gca()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(0.8)
ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)

# Layout
plt.tight_layout()
fig.savefig(output_path / "COM_OFV_ReactionVariability.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "COM_OFV_ReactionVariability.svg", bbox_inches='tight')
plt.close()

print(" Done! ✅")

# ════════════════════════════════════════════════════════════════
# BIGG data analysis
# ════════════════════════════════════════════════════════════════

print("Starting BIGG Data Analysis...")

#------------------------------------------------------------------
# write data on BIGG mapping into txt file
#------------------------------------------------------------------

txt_file = txt_path / "COM_OFV_BiGG_Matching_Summary.txt"

print_summary_to_file("All Reactions", BIGG_results["all"], txt_file)
print_summary_to_file("FBA-Active Reactions", BIGG_results["fba"], txt_file)
print_summary_to_file("FVA-Variable Reactions", BIGG_results["fva"], txt_file)

#------------------------------------------------------------------
# Process Matching BIGG IDs for var reactions 
#------------------------------------------------------------------

# Initialize dictionaries for storing matches per model
matched_reactions_dict = {}
matched_old_ids_dict = {}
still_not_found_dict = {}

# Iterate over models
for model_name in fva_reaction_variability_dict:
    if model_name == "e_coli_core":
        continue  # Skip core model

    # Step 1: Get FVA reaction IDs
    fva_reactions = fva_reaction_variability_dict[model_name]["Reaction"].unique()

    # Step 2: Match in 'bigg_id'
    found_mask = reaction_df["bigg_id"].isin(fva_reactions)
    matched_df = reaction_df[found_mask].copy()
    matched_ids = set(matched_df["bigg_id"])
    not_yet_matched = [rxn for rxn in fva_reactions if rxn not in matched_ids]

    # Step 3: Match in 'old_bigg_ids'
    matches_in_old = []
    for idx, row in reaction_df.iterrows():
        old_ids = row["old_bigg_ids"]
        if isinstance(old_ids, list):
            if any(rxn in old_ids for rxn in not_yet_matched):
                matches_in_old.append(idx)
    matched_old_df = reaction_df.loc[matches_in_old].copy()

    # Step 4: Identify still missing reactions
    found_in_old_ids = set()
    for old_list in matched_old_df["old_bigg_ids"]:
        for rxn in old_list:
            if rxn in not_yet_matched:
                found_in_old_ids.add(rxn)
    still_missing = [rxn for rxn in not_yet_matched if rxn not in found_in_old_ids]

    # Store results
    matched_reactions_dict[model_name] = matched_df
    matched_old_ids_dict[model_name] = matched_old_df
    still_not_found_dict[model_name] = still_missing

#------------------------------------------------------------------
# Checking "rare reactions" - i.e. not shared with other BIGG models
#------------------------------------------------------------------

output_file = txt_path / "COM_OFV_BiGG_rare_reactions_summary.txt"

with open(output_file, "w") as f:
    print("\n=== Reactions in <10 models ===\n", file=f)

    for model_name, df in matched_reactions_dict.items():
        # Calculate number of models for each reaction
        df["model_count"] = df["model_list"].apply(
            lambda x: len(str(x).split(";")) if pd.notnull(x) else 0
        )

        # Filter reactions that appear in fewer than 10 models
        rare_reactions = df[df["model_count"] < 10]

        if not rare_reactions.empty:
            print(f"🔎 {model_name}: {len(rare_reactions)} reactions in <10 models", file=f)
            for _, row in rare_reactions.iterrows():
                print(f" - {row['bigg_id']}: {row['name']}", file=f)
            print(file=f)
        else:
            print(f"✅ {model_name}: All matched reactions appear in ≥10 models\n", file=f)

#------------------------------------------------------------------
# Checking/ Plotting how variable reactions overlap between the 3 GEMs
#------------------------------------------------------------------

# Define color palette and labels
palette = sns.color_palette("rocket", 10)
custom_palette = [palette[i] for i in [6, 3, 0]]

models_to_plot = ['iMS520', 'iKS1119', 'iJO1366']
model_label_map = {
    'iMS520': 'B. longum',
    'iKS1119': 'B. thetaiotaomicron',
    'iJO1366': 'E. coli'
}

# Get model sets and pretty labels
# Extract 'bigg_id' columns as sets from each model DataFrame
model_bigg_ids = {
    model: set(matched_reactions_dict[model]['bigg_id'].dropna())
    for model in matched_reactions_dict
}

# Create list of sets for the models you want to plot
sets = [model_bigg_ids[model] for model in models_to_plot]
labels = [model_label_map[model] for model in models_to_plot]

# Plot Venn diagram
fig = plt.figure(figsize=(10, 5))
venn = venn3(subsets=sets, set_labels=labels)

# Apply custom colors to each circle
for patch, color in zip(venn.patches[:3], custom_palette):
    if patch:
        patch.set_facecolor(color)
        patch.set_linewidth(0.8)
        patch.set_alpha(0.8)

# Style title and layout
plt.title("", fontsize=12)
plt.tight_layout()
fig.savefig(output_path / "COM_OFV_FVA_BIGG_VENN.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "COM_OFV_FVA_BIGG_VENN.svg", bbox_inches='tight')
plt.close()

#------------------------------------------------------------------
# Checking/ Plotting distribution and share with other models
#------------------------------------------------------------------

# Style
sns.set(style="white", font_scale=1.1)

# Color palette and labels
palette = sns.color_palette("rocket", 10)
custom_palette = [palette[i] for i in [6, 3, 0]]
models_to_plot = ['iMS520', 'iKS1119', 'iJO1366']
model_label_map = {
    'iMS520': 'B. longum',
    'iKS1119': 'B. thetaiotaomicron',
    'iJO1366': 'E. coli'
}

# Prepare figure with 3 horizontal subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Set bin size
bin_width = 10
max_count = max(
    df["model_list"].apply(lambda x: len(str(x).split(";")) if pd.notnull(x) else 0).max()
    for df in matched_reactions_dict.values()
)
bins = np.arange(0, max_count + bin_width, bin_width)

# Plot each model's normalized histogram
for ax, model_name, color in zip(axs, models_to_plot, custom_palette):
    df = matched_reactions_dict[model_name]
    model_counts = df["model_list"].apply(
        lambda x: len(str(x).split(";")) if pd.notnull(x) else 0
    )

    # Histogram
    counts, bin_edges = np.histogram(model_counts, bins=bins)
    percentages = counts / counts.sum() * 100
    bin_centers = bin_edges[:-1] + bin_width / 2

    ax.bar(bin_centers, percentages, width=bin_width * 0.9, edgecolor="black", linewidth=0.8,color=color, alpha=0.8)
    ax.set_title(model_label_map[model_name], fontsize=10)
    ax.set_xlabel("Number of Models", fontsize=10)
    ax.set_xlim([0, max_count + bin_width])
    ax.set_xticks(bins)
    ax.set_ylim(0, 40)
    ax.tick_params(width=0.8, length=4, direction='out', labelsize=10)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)

# Shared y-axis label
axs[0].set_ylabel("Percentage of Reactions", fontsize=10)

# Final layout
plt.tight_layout()
fig.savefig(output_path / "COM_OFV_FVA_BIGG_dist.png", dpi=300, bbox_inches='tight')
fig.savefig(output_path / "COM_OFV_FVA_BIGG_dist.svg", bbox_inches='tight')
plt.close()

print(" Done! ✅")

# ════════════════════════════════════════════════════════════════
# Data Export
# ════════════════════════════════════════════════════════════════

print("Data Export...")

combined_df.to_csv(processed_csv_path / "Community_OFV_Analysis_Objectives.csv", index=False)
df_fuzzy_results.to_csv(processed_csv_path / "Community_OFV_Fuzzy_FBA_Fluxes_Conservation.csv", index=False)

print(" Done! ✅")
print('Everything complete! 🎉')