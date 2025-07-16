# ---------------------------------------
# Imports and Path Setup
# ---------------------------------------

import pandas as pd # type: ignore
import pickle
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from pathlib import Path
from cobra.io import read_sbml_model # type: ignore

# ---------------------------------------
# Paths
# ---------------------------------------

# Add Repo Root 
project_root = Path(__file__).resolve().parents[3] 

# processed data
processed_path = project_root / "data" / "processed"
figures_path = processed_path / "figures"

# raw data
raw_path = project_root / "data" / "raw" 
raw_sbml_path = raw_path / "sbml_files"
temp_sbml_path = raw_sbml_path / "temporary_files"
raw_csv_path = raw_path / "csv_files"
raw_pkl_path = raw_path / "pkl_files"

# ---------------------------------------
# Model Setup
# ---------------------------------------

print("Importing Data ...")

model_names = ['e_coli_core', 'iJO1366', 'iMS520', 'iKS1119']
models = {name: read_sbml_model(temp_sbml_path / f"{name}_neutral.xml") for name in model_names}

reaction_lists = {
    'e_coli_core': ['EX_fru_e', 'EX_glc__D_e'],
    'iJO1366': ['EX_fru_e', 'EX_gal_e', 'EX_glc__D_e', 'EX_sucr_e', 'EX_malt_e', 'EX_lcts_e'],
    'iKS1119': ['EX_fru_e', 'EX_gal_e', 'EX_glc_e', 'EX_sucr_e', 'EX_lcts_e', 'EX_malt_e'],
    'iMS520': ['EX_fru(e)', 'EX_gal(e)', 'EX_glc-D(e)', 'EX_lcts(e)', 'EX_malt(e)', 'EX_sucr(e)']
}

# ---------------------------------------
# Load FBA Results
# ---------------------------------------

objective_dfs = {}
flux_distributions = {}

for name in model_names:
    obj_path = raw_csv_path / f'FBA_Objectives_{name}.csv'
    flux_path = raw_pkl_path / f'FBA_Distributions_{name}.pkl'

    objective_dfs[name] = pd.read_csv(obj_path)
    with open(flux_path, 'rb') as f:
        flux_distributions[name] = pickle.load(f)

print(" Done! ✅")

# ---------------------------------------
# Convert Flux Distributions to DataFrames
# ---------------------------------------

print("Working on data accessibility ...")

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

# ---------------------------------------
# Subset Flux DataFrames and Add Row Sum
# ---------------------------------------

subset_flux_df_with_sum = {}

for name in model_names:
    df = flux_df_dict[name]
    target_rxns = reaction_lists[name]
    subset = df[target_rxns].fillna(0)
    subset['total_flux'] = subset.sum(axis=1)
    subset_flux_df_with_sum[name] = subset

print(" Done! ✅")

# ---------------------------------------
# Analyze Reaction Activity
# ---------------------------------------

print("Data Analysis ...")

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

# ---------------------------------------
# Plot: Reaction Activity by Model
# ---------------------------------------

print(" Plotting Data analysis...")

sns.set_style("ticks")

# Prepare the DataFrame
comparison_df = pd.DataFrame(normalized_activity_percentages).fillna(0)
comparison_df.index.name = 'Activity'
comparison_df.reset_index(inplace=True)

long_df = comparison_df.melt(id_vars='Activity', var_name='Model', value_name='Percentage')
model_order = ['e_coli_core', 'iMS520', 'iKS1119', 'iJO1366']
long_df['Model'] = pd.Categorical(long_df['Model'], categories=model_order, ordered=True)

# Custom colors and labels
palette = sns.color_palette("rocket", 10)
custom_palette = [palette[i] for i in [9, 6, 3, 0]]
label_map = {
    'e_coli_core': 'E. coli Core',
    'iJO1366': 'E. coli',
    'iMS520': 'B. longum',
    'iKS1119': 'B. thetaiotaomicron'
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
plt.savefig(figures_path / 'Reaction_activity_comparison.png', dpi=300)
print("     Exported: Reaction_activity_comparison.png")
plt.savefig(figures_path / "Reaction_activity_comparison.svg", bbox_inches='tight')
print("     Exported: Reaction_activity_comparison.svg\n")
#plt.show()

# ---------------------------------------
# Plot: Active Reaction Count per Combo
# ---------------------------------------

models_to_plot = ['iJO1366', 'iMS520', 'iKS1119']
plot_data = {m: reaction_usage_stats[m].squeeze() for m in models_to_plot}
df = pd.DataFrame(plot_data).reset_index().rename(columns={'index': 'Combo'})
long_df = df.melt(id_vars='Combo', var_name='Model', value_name='Value')

size_order = sorted(models_to_plot, key=lambda m: reaction_usage_stats[m].shape[0])
long_df['Model'] = pd.Categorical(long_df['Model'], categories=size_order, ordered=True)
long_df['ModelLabel'] = long_df['Model'].map(label_map)

plt.figure(figsize=(10, 4))

colors = [palette[i] for i in [9, 6, 3]]
for i, model in enumerate(size_order):
    data = long_df[long_df['Model'] == model]
    plt.plot(data['Combo'], data['Value'], label=label_map[model], color=colors[i], linewidth=2)

plt.ylim(350, 520)
plt.ylabel("Active Reactions", fontsize=10)
plt.xticks(range(64), range(64), fontsize=8)
plt.xlim(0, 63)
plt.yticks(fontsize=8)

# Legend (styled as in your bar plot)
legend = plt.legend(
    loc='upper left',
    frameon=True,
    edgecolor='black',
    fontsize=9,
    labelspacing=0.4,
    handlelength=1.5,
    handletextpad=0.5,
    borderpad=0.5
)
frame = legend.get_frame()
frame.set_linewidth(0.5)
frame.set_edgecolor('black')
frame.set_facecolor('white')
frame.set_alpha(1.0)

plt.tight_layout()
plt.savefig(figures_path / 'Active_reaction_counts.png', dpi=300)
print("     Exported: Active_reaction_counts.png")
plt.savefig(figures_path / "Active_reaction_counts.svg", bbox_inches='tight')
print("     Exported: Active_reaction_counts.svg\n")
#plt.show()

print(" Done! ✅")

# ---------------------------------------
# Analysis: B. theta unusual reactions
# ---------------------------------------

df = flux_df_dict['iKS1119']
target = "('Maltose', 'Sucrose')"

if target in df.index:
    row = df.loc[target]
    zero_cols = row[row == 0].index
    others = df.drop(index=target)

    result_cols = [
        col for col in zero_cols
        if (others[col] != 0).sum() / len(others) >= 0.95
    ]

    # Get reaction names
    rxn_names = [models['iKS1119'].reactions.get_by_id(r).name for r in result_cols]
    unusual_df = pd.DataFrame({'ID': result_cols, 'Name': rxn_names})
    #unusual_df.to_csv(raw_csv_path / 'btheta_unusual_reactions.csv', index=False)

# ---------------------------------------
# Export: Activity DataFrames
# ---------------------------------------

print("Data Export...")

for model_name, df in reaction_activity_info.items():
    activity_path = raw_csv_path / f'{model_name}_activity.csv'
    df.to_csv(activity_path, index=False)

# ---------------------------------------
# Export: Full Flux Distribution DataFrames
# ---------------------------------------

for model_name, df in flux_df_dict.items():
    flux_path = raw_csv_path / f'{model_name}_fluxes.csv'
    df.to_csv(flux_path)

print(" Done! ✅")

print('Everything complete! 🎉')
