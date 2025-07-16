# -------------------------------
# Imports
# -------------------------------

import pandas as pd  # type: ignore
import json
from pathlib import Path  # type: ignore

# -------------------------------
# File Paths
# -------------------------------

# Add Repo Root 
project_root = Path(__file__).resolve().parents[2] 

# raw data
raw_path = project_root / "data" / "raw" 
raw_tsv_path = raw_path / "tsv_files"
raw_csv_path = raw_path / "csv_files"
raw_json_path = raw_path / "json_files"

# -------------------------------
# Data
# -------------------------------

print("Loading raw sugar data from files...")

# Load raw sugar data from three organisms
B_theta = pd.read_csv(raw_tsv_path / 'Sugars_B_theta.tsv', sep='\t')
B_longum = pd.read_csv(raw_tsv_path / 'Sugars_B_longum.tsv', sep='\t')
E_coli = pd.read_csv(raw_tsv_path / 'Sugars_E_coli.tsv', sep='\t')

print(" Done! ✅")

# -------------------------------
# Create Raw Sugars Table
# -------------------------------

print("Creating raw sugars table...")

# DataFrame names for reference
df_names = ['B_theta', 'B_longum', 'E_coli']
dfs = [B_theta, B_longum, E_coli]

# Step 1: Extract all unique sugar abbreviations and full names
all_entries = pd.concat([df[['abbreviation', 'fullName']] for df in dfs], ignore_index=True)
all_entries.dropna(subset=['abbreviation'], inplace=True)

carbon_sources_df = all_entries.drop_duplicates(subset='abbreviation').reset_index(drop=True)
carbon_sources_df.rename(columns={'abbreviation': 'Sugar', 'fullName': 'Description'}, inplace=True)

# Step 2: Add boolean presence columns for each organism
for name, df in zip(df_names, dfs):
    present_sugars = set(df['abbreviation'].dropna().unique())
    carbon_sources_df[name] = carbon_sources_df['Sugar'].isin(present_sugars)

# Step 3: Add 'Utilisation' column
carbon_sources_df['Utilisation'] = carbon_sources_df[df_names].sum(axis=1)

carbon_sources_df.sort_values(by='Utilisation', ascending=False, inplace=True)
carbon_sources_df.reset_index(drop=True, inplace=True)

# Export raw sugars table
carbon_sources_df.to_csv(raw_csv_path / 'Raw_Sugars_Table.csv', index=False)
print(" Done! ✅")

# -------------------------------
# Filter & Annotate Sugars for Analysis
# -------------------------------

print("Filtering and annotating sugars for analysis...")

# Criterion: used by at least 2 of the 3 organisms
sugars_df_subset = carbon_sources_df[carbon_sources_df['Utilisation'] > 1]

# Annotate full names manually where needed
corrections = {
    "melib": "Melibiose",
    "arabinogal": "Larch arabinogalactan"
}
sugars_df_subset.loc[:, 'Description'] = sugars_df_subset.apply(
    lambda row: corrections.get(row["Sugar"], row["Description"]), axis=1
)

# Store cleaned descriptions in lowercase
sugar_descriptions = sugars_df_subset['Description'].dropna().str.lower().tolist()

print(" Done! ✅")

# -------------------------------
# Final Sugar Selection
# -------------------------------

print("Selecting final sugars for analysis...")

sugar_list = [
    'd-glucose',
    'alpha-lactose',
    'd-maltose',
    'd-fructose',
    'd-galactose',
    'sucrose'
]

# Export file
with open(raw_json_path / 'sugar_list.json', 'w') as f:
    json.dump(sugar_list, f)

print(f"    Final sugar selection complete: {sugar_list}")

print('Everything complete! 🎉')