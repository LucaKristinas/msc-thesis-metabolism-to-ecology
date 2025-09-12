
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# Load CSV
csv_path = "/Users/Luca/Desktop/Thesis_Rep/GitHub_Repo/data/processed/csv_files/ecoli_sim_iJO1366_OFVs.csv"  # Update this with your actual file path
df = pd.read_csv(csv_path, index_col=0)

# Threshold
threshold = 1e-5

# Process each condition
for condition, row in df.iterrows():
    active_reactions = [
        col for col in df.columns
        if col.startswith("EX_") and abs(row[col]) > threshold
    ]

    print(f"\n🔬 Condition: {condition}")
    if active_reactions:
        print("  ✅ Active Exchange Reactions:")
        for rxn in active_reactions:
            print(f"    - {rxn} (flux = {row[rxn]:.6f})")
    else:
        print("  ❌ No active exchange reactions.")