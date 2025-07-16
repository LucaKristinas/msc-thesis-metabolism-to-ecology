import numpy as np
import pandas as pd
import cvxpy as cp
import sys
from pathlib import Path

# ==== CONFIGURATION ====

# Add Repo Root to sys.path 
project_root = Path(__file__).resolve().parents[2]  
sys.path.append(str(project_root))

# Import Helper Functions
from src.utils import (is_in_convex_hull, convexly_independent_set)

# processed data
processed_path = project_root / "data" / "processed"
processed_csv_path = processed_path / "csv_files"

# raw data
raw_path = project_root / "data" / "raw" 
raw_csv_path = raw_path / "csv_files"

# Model names
names = ["e_coli_core", "iJO1366", "iKS1119", "iMS520"]

# Define dynamic paths
input_csv_paths = {
    name: raw_csv_path / f"{name}_OFVs.csv"
    for name in names
}

output_independent_set_csvs = {
    name: processed_csv_path / f"{name}_ciOFVs.csv"
    for name in names
}

INDEX_COLUMN = 0  

# ---------------------------------------
# Main Fxn
# ---------------------------------------

print('Starting Convex Independence Check ...')

def main():
    for name in names:
        input_path = input_csv_paths[name]
        output_path = output_independent_set_csvs[name]

        print(f"\nProcessing model: {name}")

        # Load CSV
        print(f"Loading input file: {input_path}")
        df = pd.read_csv(input_path, index_col=INDEX_COLUMN)
        print(f"Loaded DataFrame with shape: {df.shape}")

        # Convert to NumPy array
        vectors = df.to_numpy()

        # Compute convexly independent set
        print("Checking for convex independence...")
        keep_mask = convexly_independent_set(vectors)

        # Save subset
        df_independent = df[keep_mask]
        df_independent.to_csv(output_path)
        print(f"    Convexly independent vectors saved to: {output_path}")
        print(f"    Reduced from {len(df)} to {len(df_independent)} vectors.")

if __name__ == "__main__":
    try:
        main()
        print('\nEverything complete! 🎉')
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
