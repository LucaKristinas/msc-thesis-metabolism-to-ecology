# run_fba_combinations.py

import pandas as pd  # type: ignore
import itertools
import copy
import pickle
import cobra  # type: ignore
import argparse
from pathlib import Path
from cobra.io import read_sbml_model  # type: ignore
from cobra.flux_analysis import flux_variability_analysis

# === Command-Line Argument Handling ===

parser = argparse.ArgumentParser(description="Run FBA sugar combinations for one or multiple models.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-name", type=str, help="Single model name (e.g., e_coli_core)")
group.add_argument("-multiple_names", nargs="+", help="List of model names (e.g., e_coli_core iJO1366)")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
args = parser.parse_args()

# === Sugar Classification ===

mono_sugars = {'Glucose', 'Fructose', 'Galactose'}
di_sugars = {'Sucrose', 'Maltose', 'Lactose'}

# === Base Paths ===

# Add Repo Root 
project_root = Path(__file__).resolve().parents[3] 

# raw data
raw_path = project_root / "data" / "raw" 
raw_sbml_path = raw_path / "sbml_files"
temp_sbml_path = raw_sbml_path / "temporary_files"
raw_csv_path = raw_path / "csv_files"
raw_pkl_path = raw_path / "pkl_files"

# === Function to Run FBA for a Model ===

def run_fba_for_model(name: str):
    print(f"\n=== Running FBA for model: {name} ===")

    path_to_sbml = temp_sbml_path / f'{name}_neutral.xml'
    path_to_csv = raw_csv_path / f'{name}_edf.csv'
    outputpath_obj_df = raw_csv_path / f'FBA_Objectives_{name}.csv'
    outputpath_flux_dist = raw_pkl_path / f'FBA_Distributions_{name}.pkl'
    outputpath_fva_df = raw_csv_path / f'FVA_{name}.csv'

    noise_threshold = 1e-6  # Set noise threshold

    # Load model and sugar mapping
    original_model = read_sbml_model(path_to_sbml)
    df = pd.read_csv(path_to_csv)

    sugars = df['Sugar'].tolist()
    sugar_to_rxn = dict(zip(df['Sugar'], df['ID']))

    # Only run FBA for each sugar individually (conditions 1–6)
    all_combinations = [[sugar] for sugar in sugars]

    objective_results = []
    flux_distributions = {}
    all_fva_results = []

    for combo in all_combinations:
        if args.verbose:
            print(f"\n--- Running combination: {combo} ---")

        model = copy.deepcopy(original_model)

        monos = [s for s in combo if s in mono_sugars]
        dis = [s for s in combo if s in di_sugars]
        total_sugars = len(combo)

        mono_flux = -10 / total_sugars if monos else 0
        dis_flux = -5 / total_sugars if dis else 0

        for sugar in combo:
            rxn_id = sugar_to_rxn[sugar]
            rxn = model.reactions.get_by_id(rxn_id)
            rxn.lower_bound = mono_flux if sugar in mono_sugars else dis_flux
            rxn.upper_bound = rxn.lower_bound  # fixed uptake

        # Block other sugars
        for sugar in set(sugars) - set(combo):
            rxn_id = sugar_to_rxn[sugar]
            rxn = model.reactions.get_by_id(rxn_id)
            rxn.lower_bound = 0
            rxn.upper_bound = 0

        if args.verbose:
            print("Optimizing model...")
        solution = model.optimize()

        if args.verbose:
            print("Running FVA...")
        fva_result = flux_variability_analysis(model, fraction_of_optimum=1.0)

        # Apply noise threshold to min/max values
        fva_result = fva_result.applymap(lambda x: 0.0 if abs(x) < noise_threshold else x)

        fva_result['Combination'] = ','.join(combo)
        all_fva_results.append(fva_result.reset_index())

        objective_results.append({
            'Objective': solution.objective_value,
            'Combination': combo
        })

        fluxes = solution.fluxes[abs(solution.fluxes) > noise_threshold]
        flux_distributions[tuple(combo)] = fluxes.to_dict()

    if args.verbose:
        print("Saving objective results...")
    pd.DataFrame(objective_results).to_csv(outputpath_obj_df, index=False)

    if args.verbose:
        print("Saving flux distributions...")
    with open(outputpath_flux_dist, 'wb') as f:
        pickle.dump(flux_distributions, f)

    if args.verbose:
        print("Saving FVA results...")
    pd.concat(all_fva_results, ignore_index=True).to_csv(outputpath_fva_df, index=False)

    print(f"=== Finished FBA for model: {name} ===\n")

# === Run Script Safely ===

if __name__ == "__main__":
    if args.name:
        run_fba_for_model(args.name)
    elif args.multiple_names:
        for name in args.multiple_names:
            run_fba_for_model(name)
