# ════════════════════════════════════════════════════════════════
# 1. Import Packages
# ════════════════════════════════════════════════════════════════

import pandas as pd
import cobra 
import numpy as np
import sys
import copy
import re
from pathlib import Path

from cobra import Reaction
from cobra.io import read_sbml_model, save_matlab_model, write_sbml_model
from cobra.util.array import create_stoichiometric_matrix

# Add Repo Root to sys.path 
project_root = Path(__file__).resolve().parents[2]  
sys.path.append(str(project_root))

# Import Helper Functions
from src.utils import (reverse_stoichiometry)

# ════════════════════════════════════════════════════════════════
# 2. Import & Export Paths
# ════════════════════════════════════════════════════════════════

# Raw data paths
raw_data_path = project_root / "data" / "raw"
csv_raw_path = raw_data_path / "csv_files"
sbml_raw_path = raw_data_path / "sbml_files"

# Processed data paths
processed_data_path = project_root / "data" / "processed"
sbml_processed_path = processed_data_path / "sbml_files"

# ════════════════════════════════════════════════════════════════
# 3. Data Import
# ════════════════════════════════════════════════════════════════ 

print('📦 Importing Data...')

# Import stoichiometric matrix and modified flux bounds
df = pd.read_csv(csv_raw_path / "iAF1260_core_S.csv", sep=',')
reaction_bounds = pd.read_csv(csv_raw_path / "iAF1260_core_mod_FluxBounds.csv", sep=',')

print(' ✅ Done!')

# ════════════════════════════════════════════════════════════════
# 4. Modify to create model used in Martino et al., 2018
# ════════════════════════════════════════════════════════════════

print('Create model used in Martino et al., 2018...')

# Remove reactions not present in Martino model
rem_reaction = ['EX_fru(e)','EX_fum(e)','EX_gln_L(e)','EX_mal_L(e)','FRD7','FRUpts2','FUMt2_2','GLNabc','MALt2_2']
df_new = df.drop(columns=rem_reaction)

# Remove unused metabolites
mask = (df_new.iloc[:, 1:] == 0).all(axis=1)
print(f"Entries in column 0 for rows where all other columns are 0:\n{df_new.loc[mask, df_new.columns[0]]}")
df_martino = df_new[~mask].copy()

# Extract matrix data
stoch_mat = df_martino.iloc[:, 1:].to_numpy(dtype=float)
metabolites = df_martino.iloc[:, 0].tolist()
reaction_ids = df_martino.columns[1:].tolist()
bounds_list = list(zip(reaction_bounds["min"], reaction_bounds["max"]))

# ════════════════════════════════════════════════════════════════
# 5. Create cobra model
# ════════════════════════════════════════════════════════════════

model = cobra.Model("martino_model")

# Add metabolites
metabolite_objects = {}
for met in metabolites:
    metabolite = cobra.Metabolite(id=met)
    model.add_metabolites(metabolite)
    metabolite_objects[met] = metabolite  

# Add reactions with stoichiometry and bounds
for j, rxn_id in enumerate(reaction_ids):
    reaction = cobra.Reaction(id=rxn_id)
    stoich_coeffs = stoch_mat[:, j]
    reaction_metabolites = {
        metabolite_objects[metabolites[i]]: stoich_coeffs[i]
        for i in range(len(metabolites))
        if stoich_coeffs[i] != 0  
    }
    reaction.add_metabolites(reaction_metabolites)
    reaction.lower_bound, reaction.upper_bound = bounds_list[j]
    model.add_reactions([reaction])

# Add compartments
for rxn in model.reactions:
    for met in rxn.metabolites:
        if rxn.id.startswith("EX_"):
            met.compartment = 'e'
        elif met.compartment != 'e':
            met.compartment = 'c'

print(' Running Sanity Checks...')

# Sanity check: Run FBA

model.objective = "Biomass_Ecoli_core"
solution = model.optimize()
print("     Optimal Growth Rate:", round(solution.objective_value, 2))
doubling_time = np.log(2) / solution.objective_value
print(f"        Doubling Time: {doubling_time:.2f} hours")

# Export intermediate model
write_sbml_model(model, sbml_raw_path / "raw_kmmmodel.xml") # 'raw' cobra model

print(' ✅ Done!')

# ═══════════════════════════════════════════════════════════════════════════
# 6. Extreme Pathway Feasibility Modification (reversibilities of exchanges)
# ═══════════════════════════════════════════════════════════════════════════

print(' Extreme Pathway Feasibility Modification...')

# Adjust reversibilities of exchanges for computational feasibility

ep_model = copy.deepcopy(model)
ex_reactions = [rxn.id for rxn in ep_model.exchanges]
ex_directions = ['export','export','export','export','export','export','import','export','export','export','export','import','import','import','export','export']

exchange_df = pd.DataFrame({ 'Reaction': ex_reactions, 'Direction': ex_directions })

original_info = {
    rxn.id: {
        "bounds": (rxn.lower_bound, rxn.upper_bound),
        "reaction": rxn.reaction
    }
    for rxn in ep_model.reactions
    if rxn.id in exchange_df['Reaction'].values
}

for _, row in exchange_df.iterrows():
    rxn_id, direction = row['Reaction'], row['Direction'].lower()
    reaction = ep_model.reactions.get_by_id(rxn_id)

    if direction == "import":
        reaction.lower_bound = -10 if rxn_id == "EX_glc(e)" else -1000
        reaction.upper_bound = 0
        reverse_stoichiometry(reaction)
        reaction.upper_bound = abs(reaction.lower_bound)
        reaction.lower_bound = 0
    elif direction == "export":
        reaction.lower_bound = 0
        reaction.upper_bound = 1000

print(' Running Sanity Checks...')

# Sanity check
solution = ep_model.optimize()
print("     Optimal Growth Rate:", round(solution.objective_value, 2))
doubling_time = np.log(2) / solution.objective_value
print(f"        Doubling Time: {doubling_time:.2f} hours")

# Export updated model
write_sbml_model(ep_model, sbml_processed_path / "klamt_martino_model.xml")

print(' ✅ Done!')

# ════════════════════════════════════════════════════════════════
# 7. MPLRs Feasibility 
# ════════════════════════════════════════════════════════════════

print(' EFMlrs/ mplrs pre-processing...')

# External script used from: quickcheck_sbml.py by BeeAnka
# Source: https://github.com/BeeAnka/EFMlrs/blob/master/efmlrs/quickcheck_sbml.py
# Repository: https://github.com/BeeAnka/EFMlrs
# Purpose: Used for checking SBML models before running EFM calculations
# License: MIT (see original repo for full details)

# Final adjustments based quickcheck_sbml.py run with 'klamt_martino_model.xml'

ep_qc_model = copy.deepcopy(ep_model)

fva_bounds = {
    "ACALD": (-18.6717703976, 0.0),
    "ACALDt": (-18.6717703976, 0.0),
    "ACKr": (-18.6717703976, 0.0),
    "ACONTa": (0.0942873914, 18.766057789),
    "ACONTb": (0.0942873914, 18.766057789),
    "ACt2r": (-18.6717703976, 0.0),
    "ADK1": (0.0, 150.1556660394),
    "AKGt2r": (-9.3358851988, 0.0),
    "ALCD2x": (-18.6717703976, 0.0),
    "CO2t": (-56.280983331, 0.0),
    "D-LACt2": (-18.6717703976, 0.0),
    "ENO": (0.9651763907, 19.6369467884),
    "ETOHt2r": (-18.6717703976, 0.0),
    "G6PDH2r": (0.0, 56.0153111929),
    "GAPD": (1.0959150482, 19.7676854458),
    "GLUt2r": (-9.3358851988, 0.0),
    "H2Ot": (-56.9175827136, 0.0),
    "ICDHyr": (0.0942873914, 18.766057789),
    "LDH_D": (-18.6717703976, 0.0),
    "NH4t": (0.4765319193, 9.8124171181),
    "O2t": (0.0, 56.1799492656),
    "PGK": (-19.7676854458, -1.0959150482),
    "PGM": (-19.6369467884, -0.9651763907),
    "PIt2r": (0.3214895048, 3.2148950477),
    "PTAr": (0.0, 18.6717703976),
    "PYRt2r": (-18.6717703976, 0.0),
    "RPI": (-18.7345878756, -0.0628174779),
    "SUCOAS": (-18.6717703976, 0.0)
}

zero_flux_reactions = []

for rxn_id, (fva_min, fva_max) in fva_bounds.items():
    rxn = ep_qc_model.reactions.get_by_id(rxn_id)
    if fva_max <= 0:
        reverse_stoichiometry(rxn)
        rxn.lower_bound = 0
        rxn.upper_bound = 1000
    else:
        rxn.lower_bound = 0
        rxn.upper_bound = 1000

print(' ✅ Done!')

# ════════════════════════════════════════════════════════════════
# 7. Export and Information
# ════════════════════════════════════════════════════════════════

print('Model Information:')

# Model information

stoich_dense = create_stoichiometric_matrix(ep_qc_model)
rank = np.linalg.matrix_rank(stoich_dense)
num_reactions = len(ep_qc_model.reactions)
dof = num_reactions - rank

print(f"    Stoichiometric Matrix Rank: {rank}")
print(f"    Number of Reactions: {num_reactions}")
print(f"    Degrees of Freedom: {dof}")

print('💾 Exporting final model...')

# Export final model
write_sbml_model(ep_qc_model, sbml_processed_path / "klamt_martino_model_mplrs.xml") # this served as input for mplrs

print(' ✅ Done!')

print('Everything complete! 🎉')