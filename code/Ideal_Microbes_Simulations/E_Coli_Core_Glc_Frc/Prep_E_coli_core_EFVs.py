# -------------------------------
# Imports
# -------------------------------

import pandas as pd
import cobra 
import numpy as np
import copy
import re
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys

from cobra import Reaction
from cobra.io import read_sbml_model, save_matlab_model, write_sbml_model
from cobra.util.array import create_stoichiometric_matrix
from pathlib import Path  

# Add Repo Root to sys.path 
project_root = Path(__file__).resolve().parents[2]  
sys.path.append(str(project_root))

# Import Helper Functions
from src.utils import (parse_quickcheck_output)

# -------------------------------
# File Paths
# -------------------------------

# raw path
raw_path = project_root / "data" / "raw"
mplrs_path = raw_path / "mplrs_data"
sbml_path = raw_path / "sbml_files"
temp_path = sbml_path / "temporary_files"

# -------------------------------
# Data
# -------------------------------

print("Loading E. coli core...")

model = read_sbml_model(sbml_path / 'e_coli_core.xml')

print(" Done! ✅")

# -------------------------------
# Modify Exchange Fluxes
# -------------------------------

print("Adjusting Exchange fluxes...")

# Run FBA
solution = model.optimize()

# Collect data from exchange reactions
data = []
for rxn in model.exchanges:
    flux = solution.fluxes.get(rxn.id, None)
    included_in_fba = flux is not None and abs(flux) > 1e-20  # Tiny flux ≈ zero

    # Determine Export/Import values
    export = None
    import_ = None
    if included_in_fba:
        export = flux > 0
        import_ = flux < 0
    else:
        # Use bounds as heuristic if not used in FBA
        if rxn.lower_bound == 0 and rxn.upper_bound > 0:
            export = True
            import_ = False
        elif rxn.upper_bound == 0 and rxn.lower_bound < 0:
            export = False
            import_ = True

    data.append({
        'ID': rxn.id,
        'name': rxn.name,
        'LB': rxn.lower_bound,
        'UB': rxn.upper_bound,
        'FBA': included_in_fba,
        'FBA_flux': flux if included_in_fba else None,
        'Export': export,
        'Import': import_
    })

# Convert to DataFrame
exchange_df = pd.DataFrame(data)

# Sort and reset index
exchange_df.sort_values(by='ID', inplace=True)
exchange_df.reset_index(drop=True, inplace=True)

# D-Glucose exchange, D-Fructose exchange (analogous to mplrs data)

exchange_df.loc[exchange_df['ID'] == "EX_fru_e", 'Import'] = True
exchange_df.loc[exchange_df['ID'] == "EX_fru_e", 'Export'] = False

# copy model and modify directionalities

rev_model = copy.deepcopy(model)

for i, row in exchange_df.iterrows():
    rxn_id = row['ID']
    export = row['Export']
    import_ = row['Import']

    try:
        rxn = rev_model.reactions.get_by_id(rxn_id)
    except KeyError:
        continue

    original_lb = rxn.lower_bound
    original_ub = rxn.upper_bound

    # Handle Export cleanup
    if export and original_lb < 0:
        rxn.lower_bound = 0

    # Handle Import split
    elif import_:
        # Adjust original reaction bounds
        new_bound = abs(original_lb)
        rxn.lower_bound = 0
        rxn.upper_bound = new_bound

        # Create reverse reaction
        rxn_rev = Reaction(id=rxn_id + "_rev")
        rxn_rev.name = rxn.name + " (reversed)"
        rxn_rev.lower_bound = 0
        rxn_rev.upper_bound = new_bound
        rxn_rev.add_metabolites({met: -coeff for met, coeff in rxn.metabolites.items()})
        rev_model.add_reactions([rxn_rev])

        # Handle groups
        for group in rev_model.groups:
            if rxn in group.members:
                group.members.remove(rxn)
                group.members.add(rxn_rev)

        # Remove original reaction from model
        rev_model.reactions.remove(rxn)

# export model

write_sbml_model(rev_model, temp_path / 'revmodel.xml')

print(" Done! ✅")

# -------------------------------
# Generate Basis Model
# -------------------------------

print("Generate Base Model for EFMlrs and mplrs...")

# Load the model once
model_path = "/Users/Luca/Desktop/Thesis_Rep/Thesis_Rep_2/Data/raw/SBML_files/Temporary_Rep/revmodel.xml"
model = read_sbml_model(model_path)

# Apply exchange bounds (0 to 10) to the desired reactions
exchange_rxns = ["EX_fru_e_rev", "EX_glc__D_e_rev"]
for rxn_id in exchange_rxns:
    rxn = model.reactions.get_by_id(rxn_id)
    rxn.lower_bound = 0
    rxn.upper_bound = 10

# Save the modified model
write_sbml_model(model, temp_path / "Base_model")

print(" Done! ✅")

# -------------------------------
# Integrate Quickcheck Results
# -------------------------------

# Run https://github.com/BeeAnka/EFMlrs/blob/master/efmlrs/quickcheck_sbml.py on "Base_model"
# and integrate the requested adjustments

# Then run EFMlrs and mplrs on the model (to compare: Outputs found in raw / mplrs_data)

print(" Note: To continue, run quickcheck_sbml.py on Base_model, \n adjust Base_model accordingly, then execute EFMlrs and mplrs")

print('Everything complete! 🎉')