# utils.py
import cobra 
import numpy as np
import pandas as pd
from cobra import Reaction
import networkx as nx
import re
import copy
from collections import defaultdict
import cvxpy as cp
from scipy.interpolate import make_interp_spline

import cobra
from cobra import Model, Reaction
from cobra.core import Group, Reaction
from cobra.flux_analysis import flux_variability_analysis
from cobra.flux_analysis.fastcc import fastcc
from cobra.io import read_sbml_model, save_matlab_model, write_sbml_model
from cobra.util.solver import linear_reaction_coefficients
from rapidfuzz import fuzz, process
from cobra.util.array import create_stoichiometric_matrix

def lump_reaction(model, coupled_df, verbose=True, Search_COBRA_groups=False, label_type="Group"):
    """
    Create a lumped model by replacing fully coupled reaction groups with pseudo-reactions.

    Parameters:
    - model: cobra.Model (the original model, will remain unchanged)
    - coupled_df: pd.DataFrame with either:
        - columns ['Coupled_Reactions', 'COBRA_Groups'] (default format), or
        - column ['Reactions'], and Search_COBRA_groups=True
    - verbose: bool, if True print progress messages
    - Search_COBRA_groups: bool, if True, search model groups for each reaction if not supplied
    - label_type: str, either 'Group' or 'Module' — used in pseudo-reaction naming (required)

    Returns:
    - model_lumped: cobra.Model with pseudo-reactions added
    - translation_df: pd.DataFrame mapping pseudo-reactions to original reactions and groups
    """
    model_lumped = copy.deepcopy(model)
    translation_data = []

    # Handle case with only 'Reactions' column — determine COBRA_Groups if needed
    if Search_COBRA_groups and 'Reactions' in coupled_df.columns:
        cobra_groups_list = []
        for i, row in coupled_df.iterrows():
            reaction_list = row['Reactions']
            found_groups = set()
            for rxn_id in reaction_list:
                for group in model.groups:
                    if rxn_id in [rxn.id for rxn in group.members]:
                        found_groups.add(group.name)
            cobra_groups_list.append(list(found_groups))
        coupled_df = coupled_df.rename(columns={'Reactions': 'Coupled_Reactions'})
        coupled_df['COBRA_Groups'] = cobra_groups_list

    for i, row in coupled_df.iterrows():
        group_reactions = row['Coupled_Reactions']
        cobra_groups = row['COBRA_Groups']

        valid_reactions = []
        for rxn_id in group_reactions:
            if rxn_id in model_lumped.reactions:
                valid_reactions.append(rxn_id)
            else:
                if verbose:
                    print(f"⚠️ Reaction '{rxn_id}' not found in model — skipping.")

        if len(valid_reactions) < 2:
            if verbose:
                print(f"⏭️ Skipping group {i} — fewer than 2 valid reactions.")
            continue

        lower_bounds = []
        upper_bounds = []
        for rxn_id in valid_reactions:
            rxn = model_lumped.reactions.get_by_id(rxn_id)
            lower_bounds.append(rxn.lower_bound)
            upper_bounds.append(rxn.upper_bound)
        min_lb = max(lower_bounds)
        max_ub = min(upper_bounds)

        net_stoich = defaultdict(float)
        for rxn_id in valid_reactions:
            rxn = model_lumped.reactions.get_by_id(rxn_id)
            for met, coeff in rxn.metabolites.items():
                net_stoich[met] += coeff

        cleaned_stoich = {met: coeff for met, coeff in net_stoich.items() if abs(coeff) > 1e-10}

        pseudo_id = f"Pseudo_{label_type}_{i}"
        pseudo_rxn = Reaction(id=pseudo_id)
        pseudo_rxn.name = f"Lumped reaction for: {', '.join(valid_reactions)}"
        pseudo_rxn.lower_bound = min_lb
        pseudo_rxn.upper_bound = max_ub
        pseudo_rxn.add_metabolites(cleaned_stoich)

        model_lumped.add_reactions([pseudo_rxn])

        for group in model_lumped.groups:
            if group.name in cobra_groups:
                group.add_members([pseudo_rxn])

        for rxn_id in valid_reactions:
            rxn = model_lumped.reactions.get_by_id(rxn_id)
            for group in model_lumped.groups:
                if rxn in group.members:
                    group.members.remove(rxn)
            model_lumped.reactions.remove(rxn)

        translation_data.append({
            'Pseudo_Reaction_ID': pseudo_id,
            'Original_Reactions': valid_reactions,
            'COBRA_Groups': cobra_groups
        })

        if verbose:
            print(f"✅ {label_type} {i}: Created '{pseudo_id}' from {valid_reactions}")

    translation_df = pd.DataFrame(translation_data)
    return model_lumped, translation_df

def are_reactions_interconnected(model, reaction_ids):
    """
    Check whether a list of reactions are all interconnected via shared metabolites.
    
    Parameters:
    - model: cobra.Model
    - reaction_ids: list of reaction IDs to test

    Returns:
    - bool: True if all reactions are connected via shared metabolites
    """
    # Create a graph where nodes = reactions, edges = shared metabolite
    G = nx.Graph()
    G.add_nodes_from(reaction_ids)

    # Build edges based on shared metabolites
    for i, rxn1_id in enumerate(reaction_ids):
        if rxn1_id not in model.reactions:
            continue
        rxn1 = model.reactions.get_by_id(rxn1_id)
        mets1 = set(rxn1.metabolites)

        for rxn2_id in reaction_ids[i+1:]:
            if rxn2_id not in model.reactions:
                continue
            rxn2 = model.reactions.get_by_id(rxn2_id)
            mets2 = set(rxn2.metabolites)

            # Add edge if they share at least one metabolite
            if mets1 & mets2:
                G.add_edge(rxn1_id, rxn2_id)

    # Check if the graph is fully connected
    return nx.is_connected(G)

def reverse_stoichiometry(reaction: Reaction):
    """
    Reverses the stoichiometry of a given reaction in-place by negating the
    coefficients of all its metabolites.

    Parameters:
        reaction (cobra.Reaction): The COBRApy reaction object to be reversed.

    Returns:
        None: The function modifies the reaction object directly.
    """
    reversed_mets = {met: -coeff for met, coeff in reaction.metabolites.items()}
    reaction.add_metabolites(reversed_mets, combine=False)

def read_rfile(filepath, remove_lambda=True):
    """
    Reads a single-line .rfile and returns a list of reaction IDs.
    Optionally removes the artificial RS_lambda reaction at the end.
    """
    with open(filepath, 'r') as f:
        reactions = [name.strip('"') for name in f.readline().strip().split()]
    return reactions[:-1] if remove_lambda and reactions[-1] == 'RS_lambda' else reactions

def read_mfile(filepath):
    """
    Reads a single-line .mfile and returns a list of metabolite IDs.
    """
    with open(filepath, 'r') as f:
        return [name.strip('"') for name in f.readline().strip().split()]

def check_steady_state_modes(df, model, Mode="EFM"):
    """
    Checks if each row in the DataFrame represents a steady-state mode.
    Prints out any imbalanced modes and returns their indices.
    """
    df = df.copy()
    df['non_zero_count'] = (df != 0).sum(axis=1)
    df['non_zero_items'] = df.apply(lambda row: {col: row[col] for col in df.columns[:-1] if row[col] != 0}, axis=1)

    imbalanced_indices = []
    for idx, row in df.iterrows():
        rxn_dict = row['non_zero_items']
        combined_stoich = {}
        for rxn_id, factor in rxn_dict.items():
            rxn = model.reactions.get_by_id(rxn_id)
            for met, coeff in rxn.metabolites.items():
                combined_stoich[met.id] = combined_stoich.get(met.id, 0) + coeff * factor
        if any(abs(v) > 1e-6 for v in combined_stoich.values()):
            imbalanced_indices.append(idx)
    
    if not imbalanced_indices:
        print(f"    ⚖️ All {Mode}s are balanced.")
    else:
        print(f"⚠️ {len(imbalanced_indices)} {Mode}(s) are imbalanced at indices: {imbalanced_indices}")

def softmax_from_reaction(df, reaction, T):
    """
    Computes softmax-weighted probabilities across EFMs/EFVs using a given reaction.
    """
    x = df[reaction].astype(float).values
    x_scaled = x / T
    exp_x = np.exp(x_scaled - np.max(x_scaled))
    return exp_x / np.sum(exp_x)

def flux_ratio(df, reference_reaction, q):
    """
    Returns a normalised average flux profile across reactions using reference reaction.
    """
    q = np.asarray(q)
    weighted_fluxes = df.multiply(q, axis=0)
    numerator = weighted_fluxes.sum()
    denominator = (df[reference_reaction] * q).sum()
    if denominator == 0:
        raise ZeroDivisionError(f"Weighted flux of {reference_reaction} is zero.")
    return pd.DataFrame([numerator / denominator], columns=df.columns)

def clean_values(df, tiny_threshold=1e-10, round_decimals=4):
    """
    Cleans tiny numerical noise in a DataFrame and rounds values.
    """
    df_cleaned = df.copy()
    df_cleaned[np.abs(df_cleaned) < tiny_threshold] = 0
    return df_cleaned.round(round_decimals)

def softmax_from_reaction2(efm_df: pd.DataFrame, reaction: str, T: float) -> np.ndarray:
    """
    Computes a softmax probability distribution over EFMs/EFVs based on 
    the flux values of a specified reaction (e.g. biomass), scaled by temperature T.
    
    Parameters:
        efm_df (pd.DataFrame): DataFrame with EFMs or EFVs as rows and reactions as columns.
        reaction (str): The reaction name used for softmax scoring (e.g. "Biomass_Ecoli_core").
        T (float): Temperature-like parameter controlling softness of distribution.
                   Lower T → sharper distribution (more deterministic), higher T → flatter.
    
    Returns:
        np.ndarray: A softmax-normalized weight vector (1D), same length as the number of EFMs.
    """
    x = efm_df[reaction].values.astype(float)
    x_scaled = x / T
    x_stable = x_scaled - np.max(x_scaled)
    exp_x = np.exp(x_stable)
    softmax_q = exp_x / np.sum(exp_x)
    return softmax_q

def get_weighted_lambdas(df: pd.DataFrame, T: float, scale: float = 2.83222):
    """
    Computes individual normalized growth rates (λ) for each EFM/EFV 
    and returns their softmax-based weights at a given T.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing EFMs or EFVs with columns 
                           "Biomass_Ecoli_core" and "EX_glc(e)".
        T (float): Temperature-like parameter used in softmax calculation.
        scale (float): Scaling factor applied to all λ values (default: 2.83222).
    
    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Scaled λ values (1D array) for each EFM
            - Corresponding softmax weights (1D array), normalized to sum to 1
    """
    q = softmax_from_reaction2(df, reaction="Biomass_Ecoli_core", T=T)
    biomass = df['Biomass_Ecoli_core'].values
    glucose = df['EX_glc(e)'].values

    with np.errstate(divide='ignore', invalid='ignore'):
        lambda_vals = np.where(glucose != 0, biomass / glucose, np.nan)

    valid = ~np.isnan(lambda_vals)
    lambda_vals = lambda_vals[valid] * scale
    q = q[valid]
    q = q / q.sum()
    return lambda_vals, q

def parse_quickcheck_output(filepath):
    """
    Parse a QuickCheck output file containing reaction bounds and results.

    This function reads a text file where each reaction's data is grouped in a 
    fixed structure of four lines:
        1. Reaction ID line (e.g., "EX_leu-L(e): ...")
        2. Line with lower and upper bounds (e.g., "lb: -10.0 ub: 1000.0")
        3. Line with min and max values (e.g., "min: -10.0 max: 0.0")
        4. Line indicating reversibility (e.g., "reversibility: True")

    It extracts the reaction ID, bounds, min/max flux values, and reversibility
    for each reaction and returns a DataFrame summarizing this information.

    Parameters:
        filepath (str): Path to the QuickCheck output text file.

    Returns:
        pandas.DataFrame: A DataFrame with columns:
            - reaction_id (str)
            - lb (float): Lower bound
            - ub (float): Upper bound
            - min (float): Minimum flux value
            - max (float): Maximum flux value
            - reversibility (bool): Reaction reversibility
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for reaction ID line (e.g. EX_leu-L(e): ...)
        if ':' in line:
            match_id = re.match(r'^(.*?):', line)
            if match_id:
                reaction_id = match_id.group(1)

                # Expect lb/ub in next line
                lb_ub_line = lines[i + 1].strip()
                lb_match = re.search(r'lb:\s*([-\d.]+)', lb_ub_line)
                ub_match = re.search(r'ub:\s*([-\d.]+)', lb_ub_line)
                lb = float(lb_match.group(1)) if lb_match else None
                ub = float(ub_match.group(1)) if ub_match else None

                # Expect min/max in next line
                min_max_line = lines[i + 2].strip()
                min_match = re.search(r'min:\s*([-\d.]+)', min_max_line)
                max_match = re.search(r'max:\s*([-\d.]+)', min_max_line)
                min_val = float(min_match.group(1)) if min_match else None
                max_val = float(max_match.group(1)) if max_match else None

                # Expect reversibility in next line
                rev_line = lines[i + 3].strip()
                rev_match = re.search(r'reversibility:\s*(\w+)', rev_line)
                reversibility = rev_match.group(1) == 'True' if rev_match else None

                data.append({
                    'reaction_id': reaction_id,
                    'lb': lb,
                    'ub': ub,
                    'min': min_val,
                    'max': max_val,
                    'reversibility': reversibility
                })

                # Move to next block
                i += 5
            else:
                i += 1
        else:
            i += 1

    return pd.DataFrame(data)

def is_in_convex_hull(x, others, tol=1e-5):
    """
    Check if vector x lies in the convex hull of the rows in 'others'.
    """
    n = others.shape[0]
    if n == 0:
        return False

    lambdas = cp.Variable(n)
    constraints = [lambdas >= 0, cp.sum(lambdas) == 1]
    reconstruction = others.T @ lambdas
    objective = cp.Minimize(cp.norm(reconstruction - x))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    return prob.status == 'optimal' and prob.value < tol

def convexly_independent_set(vectors):
    """
    Iteratively remove vectors that lie in the convex hull of the others.
    Returns a boolean mask indicating which vectors are kept.
    """
    n = len(vectors)
    keep = np.ones(n, dtype=bool)

    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not keep[i]:
                continue
            others = vectors[keep & (np.arange(n) != i)]
            if is_in_convex_hull(vectors[i], others):
                print(f"Removing vector at index {i} (convex combination of others).")
                keep[i] = False
                changed = True
                break  # Restart loop after any removal
    return keep

def match_reactions_to_bigg(reaction_ids, reaction_df):
    found_mask = reaction_df["bigg_id"].isin(reaction_ids)
    matched_df = reaction_df[found_mask].copy()
    matched_ids = set(matched_df["bigg_id"])
    not_yet_matched = [rxn for rxn in reaction_ids if rxn not in matched_ids]

    # Search in old_bigg_ids
    matches_in_old = []
    for idx, row in reaction_df.iterrows():
        old_ids = row["old_bigg_ids"]
        if isinstance(old_ids, list):
            if any(rxn in old_ids for rxn in not_yet_matched):
                matches_in_old.append(idx)

    matched_old_df = reaction_df.loc[matches_in_old].copy()

    # Determine still missing
    found_in_old_ids = set()
    for old_list in matched_old_df["old_bigg_ids"]:
        for rxn in old_list:
            if rxn in not_yet_matched:
                found_in_old_ids.add(rxn)

    still_missing = [rxn for rxn in not_yet_matched if rxn not in found_in_old_ids]

    return matched_df, matched_old_df, still_missing

def print_summary_to_file(category_name, result_dict, file_path, total_counts=None):
    """
    Writes a formatted summary of reaction ID matching results for multiple models to a text file.

    Parameters:
    ----------
    category_name : str
        A label for the category of reactions being summarized (e.g., "Exchange Reactions").

    result_dict : dict
        A dictionary where each key is a model name (str), and each value is a tuple:
        (matched_df, matched_old_df, missing), where:
            - matched_df (pd.DataFrame): reactions matched via "bigg_id"
            - matched_old_df (pd.DataFrame): reactions matched via "old_bigg_ids"
            - missing (list of str): reaction IDs not matched at all

    file_path : Path or str
        Full path to the `.txt` file where the summary should be written.

    total_counts : dict, optional
        Dictionary mapping model names to total number of reaction IDs originally attempted for matching.
        If not provided, the total is computed from the result_dict.

    Returns:
    -------
    None
        Writes the summary to the specified text file.
    """
    with open(file_path, "a") as f:
        f.write(f"\n=== Matching Summary for {category_name} ===\n\n")
        for model_name, (matched_df, matched_old_df, missing) in result_dict.items():
            total = total_counts.get(model_name) if total_counts else (
                len(matched_df) + len(matched_old_df) + len(missing)
            )
            n_direct = len(matched_df)
            n_old = len(matched_old_df)
            n_missing = len(missing)
            coverage = 100 * (n_direct + n_old) / total if total else 0

            f.write(f"🔍 {model_name}\n")
            f.write(f"   Total:                   {total}\n")
            f.write(f"✅ Found in bigg_id:        {n_direct}\n")
            f.write(f"✅ Found in old_bigg_ids:   {n_old}\n")
            f.write(f"❌ Still not found:         {n_missing}\n")
            f.write(f"🔎 BiGG Coverage:           {coverage:.1f}%\n\n")

def smooth_line(x, y, points=300, degree=1):
    """
    Returns a smoothed curve using spline interpolation.

    Parameters:
    ----------
    x : array-like
        The x-values of the input data (independent variable). Must be sorted and strictly increasing.
    y : array-like
        The y-values corresponding to `x` (dependent variable).
    points : int, optional (default = 100)
        Number of points to generate in the smoothed output. Higher values result in a smoother, higher-resolution curve.
        Lower values give a coarser curve. Does not affect the interpolation itself, only the density of the output.
    degree : int, optional (default = 1)
        Degree of the spline used for interpolation:
            - 1 = linear spline (piecewise straight lines)
            - 2 = quadratic spline
            - 3 = cubic spline (smooth and commonly used)
        Higher degrees can produce smoother curves but require more data points. If `len(x) <= degree`, the original (x, y) is returned.

    Returns:
    -------
    x_new : ndarray
        A new x-array with `points` evenly spaced values over the range of `x`.
    y_smooth : ndarray
        The interpolated y-values corresponding to `x_new`, forming the smoothed curve.

    Notes:
    -----
    - Uses scipy's `make_interp_spline` for B-spline interpolation.
    - This function assumes `x` and `y` are numeric and of the same length.
    - It is especially useful for creating smooth visualizations of noisy or low-resolution data.
    """
    if len(x) > degree:
        x_new = np.linspace(x.min(), x.max(), points)
        spline = make_interp_spline(x, y, k=degree)
        y_smooth = spline(x_new)
        return x_new, y_smooth
    else:
        return x, y
    
def smooth_line2(x, y, points=300, degree=1):
    """Returns a smoothed curve using spline interpolation, ignoring NaNs/infs."""
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaNs and infs from both x and y
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isinf(x)) & (~np.isinf(y))
    x = x[mask]
    y = y[mask]

    if len(x) > degree:
        x_new = np.linspace(x.min(), x.max(), points)
        spline = make_interp_spline(x, y, k=degree)
        y_smooth = spline(x_new)
        return x_new, y_smooth
    else:
        return x, y
