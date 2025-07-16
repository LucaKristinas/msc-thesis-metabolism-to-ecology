# 📁 OFV Generation

This folder contains all scripts required to generate input files for simulations using the **Ideal Microbes** package developed by Vít Piskovsky. Specifically, it produces metabolic pathway representations, **Optimal Flux Vectors (OFVs)**, which are essentially FBA-derived flux profiles for different combinations of carbon sources.

The analysis included simulating minimal, anoxic growth conditions across three organisms: *E. coli*, *Bacteroides thetaiotaomicron*, and *Bifidobacterium longum*. The carbon sources explored include: **glucose, lactose, fructose, maltose, galactose, and sucrose**. Conditions were chosen to be as constrained as possible while remaining biologically valid, enabling integrative comparisons across different GEMs.

## 🧩 File Descriptions

| Script/File Name                       | Description                                                                                                                                                                      |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1_Sugar_Selection.py`                 | Loads sugar utilization data from VMH and selects relevant carbon sources for OFV computation.                                                                                   |
| `2_Model_Preprocessing.py`             | Performs manual curation to simulate minimal environments for the selected GEMs under anoxic conditions.                                                                         |
| `3_FBA_Computations.py`                | Runs FBA simulations for the selected sugars, normalizes fluxes, and saves both objective values and flux distributions for each GEM. Supports both single- and multi-GEM modes. |
| `4_Flux_Activity_Analysis.py`          | Analyzes flux distributions to determine reaction activity patterns across conditions (Reaction Activity Analysis).                                                              |
| `5_Ideal_Microbes_Input_Generation.py` | Modifies GEMs based on flux activity, generates preliminary OFVs, and creates final internal and external stoichiometric matrices for Ideal Microbes simulations.                |
| `6_Convex_Independence_Check.py`       | Identifies a convexly independent subset of OFVs, ensuring no single OFV lies within the convex hull of the others.                                                              |

## 🧬 Note on Activity Codes in `4_Flux_Activity_Analysis.py`

Each reaction is assigned a code based on its activity pattern across simulations:

| Code | Interpretation                                                       |
| ---- | -------------------------------------------------------------------- |
| `0`  | Always inactive (no flux under any condition)                        |
| `1`  | Always active in the forward direction only                          |
| `2`  | Always active in the reverse direction only                          |
| `3`  | Always active in both directions (can carry flux forward or reverse) |
| `4`  | Either inactive or active in the forward direction only              |
| `5`  | Either inactive or active in the reverse direction only              |
| `6`  | Either inactive or active in both directions                         |