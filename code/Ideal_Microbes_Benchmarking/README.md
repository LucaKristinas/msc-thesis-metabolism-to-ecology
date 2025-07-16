# 📁 Ideal Microbes Benchmarking

This folder contains code used to benchmark the **Ideal Microbes** framework for microbial metabolism modelling.

## 🧩 File Descriptions

| Script/File Name         | Description                                                                                                                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1_Model_Preparation.py` | Reconstructs the metabolic model from De Martino et al. (2018), preparing it for pathway analysis and ensuring compatibility with EFMLrs/mplrs.                                                     |
| `2_FluxMode_Analysis.py` | Main analysis script: normalizes EFMs/EFVs generated with `mplrs` for the *E. coli* core model and computes MSE values compared to experimental flux data across a range of molecular noise levels. |
| `3_FBA_Maxent_data.py`   | Calculates MSE for classical COBRApy FBA and the De Martino et al. model, comparing predictions to experimental flux data.                                                                          |
| `4_Figures.py`           | Generates all figures used in this part of the analysis.                                                                                                                                            |