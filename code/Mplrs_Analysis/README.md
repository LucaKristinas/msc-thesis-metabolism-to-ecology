# 📁 mplrs Analysis

This folder contains the analysis and quantitative evaluation of the **mplrs** algorithm—a scalable, parallelized C wrapper for the **lrs** vertex enumeration algorithm, developed by David Avis and adapted by Skip Jordan. `mplrs` enables distributed computation across multicore machines via MPI, making it highly suitable for high-performance computing (HPC) environments.

In this project, `mplrs` is used in combination with the **EFMlrs** Python package to enumerate **elementary flux modes (EFMs)** and **elementary flux vectors (EFVs)** in metabolic networks. This complements the analyses in:

* **Buchner & Zanghellini (2021)**: *EFMlrs: a Python package for elementary flux mode enumeration via lexicographic reverse search*. *BMC Bioinformatics* 22:1–21.
* **Avis & Jordan (2018)**: *mplrs: A scalable parallel vertex/facet enumeration code*. *Math. Prog. Comp.* 10:267–302. [https://doi.org/10.1007/s12532-017-0129-y](https://doi.org/10.1007/s12532-017-0129-y)

By applying these tools to models of increasing complexity, this analysis contributes to a deeper understanding of the practical limitations of EFV enumeration in metabolic modelling and highlights current computational boundaries.

## 🧩 File Descriptions

| Script/File Name             | Description                                                                                                                                                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DOF_Model_Generation.ipynb` | Generates models with varying degrees of freedom (DoF 23–26) from the *iAF1260 core* model (Orth et al., 2010). Prepares these for generic pathway computation and ensures compatibility with EFMlrs/mplrs. |
| `DOF_Figures.py`             | Analyzes the output data from `mplrs` and generates all related figures for this section of the project.                                                                                                    |
| `efmlrs_pre_template.sh`     | Shell script template demonstrating how EFMlrs preprocessing was executed in this project.                                                                                                                  |
| `efmlrs_post_template.sh`    | Shell script template demonstrating how EFMlrs postprocessing was handled.                                                                                                                                  |
| `mplrs_template.sh`          | Shell script template showing how `mplrs` was run on HPC infrastructure.                                                                                                                                    |