# ⚡️ From Genomes to Ecology: A Metabolic-Mode–Based Framework for Modeling Microbial Growth Strategies

This project integrates a hybrid framework that bridges constraint-based metabolic models (FBA) 
and consumer–resource models using elementary flux vectors (EFVs). With *E. coli* as a case study, 
we show how EFVs informed by molecular noise improve predictions of flux distributions, growth, 
and nutrient dynamics, while highlighting computational limits at larger scales.

This repository contains the code, data, and environment configuration for this project.

## 🗂️ Repository Structure

```
├── code/ # Core scripts for analyses and simulations
│ ├── GEM_Reduction/ # Scripts for GEM reduction strategies
│ ├── Ideal_Microbes_Benchmarking/ # Benchmarking Ideal Microbes framework
│ ├── Ideal_Microbes_Simulations/ # Simulations with Ideal Microbes package
│ ├── Mplrs_Analysis/ # Analysis of mode enumeration with mplrs
│ └── Optimal_Flux_Vectors/ # Scripts for OFV preparation and analysis
│
├── data/
│ ├── processed/ # Processed/derived datasets
│ └── raw/ # Raw input data
│
├── src/ # Source utilities and helper functions
│ └── utils.py # Common utility functions
│
├── .gitignore # Ignore rules for git
├── environment.yml # Conda environment for reproducibility
└── README.md # Project overview (this file)
```

## 🛠️ Note on External Tools Used

This project makes use of the following external tools:

- **[EFMlrs](https://github.com/BeeAnka/EFMlrs)**: Buchner, Bianca A., and Jürgen Zanghellini. "EFMlrs: a Python package for elementary flux mode enumeration via lexicographic reverse search." BMC bioinformatics 22.1 (2021): 547.
- **[mplrs](https://cgm.cs.mcgill.ca/~avis/C/lrs.html)**: Avis, David, and Charles Jordan. "mplrs: A scalable parallel vertex/facet enumeration code." Mathematical Programming Computation 10.2 (2018): 267-302.
- **[COBRApy](https://github.com/opencobra/cobrapy)**: Ebrahim, Ali, et al. "COBRApy: constraints-based reconstruction and analysis for python." BMC systems biology 7.1 (2013): 74.
- **[Memote](https://github.com/opencobra/memote)**: Lieven, Christian, et al. "MEMOTE for standardized genome-scale metabolic model testing." Nature biotechnology 38.3 (2020): 272-276.

## 📄 License
This repository is part of a Master's thesis and is not licensed for reuse without permission. For academic use or inquiries, please contact the author.