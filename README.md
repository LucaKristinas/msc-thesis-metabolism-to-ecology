# ⚡️ Modelling Microbial Community Metabolism: Bridging Constraint-Based and Ecological Approaches

This project explores microbial metabolic modelling by integrating concepts from systems biology and statistical mechanics. It combines classical constraint-based approaches (such as Flux Balance Analysis, or FBA) with ecological models based on consumer-resource dynamics.

Key contributions of the project include the use of different network representations to model aerobic glucose metabolism in E. coli, and the demonstration that incorporating molecular noise by allowing for "suboptimal" growth behavior can outperform classical Flux Balance Analysis (FBA) in predicting metabolic fluxes and growth. The project also quantifies the limitations of translating metabolic networks into meaningful pathway representations by applying the state-of-the-art algorithm mplrs to models of increasing complexity. Finally, it introduces the novel pathway concept optimal flux vectors to model a 3-member microbial community across distinct environmental conditions, and uses these vectors to simulate community growth on various carbon sources.

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

- **mplrs**: Used for vertex enumeration in metabolic network polyhedra
- **COBRApy**: Modelling and processing of metabolic networks
- **Memote**: For metabolic model quality assessment
- **Python**, **Bash**, and **MATLAB** are all used in various parts of the analysis

## 📄 License
This repository is part of a Master's thesis and is not licensed for reuse without permission. For academic use or inquiries, please contact the author.