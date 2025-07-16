# 📁 Genome-scale Metabolic Model Reduction

This folder contains all code related to Python-native approaches for reducing genome-scale metabolic models. The strategy is based on reaction-lumping via metabolic modules. It includes both generic and context-specific reduction attempts, primarily using the *iAF1260* and *iMS520* models.

The folder also contains `iMS520_example_cobrapy.py`, which highlights the limitations of using COBRApy alone for model validation and emphasizes the importance of external tools such as **MEMOTE** for ensuring biological consistency.

Each script is modular and contributes to the broader project on modelling microbial community metabolism.

## 🧩 File Descriptions

| Script/File Name                     | Description                                                                                                                                                                               |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `iAF1260_Subsystem_Assignment.ipynb` | Assigns metabolic subsystems to reactions in the modern SBML version of *iAF1260* using BiGG metatables and manual curation.                                                              |
| `iAF1260_Reduction.ipynb`            | Demonstrates a generic reduction strategy for *iAF1260*. This version does not yield a valid model but outlines the approach.                                                             |
| `iAF1260_Reduction_Context.ipynb`    | Applies a context-specific reduction strategy to *iAF1260*. Also non-functional but conceptually demonstrates the method.                                                                 |
| `iMS520_Reduction.py`                | Generic reduction script for the *iMS520* model. Strategy is demonstrated but the output model is not valid.                                                                              |
| `iMS520_Reduction_Context.py`        | Context-specific reduction of *iMS520* with a focus on targeted pathways. Like the generic version, it does not yield a valid model.                                                      |
| `iMS520_example_cobrapy.py`          | Illustrates the limitations of COBRApy for model validation using *iMS520*. Highlights issues in previously reduced models and produces models analysed with **MEMOTE** to identify these flaws. |
| `iMS520_F2C2.py`                     | Template for applying **F2C2** consistency checking on COBRApy-compatible models.                                                                                                         |
| `iMS520_FASTCC.py`                   | Template for applying **FASTCC** consistency checking on COBRApy-compatible models.                                                                                                       |
