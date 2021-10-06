# Validation of collective optimal transport using simulated spatial cell-cell communication


## Introduction

This repository contains all of the code, input data, and figures used as synthetic validation of the package COMMOT, as presented in Figures 2 and S1-4 of Cang et al. [1]:

More specifically, the code simulates spatial cell-cell communication using reaction-diffusion partial differential equation (PDE) models of ligand-receptor binding in 2D, where we consider multiple ligand species, some of which can bind to one or more receptor species. We consider ten cases of ligand-receptor binding of varying levels of competition, which are characterised by the number of ligands, the number of receptors, by the common receptors to which different ligands can bind. For each case, we ran ten simulations, each with a randomised initial condition.

Numerical solutions of the PDEs are obtained using the Python Package [py-pde](https://py-pde.readthedocs.io/en/latest/getting_started.html); see Zwicker [2] for more details. The output from numerical simulations is then compared to traditional optimal transport methods and the novel collective optimal transport method presented in COMMOT.

The file that `synthetic_ligand_receptor_interaction.py` contains the code that defines the PDE model using py-pde and helper functions to plot the initial data and final numerical solutions. Each case is defined in its own Python script; for example, the code to simulate Case 1, which contains two ligand species that bind to a common receptor (Fig. S1; Cang et al. [1]), is found in `synthesize_case_1.py`. The scripts to run the other cases are named analogously.

## Installation

To run the code, all one needs is py-pde and all of its dependencies. Instructions on installing py-pde and its dependencies can be found [here](https://py-pde.readthedocs.io/en/latest/getting_started.html).

## Reproducing results

To produce the results for a case, e.g. Case 1, from whichever terminal one runs Python code, one can run:

```python3 synthesize_case_1.py```

All of the parameter values can be found in each Python script. Moreover, the random state has been specified, so that users can reproduce the results in [1] exactly.

For each simulation and each case, all of the input data, saved as `.npy files`, and outputted figures, saved as `.pdf` files, used in Cang et al. [1] can be found in the [data](https://github.com/axelalmet/COMMOTSyntheticBenchmarking/tree/main/data) folder. We did not upload the original PDE solution files, as for cases with a larger number of ligand and receptor species, the files can get quite large (~100mb).

## References

1. Cang, Z., Zhao, Y., Almet, A.A., Atwood, S.X. and Nie, Q. (2021). Screening cell-cell communication in spatial transcriptomics via collective optimal transport, *Under preparation*.
2. Zwicker, D. (2020). py-pde: A Python package for solving partial differential equations. *Journal of Open Source Software*, 5(48), 2158.




