# OP_ML

Complete pipeline from LAMMPS trajectories to ML phase classification. Computes 383 structural order parameters via compiled Fortran + MPI, checks for NaNs and stops on error (never drops data), supports interactive phase merging, and trains XGBoost models with automatic feature selection.

## Install

```bash
# From GitHub (private repo — requires access)
pip install git+https://github.com/EnCRiPT-IISc/Polymorph-Classifier.git

# For OP computation, build the Fortran extension first:
cd src/op_ml/fortran && make
```

## Usage

```bash
# See all commands
OP_ML --help

# Compute OPs from trajectory files (Fortran backend, MPI parallel)
mpiexec -n 8 OP_ML compute --traj phases/ --outdir results/

# Train ML classifier on existing OP CSV
OP_ML train --data OP.csv --outdir results/

# Full pipeline: compute OPs then train ML
mpiexec -n 8 OP_ML run --traj phases/ --outdir results/

# Validate a CSV for NaN values
OP_ML validate --file OP.csv
```

## Requirements

- Python >= 3.8
- numpy, pandas, scipy, scikit-learn, xgboost, mlxtend, matplotlib
- For OP computation: Fortran compiler (mpiifx or gfortran), MPI
- For MPI parallel: mpi4py
