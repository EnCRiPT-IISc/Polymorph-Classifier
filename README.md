# OptOP

**Opt**imal **O**rder **P**arameters — a fast Fortran/MPI engine for computing
structural order parameters from molecular‑dynamics trajectories, plus an
XGBoost pipeline that learns to classify phases from those order parameters.

Originally built for water/ice and gas‑hydrate phase discrimination, but it works
for any LAMMPS `.lammpstrj` trajectory: you choose the central atom type, the
neighbour atom types (or a center‑of‑mass representation), the cutoff, the
order‑parameter families, and the ML hyper‑parameters.

---

## Features

- **Order parameters** (per central atom or per molecule COM):
  - Steinhardt `Q_l`, `W_l` and locally‑averaged `LQ_l`, `LW_l` for
    `l = 2,3,4,5,6,8,10,12,14,16`
  - Bond‑angle (`B`), distance (`D`), Fourier (`F`) and tetrahedral (`I`) families
  - Output averaged (Lechner–Dellago), local, or both
- **Flexible neighbour definition**
  - same‑type (e.g. O–O), or
  - mixed types with the molecular convention (e.g. O sees O, H and methane),
    where guests enter a central atom's local environment but not its averaging set
  - **COM–COM**: reduce each molecule to its center of mass (masses given
    explicitly or auto‑detected from element symbols)
- **Triclinic boxes** handled exactly (tilt factors `xy/xz/yz`)
- **Heterogeneous molecules** (different atom counts per molecule, e.g. water + methane)
- **MPI parallel** over trajectory frames (`mpiexec -n N`)
- **ML classifier**: XGBoost + feature‑importance ranking + RandomizedSearchCV
  tuning + Sequential Forward Selection, with confusion matrices, accuracy
  curves and per‑class reports for every model.

## Installation

Requires a Fortran compiler (`gfortran` or Intel `ifx`/`mpiifx`) at build time.

```bash
pip install git+https://github.com/dikshaiisc/OptOP.git
```

From a local clone (editable):

```bash
git clone https://github.com/dikshaiisc/OptOP.git
cd OptOP
pip install -e .
# optional MPI support:
pip install -e ".[mpi]"
```

The Fortran extension is compiled automatically at install time via
`meson-python` + `f2py`.

## Command line

```
OptOP compute   compute order parameters from .lammpstrj trajectories
OptOP train     train an XGBoost phase classifier on an OP CSV
OptOP run       compute then train (end-to-end)
```

Each `.lammpstrj` file in a `--traj` directory becomes one class, labelled by
its filename (e.g. `Ice1c.lammpstrj` → class `Ice1c`).

### Compute order parameters

```bash
# O–O Steinhardt OPs for every phase in a folder, last 50 frames
OptOP compute --traj phases/ --central-type 1 \
      --op-categories Q,W,LQ,LW --last-frames 50 --outdir out/

# O with O,H,methane(M) neighbours (molecular convention), MPI on 48 cores
mpiexec -n 48 OptOP compute --traj phases/ --central-type 1 \
      --neighbor-types 1,2,3 --rcut 3.5 --op-categories Q,W,LQ,LW --outdir out/

# COM–COM order parameters (masses auto-detected from element symbols)
OptOP compute --traj phases/ --central-type 1 --com --outdir out/

# COM–COM with explicit per-type masses
OptOP compute --traj phases/ --central-type 1 --com \
      --masses 1:15.999,2:1.008,3:16.043 --outdir out/
```

Key `compute` options:

| flag | meaning |
|---|---|
| `--traj` | trajectory file or directory of `*.lammpstrj` |
| `--central-type` | LAMMPS atom type of the central atom (e.g. `1`=O) |
| `--neighbor-types` | comma list of neighbour types, e.g. `1,2,3` (default: central only) |
| `--com` | reduce each molecule to its center of mass |
| `--masses` | COM masses `type:mass,...` (else auto from element symbols) |
| `--rcut` | neighbour cutoff in Å |
| `--op-categories` | `B,D,F,I,Q,W,LQ,LW` or `all` (default: all) |
| `--op-type` | `avg` / `local` / `both` |
| `--last-frames` | use only the last N frames per trajectory |
| `--outdir` | output directory |

### Train a classifier

```bash
OptOP train --data out/OP.csv --outdir ml/ \
      --feature-mode cumulative --imp-threshold 0.90 --max-top 20 \
      --phases Ice1c,Ice1h,S1,S2,SH --n-iter 25 --cv 5 --max-sfs 6
```

Key `train` options:

| flag | meaning |
|---|---|
| `--data` | OP CSV with a `Class` column |
| `--feature-mode` | `cumulative` (keep features until cumulative importance ≥ threshold) or `fixed` (keep `--max-top`) |
| `--max-top` / `--imp-threshold` | cap and cumulative cutoff for top features |
| `--phases` | comma list of classes to train on (default: all) |
| `--param-grid` | JSON file with a custom hyper‑parameter grid |
| `--n-iter` / `--cv` | RandomizedSearchCV iterations / CV folds |
| `--max-sfs` | max features for Sequential Forward Selection |
| `--overfit-gap` | train−test gap flagged as overfit |

### End‑to‑end

```bash
OptOP run --traj phases/ --central-type 1 --op-categories Q,W,LQ,LW \
      --last-frames 50 --feature-mode cumulative --outdir out/
```

## Outputs

**compute** → `out/OP.csv` (rows = central atoms/molecules × frames, columns =
selected order parameters, plus a `Class` column).

**train** → in `--outdir`: `model_summary.csv` (k, features, train/test/CV
accuracy, gap, overfit), `best_model.json`, `best_params.json`,
`feature_importances.{csv,png}`, `accuracy_curve.png`,
`cm_{baseline,topN,1feat…}.{png,csv}` (confusion matrices),
`best_classification_report.csv`, `hist_1feat.png`, `scatter_{2,3}feat.png`,
`pca_{k}feat.png`, and `xgb_*.pkl` models.

## Python API

```python
from optop import MPIOPCalculator
calc = MPIOPCalculator(rcut=3.5, central_type=1, neighbor_types=[1,2,3],
                       op_categories=["Q","W","LQ","LW"], last_frames=50)
df_local, df_avg = calc.compute_all("phases/Ice1c.lammpstrj")

from optop.ml import run_training
run_training("out/OP.csv", "ml/", feature_mode="cumulative",
             phases=["Ice1c","Ice1h","S1","S2","SH"])
```

## License

MIT — see [LICENSE](LICENSE).
