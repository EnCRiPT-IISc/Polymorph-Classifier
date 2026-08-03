"""Unified CLI entry point: OP_ML

Subcommands:
    OP_ML compute   — Compute OPs from trajectory files (Fortran+MPI)
    OP_ML train     — Train ML classifier on existing OP CSV
    OP_ML run       — Full pipeline: compute OPs then train ML
    OP_ML validate  — Check an OP CSV file for NaN values
"""

import argparse
import os
import sys
import glob
import json
from time import time


def _add_op_args(pa):
    """Add OP computation args to a subparser."""
    pa.add_argument("--traj", metavar="DIR", required=True,
                    help="Directory containing .lammpstrj files (one per phase)")
    pa.add_argument("--rcut", type=float, default=None,
                    help="Neighbor cutoff in Angstroms (asked if not given)")
    pa.add_argument("--central-type", type=int, default=None,
                    help="LAMMPS atom type for central atom, e.g. 1 (asked if not given)")
    pa.add_argument("--neighbor-types", default=None,
                    help="LAMMPS atom types for neighbors, comma-separated, e.g. 1,2 "
                         "(default: same as central type)")
    pa.add_argument("--op-type", choices=["avg", "local", "both"], default=None,
                    help="Which OPs to output: avg, local, both (asked if not given)")
    pa.add_argument("--op-categories", default=None,
                    help="OP categories to compute, comma-separated: B,D,F,I,Q,W,LQ,LW "
                         "or 'all' (default: all)")
    pa.add_argument("--last-frames", type=int, default=None,
                    help="Process only the last N frames of each trajectory "
                         "(default: all frames)")


def _add_ml_args(pa):
    """Add ML args to a subparser."""
    pa.add_argument("--max-sfs", type=int, default=None,
                    help="Max features for SFS (default: 6, asked if not given)")
    pa.add_argument("--overfit-gap", type=float, default=None,
                    help="Overfitting threshold (default: 0.05, asked if not given)")
    pa.add_argument("--param-grid", metavar="JSON", default=None,
                    help="Path to JSON file with custom hyperparameter search grid")
    pa.add_argument("--n-iter", type=int, default=None,
                    help="Hyperparameter search iterations (default: 25, asked if not given)")
    pa.add_argument("--no-interactive", action="store_true",
                    help="Skip all prompts, use defaults for unset parameters")


def _add_out_args(pa):
    """Add output args to a subparser."""
    pa.add_argument("--outdir", default="results",
                    help="Output directory (default: results/)")


def _get_mpi_rank():
    """Return MPI rank, or 0 if mpi4py not available or single process."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD
    except ImportError:
        return 0, 1, None


def _is_interactive(args):
    """True only if: not --no-interactive AND single process (or rank 0 of size 1)."""
    if getattr(args, "no_interactive", False):
        return False
    _, size, _ = _get_mpi_rank()
    if size > 1:
        # MPI with multiple ranks: interactive prompts don't work
        return False
    return True


def _ask(prompt, default, cast=str):
    """Prompt user for a value. Return default if empty input."""
    val = input(f"  {prompt} [default={default}]: ").strip()
    return cast(val) if val else default


def _detect_atoms(traj_dir):
    """Scan first trajectory. Returns FrameMeta or None."""
    traj_files = sorted(glob.glob(os.path.join(traj_dir, "*.lammpstrj")))
    if not traj_files:
        return None
    from ._trajectory import scan_first_frame
    return scan_first_frame(traj_files[0])


def _resolve_op_args(args):
    """Fill in missing OP args. Interactive if single process, auto-detect if MPI."""
    interactive = _is_interactive(args)
    rank, size, _ = _get_mpi_rank()

    # Scan trajectory for atom types
    meta = _detect_atoms(args.traj)

    if args.rcut is None:
        if interactive:
            args.rcut = _ask("Neighbor cutoff (Angstroms)", 4.5, float)
        else:
            args.rcut = 4.5

    if args.central_type is None:
        if interactive and meta:
            print(f"\n  Atom types found in trajectory:")
            for t, name in sorted(meta.type_to_name.items()):
                print(f"    Type {t} = {name}")
            default_type = meta.atom_types[0] if meta.atom_types else 1
            args.central_type = _ask("Central atom type (integer)", default_type, int)
        elif meta:
            args.central_type = meta.atom_types[0]
            if rank == 0:
                print(f"  Auto-detected central type: {args.central_type} "
                      f"({meta.type_to_name.get(args.central_type, '?')})")
        else:
            args.central_type = 1

    if args.neighbor_types is None:
        if interactive and meta:
            print(f"\n  Available atom types: "
                  + ", ".join(f"{t}={n}" for t, n in sorted(meta.type_to_name.items())))
            val = input(f"  Neighbor atom types (comma-separated, "
                        f"Enter={args.central_type} only): ").strip()
            if val:
                args.neighbor_types = [int(x.strip()) for x in val.split(",")]
            else:
                args.neighbor_types = None  # = central type only
        # else: stays None = central type only
    elif isinstance(args.neighbor_types, str):
        args.neighbor_types = [int(x.strip()) for x in args.neighbor_types.split(",")]

    if args.op_type is None:
        if interactive:
            args.op_type = _ask("OP output type (avg/local/both)", "avg")
        else:
            args.op_type = "avg"

    if getattr(args, "op_categories", None) is not None:
        if isinstance(args.op_categories, str):
            args.op_categories = [x.strip() for x in args.op_categories.split(",")]
    elif interactive:
        print(f"\n  OP categories: B, D, F, I, Q, W, LQ, LW, all")
        val = input(f"  Which categories to compute (Enter=all): ").strip()
        if val and val.lower() != "all":
            args.op_categories = [x.strip() for x in val.split(",")]
        else:
            args.op_categories = None  # all

    if getattr(args, "last_frames", None) is None and interactive:
        val = input("  Number of last frames to process (Enter=all): ").strip()
        if val:
            args.last_frames = int(val)


def _resolve_ml_args(args):
    """Fill in missing ML args."""
    interactive = _is_interactive(args)

    if args.max_sfs is None:
        if interactive:
            args.max_sfs = _ask("Max features for SFS", 6, int)
        else:
            args.max_sfs = 6

    if args.overfit_gap is None:
        if interactive:
            args.overfit_gap = _ask("Overfitting gap threshold", 0.05, float)
        else:
            args.overfit_gap = 0.05

    if args.n_iter is None:
        if interactive:
            args.n_iter = _ask("Hyperparameter search iterations", 25, int)
        else:
            args.n_iter = 25

    if args.param_grid is None and interactive:
        val = input("  Custom hyperparameter grid JSON (Enter for default): ").strip()
        if val:
            args.param_grid = val


# ── Subcommand handlers ──────────────────────────────────────────────

def cmd_compute(args):
    """OP_ML compute — compute OPs from trajectory files."""
    from .op_calculator import MPIOPCalculator

    traj_files = sorted(glob.glob(os.path.join(args.traj, "*.lammpstrj")))
    if not traj_files:
        print(f"Error: No .lammpstrj files in '{args.traj}'", file=sys.stderr)
        sys.exit(1)

    _resolve_op_args(args)

    rank, _, _ = _get_mpi_rank()
    nt = args.neighbor_types or [args.central_type]
    if rank == 0:
        phases = [os.path.splitext(os.path.basename(f))[0] for f in traj_files]
        print(f"\n[OP_ML compute] {len(traj_files)} phase(s): {phases}")
        print(f"  rcut={args.rcut} A, central_type={args.central_type}, "
              f"neighbor_types={nt}, op_type={args.op_type}")
        if args.op_categories:
            print(f"  op_categories={args.op_categories}")
        print()

    calc = MPIOPCalculator(
        rcut=args.rcut,
        central_type=args.central_type,
        neighbor_types=args.neighbor_types,
        op_categories=getattr(args, "op_categories", None),
        last_frames=getattr(args, "last_frames", None),
    )
    os.makedirs(args.outdir, exist_ok=True)
    calc.compute_batch(traj_files, out_prefix="OP", out_dir=args.outdir,
                       output_type=args.op_type)


def cmd_train(args):
    """OP_ML train — ML classification on existing OP CSV."""
    from .validate import validate_or_exit
    from .ml import (load_and_validate, ask_phase_clubbing, apply_clubbing,
                     prepare_splits, train_baseline, select_top_features,
                     tune_hyperparams, run_sfs)
    from .ml import plots
    import numpy as np
    import pandas as pd

    _resolve_ml_args(args)

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time()

    # Load & validate (stops on NaN)
    print("\n[OP_ML train] Loading data...")
    df = load_and_validate(args.data)
    classes = sorted(df["Class"].unique())
    print(f"  {len(df)} samples, {df.shape[1]-1} features, {len(classes)} classes: {classes}")

    # Phase clubbing
    if not args.no_interactive:
        mapping = ask_phase_clubbing(classes)
    else:
        mapping = {c: c for c in classes}
    df = apply_clubbing(df, mapping)
    classes = sorted(df["Class"].unique())
    print(f"  Classes after mapping: {classes}\n")

    # Split
    X, y, le, X_tr, X_te, y_tr, y_te = prepare_splits(df)

    # Baseline
    print("Training baseline (all features)...")
    importances, tr, te = train_baseline(X_tr, y_tr, X_te, y_te)
    gap = tr - te
    flag = f"  *** OVERFIT (gap={gap:.3f}) ***" if gap > args.overfit_gap else ""
    print(f"  Train={tr:.4f}  Test={te:.4f}{flag}\n")

    # Feature ranking
    names = np.array(X.columns)
    top_names, top_idx = select_top_features(importances, names)
    print(f"Top {len(top_names)} features:")
    for i, n in enumerate(top_names):
        print(f"  {i+1:2d}. {n:25s} {importances[top_idx[i]]:.4f}")
    plots.feature_importance(importances, names, len(top_names), args.outdir)

    # Re-split on top features
    df_top = df[list(top_names) + ["Class"]]
    _, _, _, X_tr_top, X_te_top, y_tr, y_te = prepare_splits(df_top)

    # Hyperparameter tuning
    print("\nTuning hyperparameters...")
    pg = None
    if args.param_grid:
        with open(args.param_grid) as f:
            pg = json.load(f)
        print(f"  Loaded grid from {args.param_grid}")
    best_params = tune_hyperparams(X_tr_top, y_tr, param_grid=pg, n_iter=args.n_iter)
    print(f"  Best: {best_params}\n")

    # SFS
    print(f"Running SFS (k=1..{args.max_sfs}) on {len(top_names)} features...")
    results = run_sfs(best_params, top_names, X_tr_top, X_te_top, y_tr, y_te,
                      args.outdir, max_k=args.max_sfs, overfit_gap=args.overfit_gap)

    for r in results:
        f = "  *** OVERFIT ***" if r["overfit"] else ""
        print(f"  k={r['k']}: {r['features']}  "
              f"Train={r['train_acc']:.4f} Test={r['test_acc']:.4f}{f}")

    # Plots
    print("\nGenerating plots...")
    plots.accuracy_curve(results, args.outdir)
    for r in results:
        plots.conf_matrix(y_te, r["y_pred"], le, r["k"], args.outdir)
        sel = r["features"]
        if r["k"] == 1:
            plots.histogram_1f(df, sel[0], classes, args.outdir)
        elif r["k"] == 2:
            plots.scatter_2f(df, sel, classes, args.outdir)
        elif r["k"] == 3:
            plots.scatter_3f(df, sel, classes, args.outdir)
        else:
            plots.pca_2d(X_te_top[sel].values, y_te, classes, le, r["k"], args.outdir)

    # Summary
    print(f"\n{'='*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'k':>3}  {'Train':>8}  {'Test':>8}  {'CV':>8}  {'Gap':>8}  {'Status':10}  Features")
    print("-" * 80)
    for r in results:
        g = r["train_acc"] - r["test_acc"]
        st = "OVERFIT" if r["overfit"] else "OK"
        print(f"{r['k']:3d}  {r['train_acc']:8.4f}  {r['test_acc']:8.4f}  "
              f"{r['cv_acc']:8.4f}  {g:8.4f}  {st:10s}  {r['features']}")

    ok = [r for r in results if not r["overfit"]]
    best = max(ok, key=lambda r: r["test_acc"]) if ok else max(results, key=lambda r: r["test_acc"])
    if not ok:
        print("\n  WARNING: All models show overfitting.")
    bk = best["k"]
    print(f"\n  BEST: k={bk}  Test={best['test_acc']:.4f}  Features={best['features']}")
    print(f"  Model: {os.path.join(args.outdir, f'xgb_{bk}feat.pkl')}")

    rows = [{k: v for k, v in r.items() if k != "y_pred"} for r in results]
    for r in rows:
        r["features"] = ", ".join(r["features"])
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "model_summary.csv"), index=False)
    print(f"  Summary: {os.path.join(args.outdir, 'model_summary.csv')}")
    print(f"\n  Total time: {(time()-t0)/60:.1f} min\n")


def cmd_run(args):
    """OP_ML run — full pipeline: compute OPs then train ML."""
    print("\n[OP_ML run] Step 1/2: Computing Order Parameters (Fortran backend)...")
    cmd_compute(args)

    op_csv = os.path.join(args.outdir, "OP.csv")
    if not os.path.isfile(op_csv):
        op_csv = os.path.join(args.outdir, "OP_local.csv")
    if not os.path.isfile(op_csv):
        print("Error: No OP output file found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[OP_ML run] Step 2/2: Training ML classifier...")
    args.data = op_csv
    ml_dir = os.path.join(args.outdir, "ml_results")
    args.outdir = ml_dir
    cmd_train(args)


def cmd_validate(args):
    """OP_ML validate — check CSV for NaN."""
    import pandas as pd
    from .validate import check_op_dataframe, ValidationError

    df = pd.read_csv(args.file)
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)
    try:
        check_op_dataframe(df, args.file)
        print(f"OK: {args.file} — {len(df)} rows, no NaN values found.")
    except ValidationError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


# ── Main entry point ─────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        prog="OP_ML",
        description="Order Parameter computation (Fortran+MPI) & ML phase classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
================================================================================
  COMMANDS
================================================================================

  OP_ML compute   Compute 383 order parameters from .lammpstrj trajectory files
  OP_ML train     Train XGBoost ML classifier on an existing OP CSV file
  OP_ML run       Full end-to-end pipeline: compute OPs then train ML
  OP_ML validate  Check a CSV file for NaN values (stops if found)

================================================================================
  OP COMPUTATION PARAMETERS  (for: compute, run)
================================================================================

  --traj DIR            Directory with .lammpstrj files (one per phase)
                        Filename = class label, e.g. Ice1c.lammpstrj -> "Ice1c"
  --rcut FLOAT          Neighbor cutoff in Angstroms (default: 4.5)
  --central-type INT    LAMMPS atom type for the central atom (e.g. 1 for O)
                        Auto-detected from trajectory if not given
  --neighbor-types LIST LAMMPS atom types for neighbor atoms, comma-separated
                        e.g. "1,2" for O and H.  Default: central type only
                        Examples:
                          Water O-O:    --central-type 1
                          CaCO3 Ca-Ca+C: --central-type 1 --neighbor-types 1,2
                          All atoms:    --central-type 1 --neighbor-types 1,2,3
  --op-type TYPE        Which OPs to output: avg, local, both (default: avg)
  --op-categories LIST  Which OP categories: B,D,F,I,Q,W,LQ,LW or "all"
                        e.g. "Q,W,LQ,LW" for Steinhardt only
                        e.g. "B" for bond-angle OPs only (default: all)
  --last-frames N       Process only the last N frames of each trajectory
                        e.g. --last-frames 10 for last 10 frames (default: all)

================================================================================
  ML TRAINING PARAMETERS  (for: train, run)
================================================================================

  --data CSV            Input OP CSV file with "Class" column (for train only)
  --max-sfs INT         Max features for Sequential Forward Selection (default: 6)
  --overfit-gap FLOAT   Train-test accuracy gap threshold (default: 0.05)
                        Models with gap > this are flagged as OVERFIT
  --n-iter INT          Hyperparameter search iterations (default: 25)
  --param-grid FILE     JSON file with custom hyperparameter grid, e.g.:
                        {"n_estimators":[100,500], "max_depth":[5,10]}
  --no-interactive      Skip all prompts, use defaults for missing parameters

================================================================================
  OUTPUT
================================================================================

  --outdir DIR          Output directory (default: results/)

  Output files produced:
    results/OP.csv                  Combined OPs with Class column
    results/ml_results/
      model_summary.csv             k, features, train/test/CV accuracy, overfit
      xgb_1feat.pkl ... xgb_6feat.pkl   Trained XGBoost models
      feature_importances.png       Top-N feature importance chart
      accuracy_curve.png            Train/test accuracy + overfitting gap
      hist_1feat.png                Per-class histogram (1-feature model)
      scatter_2feat.png             2D scatter plot (2-feature model)
      scatter_3feat.png             3D scatter plot (3-feature model)
      pca_4feat.png ...             PCA projections (4+ feature models)
      cm_1feat.png ... cm_6feat.png Confusion matrices

================================================================================
  EXAMPLES
================================================================================

  # Interactive mode (asks for types, cutoff, ML params):
  OP_ML compute --traj phases/ --outdir results/
  OP_ML train --data OP.csv --outdir results/

  # Water: O-O order parameters (type 1 = O)
  OP_ML compute --traj phases/ --central-type 1 --rcut 4.5

  # CaCO3: Ca as central, Ca+C as neighbors (type 1=Ca, 2=C)
  OP_ML compute --traj phases/ --central-type 1 --neighbor-types 1,2 --rcut 5.0

  # Only Steinhardt OPs (Q, W, LQ, LW):
  OP_ML compute --traj phases/ --central-type 1 --op-categories Q,W,LQ,LW

  # Only bond-angle OPs (B):
  OP_ML compute --traj phases/ --central-type 1 --op-categories B

  # Last 10 frames only (useful for testing or equilibrated portion):
  OP_ML compute --traj phases/ --central-type 1 --last-frames 10

  # Last 50 frames with MPI:
  mpiexec -n 8 OP_ML compute --traj phases/ --central-type 1 --last-frames 50

  # Full pipeline with all options:
  OP_ML run --traj phases/ --central-type 1 --rcut 4.5 --max-sfs 8 --no-interactive

  # MPI parallel (fast OP computation):
  mpiexec -n 8 OP_ML compute --traj phases/ --central-type 1 --rcut 5.0
  mpiexec -n 8 OP_ML run --traj phases/ --central-type 1 --rcut 4.5 --no-interactive

  # Validate CSV for NaN (stops with error if found):
  OP_ML validate --file OP.csv

  # Custom hyperparameter grid:
  OP_ML train --data OP.csv --param-grid my_grid.json --n-iter 50

================================================================================
  INSTALL
================================================================================

  pip install git+https://github.com/dikshaiisc/OrderParameter.git

  For OP computation, build the Fortran extension:
    cd $(python -c "import op_ml; import os; print(os.path.dirname(op_ml.__file__))")/fortran
    make

================================================================================
  NOTES
================================================================================

  - Atom selection uses LAMMPS integer types (not element names).
    The trajectory is scanned to show you: "Type 1 = O, Type 2 = H" etc.
  - --neighbor-types controls which atoms are used as neighbors in OP calc.
    Default: same as central type (e.g. O-O). Set to 1,2 for O+H neighbors.
  - --op-categories filters output columns. "all" computes everything (383 OPs).
    Use "Q,W" for Steinhardt only, "B" for bond-angle only, etc.
  - With MPI (mpiexec -n N), interactive prompts are disabled automatically.
    Provide --central-type, --rcut etc. via flags.
  - Without MPI, missing parameters are asked interactively.
  - NaN validation: the pipeline NEVER drops data. It stops with an error.
  - Phase clubbing: in interactive mode, you can merge phases.
""",
    )
    sub = pa.add_subparsers(dest="command", help="Available commands")

    # ── compute ──
    p_comp = sub.add_parser("compute",
        help="Compute order parameters from trajectory files (Fortran backend)")
    _add_op_args(p_comp)
    _add_out_args(p_comp)

    # ── train ──
    p_train = sub.add_parser("train",
        help="Train ML classifier on an existing OP CSV file")
    p_train.add_argument("--data", metavar="CSV", required=True,
                         help="OP CSV file with Class column")
    _add_ml_args(p_train)
    _add_out_args(p_train)

    # ── run ──
    p_run = sub.add_parser("run",
        help="Full pipeline: compute OPs then train ML (end-to-end)")
    _add_op_args(p_run)
    _add_ml_args(p_run)
    _add_out_args(p_run)

    # ── validate ──
    p_val = sub.add_parser("validate",
        help="Check a CSV file for NaN values")
    p_val.add_argument("--file", required=True, help="CSV file to validate")

    args = pa.parse_args()

    if args.command is None:
        pa.print_help()
        sys.exit(0)

    {"compute": cmd_compute, "train": cmd_train,
     "run": cmd_run, "validate": cmd_validate}[args.command](args)
