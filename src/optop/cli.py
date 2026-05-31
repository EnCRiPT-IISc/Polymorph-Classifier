"""OptOP unified command-line interface.

    OptOP compute  — compute order parameters from .lammpstrj trajectories
    OptOP train    — train an XGBoost phase classifier on an OP CSV
    OptOP run      — compute OPs then train (end-to-end)

MPI:  prefix with `mpiexec -n N` to parallelise the (compute) stage.
"""
from __future__ import annotations
import argparse, glob, json, os, sys


# ── small parsers ────────────────────────────────────────────────────────
def _int_list(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _str_list(s):
    return [x.strip() for x in s.split(",") if x.strip() != ""]


def _mass_map(s):
    """'1:15.999,2:1.008,3:16.043' -> {1:15.999, 2:1.008, 3:16.043}"""
    if not s:
        return None
    out = {}
    for tok in s.split(","):
        t, m = tok.split(":")
        out[int(t)] = float(m)
    return out


def _rank():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except Exception:
        return 0


# ── compute ───────────────────────────────────────────────────────────────
def cmd_compute(a):
    from .calculator import MPIOPCalculator
    traj = a.traj
    if os.path.isdir(traj):
        files = sorted(glob.glob(os.path.join(traj, "*.lammpstrj")))
    else:
        files = [traj]
    if not files:
        print(f"Error: no .lammpstrj files in {traj}", file=sys.stderr); sys.exit(1)

    calc = MPIOPCalculator(
        rcut=a.rcut, central_type=a.central_type,
        neighbor_types=_int_list(a.neighbor_types) if a.neighbor_types else None,
        op_categories=_str_list(a.op_categories) if a.op_categories else None,
        last_frames=a.last_frames, com=a.com, masses_map=_mass_map(a.masses),
    )
    if _rank() == 0:
        os.makedirs(a.outdir, exist_ok=True)
        print(f"[OptOP compute] {len(files)} phase(s); central_type={a.central_type} "
              f"neighbors={a.neighbor_types or a.central_type} com={a.com} "
              f"rcut={a.rcut} ops={a.op_categories or 'all'} "
              f"last_frames={a.last_frames}", flush=True)
    calc.compute_batch(files, out_prefix=a.out_prefix, out_dir=a.outdir,
                       output_type=a.op_type)
    if _rank() == 0:
        print(f"[OptOP compute] wrote {os.path.join(a.outdir, a.out_prefix + '.csv')}",
              flush=True)


# ── train ───────────────────────────────────────────────────────────────--
def cmd_train(a):
    from .ml import run_training
    pg = None
    if a.param_grid:
        pg = json.load(open(a.param_grid))
    run_training(
        a.data, a.outdir, tag=a.tag or os.path.basename(a.outdir.rstrip("/")),
        feature_mode=a.feature_mode, max_top=a.max_top, imp_threshold=a.imp_threshold,
        param_grid=pg, n_iter=a.n_iter, cv=a.cv, overfit_gap=a.overfit_gap,
        max_sfs=a.max_sfs, phases=_str_list(a.phases) if a.phases else None,
    )


def cmd_run(a):
    cmd_compute(a)
    if _rank() != 0:
        return
    a.data = os.path.join(a.outdir, a.out_prefix + ".csv")
    a.outdir = os.path.join(a.outdir, "ml")
    cmd_train(a)


# ── argument wiring ────────────────────────────────────────────────────────
def _add_compute_args(p):
    p.add_argument("--traj", required=True, help="trajectory file or directory of *.lammpstrj (filename = Class)")
    p.add_argument("--central-type", type=int, default=1, help="LAMMPS atom type of the central atom (e.g. 1=O)")
    p.add_argument("--neighbor-types", default=None, help="comma list of neighbour atom types, e.g. 1,2,3 (default: central only)")
    p.add_argument("--com", action="store_true", help="COM-COM: reduce each molecule to its center of mass")
    p.add_argument("--masses", default=None, help="COM masses 'type:mass,...' (else auto from element symbols)")
    p.add_argument("--rcut", type=float, default=4.5, help="neighbour cutoff (Angstrom)")
    p.add_argument("--op-categories", default=None, help="OP categories: B,D,F,I,Q,W,LQ,LW or 'all' (default all)")
    p.add_argument("--op-type", choices=["avg", "local", "both"], default="avg", help="output averaged/local/both")
    p.add_argument("--last-frames", type=int, default=None, help="use only the last N frames per trajectory")
    p.add_argument("--out-prefix", default="OP", help="output CSV prefix (default OP)")
    p.add_argument("--outdir", default="results", help="output directory")


def _add_train_args(p):
    p.add_argument("--feature-mode", choices=["fixed", "cumulative"], default="cumulative",
                   help="top-feature selection: fixed count or cumulative-importance")
    p.add_argument("--max-top", type=int, default=20, help="max top features kept (cap)")
    p.add_argument("--imp-threshold", type=float, default=0.90, help="cumulative-importance cutoff (cumulative mode)")
    p.add_argument("--param-grid", default=None, help="JSON file with a custom hyperparameter grid")
    p.add_argument("--n-iter", type=int, default=25, help="RandomizedSearchCV iterations")
    p.add_argument("--cv", type=int, default=5, help="cross-validation folds")
    p.add_argument("--max-sfs", type=int, default=6, help="max features for Sequential Forward Selection")
    p.add_argument("--overfit-gap", type=float, default=0.05, help="train-test gap flagged as overfit")
    p.add_argument("--phases", default=None, help="comma list of Class labels to train on (default: all)")
    p.add_argument("--tag", default=None, help="label used in plot titles")


def main(argv=None):
    pa = argparse.ArgumentParser(prog="OptOP",
        description="Order-parameter computation (Fortran/MPI) + ML phase classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # O-O Steinhardt OPs for every phase in a folder, last 50 frames:
  OptOP compute --traj phases/ --central-type 1 --op-categories Q,W,LQ,LW --last-frames 50 --outdir out/

  # O with O,H,methane(M) neighbours (molecular convention):
  mpiexec -n 48 OptOP compute --traj phases/ --central-type 1 --neighbor-types 1,2,3 --rcut 3.5 --outdir out/

  # COM-COM order parameters (masses auto-detected from element symbols):
  OptOP compute --traj phases/ --central-type 1 --com --outdir out/
  # COM with explicit masses:
  OptOP compute --traj phases/ --central-type 1 --com --masses 1:15.999,2:1.008,3:16.043 --outdir out/

  # train a classifier (cumulative-importance feature selection) on chosen phases:
  OptOP train --data out/OP.csv --outdir ml/ --feature-mode cumulative --phases Ice1c,Ice1h,S1,S2,SH

  # end-to-end:
  OptOP run --traj phases/ --central-type 1 --op-categories Q,W,LQ,LW --last-frames 50 --outdir out/
""")
    sub = pa.add_subparsers(dest="command")
    pc = sub.add_parser("compute", help="compute order parameters"); _add_compute_args(pc)
    pt = sub.add_parser("train", help="train ML classifier")
    pt.add_argument("--data", required=True, help="OP CSV with a 'Class' column")
    pt.add_argument("--outdir", default="ml_results"); _add_train_args(pt)
    pr = sub.add_parser("run", help="compute then train"); _add_compute_args(pr); _add_train_args(pr)

    a = pa.parse_args(argv)
    if not a.command:
        pa.print_help(); sys.exit(0)
    {"compute": cmd_compute, "train": cmd_train, "run": cmd_run}[a.command](a)


if __name__ == "__main__":
    main()
