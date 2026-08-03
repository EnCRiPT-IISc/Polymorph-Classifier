"""MPIOPCalculator — MPI-parallel order parameter calculator.

Uses LAMMPS atom types (integers) for central and neighbor selection.
Supports OP category filtering (B, D, F, I, Q, W, LQ, LW).
"""

from __future__ import annotations
import os
import time

import numpy as np
import pandas as pd

from ._trajectory import scan_first_frame, count_frames, read_frame, \
                          skip_frames, organise_coords
from ._fortran_ext import init, compute_frame as _fortran_compute
from ._constants import OP_COLUMNS, OP_COLUMNS_AVG


# ── OP category filtering ────────────────────────────────────────────

OP_CATEGORY_INDICES = {
    "B":  list(range(0, 42)),
    "D":  list(range(42, 67)),
    "F":  list(range(67, 342)),
    "I":  [342],
    "Q":  [343 + i * 4 + 0 for i in range(10)],
    "W":  [343 + i * 4 + 1 for i in range(10)],
    "LQ": [343 + i * 4 + 2 for i in range(10)],
    "LW": [343 + i * 4 + 3 for i in range(10)],
}
ALL_CATEGORIES = list(OP_CATEGORY_INDICES.keys())


def filter_op_columns(df, categories):
    """Keep only columns from selected OP categories."""
    if categories is None or "all" in [c.lower() for c in categories]:
        return df

    keep_idx = []
    for cat in categories:
        cu = cat.upper()
        if cu not in OP_CATEGORY_INDICES:
            raise ValueError(f"Unknown OP category '{cat}'. "
                             f"Choose from: {ALL_CATEGORIES} or 'all'")
        keep_idx.extend(OP_CATEGORY_INDICES[cu])

    keep_idx = sorted(set(keep_idx))
    op_cols = [c for c in df.columns if c not in ("mol_id", "Class")]
    keep = ["mol_id"] + [op_cols[i] for i in keep_idx]
    if "Class" in df.columns:
        keep.append("Class")
    return df[keep]


class MPIOPCalculator:
    """MPI-parallel OP calculator backed by compiled Fortran.

    Parameters
    ----------
    rcut : float
        Neighbor cutoff in Angstroms.
    central_type : int
        LAMMPS atom type for the central atom.
    neighbor_types : list of int or None
        LAMMPS atom types to use as neighbors.
        None = central type only (one atom per molecule).
    op_categories : list of str or None
        OP categories to include. None = all 383.
        Options: B, D, F, I, Q, W, LQ, LW, all
    """

    def __init__(self, rcut=4.5, central_type=1, neighbor_types=None,
                 op_categories=None, last_frames=None):
        self.rcut = rcut
        self.rcutsq = rcut * rcut
        self.central_type = central_type
        self.neighbor_types = neighbor_types
        self.op_categories = op_categories
        self.last_frames = last_frames   # None = all frames

    def compute_all(self, traj_path, out_prefix="OP", out_dir=".",
                    verbose=True, write_csv=True):
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        except ImportError:
            comm, rank, size = None, 0, 1

        meta = scan_first_frame(traj_path)
        n_frames = count_frames(traj_path, meta)
        nt = self.neighbor_types or [self.central_type]

        # Determine n_mol for Fortran init
        nt_set = set(self.neighbor_types or [self.central_type])
        mixed_atomic = (meta.atoms_per_mol == 1 and nt_set != {self.central_type})

        if meta.atoms_per_mol == 1:
            with open(traj_path, "r") as fh:
                result = read_frame(fh, meta)
            _, _, _, atm_types_0, _ = result
            if mixed_atomic:
                # Mixed neighbor types in atomic system: Fortran sees ALL selected types
                n_mol = int(np.sum(np.isin(atm_types_0, list(nt_set))))
                n_central_only = int(np.sum(atm_types_0 == self.central_type))
            else:
                n_mol = int(np.sum(atm_types_0 == self.central_type))
                n_central_only = n_mol
        else:
            n_mol = meta.n_molecules
            n_central_only = n_mol
            mixed_atomic = False

        # Determine which frames to process
        if self.last_frames is not None and self.last_frames < n_frames:
            start_frame = n_frames - self.last_frames
            use_n_frames = self.last_frames
        else:
            start_frame = 0
            use_n_frames = n_frames

        if rank == 0 and verbose:
            frame_info = f"{use_n_frames} frames" if use_n_frames == n_frames else \
                         f"last {use_n_frames} of {n_frames} frames"
            extra = ""
            if mixed_atomic:
                extra = f" (Fortran sees {n_mol} atoms, output filtered to {n_central_only})"
            print(f"[OP_ML] {frame_info}, {n_central_only} central atoms, "
                  f"{size} rank(s), rcut={self.rcut} A, "
                  f"central_type={self.central_type} ({meta.type_to_name.get(self.central_type,'?')}), "
                  f"neighbor_types={list(nt_set)}{extra}",
                  flush=True)

        init(n_mol)
        # Round-robin assignment over the selected frame range
        my_frames = list(range(start_frame + rank, start_frame + use_n_frames, size))

        local_rows, avg_rows = [], []
        t0 = time.time()

        with open(traj_path, "r") as fh:
            cur_frame = 0
            if my_frames:
                skip_frames(fh, my_frames[0], meta)
                cur_frame = my_frames[0]

            for target in my_frames:
                if cur_frame < target:
                    skip_frames(fh, target - cur_frame, meta)
                    cur_frame = target

                result = read_frame(fh, meta)
                cur_frame += 1
                if result is None:
                    break

                ts, box, mol_ids, atm_types, coords = result
                org_result = organise_coords(
                    coords, mol_ids, atm_types, meta,
                    self.central_type, self.neighbor_types,
                )

                # organise_coords returns 3 or 4 values
                if len(org_result) == 4:
                    r_central, r_list, mol_list, central_indices = org_result
                else:
                    r_central, r_list, mol_list = org_result
                    central_indices = None

                local_op, avg_op = _fortran_compute(
                    r_central, r_list, mol_list, box, self.rcutsq,
                )

                # For mixed atomic: filter to central type atoms only
                if central_indices is not None:
                    local_op = local_op[central_indices]
                    avg_op = avg_op[central_indices]

                n_out = local_op.shape[0]
                mol_id_col = np.arange(1, n_out + 1, dtype=np.int32)
                local_rows.append(pd.DataFrame(
                    np.column_stack([mol_id_col, local_op]),
                    columns=["mol_id"] + OP_COLUMNS,
                ))
                avg_rows.append(pd.DataFrame(
                    np.column_stack([mol_id_col, avg_op]),
                    columns=["mol_id"] + OP_COLUMNS_AVG,
                ))

                if verbose:
                    elapsed = time.time() - t0
                    done = target - start_frame + 1
                    print(f"  rank {rank}: frame {done}/{use_n_frames} "
                          f"(ts={ts}) {elapsed:.1f}s", flush=True)

        df_local = pd.concat(local_rows, ignore_index=True) if local_rows else \
                   pd.DataFrame(columns=["mol_id"] + OP_COLUMNS)
        df_avg = pd.concat(avg_rows, ignore_index=True) if avg_rows else \
                 pd.DataFrame(columns=["mol_id"] + OP_COLUMNS_AVG)
        df_local["mol_id"] = df_local["mol_id"].astype(int)
        df_avg["mol_id"] = df_avg["mol_id"].astype(int)

        if comm is not None and size > 1:
            all_local = comm.gather(df_local, root=0)
            all_avg = comm.gather(df_avg, root=0)
            if rank == 0:
                df_local = pd.concat(all_local, ignore_index=True)
                df_avg = pd.concat(all_avg, ignore_index=True)

        if rank == 0:
            df_local = filter_op_columns(df_local, self.op_categories)
            df_avg = filter_op_columns(df_avg, self.op_categories)

            if write_csv:
                os.makedirs(out_dir, exist_ok=True)
                df_local.to_csv(os.path.join(out_dir, f"{out_prefix}_unavg.csv"), index=False)
                df_avg.to_csv(os.path.join(out_dir, f"{out_prefix}_avg.csv"), index=False)
            if verbose:
                print(f"[OP_ML] Done in {time.time()-t0:.1f}s", flush=True)
            return df_local, df_avg
        return None, None

    def compute_batch(self, traj_paths, out_prefix="OP", out_dir=".",
                      output_type="avg", verbose=True):
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        except ImportError:
            rank = 0

        phase_dfs_local, phase_dfs_avg = [], []

        for traj_path in traj_paths:
            class_name = os.path.splitext(os.path.basename(traj_path))[0]
            if rank == 0 and verbose:
                print(f"\n{'='*50}\n  Phase: {class_name}  ({traj_path})\n{'='*50}",
                      flush=True)

            df_local, df_avg = self.compute_all(
                traj_path, out_prefix=f"_tmp_{class_name}",
                out_dir=out_dir, verbose=verbose, write_csv=False,
            )

            if rank == 0:
                if df_local is not None:
                    df_local["Class"] = class_name
                    phase_dfs_local.append(df_local)
                if df_avg is not None:
                    df_avg["Class"] = class_name
                    phase_dfs_avg.append(df_avg)

        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            if output_type in ("avg", "both") and phase_dfs_avg:
                df = pd.concat(phase_dfs_avg, ignore_index=True)
                path = os.path.join(out_dir, f"{out_prefix}.csv")
                df.to_csv(path, index=False)
                if verbose:
                    print(f"\n[OP_ML] Combined -> {path} ({len(df)} rows)", flush=True)
            if output_type in ("local", "both") and phase_dfs_local:
                df = pd.concat(phase_dfs_local, ignore_index=True)
                path = os.path.join(out_dir, f"{out_prefix}_local.csv")
                df.to_csv(path, index=False)

            if output_type == "avg":
                return pd.concat(phase_dfs_avg, ignore_index=True) if phase_dfs_avg else None
            elif output_type == "local":
                return pd.concat(phase_dfs_local, ignore_index=True) if phase_dfs_local else None
            else:
                return (pd.concat(phase_dfs_local, ignore_index=True) if phase_dfs_local else None,
                        pd.concat(phase_dfs_avg, ignore_index=True) if phase_dfs_avg else None)
        return None
