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
                          skip_frames, organise_coords, organise_coords_com, \
                          build_masses_by_type
from ._fortran_ext import init, compute_frame as _fortran_compute, \
                          compute_frame_partial as _fortran_compute_partial, \
                          compute_local_range as _fortran_local_range, \
                          compute_avg_range as _fortran_avg_range, \
                          build_lat_vecs
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
                 op_categories=None, last_frames=None, com=False, masses_map=None,
                 structure="auto", atom_ranks=1):
        self.rcut = rcut
        self.rcutsq = rcut * rcut
        self.central_type = central_type
        self.neighbor_types = neighbor_types
        self.op_categories = op_categories
        self.last_frames = last_frames   # None = all frames
        self.com = com                   # COM-COM mode (one COM per molecule)
        self.masses_map = masses_map     # {atom_type: mass} overrides for COM
        if structure not in ("auto", "orthorhombic", "triclinic"):
            raise ValueError(f"structure must be auto/orthorhombic/triclinic, got {structure!r}")
        self.structure = structure       # PBC treatment (see _trajectory.read_frame)
        self.atom_ranks = max(1, int(atom_ranks))  # MPI ranks per frame-group (2D decomp)

    def compute_all(self, traj_path, out_prefix="OP", out_dir=".",
                    verbose=True, write_csv=True):
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        except ImportError:
            comm, rank, size = None, 0, 1

        # 2D (frame x atom/molecule) decomposition is only meaningful with >1
        # rank; otherwise fall through to the (frame-only) path below.
        if self.atom_ranks > 1 and comm is not None and size > 1:
            return self._compute_all_2d(comm, rank, size, traj_path,
                                        out_prefix, out_dir, verbose, write_csv)

        meta = scan_first_frame(traj_path)
        n_frames = count_frames(traj_path, meta)
        nt = self.neighbor_types or [self.central_type]

        # Determine the OP-array size for Fortran init — purely type-based, so
        # it is robust to heterogeneous molecules (gas hydrates: water O/H +
        # methane).  In BOTH cases the OP arrays / averaging set are sized to
        # the central molecules (n_central_only); for mixed neighbour types the
        # larger connection matrix is built inside compute_frame_partial from
        # the per-molecule centres.
        nt_set = set(self.neighbor_types or [self.central_type])
        mixed_atomic = (nt_set != {self.central_type})

        with open(traj_path, "r") as fh:
            _, _, mol_ids_0, atm_types_0, _ = read_frame(fh, meta)
        n_total_mol = int(np.unique(mol_ids_0).shape[0])

        masses_by_type = None
        if self.com:
            masses_by_type = build_masses_by_type(meta, self.masses_map)
            # central molecules = molecules containing the central type
            cen_mask = np.isin(mol_ids_0, mol_ids_0[atm_types_0 == self.central_type])
            n_central_only = int(np.unique(mol_ids_0[cen_mask]).shape[0])
        else:
            n_central_only = int(np.sum(atm_types_0 == self.central_type))
        n_mol = n_central_only

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
                extra = (f" (molecular: connection matrix over {n_total_mol} "
                         f"molecule centres, OPs for {n_central_only} central)")
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

                result = read_frame(fh, meta, structure=self.structure)
                cur_frame += 1
                if result is None:
                    break

                ts, box, mol_ids, atm_types, coords = result
                if self.com:
                    lat_vecs, rlat_vecs = build_lat_vecs(box)
                    org_result = organise_coords_com(
                        coords, mol_ids, atm_types,
                        self.central_type, self.neighbor_types,
                        masses_by_type, lat_vecs, rlat_vecs,
                    )
                else:
                    org_result = organise_coords(
                        coords, mol_ids, atm_types, meta,
                        self.central_type, self.neighbor_types,
                    )

                # 3-tuple -> same-type neighbours (O-O); 5-tuple -> mixed
                # neighbour types via the molecular (hydrate) convention.
                if len(org_result) == 5:
                    r_central, r_mol, r_list, mol_list, _n_cen = org_result
                    local_op, avg_op = _fortran_compute_partial(
                        r_central, r_mol, r_list, mol_list, box, self.rcutsq,
                    )
                else:
                    r_central, r_list, mol_list = org_result
                    local_op, avg_op = _fortran_compute(
                        r_central, r_list, mol_list, box, self.rcutsq,
                    )

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

    # ── 2D (frame x atom/molecule) MPI decomposition ────────────────────────
    @staticmethod
    def _unpack_org(org):
        """Map organise_coords output to a uniform 5-tuple for the 2D kernels.
        OO/COM-COM (3-tuple) -> r_mol == r_central, n_mol == n_cen."""
        if len(org) == 5:
            r_central, r_mol, r_list, mol_list, n_cen = org
        else:
            r_central, r_list, mol_list = org
            r_mol = r_central
            n_cen = r_central.shape[0]
        return r_central, r_mol, r_list, mol_list, n_cen

    def _compute_all_2d(self, comm, rank, size, traj_path,
                        out_prefix, out_dir, verbose, write_csv):
        """2D decomposition: COMM_WORLD is split into G frame-groups, each of P
        = ``self.atom_ranks`` ranks (so size = G*P, P must divide size).  Frames
        are distributed round-robin across the G groups; within a group the P
        ranks split the central molecules and cooperatively compute each frame
        via compute_local_range -> Allreduce -> compute_avg_range -> Allreduce.
        The per-molecule result is bit-identical to the serial computation."""
        from mpi4py import MPI

        P = self.atom_ranks
        if size % P != 0:
            # Collective misconfiguration — every rank sees it, so every rank
            # raises (raising on rank 0 alone would deadlock the others).
            raise ValueError(f"--atom-ranks ({P}) must divide the total number "
                             f"of MPI ranks ({size}).")
        G = size // P
        group_id   = rank // P          # which frame-group (0..G-1)
        group_rank = rank % P           # position within the group (0..P-1)
        group_comm = comm.Split(color=group_id, key=group_rank)

        # ---- metadata (rank 0 counts frames, broadcasts) ----
        meta = scan_first_frame(traj_path)
        n_frames = count_frames(traj_path, meta) if rank == 0 else None
        n_frames = comm.bcast(n_frames, root=0)

        nt_set = set(self.neighbor_types or [self.central_type])
        mixed_atomic = (nt_set != {self.central_type})

        with open(traj_path, "r") as fh:
            _, _, mol_ids_0, atm_types_0, _ = read_frame(fh, meta, structure=self.structure)
        masses_by_type = None
        if self.com:
            masses_by_type = build_masses_by_type(meta, self.masses_map)
            cen_mask = np.isin(mol_ids_0, mol_ids_0[atm_types_0 == self.central_type])
            n_cen = int(np.unique(mol_ids_0[cen_mask]).shape[0])
        else:
            n_cen = int(np.sum(atm_types_0 == self.central_type))

        if self.last_frames is not None and self.last_frames < n_frames:
            start_frame, use_n_frames = n_frames - self.last_frames, self.last_frames
        else:
            start_frame, use_n_frames = 0, n_frames

        if rank == 0 and verbose:
            print(f"[OP_ML] 2D decomposition: {size} ranks = {G} frame-group(s) x {P} "
                  f"atom-rank(s); {use_n_frames} frames, {n_cen} central molecules, "
                  f"rcut={self.rcut} A, structure={self.structure}", flush=True)

        init(n_cen)

        # molecule slice owned by this rank within its group (1-based, inclusive)
        bounds = np.linspace(0, n_cen, P + 1).astype(int)
        mol_lo, mol_hi = int(bounds[group_rank]) + 1, int(bounds[group_rank + 1])

        # frames handled by THIS group (round-robin over G groups)
        my_frames = list(range(start_frame + group_id, start_frame + use_n_frames, G))

        local_rows, avg_rows = [], []
        t0 = time.time()
        fh = open(traj_path, "r") if group_rank == 0 else None
        cur_frame = 0
        if fh is not None and my_frames:
            skip_frames(fh, my_frames[0], meta)
            cur_frame = my_frames[0]

        for target in my_frames:
            # group-root reads + organises the frame, broadcasts to the group
            if group_rank == 0:
                if cur_frame < target:
                    skip_frames(fh, target - cur_frame, meta)
                    cur_frame = target
                result = read_frame(fh, meta, structure=self.structure)
                cur_frame += 1
                ts, box, mol_ids, atm_types, coords = result
                if self.com:
                    lat_vecs, rlat_vecs = build_lat_vecs(box)
                    org = organise_coords_com(coords, mol_ids, atm_types,
                                              self.central_type, self.neighbor_types,
                                              masses_by_type, lat_vecs, rlat_vecs)
                else:
                    org = organise_coords(coords, mol_ids, atm_types, meta,
                                          self.central_type, self.neighbor_types)
                payload = (box, self._unpack_org(org), ts)
            else:
                payload = None
            box, org5, ts = group_comm.bcast(payload, root=0)
            r_central, r_mol, r_list, mol_list, n_cen_f = org5

            lat_vecs, rlat_vecs = build_lat_vecs(box)

            # Stage 1: local OPs + q_lm for this rank's molecule slice
            local, qlm, qnorm = _fortran_local_range(
                mol_lo, mol_hi, r_central, r_mol, r_list, mol_list,
                lat_vecs, rlat_vecs, self.rcutsq)
            group_comm.Allreduce(MPI.IN_PLACE, local, op=MPI.SUM)
            group_comm.Allreduce(MPI.IN_PLACE, qlm,   op=MPI.SUM)
            group_comm.Allreduce(MPI.IN_PLACE, qnorm, op=MPI.SUM)

            # Stage 2: averaged OPs for this rank's slice, given full state
            _, avg = _fortran_avg_range(
                mol_lo, mol_hi, r_mol, lat_vecs, rlat_vecs, self.rcutsq,
                local, qlm, qnorm)
            group_comm.Allreduce(MPI.IN_PLACE, avg, op=MPI.SUM)

            if group_rank == 0:
                local_op, avg_op = local.T, avg.T          # (n_cen, 383)
                mol_id_col = np.arange(1, local_op.shape[0] + 1, dtype=np.int32)
                local_rows.append(pd.DataFrame(
                    np.column_stack([mol_id_col, local_op]),
                    columns=["mol_id"] + OP_COLUMNS))
                avg_rows.append(pd.DataFrame(
                    np.column_stack([mol_id_col, avg_op]),
                    columns=["mol_id"] + OP_COLUMNS_AVG))
                if verbose:
                    done = target - start_frame + 1
                    print(f"  group {group_id}: frame {done}/{use_n_frames} "
                          f"(ts={ts}) {time.time()-t0:.1f}s", flush=True)
        if fh is not None:
            fh.close()
        group_comm.Free()

        df_local = pd.concat(local_rows, ignore_index=True) if local_rows else \
                   pd.DataFrame(columns=["mol_id"] + OP_COLUMNS)
        df_avg = pd.concat(avg_rows, ignore_index=True) if avg_rows else \
                 pd.DataFrame(columns=["mol_id"] + OP_COLUMNS_AVG)
        df_local["mol_id"] = df_local["mol_id"].astype(int)
        df_avg["mol_id"] = df_avg["mol_id"].astype(int)

        all_local = comm.gather(df_local, root=0)
        all_avg = comm.gather(df_avg, root=0)
        if rank == 0:
            df_local = pd.concat(all_local, ignore_index=True)
            df_avg = pd.concat(all_avg, ignore_index=True)
            df_local = filter_op_columns(df_local, self.op_categories)
            df_avg = filter_op_columns(df_avg, self.op_categories)
            if write_csv:
                os.makedirs(out_dir, exist_ok=True)
                df_local.to_csv(os.path.join(out_dir, f"{out_prefix}_unavg.csv"), index=False)
                df_avg.to_csv(os.path.join(out_dir, f"{out_prefix}_avg.csv"), index=False)
            if verbose:
                print(f"[OP_ML] Done (2D) in {time.time()-t0:.1f}s", flush=True)
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
