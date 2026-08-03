"""LAMMPS .lammpstrj trajectory reader.

Handles:
  - Orthogonal and triclinic boxes
  - Any ATOMS column order (detects from ITEM: ATOMS header)
  - Both wrapped (x y z) and unwrapped (xu yu zu) coordinates
  - Charge column (q) present or absent
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


HEADER_LINES = 9   # lines per frame before the atom block


@dataclass
class FrameMeta:
    n_atoms: int
    n_molecules: int
    atoms_per_mol: int
    atom_names: list
    atom_types: list
    type_to_name: dict = field(default_factory=dict)
    lines_per_frame: int = 0
    triclinic: bool = False
    # Column indices (0-based) in the ATOMS line
    col_id: int = 0
    col_mol: int = 1
    col_type: int = 2
    col_element: int = 3
    col_x: int = 4
    col_y: int = 5
    col_z: int = 6


def _parse_atoms_header(header_line: str):
    """Parse 'ITEM: ATOMS id mol type element x y z' to get column indices."""
    # Remove 'ITEM: ATOMS' prefix
    parts = header_line.strip().split()
    if len(parts) < 3:
        # fallback
        return 0, 1, 2, 3, 4, 5, 6

    cols = parts[2:]  # skip 'ITEM:' and 'ATOMS'
    col_map = {name: i for i, name in enumerate(cols)}

    col_id = col_map.get("id", 0)
    col_mol = col_map.get("mol", col_map.get("mol_id", 1))
    col_type = col_map.get("type", 2)
    col_element = col_map.get("element", 3)

    # Prefer unwrapped coords (xu,yu,zu), fall back to (x,y,z)
    if "xu" in col_map:
        col_x, col_y, col_z = col_map["xu"], col_map["yu"], col_map["zu"]
    elif "x" in col_map:
        col_x, col_y, col_z = col_map["x"], col_map["y"], col_map["z"]
    else:
        # Guess: last 3 columns are coords
        col_x, col_y, col_z = len(cols) - 3, len(cols) - 2, len(cols) - 1

    return col_id, col_mol, col_type, col_element, col_x, col_y, col_z


def scan_first_frame(path: str) -> FrameMeta:
    """Read first frame to detect system structure."""
    with open(path, "r") as fh:
        fh.readline()                     # ITEM: TIMESTEP
        fh.readline()                     # timestep
        fh.readline()                     # ITEM: NUMBER OF ATOMS
        n_atoms = int(fh.readline())

        box_header = fh.readline()        # ITEM: BOX BOUNDS ...
        triclinic = "xy" in box_header

        # Read box lines
        fh.readline()   # xlo xhi [xy]
        fh.readline()   # ylo yhi [xz]
        fh.readline()   # zlo zhi [yz]

        atoms_header = fh.readline()      # ITEM: ATOMS ...
        ci, cm, ct, ce, cx, cy, cz = _parse_atoms_header(atoms_header)

        mol_ids, atom_names, atom_types = [], [], []
        for _ in range(n_atoms):
            p = fh.readline().split()
            mol_ids.append(int(p[cm]))
            atom_types.append(int(p[ct]))
            atom_names.append(p[ce])

    first_mol = mol_ids[0]
    atoms_per_mol = sum(1 for m in mol_ids if m == first_mol)
    n_molecules = n_atoms // atoms_per_mol

    mol1_names = [atom_names[i] for i, m in enumerate(mol_ids) if m == first_mol]
    mol1_types = [atom_types[i] for i, m in enumerate(mol_ids) if m == first_mol]

    type_to_name = {}
    for t, n in zip(atom_types, atom_names):
        if t not in type_to_name:
            type_to_name[t] = n

    return FrameMeta(
        n_atoms=n_atoms,
        n_molecules=n_molecules,
        atoms_per_mol=atoms_per_mol,
        atom_names=mol1_names,
        atom_types=mol1_types,
        type_to_name=type_to_name,
        lines_per_frame=HEADER_LINES + n_atoms,
        triclinic=triclinic,
        col_id=ci, col_mol=cm, col_type=ct, col_element=ce,
        col_x=cx, col_y=cy, col_z=cz,
    )


def count_frames(path: str, meta: FrameMeta) -> int:
    with open(path, "rb") as fh:
        total_lines = sum(1 for _ in fh)
    return total_lines // meta.lines_per_frame


def read_frame(fh, meta: FrameMeta):
    """Read one frame. Returns (timestep, box, mol_ids, atom_types, coords) or None.

    For triclinic boxes, box is still (lx, ly, lz) — the orthogonal extents.
    Tilt factors are ignored for the neighbor cutoff (approximation).
    """
    line = fh.readline()
    if not line:
        return None

    ts = int(fh.readline())
    fh.readline()                          # ITEM: NUMBER OF ATOMS
    fh.readline()                          # n_atoms
    fh.readline()                          # ITEM: BOX BOUNDS ...

    # Read 3 box lines — handle both orthogonal (2 vals) and triclinic (3 vals)
    bx = fh.readline().split()
    by = fh.readline().split()
    bz = fh.readline().split()

    x_lo, x_hi = float(bx[0]), float(bx[1])
    y_lo, y_hi = float(by[0]), float(by[1])
    z_lo, z_hi = float(bz[0]), float(bz[1])

    # For triclinic, LAMMPS prints (lo, hi, tilt). The actual box extent is hi-lo.
    # Tilt factors shift the box but don't change the orthogonal lengths.
    box = np.array([x_hi - x_lo, y_hi - y_lo, z_hi - z_lo])

    fh.readline()                          # ITEM: ATOMS ...

    n = meta.n_atoms
    mol_ids   = np.empty(n, dtype=np.int32)
    atm_types = np.empty(n, dtype=np.int32)
    coords    = np.empty((n, 3), dtype=np.float64)

    cm, ct = meta.col_mol, meta.col_type
    cx, cy, cz = meta.col_x, meta.col_y, meta.col_z

    for i in range(n):
        p = fh.readline().split()
        mol_ids[i]   = int(p[cm])
        atm_types[i] = int(p[ct])
        coords[i, 0] = float(p[cx])
        coords[i, 1] = float(p[cy])
        coords[i, 2] = float(p[cz])

    return ts, box, mol_ids, atm_types, coords


def skip_frames(fh, n_frames: int, meta: FrameMeta):
    skip = n_frames * meta.lines_per_frame
    for _ in range(skip):
        fh.readline()


def organise_coords(coords, mol_ids, atm_types, meta, central_type, neighbor_types=None):
    """Extract central atom positions and neighbor list from raw atom data.

    Handles:
      - Molecular systems (water): multiple atoms per molecule
      - Atomic systems (CaCO3): each atom is its own molecule (atoms_per_mol=1)
    """
    if neighbor_types is None:
        neighbor_types = [central_type]

    neigh_set = set(neighbor_types)

    if meta.atoms_per_mol == 1:
        # ── Atomic system: each atom is its own "molecule" ──────────
        # For the Fortran backend, mol_list values must be in 1..n_central.
        # The connection matrix is (n_central x n_central).
        #
        # Strategy: central atoms = all atoms of central_type.
        # For same-type neighbors: straightforward 1..n_central mapping.
        # For mixed-type neighbors: include them in r_list but assign
        # each non-central neighbor a mol_id that maps to a central atom.
        # We assign each non-central neighbor to the central atom with
        # the same sequential position (wrapping). The Fortran uses
        # ConnectionMatrix(mol1, mol_list(atm)) to decide whether to
        # include that neighbor — since we set mol_list = mol1 for
        # all non-central neighbors, they will always be "connected"
        # to themselves, meaning every central atom will consider every
        # non-central neighbor (distance filtering happens via rij_m).
        #
        # Actually, the Fortran checks ConnectionMatrix(mol1, mol2)
        # where mol2 = mol_list(neighbor). If mol2 == mol1, then
        # ConnectionMatrix(mol1, mol1) is always True (self-connected).
        # So if we set all non-central neighbors' mol_list = mol1 for
        # each mol1... that doesn't work because mol_list is per-atom,
        # not per-mol1.
        #
        # Cleanest approach: set non-central neighbor mol_list values
        # to 1 (or any valid central index). ConnectionMatrix(mol1, 1)
        # will be True if mol1 and central_atom_1 are within rcut.
        # This means non-central neighbors are only considered if the
        # first central atom is within rcut of mol1 — which is wrong.
        #
        # Correct approach: DON'T use mixed neighbor types with the
        # Fortran connection-matrix backend for atomic systems.
        # Instead, include ALL selected types in BOTH r_central and r_list.
        # Compute OPs for all of them, then filter output to central type only.

        central_mask = atm_types == central_type
        r_central_only = coords[central_mask].copy()
        n_central = r_central_only.shape[0]

        if neigh_set == {central_type}:
            mol_list = np.arange(1, n_central + 1, dtype=np.int32)
            return r_central_only, r_central_only.copy(), mol_list

        # Mixed neighbor types: include ALL selected types as "molecules"
        # for the Fortran backend. Filter output later to central type only.
        all_mask = np.isin(atm_types, list(neigh_set))
        r_all_selected = coords[all_mask].copy()
        n_all = r_all_selected.shape[0]
        mol_list = np.arange(1, n_all + 1, dtype=np.int32)

        # Track which of the selected atoms are central type (for output filtering)
        selected_types = atm_types[all_mask]
        central_indices = np.where(selected_types == central_type)[0]

        # Return r_central = ALL selected (Fortran treats them all as molecules)
        # The caller must filter output rows to central_indices only
        return r_all_selected, r_all_selected.copy(), mol_list, central_indices

    else:
        # ── Molecular system ────────────────────────────────────────
        n_mol = meta.n_molecules
        apm = meta.atoms_per_mol

        central_pos = None
        for i, t in enumerate(meta.atom_types):
            if t == central_type:
                central_pos = i
                break
        if central_pos is None:
            avail = {t: meta.type_to_name.get(t, "?")
                     for t in sorted(set(meta.atom_types))}
            raise ValueError(
                f"Central atom type {central_type} not found in molecule.\n"
                f"  Available types: {avail}"
            )

        r_all = np.empty((n_mol, apm, 3), dtype=np.float64)
        slot = np.zeros(n_mol, dtype=np.int32)
        for i in range(len(mol_ids)):
            m = mol_ids[i] - 1
            r_all[m, slot[m]] = coords[i]
            slot[m] += 1

        r_central = r_all[:, central_pos, :].copy()

        if neigh_set == {central_type}:
            mol_list = np.arange(1, n_mol + 1, dtype=np.int32)
            return r_central, r_central.copy(), mol_list

        neigh_positions = [i for i, t in enumerate(meta.atom_types)
                          if t in neigh_set]
        n_neigh_per_mol = len(neigh_positions)
        n_list = n_mol * n_neigh_per_mol
        r_list = np.empty((n_list, 3), dtype=np.float64)
        mol_list = np.empty(n_list, dtype=np.int32)

        idx = 0
        for mol in range(n_mol):
            for pos in neigh_positions:
                r_list[idx] = r_all[mol, pos]
                mol_list[idx] = mol + 1
                idx += 1

        return r_central, r_list, mol_list
