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


def read_frame(fh, meta: FrameMeta, structure: str = "auto"):
    """Read one frame. Returns (timestep, box, mol_ids, atom_types, coords) or None.

    ``structure`` selects how the periodic box / PBC are built:
      - "auto"         : triclinic tilt is applied iff the dump header carries
                         tilt factors (``ITEM: BOX BOUNDS xy xz yz ...``);
                         otherwise the box is treated as orthogonal.
      - "orthorhombic" : tilt factors are IGNORED — the raw (lo, hi) bounds of
                         each box line are used as the orthogonal edge lengths
                         (lx = xhi - xlo, ...).  This reproduces the original
                         reference code/hydrate_*.f90, which read only the two
                         lo/hi numbers and used a diagonal lattice even for the
                         triclinic sH structure.  Use this to match features_SC.
      - "triclinic"    : the LAMMPS bounding box is converted back to the true
                         cell edges and the tilt factors (xy, xz, yz) are kept,
                         giving the physically-exact minimum image for sloped
                         cells.  (For an orthogonal dump this is identical to
                         "orthorhombic".)

    box is returned as (lx, ly, lz, xy, xz, yz); tilt factors are zero unless a
    triclinic treatment is in effect.
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

    # Decide whether to apply a triclinic treatment for THIS frame.
    #   "triclinic"    -> always (when tilt columns are present)
    #   "auto"         -> only if the header advertised tilt (meta.triclinic)
    #   "orthorhombic" -> never (ignore tilt; use raw lo/hi bounds)
    has_tilt_cols = len(bx) >= 3 and len(by) >= 3 and len(bz) >= 3
    if structure == "triclinic":
        use_triclinic = has_tilt_cols
    elif structure == "orthorhombic":
        use_triclinic = False
    else:  # "auto"
        use_triclinic = meta.triclinic and has_tilt_cols

    # For a true triclinic cell LAMMPS dumps the *bounding* box: each line is
    # (lo_bound, hi_bound, tilt). Convert the bounding box back to the real box
    # edges so the lattice vectors are exact.
    #   xlo = xlo_bound - MIN(0, xy, xz, xy+xz);  xhi = xhi_bound - MAX(...)
    #   ylo = ylo_bound - MIN(0, yz);             yhi = yhi_bound - MAX(0, yz)
    if use_triclinic:
        xy, xz, yz = float(bx[2]), float(by[2]), float(bz[2])
        x_lo -= min(0.0, xy, xz, xy + xz)
        x_hi -= max(0.0, xy, xz, xy + xz)
        y_lo -= min(0.0, yz)
        y_hi -= max(0.0, yz)
    else:
        # orthogonal treatment: raw lo/hi bounds, zero tilt (legacy behaviour)
        xy = xz = yz = 0.0

    # box = (lx, ly, lz, xy, xz, yz); tilt factors are zero for orthogonal cells.
    box = np.array([x_hi - x_lo, y_hi - y_lo, z_hi - z_lo, xy, xz, yz])

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
    """Extract central-atom positions and the neighbour list, purely by ATOM TYPE.

    Type-based selection works uniformly for:
      - pure molecular systems (water: select every O atom)
      - heterogeneous molecular systems (gas hydrates: water O/H + a methane
        bead) where molecules have *different* atom counts
      - atomic systems (each atom is its own molecule)

    The central atoms are every atom whose type == central_type.  When the
    neighbour types differ from the central type, every selected atom (central
    + neighbour types) is handed to the Fortran backend as a "molecule"; the
    caller filters the output rows back to the central atoms using the returned
    ``central_indices``.  This avoids any assumption of a fixed number of atoms
    per molecule.
    """
    if neighbor_types is None:
        neighbor_types = [central_type]

    neigh_set = set(neighbor_types)

    central_mask = atm_types == central_type
    r_central_only = coords[central_mask].copy()
    n_central = r_central_only.shape[0]
    if n_central == 0:
        avail = {t: meta.type_to_name.get(t, "?")
                 for t in sorted(set(atm_types.tolist()))}
        raise ValueError(
            f"Central atom type {central_type} not present in frame.\n"
            f"  Available types: {avail}"
        )

    if neigh_set == {central_type}:
        mol_list = np.arange(1, n_central + 1, dtype=np.int32)
        return r_central_only, r_central_only.copy(), mol_list

    # ── Mixed neighbour types (e.g. O-O,H,M) — molecular convention ──────
    # Reproduces code/hydrate_OHM.f90:
    #   * every molecule has a "centre" (the central-type atom if the molecule
    #     contains one — water O — otherwise its first atom — guest M);
    #   * molecules that contain the central type are ordered FIRST (1..n_cen),
    #     guests after; n_cen = number of central atoms;
    #   * the connection matrix is built over ALL molecule centres (r_mol), so a
    #     central O sees a guest as a neighbour when their centres are < rcut;
    #   * r_list holds every atom of a selected neighbour type, tagged with its
    #     molecule index, and is the neighbour pool for the local Steinhardt.
    # Returns a 5-tuple so the caller routes to compute_frame_partial.
    unique_mols = np.unique(mol_ids)                      # sorted == appearance order
    n_total = unique_mols.shape[0]
    mol_of_atom = np.searchsorted(unique_mols, mol_ids)   # 0-based molecule index per atom

    # default centre = first atom of each molecule; override with central atom
    first_idx = np.full(n_total, -1, dtype=np.int64)
    for i in range(mol_of_atom.shape[0] - 1, -1, -1):     # last write wins -> lowest index
        first_idx[mol_of_atom[i]] = i
    center_idx = first_idx.copy()
    has_central = np.zeros(n_total, dtype=bool)
    cen_atom_idx = np.where(central_mask)[0]
    cen_mol = mol_of_atom[cen_atom_idx]
    center_idx[cen_mol] = cen_atom_idx
    has_central[cen_mol] = True

    # order: central-containing molecules first, then guests
    order = np.concatenate([np.where(has_central)[0], np.where(~has_central)[0]])
    n_cen = int(has_central.sum())
    new_index = np.empty(n_total, dtype=np.int64)
    new_index[order] = np.arange(n_total)

    r_mol = coords[center_idx[order]].copy()              # centres in new order
    r_central = r_mol[:n_cen].copy()                      # central atoms (water O) = r1

    sel = np.isin(atm_types, list(neigh_set))
    r_list = coords[sel].copy()
    mol_list = (new_index[mol_of_atom[sel]] + 1).astype(np.int32)   # 1-based molecule idx
    return r_central, r_mol, r_list, mol_list, n_cen


# ── Center-of-mass (COM) support ─────────────────────────────────────────
# Minimal periodic table (g/mol) for auto mass inference from element symbols.
ATOMIC_MASSES = {
    "H": 1.008, "D": 2.014, "He": 4.0026, "Li": 6.94, "Be": 9.0122,
    "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085,
    "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098,
    "Ca": 40.078, "Fe": 55.845, "Cu": 63.546, "Zn": 65.38, "Br": 79.904,
    "I": 126.90, "M": 16.043, "CH4": 16.043,   # M / CH4 united-atom methane
}


def _clean_symbol(name: str) -> str:
    """Normalise an element/atom label to a periodic-table key."""
    s = "".join(name.split()).strip()
    if s in ATOMIC_MASSES:
        return s
    # strip trailing digits/charges, capitalise first letter
    core = "".join(ch for ch in s if ch.isalpha())
    if core in ATOMIC_MASSES:
        return core
    if core[:2].capitalize() in ATOMIC_MASSES:
        return core[:2].capitalize()
    return core[:1].upper()


def build_masses_by_type(meta, masses_map=None):
    """Return {atom_type: mass}. `masses_map` (type->mass) overrides; otherwise
    the mass is inferred from the element symbol via ATOMIC_MASSES."""
    masses_map = masses_map or {}
    out = {}
    for t, name in meta.type_to_name.items():
        if t in masses_map:
            out[t] = float(masses_map[t])
        else:
            out[t] = ATOMIC_MASSES.get(_clean_symbol(name))
    return out


def _com_per_molecule(coords, mol_idx, mass, n_mol, lat_vecs, rlat_vecs):
    """Mass-weighted COM of every molecule, with minimum-image unwrapping
    relative to each molecule's first atom (handles PBC-split molecules)."""
    first = np.full(n_mol, -1, dtype=np.int64)
    for i in range(mol_idx.shape[0] - 1, -1, -1):
        first[mol_idx[i]] = i
    refpos = coords[first[mol_idx]]
    d = coords - refpos
    frac = d @ rlat_vecs.T
    frac -= np.round(frac)
    unwrapped = refpos + frac @ lat_vecs.T
    com = np.zeros((n_mol, 3)); msum = np.zeros(n_mol)
    np.add.at(com, mol_idx, unwrapped * mass[:, None])
    np.add.at(msum, mol_idx, mass)
    return com / msum[:, None]


def organise_coords_com(coords, mol_ids, atm_types, central_type, neighbor_types,
                        masses_by_type, lat_vecs, rlat_vecs):
    """COM-COM organisation: each molecule is reduced to its center of mass.

    central molecules  = molecules containing >=1 atom of `central_type`
    neighbour molecules = molecules containing any `neighbor_types` atom
    Returns the same shapes as organise_coords: a 3-tuple (COM-COM, same set)
    or a 5-tuple (mixed molecule sets -> compute_frame_partial)."""
    if neighbor_types is None:
        neighbor_types = [central_type]
    neigh_set = set(neighbor_types)

    miss = [t for t, m in masses_by_type.items() if m is None]
    if miss:
        raise ValueError(f"No mass for atom type(s) {miss}. "
                         f"Pass --masses TYPE:MASS,... for COM mode.")
    mass = np.array([masses_by_type[int(t)] for t in atm_types], dtype=np.float64)

    unique_mols = np.unique(mol_ids)
    n_total = unique_mols.shape[0]
    mol_of_atom = np.searchsorted(unique_mols, mol_ids)
    com_all = _com_per_molecule(coords, mol_of_atom, mass, n_total, lat_vecs, rlat_vecs)

    has_central = np.zeros(n_total, dtype=bool)
    has_central[mol_of_atom[atm_types == central_type]] = True
    has_neigh = np.zeros(n_total, dtype=bool)
    for t in neigh_set:
        has_neigh[mol_of_atom[atm_types == t]] = True

    cen_mols = np.where(has_central)[0]
    if cen_mols.size == 0:
        raise ValueError(f"No molecule contains central type {central_type}.")
    r_central = com_all[cen_mols].copy()

    if neigh_set == {central_type} or np.array_equal(np.where(has_neigh)[0], cen_mols):
        # COM-COM, identical molecule set
        mol_list = np.arange(1, r_central.shape[0] + 1, dtype=np.int32)
        return r_central, r_central.copy(), mol_list

    # mixed molecule sets: central molecules first, neighbour-only molecules after
    neigh_only = np.where(has_neigh & ~has_central)[0]
    order = np.concatenate([cen_mols, neigh_only])
    n_cen = cen_mols.shape[0]
    r_mol = com_all[order].copy()
    # r_list = one COM per neighbour molecule (central + neighbour-only)
    r_list = r_mol.copy()
    mol_list = np.arange(1, r_mol.shape[0] + 1, dtype=np.int32)
    return r_central, r_mol, r_list, mol_list, n_cen
