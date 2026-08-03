"""OP column name definitions — mirrors calc_op.f90 exactly."""

import numpy as np

PHI0 = 180.0 / 109.5   # Fortran: phi0 = 180._PR / 109.5_PR
STEINHARDT_L = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16]

_B_PHI = [0.0, 2.0 * np.pi / 3.0, np.pi / 2.0, np.pi / 3.0,
          np.pi / 4.0, np.pi / 5.0, np.pi / 6.0]

B_PARAMS = [(n1, n2, phi) for n1 in [1, 2] for n2 in [1, 2, 3] for phi in _B_PHI]
D_PARAMS = [(n1, n1, n2) for n1 in range(1, 6) for n2 in range(1, 6)]
F_PARAMS = [(n1, n2, a)
            for n1 in range(1, 6) for n2 in range(1, 6)
            for a in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0,
                      PHI0, 2*PHI0, 3*PHI0, 4*PHI0, 6*PHI0]]


def _r2u(val: float) -> str:
    return f"{val:.2f}".replace(".", "_")


def _make_cols(avg: bool) -> list:
    s = "avg" if avg else ""
    cols = []
    for (n1, n2, phi) in B_PARAMS:
        cols.append(f"B_{n1}_{n2}_{_r2u(phi)}{s}")
    for (na, nb, nc) in D_PARAMS:
        cols.append(f"D_{na}_{nb}_{nc}{s}")
    for (na, nb, a) in F_PARAMS:
        cols.append(f"F_{na}_{nb}_{_r2u(a)}{s}")
    cols.append(f"I{s}")
    for l in [2, 3, 4, 5, 6]:
        for t in ["Q", "W", "LQ", "LW"]:
            cols.append(f"{t}_{l}{s}")
    for l in [8, 10, 12, 14, 16]:
        for t in ["Q", "W", "LQ", "LW"]:
            cols.append(f"{t}_{l}{s}")
    assert len(cols) == 383
    return cols


OP_COLUMNS     = _make_cols(avg=False)
OP_COLUMNS_AVG = _make_cols(avg=True)
