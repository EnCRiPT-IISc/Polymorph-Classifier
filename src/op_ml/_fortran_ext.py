"""Lazy loader for the compiled _op_fortran extension.

If the .so is missing, attempts to build it automatically by running make.
"""

from __future__ import annotations
import os
import sys
import subprocess
import numpy as np

_ext = None

_FORTRAN_DIR = os.path.join(os.path.dirname(__file__), "fortran")
_PKG_DIR = os.path.dirname(__file__)


def _auto_build():
    """Try to build the Fortran extension automatically."""
    makefile = os.path.join(_FORTRAN_DIR, "Makefile")
    if not os.path.isfile(makefile):
        return False

    print("[OP_ML] Fortran extension not found. Attempting automatic build...",
          flush=True)
    try:
        # Run f2py build
        result = subprocess.run(
            ["python3", "-m", "numpy.f2py", "-c",
             "consts.f90", "spherical_harmonics.f90",
             "order_parameters.f90", "op_frame_wrapper.f90",
             "--f90exec=mpiifx", "--f90flags=-O3",
             "-m", "_op_fortran", "--build-dir", "/tmp/f2py_build_op"],
            cwd=_FORTRAN_DIR,
            capture_output=True, text=True, timeout=300,
        )

        if result.returncode != 0:
            # Try with gfortran as fallback
            result = subprocess.run(
                ["python3", "-m", "numpy.f2py", "-c",
                 "consts.f90", "spherical_harmonics.f90",
                 "order_parameters.f90", "op_frame_wrapper.f90",
                 "--f90exec=gfortran", "--f90flags=-O3",
                 "-m", "_op_fortran", "--build-dir", "/tmp/f2py_build_op"],
                cwd=_FORTRAN_DIR,
                capture_output=True, text=True, timeout=300,
            )

        if result.returncode != 0:
            print(f"[OP_ML] Auto-build failed. Error:\n{result.stderr[-500:]}", flush=True)
            return False

        # Run meson build step
        build_dir = "/tmp/f2py_build_op"
        bbdir = os.path.join(build_dir, "bbdir")
        for fc_cc in [("mpiifx", "mpiicx"), ("gfortran", "gcc")]:
            env = os.environ.copy()
            env["FC"] = fc_cc[0]
            env["CC"] = fc_cc[1]
            r1 = subprocess.run(
                ["meson", "setup", bbdir, build_dir, "--wipe"],
                env=env, capture_output=True, text=True, timeout=120,
            )
            if r1.returncode == 0:
                r2 = subprocess.run(
                    ["meson", "compile", "-C", bbdir],
                    env=env, capture_output=True, text=True, timeout=300,
                )
                if r2.returncode == 0:
                    break

        # Copy .so to package directory
        import glob
        so_files = glob.glob(os.path.join(bbdir, "_op_fortran*.so"))
        if not so_files:
            # Maybe built directly in fortran/ dir
            so_files = glob.glob(os.path.join(_FORTRAN_DIR, "_op_fortran*.so"))
        if not so_files:
            so_files = glob.glob(os.path.join(build_dir, "_op_fortran*.so"))

        if so_files:
            import shutil
            dest = os.path.join(_PKG_DIR, os.path.basename(so_files[0]))
            shutil.copy2(so_files[0], dest)
            print(f"[OP_ML] Build successful! Extension installed to {dest}", flush=True)
            return True
        else:
            print("[OP_ML] Build completed but .so file not found.", flush=True)
            return False

    except FileNotFoundError as e:
        print(f"[OP_ML] Auto-build failed: {e}", flush=True)
        return False
    except subprocess.TimeoutExpired:
        print("[OP_ML] Auto-build timed out.", flush=True)
        return False


def _load():
    global _ext
    if _ext is not None:
        return _ext
    try:
        from op_ml import _op_fortran as ext
        _ext = ext
    except ImportError:
        # Try auto-build
        if _auto_build():
            try:
                # Reload after build
                import importlib
                import op_ml
                importlib.reload(op_ml)
                from op_ml import _op_fortran as ext
                _ext = ext
            except ImportError:
                pass

        if _ext is None:
            raise ImportError(
                "\n" + "=" * 60 + "\n"
                "  Fortran extension (_op_fortran) not available.\n"
                "  Auto-build was attempted but failed.\n\n"
                "  To build manually:\n"
                f"    cd {_FORTRAN_DIR}\n"
                "    make\n\n"
                "  Requirements: Fortran compiler (mpiifx or gfortran),\n"
                "  meson, ninja, numpy\n"
                + "=" * 60
            )
    return _ext


def init(n_mol: int):
    _load().init_op_calc(n_mol)


def compute_frame(
    r_central: np.ndarray,
    r_list: np.ndarray,
    mol_list: np.ndarray,
    box: np.ndarray,
    rcutsq: float,
) -> tuple:
    ext = _load()
    lat_vecs, rlat_vecs = build_lat_vecs(box)
    rc_f = np.asfortranarray(r_central.T)
    rl_f = np.asfortranarray(r_list.T)
    ml = np.ascontiguousarray(mol_list, dtype=np.int32)
    local_f, avg_f = ext.compute_frame(rc_f, rl_f, ml, lat_vecs, rlat_vecs, rcutsq)
    return local_f.T.copy(), avg_f.T.copy()


def build_lat_vecs(box: np.ndarray):
    lx, ly, lz = box
    lat = np.array([[lx, 0., 0.], [0., ly, 0.], [0., 0., lz]],
                   dtype=np.float64, order='F')
    rlat = np.asfortranarray(np.linalg.inv(lat))
    return lat, rlat
