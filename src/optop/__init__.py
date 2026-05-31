"""OptOP — order-parameter computation (Fortran/MPI) + ML phase classification.

Quick API:
    from optop import MPIOPCalculator
    from optop.ml import run_training
"""
from .calculator import MPIOPCalculator

__version__ = "1.0.0"
__all__ = ["MPIOPCalculator"]
