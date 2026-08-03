"""OP_ML — Order Parameter computation (Fortran+MPI) and ML phase classification.

Install from GitHub:
    pip install git+https://github.com/dikshaiisc/OrderParameter.git

Usage:
    OP_ML --help
    OP_ML compute --traj phases/ --outdir results/
    OP_ML train --data OP.csv --outdir results/
    OP_ML run --traj phases/ --outdir results/
"""

__version__ = "1.1.0"

from .op_calculator import MPIOPCalculator, ALL_CATEGORIES
from ._constants import OP_COLUMNS, OP_COLUMNS_AVG
