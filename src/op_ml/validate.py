"""Data validation — stops the pipeline on bad data. Never drops or removes data."""

import sys
import pandas as pd


class ValidationError(Exception):
    pass


def check_op_dataframe(df: pd.DataFrame, label: str = "OP data"):
    """Validate an OP DataFrame. Raises ValidationError on any problem.

    IMPORTANT: This function does NOT drop or modify any data.
    If NaN is found, it stops and tells the user to fix the input file.
    """
    # 1. NaN in numeric columns
    numeric = df.select_dtypes(include="number")
    nan_counts = numeric.isna().sum()
    bad = nan_counts[nan_counts > 0]
    if len(bad) > 0:
        msg = (f"\n{'='*60}\n"
               f"  ERROR: NaN values detected in {label}!\n"
               f"{'='*60}\n"
               f"  Columns with NaN values:\n")
        for col, cnt in bad.items():
            msg += f"    {col}: {cnt} NaN values\n"
        msg += (f"\n  Total NaN cells: {int(nan_counts.sum())}\n"
                f"\n  The pipeline CANNOT proceed with NaN values.\n"
                f"  Please check your input file and fix the issue.\n"
                f"  Common causes:\n"
                f"    - Molecule with zero neighbors (bad cutoff)\n"
                f"    - Corrupted trajectory frame\n"
                f"    - Incomplete OP computation\n"
                f"{'='*60}")
        raise ValidationError(msg)

    # 2. NaN in Class column
    if "Class" in df.columns:
        nan_class = df["Class"].isna().sum()
        if nan_class > 0:
            raise ValidationError(
                f"\n{'='*60}\n"
                f"  ERROR: {nan_class} rows have NaN in the 'Class' column.\n"
                f"  Every row must have a phase label.\n"
                f"  Please check your input file.\n"
                f"{'='*60}"
            )

    # 3. Empty file
    if len(df) == 0:
        raise ValidationError(
            f"\n{'='*60}\n"
            f"  ERROR: {label} is empty (0 rows).\n"
            f"  Please check your input file.\n"
            f"{'='*60}"
        )

    return True


def validate_or_exit(df: pd.DataFrame, label: str = "OP data"):
    """Validate and exit with clear message on failure. Never modifies data."""
    try:
        check_op_dataframe(df, label)
    except ValidationError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
