"""Data loading, phase clubbing (interactive), and train/test split."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from .config import TEST_SIZE, RANDOM_STATE
from op_ml.validate import validate_or_exit


def load_and_validate(path: str) -> pd.DataFrame:
    """Load OP CSV and validate. Stops on NaN — never drops data."""
    df = pd.read_csv(path)
    # Only drop auto-generated pandas index columns (Unnamed:0 etc.)
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)
    # Validate — stops pipeline if any NaN found anywhere
    validate_or_exit(df, path)
    return df


# ── Phase clubbing ───────────────────────────────────────────────────

def ask_phase_clubbing(classes: list) -> dict:
    """Interactive: ask user whether to club phases.

    Returns mapping dict, e.g. {'Ice1c': 'Ice', 'Ice1h': 'Ice', 'Liquid': 'Liquid'}.
    """
    print(f"\n{'='*50}")
    print("  PHASE CONFIGURATION")
    print(f"{'='*50}")
    print(f"\n  Phases found in training data ({len(classes)}):\n")
    for i, c in enumerate(classes):
        print(f"    [{i+1}] {c}")

    print(f"\n  Options:")
    print(f"    1 — Classify ALL phases separately (default)")
    print(f"    2 — Club (merge) some phases together")

    choice = input("\n  Your choice [1/2]: ").strip()
    if choice != "2":
        print("  Using all phases separately.\n")
        return {c: c for c in classes}

    print(f"\n  Enter groups. Each group becomes one class label.")
    print(f"  Type phase numbers separated by commas, then name the group.")
    print(f"  Type 'done' when finished. Unassigned phases keep original name.\n")

    mapping = {}
    while True:
        raw = input("  Phases to merge (comma-separated numbers, or 'done'): ").strip()
        if raw.lower() == "done":
            break
        try:
            indices = [int(x.strip()) - 1 for x in raw.split(",")]
            selected = [classes[i] for i in indices]
        except (ValueError, IndexError):
            print("    Invalid input. Use numbers like: 1,2,3")
            continue
        name = input(f"  Group name for {selected}: ").strip()
        if not name:
            print("    Name cannot be empty.")
            continue
        for c in selected:
            mapping[c] = name
        print(f"    Merged {selected} -> '{name}'\n")

    for c in classes:
        if c not in mapping:
            mapping[c] = c

    print(f"\n  Final class mapping:")
    for orig, new in sorted(mapping.items()):
        tag = " (merged)" if orig != new else ""
        print(f"    {orig:15s} -> {new}{tag}")
    print()
    return mapping


def apply_clubbing(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    df = df.copy()
    df["Class"] = df["Class"].map(mapping)
    return df


def prepare_splits(df: pd.DataFrame):
    """Split into train/test. Returns X, y, le, X_tr, X_te, y_tr, y_te."""
    X = df.drop("Class", axis=1)
    le = LabelEncoder()
    y = le.fit_transform(df["Class"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y, shuffle=True,
    )
    return X, y, le, X_tr, X_te, y_tr, y_te
