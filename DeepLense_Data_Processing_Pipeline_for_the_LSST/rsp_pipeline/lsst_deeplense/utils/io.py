"""
io.py
-----
Convenience functions for saving and loading lens-candidate tables and
pre-fetched numpy cutout arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def save_candidates_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """
    Save a lens-candidate DataFrame to CSV.

    Required columns: ``ra``, ``dec``.  Optional: ``label``, ``objectId``.

    Parameters
    ----------
    df : pd.DataFrame
    path : str or Path
    overwrite : bool
        If ``False`` (default) and ``path`` already exists, raise
        ``FileExistsError``.

    Returns
    -------
    Path  – the resolved path where the file was written.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists.  Pass overwrite=True to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_candidates_csv(
    path: Union[str, Path],
    required_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load a lens-candidate CSV and validate required columns.

    Parameters
    ----------
    path : str or Path
    required_cols : list[str], optional
        Extra columns that must be present (beyond ``'ra'`` and ``'dec'``).

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {path}")

    df = pd.read_csv(path)
    must_have = {"ra", "dec"} | set(required_cols or [])
    missing = must_have - set(df.columns)
    if missing:
        raise ValueError(f"CSV '{path}' is missing columns: {missing}")
    return df
