"""Load CSV / Excel / Stata tables from a path or file-like object."""

from __future__ import annotations

from typing import BinaryIO, TextIO, Union

import numpy as np
import pandas as pd
from pandas.errors import ParserError

FileLike = Union[str, BinaryIO, TextIO]

FILE_TYPE_OPTIONS: tuple[str, ...] = ("CSV", "Excel", "Stata (.dta)")

_LOAD_ERRORS = (
    OSError,
    ValueError,
    UnicodeDecodeError,
    ImportError,
    ParserError,
    TypeError,
)


class FileLoadError(ValueError):
    """Raised when a selected file cannot be parsed as a table."""


def guess_file_type(filename: str) -> str:
    """Map a file name to one of ``FILE_TYPE_OPTIONS``."""
    lower = (filename or "").lower()
    if lower.endswith(".csv"):
        return "CSV"
    if lower.endswith((".xlsx", ".xls")):
        return "Excel"
    if lower.endswith(".dta"):
        return "Stata (.dta)"
    raise FileLoadError(f"Unsupported file name: {filename}")


def load_dataframe(file_obj: FileLike, file_type: str) -> pd.DataFrame:
    """Read ``file_obj`` according to the Streamlit file-type label."""
    if file_type not in FILE_TYPE_OPTIONS:
        raise FileLoadError(f"Unsupported file type: {file_type}")
    try:
        if file_type == "CSV":
            frame = pd.read_csv(file_obj)
        elif file_type == "Excel":
            frame = pd.read_excel(file_obj)
        else:
            frame = pd.read_stata(file_obj)
    except _LOAD_ERRORS as exc:
        raise FileLoadError(f"Failed to load file: {exc}") from exc
    if not isinstance(frame, pd.DataFrame):
        raise FileLoadError("File did not produce a table.")
    return frame


def numeric_column_names(df: pd.DataFrame) -> list[str]:
    """Column names whose dtypes are NumPy number subtypes."""
    return df.select_dtypes(include=[np.number]).columns.tolist()
