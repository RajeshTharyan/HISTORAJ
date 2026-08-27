"""Row-range and pandas-query filters applied before plotting."""

from __future__ import annotations

import pandas as pd
from pandas.errors import UndefinedVariableError


class ObservationRangeError(ValueError):
    """Raised when the 1-based start/end observation fields are not a valid slice."""


class FilterQueryError(ValueError):
    """Raised when the optional pandas ``query`` condition cannot be applied."""


def parse_observation_range(
    start_str: str | None,
    end_str: str | None,
    n_rows: int,
) -> tuple[int, int]:
    """Return a 0-based half-open ``[start, end)`` slice.

    Empty fields mean "from the first row" and "through the last row".
    Inputs are 1-based, matching the Streamlit labels.
    """
    if n_rows < 0:
        raise ObservationRangeError("Row count cannot be negative.")

    start_idx = 0
    end_idx = n_rows
    start_raw = (start_str or "").strip()
    end_raw = (end_str or "").strip()

    try:
        if start_raw:
            start_idx = int(start_raw) - 1
        if end_raw:
            end_idx = int(end_raw)
    except ValueError as exc:
        raise ObservationRangeError(
            "Observation range must be integers (1-based)."
        ) from exc

    if not (0 <= start_idx < end_idx <= n_rows):
        raise ObservationRangeError("Invalid observation range.")
    return start_idx, end_idx


def apply_condition(df: pd.DataFrame, condition: str | None) -> pd.DataFrame:
    """Filter ``df`` with ``DataFrame.query``; blank conditions are a no-op."""
    expr = (condition or "").strip()
    if not expr:
        return df
    try:
        return df.query(expr)
    except (ValueError, SyntaxError, TypeError, KeyError, UndefinedVariableError) as exc:
        raise FilterQueryError(f"Invalid condition: {exc}") from exc


def apply_filters(
    df: pd.DataFrame,
    start_obs: str | None = None,
    end_obs: str | None = None,
    condition: str | None = None,
) -> pd.DataFrame:
    """Apply observation-range slice, then the optional query condition."""
    start_idx, end_idx = parse_observation_range(start_obs, end_obs, len(df))
    sliced = df.iloc[start_idx:end_idx]
    return apply_condition(sliced, condition)
