"""Figure grid geometry for one histogram per selected column."""

from __future__ import annotations

import math


def subplot_grid(n_plots: int, max_cols: int = 2) -> tuple[int, int]:
    """Return ``(rows, cols)`` for ``n_plots`` axes, at most ``max_cols`` wide.

    This matches the original Streamlit layout: one column for a single
    series, two columns once there are two or more series.
    """
    if n_plots < 0:
        raise ValueError("n_plots cannot be negative.")
    if max_cols < 1:
        raise ValueError("max_cols must be at least 1.")
    if n_plots == 0:
        return (0, 0)
    cols = min(n_plots, max_cols)
    rows = math.ceil(n_plots / cols)
    return rows, cols
