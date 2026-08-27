"""Core analysis helpers for the HistoraJ Streamlit app.

The Streamlit UI lives in ``historajstreamlit.py``. Import from this package
when you want the parsers, filters, stats, or plotting without a GUI.
"""

from historaj.filters import (
    FilterQueryError,
    ObservationRangeError,
    apply_condition,
    apply_filters,
    parse_observation_range,
)
from historaj.io import (
    FILE_TYPE_OPTIONS,
    FileLoadError,
    guess_file_type,
    load_dataframe,
    numeric_column_names,
)
from historaj.layout import subplot_grid
from historaj.stats import ColumnStats, is_numeric_series, summarize_numeric

__all__ = [
    "FILE_TYPE_OPTIONS",
    "ColumnStats",
    "FileLoadError",
    "FilterQueryError",
    "ObservationRangeError",
    "apply_condition",
    "apply_filters",
    "guess_file_type",
    "is_numeric_series",
    "load_dataframe",
    "numeric_column_names",
    "parse_observation_range",
    "subplot_grid",
    "summarize_numeric",
]
