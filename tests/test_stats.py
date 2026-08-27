"""Tests for descriptive statistics (no GUI, no network)."""

import numpy as np
import pandas as pd
import pytest

from historaj.stats import (
    format_number,
    format_stats_markdown,
    format_stats_plot_text,
    is_numeric_series,
    summarize_numeric,
)


def test_is_numeric_series():
    assert is_numeric_series(pd.Series([1, 2, 3]))
    assert is_numeric_series(pd.Series([1.5, np.nan, 2.5]))
    assert not is_numeric_series(pd.Series(["a", "b"]))
    assert not is_numeric_series(pd.Series([True, False]))


def test_summarize_simple_sequence():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = summarize_numeric(series, show_percentiles=True)
    assert stats is not None
    assert stats.n_obs == 5
    assert stats.mean == pytest.approx(3.0)
    assert stats.median == pytest.approx(3.0)
    assert stats.std == pytest.approx(series.std())
    assert stats.valid_std is True
    assert [label for label, _ in stats.sd_markers] == [
        "-3sd",
        "-2sd",
        "-1sd",
        "+1sd",
        "+2sd",
        "+3sd",
    ]
    plus_one = dict(stats.sd_markers)["+1sd"]
    assert plus_one == pytest.approx(stats.mean + stats.std)
    assert set(stats.percentiles) == {"P5", "P10", "P25(Q1)", "P75(Q3)", "P90", "P95"}
    assert stats.percentiles["P25(Q1)"] == pytest.approx(series.quantile(0.25))


def test_summarize_drops_nans():
    series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    stats = summarize_numeric(series)
    assert stats is not None
    assert stats.n_obs == 3
    assert stats.mean == pytest.approx(3.0)


def test_summarize_empty_after_dropna():
    assert summarize_numeric(pd.Series([np.nan, np.nan])) is None
    assert summarize_numeric(pd.Series([], dtype=float)) is None


def test_constant_column_has_no_valid_std():
    stats = summarize_numeric(pd.Series([4.0, 4.0, 4.0]))
    assert stats is not None
    assert stats.std == pytest.approx(0.0)
    assert stats.valid_std is False
    assert stats.sd_markers == ()


def test_single_value_has_no_valid_std():
    stats = summarize_numeric(pd.Series([10.0]))
    assert stats is not None
    assert stats.n_obs == 1
    assert stats.valid_std is False
    assert not np.isfinite(stats.std) or stats.std == 0 or np.isnan(stats.std)


def test_percentiles_omitted_unless_requested():
    stats = summarize_numeric(pd.Series(range(20)), show_percentiles=False)
    assert stats is not None
    assert stats.percentiles == {}


def test_format_number_and_stats_text():
    assert format_number(1.234567) == "1.23457"
    assert format_number(float("nan")) == "N/A"
    assert format_number(None) == "N/A"

    stats = summarize_numeric(pd.Series([1.0, 2.0, 3.0]), show_percentiles=True)
    md = format_stats_markdown("x", stats)
    assert "Stats for x" in md
    assert "Percentiles" in md
    plot_text = format_stats_plot_text(stats)
    assert "Obs   : 3" in plot_text
    assert "P5" in plot_text
