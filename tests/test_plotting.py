"""Tests for subplot grid geometry and matplotlib drawing (Agg backend)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from historaj.layout import subplot_grid
from historaj.plotting import BIN_COUNT, draw_histogram, plot_column
from historaj.stats import summarize_numeric


def test_subplot_grid_matches_original_layout():
    assert subplot_grid(0) == (0, 0)
    assert subplot_grid(1) == (1, 1)
    assert subplot_grid(2) == (1, 2)
    assert subplot_grid(3) == (2, 2)
    assert subplot_grid(4) == (2, 2)
    assert subplot_grid(5) == (3, 2)


def test_subplot_grid_rejects_bad_inputs():
    with pytest.raises(ValueError):
        subplot_grid(-1)
    with pytest.raises(ValueError):
        subplot_grid(3, max_cols=0)


def test_plot_column_numeric_draws_histogram_and_overlay():
    fig, ax = plt.subplots()
    series = pd.Series(np.linspace(0, 10, 200))
    stats = plot_column(ax, series, "x", "density", show_percentiles=True)
    assert stats is not None
    assert stats.n_obs == 200
    bars = [p for p in ax.patches if p.get_width() > 0]
    assert len(bars) == BIN_COUNT
    # Histogram + normal overlay (at least one Line2D beyond the axes spines).
    assert len(ax.lines) >= 1
    plt.close(fig)


def test_plot_column_frequency_and_percentage():
    series = pd.Series(np.random.default_rng(0).normal(size=80))
    for yaxis in ("frequency", "fraction", "percentage"):
        fig, ax = plt.subplots()
        stats = plot_column(ax, series, "z", yaxis)
        assert stats is not None
        assert ax.get_ylabel() == yaxis.capitalize()
        plt.close(fig)


def test_plot_column_non_numeric_placeholder():
    fig, ax = plt.subplots()
    stats = plot_column(ax, pd.Series(["a", "b"]), "label", "density")
    assert stats is None
    texts = [t.get_text() for t in ax.texts]
    assert any("not numeric" in text for text in texts)
    plt.close(fig)


def test_plot_column_all_nan_placeholder():
    fig, ax = plt.subplots()
    stats = plot_column(ax, pd.Series([np.nan, np.nan]), "x", "density")
    assert stats is None
    texts = [t.get_text() for t in ax.texts]
    assert any("No data available" in text for text in texts)
    plt.close(fig)


def test_draw_histogram_rejects_unknown_yaxis():
    fig, ax = plt.subplots()
    series = pd.Series([1.0, 2.0, 3.0])
    stats = summarize_numeric(series)
    with pytest.raises(ValueError, match="Unsupported y-axis"):
        draw_histogram(ax, series, stats, "bogus", "x")
    plt.close(fig)
