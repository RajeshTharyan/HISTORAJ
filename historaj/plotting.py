"""Matplotlib histogram + normal overlay. No Streamlit imports."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from historaj.stats import (
    ColumnStats,
    format_stats_plot_text,
    is_numeric_series,
    summarize_numeric,
)

BIN_COUNT = 100
YAXIS_CHOICES: tuple[str, ...] = ("density", "frequency", "fraction", "percentage")
SD_COLORS = {
    "-3sd": "red",
    "+3sd": "red",
    "-2sd": "orange",
    "+2sd": "orange",
    "-1sd": "green",
    "+1sd": "green",
}


def configure_matplotlib() -> None:
    """Apply the figure defaults used by the Streamlit app."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.family": "sans-serif",
            "font.size": 10,
        }
    )


def _placeholder(ax, message: str, col_name: str, title_text: str, yaxis_choice: str) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize=10,
        color="red",
    )
    ax.set_title(title_text if title_text else col_name)
    ax.set_xlabel(col_name)
    ax.set_ylabel(yaxis_choice.capitalize())


def _hist_weights(values: np.ndarray, yaxis_choice: str):
    density_param = False
    weights = None
    if yaxis_choice == "density":
        density_param = True
    elif yaxis_choice == "fraction":
        weights = np.ones_like(values) / len(values)
    elif yaxis_choice == "percentage":
        weights = np.ones_like(values) / len(values) * 100
    return density_param, weights


def _overlay_normal(
    ax,
    values: np.ndarray,
    stats: ColumnStats,
    yaxis_choice: str,
    bins: np.ndarray,
    plot_xmin: float,
    plot_xmax: float,
) -> None:
    x_norm = np.linspace(plot_xmin, plot_xmax, 200)
    pdf = norm.pdf(x_norm, stats.mean, stats.std)
    bin_width_approx = (bins[-1] - bins[0]) / (len(bins) - 1) if len(bins) > 1 else 1.0

    if yaxis_choice == "density":
        ax.plot(x_norm, pdf, "k", linewidth=2)
    elif yaxis_choice == "fraction":
        ax.plot(x_norm, pdf * bin_width_approx, "k", linewidth=2)
    elif yaxis_choice == "percentage":
        ax.plot(x_norm, pdf * bin_width_approx * 100, "k", linewidth=2)
    else:
        ax.plot(x_norm, pdf * len(values) * bin_width_approx, "k", linewidth=2)


def draw_histogram(
    ax,
    data: pd.Series,
    stats: ColumnStats,
    yaxis_choice: str,
    col_name: str,
) -> None:
    """Draw histogram, optional normal curve, SD lines, and the stats box."""
    if yaxis_choice not in YAXIS_CHOICES:
        raise ValueError(f"Unsupported y-axis type: {yaxis_choice}")

    values = np.asarray(data.dropna(), dtype=float)
    density_param, weights = _hist_weights(values, yaxis_choice)
    _counts, bins, _patches = ax.hist(
        values,
        bins=BIN_COUNT,
        density=density_param,
        weights=weights,
        alpha=0.7,
        edgecolor="black",
    )

    xmin, xmax_hist = ax.get_xlim()
    data_min, data_max = float(values.min()), float(values.max())
    plot_xmin = min(
        xmin,
        data_min - 0.1 * abs(data_min) if stats.valid_std else data_min,
    )
    plot_xmax = max(
        xmax_hist,
        data_max + 0.1 * abs(data_max) if stats.valid_std else data_max,
    )
    ax.set_xlim(plot_xmin, plot_xmax)

    if stats.valid_std:
        _overlay_normal(ax, values, stats, yaxis_choice, bins, plot_xmin, plot_xmax)
        for lab, marker_val in stats.sd_markers:
            ax.axvline(marker_val, linestyle="dashed", linewidth=1, color=SD_COLORS[lab])

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        marker_positions = [val for (_, val) in stats.sd_markers]
        marker_labels = [lab for (lab, _) in stats.sd_markers]
        ax_top.set_xticks(marker_positions)
        ax_top.set_xticklabels(marker_labels, fontweight="bold", fontsize=9)
        ax_top.tick_params(axis="x", pad=5)
        for tick, lab in zip(ax_top.get_xticklabels(), marker_labels):
            tick.set_color(SD_COLORS[lab])

    ax.set_xlabel(f"{col_name}: {yaxis_choice.capitalize()} Plot")
    ax.set_ylabel(yaxis_choice.capitalize())
    ax.set_title("")
    ax.text(
        0.98,
        0.95,
        format_stats_plot_text(stats),
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="lightgrey"),
    )


def plot_column(
    ax,
    series: pd.Series,
    col_name: str,
    yaxis_choice: str,
    show_percentiles: bool = False,
    title_text: str = "",
) -> ColumnStats | None:
    """Plot one column. Returns stats when there is numeric data, else None."""
    if yaxis_choice not in YAXIS_CHOICES:
        raise ValueError(f"Unsupported y-axis type: {yaxis_choice}")

    if not is_numeric_series(series):
        _placeholder(
            ax,
            f"Column {col_name} is not numeric.",
            col_name,
            title_text,
            yaxis_choice,
        )
        return None

    data = series.dropna()
    if data.empty:
        _placeholder(
            ax,
            f"No data available for {col_name} after filtering.",
            col_name,
            title_text,
            yaxis_choice,
        )
        return None

    stats = summarize_numeric(data, show_percentiles=show_percentiles)
    if stats is None:
        _placeholder(
            ax,
            f"No data available for {col_name} after filtering.",
            col_name,
            title_text,
            yaxis_choice,
        )
        return None

    draw_histogram(ax, data, stats, yaxis_choice, col_name)
    return stats
