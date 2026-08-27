"""Descriptive statistics used by HistoraJ plots and the Streamlit expanders."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PERCENTILE_SPEC: tuple[tuple[float, str], ...] = (
    (0.05, "P5"),
    (0.10, "P10"),
    (0.25, "P25(Q1)"),
    (0.75, "P75(Q3)"),
    (0.90, "P90"),
    (0.95, "P95"),
)

SD_LABELS: tuple[str, ...] = ("-3sd", "-2sd", "-1sd", "+1sd", "+2sd", "+3sd")
SD_MULTIPLIERS: tuple[int, ...] = (-3, -2, -1, 1, 2, 3)


@dataclass(frozen=True)
class ColumnStats:
    """Summary of one numeric column after NaNs are dropped."""

    n_obs: int
    mean: float
    median: float
    std: float
    skew: float
    kurtosis: float
    valid_std: bool
    sd_markers: tuple[tuple[str, float], ...] = ()
    percentiles: dict[str, float] = field(default_factory=dict)


def is_numeric_series(series: pd.Series) -> bool:
    """Return True when the series dtype is a NumPy number subtype."""
    return np.issubdtype(series.dtype, np.number)


def _is_finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(value))
    except TypeError:
        return False


def summarize_numeric(series: pd.Series, show_percentiles: bool = False) -> ColumnStats | None:
    """Return descriptive stats for a numeric series, or None if nothing remains.

    Standard deviation uses pandas' default (sample, ``ddof=1``). Kurtosis is
    pandas' excess kurtosis. ``valid_std`` is True only when std is finite and
    strictly positive — a single value or a constant column cannot support a
    normal overlay or SD markers.
    """
    data = series.dropna()
    if data.empty:
        return None

    mean = float(data.mean())
    median = float(data.median())
    std = float(data.std())
    skew = float(data.skew())
    kurtosis = float(data.kurt())
    valid_std = _is_finite_number(std) and std > 0

    markers: tuple[tuple[str, float], ...] = ()
    if valid_std:
        markers = tuple(
            (label, mean + multiplier * std)
            for label, multiplier in zip(SD_LABELS, SD_MULTIPLIERS)
        )

    percentiles: dict[str, float] = {}
    if show_percentiles:
        quantiles = data.quantile([q for q, _ in PERCENTILE_SPEC])
        percentiles = {
            label: float(quantiles.iloc[i]) for i, (_, label) in enumerate(PERCENTILE_SPEC)
        }

    return ColumnStats(
        n_obs=int(len(data)),
        mean=mean,
        median=median,
        std=std,
        skew=skew,
        kurtosis=kurtosis,
        valid_std=valid_std,
        sd_markers=markers,
        percentiles=percentiles,
    )


def format_number(value: object) -> str:
    """Format a statistic the same way the UI and plot annotation do."""
    if value is None or not _is_finite_number(value):
        return "N/A"
    return f"{float(value):.5f}"


def format_stats_markdown(col_name: str, stats: ColumnStats) -> str:
    """Markdown block shown in the Streamlit expander."""
    lines = [
        f"**Stats for {col_name}:**",
        f"- Obs    : {stats.n_obs}",
        f"- Mean   : {format_number(stats.mean)}",
        f"- Median : {format_number(stats.median)}",
        f"- StdDev : {format_number(stats.std)}",
        f"- Skew   : {format_number(stats.skew)}",
        f"- Kurtos : {format_number(stats.kurtosis)}",
    ]
    if stats.valid_std:
        for label, value in stats.sd_markers:
            lines.append(f"- {label:<6}: {format_number(value)}")
    if stats.percentiles:
        lines.append("**Percentiles:**")
        for label, value in stats.percentiles.items():
            lines.append(f"- {label:<8}: {format_number(value)}")
    return "\n".join(lines)


def format_stats_plot_text(stats: ColumnStats) -> str:
    """Compact stats box drawn on the matplotlib axes."""
    parts = [
        f"Obs   : {stats.n_obs}",
        f"Mean  : {format_number(stats.mean)}",
        f"Median: {format_number(stats.median)}",
        f"StdDev: {format_number(stats.std)}",
        f"Skew  : {format_number(stats.skew)}",
        f"Kurtos: {format_number(stats.kurtosis)}",
    ]
    if stats.percentiles:
        parts.append("---- Percentiles ----")
        for label, value in stats.percentiles.items():
            parts.append(f"{label:<8}: {format_number(value)}")
    return "\n".join(parts)
