"""HistoraJ Streamlit UI.

Core parsing, filtering, stats, and plotting live in the ``historaj`` package
so they can be imported and tested without a browser session.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from historaj.filters import FilterQueryError, ObservationRangeError, apply_filters
from historaj.io import (
    FILE_TYPE_OPTIONS,
    FileLoadError,
    guess_file_type,
    load_dataframe,
    numeric_column_names,
)
from historaj.layout import subplot_grid
from historaj.plotting import YAXIS_CHOICES, configure_matplotlib, plot_column
from historaj.stats import format_stats_markdown

APP_INTRO = """
## What this app does
Historaj helps you quickly analyze and visualize the distribution of numeric data. For each selected column in your dataset, it generates a histogram showing the data distribution, an overlaid normal distribution curve (if standard deviation is valid), vertical lines marking standard deviations from the mean (-3sd to +3sd), and a summary of key statistics (mean, median, standard deviation, skewness, kurtosis, and optionally, key percentiles).

## How to use HistoraJ
1.  **Upload Your Data**: In the sidebar on the left, click "Select input file" to upload your data (supported formats: CSV, Excel (.xlsx, .xls), Stata (.dta)).
2.  **Filter Data (Optional)**:
    *   **By Observation Range**: Enter 1-based start and/or end observation numbers to analyze a specific slice of your data.
    *   **By Condition**: Apply a filter based on column values (e.g., `age > 30 and income < 50000`). Refer to column names exactly as they appear in your file.
3.  **Customize Plot Appearance (Optional)**:
    *   **Title & Note**: Add an overall title for plots and/or an overall note/caption for the figure (Markdown supported for note).
    *   **Y-axis Type**: Choose how the y-axis is scaled (Density, Frequency, Fraction, Percentage).
    *   **Show Key Percentiles**: Check this box to include detailed percentile information in the statistics.
4.  **Select Columns for Analysis**: Choose one or more numeric columns from your dataset that you wish to analyze.
5.  **Run Analysis**: Click the "Run Analysis" button in the sidebar.

The results, including plots and detailed statistics (in an expandable section for each variable), will appear in this main area.
"""

_SESSION_DEFAULTS = {
    "df": None,
    "numeric_cols": [],
    "run_analysis_triggered": False,
    "last_selected_columns": [],
    "last_df_filtered_for_plot": None,
    "loaded_file_name": None,
    "file_type_at_load": None,
}


def init_session_state() -> None:
    for key, value in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_run_analysis_triggered() -> None:
    st.session_state.run_analysis_triggered = True
    st.session_state.last_df_filtered_for_plot = None


def _clear_loaded_file() -> None:
    st.session_state.df = None
    st.session_state.numeric_cols = []
    st.session_state.run_analysis_triggered = False
    st.session_state.last_df_filtered_for_plot = None
    st.session_state.loaded_file_name = None
    st.session_state.file_type_at_load = None


def _load_new_file(uploaded_file, file_type: str) -> None:
    try:
        current_df = load_dataframe(uploaded_file, file_type)
    except FileLoadError as exc:
        st.sidebar.error(str(exc))
        _clear_loaded_file()
        return

    st.session_state.df = current_df
    st.session_state.numeric_cols = numeric_column_names(current_df)
    st.session_state.run_analysis_triggered = False
    st.session_state.last_df_filtered_for_plot = None
    st.session_state.loaded_file_name = uploaded_file.name
    st.session_state.file_type_at_load = file_type
    if not st.session_state.numeric_cols:
        st.sidebar.warning("No numeric columns found in the uploaded file.")


def _guessed_file_type_index(uploaded_file) -> int:
    if uploaded_file is None:
        return 0
    try:
        return FILE_TYPE_OPTIONS.index(guess_file_type(uploaded_file.name))
    except (FileLoadError, ValueError):
        return 0


def _prepare_plot_frame(selected_columns: list[str], start_obs: str, end_obs: str, condition: str):
    """Filter the loaded frame after Run Analysis. Returns a frame or empty."""
    if st.session_state.df is None or not selected_columns:
        return None

    df_current = st.session_state.df.copy()
    try:
        df_filtered = apply_filters(df_current, start_obs, end_obs, condition)
    except ObservationRangeError:
        st.error("Invalid observation range.")
        st.session_state.last_df_filtered_for_plot = pd.DataFrame()
        return pd.DataFrame()
    except FilterQueryError as exc:
        st.error(str(exc))
        st.session_state.last_df_filtered_for_plot = pd.DataFrame()
        return pd.DataFrame()

    if df_filtered.empty:
        st.warning("No data after filtering.")
        st.session_state.last_df_filtered_for_plot = pd.DataFrame()
        return pd.DataFrame()

    st.session_state.last_df_filtered_for_plot = df_filtered.copy()
    st.session_state.last_selected_columns = selected_columns[:]
    return df_filtered


def _render_plots(
    df_for_plotting: pd.DataFrame,
    selected_columns: list[str],
    title_str: str,
    note_str: str,
    yaxis_choice: str,
    show_percentiles: bool,
) -> None:
    st.subheader("Analysis Results")
    n = len(selected_columns)
    rows_plot, cols_plot = subplot_grid(n)
    fig, axes = plt.subplots(
        rows_plot,
        cols_plot,
        figsize=(7 * cols_plot, 6 * rows_plot),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    last_i = 0
    for i, col_name in enumerate(selected_columns):
        last_i = i
        if i >= len(axes_flat):
            break
        if col_name not in df_for_plotting.columns:
            st.warning(
                f"Column {col_name} not found in the filtered data for plotting. "
                "It might have been filtered out."
            )
            continue
        stats = plot_column(
            axes_flat[i],
            df_for_plotting[col_name],
            col_name,
            yaxis_choice,
            show_percentiles=show_percentiles,
            title_text=title_str,
        )
        if stats is not None:
            with st.expander(f"Detailed Statistics for {col_name}", expanded=False):
                st.markdown(format_stats_markdown(col_name, stats))
        else:
            if not pd.api.types.is_numeric_dtype(df_for_plotting[col_name]):
                st.info(f"Column {col_name} is not numeric; skipping.")
            else:
                st.warning(f"No data available for {col_name} after filtering/dropping NaNs.")

    for j in range(last_i + 1, len(axes_flat)):
        axes_flat[j].axis("off")
    plt.tight_layout(pad=3.0)
    if note_str and n > 0:
        fig.suptitle(note_str, y=1.00, fontsize=12, va="bottom")
    st.pyplot(fig, dpi=300)


def main() -> None:
    st.set_page_config(layout="wide", page_title="HistoraJ: Summary Statistics & Visualization")
    configure_matplotlib()
    init_session_state()

    st.title("Historaj: Summary Statistics & Visualization")
    st.markdown("**By: Prof. Rajesh Tharyan**")
    st.markdown(APP_INTRO)
    st.divider()

    uploaded_file = st.sidebar.file_uploader(
        "Select input file", type=["csv", "xlsx", "xls", "dta"]
    )
    file_type = st.sidebar.selectbox(
        "File type:",
        FILE_TYPE_OPTIONS,
        index=_guessed_file_type_index(uploaded_file),
    )

    if uploaded_file is not None:
        new_name = uploaded_file.name != st.session_state.get("loaded_file_name")
        same_name_new_type = (
            uploaded_file.name == st.session_state.get("loaded_file_name")
            and file_type != st.session_state.get("file_type_at_load")
        )
        if new_name or same_name_new_type:
            _load_new_file(uploaded_file, file_type)

    if (
        st.session_state.df is not None
        and st.session_state.get("last_df_filtered_for_plot") is None
        and not st.session_state.get("run_analysis_triggered", False)
    ):
        st.subheader("Data Preview (First 5 rows)")
        st.dataframe(st.session_state.df.head())

    condition_str = st.sidebar.text_input(
        "Condition (e.g., col1 > 45 and col2 < 100):",
        value="",
    )
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_obs_str = st.text_input("Start Obs (1-based):", value="")
    with col2:
        end_obs_str = st.text_input("End Obs (1-based):", value="")

    title_str = st.sidebar.text_input("Overall Title for plots (optional):", value="")
    note_str = st.sidebar.text_area(
        "Overall Note for plots (optional, Markdown supported):",
        value="",
        height=100,
    )
    yaxis_choice = st.sidebar.selectbox("Y-axis type:", YAXIS_CHOICES, index=0)
    show_percentiles = st.sidebar.checkbox("Show Key Percentiles in plot stats", value=False)

    selected_columns: list[str] = []
    run_button = False
    if st.session_state.df is not None and st.session_state.numeric_cols:
        selected_columns = st.sidebar.multiselect(
            "Select numeric columns for analysis:",
            st.session_state.numeric_cols,
            default=st.session_state.numeric_cols[0] if st.session_state.numeric_cols else [],
        )
        run_button = st.sidebar.button("Run Analysis", on_click=set_run_analysis_triggered)
    else:
        st.sidebar.info("Upload a file and ensure it has numeric columns to proceed.")

    should_plot_now = False
    df_for_plotting = None

    if st.session_state.run_analysis_triggered:
        should_plot_now = True
        df_for_plotting = _prepare_plot_frame(
            selected_columns, start_obs_str, end_obs_str, condition_str
        )
        if df_for_plotting is None:
            should_plot_now = False
        st.session_state.run_analysis_triggered = False
    elif (
        st.session_state.last_df_filtered_for_plot is not None
        and st.session_state.last_selected_columns
    ):
        should_plot_now = True
        df_for_plotting = st.session_state.last_df_filtered_for_plot
        selected_columns = st.session_state.last_selected_columns

    if (
        should_plot_now
        and df_for_plotting is not None
        and not df_for_plotting.empty
        and selected_columns
    ):
        _render_plots(
            df_for_plotting,
            selected_columns,
            title_str,
            note_str,
            yaxis_choice,
            show_percentiles,
        )
    elif run_button and (st.session_state.df is None or not selected_columns):
        st.warning(
            "Run Analysis clicked, but please upload a file and select at least one numeric column."
        )


if __name__ == "__main__":
    main()
