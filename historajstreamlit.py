import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

# Set DPI and font parameters for sharper text - Streamlit handles this differently,
# but we can still set rcParams for Matplotlib plots.
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'sans-serif',
    'font.size': 10
})

def historaj(df_col_data, col_name, title_text, note_text, yaxis_choice, show_percentiles_flag, ax):
    """
    Generates a histogram and statistical analysis for a given column.
    This function is adapted for use within Streamlit and Matplotlib.
    df_col_data: pandas Series containing the data for the column.
    ax: Matplotlib axis object to plot on.
    """
    if not np.issubdtype(df_col_data.dtype, np.number):
        st.info(f"Column {col_name} is not numeric; skipping.")
        ax.text(0.5, 0.5, f"Column {col_name} is not numeric.",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=10, color='red')
        ax.set_title(title_text if title_text else col_name)
        ax.set_xlabel(col_name)
        ax.set_ylabel(yaxis_choice.capitalize())
        return

    data = df_col_data.dropna()

    if data.empty:
        st.warning(f"No data available for {col_name} after filtering/dropping NaNs.")
        ax.text(0.5, 0.5, f"No data available for {col_name} after filtering.",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=10, color='red')
        ax.set_title(title_text if title_text else col_name)
        ax.set_xlabel(col_name)
        ax.set_ylabel(yaxis_choice.capitalize())
        return

    full_mean = data.mean()
    full_std = data.std()
    median_val = data.median()
    skew_val = data.skew()
    kurt_val = data.kurt()

    valid_std_dev = full_std is not None and not np.isnan(full_std) and full_std > 0

    markers = []
    if valid_std_dev:
        markers = [
            ("-3sd", full_mean - 3 * full_std),
            ("-2sd", full_mean - 2 * full_std),
            ("-1sd", full_mean - 1 * full_std),
            ("+1sd", full_mean + 1 * full_std),
            ("+2sd", full_mean + 2 * full_std),
            ("+3sd", full_mean + 3 * full_std),
        ]
    sd_colors = {
        "-3sd": 'red', "+3sd": 'red',
        "-2sd": 'orange', "+2sd": 'orange',
        "-1sd": 'green', "+1sd": 'green'
    }

    # Streamlit alternative for console print (can be extensive, consider st.expander)
    stats_summary_md = f"""
    **Stats for {col_name}:**
    - Obs    : {len(data)}
    - Mean   : {f'{full_mean:.5f}' if pd.notna(full_mean) else 'N/A'}
    - Median : {f'{median_val:.5f}' if pd.notna(median_val) else 'N/A'}
    - StdDev : {f'{full_std:.5f}' if pd.notna(full_std) else 'N/A'}
    - Skew   : {f'{skew_val:.5f}' if pd.notna(skew_val) else 'N/A'}
    - Kurtos : {f'{kurt_val:.5f}' if pd.notna(kurt_val) else 'N/A'}
    """
    if valid_std_dev:
        for lab, val in markers:
            stats_summary_md += f"\n    - {lab:<6}: {f'{val:.5f}' if pd.notna(val) else 'N/A'}"

    percentile_values_dict = {}
    if show_percentiles_flag and not data.empty:
        percentiles_to_calc = [0.05, 0.10, 0.25, 0.75, 0.90, 0.95]
        calculated_percentiles = data.quantile(percentiles_to_calc)
        percentile_labels = ["P5", "P10", "P25(Q1)", "P75(Q3)", "P90", "P95"]
        stats_summary_md += "\n    **Percentiles:**"
        for label, val in zip(percentile_labels, calculated_percentiles):
            percentile_values_dict[label] = val
            stats_summary_md += f"\n    - {label:<8}: {f'{val:.5f}' if pd.notna(val) else 'N/A'}"
    
    with st.expander(f"Detailed Statistics for {col_name}", expanded=False):
        st.markdown(stats_summary_md)


    weights = None
    density_param = False
    ylabel = yaxis_choice.capitalize()

    if yaxis_choice == "density":
        density_param = True
    elif yaxis_choice == "fraction":
        weights = np.ones_like(data) / len(data)
    elif yaxis_choice == "percentage":
        weights = np.ones_like(data) / len(data) * 100

    counts, bins, _ = ax.hist(data, bins=100, density=density_param, weights=weights, alpha=0.7, edgecolor='black')
    xmin, xmax_hist = ax.get_xlim() # Use xmax_hist to avoid conflict later
    
    # Ensure x-axis for normal curve covers data range even if hist range is small
    data_min, data_max = data.min(), data.max()
    plot_xmin = min(xmin, data_min - 0.1 * abs(data_min) if valid_std_dev else data_min) # Add buffer if std_dev is valid
    plot_xmax = max(xmax_hist, data_max + 0.1 * abs(data_max) if valid_std_dev else data_max)
    ax.set_xlim(plot_xmin, plot_xmax) # Update axis limit based on data + hist range

    x_norm = np.linspace(plot_xmin, plot_xmax, 200)


    if valid_std_dev:
        p = norm.pdf(x_norm, full_mean, full_std)
        if yaxis_choice == "density":
            ax.plot(x_norm, p, 'k', linewidth=2)
        else: # frequency, fraction, or percentage
            # For non-density, scale PDF by N * bin_width.
            # For fraction/percentage, histogram is already scaled by weights.
            # The PDF curve should then align with the overall shape irrespective of hist's weights.
            # So, we scale the PDF as if it were matching a frequency histogram,
            # and it should visually align if the shapes are similar.
            bin_width_approx = (bins[-1] - bins[0]) / (len(bins) -1) if len(bins) > 1 else 1
            
            if yaxis_choice == "fraction":
                 ax.plot(x_norm, p * bin_width_approx, 'k', linewidth=2) # Scale PDF by bin_width for fraction
            elif yaxis_choice == "percentage":
                 ax.plot(x_norm, p * bin_width_approx * 100, 'k', linewidth=2) # Scale PDF by bin_width*100 for percentage
            else: # frequency
                 ax.plot(x_norm, p * len(data) * bin_width_approx, 'k', linewidth=2)


        for lab, marker_val in markers:
            ax.axvline(marker_val, linestyle='dashed', linewidth=1, color=sd_colors[lab])

    ax.set_xlabel(f"{col_name}: {yaxis_choice.capitalize()} Plot")
    ax.set_ylabel(ylabel)
    ax.set_title("")

    if valid_std_dev:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        marker_positions = [val for (_, val) in markers]
        marker_labels = [lab for (lab, _) in markers]
        ax_top.set_xticks(marker_positions)
        ax_top.set_xticklabels(marker_labels, fontweight='bold', fontsize=9) # Reduced fontsize
        ax_top.tick_params(axis='x', pad=5) # Reduced pad
        for tick, lab in zip(ax_top.get_xticklabels(), marker_labels):
            tick.set_color(sd_colors[lab])

    stats_text_parts = [
        f"Obs   : {len(data)}",
        f"Mean  : {f'{full_mean:.5f}' if pd.notna(full_mean) else 'N/A'}",
        f"Median: {f'{median_val:.5f}' if pd.notna(median_val) else 'N/A'}",
        f"StdDev: {f'{full_std:.5f}' if pd.notna(full_std) else 'N/A'}",
        f"Skew  : {f'{skew_val:.5f}' if pd.notna(skew_val) else 'N/A'}",
        f"Kurtos: {f'{kurt_val:.5f}' if pd.notna(kurt_val) else 'N/A'}"
    ]
    if show_percentiles_flag and percentile_values_dict:
        stats_text_parts.append("---- Percentiles ----")
        for label, val in percentile_values_dict.items():
            stats_text_parts.append(f"{label:<8}: {f'{val:.5f}' if pd.notna(val) else 'N/A'}")
    stats_text_on_plot = "\n".join(stats_text_parts)

    ax.text(0.98, 0.95, stats_text_on_plot, transform=ax.transAxes,
            fontsize=7, verticalalignment='top', horizontalalignment='right', #Reduced fontsize
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='lightgrey'))


def main():
    st.set_page_config(layout="wide", page_title="HistoraJ: Summary Statistics & Visualization")
    st.title("Historaj: Summary Statistics & Visualization")
    st.markdown("**By: Prof. Rajesh Tharyan**")

    # --- App Description and Instructions ---
    st.markdown("""
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
    """)
    st.markdown("---_" ) # Visual separator

    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'numeric_cols' not in st.session_state:
        st.session_state.numeric_cols = []
    if 'run_analysis_triggered' not in st.session_state: # Combined flag
        st.session_state.run_analysis_triggered = False
    if 'last_selected_columns' not in st.session_state:
        st.session_state.last_selected_columns = []
    if 'last_df_filtered_for_plot' not in st.session_state:
        st.session_state.last_df_filtered_for_plot = None
    if 'loaded_file_name' not in st.session_state:
        st.session_state.loaded_file_name = None
    if 'file_type_at_load' not in st.session_state: # To track file type for accurate new file detection
        st.session_state.file_type_at_load = None

    # --- Callback Functions ---
    def trigger_rerun_for_note():
        # This callback is mainly to ensure Streamlit registers the change and reruns.
        # The actual logic to decide if we replot will be in the main flow.
        pass # Rerun happens automatically on widget change with on_change

    def set_run_analysis_triggered():
        st.session_state.run_analysis_triggered = True
        # When run analysis is explicitly clicked, we want to use the new data.
        st.session_state.last_df_filtered_for_plot = None 

    # --- Sidebar for Inputs ---
    # st.sidebar.header("Input Options") # Removed for vertical compactness

    uploaded_file = st.sidebar.file_uploader("Select input file", type=["csv", "xlsx", "xls", "dta"])
    
    file_type_options = ["CSV", "Excel", "Stata (.dta)"]
    guessed_ft_index = 0 
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            guessed_ft_index = 0
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            guessed_ft_index = 1
        elif uploaded_file.name.endswith('.dta'):
            guessed_ft_index = 2
            
    file_type = st.sidebar.selectbox("File type:", file_type_options, index=guessed_ft_index, on_change=trigger_rerun_for_note) # on_change to allow reprocessing if file type changes for an already uploaded file name

    # File processing logic
    if uploaded_file is not None:
        # Process if new file name, or if file type changed for the same file name
        if uploaded_file.name != st.session_state.get('loaded_file_name', None) or \
           (uploaded_file.name == st.session_state.get('loaded_file_name', None) and file_type != st.session_state.get('file_type_at_load', None)):
            try:
                current_df = None
                if file_type == "CSV":
                    current_df = pd.read_csv(uploaded_file)
                elif file_type == "Excel":
                    current_df = pd.read_excel(uploaded_file)
                elif file_type == "Stata (.dta)":
                    current_df = pd.read_stata(uploaded_file)
                
                st.session_state.df = current_df
                st.session_state.numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
                st.session_state.run_analysis_triggered = False 
                st.session_state.last_df_filtered_for_plot = None # Reset for new data, enables preview
                st.session_state.loaded_file_name = uploaded_file.name
                st.session_state.file_type_at_load = file_type

                if not st.session_state.numeric_cols :
                     st.sidebar.warning("No numeric columns found in the uploaded file.")
            except Exception as e:
                st.sidebar.error(f"Failed to load file: {e}")
                st.session_state.df = None
                st.session_state.numeric_cols = []
                st.session_state.run_analysis_triggered = False
                st.session_state.last_df_filtered_for_plot = None
                st.session_state.loaded_file_name = None 
                st.session_state.file_type_at_load = None

    # Conditional Data Preview
    if st.session_state.df is not None and \
       st.session_state.get('last_df_filtered_for_plot') is None and \
       not st.session_state.get('run_analysis_triggered', False):
        st.subheader("Data Preview (First 5 rows)")
        st.dataframe(st.session_state.df.head())

    condition_str = st.sidebar.text_input("Condition (e.g., col1 > 45 and col2 < 100):", value="", on_change=trigger_rerun_for_note)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_obs_str = st.text_input("Start Obs (1-based):", value="", on_change=trigger_rerun_for_note)
    with col2:
        end_obs_str = st.text_input("End Obs (1-based):", value="", on_change=trigger_rerun_for_note)

    title_str = st.sidebar.text_input("Overall Title for plots (optional):", value="", on_change=trigger_rerun_for_note)
    note_str = st.sidebar.text_area("Overall Note for plots (optional, Markdown supported):", value="", height=100, on_change=trigger_rerun_for_note)
    
    yaxis_options = ["density", "frequency", "fraction", "percentage"]
    yaxis_choice = st.sidebar.selectbox("Y-axis type:", yaxis_options, index=0, on_change=trigger_rerun_for_note)
    
    show_percentiles = st.sidebar.checkbox("Show Key Percentiles in plot stats", value=False, on_change=trigger_rerun_for_note)

    selected_columns = []
    if st.session_state.df is not None and st.session_state.numeric_cols:
        selected_columns = st.sidebar.multiselect("Select numeric columns for analysis:", 
                                                  st.session_state.numeric_cols, 
                                                  default=st.session_state.numeric_cols[0] if st.session_state.numeric_cols else [],
                                                  on_change=trigger_rerun_for_note) # Trigger rerun if cols change
        
        run_button = st.sidebar.button("Run Analysis", on_click=set_run_analysis_triggered)
    else:
        st.sidebar.info("Upload a file and ensure it has numeric columns to proceed.")
        run_button = False # Ensure run_button is defined

    # --- Main Area for Plots and Outputs ---
    # Plot if run analysis was clicked OR if settings changed and a previous plot existed and data is ready
    should_plot_now = False
    df_for_plotting = None

    if st.session_state.run_analysis_triggered:
        should_plot_now = True
        # Process data fully if run_button was the trigger
        if st.session_state.df is not None and selected_columns:
            df_current = st.session_state.df.copy()
            original_len = len(df_current)
            start_idx, end_idx, valid_obs_range = 0, original_len, True
            # (Observation range parsing and validation logic - condensed for brevity, assume it sets df_processed)
            if start_obs_str: # Simplified validation logic for brevity
                try: start_idx = int(start_obs_str) - 1
                except: valid_obs_range = False
            if end_obs_str: 
                try: end_idx = int(end_obs_str)
                except: valid_obs_range = False
            if not (0 <= start_idx < end_idx <= original_len and valid_obs_range):
                st.error("Invalid observation range.")
                df_processed = pd.DataFrame() # Empty df
            else: 
                df_processed = df_current.iloc[start_idx:end_idx]
            
            df_filtered = df_processed
            if condition_str and not df_processed.empty:
                try: df_filtered = df_processed.query(condition_str)
                except Exception as e: st.error(f"Invalid condition: {e}"); df_filtered = pd.DataFrame()
            
            if df_filtered.empty:
                st.warning("No data after filtering.")
                # Store a marker (e.g., empty DataFrame) to indicate analysis was attempted
                st.session_state.last_df_filtered_for_plot = pd.DataFrame() 
            else:
                st.session_state.last_df_filtered_for_plot = df_filtered.copy()
                st.session_state.last_selected_columns = selected_columns[:]
                df_for_plotting = df_filtered
        else:
            should_plot_now = False # Not enough data
        st.session_state.run_analysis_triggered = False # Reset trigger
    
    elif st.session_state.last_df_filtered_for_plot is not None and st.session_state.last_selected_columns:
        # If not triggered by run button, but a previous plot existed, consider replotting with current settings
        # This handles changes from on_change callbacks for note, title, y-axis etc.
        should_plot_now = True
        df_for_plotting = st.session_state.last_df_filtered_for_plot
        selected_columns = st.session_state.last_selected_columns # Use last successfully plotted columns
        # Note: if selected_columns widget itself changed, this might need more sophisticated handling
        # to re-filter data. For now, assume other on_change widgets are for plot cosmetics.
        # A better approach might be to always re-filter if any data-affecting param changes.
        # For this iteration, we only re-filter fully on "Run Analysis"

    if should_plot_now and df_for_plotting is not None and not df_for_plotting.empty and selected_columns:
        st.subheader("Analysis Results")
        n = len(selected_columns)
        cols_plot = min(n, 2)
        rows_plot = math.ceil(n / cols_plot)
        fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(7 * cols_plot, 6 * rows_plot), squeeze=False)
        axes_flat = axes.flatten()

        for i, col_name in enumerate(selected_columns):
            if i < len(axes_flat):
                if col_name in df_for_plotting.columns:
                    current_ax = axes_flat[i]
                    historaj(df_for_plotting[col_name], col_name, title_str, note_str, yaxis_choice, show_percentiles, current_ax)
                else:
                    st.warning(f"Column {col_name} not found in the filtered data for plotting. It might have been filtered out.")
        
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')
        plt.tight_layout(pad=3.0)
        if note_str and n > 0:
            fig.suptitle(note_str, y=1.00, fontsize=12, va='bottom')
        st.pyplot(fig, dpi=300)
    
    elif run_button and (st.session_state.df is None or not selected_columns):
        st.warning("Run Analysis clicked, but please upload a file and select at least one numeric column.")


if __name__ == "__main__":
    main() 