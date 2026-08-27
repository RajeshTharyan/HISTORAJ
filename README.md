# HistoraJ

[![Tests](https://github.com/RajeshTharyan/HISTORAJ/actions/workflows/tests.yml/badge.svg)](https://github.com/RajeshTharyan/HISTORAJ/actions/workflows/tests.yml)
[![Open in GitHub Codespaces](https://img.shields.io/badge/Open_in-GitHub_Codespaces-2f81f7?logo=github)](https://codespaces.new/RajeshTharyan/HISTORAJ)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A small Streamlit app for **exploratory histograms and descriptive statistics** on a table you already have (CSV, Excel, or Stata). Upload a file, optionally slice rows or apply a pandas `query` filter, pick numeric columns, and get a histogram with a normal overlay, SD markers, and a stats box.

There is **no hosted demo URL** for this repository. Run a local copy, open it in GitHub Codespaces, or deploy your own Streamlit Community Cloud app using the steps below.

This repo is meant to be read as a portfolio piece: a teaching-style EDA tool with a thin UI over testable Python modules — not a statistics library, and not production analytics infrastructure.

---

## The problem

When you first open a dataset, the useful questions are ordinary: How is this variable shaped? Is it skewed? Where are the tails relative to ±1/2/3 standard deviations? How do a couple of columns look side by side after a simple filter?

Spreadsheets and full stats packages can answer that, but they are heavy for a “show me the distribution of these columns” pass. HistoraJ is a single-page Streamlit workflow for that pass: load a file, constrain the sample, plot histograms, and read mean / median / SD / skew / excess kurtosis (and optional percentiles) without leaving the browser.

It does **not** run formal goodness-of-fit tests, choose bin widths for you, or replace Stata/R/Python for research analysis. It visualizes a normal curve on top of the histogram so you can *see* whether a rough Gaussian overlay is plausible.

---

## What a visitor should infer

| Skill | What this repo actually shows | What it does not show |
| --- | --- | --- |
| Python data wrangling | pandas I/O for CSV / `.xlsx` / `.dta`, numeric-column detection, 1-based row slices, `DataFrame.query` filters | A query planner, SQL engine, or generic ETL framework |
| Descriptive statistics | Sample mean, median, SD (`ddof=1`), skew, excess kurtosis, SD markers, selected quantiles | Inferential stats, robust estimators, or a custom math library |
| Visualization | Matplotlib histograms, PDF overlay scaled to density / frequency / fraction / percentage, twin-axis SD labels | Interactive Plotly dashboards or publication layout systems |
| App structure | Streamlit sidebar + `session_state` for upload / rerun; plotting and stats importable without Streamlit | Multi-page apps, auth, databases, or background jobs |
| Engineering hygiene | Named exceptions, no module globals for UI state, pytest on logic, GitHub Actions | Large-scale architecture, typed public APIs with versioning, or high test coverage of the GUI |

If you are skimming for hiring signal: the interesting part is the split between `historajstreamlit.py` (widgets and session) and `historaj/` (I/O, filters, stats, layout, matplotlib). The tests exercise that package with in-memory buffers and the Agg backend — no live network and no browser.

---

## Architecture

```
historajstreamlit.py          Streamlit entry (upload, sidebar, session_state)
        │
        ▼
historaj/
  io.py                       CSV / Excel / Stata loaders
  filters.py                  observation range + pandas query
  stats.py                    ColumnStats + markdown/plot text
  layout.py                   subplot grid (rows × cols)
  plotting.py                 histogram / normal overlay (no Streamlit)
```

**Rerun model.** Uploading a file stores the DataFrame in `st.session_state`. **Run Analysis** applies filters and caches the filtered frame. Later widget changes (y-axis, percentiles, note) redraw from that cache so you do not have to click Run again for cosmetics. Changing the row range, query, or selected columns requires Run Analysis again — that is deliberate, not a hidden refresh of the sample.

**Plotting is headless-testable.** `plot_column` takes a Matplotlib axis and a Series. Tests set `matplotlib.use("Agg")` and assert bar counts / placeholders. Streamlit only calls `st.pyplot` and expanders.

**Filters are explicit.** Observation bounds are 1-based inclusive end, converted to a half-open `iloc` slice. The condition box is pandas `query` syntax, not a custom DSL. Invalid ranges and queries raise `ObservationRangeError` / `FilterQueryError` and surface as Streamlit errors.

---

## Using the app in the browser

Once Streamlit is running (see below), the UI is one page:

1. **Sidebar → Select input file** — `.csv`, `.xlsx`, `.xls`, or `.dta`. Confirm **File type** if the guess is wrong.
2. A **Data Preview** of the first five rows appears in the main pane (until you run an analysis).
3. Optional **Start Obs / End Obs** (1-based) and **Condition** (e.g. `age > 30 and income < 50000`). Column names must match the file, including spaces/case. Identifiers that are not valid Python names need pandas backtick quoting: `` `GDP growth` > 0 ``.
4. Optional overall **Title** (used on skip/placeholder axes) and **Note** (drawn as the figure `suptitle` after a successful run — same as the original app).
5. **Y-axis type**: density, frequency, fraction, or percentage.
6. **Select numeric columns** and click **Run Analysis**.

Each selected column gets a histogram. If sample SD is finite and &gt; 0, a black normal curve and dashed ±1/2/3 SD lines are overlaid. Expand **Detailed Statistics** under the figure for the same numbers in Markdown.

---

## Run a copy

Python 3.11 or 3.12. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run historajstreamlit.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

**Tests** (no Streamlit server required):

```bash
pip install -r requirements-dev.txt
pytest
```

### GitHub Codespaces

The [Codespaces badge](https://codespaces.new/RajeshTharyan/HISTORAJ) opens this repo in a Python 3.12 dev container that installs `requirements.txt` and `requirements-dev.txt`. Then run `streamlit run historajstreamlit.py` in the terminal (port 8501 is forwarded).

### Streamlit Community Cloud

There is no pre-deployed Cloud app linked from this repo. To host your own copy: fork (or connect) the repository in [Streamlit Community Cloud](https://share.streamlit.io/), set the main file to `historajstreamlit.py`, and use this `requirements.txt`. Treat a public Cloud deploy as **your data in someone else's process** — see [SECURITY.md](SECURITY.md).

---

## Honest limits

- **Not a stats package.** Bin count is fixed at 100. The normal curve is a visual overlay, not a test of normality. Kurtosis is pandas excess kurtosis.
- **Title vs note.** After a successful run the figure `suptitle` is the **Note** field, matching the original behavior. The **Overall Title** field is used on non-numeric / empty-data placeholder axes, not as the figure title.
- **Filters apply on Run Analysis**, not on every keystroke. Cosmetic controls (y-axis, percentiles, note) redraw from the last successful filtered frame.
- **`.xls` (Excel 97–2003)** is offered in the uploader; `requirements.txt` includes `openpyxl` for `.xlsx`. Old `.xls` typically needs `xlrd`, which is not listed — those files may fail to load.
- **Memory-bound.** Files live in the Streamlit session. Very large tables will strain the browser/process. Streamlit’s default upload cap is 200 MB unless you raise it.
- **`DataFrame.query` is not a sandbox.** Do not expose this app on a shared host to untrusted users. See [SECURITY.md](SECURITY.md).
- **No persistence, accounts, or sharing.** Close the tab and the session is gone.
- **No hosted demo.** Badges above are CI, Codespaces, and license — not a live dataset explorer.

---

## License

MIT. See [LICENSE](LICENSE).
