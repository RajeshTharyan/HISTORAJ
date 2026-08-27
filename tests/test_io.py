"""Tests for CSV/Excel/Stata loaders and numeric-column detection."""

import io

import pandas as pd
import pytest

from historaj.io import (
    FileLoadError,
    guess_file_type,
    load_dataframe,
    numeric_column_names,
)


def test_guess_file_type():
    assert guess_file_type("data.CSV") == "CSV"
    assert guess_file_type("book.xlsx") == "Excel"
    assert guess_file_type("old.xls") == "Excel"
    assert guess_file_type("panel.dta") == "Stata (.dta)"
    with pytest.raises(FileLoadError):
        guess_file_type("notes.txt")


def test_load_csv_from_stringio():
    buf = io.StringIO("x,y,label\n1,2,a\n3,4,b\n")
    df = load_dataframe(buf, "CSV")
    assert list(df.columns) == ["x", "y", "label"]
    assert len(df) == 2
    assert numeric_column_names(df) == ["x", "y"]


def test_load_excel_from_bytesio():
    source = pd.DataFrame({"height": [1.5, 1.8], "name": ["a", "b"]})
    buf = io.BytesIO()
    source.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    df = load_dataframe(buf, "Excel")
    pd.testing.assert_frame_equal(df, source)


def test_load_stata_from_bytesio():
    source = pd.DataFrame({"income": [10.0, 20.0, 30.0], "year": [2000, 2001, 2002]})
    buf = io.BytesIO()
    source.to_stata(buf, write_index=False)
    buf.seek(0)
    df = load_dataframe(buf, "Stata (.dta)")
    assert list(df["income"]) == [10.0, 20.0, 30.0]
    assert numeric_column_names(df) == ["income", "year"]


def test_load_rejects_unknown_type():
    with pytest.raises(FileLoadError):
        load_dataframe(io.StringIO("a,b\n1,2\n"), "Parquet")


def test_numeric_column_names_ignores_object_and_bool():
    df = pd.DataFrame(
        {
            "n": [1, 2],
            "f": [0.5, 1.5],
            "s": ["x", "y"],
            "b": [True, False],
        }
    )
    assert numeric_column_names(df) == ["n", "f"]
