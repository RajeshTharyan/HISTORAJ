"""Tests for observation-range parsing and pandas-query filters."""

import pandas as pd
import pytest

from historaj.filters import (
    FilterQueryError,
    ObservationRangeError,
    apply_condition,
    apply_filters,
    parse_observation_range,
)


def test_parse_observation_range_defaults():
    assert parse_observation_range("", "", 10) == (0, 10)
    assert parse_observation_range(None, None, 10) == (0, 10)


def test_parse_observation_range_one_based_slice():
    assert parse_observation_range("1", "5", 10) == (0, 5)
    assert parse_observation_range("3", "", 10) == (2, 10)
    assert parse_observation_range("", "4", 10) == (0, 4)
    assert parse_observation_range("10", "10", 10) == (9, 10)


def test_parse_observation_range_rejects_invalid():
    with pytest.raises(ObservationRangeError):
        parse_observation_range("abc", "", 10)
    with pytest.raises(ObservationRangeError):
        parse_observation_range("0", "5", 10)
    with pytest.raises(ObservationRangeError):
        parse_observation_range("5", "3", 10)
    with pytest.raises(ObservationRangeError):
        parse_observation_range("1", "11", 10)
    with pytest.raises(ObservationRangeError):
        parse_observation_range("", "", 0)


def test_apply_condition_blank_is_noop():
    df = pd.DataFrame({"age": [10, 20, 30]})
    out = apply_condition(df, "  ")
    pd.testing.assert_frame_equal(out, df)


def test_apply_condition_filters_rows():
    df = pd.DataFrame({"age": [10, 20, 30], "score": [1, 2, 3]})
    out = apply_condition(df, "age > 15 and score < 3")
    assert list(out["age"]) == [20]


def test_apply_condition_rejects_bad_query():
    df = pd.DataFrame({"age": [10, 20]})
    with pytest.raises(FilterQueryError):
        apply_condition(df, "not a valid query ???")
    with pytest.raises(FilterQueryError):
        apply_condition(df, "missing_col > 1")


def test_apply_filters_combines_range_and_condition():
    df = pd.DataFrame({"age": [10, 20, 30, 40, 50]})
    out = apply_filters(df, start_obs="2", end_obs="4", condition="age >= 30")
    assert list(out["age"]) == [30, 40]
