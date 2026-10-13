"""Tests for house_price_analysis.py. No network; small fixture in tests/fixtures."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import house_price_analysis as hpa

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_listings.csv"


@pytest.fixture
def prepared():
    return hpa.load_and_prepare_data(str(FIXTURE))


def test_load_drops_redundant_columns_and_adds_converted_ones(prepared):
    assert "SQUARE_FT" not in prepared.columns
    assert "BHK_OR_RK" not in prepared.columns
    assert "TARGET(PRICE_IN_LACS)" not in prepared.columns
    assert {"SQUARE_M", "TARGET(PRICE_IN_BGP)"} <= set(prepared.columns)
    assert len(prepared) == 6


def test_load_converts_square_feet_to_square_metres(prepared):
    assert prepared.loc[0, "SQUARE_M"] == pytest.approx(1300.0 / 10.764)


def test_load_converts_lacs_to_pounds_at_94_94(prepared):
    # 55 lacs = 5,500,000 INR; / 94.94 INR per GBP
    assert prepared.loc[0, "TARGET(PRICE_IN_BGP)"] == pytest.approx(5_500_000 / 94.94)


def test_load_turns_flag_columns_into_booleans(prepared):
    for col in ["UNDER_CONSTRUCTION", "RERA", "READY_TO_MOVE", "RESALE"]:
        assert prepared[col].dtype == bool
    assert bool(prepared.loc[3, "RERA"]) is True


def test_quoted_address_with_comma_stays_one_field(prepared):
    assert prepared.loc[0, "ADDRESS"] == "Ksfc Layout,Bangalore"


def test_missing_file_returns_none(capsys):
    assert hpa.load_and_prepare_data("does_not_exist.csv") is None
    assert "was not found" in capsys.readouterr().out


def test_missing_column_is_swallowed_and_returns_none(tmp_path, capsys):
    # Documents current behaviour: any error becomes a printed message and None.
    bad = tmp_path / "bad.csv"
    bad.write_text("A,B\n1,2\n")
    assert hpa.load_and_prepare_data(str(bad)) is None
    assert "An error occurred" in capsys.readouterr().out


def test_build_and_evaluate_recovers_a_perfect_linear_relationship(capsys):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    y = 3.0 + 2.0 * X["a"] - 1.0 * X["b"] + 0.5 * X["c"]
    hpa.build_and_evaluate_model(X, y, "synthetic")
    out = capsys.readouterr().out
    assert "Evaluating Model: synthetic" in out
    assert "Coefficient of Determination (R^2): 1.00" in out
    assert "Intercept: 3.00" in out


def test_split_is_reproducible(capsys):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"])
    y = X["a"] + rng.normal(size=100)
    hpa.build_and_evaluate_model(X, y, "m")
    first = capsys.readouterr().out
    hpa.build_and_evaluate_model(X, y, "m")
    assert capsys.readouterr().out == first  # random_state=42 fixed


@pytest.mark.skipif(not (ROOT / "train.csv").exists(), reason="train.csv not present")
def test_full_pipeline_on_real_data_matches_verified_numbers(capsys):
    hpa.main()
    out = capsys.readouterr().out
    assert "Original dataset size: 29451" in out
    assert "after removing price outliers: 26367" in out
    for name in ["Builder", "Dealer", "Owner", "All Listings"]:
        assert name in out


@pytest.mark.xfail(strict=True, reason=(
    "Known bug (review.md P1): train.csv has 401 exact duplicate rows that the Java "
    "DublicateRemover step removes but the Python script never does, so the Python-only "
    "path models duplicated listings (and can leak them across the train/test split)."))
def test_python_path_removes_duplicate_listings():
    df = hpa.load_and_prepare_data(str(ROOT / "train.csv"))
    assert not df.duplicated().any()
