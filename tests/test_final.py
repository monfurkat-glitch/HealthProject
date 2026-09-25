"""Tests for the final-model helpers. They never touch the real test set."""
import numpy as np
import pytest

from src.evaluate import bootstrap_ci
from src.final import band_table, calibration_table, risk_band


def test_risk_band_boundaries():
    cutoffs = [0.1, 0.3]
    assert list(risk_band([0.05, 0.1, 0.2, 0.3, 0.9], cutoffs)) == ["Low", "Medium", "Medium", "High", "High"]
    assert list(risk_band(0.5, cutoffs)) == ["High"]          # a single probability works too


def test_band_table_shares_add_up():
    table = band_table([0, 1, 0, 1, 1, 0], ["Low", "Low", "Medium", "High", "High", "High"])
    assert table["appointments"].sum() == 6
    assert table["share_of_all_no_shows"].sum() == pytest.approx(1.0)
    assert table.loc["High", "no_show_rate"] == pytest.approx(2 / 3)


def test_calibration_table_has_deciles():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=1000)
    y = rng.uniform(size=1000) < p
    table = calibration_table(y, p)
    assert len(table) == 10
    assert (table["observed_rate"] - table["mean_predicted"]).abs().max() < 0.1   # well calibrated by construction


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = 1 / (1 + np.exp(-(y + rng.normal(size=500))))     # probabilities in (0, 1)
    ci = bootstrap_ci(y, s, n_boot=200)
    assert (ci["ci_low"] <= ci["estimate"]).all() and (ci["estimate"] <= ci["ci_high"]).all()
