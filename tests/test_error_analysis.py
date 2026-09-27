"""Tests for the error-analysis helpers (synthetic data only)."""
import pandas as pd
import pytest

from src.error_analysis import slice_metrics


def test_slice_metrics_error_rates():
    df = pd.DataFrame({
        "group":       ["a", "a", "a", "a", "b", "b"],
        "no_show":     [1, 1, 0, 0, 1, 0],
        "probability": [0.9, 0.2, 0.8, 0.1, 0.7, 0.3],
        "flagged":     [True, False, True, False, True, False],
    })
    t = slice_metrics(df, "group")
    assert t.loc["a", "recall_in_high"] == pytest.approx(0.5)      # 1 of 2 no-shows flagged
    assert t.loc["a", "false_alarm_rate"] == pytest.approx(0.5)    # 1 of 2 attenders flagged
    assert t.loc["a", "precision_in_high"] == pytest.approx(0.5)
    assert t.loc["b", "roc_auc"] == 1.0
    assert t["appointments"].sum() == 6
