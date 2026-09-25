"""Tests for the metrics, the rule baseline, and the experiment log."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate, top_share_mask
from src.experiments import load_runs, log_run
from src.train import LeadTimeRule


def test_top_share_flags_exact_count_even_with_ties():
    assert top_share_mask(np.zeros(10), 0.2).sum() == 2
    assert list(np.flatnonzero(top_share_mask(np.array([0.1, 0.9, 0.5, 0.8, 0.2]), 0.4))) == [1, 3]


def test_perfect_and_random_scores():
    y = np.array([0] * 80 + [1] * 20)
    perfect = evaluate(y, y.astype(float))
    assert perfect["roc_auc"] == 1.0 and perfect["recall_top20"] == 1.0 and perfect["precision_top20"] == 1.0
    constant = evaluate(y, np.full(100, 0.2))
    assert constant["roc_auc"] == 0.5
    assert constant["pr_auc"] == pytest.approx(0.2)


def test_lead_time_rule_learns_group_rates():
    X = pd.DataFrame({"lead_days": [0, 0, 10, 10, 10, 10]})
    y = [0, 0, 1, 1, 0, 0]
    p = LeadTimeRule().fit(X, y).predict_proba(pd.DataFrame({"lead_days": [0, 12, 500]}))[:, 1]
    assert p[0] == 0.0 and p[1] == 0.5
    assert p[2] == pytest.approx(2 / 6)     # unseen group falls back to the overall rate


def test_log_run_appends_rows(tmp_path):
    path = tmp_path / "runs.csv"
    log_run("exp", "a", "Model", {"C": 1}, False, {"roc_auc": 0.7}, path=path)
    log_run("exp", "b", "Model", {"C": 2}, True, {"roc_auc": 0.8}, path=path)
    runs = load_runs(path)
    assert list(runs["run_name"]) == ["a", "b"]
    assert runs.loc[1, "roc_auc"] == 0.8


def test_log_keeps_numeric_looking_commit_hash(tmp_path, monkeypatch):
    import src.experiments as ex
    monkeypatch.setattr(ex, "git_commit", lambda: "1913e50")
    path = tmp_path / "runs.csv"
    for name in ("a", "b"):
        ex.log_run("exp", name, "Model", {}, False, {"roc_auc": 0.7}, path=path)
    assert list(ex.load_runs(path)["git_commit"]) == ["1913e50", "1913e50"]
