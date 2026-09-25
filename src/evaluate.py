"""Evaluation metrics for no-show prediction.

The classes are imbalanced (about 20% no-shows), so accuracy is not used. Metrics:

- ROC-AUC: how well the model ranks no-shows above attended appointments (0.5 = random).
- PR-AUC (average precision): ranking quality focused on the no-show class; random = the no-show rate.
- precision / recall @ top 20%: staff can only follow up a limited number of appointments. If they
  contact the 20% riskiest, what share of those are real no-shows (precision) and what share of all
  no-shows do they reach (recall)? Random selection gives recall = 20%.
- Brier score: mean squared error of the probabilities (lower is better); checks they are sensible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

TOP_SHARE = 0.20


def top_share_mask(y_score: np.ndarray, share: float = TOP_SHARE) -> np.ndarray:
    """Boolean mask of the `share` highest-scoring rows. Ties are broken by row order, so exactly
    round(share * n) rows are flagged even when many scores are equal (e.g. a constant baseline)."""
    n_flag = int(round(share * len(y_score)))
    order = np.argsort(-np.asarray(y_score), kind="stable")
    mask = np.zeros(len(y_score), dtype=bool)
    mask[order[:n_flag]] = True
    return mask


def evaluate(y_true, y_score, share: float = TOP_SHARE) -> dict[str, float]:
    """All metrics for one set of predicted no-show probabilities."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    flagged = top_share_mask(y_score, share)
    caught = y_true[flagged].sum()
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        f"precision_top{int(share * 100)}": caught / flagged.sum(),
        f"recall_top{int(share * 100)}": caught / y_true.sum(),
        "brier": brier_score_loss(y_true, y_score),
        "no_show_rate": y_true.mean(),
    }


def paired_bootstrap(y_true, scores: dict[str, np.ndarray], reference: str, n_boot: int = 1000,
                     seed: int = 0) -> pd.DataFrame:
    """Is each model really better than `reference`, or is the gap just noise?

    Resamples the evaluation rows with replacement `n_boot` times. In each resample every model is
    scored on the SAME rows (a paired comparison), and the difference to the reference is recorded.
    Returns the mean difference, its 95% interval, and how often the model beat the reference.
    If the interval contains 0, the difference is not reliable.
    """
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    metrics = {"roc_auc": lambda y, s: roc_auc_score(y, s),
               f"recall_top{int(TOP_SHARE * 100)}": lambda y, s: y[top_share_mask(s)].sum() / y.sum()}
    diffs = {(name, m): [] for name in scores if name != reference for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        y = y_true[idx]
        for m, fn in metrics.items():
            ref = fn(y, np.asarray(scores[reference])[idx])
            for name in scores:
                if name != reference:
                    diffs[(name, m)].append(fn(y, np.asarray(scores[name])[idx]) - ref)
    rows = []
    for (name, m), d in diffs.items():
        d = np.array(d)
        rows.append({"model": name, "metric": m, "mean_diff": d.mean(), "ci_low": np.percentile(d, 2.5),
                     "ci_high": np.percentile(d, 97.5), "share_better": (d > 0).mean()})
    return pd.DataFrame(rows)


def bootstrap_ci(y_true, y_score, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """95% bootstrap interval for every metric of one model: how much the score could move
    with a different sample of appointments."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        samples.append(evaluate(y_true[idx], y_score[idx]))
    samples = pd.DataFrame(samples).drop(columns="no_show_rate")
    point = pd.Series(evaluate(y_true, y_score)).drop("no_show_rate")
    return pd.DataFrame({"estimate": point, "ci_low": samples.quantile(0.025), "ci_high": samples.quantile(0.975)})
