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
