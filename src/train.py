"""Train and compare models. Every run is scored on the validation set and logged.

Usage:
    python -m src.train baselines
    python -m src.train logreg_tuning
    python -m src.train random_forest
    python -m src.train gradient_boosting
    python -m src.train compare        # bootstrap test: are the best models really better than logreg?

The test set is NOT used here. It is kept untouched until the final model is chosen.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.evaluate import evaluate, paired_bootstrap
from src.experiments import log_run
from src.preprocess import REPO_ROOT
from src.preprocess import TARGET, build_dataset, build_preprocessor, feature_columns, time_split

RANDOM_STATE = 42


class LeadTimeRule(BaseEstimator, ClassifierMixin):
    """Rule-based baseline: predict the training no-show rate of the appointment's lead-time group.

    Encodes the single strongest pattern from the EDA ("the longer the wait, the more no-shows")
    without any machine learning.
    """

    bins = [-1, 0, 2, 7, 14, 30, 60, np.inf]

    def fit(self, X: pd.DataFrame, y):
        groups = pd.cut(X["lead_days"], self.bins)
        self.rates_ = pd.Series(np.asarray(y)).groupby(groups.to_numpy(), observed=True).mean()
        self.default_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        groups = pd.cut(X["lead_days"], self.bins)
        p = groups.map(self.rates_).astype(float).fillna(self.default_).to_numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass
class Experiment:
    run_name: str
    build: callable                   # include_sms -> unfitted estimator
    params: dict = field(default_factory=dict)
    include_sms: bool = False
    notes: str = ""


def logistic(include_sms: bool, **params) -> Pipeline:
    return Pipeline([("pre", build_preprocessor(scale_numeric=True, include_sms=include_sms)),
                     ("model", LogisticRegression(max_iter=2000, **params))])


def random_forest(include_sms: bool, **params) -> Pipeline:
    return Pipeline([("pre", build_preprocessor(scale_numeric=False, include_sms=include_sms)),
                     ("model", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE,
                                                      **params))])


def gradient_boosting(include_sms: bool, **params) -> Pipeline:
    """Early stopping holds out 10% of the training rows internally, so validation stays unseen."""
    return Pipeline([("pre", build_preprocessor(scale_numeric=False, include_sms=include_sms)),
                     ("model", HistGradientBoostingClassifier(max_iter=1000, early_stopping=True,
                                                              random_state=RANDOM_STATE, **params))])


SHORT_NAMES = {"learning_rate": "lr", "max_leaf_nodes": "leaves", "min_samples_leaf": "leaf", "max_features": "feat"}


def grid(prefix: str, builder, grid_params: list[dict], notes: str = "") -> list[Experiment]:
    """One experiment per parameter combination, named e.g. 'hgb_lr0.1_leaves15_leaf20'."""
    def name(p):
        return prefix + "_" + ("_".join(f"{SHORT_NAMES.get(k, k)}{v}" for k, v in p.items()) or "default")
    return [Experiment(name(p), lambda sms, p=p: builder(sms, **p), p, notes=notes) for p in grid_params]


EXPERIMENTS: dict[str, list[Experiment]] = {
    "baselines": [
        Experiment("majority_class", lambda sms: DummyClassifier(strategy="prior"),
                   {"strategy": "prior"}, notes="Always predicts the training no-show rate"),
        Experiment("lead_time_rule", lambda sms: LeadTimeRule(), {"bins": "0,1-2,3-7,8-14,15-30,31-60,61+"},
                   notes="Training no-show rate of the lead-time group; no ML"),
        Experiment("logreg", lambda sms: logistic(sms, C=1.0), {"C": 1.0},
                   notes="Logistic regression, all default features"),
        Experiment("logreg_balanced", lambda sms: logistic(sms, C=1.0, class_weight="balanced"),
                   {"C": 1.0, "class_weight": "balanced"}, notes="Class weights for the 80/20 imbalance"),
        Experiment("logreg_with_sms", lambda sms: logistic(sms, C=1.0), {"C": 1.0}, include_sms=True,
                   notes="Comparison only: SMS status is not known at booking time"),
    ],
    # Regularisation strength: smaller C = simpler model
    "logreg_tuning": grid("logreg", logistic, [{"C": c} for c in (0.001, 0.01, 0.1, 1.0, 10.0)]),
    # Tree size: min_samples_leaf controls how specific each leaf can be (1 = fully grown trees)
    "random_forest": grid("rf", random_forest, [
        {"min_samples_leaf": 1, "max_features": "sqrt"},
        {"min_samples_leaf": 20, "max_features": "sqrt"},
        {"min_samples_leaf": 100, "max_features": "sqrt"},
        {"min_samples_leaf": 20, "max_features": 0.5},
    ]),
    "gradient_boosting": grid("hgb", gradient_boosting, [{}] + [
        {"learning_rate": lr, "max_leaf_nodes": leaves, "min_samples_leaf": leaf}
        for lr in (0.03, 0.1) for leaves in (15, 63) for leaf in (20, 200)
    ]),
}


def run(experiment: str) -> pd.DataFrame:
    train, val, _ = time_split(build_dataset())
    results = []
    for exp in EXPERIMENTS[experiment]:
        cols = feature_columns(include_sms=exp.include_sms)
        model = exp.build(exp.include_sms).fit(train[cols], train[TARGET])
        metrics = evaluate(val[TARGET], model.predict_proba(val[cols])[:, 1])
        # Training score, to spot overfitting (a large gap to validation)
        metrics["train_roc_auc"] = roc_auc_score(train[TARGET], model.predict_proba(train[cols])[:, 1])
        estimator = model[-1] if isinstance(model, Pipeline) else model
        if isinstance(estimator, HistGradientBoostingClassifier):   # boosting rounds chosen by early stopping
            exp.params = {**exp.params, "n_iter_": int(estimator.n_iter_)}
        log_run(experiment, exp.run_name, type(model[-1] if isinstance(model, Pipeline) else model).__name__,
                exp.params, exp.include_sms, metrics, notes=exp.notes)
        results.append({"run": exp.run_name, **metrics})
    return pd.DataFrame(results).set_index("run")


# Best run of each model family (by validation ROC-AUC, see experiments/runs.csv), compared to logistic regression
CANDIDATES = {
    "logreg": ("baselines", "logreg"),
    "random_forest": ("random_forest", "rf_leaf20_featsqrt"),
    "gradient_boosting": ("gradient_boosting", "hgb_lr0.03_leaves15_leaf200"),
}


def compare(n_boot: int = 1000) -> pd.DataFrame:
    """Refit the candidates and run a paired bootstrap on the validation set."""
    train, val, _ = time_split(build_dataset())
    cols = feature_columns()
    by_name = {e.run_name: e for exps in EXPERIMENTS.values() for e in exps}
    scores = {}
    for label, (_, run_name) in CANDIDATES.items():
        model = by_name[run_name].build(False).fit(train[cols], train[TARGET])
        scores[label] = model.predict_proba(val[cols])[:, 1]
    result = paired_bootstrap(val[TARGET], scores, reference="logreg", n_boot=n_boot)
    result.round(4).to_csv(REPO_ROOT / "experiments" / "comparison.csv", index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS) + ["compare"])
    args = parser.parse_args()
    if args.experiment == "compare":
        print("Paired bootstrap on validation (difference to logistic regression, 1000 resamples):")
        print(compare().round(4).to_string(index=False))
        print("\nSaved to experiments/comparison.csv")
        raise SystemExit
    table = run(args.experiment)
    print(f"Validation results ({args.experiment}):")
    print(table.round(4).to_string())
    print("\nLogged to experiments/runs.csv")
