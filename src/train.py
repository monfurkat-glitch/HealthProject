"""Train and compare models. Every run is scored on the validation set and logged.

Usage:
    python -m src.train baselines

The test set is NOT used here. It is kept untouched until the final model is chosen.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluate import evaluate
from src.experiments import log_run
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
}


def run(experiment: str) -> pd.DataFrame:
    train, val, _ = time_split(build_dataset())
    results = []
    for exp in EXPERIMENTS[experiment]:
        cols = feature_columns(include_sms=exp.include_sms)
        model = exp.build(exp.include_sms).fit(train[cols], train[TARGET])
        metrics = evaluate(val[TARGET], model.predict_proba(val[cols])[:, 1])
        log_run(experiment, exp.run_name, type(model[-1] if isinstance(model, Pipeline) else model).__name__,
                exp.params, exp.include_sms, metrics, notes=exp.notes)
        results.append({"run": exp.run_name, **metrics})
    return pd.DataFrame(results).set_index("run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    args = parser.parse_args()
    table = run(args.experiment)
    print(f"Validation results ({args.experiment}):")
    print(table.round(4).to_string())
    print("\nLogged to experiments/runs.csv")
