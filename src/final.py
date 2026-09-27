"""Train the final model and evaluate it ONCE on the test set, as pre-registered in docs/final_model.md.

Usage:
    python -m src.final

Steps:
1. Refit the final model (logistic regression), the comparison model (gradient boosting), and the
   baselines on train + validation.
2. Fix the risk-band cut-offs from the final model's probabilities on train + validation.
3. Score every model on the test set; bootstrap intervals; paired comparison; bands; calibration.
4. Save the final model with its cut-offs to models/final_model.joblib, and results to reports/.
"""
from __future__ import annotations

import json
from datetime import date

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.ticker import PercentFormatter
from sklearn.dummy import DummyClassifier

from src.evaluate import bootstrap_ci, evaluate, paired_bootstrap
from src.experiments import git_commit, log_run
from src.preprocess import REPO_ROOT, TARGET, build_dataset, feature_columns, time_split
from src.train import LeadTimeRule, gradient_boosting, logistic

MODEL_PATH = REPO_ROOT / "models" / "final_model.joblib"
REPORTS = REPO_ROOT / "reports"
BAND_PERCENTILES = (50, 80)          # Low < 50th <= Medium < 80th <= High
BANDS = ["Low", "Medium", "High"]

FINAL_PARAMS = {"C": 1.0}
COMPARISON_PARAMS = {"learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 200}


def risk_band(probability, cutoffs) -> np.ndarray:
    """Map no-show probabilities to Low / Medium / High using the saved cut-offs."""
    idx = np.searchsorted(np.asarray(cutoffs), np.atleast_1d(probability), side="right")
    return np.array(BANDS)[idx]


def reference_means(model, X: pd.DataFrame) -> list[float]:
    """Average of each encoded model input over the training rows. Predictions are explained
    relative to this 'average appointment' (see src/predict.py)."""
    return model[0].transform(X).mean(axis=0).round(6).tolist()


def band_table(y_true, bands) -> pd.DataFrame:
    frame = pd.DataFrame({"band": bands, "y": np.asarray(y_true)})
    table = frame.groupby("band")["y"].agg(appointments="size", no_shows="sum").reindex(BANDS)
    table["share_of_appointments"] = table["appointments"] / table["appointments"].sum()
    table["no_show_rate"] = table["no_shows"] / table["appointments"]
    table["share_of_all_no_shows"] = table["no_shows"] / table["no_shows"].sum()
    return table


def calibration_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability vs observed no-show rate, by decile of predicted probability."""
    frame = pd.DataFrame({"p": y_score, "y": np.asarray(y_true)})
    frame["decile"] = pd.qcut(frame["p"], n_bins, labels=False, duplicates="drop") + 1
    return frame.groupby("decile").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"),
                                       appointments=("y", "size"))


def plot_results(calibration: pd.DataFrame, bands: pd.DataFrame, path_prefix) -> None:
    blue, grey, ink2, grid = "#2a78d6", "#a8a7a2", "#52514e", "#e4e3df"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": grid,
                         "axes.grid": True, "grid.color": grid, "axes.axisbelow": True, "font.size": 10,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    lim = max(calibration["mean_predicted"].max(), calibration["observed_rate"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color=grey, ls="--", lw=1, label="Perfect calibration")
    ax.plot(calibration["mean_predicted"], calibration["observed_rate"], color=blue, lw=2, marker="o", ms=6,
            label="Final model (test set)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Mean predicted no-show probability (decile)"); ax.set_ylabel("Observed no-show rate")
    ax.set_title("Calibration on the test set"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(f"{path_prefix}_calibration.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(bands.index, bands["no_show_rate"], color=blue, width=0.6)
    ax.bar_label(bars, labels=[f"{r:.0%} no-shows\n{s:.0%} of appointments" for r, s in
                               zip(bands["no_show_rate"], bands["share_of_appointments"])],
                 padding=3, fontsize=8, color=ink2)
    ax.set_ylim(0, bands["no_show_rate"].max() * 1.35); ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("No-show rate"); ax.set_title("Risk bands on the test set")
    fig.tight_layout(); fig.savefig(f"{path_prefix}_risk_bands.png", dpi=130); plt.close(fig)


def main() -> dict:
    train, val, test = time_split(build_dataset())
    fit_data = pd.concat([train, val], ignore_index=True)
    cols = feature_columns()
    X_fit, y_fit, X_test, y_test = fit_data[cols], fit_data[TARGET], test[cols], test[TARGET]

    models = {
        "final_logreg": (logistic(False, **FINAL_PARAMS), FINAL_PARAMS),
        "comparison_hgb": (gradient_boosting(False, **COMPARISON_PARAMS), COMPARISON_PARAMS),
        "baseline_majority": (DummyClassifier(strategy="prior"), {"strategy": "prior"}),
        "baseline_lead_time_rule": (LeadTimeRule(), {}),
    }
    scores, metrics = {}, {}
    for name, (model, params) in models.items():
        model.fit(X_fit, y_fit)
        scores[name] = model.predict_proba(X_test)[:, 1]
        metrics[name] = evaluate(y_test, scores[name])
        log_run("final_test", name, type(model[-1] if hasattr(model, "steps") else model).__name__, params,
                False, metrics[name], split="test", notes="Trained on train+validation; pre-registered in docs/final_model.md")

    final = models["final_logreg"][0]
    cutoffs = np.percentile(final.predict_proba(X_fit)[:, 1], BAND_PERCENTILES)
    bands = band_table(y_test, risk_band(scores["final_logreg"], cutoffs))
    calibration = calibration_table(y_test, scores["final_logreg"])
    ci = bootstrap_ci(y_test, scores["final_logreg"])
    vs_hgb = paired_bootstrap(y_test, {"final_logreg": scores["final_logreg"], "comparison_hgb": scores["comparison_hgb"]},
                              reference="final_logreg")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": final,
        "features": cols,
        "band_cutoffs": cutoffs.tolist(),
        "band_percentiles": list(BAND_PERCENTILES),
        "reference_means": reference_means(final, X_fit),
        "params": FINAL_PARAMS,
        "trained_on": f"{fit_data['appointment_day'].min().date()} to {fit_data['appointment_day'].max().date()}",
        "training_rows": len(fit_data),
        "sklearn_version": sklearn.__version__,
        "git_commit": git_commit(),
        "created": date.today().isoformat(),
    }, MODEL_PATH)

    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    plot_results(calibration, bands, REPORTS / "figures" / "11")
    results = {
        "test_rows": len(test), "test_no_show_rate": float(y_test.mean()),
        "metrics": {k: {m: round(float(v), 4) for m, v in d.items()} for k, d in metrics.items()},
        "final_logreg_95ci": ci.round(4).to_dict(orient="index"),
        "hgb_minus_logreg": vs_hgb.round(4).to_dict(orient="records"),
        "band_cutoffs": [round(float(c), 4) for c in cutoffs],
        "bands": bands.round(4).reset_index().to_dict(orient="records"),
        "calibration": calibration.round(4).reset_index().to_dict(orient="records"),
    }
    (REPORTS / "test_results.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    r = main()
    print(f"Test set: {r['test_rows']:,} appointments, no-show rate {r['test_no_show_rate']:.1%}\n")
    print(pd.DataFrame(r["metrics"]).T.round(4).to_string())
    print("\nFinal model, 95% bootstrap intervals:")
    print(pd.DataFrame(r["final_logreg_95ci"]).T.to_string())
    print("\nGradient boosting minus logistic regression (paired bootstrap):")
    print(pd.DataFrame(r["hgb_minus_logreg"]).to_string(index=False))
    print(f"\nRisk-band cut-offs (probability): {r['band_cutoffs']}")
    print(pd.DataFrame(r["bands"]).to_string(index=False))
    print("\nCalibration by decile:")
    print(pd.DataFrame(r["calibration"]).to_string(index=False))
    print(f"\nSaved {MODEL_PATH.relative_to(REPO_ROOT)}, reports/test_results.json, reports/figures/11_*.png")
