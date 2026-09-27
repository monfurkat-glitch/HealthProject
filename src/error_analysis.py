"""Error analysis and fairness checks for the final model on the test set.

Usage:
    python -m src.error_analysis

Analysis only: the model and its risk bands are loaded from models/final_model.joblib and
nothing is re-tuned. "Flagged" means the appointment falls in the High risk band, which is the
group staff would follow up.

Outputs: reports/error_analysis.json and reports/figures/12_*.png, 13_*.png, 14_*.png
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_auc_score

from src.final import MODEL_PATH, risk_band
from src.preprocess import REPO_ROOT, TARGET, build_dataset, time_split

REPORTS = REPO_ROOT / "reports"
MIN_GROUP = 30          # smaller groups are reported but not interpreted

BLUE, GREY, INK2, GRID = "#2a78d6", "#a8a7a2", "#52514e", "#e4e3df"


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Readable groups to slice the results by."""
    out = df.copy()
    out["lead_time"] = pd.cut(out["lead_days"], [-1, 0, 2, 7, 14, 30, 1000],
                              labels=["same day", "1-2 days", "3-7 days", "8-14 days", "15-30 days", "31+ days"])
    out["age_group"] = pd.cut(out["age"], [-1, 12, 17, 25, 45, 65, 200],
                              labels=["0-12", "13-17", "18-25", "26-45", "46-65", "66+"])
    out["gender"] = out["male"].map({1: "male", 0: "female"})
    out["welfare"] = out["scholarship"].map({1: "welfare (Scholarship)", 0: "no welfare"})
    out["history"] = np.select([out["prior_appointments"] == 0, out["prior_no_shows"] == 0],
                               ["no known history", "history, no prior no-show"], "history, 1+ prior no-show")
    return out


def slice_metrics(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Per group: size, no-show rate, ranking quality, calibration, and error rates of the High band."""
    rows = []
    for name, g in df.groupby(group, observed=True):
        y, p, flag = g[TARGET].to_numpy(), g["probability"].to_numpy(), g["flagged"].to_numpy()
        rows.append({
            group: name,
            "appointments": len(g),
            "no_show_rate": y.mean(),
            "mean_predicted": p.mean(),
            "roc_auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
            "flag_rate": flag.mean(),                                           # share put in High band
            "recall_in_high": flag[y == 1].mean() if y.sum() else np.nan,       # no-shows caught (TPR)
            "false_alarm_rate": flag[y == 0].mean() if (y == 0).any() else np.nan,  # attenders flagged (FPR)
            "precision_in_high": y[flag].mean() if flag.any() else np.nan,
        })
    return pd.DataFrame(rows).set_index(group)


def coefficients(model) -> pd.DataFrame:
    """Logistic regression coefficients as odds ratios. Numeric features are standardised, so their
    odds ratio is per one standard deviation; binary and one-hot features are per 'yes'."""
    names = model[:-1].get_feature_names_out()
    coef = model[-1].coef_[0]
    table = pd.DataFrame({"feature": names, "coefficient": coef, "odds_ratio": np.exp(coef)})
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def error_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average characteristics of each outcome type (flag = High band)."""
    kind = np.select(
        [df["flagged"] & (df[TARGET] == 1), df["flagged"] & (df[TARGET] == 0),
         ~df["flagged"] & (df[TARGET] == 1), ~df["flagged"] & (df[TARGET] == 0)],
        ["caught no-show (TP)", "false alarm (FP)", "missed no-show (FN)", "correctly not flagged (TN)"])
    cols = {"appointments": ("age", "size"), "mean_age": ("age", "mean"), "median_lead_days": ("lead_days", "median"),
            "same_day_share": ("same_day", "mean"), "welfare_share": ("scholarship", "mean"),
            "has_history_share": ("prior_appointments", lambda s: (s > 0).mean()),
            "mean_probability": ("probability", "mean")}
    return df.assign(outcome=kind).groupby("outcome").agg(**cols)


def _style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                         "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "font.size": 10,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})


def plot_coefficients(coef: pd.DataFrame, path) -> None:
    top = coef[~coef["feature"].str.startswith("neighbourhood_")].head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = [BLUE if c > 0 else GREY for c in top["coefficient"]]
    bars = ax.barh(top["feature"], top["odds_ratio"] - 1, left=1, color=colors, height=0.65)
    ax.bar_label(bars, labels=[f"×{v:.2f}" for v in top["odds_ratio"]], padding=3, fontsize=8, color=INK2)
    ax.axvline(1, color=INK2, lw=1); ax.grid(axis="y", visible=False)
    lo, hi = top["odds_ratio"].min(), top["odds_ratio"].max()
    ax.set_xlim(min(lo, 1) - 0.15, max(hi, 1) + 0.25)
    ax.set_xlabel("Odds ratio (blue = raises no-show odds; numeric features: per 1 standard deviation)")
    ax.set_title("What the final model learned (excluding neighbourhoods)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_slices(slices: dict[str, pd.DataFrame], overall_auc: float, path) -> None:
    groups = ["lead_time", "age_group", "history"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True, gridspec_kw={"width_ratios": [6, 6, 3]})
    for ax, g in zip(axes, groups):
        t = slices[g]
        bars = ax.bar(range(len(t)), t["roc_auc"], color=BLUE, width=0.65)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in t["roc_auc"]], padding=2, fontsize=8, color=INK2)
        ax.axhline(overall_auc, color=INK2, ls="--", lw=1, label=f"Overall {overall_auc:.3f}")
        ax.set_xticks(range(len(t)), [str(i).replace(", ", ",\n") for i in t.index], fontsize=8)
        ax.set_title(g.replace("_", " ").capitalize()); ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0.5, 0.8); axes[0].set_ylabel("ROC-AUC within the group"); axes[0].legend(loc="upper right")
    fig.suptitle("Ranking quality inside each segment (test set)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_fairness(slices: dict[str, pd.DataFrame], path) -> None:
    rows = []
    for g in ["gender", "welfare", "age_group"]:
        for name, r in slices[g].iterrows():
            rows.append((f"{name}", r["recall_in_high"], r["false_alarm_rate"], r["no_show_rate"]))
    labels, tpr, fpr, rate = zip(*rows)
    y = np.arange(len(labels)); h = 0.38
    fig, ax = plt.subplots(figsize=(8, 5.2))
    b1 = ax.barh(y - h / 2 - 0.01, tpr, h, color=BLUE, label="No-shows caught (recall in High band)")
    b2 = ax.barh(y + h / 2 + 0.01, fpr, h, color=GREY, label="Attenders flagged (false-alarm rate)")
    for bb in (b1, b2):
        ax.bar_label(bb, labels=[f"{v:.0%}" for v in bb.datavalues], padding=3, fontsize=8, color=INK2)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    for sep in (1.5, 3.5):
        ax.axhline(sep, color=GRID, lw=1.2)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.set_xlim(0, max(tpr + fpr) * 1.25)
    ax.set_title("Fairness check: error rates by group (test set)"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> dict:
    bundle = joblib.load(MODEL_PATH)
    model, cutoffs = bundle["model"], bundle["band_cutoffs"]
    _, _, test = time_split(build_dataset())
    test = add_segments(test)
    test["probability"] = model.predict_proba(test[bundle["features"]])[:, 1]
    test["band"] = risk_band(test["probability"], cutoffs)
    test["flagged"] = test["band"] == "High"

    overall_auc = roc_auc_score(test[TARGET], test["probability"])
    groups = ["lead_time", "age_group", "gender", "welfare", "history", "appointment_weekday"]
    slices = {g: slice_metrics(test, g) for g in groups}
    nb = slice_metrics(test, "neighbourhood")
    coef = coefficients(model)
    profile = error_profile(test)

    y, flag = test[TARGET], test["flagged"]
    confusion = {"caught_no_shows_TP": int((flag & (y == 1)).sum()), "false_alarms_FP": int((flag & (y == 0)).sum()),
                 "missed_no_shows_FN": int((~flag & (y == 1)).sum()), "correct_not_flagged_TN": int((~flag & (y == 0)).sum())}

    _style()
    figs = REPORTS / "figures"
    plot_coefficients(coef, figs / "12_coefficients.png")
    plot_slices(slices, overall_auc, figs / "13_segments.png")
    plot_fairness(slices, figs / "14_fairness.png")

    to_json = lambda t: json.loads(t.round(4).reset_index().to_json(orient="records"))
    results = {
        "overall_roc_auc": round(overall_auc, 4),
        "confusion_high_band": confusion,
        "coefficients": to_json(coef),
        "error_profile": to_json(profile),
        "slices": {g: to_json(t) for g, t in slices.items()},
        "neighbourhoods": {"groups": int(len(nb)),
                           "roc_auc_range_min_300": [round(float(v), 4) for v in
                                                     nb.loc[nb["appointments"] >= 300, "roc_auc"].agg(["min", "max"])],
                           "top_by_no_show_rate": to_json(nb[nb["appointments"] >= 100].nlargest(5, "no_show_rate")),
                           "unseen_in_training": sorted(set(test["neighbourhood"]) -
                                                        set(pd.concat(time_split(build_dataset())[:2])["neighbourhood"]))},
    }
    (REPORTS / "error_analysis.json").write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    r = main()
    pd.set_option("display.width", 200)
    print(f"Overall test ROC-AUC: {r['overall_roc_auc']}\nHigh-band confusion: {r['confusion_high_band']}\n")
    print("Top coefficients:"); print(pd.DataFrame(r["coefficients"]).head(20).to_string(index=False))
    print("\nError profile:"); print(pd.DataFrame(r["error_profile"]).to_string(index=False))
    for g, rows in r["slices"].items():
        print(f"\n{g}:"); print(pd.DataFrame(rows).to_string(index=False))
    print("\nNeighbourhoods:", {k: v for k, v in r["neighbourhoods"].items() if k != "top_by_no_show_rate"})
    print(pd.DataFrame(r["neighbourhoods"]["top_by_no_show_rate"]).to_string(index=False))
