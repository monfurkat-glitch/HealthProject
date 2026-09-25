"""Data loading, cleaning, feature engineering, and splitting for no-show prediction.

The same functions are used for training and for prediction, so a new appointment
goes through exactly the same steps as the training data.

Prediction moment: at booking time. Every feature must be known when the
appointment is booked (``scheduled_day``). See docs/project_brief.md, section 12.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "Data" / "KaggleV2-May-2016.csv"
DATA_URL = "https://raw.githubusercontent.com/monfurkat-glitch/HealthProject/main/Data/KaggleV2-May-2016.csv"

TARGET = "no_show"

# Original Kaggle column -> clean snake_case name
COLUMN_NAMES = {
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "ScheduledDay": "scheduled_day",
    "AppointmentDay": "appointment_day",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship",
    "Hipertension": "hypertension",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handicap_count",
    "SMS_received": "sms_received",
    "No-show": TARGET,
}

# Chronological split by appointment date (see notebooks/01_eda.ipynb, section 9).
# The train/validation boundary falls in a 4-day gap with no appointments (2016-05-26 to 05-29).
VAL_START = pd.Timestamp("2016-05-30")
TEST_START = pd.Timestamp("2016-06-03")

# Model features, grouped by how they are encoded.
NUMERIC_FEATURES = ["age", "lead_days", "prior_appointments", "prior_no_shows", "prior_no_show_rate"]
BINARY_FEATURES = ["male", "scholarship", "hypertension", "diabetes", "alcoholism", "disability", "same_day"]
CATEGORICAL_FEATURES = ["neighbourhood", "appointment_weekday"]
# SMS reminders are sent after booking and are the very action this model is meant to target,
# so SMS status is excluded by default. It can be switched on for comparison experiments.
SMS_FEATURE = "sms_received"

# Neighbourhoods with fewer training appointments than this are grouped together.
MIN_NEIGHBOURHOOD_COUNT = 50


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the Kaggle CSV from the repo, or from GitHub if the file is not available (e.g. in Colab)."""
    source = Path(path) if path else DATA_PATH
    if not source.exists():
        source = DATA_URL
    return pd.read_csv(source, dtype={"PatientId": str})


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, fix types, and drop rows that are impossible in real use.

    Dropped: negative age, and appointments dated before their booking day.
    """
    df = raw.rename(columns=COLUMN_NAMES).copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["scheduled_day"] = pd.to_datetime(df["scheduled_day"]).dt.tz_localize(None)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"]).dt.tz_localize(None).dt.normalize()
    df["neighbourhood"] = df["neighbourhood"].str.strip().str.upper()
    df["gender"] = df["gender"].str.strip().str.upper()
    if df[TARGET].dtype == object:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)

    booked_after = df["appointment_day"] < df["scheduled_day"].dt.normalize()
    valid = (df["age"] >= 0) & ~booked_after
    return df.loc[valid].reset_index(drop=True)


def add_history_features(df: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add each patient's attendance history as known at booking time.

    Only appointments dated strictly before the booking day are counted: their
    outcome was known when the new appointment was booked. Later appointments,
    including ones on the booking day itself, are ignored, which prevents leakage.

    ``history`` holds past appointments with known outcomes (``patient_id``,
    ``appointment_day``, ``no_show``). By default it is ``df`` itself, which is
    safe because of the "strictly before the booking day" rule.
    """
    history = df if history is None else history
    # Per patient and day: appointments and no-shows, accumulated over time
    daily = (history.groupby(["patient_id", "appointment_day"])[TARGET]
             .agg(n="size", ns="sum").reset_index())
    daily["prior_appointments"] = daily.groupby("patient_id")["n"].cumsum()
    daily["prior_no_shows"] = daily.groupby("patient_id")["ns"].cumsum()

    out = df.copy()
    out["_row"] = np.arange(len(out))
    out["_booking_day"] = out["scheduled_day"].dt.normalize()
    merged = pd.merge_asof(
        out.sort_values("_booking_day"),
        daily.sort_values("appointment_day")[["patient_id", "appointment_day", "prior_appointments", "prior_no_shows"]]
             .rename(columns={"appointment_day": "_history_day"}),
        left_on="_booking_day", right_on="_history_day", by="patient_id",
        direction="backward", allow_exact_matches=False,  # strictly before the booking day
    ).sort_values("_row")

    out["prior_appointments"] = merged["prior_appointments"].fillna(0).astype(int).to_numpy()
    out["prior_no_shows"] = merged["prior_no_shows"].fillna(0).astype(int).to_numpy()
    out["prior_no_show_rate"] = np.where(
        out["prior_appointments"] > 0, out["prior_no_shows"] / out["prior_appointments"].clip(lower=1), 0.0)
    return out.drop(columns=["_row", "_booking_day"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features derived from a single appointment (no other rows needed)."""
    out = df.copy()
    out["lead_days"] = (out["appointment_day"] - out["scheduled_day"].dt.normalize()).dt.days
    out["same_day"] = (out["lead_days"] == 0).astype(int)
    out["appointment_weekday"] = out["appointment_day"].dt.day_name()
    out["male"] = (out["gender"] == "M").astype(int)
    out["disability"] = (out["handicap_count"] > 0).astype(int)
    return out


def build_dataset(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw CSV -> cleaned table with all features and the target."""
    raw = load_raw() if raw is None else raw
    return add_features(add_history_features(clean(raw)))


def time_split(df: pd.DataFrame, val_start: pd.Timestamp = VAL_START,
               test_start: pd.Timestamp = TEST_START) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically by appointment day: train < val_start <= validation < test_start <= test."""
    day = df["appointment_day"]
    return (df[day < val_start].reset_index(drop=True),
            df[(day >= val_start) & (day < test_start)].reset_index(drop=True),
            df[day >= test_start].reset_index(drop=True))


def feature_columns(include_sms: bool = False) -> list[str]:
    """Names of the model input columns."""
    return NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + ([SMS_FEATURE] if include_sms else [])


def build_preprocessor(scale_numeric: bool = True, include_sms: bool = False) -> ColumnTransformer:
    """Encoder for model inputs. It is fitted on training data only, inside the model pipeline.

    - numeric: standardised (needed by logistic regression, harmless for trees);
    - categorical: one-hot; rare and unseen neighbourhoods fall into one "infrequent" group;
    - binary: passed through unchanged.
    """
    numeric = StandardScaler() if scale_numeric else "passthrough"
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_NEIGHBOURHOOD_COUNT,
                                sparse_output=False)
    binary = BINARY_FEATURES + ([SMS_FEATURE] if include_sms else [])
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES),
         ("categorical", categorical, CATEGORICAL_FEATURES),
         ("binary", "passthrough", binary)],
        verbose_feature_names_out=False,
    )


def split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Size, date range, and no-show rate of each split."""
    rows = {}
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        rows[name] = {
            "rows": len(part),
            "share": len(part) / (len(train) + len(val) + len(test)),
            "first day": part["appointment_day"].min().date(),
            "last day": part["appointment_day"].max().date(),
            "no-show rate": part[TARGET].mean(),
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    raw = load_raw()
    data = build_dataset(raw)
    print(f"Raw rows: {len(raw):,}   after cleaning: {len(data):,}   dropped: {len(raw) - len(data)}")
    train, val, test = time_split(data)
    summary = split_summary(train, val, test)
    summary["share"] = summary["share"].map("{:.1%}".format)
    summary["no-show rate"] = summary["no-show rate"].map("{:.1%}".format)
    print(summary.to_string())
    print(f"\nFeatures ({len(feature_columns())}): {feature_columns()}")
