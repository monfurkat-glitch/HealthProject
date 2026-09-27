"""Predict the no-show risk of new appointments with the saved final model.

Usage (command line):
    python -m src.predict examples/appointment.json
    python -m src.predict examples/appointments.json --history     # use past visits from the dataset

Usage (Python):
    from src.predict import NoShowPredictor
    predictor = NoShowPredictor()
    predictor.predict({"age": 25, "gender": "F", "neighbourhood": "CENTRO",
                       "scheduled_day": "2016-06-01", "appointment_day": "2016-06-20"})

Every input is validated first. Invalid inputs raise ``InvalidAppointment`` with a message for
each problem; inputs that are valid but unusual (e.g. an unknown neighbourhood) are accepted and
returned with a warning.

The output is decision support for clinic staff: it never cancels, rebooks, or refuses care.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.final import MODEL_PATH, risk_band
from src.preprocess import TARGET, add_features, add_history_features, clean, load_raw

REQUIRED = ["age", "gender", "scheduled_day", "appointment_day"]
FLAGS = ["scholarship", "hypertension", "diabetes", "alcoholism"]
MAX_AGE = 115                 # oldest patient in the data
MAX_LEAD_DAYS_SEEN = 179      # longest wait in the training data
TRUE_WORDS, FALSE_WORDS = {"1", "true", "yes", "y"}, {"0", "false", "no", "n", ""}
GENDERS = {"F": "F", "FEMALE": "F", "M": "M", "MALE": "M"}


class InvalidAppointment(ValueError):
    """Raised when an appointment cannot be scored. ``errors`` lists every problem found."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid appointment: " + "; ".join(errors))


def _to_flag(value, name: str, errors: list[str], warnings: list[str], max_value: int = 1) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        warnings.append(name)            # collected into one "assumed 0" warning by validate()
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in TRUE_WORDS:
            return 1
        if text in FALSE_WORDS:
            return 0
        if text.isdigit():
            value = int(text)
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer() \
            and 0 <= value <= max_value:
        return int(value)
    errors.append(f"'{name}' must be {'0 or 1' if max_value == 1 else f'an integer from 0 to {max_value}'}"
                  f" (or yes/no), got {value!r}")
    return 0


def _to_date(value, name: str, errors: list[str]):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None                      # reported as missing elsewhere
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        errors.append(f"'{name}' is not a valid date: {value!r} (use e.g. 2016-06-01)")
        return None
    if ts is pd.NaT:
        errors.append(f"'{name}' is not a valid date: {value!r}")
        return None
    return ts.tz_convert(None) if ts.tzinfo else ts


def validate(appointment: dict, known_neighbourhoods: set[str]) -> tuple[dict, list[str]]:
    """Check one raw appointment and convert it to the cleaned format.

    Returns (clean record, warnings). Raises InvalidAppointment listing every error.
    """
    if not isinstance(appointment, dict):
        raise InvalidAppointment([f"expected a dictionary of fields, got {type(appointment).__name__}"])
    a = {str(k).strip().lower(): v for k, v in appointment.items()}
    errors, warnings = [], []

    for field in REQUIRED:
        if a.get(field) is None or (isinstance(a.get(field), str) and not a[field].strip()):
            errors.append(f"'{field}' is required")

    age = a.get("age")
    if age is not None and not (isinstance(age, str) and not age.strip()):
        try:
            age_num = float(age)
            if not age_num.is_integer():
                raise ValueError
            age = int(age_num)
            if age < 0 or age > MAX_AGE:
                errors.append(f"'age' must be between 0 and {MAX_AGE}, got {age}")
            elif age > 100:
                warnings.append(f"age {age} is very rare in the training data; the prediction is less reliable")
        except (ValueError, TypeError):
            errors.append(f"'age' must be a whole number, got {age!r}")

    gender = a.get("gender")
    if gender is not None and str(gender).strip():
        gender = GENDERS.get(str(gender).strip().upper())
        if gender is None:
            errors.append(f"'gender' must be F or M, got {a['gender']!r}")

    scheduled = _to_date(a.get("scheduled_day"), "scheduled_day", errors)
    appointment_day = _to_date(a.get("appointment_day"), "appointment_day", errors)
    if scheduled is not None and appointment_day is not None:
        lead = (appointment_day.normalize() - scheduled.normalize()).days
        if lead < 0:
            errors.append(f"'appointment_day' ({appointment_day.date()}) is before 'scheduled_day' "
                          f"({scheduled.date()}): an appointment cannot be booked after it happens")
        elif lead > MAX_LEAD_DAYS_SEEN:
            warnings.append(f"booked {lead} days ahead; the training data only goes up to {MAX_LEAD_DAYS_SEEN} "
                            "days, so the prediction is less reliable")
        if appointment_day.dayofweek == 6:
            warnings.append("appointment on a Sunday: no Sunday appointments in the training data")

    neighbourhood = str(a.get("neighbourhood") or "").strip().upper()
    if not neighbourhood:
        warnings.append("'neighbourhood' missing: treated as an unknown neighbourhood")
    elif neighbourhood not in known_neighbourhoods:
        warnings.append(f"neighbourhood {neighbourhood!r} not seen in training: treated as a rare neighbourhood")

    missing_flags: list[str] = []
    flags = {f: _to_flag(a.get(f), f, errors, missing_flags) for f in FLAGS}
    handicap = _to_flag(a.get("handicap", a.get("handcap")), "handicap", errors, missing_flags, max_value=4)
    if missing_flags:
        warnings.append(f"not given, assumed 0 (no): {', '.join(missing_flags)}")

    prior = {}
    for field in ("prior_appointments", "prior_no_shows"):
        if a.get(field) is not None:
            try:
                value = float(a[field])
                if not value.is_integer() or value < 0:
                    raise ValueError
                prior[field] = int(value)
            except (ValueError, TypeError):
                errors.append(f"'{field}' must be a whole number of 0 or more, got {a[field]!r}")
    if len(prior) == 1:
        errors.append("give both 'prior_appointments' and 'prior_no_shows', or neither")
    elif len(prior) == 2 and prior["prior_no_shows"] > prior["prior_appointments"]:
        errors.append("'prior_no_shows' cannot be larger than 'prior_appointments'")

    if errors:
        raise InvalidAppointment(errors)

    record = {
        "patient_id": str(a.get("patient_id") or "new-patient"),
        "gender": gender, "age": age, "neighbourhood": neighbourhood or "UNKNOWN",
        **flags, "handicap_count": handicap,
        "scheduled_day": scheduled, "appointment_day": appointment_day.normalize(),
        **prior,
    }
    return record, warnings


def load_history() -> pd.DataFrame:
    """Past appointments with known outcomes, used to look up a patient's attendance history.

    In this prototype the history is the Kaggle dataset itself; a clinic would use its own records.
    Only appointments dated before each new booking are ever counted (see add_history_features).
    """
    return clean(load_raw())[["patient_id", "appointment_day", TARGET]]


class NoShowPredictor:
    """Loads the saved model once and scores appointments."""

    def __init__(self, model_path: str | Path = MODEL_PATH, history: pd.DataFrame | None = None):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.cutoffs = bundle["band_cutoffs"]
        self.reference = np.asarray(bundle.get("reference_means", 0.0))
        self.info = {k: v for k, v in bundle.items() if k != "model"}
        self.history = history
        self.version_warning = None
        if bundle.get("sklearn_version") != sklearn.__version__:
            self.version_warning = (f"model saved with scikit-learn {bundle.get('sklearn_version')}, running "
                                    f"{sklearn.__version__}; install requirements.txt for identical results")
        encoder = self.model[0].named_transformers_["categorical"]
        self.known_neighbourhoods = set(encoder.categories_[list(encoder.feature_names_in_).index("neighbourhood")])

    def _features(self, records: list[dict]) -> pd.DataFrame:
        frame = pd.DataFrame(records)
        given = frame[["prior_appointments", "prior_no_shows"]].copy() \
            if {"prior_appointments", "prior_no_shows"} & set(frame.columns) else None
        frame = frame.drop(columns=["prior_appointments", "prior_no_shows"], errors="ignore")
        if self.history is not None:
            frame = add_history_features(frame, history=self.history)
        else:
            frame = frame.assign(prior_appointments=0, prior_no_shows=0)
        if given is not None:                              # explicit values override the lookup
            for col in given.columns:
                frame[col] = given[col].fillna(frame[col]).astype(int).to_numpy()
        frame["prior_no_show_rate"] = np.where(frame["prior_appointments"] > 0,
                                               frame["prior_no_shows"] / frame["prior_appointments"].clip(lower=1), 0.0)
        return add_features(frame)

    def explain(self, row: pd.DataFrame, top: int = 3) -> list[str]:
        """The features that move this appointment's risk most, compared with an average appointment.

        In logistic regression each encoded input adds coefficient x value to the log-odds, so
        coefficient x (value - training average) is how much this input moves the prediction away
        from the average appointment's.
        """
        pre, clf = self.model[0], self.model[-1]
        encoded = pre.transform(row[self.features])[0]
        contributions = (encoded - self.reference) * clf.coef_[0]
        r = row.iloc[0]
        phrases = []
        for i in np.argsort(-np.abs(contributions)):
            name = pre.get_feature_names_out()[i]
            if abs(contributions[i]) < 0.01 or len(phrases) == top:
                break
            if name.startswith(("neighbourhood_", "appointment_weekday_")) and encoded[i] == 0:
                continue                 # "not in neighbourhood X" is not a useful explanation
            phrases.append(f"{self._describe(name, r)} {'raises' if contributions[i] > 0 else 'lowers'} risk")
        return phrases

    @staticmethod
    def _describe(name: str, r: pd.Series) -> str:
        """Plain-language description of one model input for this appointment."""
        yes_no = {"same_day": ("booked for the same day", "not booked for the same day"),
                  "scholarship": ("welfare (Scholarship)", "no welfare"), "male": ("male", "female"),
                  "hypertension": ("hypertension", "no hypertension"), "diabetes": ("diabetes", "no diabetes"),
                  "alcoholism": ("alcoholism", "no alcoholism"), "disability": ("disability", "no disability")}
        if name in yes_no:
            return yes_no[name][0] if r[name] == 1 else yes_no[name][1]
        numeric = {"age": f"age {r['age']}", "lead_days": f"booked {r['lead_days']} days ahead",
                   "prior_appointments": f"{r['prior_appointments']} earlier appointments",
                   "prior_no_shows": f"{r['prior_no_shows']} earlier no-shows",
                   "prior_no_show_rate": f"earlier no-show rate {r['prior_no_show_rate']:.0%}"}
        if name in numeric:
            return numeric[name]
        if name.startswith("neighbourhood_"):
            return "rare or unknown neighbourhood" if name.endswith("infrequent_sklearn") \
                else f"neighbourhood {name.removeprefix('neighbourhood_').title()}"
        return f"appointment on a {r['appointment_weekday']}"

    def predict(self, appointments: dict | list[dict]) -> dict | list[dict]:
        """Score one appointment (dict) or several (list of dicts)."""
        single = isinstance(appointments, dict)
        items = [appointments] if single else list(appointments)
        if not items:
            return []
        records, warnings = [], []
        for i, item in enumerate(items):
            try:
                record, warn = validate(item, self.known_neighbourhoods)
            except InvalidAppointment as exc:
                if single:
                    raise
                raise InvalidAppointment([f"appointment {i}: {e}" for e in exc.errors]) from None
            records.append(record)
            warnings.append(warn + ([self.version_warning] if self.version_warning else []))

        features = self._features(records)
        probabilities = self.model.predict_proba(features[self.features])[:, 1]
        bands = risk_band(probabilities, self.cutoffs)
        results = []
        for i, (p, band) in enumerate(zip(probabilities, bands)):
            row = features.iloc[[i]]
            results.append({
                "no_show_probability": round(float(p), 4),
                "risk_band": str(band),
                "main_factors": self.explain(row),
                "lead_days": int(row["lead_days"].iloc[0]),
                "prior_appointments": int(row["prior_appointments"].iloc[0]),
                "prior_no_shows": int(row["prior_no_shows"].iloc[0]),
                "warnings": warnings[i],
            })
        return results[0] if single else results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict no-show risk for appointments in a JSON file.")
    parser.add_argument("json_file", help="a JSON object (one appointment) or a list of objects")
    parser.add_argument("--history", action="store_true",
                        help="look up each patient_id's past appointments in the dataset")
    args = parser.parse_args()
    appointments = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    predictor = NoShowPredictor(history=load_history() if args.history else None)
    try:
        print(json.dumps(predictor.predict(appointments), indent=2))
    except InvalidAppointment as exc:
        print("Cannot score this input:")
        for e in exc.errors:
            print(" -", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
