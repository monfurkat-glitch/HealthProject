"""Tests for src/predict.py: saved-model reload, validation, edge cases, and consistency with training."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.final import MODEL_PATH
from src.predict import InvalidAppointment, NoShowPredictor, load_history
from src.preprocess import REPO_ROOT, build_dataset, load_raw, time_split

VALID = {"patient_id": "p1", "age": 30, "gender": "F", "neighbourhood": "CENTRO", "scholarship": 0,
         "hypertension": 0, "diabetes": 0, "alcoholism": 0, "handicap": 0,
         "scheduled_day": "2016-06-01", "appointment_day": "2016-06-15"}


@pytest.fixture(scope="module")
def predictor():
    return NoShowPredictor()


def with_(**changes):
    return {**VALID, **changes}


# ---------- saved model ----------

def test_saved_model_reloads_with_metadata():
    bundle = joblib.load(MODEL_PATH)
    assert {"model", "features", "band_cutoffs", "reference_means", "sklearn_version"} <= set(bundle)
    assert bundle["band_cutoffs"][0] < bundle["band_cutoffs"][1]


def test_valid_appointment(predictor):
    r = predictor.predict(VALID)
    assert 0 < r["no_show_probability"] < 1
    assert r["risk_band"] in {"Low", "Medium", "High"}
    assert r["lead_days"] == 14 and r["warnings"] == []
    assert 1 <= len(r["main_factors"]) <= 3


def test_prediction_matches_training_pipeline(predictor):
    """Raw rows scored through predict() must get exactly the probabilities of the training pipeline."""
    raw = load_raw()
    data = build_dataset(raw)
    _, _, test = time_split(data)
    sample = test.sample(40, random_state=1)
    raw_rows = raw.set_index("AppointmentID").loc[sample["appointment_id"]].reset_index()
    inputs = [{"patient_id": r.PatientId, "age": r.Age, "gender": r.Gender, "neighbourhood": r.Neighbourhood,
               "scholarship": r.Scholarship, "hypertension": r.Hipertension, "diabetes": r.Diabetes,
               "alcoholism": r.Alcoholism, "handicap": r.Handcap, "scheduled_day": r.ScheduledDay,
               "appointment_day": r.AppointmentDay} for r in raw_rows.itertuples()]
    with_history = NoShowPredictor(history=load_history())
    got = np.array([r["no_show_probability"] for r in with_history.predict(inputs)])
    expected = predictor.model.predict_proba(sample[predictor.features])[:, 1]
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_batch_prediction(predictor):
    out = predictor.predict([VALID, with_(age=70)])
    assert len(out) == 2 and predictor.predict([]) == []


def test_example_files_run(predictor):
    for name in ("appointment.json", "appointments.json"):
        data = json.loads((REPO_ROOT / "examples" / name).read_text(encoding="utf-8"))
        assert predictor.predict(data)
    with pytest.raises(InvalidAppointment):
        predictor.predict(json.loads((REPO_ROOT / "examples" / "invalid_appointment.json").read_text()))


# ---------- sensible behaviour ----------

def test_same_day_booking_is_lower_risk(predictor):
    same_day = predictor.predict(with_(appointment_day="2016-06-01"))
    later = predictor.predict(with_(appointment_day="2016-06-30"))
    assert same_day["no_show_probability"] < later["no_show_probability"]


def test_history_raises_risk(predictor):
    base = predictor.predict(VALID)["no_show_probability"]
    missed = predictor.predict(with_(prior_appointments=4, prior_no_shows=4))["no_show_probability"]
    assert missed > base


# ---------- invalid inputs: clear errors ----------

@pytest.mark.parametrize("changes, message", [
    ({"age": -1}, "'age' must be between"),
    ({"age": 130}, "'age' must be between"),
    ({"age": "thirty"}, "'age' must be a whole number"),
    ({"age": 30.5}, "'age' must be a whole number"),
    ({"age": None}, "'age' is required"),
    ({"age": ""}, "'age' is required"),
    ({"gender": "X"}, "'gender' must be F or M"),
    ({"scheduled_day": "not a date"}, "not a valid date"),
    ({"appointment_day": None}, "'appointment_day' is required"),
    ({"appointment_day": "2016-05-01"}, "before 'scheduled_day'"),
    ({"diabetes": 2}, "'diabetes' must be 0 or 1"),
    ({"handicap": 7}, "'handicap' must be an integer from 0 to 4"),
    ({"prior_appointments": 2, "prior_no_shows": 3}, "cannot be larger"),
    ({"prior_no_shows": 1}, "give both"),
    ({"prior_appointments": -1, "prior_no_shows": 0}, "0 or more"),
])
def test_invalid_inputs_raise_clear_errors(predictor, changes, message):
    with pytest.raises(InvalidAppointment) as exc:
        predictor.predict(with_(**changes))
    assert any(message in e for e in exc.value.errors)


def test_all_errors_reported_at_once(predictor):
    with pytest.raises(InvalidAppointment) as exc:
        predictor.predict(with_(age=-5, gender="?", diabetes="maybe"))
    assert len(exc.value.errors) == 3


def test_not_a_dictionary(predictor):
    with pytest.raises(InvalidAppointment):
        predictor.predict(["not", "a", "dict"])


def test_batch_error_names_the_row(predictor):
    with pytest.raises(InvalidAppointment) as exc:
        predictor.predict([VALID, with_(age=-1)])
    assert exc.value.errors[0].startswith("appointment 1:")


# ---------- unusual but valid inputs: prediction plus warning ----------

@pytest.mark.parametrize("changes, warning", [
    ({"neighbourhood": "ATLANTIS"}, "not seen in training"),
    ({"neighbourhood": ""}, "'neighbourhood' missing"),
    ({"diabetes": None}, "assumed 0"),
    ({"appointment_day": "2017-06-01"}, "less reliable"),
    ({"age": 105}, "very rare"),
    ({"appointment_day": "2016-06-05"}, "Sunday"),
])
def test_unusual_inputs_give_warnings(predictor, changes, warning):
    r = predictor.predict(with_(**changes))
    assert 0 < r["no_show_probability"] < 1
    assert any(warning in w for w in r["warnings"])


def test_flexible_formats(predictor):
    a = predictor.predict(with_(gender="female", scholarship="yes", diabetes=True, neighbourhood=" centro "))
    b = predictor.predict(with_(gender="F", scholarship=1, diabetes=1, neighbourhood="CENTRO"))
    assert a["no_show_probability"] == b["no_show_probability"] and a["warnings"] == []


def test_timezone_aware_dates(predictor):
    a = predictor.predict(with_(scheduled_day="2016-06-01T10:00:00Z", appointment_day="2016-06-15T00:00:00Z"))
    assert a["lead_days"] == 14


def test_extreme_ages_stay_valid_probabilities(predictor):
    for age in (0, 115):
        p = predictor.predict(with_(age=age))["no_show_probability"]
        assert 0 < p < 1
