"""Tests for src/preprocess.py. Run with:  python -m pytest"""
import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    TARGET, add_history_features, build_dataset, build_preprocessor, clean, feature_columns, load_raw, time_split,
)


def make_raw(rows):
    """Build a tiny raw table in the original Kaggle format; each row sets only what the test needs."""
    base = {"PatientId": "1", "AppointmentID": 0, "Gender": "F", "ScheduledDay": "2016-05-01T10:00:00Z",
            "AppointmentDay": "2016-05-10T00:00:00Z", "Age": 30, "Neighbourhood": "CENTRO", "Scholarship": 0,
            "Hipertension": 0, "Diabetes": 0, "Alcoholism": 0, "Handcap": 0, "SMS_received": 0, "No-show": "No"}
    return pd.DataFrame([{**base, "AppointmentID": i, **r} for i, r in enumerate(rows)])


@pytest.fixture(scope="module")
def data():
    return build_dataset(load_raw())


# ---------- cleaning ----------

def test_clean_drops_invalid_rows():
    raw = make_raw([
        {},                                                                   # valid
        {"Age": -1},                                                          # invalid age
        {"ScheduledDay": "2016-05-11T09:00:00Z"},                             # booked after the appointment
        {"ScheduledDay": "2016-05-10T09:00:00Z"},                             # same-day booking: valid
    ])
    cleaned = clean(raw)
    assert list(cleaned["appointment_id"]) == [0, 3]


def test_clean_encodes_target_and_types():
    cleaned = clean(make_raw([{"No-show": "Yes"}, {"No-show": "No"}]))
    assert list(cleaned[TARGET]) == [1, 0]
    assert cleaned["patient_id"].dtype == object
    assert cleaned["appointment_day"].dt.hour.eq(0).all()


def test_real_data_cleaning(data):
    assert len(data) == 110_527 - 6
    assert (data["age"] >= 0).all()
    assert (data["lead_days"] >= 0).all()
    assert set(data["disability"].unique()) <= {0, 1}
    assert data[feature_columns(include_sms=True)].notna().all().all()


# ---------- patient history (leakage) ----------

def test_history_counts_only_appointments_before_booking_day():
    raw = make_raw([
        # patient 1: two past appointments (one no-show), then a new booking on 05-05
        {"ScheduledDay": "2016-04-20T08:00:00Z", "AppointmentDay": "2016-05-01T00:00:00Z", "No-show": "Yes"},
        {"ScheduledDay": "2016-04-20T08:00:00Z", "AppointmentDay": "2016-05-03T00:00:00Z", "No-show": "No"},
        {"ScheduledDay": "2016-05-05T08:00:00Z", "AppointmentDay": "2016-05-12T00:00:00Z", "No-show": "No"},
        # appointment ON the booking day: outcome not known yet at booking time, must not count
        {"ScheduledDay": "2016-05-01T08:00:00Z", "AppointmentDay": "2016-05-05T00:00:00Z", "No-show": "Yes"},
        # another patient: must never be mixed in
        {"PatientId": "2", "ScheduledDay": "2016-04-01T08:00:00Z", "AppointmentDay": "2016-04-02T00:00:00Z",
         "No-show": "Yes"},
    ])
    hist = add_history_features(clean(raw)).set_index("appointment_id")
    assert hist.loc[2, "prior_appointments"] == 2
    assert hist.loc[2, "prior_no_shows"] == 1
    assert hist.loc[2, "prior_no_show_rate"] == pytest.approx(0.5)
    assert hist.loc[0, "prior_appointments"] == 0      # booked before any appointment happened
    assert hist.loc[4, "prior_appointments"] == 0      # other patient's history is separate


def test_history_ignores_future_outcomes():
    """Changing the outcome of any appointment on/after the booking day must not change the features."""
    rows = [
        {"ScheduledDay": "2016-04-20T08:00:00Z", "AppointmentDay": "2016-05-01T00:00:00Z", "No-show": "Yes"},
        {"ScheduledDay": "2016-05-02T08:00:00Z", "AppointmentDay": "2016-05-09T00:00:00Z", "No-show": "No"},
        {"ScheduledDay": "2016-05-02T08:00:00Z", "AppointmentDay": "2016-05-20T00:00:00Z", "No-show": "Yes"},
    ]
    before = add_history_features(clean(make_raw(rows)))
    rows[2]["No-show"] = "No"      # flip a future outcome
    rows[1]["No-show"] = "Yes"     # and another appointment after the booking day
    after = add_history_features(clean(make_raw(rows)))
    cols = ["prior_appointments", "prior_no_shows"]
    # rows 1 and 2 were both booked on 05-02: only the 05-01 appointment is known
    pd.testing.assert_frame_equal(before[cols], after[cols])


def test_history_matches_brute_force_on_real_data(data):
    """Recompute history the slow, obvious way for a random sample and compare."""
    sample = data.sample(300, random_state=0)
    by_patient = {pid: g for pid, g in data.groupby("patient_id")}
    for _, row in sample.iterrows():
        g = by_patient[row["patient_id"]]
        known = g[g["appointment_day"] < row["scheduled_day"].normalize()]
        assert row["prior_appointments"] == len(known)
        assert row["prior_no_shows"] == known[TARGET].sum()


def test_history_uses_separate_history_table():
    """At prediction time, history comes from a table of past appointments."""
    past = clean(make_raw([
        {"ScheduledDay": "2016-04-01T08:00:00Z", "AppointmentDay": "2016-04-05T00:00:00Z", "No-show": "Yes"},
    ]))
    new = clean(make_raw([{"AppointmentID": 99, "ScheduledDay": "2016-05-01T08:00:00Z"},
                          {"PatientId": "unknown", "ScheduledDay": "2016-05-01T08:00:00Z"}]))
    out = add_history_features(new, history=past)
    assert list(out["prior_no_shows"]) == [1, 0]      # new patient gets zeros, not NaN


# ---------- split ----------

def test_time_split_is_chronological_and_complete(data):
    train, val, test = time_split(data)
    assert len(train) + len(val) + len(test) == len(data)
    assert train["appointment_day"].max() < val["appointment_day"].min()
    assert val["appointment_day"].max() < test["appointment_day"].min()
    assert not set(train["appointment_id"]) & set(test["appointment_id"])
    for part in (train, val, test):
        assert 0.10 < len(part) / len(data)
        assert part[TARGET].sum() > 1000             # enough no-shows to evaluate


# ---------- encoder ----------

def test_preprocessor_handles_unseen_and_rare_categories(data):
    train, _, _ = time_split(data)
    pre = build_preprocessor().fit(train[feature_columns()])
    new = train[feature_columns()].head(2).copy()
    new["neighbourhood"] = ["NOT A REAL PLACE", "PARQUE INDUSTRIAL"]
    X = pre.transform(new)
    assert X.shape[0] == 2 and np.isfinite(X).all()
    # both land in the same "infrequent" column group, so they encode identically
    names = list(pre.get_feature_names_out())
    nb_cols = [i for i, n in enumerate(names) if n.startswith("neighbourhood_")]
    assert (X[0, nb_cols] == X[1, nb_cols]).all()


def test_sms_is_excluded_by_default():
    assert "sms_received" not in feature_columns()
    assert "sms_received" in feature_columns(include_sms=True)
