# MED-01 | Hospital Appointment No-Show Prediction

A machine learning model that predicts each outpatient appointment's **no-show risk** before it happens, so clinic staff can focus reminders, confirmations and follow-up calls on the appointments most likely to be missed.

The model only **informs** staff. It never cancels or rebooks appointments automatically.

| | |
|---|---|
| **Student** | Furkat Toshmatov |
| **Project track** | _To confirm_ |
| **Domain** | MedTech |
| **ML task** | Supervised binary classification |
| **Repository** | https://github.com/monfurkat-glitch/HealthProject |

> 🚧 Work in progress. Progress against the capstone criteria is tracked in [docs/ROADMAP.md](docs/ROADMAP.md) and [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Problem statement

Unannounced no-shows waste clinic capacity and lengthen waiting times for other patients. Without a risk estimate, staff spread reminder efforts evenly across all appointments instead of targeting the ones most likely to be missed. See the full [project brief](docs/project_brief.md).

## ML task

| | |
|---|---|
| **Input** | One scheduled appointment: patient age, gender, neighbourhood, welfare status, health conditions, booking date, appointment date, and reminder status |
| **Target** | `No-show` (`Yes` = the patient did not attend) |
| **Output** | Probability of a no-show, plus a Low / Medium / High risk band |
| **Prediction moment** | Before the appointment, using only information known at that time |

## Success criteria

Measured on the held-out test period (the latest appointments, never used for training or tuning):

1. The final model beats both baselines (always "shows up" and logistic regression) on ROC-AUC and PR-AUC.
2. Among the 20% of appointments ranked riskiest, the model catches a larger share of real no-shows than the logistic regression baseline.
3. No leakage: every feature is computable at prediction time, verified by an automated test.
4. The Colab demo runs end to end in a fresh runtime from this repository alone.

## Dataset

[Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments) (Kaggle): 110,527 appointments from public health clinics in Vitória, Brazil (April–June 2016). The CSV is included in [`Data/`](Data/) under its CC BY-NC-SA 4.0 license; see [Data/README.md](Data/README.md) for the column dictionary, license, and known data issues.

### Exploratory data analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/monfurkat-glitch/HealthProject/blob/main/notebooks/01_eda.ipynb)

The full analysis, with charts, is in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). It runs in Google Colab with no setup. Charts are saved in [`reports/figures/`](reports/figures/).

![No-show rate by lead time](reports/figures/02_lead_time.png)
![SMS reminders are confounded with lead time](reports/figures/03_sms_confounding.png)

### Key data findings so far

- 62,299 patients, no missing values, no duplicate rows. 20.2% of appointments are no-shows.
- Invalid records: 1 row with `Age = -1`, and 5 appointments booked after the appointment date.
- `Handcap` is a 0–4 count, not a yes/no flag.
- Appointment dates only span about 6 weeks (2016-04-29 to 2016-06-08).
- Lead time (days between booking and appointment) is the strongest signal found so far: same-day bookings have a 4.6% no-show rate, against 28.5% for all other appointments.
- SMS reminders are confounded with lead time (they were only sent for bookings made 3 or more days ahead). Raw, SMS looks linked to *more* no-shows (27.6% vs 16.7%); within the same lead-time group it is linked to 3–8 points *fewer*.
- No-show rate peaks for ages 13–17 (27%) and is lowest for ages 66–80 (15%). Health flags, gender, and weekday each move it by less than 4 points.
- Leakage-safe patient history is only a modest signal: just 28% of appointments have any known prior appointment.

## Pipeline

```
raw appointment ─► clean() ─► add_history_features() ─► add_features() ─► encoder (fitted on train) ─► model ─► probability ─► risk band
```

All steps live in [`src/preprocess.py`](src/preprocess.py) and are shared by training and prediction.

| Step | What it does |
|---|---|
| `clean` | Renames columns, parses dates, encodes the target, and drops impossible rows (6 rows: 1 with negative age, 5 booked after the appointment) |
| `add_history_features` | `prior_appointments`, `prior_no_shows`, `prior_no_show_rate`: the patient's appointments dated **strictly before the booking day**, so only outcomes known at booking time are used |
| `add_features` | `lead_days`, `same_day`, `appointment_weekday`, `male`, `disability` (from the 0–4 `Handcap` count) |
| `build_preprocessor` | Standardises numeric features; one-hot encodes neighbourhood and weekday, grouping rare (< 50 training rows) and unseen neighbourhoods together |
| `time_split` | Chronological split by appointment date (below) |

### Data split

| Split | Appointment dates | Rows | Share | No-show rate |
|---|---|---|---|---|
| Train | 2016-04-29 → 2016-05-25 | 75,278 | 68.1% | 21.0% |
| Validation | 2016-05-30 → 2016-06-02 | 17,567 | 15.9% | 18.6% |
| Test | 2016-06-03 → 2016-06-08 | 17,676 | 16.0% | 18.5% |

The split is by time, not random, because the model will always predict *future* appointments from *past* data. The train/validation boundary falls in a 4-day gap with no appointments. The later periods have a slightly lower no-show rate, which the evaluation must take into account.

### Design decisions

- **Prediction moment: at booking time.** Every feature is known when the appointment is booked.
- **`SMS_received` is excluded by default.** Reminders are sent *after* booking, and deciding who gets one is exactly what the model is for, so SMS status is not known at prediction time. The EDA also shows it is confounded with lead time. It can be switched on (`include_sms=True`) for a comparison experiment.
- **Leakage is tested automatically.** [`tests/test_preprocess.py`](tests/test_preprocess.py) checks the history features against a brute-force recomputation on real data, and checks that changing future outcomes never changes a feature.

## Approach

- **Baselines:** majority class, a lead-time rule (no ML), and logistic regression.
- **Models to compare:** Random Forest and gradient boosting (scikit-learn `HistGradientBoostingClassifier`).
- **Metrics** ([`src/evaluate.py`](src/evaluate.py)): ROC-AUC, PR-AUC, precision and recall among the 20% riskiest appointments (matching limited staff capacity), and the Brier score. Accuracy is not used because of the 80/20 imbalance.
- **Experiment tracking:** every run is logged with its git commit to [`experiments/runs.csv`](experiments/runs.csv); results and analysis are in [`experiments/README.md`](experiments/README.md).

### Baseline results (validation set)

| Model | ROC-AUC | PR-AUC | recall@top20 |
|---|---|---|---|
| Majority class | 0.500 | 0.186 | 0.206 |
| Lead-time rule (no ML) | 0.693 | 0.284 | 0.328 |
| Logistic regression | **0.726** | **0.332** | **0.365** |

### Model comparison (validation set)

| Model (best settings) | ROC-AUC | PR-AUC | recall@top20 |
|---|---|---|---|
| Logistic regression | 0.726 | 0.332 | 0.365 |
| Random forest (min 20 samples per leaf) | 0.726 | 0.323 | 0.363 |
| Gradient boosting (learning rate 0.03, 15 leaves, min 200 samples per leaf) | 0.729 | 0.331 | 0.368 |

23 runs in total. A paired bootstrap shows **no model is reliably better than logistic regression** (all 95% intervals for the difference include zero). See [`experiments/README.md`](experiments/README.md) for every run and the analysis.

## Repository structure

```
HealthProject/
├── Data/            dataset (CSV) and data dictionary
├── docs/            project brief, roadmap, and final-model decision
├── examples/        example inputs for prediction
├── notebooks/       EDA and Colab demo notebooks
├── src/             preprocessing, training, and prediction code
├── models/          saved model and preprocessing artifacts
├── experiments/     experiment log (runs.csv) and analysis
├── reports/         evaluation results, figures, and error analysis
├── tests/           automated tests (pytest)
├── requirements.txt pinned dependencies
└── PROJECT_STATUS.md
```

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/monfurkat-glitch/HealthProject.git
cd HealthProject
pip install -r requirements.txt
```

## Running the pipeline and tests

```bash
python -m src.preprocess     # builds the dataset and prints the split summary
python -m src.train baselines  # trains the baselines, scores them on validation, logs the runs
python -m src.train logreg_tuning       # also: random_forest, gradient_boosting
python -m src.train compare    # paired bootstrap: best of each model family vs logistic regression
python -m src.final           # trains the final model, saves it, and scores it once on the test set
python -m src.error_analysis  # error analysis and fairness checks of the final model
python -m src.predict examples/appointment.json  # scores an appointment with the saved model
python -m pytest             # runs the automated tests
```

## Prediction (inference)

[`src/predict.py`](src/predict.py) loads the saved model and scores new appointments. Every input is validated first.

```bash
python -m src.predict examples/appointment.json              # one appointment
python -m src.predict examples/appointments.json             # a list of appointments
python -m src.predict examples/appointments.json --history   # also look up each patient's past visits
python -m src.predict examples/invalid_appointment.json      # shows the validation errors
```

From Python:

```python
from src.predict import NoShowPredictor
predictor = NoShowPredictor()
predictor.predict({"age": 25, "gender": "F", "neighbourhood": "CENTRO",
                   "scheduled_day": "2016-06-01", "appointment_day": "2016-06-20"})
```

### Example input and output

Input ([`examples/appointment.json`](examples/appointment.json)):

```json
{
  "patient_id": "new-patient-001",
  "age": 19,
  "gender": "F",
  "neighbourhood": "JARDIM CAMBURI",
  "scholarship": 1,
  "hypertension": 0,
  "diabetes": 0,
  "alcoholism": 0,
  "handicap": 0,
  "scheduled_day": "2016-06-01T09:30:00",
  "appointment_day": "2016-06-29"
}
```

Output:

```json
{
  "no_show_probability": 0.3509,
  "risk_band": "High",
  "main_factors": [
    "not booked for the same day raises risk",
    "welfare (Scholarship) raises risk",
    "age 19 raises risk"
  ],
  "lead_days": 28,
  "prior_appointments": 0,
  "prior_no_shows": 0,
  "warnings": []
}
```

- `no_show_probability`: the model's estimated chance of a no-show; treat it as relative risk (see Results).
- `risk_band`: **Low** < 0.233 ≤ **Medium** < 0.323 ≤ **High**. The cut-offs are the 50th and 80th percentiles of the training predictions.
- `main_factors`: the inputs that move this appointment's risk most, compared with an average appointment.

### Input rules

| Field | Required | Accepted values |
|---|---|---|
| `age` | yes | whole number 0–115 (over 100: warning) |
| `gender` | yes | `F`/`M` (or `female`/`male`) |
| `scheduled_day`, `appointment_day` | yes | dates such as `2016-06-01`; the appointment cannot be before the booking. More than 179 days ahead or on a Sunday: warning |
| `neighbourhood` | no | name from the dataset; unknown or missing → treated as a rare neighbourhood, with a warning |
| `scholarship`, `hypertension`, `diabetes`, `alcoholism` | no | `0`/`1`, `yes`/`no`, `true`/`false`; missing → 0, with a warning |
| `handicap` | no | 0–4; missing → 0, with a warning |
| `prior_appointments`, `prior_no_shows` | no (both or neither) | whole numbers ≥ 0; otherwise looked up with `--history`, or 0 for a new patient |
| `patient_id` | no | used with `--history` |

Invalid inputs are rejected with **every** problem listed, e.g. for [`examples/invalid_appointment.json`](examples/invalid_appointment.json):

```
Cannot score this input:
 - 'age' must be between 0 and 115, got -4
 - 'gender' must be F or M, got 'X'
 - 'appointment_day' (2016-06-01) is before 'scheduled_day' (2016-06-10): an appointment cannot be booked after it happens
 - 'diabetes' must be 0 or 1 (or yes/no), got 'maybe'
```

55 automated tests ([`tests/`](tests/)) cover the saved model reloading, 15 kinds of invalid input, 6 kinds of unusual input, and a check that `predict()` gives exactly the same probabilities as the training pipeline for real appointments.

## Final model

**Logistic regression** (`C = 1.0`, 14 features, SMS excluded), refitted on train + validation and saved to [`models/final_model.joblib`](models/final_model.joblib) together with its risk-band cut-offs.

It was chosen **on validation evidence only** and the choice was committed before the test set was used ([docs/final_model.md](docs/final_model.md)): it matched the best tree models within noise, it is simpler, and its coefficients can be explained to staff.

## Results (test set, used once)

Appointments 2016-06-03 to 2016-06-08 (17,676 rows, 18.5% no-shows). Full report: [reports/test_results.md](reports/test_results.md).

| Model | ROC-AUC | PR-AUC | recall@top20 |
|---|---|---|---|
| Majority class | 0.500 | 0.185 | 0.182 |
| Lead-time rule (no ML) | 0.692 | 0.278 | 0.305 |
| **Logistic regression (final)** | **0.723** (95% CI 0.715–0.732) | **0.328** | **0.362** |
| Gradient boosting (comparison) | 0.737 | 0.342 | 0.385 |

| Risk band | Share of appointments | No-show rate | Share of all no-shows |
|---|---|---|---|
| Low | 49.7% | 8.1% | 21.7% |
| Medium | 27.8% | 25.1% | 37.8% |
| High | 22.6% | 33.2% | 40.5% |

- The final model's test score matches its validation score (0.726), so it generalises as expected, and it clearly beats both baselines.
- Following up only the **High** band (23% of appointments) reaches **41% of all no-shows**.
- **Gradient boosting was reliably better on the test set** (+0.013 ROC-AUC, 95% CI +0.009 to +0.018), unlike on validation. The pre-registered choice is kept so the test score stays honest; gradient boosting is recommended for a future version.
- Probabilities are slightly too high (21.2% predicted vs 18.5% observed), probably because the test period had fewer no-shows than the training period. Treat them as relative risk.

## Error analysis and fairness

Full report: [reports/error_analysis.md](reports/error_analysis.md) (`python -m src.error_analysis`).

- **Main limitation:** within a single lead-time group, the model ranks only a little better than chance (ROC-AUC 0.57–0.62). Most of its power comes from separating short from long lead times.
- **Errors:** caught no-shows and false alarms have almost identical profiles (young, booked weeks ahead), so the available features cannot separate them. Missed no-shows look like typical attenders (older, short lead time).
- **New patients** without history are ranked as well as everyone else (ROC-AUC 0.717).
- **Fairness:** no material gender gap. **Welfare patients are flagged twice as often** (42% vs 20%) with twice the false-alarm rate, because their no-show rate dropped in the test week. **Children are over-flagged and elderly no-shows are rarely caught** (7% recall for 66+), partly because the model treats age as a straight line.

## Limitations and Responsible AI

_To be completed (Day 10); current points:_
- The data comes from one Brazilian city's public health system in 2016 and may not generalise to other clinics or populations.
- Patient IDs are de-identified, but combinations of age, neighbourhood, and health conditions could still act as quasi-identifiers.
- A high risk score must never be used to deny, delay, or deprioritise care. It is meant only for **supportive** follow-up (reminders, confirmation calls).
- Welfare patients and children receive more false alarms, and elderly no-shows are rarely flagged (see the error analysis). Error rates by group should be monitored if the tool is used.
- Probabilities are slightly too high on recent data; treat them as relative risk.

## License and acknowledgements

- **Dataset:** *Medical Appointment No Shows* by Joni Hoppen and Aquarela Analytics, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Redistributed unmodified for non-commercial educational use.
- **AI assistance:** Parts of this project's code and documentation were drafted with the help of Claude (Anthropic), an AI coding assistant, as allowed by the course rules. All design decisions were reviewed, and the student is responsible for and can explain every component.
