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

## Approach

- **Baselines:** majority class ("always shows up") and logistic regression.
- **Models to compare:** Random Forest and gradient boosting (scikit-learn `HistGradientBoostingClassifier`).
- **Validation:** time-based split (train on earlier appointments, validate and test on later ones).
- **Metrics:** ROC-AUC, PR-AUC, and precision/recall on the no-show class (the classes are imbalanced, about 80/20).
- **Leakage guard:** patient-history features only use past appointments dated before the current booking's `ScheduledDay`.

## Repository structure

```
HealthProject/
├── Data/            dataset (CSV) and data dictionary
├── docs/            project brief and roadmap
├── notebooks/       EDA and Colab demo notebooks
├── src/             preprocessing, training, and prediction code
├── models/          saved model and preprocessing artifacts
├── experiments/     experiment log
├── reports/         evaluation results, figures, and error analysis
├── tests/           automated tests
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

## Training, demo, and inference

_Coming in later milestones (see [ROADMAP](docs/ROADMAP.md))._

## Results

_No results yet. Every number reported here will come from runs recorded in this repository._

## Limitations and Responsible AI

_To be completed; initial concerns:_
- The data comes from one Brazilian city's public health system in 2016 and may not generalise to other clinics or populations.
- Patient IDs are de-identified, but combinations of age, neighbourhood, and health conditions could still act as quasi-identifiers.
- A high risk score must never be used to deny, delay, or deprioritise care.

## License and acknowledgements

- **Dataset:** *Medical Appointment No Shows* by Joni Hoppen and Aquarela Analytics, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Redistributed unmodified for non-commercial educational use.
- **AI assistance:** Parts of this project's code and documentation were drafted with the help of Claude (Anthropic), an AI coding assistant, as allowed by the course rules. All design decisions were reviewed, and the student is responsible for and can explain every component.
