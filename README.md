# MED-01 | Hospital Appointment No-Show Prediction

Machine learning project that estimates the probability that a scheduled outpatient appointment will be a **no-show**, so that clinic staff can prioritise reminders, confirmations and follow-up calls for high-risk appointments.

The model only **informs** staff. It never cancels or rebooks appointments automatically.

## Dataset

[Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments) (Kaggle): about 110k appointments from public health clinics in Vitória, Brazil (Apr–Jun 2016).

The dataset is **not included** in this repository. To run the project:

1. Download it from the Kaggle link above.
2. Unzip it and place `KaggleV2-May-2016.csv` in the `Data/` folder.

## Approach

- **Problem:** supervised binary classification (`No-show` = Yes/No).
- **Baselines:** majority class ("always shows up") and logistic regression.
- **Models:** Random Forest and gradient boosting.
- **Validation:** time-based split (train on earlier appointments, test on later ones).
- **Metrics:** ROC-AUC, plus precision/recall on the no-show class (the classes are imbalanced, about 80/20).
- **Output:** a no-show probability and a Low / Medium / High risk band.

## Key data findings

- 110,527 appointments, 62,299 patients, no missing values.
- Invalid records: 1 row with `Age = -1`, and 5 appointments scheduled after the appointment date.
- `Handcap` is a 0–4 count, not a yes/no flag.
- Appointment dates only span about 6 weeks (2016-04-29 to 2016-06-08).
- Lead time (days between booking and appointment) is the strongest signal: same-day bookings have a 4.6% no-show rate, versus 28.5% for all other appointments.
- SMS reminders are confounded with lead time (they were only sent for bookings made 3 or more days ahead), so their raw association with no-shows is misleading.
- Leakage guard: patient-history features only use past appointments whose date is before the current booking's `ScheduledDay`.

## Author

Furkat Toshmatov
