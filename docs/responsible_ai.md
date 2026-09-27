# Model card and Responsible AI

This document describes what the no-show model is for, what it must not be used for, who it may treat unfairly, and how patient data must be protected. Evidence comes from [reports/test_results.md](../reports/test_results.md) and [reports/error_analysis.md](../reports/error_analysis.md).

## 1. Model details

| | |
|---|---|
| Model | Logistic regression (`C = 1.0`), scikit-learn 1.7.2 |
| File | [`models/final_model.joblib`](../models/final_model.joblib) |
| Trained on | 92,845 appointments, 2016-04-29 to 2016-06-02 |
| Evaluated on | 17,676 later appointments, 2016-06-03 to 2016-06-08 (used once) |
| Inputs | age, gender, neighbourhood, welfare status, hypertension, diabetes, alcoholism, disability, booking and appointment dates, and the patient's past attendance known at booking time |
| Output | no-show probability, Low / Medium / High band, top factors, warnings |
| Test performance | ROC-AUC 0.723 (95% CI 0.715–0.732); the High band (23% of appointments) contains 41% of no-shows |
| Author | Furkat Toshmatov (student capstone project, not a certified medical device) |

## 2. Intended use

- **Purpose:** help outpatient clinic **operations staff** decide which upcoming appointments should get **supportive** follow-up: an extra reminder, a confirmation call, or an offer to reschedule.
- **How:** staff see a list ranked by risk and choose whom to contact. **A person always makes the decision**; the model never acts on its own.
- **Setting:** a prototype for the public clinics represented in the data (Vitória, Brazil, 2016). Any other setting requires re-validation first (section 7).

## 3. Uses that are out of scope or prohibited

| Use | Why not |
|---|---|
| Cancelling, delaying, or refusing an appointment because of a high score | The score is wrong about 2 times in 3 in the High band (precision 33%). Most flagged patients would have come. Restricting access to care is harmful and against the project's requirements |
| Overbooking the slots of high-risk patients | When a flagged patient does come, they face a double-booked, longer wait. This would fall hardest on the groups that are flagged most (welfare patients, children). Not evaluated, not supported |
| Penalising patients (fees, lower priority, removal from lists) | Punishes people for a prediction, and the risk factors include poverty and illness |
| Clinical decisions of any kind | The model predicts attendance, not health |
| Insurance, credit, employment, or eligibility decisions | Completely different purpose; would misuse health data |
| Treating the probability as an exact chance | Probabilities are slightly too high on recent data (21.2% predicted vs 18.5% observed); use them as a **ranking** |
| Use in another city, country, or year without re-validation | Trained on one Brazilian city's public system over 6 weeks in 2016 |

## 4. Sensitive attributes and fairness

The model uses several attributes that are **sensitive**: gender, age, welfare status (a proxy for poverty), neighbourhood (which can proxy for income or ethnicity), and health conditions (alcoholism, disability, hypertension, diabetes).

**What the model does with them:** welfare status (×1.27 odds), alcoholism (×1.43), and disability (×1.13) *raise* the predicted risk, and younger age raises it. So people who are poorer, younger, or living with alcoholism or a disability are flagged more often. For a **supportive** intervention this can be appropriate, since the people most likely to miss care get more help. It would be harmful for any **restrictive** use (section 3), which is why those uses are excluded.

**Measured on the test set** (details in [error_analysis.md](../reports/error_analysis.md#4-fairness)):

| Group | Finding |
|---|---|
| Gender | No material disparity: similar no-show rates, recall 40% vs 42%, false alarms 17% vs 21% |
| Welfare (Scholarship) | Flagged **twice as often** (42% vs 20%) with **twice the false-alarm rate** (34% vs 17%), while its real no-show rate was only slightly higher (20.5% vs 18.3%). The model was accurate for this group on training data; the group's no-show rate fell in the test week |
| Children (0–12) | Over-flagged (43%; mean predicted 24.5% vs 18.4% observed), partly because the model treats age as a straight line |
| Older patients (66+) | Rarely flagged (4.5%); only **7%** of their no-shows are caught |

**Not done in this project** (future work): training a version *without* the sensitive attributes to measure how much performance they add; formal fairness constraints or group-specific thresholds; checks by ethnicity or income (not in the data).

## 5. Privacy and data protection

- **The data:** the public Kaggle dataset is de-identified. Patient IDs are pseudonymous numbers, and there are no names, addresses, or dates of birth. It is used under its CC BY-NC-SA 4.0 license for non-commercial education.
- **Re-identification risk:** the CSV in this repository still contains patient-level records. A combination of age, neighbourhood, conditions, and appointment dates could, in principle, single out a person (quasi-identifiers). The data must not be combined with other sources to try to identify anyone.
- **The model file contains no patient records:** only coefficients, the scaler's averages, band cut-offs, and average feature values.
- **Predictions** return only the probability, band, factors, and warnings. Nothing is stored or sent anywhere.
- **In a real deployment:** health data is *sensitive personal data* under Brazil's LGPD (Lei Geral de Proteção de Dados), and under similar laws elsewhere (e.g. GDPR). A clinic would need a legal basis and patient information, access control limited to scheduling staff, the minimum necessary data (the history lookup only needs past appointment dates and outcomes), audit logs, and no transfer of patient data to external services.

## 6. Safety and misuse risks

| Risk | Mitigation in this project |
|---|---|
| **Automation bias:** staff trust the score too much | Output is labelled decision support; each prediction lists its main factors; documentation states the precision (1 in 3 flagged patients is a real no-show) |
| Unusual inputs give unreliable scores (e.g. booked 244 days ahead: the linear model extrapolates) | Validation rejects impossible inputs and **warns** on inputs outside the training range (age > 100, lead time > 179 days, unknown neighbourhood, Sunday) |
| Scores drift as behaviour changes (base rate fell from 20.5% to 18.5%) | Documented; monitoring and recalibration recommended (section 7) |
| Stigmatising language towards patients | Bands describe *appointments* ("High risk of no-show"), not people; the tool is internal to staff |

## 7. Limitations

1. **Moderate accuracy:** ROC-AUC 0.72. Within appointments booked equally far ahead, the model ranks only a little better than chance (0.57–0.62). Most of its power comes from lead time.
2. **The data cannot separate the hard cases:** caught no-shows and false alarms have almost identical profiles, so no model built on these features can separate them well.
3. **Short, old, local data:** 6 weeks of appointments in 2016 from one city's public system. There are no seasonal effects, no appointment type or time of day, no distance to the clinic, and no weather.
4. **Linear age effect:** the model over-predicts for children and under-predicts for teenagers. Gradient boosting handles this and was better on the test set (+0.013 ROC-AUC); it is the recommended candidate for a future version.
5. **Calibration drift:** probabilities were about 2.7 points too high on the test week.
6. **Gender is recorded as F/M only** in the source data.
7. **SMS reminders are excluded** (not known at booking time), so the model cannot say whether a reminder would change the outcome. It predicts risk; it does not measure the effect of an intervention.

## 8. Recommendations before any real use

1. Re-validate on the clinic's own recent data (performance, calibration, and group error rates).
2. Recalibrate the probabilities on recent data, and re-check the band cut-offs against staff capacity.
3. Monitor recall and false-alarm rates by welfare status, age group, and gender every month; investigate if gaps widen.
4. Consider gradient boosting or non-linear age terms, confirmed on new data.
5. Keep the tool strictly supportive: write the allowed uses (section 2) into the clinic's procedures, and review them with staff.
