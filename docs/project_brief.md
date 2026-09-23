# Project Brief: MED-01 | Hospital Appointment No-Show Prediction

*Individual Capstone Project. Source: `Medication scenario.docx`.*

> **Purpose:** Complete this brief and submit it to the mentors for approval before full implementation. Define the business problem and propose a technically realistic ML solution without writing a step-by-step implementation recipe.

## 1. Project Identification

| Decision / Question | Student Response |
|---|---|
| Student full name | Furkat Toshmatov |
| Project title | MED-01 \| Hospital Appointment No-Show Prediction |
| Industry / field | MedTech |
| Intended client / user / stakeholder | Private Healthcare Network |
| One-sentence project summary | A machine learning model that predicts each outpatient appointment's no-show risk before it happens, so clinic staff can focus reminders and follow-up calls on the appointments most likely to be missed. |

## 2. Client / User Background

A private healthcare network managing outpatient clinics faces reduced medical capacity and increased patient wait times caused by unannounced appointment no-shows.

## 3. Business Problem

The lack of a predictive risk assessment forces clinic staff to distribute reminder efforts broadly rather than targeting appointments at high risk of non-attendance.

## 4. Requested Solution

The objective is to build a machine learning model that estimates pre-appointment no-show probabilities to enable prioritized reminders, confirmations, and manual follow-ups by operations staff.

## 5. Available / Expected Information

The project will leverage pre-appointment data (including scheduling timestamps, patient demographics, attendance history, and appointment context) from an appropriate public dataset while evaluating any data limitations.

## 6. Data & Problem Discovery

| Decision / Question | Student Response |
|---|---|
| Selected dataset and source | Medical Appointment No Shows: <https://www.kaggle.com/datasets/joniarroba/noshowappointments> |
| What does one record / sample represent? | One scheduled outpatient appointment for one patient |
| Proposed target or ML objective | Binary classification predicting whether the appointment will be a no-show, using the `No-show` column as the label |
| Key information available at prediction / inference time | Patient demographics (age, gender, neighbourhood, scholarship status, health conditions), scheduling dates and days of advance notice, and SMS reminder status |
| Main data quality issues | Invalid values (e.g. Age = -1), a few ScheduledDay timestamps after AppointmentDay, a miscoded Handcap field, high-cardinality neighbourhoods, and class imbalance (~80/20) |
| Potential leakage risks | Building a patient's "prior no-show rate" feature without strict time-ordering would leak future outcomes into past predictions |
| Privacy / fairness / licensing concerns | Patient IDs are de-identified but quasi-identifiers remain; the data reflects one city's public health population |

## 7. Technical Proposal

| Decision / Question | Student Response |
|---|---|
| ML problem formulation | Supervised binary classification, predicting no-show (yes/no) for each scheduled appointment |
| Proposed baseline | A simple majority-class or rule-based baseline (e.g. always predict "will show up"), plus logistic regression as the first real model to beat it |
| Main modeling approach(es) to investigate | Logistic regression as the interpretable baseline model, then tree-based models like Random Forest or Gradient Boosting since they handle mixed categorical/numeric health data well |
| Data splitting / validation strategy | Split by time (train on earlier appointments, validate/test on later ones) rather than random shuffling, since in real use, you're always predicting future appointments from past data |
| Primary evaluation metric(s) and why | ROC-AUC and recall/precision on the no-show class, because the classes are imbalanced (~80/20) and accuracy alone would look good even if the model just predicted "shows up" every time |
| Expected inference input | The details of one new scheduled appointment (patient demographics, health flags, scheduling/appointment dates, reminder status) before it happens |
| Expected inference output | A probability or risk score (e.g. 0–100%) that the appointment will be a no-show, possibly grouped into risk levels like Low/Medium/High |
| Main technical risks / assumptions | Assumes the historical Brazilian clinic data generalizes to the client's own patients, and that features like SMS timing and patient history stay truly "before the appointment" with no accidental leakage |

## 8. Functional Requirements

1. Given the details of a new scheduled appointment, the system must output a no-show risk score or probability before the appointment happens.
2. The system must only use information that is known at scheduling time, never anything from after the appointment occurs.
3. Clinic staff must be able to see which appointments are high-risk, so they can prioritize reminders or follow-up calls.
4. The system must handle missing or invalid inputs (e.g. a blank age or unknown neighborhood) without crashing or giving a nonsense prediction.
5. The system must be testable on appointments it has never seen before, and its accuracy on that unseen data must be reported.
6. The system must never make an automatic decision like cancelling or rebooking an appointment. It only informs staff, who decide.
7. Anyone should be able to run the whole pipeline (data prep → training → prediction) from the repository using clear, written instructions.

## 9. Expected Deliverables

- A working ML solution that addresses the approved problem.
- A trained model or ML pipeline with documented methodology.
- Documented data source and assumptions.
- Evaluation on appropriate unseen/held-out data.
- A usable inference interface or service appropriate to the project.
- A reproducible repository with clear run instructions.
- Documented limitations, risks and recommended next steps.

*Note: General capstone requirements are defined in the official Capstone Evaluation Criteria and Helper documents.*

## 10. Acceptance Criteria

- A dataset has been chosen, cleaned, and clearly documented, including its known limitations.
- A working model has been trained that predicts no-show risk using only information available before the appointment.
- The model has been tested on data it did not train on, and the real results (not guesses) are recorded.
- Invalid or missing inputs are handled gracefully instead of breaking the system.
- Someone else could clone the repository, follow the written instructions, and get the same results without asking you anything.

## 11. Constraints: In Scope / Out of Scope

| In scope | Out of scope |
|---|---|
| Building a prototype ML model using the public Kaggle no-show dataset | Connecting to a real hospital's live scheduling system |
| Cleaning and preparing that data (fixing bad values, removing leakage-prone features) | Building enterprise-level infrastructure (load balancing, multi-server deployment, etc.) |
| Evaluating the model honestly on data it hasn't seen | Letting the system automatically cancel, reschedule, or make clinical decisions |

## 12. Questions You Must Resolve

1. **When is the prediction made, and is `SMS_received` known at that moment?** SMS reminders are sent after booking (only for appointments booked 3 or more days ahead). If the model scores appointments at booking time, SMS status is not yet known and must be excluded; if it scores them the day before the appointment, SMS status can be used. The prediction moment must be fixed first, because it decides which features are legal.
2. **How can patient history be used without leakage, and what happens for new patients?** A patient's past no-shows are only known once those appointments have happened, so history features may only count appointments dated before the current booking's `ScheduledDay`. Patients with no history need a sensible default (e.g. "0 prior appointments") rather than a missing value.
3. **How should the data be split in time?** Appointment dates only cover about 6 weeks (2016-04-29 to 2016-06-08). The split must keep training, validation, and test periods in chronological order and still leave enough no-shows in each period to measure performance reliably.
4. **What operating threshold matches staff capacity?** Staff can only call a limited number of patients per day. The threshold that turns probabilities into Low/Medium/High bands should be chosen from the precision/recall trade-off at a realistic follow-up capacity (e.g. the top 10–20% riskiest appointments), not left at the default 0.5.
5. **Does the model perform equally well across patient groups?** Performance must be compared by gender, age group, neighbourhood, and welfare (`Scholarship`) status, so that the tool does not systematically miss or over-flag particular groups.

## 13. Optional Directions

- A simple staff-facing web app (e.g. Gradio or Streamlit) that shows a day's appointments ranked by risk.
- Per-prediction explanations (e.g. SHAP or feature contributions), so staff can see *why* an appointment was flagged.
- A cost-based threshold that weighs the cost of an empty slot against the cost of a reminder call, plus probability calibration.

## 14. Mentor Review & Approval

| Field | Value |
|---|---|
| Decision | ☐ Approved ☐ Approved with revisions ☐ Revision required |
| Required revisions / comments | |
| Approved scope / special conditions | |
| Mentor name | |
| Date | |
