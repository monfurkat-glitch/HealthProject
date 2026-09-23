# Roadmap: Capstone Criteria Checklist

Tracks every item in the *Capstone Project Evaluation Criteria (v1.1)* against this project.
Tick items only when they are **done and verified** in the repository.

Legend: ✅ done · 🟡 partial · ⬜ not started

## Score tracker

| # | Criterion | Max | Status |
|---|---|---|---|
| 1 | Problem Definition and Project Alignment | 10 | 🟡 brief written, not yet approved |
| 2 | Data and Preprocessing Pipeline | 15 | 🟡 data profiled, no pipeline yet |
| 3 | Modeling and Experiments | 20 | ⬜ |
| 4 | Evaluation and Error Analysis | 15 | ⬜ |
| 5 | End-to-End Implementation and Delivery | 20 | ⬜ |
| 6 | Documentation and Reproducibility | 10 | 🟡 README started |
| 7 | Responsible AI and Limitations | 5 | ⬜ |
| 8 | Presentation, Demo, and Q&A | 5 | ⬜ |

## Detailed checklist

### 1. Problem Definition and Project Alignment (10)
- [ ] (3) Problem statement and practical relevance: README explains who suffers from no-shows and why it matters
- [ ] (3) ML task framing: task type, input, output, and target are stated explicitly
- [ ] (2) Measurable success criteria: e.g. "beat logistic regression ROC-AUC", "recall ≥ X at a staff-capacity threshold"
- [ ] (2) Alignment with approved brief: brief approved, and the implementation matches it (any deviations explained)

### 2. Data and Preprocessing Pipeline (15)
- [ ] (2) Dataset source, size, and structure documented, with column dictionary
- [ ] (3) EDA notebook: distributions, target rate, lead-time effect, SMS confounding, data issues
- [ ] (4) Cleaning: Age = -1, negative lead time, Handcap to yes/no, neighbourhood encoding, scaling where needed
- [ ] (2) Feature engineering: lead days, weekday, age groups, leakage-safe patient history
- [ ] (2) Time-based train / validation / test split, justified
- [ ] (2) Leakage prevention: history features use only outcomes known at `ScheduledDay`; encoders fitted on train only; a test proves it

### 3. Modeling and Experiments (20)
- [ ] (3) Baselines: majority class and logistic regression
- [ ] (5) Main model trained (Random Forest / HistGradientBoosting)
- [ ] (4) At least two approaches compared on the same validation set
- [ ] (3) Several experiments / hyperparameter changes, analysed
- [ ] (2) Experiment tracking: every run logged (MLflow or `experiments/runs.csv`)
- [ ] (3) Final model choice justified with evidence

### 4. Evaluation and Error Analysis (15)
- [ ] (3) Metrics explained: ROC-AUC, PR-AUC, precision/recall/F1 on no-show class
- [ ] (3) Final results reported on the untouched test period
- [ ] (2) Final model vs baseline comparison table
- [ ] (3) Error analysis: false negatives / false positives by segment (lead time, age, neighbourhood, SMS)
- [ ] (2) Threshold selection, calibration check, edge cases
- [ ] (2) Honest conclusion, including what the model cannot do

### 5. End-to-End Implementation and Delivery (20)
- [ ] (5) `predict` pipeline: raw appointment → probability + Low/Medium/High band
- [ ] (4) Google Colab demo notebook that runs from the GitHub repo
- [ ] (3) Input validation: missing/invalid age, unknown neighbourhood, bad dates → clear errors or safe defaults
- [ ] (2) Model and preprocessor saved (`joblib`) and reloaded in the demo
- [ ] (4) Clean-runtime check: fresh Colab runs the demo with no hidden local files
- [ ] (2) Edge-case tests pass

### 6. Documentation and Reproducibility (10)
- [ ] (3) README has every required section (see list below)
- [ ] (3) `requirements.txt`, setup, training, and demo instructions complete
- [ ] (2) Logical repo layout (`src/`, `notebooks/`, `models/`, `reports/`, `tests/`)
- [ ] (2) Reproduced end to end from a fresh clone

### 7. Responsible AI and Limitations (5)
- [ ] (2) Bias / fairness: performance by gender, age group, neighbourhood, and scholarship (welfare) status
- [ ] (1) Privacy: quasi-identifiers, de-identification, misuse risk (e.g. denying care to "high-risk" patients)
- [ ] (2) Limitations and inappropriate uses stated

### 8. Presentation, Demo, and Q&A (5)
- [ ] Slides for the final defense
- [ ] Live demo rehearsed in Colab
- [ ] Q&A prep: be able to explain every design choice and every file

## README required sections
- [ ] Project title
- [ ] Problem statement
- [ ] Selected project track
- [ ] Dataset source
- [ ] ML task type
- [ ] Pipeline / system architecture
- [ ] Models or approaches tested
- [ ] Final model and justification
- [ ] Evaluation metrics and results
- [ ] Installation instructions
- [ ] Training instructions
- [ ] Demo and inference run instructions (Colab first)
- [ ] Example input and output
- [ ] Known limitations
- [ ] Responsible AI considerations
- [ ] Student's full name
- [ ] License information and source acknowledgements (including AI-assistant use)

## Daily plan

| Day | Focus | Criteria |
|---|---|---|
| 1 | Finish brief (sections 12–13, summary); repo structure; `requirements.txt`; data-access decision | 1, 6 |
| 2 | EDA notebook with charts and written findings | 2 |
| 3 | `src/preprocess.py`: cleaning, features, leakage-safe history, time split; leakage test | 2 |
| 4 | Baselines + experiment logging | 3, 4 |
| 5 | Random Forest / gradient boosting experiments and tuning | 3 |
| 6 | Final model choice; test-set evaluation; threshold and calibration | 3, 4 |
| 7 | Error analysis and fairness slices | 4, 7 |
| 8 | `src/predict.py` with input validation; saved artifacts; tests | 5 |
| 9 | Colab demo; clean-runtime check from a fresh clone | 5, 6 |
| 10 | Complete README, Responsible AI section, limitations | 6, 7 |
| 11 | Slides, demo rehearsal, Q&A prep, LMS submission file with repo link | 8 |

## Rules to keep in mind
- **Never fabricate results.** Every number in the README must come from a run in this repo.
- **Understand every file.** You must be able to explain and defend all code during Q&A.
- **Acknowledge AI assistance** in the README (course rule: AI coding assistants are allowed, but must be acknowledged).
- **Commit little and often.** Mentors can check the commit history, and nothing may be added after the deadline.
