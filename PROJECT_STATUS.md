# PROJECT_STATUS

> Update this after a verified milestone. This file does not replace `README.md`.

## Current verified state
- Repository: https://github.com/monfurkat-glitch/HealthProject
- Current branch: `main`
- Current verified commit / version: see latest commit on `main` (Day 11 complete)
- Lesson 3 result: _TBD (fill in)_
- CHECK status: PASS (`python -m pytest`: 55 tests)

## Limits and unfinished work
- Current limitations:
  - Final model: logistic regression, test ROC-AUC 0.723 (95% CI 0.715–0.732); High band captures 41% of no-shows. Gradient boosting was reliably better on test (+0.013) but the pre-registered choice was kept (`reports/test_results.md`).
  - Probabilities slightly too high on test (base-rate shift).
  - Error analysis: weak ranking within a lead-time group; welfare patients and children over-flagged, elderly no-shows rarely caught (`reports/error_analysis.md`).
  - Prediction module (`src/predict.py`) and demo notebook (`notebooks/02_demo.ipynb`) done. Whole project reproduced from a fresh clone in a clean environment with identical results.
  - README finalised; Responsible AI model card in `docs/responsible_ai.md`.
  - Defense slides in `presentation/` (16 slides, speaker notes). LMS submission file prepared outside the repo.
  - `02_demo.ipynb` confirmed running cleanly in Google Colab (Run all) from GitHub, 2026-09-27.
  - Appointment dates only cover about 6 weeks (2016-04-29 to 2016-06-08), which limits the time-based split.
- Unfinished / blocked:
  - Mentor approval of the brief is pending (sections 12–13 now filled in `docs/project_brief.md`; copy them into the .docx before submitting).

## Personal project
- Brief: `docs/project_brief.md`
- Brief status: PENDING
- Required revision, if any: None received yet.

## Next step
- Next concrete action: rehearse the presentation and live demo; get brief approval; push; upload the submission file to the LMS before the deadline.
- Last updated (date): 2026-09-27
