# PROJECT_STATUS

> Update this after a verified milestone. This file does not replace `README.md`.

## Current verified state
- Repository: https://github.com/monfurkat-glitch/HealthProject
- Current branch: `main`
- Current verified commit / version: see latest commit on `main` (Day 5 complete)
- Lesson 3 result: _TBD (fill in)_
- CHECK status: PASS (`python -m pytest`: 16 tests)

## Limits and unfinished work
- Current limitations:
  - 23 experiment runs: logistic regression, random forest, and gradient boosting all reach ~0.73 validation ROC-AUC; no model is reliably better than logistic regression (paired bootstrap). No test evaluation or demo yet.
  - Project track in README still to confirm.
  - Appointment dates only cover about 6 weeks (2016-04-29 to 2016-06-08), which limits the time-based split.
- Unfinished / blocked:
  - Mentor approval of the brief is pending (sections 12–13 now filled in `docs/project_brief.md`; copy them into the .docx before submitting).

## Personal project
- Brief: `docs/project_brief.md`
- Brief status: PENDING
- Required revision, if any: None received yet.

## Next step
- Criteria checklist and daily plan: `docs/ROADMAP.md`
- Next concrete action (Day 6): choose the final model on validation evidence, choose the risk threshold, then score it once on the test set.
- Last updated (date): 2026-09-25
