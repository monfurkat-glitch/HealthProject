# Final model decision (pre-registered)

This decision was made **using validation results only** and committed to git **before** any model was scored on the test set. The commit date is the proof. The test result is reported whatever it turns out to be, and the choice below does not change based on it.

## Final model

**Logistic regression** (`C = 1.0`, no class weights), using the default feature set from `src/preprocess.py` (14 features, SMS status excluded).

### Why (validation evidence, see [experiments/README.md](../experiments/README.md))

1. **Equal performance:** ROC-AUC 0.726, the same as the best random forest (0.726) and within noise of the best gradient boosting (0.729). A paired bootstrap found no model reliably better than logistic regression (every 95% interval for the difference includes zero).
2. **Simpler and explainable:** each feature has one coefficient, so staff and reviewers can see why an appointment is flagged.
3. **No overfitting:** train and validation ROC-AUC are almost equal (0.728 vs 0.726).
4. **Good probabilities:** Brier score 0.138, as good as any other model tried.

## Comparison model

Gradient boosting (`HistGradientBoostingClassifier`, learning rate 0.03, 15 leaves, min 200 samples per leaf) is also scored on the test set, **for reporting only**.

## Procedure

1. **Training data:** the final model is refitted on **train + validation** (appointments 2016-04-29 to 2016-06-02), so it learns from the most recent data. Its settings are those chosen on validation. The comparison model and the baselines (majority class, lead-time rule) are trained on the same data.
2. **Risk bands:** from the refitted model's predicted probabilities on the train + validation rows:
   - **High:** at or above the 80th percentile (the riskiest 20%, matching the staff-capacity assumption);
   - **Medium:** between the 50th and 80th percentiles;
   - **Low:** below the 50th percentile.

   The two probability cut-offs are fixed at this point and saved with the model, so a single new appointment can be banded.
3. **Test set:** appointments 2016-06-03 to 2016-06-08 (17,676 rows), used **once**.
4. **Reported on test:** ROC-AUC, PR-AUC, precision and recall among the top 20%, Brier score, 95% bootstrap intervals for the final model, a paired bootstrap against gradient boosting, the no-show rate in each risk band, and a calibration check.

## Success criteria (from the README, unchanged)

Criteria 1 and 2 were written before any model was trained, and assumed that a more complex model would beat the logistic regression baseline. **On validation that did not happen:** no model beat logistic regression reliably. This is reported as a result, not hidden by rewording the criteria. On the test set we check:

1. *"The final model beats both baselines (always 'shows up' and logistic regression) on ROC-AUC and PR-AUC."* The final model **is** logistic regression, so the part about beating logistic regression cannot be met. We check whether it beats the majority-class and lead-time-rule baselines, and report the gradient boosting comparison.
2. *"Among the 20% of appointments ranked riskiest, the model catches a larger share of real no-shows than the logistic regression baseline."* Not met, for the same reason. We report recall@top20 for all models.
3. No leakage: covered by the automated tests.
4. The Colab demo runs end to end: Day 9.
