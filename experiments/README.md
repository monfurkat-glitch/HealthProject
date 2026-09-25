# Experiments

Every training run is logged automatically to [`runs.csv`](runs.csv) by `src/experiments.py`. Each row records the time, the git commit of the code that produced it, the model and its settings, whether SMS status was used, and the metrics.

All runs below are scored on the **validation set** (appointments 2016-05-30 to 2016-06-02; 17,567 rows; 18.6% no-shows). The test set is untouched until the final model is chosen.

To reproduce a group of runs:

```bash
python -m src.train baselines
```

## Metrics

| Metric | Meaning | Random guess |
|---|---|---|
| ROC-AUC | How well no-shows are ranked above attended appointments | 0.50 |
| PR-AUC | Ranking quality focused on the no-show class | 0.186 (the no-show rate) |
| precision@top20 | Of the 20% riskiest appointments, the share that are real no-shows | 0.186 |
| recall@top20 | Share of all no-shows found in the 20% riskiest appointments | 0.20 |
| Brier | Mean squared error of the probabilities (lower is better) | — |

"Top 20%" reflects staff capacity: clinic staff can only follow up a fraction of appointments, so what matters is how many no-shows sit at the top of the list.

## 1. Baselines (commit `a8ec4ae`)

| Run | ROC-AUC | PR-AUC | precision@top20 | recall@top20 | Brier |
|---|---|---|---|---|---|
| `majority_class`: always predicts the training no-show rate | 0.500 | 0.186 | 0.192 | 0.206 | 0.152 |
| `lead_time_rule`: training no-show rate of the lead-time group (no ML) | 0.693 | 0.284 | 0.305 | 0.328 | 0.141 |
| `logreg`: logistic regression, default features | **0.726** | **0.332** | **0.340** | **0.365** | **0.138** |
| `logreg_balanced`: + `class_weight="balanced"` | 0.725 | 0.332 | 0.340 | 0.365 | 0.221 |
| `logreg_with_sms`: + SMS status (comparison only) | 0.727 | 0.334 | 0.341 | 0.367 | 0.138 |

### Findings

1. **The majority baseline is useless for this task.** It would be about 81% accurate on validation, yet it cannot rank anyone, which is why accuracy is not used.
2. **Lead time alone gets most of the way.** The rule with no ML reaches ROC-AUC 0.693 and finds 33% of no-shows in the top 20%.
3. **Logistic regression beats the rule** (ROC-AUC 0.726 vs 0.693; recall@top20 36.5% vs 32.8%). The other features add real, but modest, information. Contacting its top 20% would reach about 1.8× as many no-shows as contacting 20% of patients at random.
4. **Class weights don't help.** `balanced` leaves the ranking unchanged (same ROC-AUC and recall) but distorts the probabilities (Brier 0.221 vs 0.138). Because staff act on a ranked list, the imbalance is better handled by choosing the decision threshold. The model is kept unweighted.
5. **SMS status adds almost nothing** (+0.001 ROC-AUC). Excluding it, which is required because it is not known at booking time, costs practically no performance.

**Model to beat:** `logreg`, validation ROC-AUC 0.726, recall@top20 0.365.

## 2. Logistic regression tuning (commit `fcdbff4`)

`C` controls regularisation (smaller `C` = simpler model).

| Run | ROC-AUC | PR-AUC | recall@top20 | Train ROC-AUC |
|---|---|---|---|---|
| `logreg_C0.001` | 0.722 | 0.322 | 0.359 | 0.723 |
| `logreg_C0.01` | 0.726 | 0.332 | 0.363 | 0.727 |
| `logreg_C0.1` | 0.726 | 0.333 | 0.365 | 0.728 |
| `logreg_C1.0` | 0.726 | 0.332 | 0.365 | 0.728 |
| `logreg_C10.0` | 0.726 | 0.332 | 0.365 | 0.728 |

**Finding:** results are flat from `C = 0.01` upwards, and train and validation scores are almost equal, so the model is not overfitting. Only very strong regularisation (`C = 0.001`) hurts. The default `C = 1.0` is kept.

## 3. Random forest (commit `fcdbff4`)

300 trees. `leaf` = minimum samples per leaf; `feat` = share of features tried at each split.

| Run | ROC-AUC | PR-AUC | recall@top20 | Train ROC-AUC |
|---|---|---|---|---|
| `rf_leaf1_featsqrt` (fully grown trees) | 0.700 | 0.297 | 0.337 | **0.999** |
| `rf_leaf20_featsqrt` | **0.726** | 0.323 | 0.363 | 0.773 |
| `rf_leaf100_featsqrt` | 0.723 | 0.312 | 0.356 | 0.742 |
| `rf_leaf20_feat0.5` | 0.722 | 0.320 | 0.357 | 0.817 |

**Finding:** fully grown trees memorise the training data (train ROC-AUC 0.999) and do *worse* than logistic regression on validation. This is clear overfitting. Limiting leaf size to 20 fixes it and brings the forest level with logistic regression, but not above it.

## 4. Gradient boosting (commit `fcdbff4`)

scikit-learn `HistGradientBoostingClassifier` with early stopping on 10% of the training rows (the number of boosting rounds it chose is in `runs.csv`). `lr` = learning rate, `leaves` = maximum leaves per tree, `leaf` = minimum samples per leaf.

| Run | ROC-AUC | PR-AUC | recall@top20 | Train ROC-AUC |
|---|---|---|---|---|
| `hgb_default` | 0.728 | 0.330 | 0.364 | 0.777 |
| `hgb_lr0.03_leaves15_leaf20` | 0.728 | **0.335** | 0.365 | 0.755 |
| `hgb_lr0.03_leaves15_leaf200` | **0.729** | 0.331 | 0.368 | 0.759 |
| `hgb_lr0.03_leaves63_leaf20` | 0.725 | 0.328 | 0.357 | 0.800 |
| `hgb_lr0.03_leaves63_leaf200` | 0.726 | 0.326 | 0.368 | 0.777 |
| `hgb_lr0.1_leaves15_leaf20` | 0.728 | 0.334 | 0.360 | 0.758 |
| `hgb_lr0.1_leaves15_leaf200` | 0.728 | 0.331 | **0.371** | 0.763 |
| `hgb_lr0.1_leaves63_leaf20` | 0.726 | 0.330 | 0.363 | 0.787 |
| `hgb_lr0.1_leaves63_leaf200` | 0.725 | 0.326 | 0.367 | 0.772 |

**Finding:** small trees (15 leaves) work best; bigger trees (63 leaves) overfit slightly more and score lower. All nine settings fall between 0.725 and 0.729, so tuning makes little difference.

## 5. Are the best models really better? (commit `4410d30`)

The best run of each family was compared with logistic regression using a **paired bootstrap** on the validation set (`python -m src.train compare`): the validation rows are resampled 1,000 times, both models are scored on the same resample, and the difference is recorded. Full output: [`comparison.csv`](comparison.csv).

| Model vs logistic regression | Metric | Mean difference | 95% interval | Better in |
|---|---|---|---|---|
| Random forest (`rf_leaf20_featsqrt`) | ROC-AUC | +0.0005 | −0.0042 to +0.0053 | 60% of resamples |
| Random forest | recall@top20 | −0.0033 | −0.0181 to +0.0097 | 32% |
| Gradient boosting (`hgb_lr0.03_leaves15_leaf200`) | ROC-AUC | +0.0031 | −0.0021 to +0.0083 | 87% |
| Gradient boosting | recall@top20 | +0.0021 | −0.0119 to +0.0152 | 62% |

**Finding:** every 95% interval contains zero, so **no model is reliably better than logistic regression.** Gradient boosting is slightly ahead most of the time, but by an amount smaller than the noise.

## Conclusions so far

1. **All reasonable models reach about 0.73 ROC-AUC on validation.** The limit comes from the information in the features, not from the choice of algorithm. With lead time dominating and the other features weak (see the EDA), there is little non-linear structure for trees to exploit.
2. **Overfitting is real and visible** in unrestricted trees (random forest train ROC-AUC 0.999, validation 0.700), and regularisation fixes it.
3. **Choice for the final model (Day 6):** logistic regression performs as well as the best tree model, is simpler, has an equally good Brier score, and its coefficients can be explained to clinic staff. Gradient boosting is the only close alternative. The choice is made on this validation evidence; only the chosen model is then scored **once** on the test set, so the test result stays an unbiased estimate.
