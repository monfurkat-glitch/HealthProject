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
