# ACE baseline model report

## Classification data

- Records: 1,792
- Positive records: 896
- Candidate-negative records: 896
- Length scope: 5–18 amino acids
- Similarity clusters: 1,576
- Selected exploratory model: `sequence_extra_trees`

The classes were sampled to the same count at each exact peptide length. This
removes the strongest length shortcut but does not turn AI4ACEIP candidate
negatives into experimentally confirmed negatives.

### Clustered five-fold cross-validation

```
                     roc_auc          pr_auc         balanced_accuracy              f1             mcc        
                        mean     std    mean     std              mean     std    mean     std    mean     std
model                                                                                                         
length_only_logistic  0.4806  0.0127  0.4876  0.0093            0.4732  0.0265  0.4770  0.0244 -0.0540  0.0535
sequence_extra_trees  0.7679  0.0081  0.7646  0.0189            0.6853  0.0133  0.6894  0.0111  0.3711  0.0263
sequence_logistic     0.6821  0.0163  0.6548  0.0342            0.6490  0.0134  0.6519  0.0091  0.2985  0.0264
```

The length-only model is reported as a shortcut diagnostic. A useful sequence
model should materially outperform it.

## pIC50 regression data

- Unique peptide targets: 1,176
- Length scope: 2–30 amino acids
- Similarity clusters: 1,050
- Sequences with repeated measurements: 599
- Selected exploratory model: `sequence_extra_trees`

Repeated exact measurements were summarized by their median at sequence level.
The minimum, maximum, count and span remain in `regression_dataset.csv` for
heterogeneity review.

### Clustered five-fold cross-validation

```
                     mae_pIC50         rmse_pIC50              r2         spearman        
                          mean     std       mean     std    mean     std     mean     std
model                                                                                     
sequence_extra_trees    0.7164  0.0335     0.9029  0.0600  0.1860  0.0430   0.4236  0.0468
sequence_ridge          1.0714  0.0800     1.5967  0.2007 -1.6367  0.9473   0.2398  0.0577
```

## Use restrictions

1. Do not use the classifier outside 5–18 amino acids.
2. Do not interpret its probability as clinical efficacy or absorption.
3. Do not label unknown DigiDigest fragments as negative.
4. Do not claim INFOGEST validity from BIOPEP rule-based fragments.
5. Treat both fitted models as baselines until external and laboratory validation.
6. Rank candidates with separate activity, digestion, transport and safety evidence rather than one unexplained probability.
