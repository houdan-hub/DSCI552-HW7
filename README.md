# DSCI-552 Homework 7 — Multi-Label Classification and Clustering

Tianbo Hou · GitHub: houdan-hub · USC ID: 8943381210

## What this is

Machine-learning homework for DSCI 552 (USC). Part 1 classifies the
[Anuran Calls (MFCCs)](https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29)
data set — 7,195 frog syllables, 22 MFCCs, three nested labels (4 families,
8 genera, 10 species) — with support vector machines under binary relevance.
Part 2 runs a 50-repetition Monte-Carlo k-means study on the same features and
scores the cluster-majority triplets against the true labels. Part 3 is the
written answer to ISLR exercise 12.6.2.

## Contents

- `Hou_Tianbo_HW7.ipynb` — the full report: code, figures, and written discussion
  for every question. Executed top to bottom with no errors.
- `data/Frogs_MFCCs.csv` — the UCI data file (unmodified).
- `data/Readme.txt` — the data set description distributed with the data.
- `requirements.txt` — Python packages needed to re-run the notebook.

## How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute Hou_Tianbo_HW7.ipynb
```

Everything uses relative paths under the repo root, so a fresh clone runs as is.
Full execution takes roughly ten minutes on a laptop; the SMOTE pipelines in
1(b)iv and the 50 k-means repetitions in part 2 are the slowest cells. All
random seeds are fixed, so the numbers below reproduce exactly.

## Main results (test set, 2,159 syllables held out)

Metrics for the three-label prediction: exact match ratio (all three labels
correct), average Hamming distance (wrong labels per instance, 0-3) and Hamming
score (1 - Hamming loss).

| model | exact match | Hamming dist. | Hamming score | Family acc | Genus acc | Species acc |
|---|---|---|---|---|---|---|
| SVM, RBF kernel, raw attributes | 0.9907 | 0.0204 | 0.9932 | 0.9944 | 0.9926 | 0.9926 |
| SVM, RBF kernel, standardized | 0.9847 | 0.0301 | 0.9900 | 0.9931 | 0.9889 | 0.9880 |
| L1-SVM, standardized | 0.9176 | 0.1510 | 0.9497 | 0.9398 | 0.9486 | 0.9606 |
| L1-SVM + SMOTE (inside folds) | 0.8606 | 0.2089 | 0.9304 | 0.9152 | 0.9171 | 0.9588 |
| Classifier chain (L1) | 0.9240 | 0.1927 | 0.9358 | 0.9393 | 0.9347 | 0.9333 |

- Hyperparameters ($C$, $\sigma$ / $\gamma$) are tuned by 10-fold stratified
  cross-validation on the training set only.
- SMOTE is applied inside the cross-validation folds via `imblearn.Pipeline`,
  never before the split and never to the test set.
- Per-label confusion matrices, macro-averaged precision/recall/F1 and the
  micro-averaged one-versus-rest ROC over all (label, class) binary problems
  (AUC = 0.9992) are reported in §1(b)vi.
- Extra practice 1(b)v (classifier chains) is included: chaining the three labels
  gives 0.9240 exact match against 0.9176 for independent binary relevance.

## Clustering (part 2)

k-means on the standardized 22 MFCCs, Monte-Carlo over 50 repetitions, $k$
chosen by silhouette (Calinski-Harabasz is reported for comparison but decreases
monotonically and always returns $k=2$):

- chosen $k$: mean 3.9, sd 0.9, mode 4, range 3-6
- Hamming distance of cluster-majority triplets: **0.7401 ± 0.1001** over 50 runs
- Hamming score 0.7533, Hamming loss 0.2467

## References

- "Anuran Calls (MFCCs) Data Set." *UCI Machine Learning Repository*, archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29. Accessed 2026.
- Colonna, Juan Gabriel, et al. "An Incremental Technique for Real-Time Bioacoustic Signal Segmentation." *Expert Systems with Applications*, vol. 42, 2015, pp. 7367-7374.
- Sorower, Mohammad S. "A Literature Survey on Algorithms for Multi-label Learning." Oregon State University, 2010.
- Read, Jesse, et al. "Classifier Chains for Multi-label Classification." *Machine Learning*, vol. 85, no. 3, 2011, pp. 333-359.
- Chawla, Nitesh V., et al. "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, vol. 16, 2002, pp. 321-357.
- James, Gareth, et al. *An Introduction to Statistical Learning: with Applications in Python*. Springer, 2023.
