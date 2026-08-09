### full - Full data (five classical models; TabPFN cannot ingest this many rows)

| Model | Folds | Train rows | MAE (log10) | R² | MdAPE % | Fit s | Predict s | Peak MB |
|---|---|---|---|---|---|---|---|---|
| random_forest | 25 | 644,672 | 0.0394 ± 0.0015 | 0.893 | 4.7 | 75.42 | 0.868 | 2260 |
| catboost | 25 | 644,672 | 0.0434 ± 0.0015 | 0.902 | 6.6 | 179.32 | 0.201 | 664 |
| lightgbm | 25 | 644,672 | 0.0483 ± 0.0014 | 0.889 | 7.7 | 2.21 | 0.504 | 79 |
| mlp | 25 | 644,672 | 0.0499 ± 0.0018 | 0.882 | 7.9 | 38.08 | 0.185 | 312 |
| ridge | 25 | 644,672 | 0.0784 ± 0.0018 | 0.735 | 13.4 | 1.18 | 0.130 | 323 |

### regime_limited - Regime-limited (all six models, inside TabPFN's CPU envelope)

| Model | Folds | Train rows | MAE (log10) | R² | MdAPE % | Fit s | Predict s | Peak MB |
|---|---|---|---|---|---|---|---|---|
| catboost | 25 | 4,760 | 0.0735 ± 0.0256 | 0.692 | 11.5 | 27.71 | 0.006 | 2 |
| tabpfn | 20 | 4,631 | 0.0747 ± 0.0259 | 0.674 | 10.4 | 1.16 | 198.435 | 507 |
| lightgbm | 25 | 4,760 | 0.0787 ± 0.0247 | 0.649 | 13.1 | 0.34 | 0.011 | 0 |
| random_forest | 25 | 4,760 | 0.0792 ± 0.0258 | 0.630 | 13.1 | 0.38 | 0.059 | 4 |
| mlp | 25 | 4,760 | 0.0859 ± 0.0225 | 0.610 | 14.0 | 0.61 | 0.009 | 0 |
| ridge | 25 | 4,760 | 0.0907 ± 0.0228 | 0.603 | 15.4 | 0.02 | 0.008 | 0 |
