### full - Full data (five classical models; TabPFN cannot ingest this many rows)

| Model | Folds | Train rows | MAE (log10) | R² | MdAPE % | Fit s | Predict s | Peak MB |
|---|---|---|---|---|---|---|---|---|
| random_forest | 25 | 644,672 | 0.0394 ± 0.0015 | 0.893 | 4.7 | 76.38 | 0.972 | 2258 |
| catboost | 25 | 644,672 | 0.0435 ± 0.0014 | 0.902 | 6.6 | 170.01 | 0.215 | 676 |
| lightgbm | 25 | 644,672 | 0.0483 ± 0.0014 | 0.889 | 7.7 | 2.23 | 0.477 | 88 |
| mlp | 25 | 644,672 | 0.0494 ± 0.0017 | 0.883 | 7.8 | 38.19 | 0.189 | 325 |
| ridge | 25 | 644,672 | 0.0784 ± 0.0018 | 0.735 | 13.4 | 1.23 | 0.128 | 334 |

### regime_limited - Regime-limited (all six models, inside TabPFN's CPU envelope)

| Model | Folds | Train rows | MAE (log10) | R² | MdAPE % | Fit s | Predict s | Peak MB |
|---|---|---|---|---|---|---|---|---|
| catboost | 25 | 4,760 | 0.0731 ± 0.0265 | 0.683 | 11.1 | 23.82 | 0.004 | 3 |
| tabpfn | 20 | 4,631 | 0.0755 ± 0.0256 | 0.673 | 10.7 | 0.86 | 147.474 | 496 |
| lightgbm | 25 | 4,760 | 0.0790 ± 0.0251 | 0.645 | 13.1 | 0.16 | 0.008 | 0 |
| random_forest | 25 | 4,760 | 0.0794 ± 0.0270 | 0.614 | 12.9 | 0.29 | 0.045 | 4 |
| ridge | 25 | 4,760 | 0.0907 ± 0.0228 | 0.603 | 15.4 | 0.02 | 0.007 | 0 |
| mlp | 25 | 4,760 | 0.0911 ± 0.0268 | 0.548 | 15.0 | 0.52 | 0.008 | 0 |
