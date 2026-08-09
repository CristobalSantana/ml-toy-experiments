# Pre-registered decision rules, evaluated

Rules fixed in CRITERIA.md before any result was seen.

Evaluated on the 20 paired (seed, fold) cells where every model ran.

- Best classical model: **catboost**, E_c = 0.0702 ± 0.0265
- TabPFN:               E_t = 0.0755 ± 0.0256
- Pooled spread s = 0.0521; |E_c - E_t| = 0.0053

- Cost ratios (TabPFN / catboost): total time 6.1x, fit time 0.04x, peak memory 173.5x

**Classical methods win** (rule 2): accuracy is statistically indistinguishable (|E_c - E_t| = 0.0053 <= s = 0.0521) and catboost reaches it at >=5x lower total time and >=5x lower peak memory.
