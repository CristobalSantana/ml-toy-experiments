# Pre-registered decision rules, evaluated

Rules fixed in CRITERIA.md before any result was seen.

Evaluated on the 20 paired (seed, fold) cells where every model ran.

- Best classical model: **catboost**, E_c = 0.0704 ± 0.0259
- TabPFN:               E_t = 0.0747 ± 0.0259
- Pooled spread s = 0.0518; |E_c - E_t| = 0.0043

- Cost ratios (TabPFN / catboost): total time 7.1x, fit time 0.04x, peak memory 224.4x

**Classical methods win** (rule 2): accuracy is statistically indistinguishable (|E_c - E_t| = 0.0043 <= s = 0.0518) and catboost reaches it at >=5x lower total time and >=5x lower peak memory.
