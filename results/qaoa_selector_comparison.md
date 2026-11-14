**EXPLORATORY: designed after the seed-42 pilot; not a pre-registered hypothesis and not part of any verdict.**

Same 16 candidates, same head, same rows; only the selector differs. RMSE / persistence: below 1 beats no-change.

| Selector | Seed | Features | n | QUBO value | Rank of 65,536 | Validation | Test |
|---|---|---|---|---|---|---|---|
| Classical (top-k by correlation) | - | [0, 1, 2, 5, 7, 11, 28, 29] | 8 | 11.808 | 35,126 | 1.000000 | 1.000000 |
| Exact QUBO optimum | - | [0, 1, 2, 3, 17, 21, 29, 30] | 8 | 1.839 | 1 | 0.999997 | 0.999999 |
| QAOA | 42 | [1, 2, 3, 11, 16, 17, 21, 26] | 8 | 3.884 | 2,566 | 0.999992 | 1.000022 |
| QAOA | 43 | [1, 2, 3, 11, 16, 17, 21, 26] | 8 | 3.884 | 2,566 | 0.999992 | 1.000022 |
| QAOA | 44 | [0, 1, 2, 3, 17, 21, 29, 30] | 8 | 1.839 | 1 | 0.999997 | 0.999999 |
| QAOA | 45 | [0, 1, 2, 3, 17, 21, 29, 30] | 8 | 1.839 | 1 | 0.999997 | 0.999999 |
| QAOA | 46 | [1, 5, 13, 17, 21, 26, 28, 29] | 8 | 4.848 | 3,694 | 0.999986 | 0.999995 |
| QAOA | 47 | [0, 1, 2, 3, 17, 21, 29, 30] | 8 | 1.839 | 1 | 0.999997 | 0.999999 |
| QAOA | 48 | [0, 1, 2, 3, 17, 21, 29, 30] | 8 | 1.839 | 1 | 0.999997 | 0.999999 |
| QAOA | 49 | [0, 1, 2, 3, 17, 21, 29, 30] | 8 | 1.839 | 1 | 0.999997 | 0.999999 |
| QAOA | 50 | [0, 1, 2, 3, 17, 21, 29, 30] | 8 | 1.839 | 1 | 0.999997 | 0.999999 |
| QAOA | 51 | [0, 1, 2, 3, 17, 21, 29, 30] | 8 | 1.839 | 1 | 0.999997 | 0.999999 |

QAOA over 10 seeds: test ratio mean 1.000004 (min 0.999995, max 1.000022); 8 seed(s) beat no-change on test.
QAOA minus classical: +3.59e-06 (95% CI [-3.48e-06, +1.07e-05]); QAOA minus exact: +4.11e-06 (95% CI [-2.97e-06, +1.12e-05]). The seeds share one dataset and gave only 3 distinct selections, so read these intervals as a description of seed-to-seed spread, not as a test.
QAOA found the exact optimum in 7 of 10 seeds; median rank 1 (best 1, worst 3,694); 10 seed(s) returned exactly 8 features; 3 distinct selections.
