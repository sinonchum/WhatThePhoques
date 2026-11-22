# Stability of the transition fingerprint

Run `python3 work/variant_stability_next/compare.py` for the practice replays and `python3 work/variant_stability_next/formal_diagnostics.py` to summarize existing formal replies. Both use local files only.

The three blocks contain ten disjoint questions each. For each block, 1,024 trials independently draw one of the three real cached replies for every endpoint and question. All methods see the same draw in a trial. These are replays on the same nine reference endpoints. They provide no independent test of transfer to new endpoint families. The feature set and sensitivity analyses are exploratory.

| Method | Fraction of exact-formal-question replays reaching MCC 0.65 | Published holdout | Remaining ten | Median on remaining ten |
| --- | ---: | ---: | ---: | ---: |
| Original transition score | 0.785 | 0.995 | 0.303 | 0.231 |
| Retain integer alignment; use answer-order signal only on choice prompts | 0.832 | 0.997 | 0.350 | 0.231 |
| Cap leave-one-question influences at each endpoint's 90th percentile | 0.753 | 0.997 | 0.489 | 0.231 |
| Average raw ranks and capped-influence consensus ranks | 0.683 | 0.984 | 0.464 | 0.524 |
| Require a variant flag in two thirds of leave-one-choice/one-integer runs | 0.630 | 0.955 | 0.177 | 0.524 |

The 90th-percentile influence cap raised the hardest panel's threshold hit rate, while its median remained below target. Confidence gating trades false splits for false merges. On the hardest 36-pair panel the original mean was 7.79 false merges and 2.79 false splits. The two-thirds gate produced 9.04 false merges and 1.91 false splits. The 90th-percentile cap produced 7.05 false merges and 2.05 false splits.

The diagnostic can abstain. A leave-two-out variant vote above or equal to two thirds supports a flag; votes between one third and two thirds leave the endpoint unresolved. A stable variant at either end resolves a pair as different. In the hardest practice block 91% of pairs received a diagnostic decision and 28% of those decisions were wrong. The official CSV requires every endpoint and cannot express abstention.

In `formal_uncertainty.csv` fewer than two distinct questions with a noncanonical grammar event or an order signal gives the status `insufficient_noncanonical_evidence`. This applies to 29 of the 33 formal endpoints under the existing cached ten-question panel. Absence of observed evidence for a split cannot verify identical weights or quantization. The file contains descriptive scores under a previously estimated family map.

`results.json` holds all thresholds, feature views, pair errors, and endpoint-specific rates. The adaptive rules select a cut by minimizing expected false-merge and false-split cost under leave-two-out partitions. These bag votes estimate perturbation sensitivity. They are not calibrated identity probabilities.
