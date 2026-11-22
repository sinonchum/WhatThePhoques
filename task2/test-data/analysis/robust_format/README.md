# Conditional formatting experiment

Run `python3 work/robust_format/evaluate.py` from this workspace. It reads the two cached answer logs, the original question bank, practice labels and the previously estimated formal family map. All calculations run offline.

The estimator compares each endpoint with other endpoints on the same prompt. Binary events are counted only on eligible responses. For example, changing a copied option's capitalization is counted only when the text starts by naming an option and copying an answer choice. The narrow version uses changes in delimiter, terminal punctuation and capitalization. The extended version adds answer labels and standardized prose headings. The final version also measures whether a response reprints several option labels and whether a short declarative answer is followed by extra prose.

For each endpoint and feature, the other endpoints provide a prompt-specific event rate. Four pseudo-observations at the pooled same-format rate of the other prompts regularize sparse rate estimates. The endpoint score is the total observed excess divided by the square root of expected Bernoulli variance plus one. Overlapping features make this a ranking score. The reported numeric thresholds are operating points with no nominal false-positive guarantee. Decisions use only response features and the current family. Endpoint labels and practice identities enter scoring after prediction.

The three ten-question blocks are disjoint. Each contains two prompts from each of the five domains. For each block the report includes all three replies per cell, three fixed-repeat rank comparisons, and 1,024 draws that independently select a cached reply per endpoint and prompt. These simulations reuse the same nine endpoints and three observations per cell. Feature design was exploratory after inspecting the available practice answers.

| Ten-question block | Median one-draw MCC | Fraction reaching 0.65 | Weak LoRA split | System wrapper incorrectly split |
| --- | ---: | ---: | ---: | ---: |
| Exact formal question IDs | 0.734 | 54% | 68% | 18% |
| Published holdout | 0.734 | 96% | >99% | 2% |
| Remaining development prompts | 0.231 | 25% | 42% | 49% |

These rows use all seven features and assign the upper third of anomaly scores to individual model groups. On the remaining development block the system wrapper more often outranks the weak LoRA. All three cached repeats together still yield MCC 0.231 on that block. The feature is sensitive to the question panel and does not meet the requested stability target.

`report.json` records every evaluated feature set, cutoff, rank result and endpoint-specific rate. `formal_profiles.csv` contains continuous scores by endpoint for offline diagnostics under the estimated family map. No formal identity score can be validated from the caches.
