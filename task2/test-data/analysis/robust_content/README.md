# Offline test of content-based model identity

The fixed rule failed the requested 0.65 model MCC target on all nine one-shot replays. The official local practice scorer verified all 12 values below, including the diagnostic runs pooling three samples. Each cell gives the model MCC over 36 reference pairs. The letters represent the same three preexisting sampling waves across all question blocks.

| Ten-question block | Repeat 0 | Repeat 1 | Repeat 2 | Pool all 3 |
|---|---:|---:|---:|---:|
| Exact ten queried on the 33 anonymous endpoints | 0.000 | 0.000 | 0.000 | 0.000 |
| Preassigned held-out ten from the prompt bank | 0.000 | 0.373 | 0.120 | 0.000 |
| Remaining disjoint ten | 0.000 | 0.000 | 0.194 | 0.000 |

The question partitions come from the team's prompt bank. The nine reference endpoints, identity labels, and scoring script come from the official starter. All windows reuse those same nine reference identities.

## Method and signal

An agglomerative algorithm compares the integrated evidence for sharing a categorical answer distribution with the evidence for two separate distributions. The prior comes from the unlabeled family answering that question. Its concentration is integrated over three fixed settings. The model handles a partial numeric or option response as a downweighted distribution and gives another view to error profiles relative to question difficulty. It merges when joint log evidence exceeds log(2). plan.txt specifies the primary method and the sensitivity cases before evaluation. The formal candidate uses the three previously inferred family labels.

Across nine one-shot panels, the mean signed MCC was −0.025. An exact permutation test preserving the 5+1+1+1+1 practice identity structure produced a mean at least as large on 74 of 126 relabelings (58.7%). An optimistic diagnostic retuned the cut using truth after freezing the trees. Its highest MCC over all nine primary trees was 0.373. This separates a failure in the proposed merge ordering from a failure in its chosen stopping rule. The threshold selected by that diagnostic was never used to generate a prediction.

Over all 30 questions with three draws, per-question strict answer agreement averaged 30.3% across repeats at the same endpoint, 27.4% across different deployments of the same model, and 25.0% across distinct models within the family. On the exact scored ten, those figures were 26.9%, 20.9%, and 25.5%. Conditioning on both replies being parseable can select different questions for different pairs. The soft overlap measure, which includes partial replies, yielded an even smaller 30-question gap (22.3% versus 21.3%). These diagnostics describe this bank, mapping and reference cohort.

## Reproduce

Run from the workspace root with Python, SciPy and NumPy, and retain the sibling work/identity_v2 modules. No script accesses a service. Input hashes for the scoring experiment and outputs per question block are in results.json.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 work/robust_content/panel.py
PYTHONDONTWRITEBYTECODE=1 python3 work/robust_content/noise.py
PYTHONDONTWRITEBYTECODE=1 python3 work/robust_content/verify.py
```

candidate.csv contains 33 endpoint assignments and 24 model labels. pair_evidence.csv has 528 unique pairs. Cross-family pairs carry no within-family log evidence. verified.json records the organizer's capped scores. The new partition has insufficient practice support for replacing an existing submission.
