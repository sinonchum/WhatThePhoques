# Offline model-identity experiments

The snapshot covers 810 reference replies from 9 endpoints, with 30 questions sampled 3 times each, and 330 scored-endpoint replies from 33 endpoints, with 10 questions sampled once. The hand-written answer key in collision.py describes the questions in this repository's prompt bank. It contains no endpoint identity information. The first live CSV supplies the existing family grouping. Every analysis runs offline.

Run from the workspace root with Python, NumPy and SciPy. The data paths in the scripts refer to the task2 folder under outputs. inputs_manifest.json records input hashes and dependency versions for this snapshot.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 work/identity_v2/identity.py
PYTHONDONTWRITEBYTECODE=1 python3 work/identity_v2/collision.py
PYTHONDONTWRITEBYTECODE=1 python3 work/identity_v2/stability.py
```

## Files to inspect

- stable_scores.csv gives continuous rarity evidence for all 528 pairs. Within the existing families, it also gives the mean percentile across leave-one-question-out runs, the fraction of runs placing the pair in the top quartile, the number of distinct questions providing positive evidence, and the final cluster decision. Cross-family pairs have a raw evidence score but no within-family vote. A zero gated vote may mean fewer than two supporting questions.
- stable_submission.csv contains the exploratory partition with 22 identities and 20 positive pairs. The method keeps prior family labels. It applies average linkage to the gated votes using a cut of 0.70. The six non-singleton groups are [01,02,18,23], [04,12,14,24,25], [17,33], [21,32], [22,26], and [03,28]. Checks verified 33 unique endpoint IDs and a unique family per model.
- scores.csv and diagnostics.json contain the results for the lexical method. It compares strict normalized answers, weak numeric or choice evidence, ordered token shingles, character four-grams, and identical semantic text. Each prompt’s evidence uses inverse frequency and discounting for copied prompt text. The training data supported essentially only the exact long-answer feature. The numeric cutoff returned 33 singleton models on these formal replies.
- collision_scores.csv and collision_diagnostics.json contain the answer-collision and capability-profile comparison. The method gives more credit for sharing a rare numeric or choice response within a family. It reduces the weight of ambiguous answers and discounts topics with scarce shared evidence. Applying the development cutoff directly returned one model per prior family. stable_scores.csv addresses this threshold shift with relative ranks among all eligible same-family pairs.

## Validation

The 20 development questions and 10 held-out questions span five categories. On each reference run, the test side uses one reply per question and endpoint to match the formal query depth. These are different questions for the same 9 reference endpoints. The three runs reuse the same 36 pairs. Signed model MCCs below zero count as zero in official scoring.

| Run | Reference held-out model MCC, repeats 0 / 1 / 2 |
|---|---|
| Lexical features, fixed fit and threshold | -0.246 / 0.262 / -0.022 |
| Semantic rarity, development-selected fixed threshold | 0 / 0 / 0 |
| Capability profiles, development-selected fixed threshold | 0.231 / 0.308 / 0.009 |
| Rare agreement with rank voting and two-question gate | -0.022 / 0.636 / -0.022 |

The first two scripts also test individual pairs by excluding both endpoints from supervised weight or threshold fitting. Unlabeled per-prompt cohort statistics still include the endpoints being compared. The stability rule uses no endpoint labels or trained weights. Its held-out AUC by repetition was 0.565, 0.725 and 0.473. All nine references share one family, with one identity containing five endpoints and four singleton identities. The family hypothesis and transfer to the 33 anonymous endpoints have not been independently verified. The results support using stable pairs as an additional source of evidence. They give limited support for replacing the existing partition wholesale.
