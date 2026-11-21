# Variant-specific content and sequence audit

The existing transition classifier detached the weak LoRA in 6 of 9 one-shot panels and the medium and strong LoRAs in 9 of 9. It detached FP8 in 0 of 9. Capturing all three LoRAs while absorbing FP8 into the five-endpoint base group already gives MCC 0.734 (10 TP, 5 FP, 21 TN, 0 FN). Aggregate MCC alone gives incomplete evidence about quantization.

The panels are the exact ten questions from the formal cache, the ten held-out questions in the team's prompt bank, and the remaining ten. Each block replays three actual single-response waves. All nine panels use the same nine reference deployments.

| Reference deployment | Baseline detaches | Unsupervised content addition | Nested supervised addition |
|---|---:|---:|---:|
| Plain | 0/9 | 0/9 | 0/9 |
| FP8 | 0/9 | 0/9 | 0/9 |
| Changed system prompt | 1/9 | 2/9 | 1/9 |
| LoRA 0.7 | 9/9 | 9/9 | 9/9 |
| Persona | 0/9 | 0/9 | 0/9 |
| Watermark | 1/9 | 1/9 | 1/9 |
| LoRA 1.5 | 9/9 | 9/9 | 9/9 |
| Temperature 1.0 | 1/9 | 1/9 | 1/9 |
| LoRA 0.3 | 6/9 | 6/9 | 6/9 |

## Experiments

variant.py reproduces the predictor in work/robust_latent/transitions.py, then computes semantic errors, disagreement between a marked choice and its echoed option, soft uncertainty, numeric deviations, arithmetic internal consistency, copied/repeated wording, and DEFLATE compression features. It estimates their expected values using peer endpoints on the same question weighted by similarity of response length. No endpoint ID is a model feature. A conservative, label-free alternative allows an unusual content profile to split a borderline endpoint. Its only change on the formal cache is an extra split of ep-32, which lacks supporting practice evidence.

A second alternative learns a shrunk correction to the sequence scores. Every outer fit excludes the entire test question block and both a variant endpoint and a nuisance endpoint, including their labels. Inner tuning leaves an additional training endpoint out and switches between the two remaining question blocks. Unlabeled cohort statistics are available at inference. Three correction strengths include zero; zero was selected in 51 of 60 outer fits. Nested per-endpoint votes are summarized in report.json alongside the original fold-level decisions. The final majority decisions match the baseline on every one-shot panel. There is only one reference FP8 endpoint, so a fold withholding it has no independently labeled FP8 example.

repeated.py evaluates all three two-draw combinations from the existing cache. It adds semantic self-consistency, first- and last-token overlap, normalized cross-reply compression, response-length changes, and discordant choices to the same nested evaluation. Model MCC remains identical to the original transition scores on all nine two-draw tests. The FP8 endpoint's rate of identical strict normalized answers on comparable repeated replies was 0.150, 0.400, and 0.167 across the three prompt blocks. The plain endpoint yielded 0.111, 0.421, and 0.250. The direction of that difference changes with question mix.

repeat-and-threshold.json also reports a diagnostic that relaxes the baseline threshold until a known target endpoint would be detected. Catching FP8 this way would falsely detach a median of three of the five nuisance/base deployments per panel (range 0 to 5). Detecting weak LoRA this way would cost a median of zero false detaches (range 0 to 2). This post-hoc calculation was never used for predictions.

## Reproduction

Run from the workspace root. Both scripts operate only on the local caches and use NumPy, SciPy, and the code in work/robust_latent, work/robust_format and work/identity_v2.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 work/variant_content_next/variant.py
PYTHONDONTWRITEBYTECODE=1 python3 work/variant_content_next/repeated.py
```

The transition scores produced the same baseline decisions as the original predictor on all nine one-shot panels. candidate.csv contains 33 unique formal endpoint IDs with one family per model. No candidate was submitted.
