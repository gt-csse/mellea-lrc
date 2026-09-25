# Identity at `reporter_root_exact_lookup`

<!-- Generated from saved counts by evaluations.render_identity_report. -->

**Score input:** `evaluations/results/2026-09-25-reporter-exact-identity/summary.json`. This report makes no provider or model calls.

Prediction run: `stable` extraction rules; docket site hunting off.

A decided root is a CORRECT_IDENTITY or WRONG_IDENTITY verdict; deferred roots abstain. Decision precision divides correct decided roots by scored decided roots. Decision recall uses every labeled gold reporter root represented by an unmasked full citation, including roots missed by extraction. Admission precision and recall count only correct-identity admissions, with all gold-correct reporter roots as the recall denominator. Unlabeled identity sets show a dash, not a zero score.

| Set | Documents | Gold reporter roots | Predicted roots | Decided | Decision precision | Decision recall | Admission precision | Admission recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| primary | 26 | 385 | 385 | 67 | 67/67 (100.0%) | 67/385 (17.4%) | 67/67 (100.0%) | 67/323 (20.7%) |
| hallucination-set-1 | 10 | 319 | 322 | 77 | 75/77 (97.4%) | 75/319 (23.5%) | 75/77 (97.4%) | 75/266 (28.2%) |
| Total | 36 | 704 | 707 | 144 | 142/144 (98.6%) | 142/704 (20.2%) | 142/144 (98.6%) | 142/589 (24.1%) |

Decision recall across all annotated root types: 142/762 (18.6%).

Deferred reporter roots: 563 (review 350, ambiguity 85, search 128).

The occurrence file beside this summary lists each prediction and each gold root that was not correctly decided, including its exact locator span.
