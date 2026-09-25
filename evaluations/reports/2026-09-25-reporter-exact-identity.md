# Identity at `reporter_root_exact_lookup`

<!-- Generated from saved counts by evaluations.render_identity_report. -->

**Score input:** `evaluations/results/2026-09-25-reporter-exact-identity/summary.json`. This report makes no provider or model calls.

Prediction run: `stable` extraction rules; docket site hunting off; linked docket court retrieval off.

A decided root is a CORRECT_IDENTITY or WRONG_IDENTITY verdict; deferred roots abstain. Decision precision divides correct decided roots by scored decided roots. Decision recall uses every labeled gold reporter root represented by an unmasked full citation, including roots missed by extraction. Admission precision and recall count only correct-identity admissions, with all gold-correct reporter roots as the recall denominator. Unlabeled identity sets show a dash, not a zero score.

| Set | Documents | Gold reporter roots | Predicted roots | Decided | Decision precision | Decision recall | Admission precision | Admission recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| primary | 26 | 385 | 385 | 67 | 67/67 (100.0%) | 67/385 (17.4%) | 67/67 (100.0%) | 67/323 (20.7%) |
| hallucination-set-1 | 10 | 319 | 322 | 77 | 75/77 (97.4%) | 75/319 (23.5%) | 75/77 (97.4%) | 75/266 (28.2%) |
| Total | 36 | 704 | 707 | 144 | 142/144 (98.6%) | 142/704 (20.2%) | 142/144 (98.6%) | 142/589 (24.1%) |

Decision recall across all annotated root types: 142/762 (18.6%).

Deferred reporter roots: 563 (review 350, ambiguity 85, search 128).

## Field judgments

Gold field labels belong to the annotated root occurrence. A predicted root represented by a later citation is unscored for fields, even when its identity can be scored. A field comparison is scored only when its reading matches the annotated quote, span, and available normalized value; a court inferred from its reporter must have the annotated court ID. Misaligned readings are extraction or normalization errors. Gold marked `not_stated` is excluded from field accuracy and recall. An undetermined judgment is an abstention.

Annotated representative roots available for field scoring: 567 of 704 identity-labeled reporter roots.

Comparison accuracy is correct determinate judgments divided by scored determinate judgments. Judgment coverage is determinate judgments divided by aligned readings at unique exact lookups. Correct / stated gold shows this stage's field recall across the available unmasked annotated root occurrences with that field stated, including those without a unique exact lookup.

| Set | Field | Stated gold | Unique exact | Aligned reading | Comparison accuracy | Judgment coverage | Correct / stated gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary | Case name | 280 | 188 | 175 | 155/170 (91.2%) | 170/175 (97.1%) | 155/280 (55.4%) |
| primary | Court | 262 | 182 | 168 | — | 0/168 (0.0%) | 0/262 (0.0%) |
| primary | Date | 270 | 186 | 186 | 183/186 (98.4%) | 186/186 (100.0%) | 183/270 (67.8%) |
| hallucination-set-1 | Case name | 284 | 208 | 177 | 143/164 (87.2%) | 164/177 (92.7%) | 143/284 (50.4%) |
| hallucination-set-1 | Court | 246 | 181 | 171 | — | 0/171 (0.0%) | 0/246 (0.0%) |
| hallucination-set-1 | Date | 248 | 184 | 182 | 180/182 (98.9%) | 182/182 (100.0%) | 180/248 (72.6%) |
| Total | Case name | 564 | 396 | 352 | 298/334 (89.2%) | 334/352 (94.9%) | 298/564 (52.8%) |
| Total | Court | 508 | 363 | 339 | — | 0/339 (0.0%) | 0/508 (0.0%) |
| Total | Date | 518 | 370 | 368 | 363/368 (98.6%) | 368/368 (100.0%) | 363/518 (70.1%) |

Gold fields marked `not_stated`: 3 case name, 59 court, 49 date. These are not counted as agreeing or disagreeing fields.

A field can lack a unique exact lookup, have no aligned reading, or receive no determinate judgment. Unscored outputs include judgments on later representative occurrences, misaligned readings, and unmatched or unlabeled occurrences; they do not enter comparison accuracy.

| Set | Field | No unique exact | Missing reading | Misaligned reading | Wrong judgment | Undetermined | Omitted judgment | Unscored outputs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| primary | Case name | 92 | 6 | 7 | 15 | 5 | 0 | 67 |
| primary | Court | 80 | 14 | 0 | 0 | 124 | 44 | 41 |
| primary | Date | 84 | 0 | 0 | 3 | 0 | 0 | 60 |
| hallucination-set-1 | Case name | 76 | 10 | 21 | 21 | 13 | 0 | 52 |
| hallucination-set-1 | Court | 65 | 9 | 1 | 0 | 104 | 67 | 23 |
| hallucination-set-1 | Date | 64 | 2 | 0 | 2 | 0 | 0 | 32 |
| Total | Case name | 168 | 16 | 28 | 36 | 18 | 0 | 119 |
| Total | Court | 145 | 23 | 1 | 0 | 228 | 111 | 64 |
| Total | Date | 148 | 2 | 0 | 5 | 0 | 0 | 92 |

Of the unscored outputs, judgments on later representative occurrences account for 91 case name, 63 court, 92 date.

### Comparison directions across scored sets

`Agrees → mismatch` rejects an agreeing field; `Disagrees → match` accepts a disagreeing field. These counts use only aligned readings of the annotated root occurrence.

| Field | Agrees → match | Agrees → mismatch | Disagrees → match | Disagrees → mismatch | Agrees → undetermined | Disagrees → undetermined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Case name | 269 | 36 | 0 | 29 | 16 | 2 |
| Court | 0 | 0 | 0 | 0 | 204 | 24 |
| Date | 356 | 5 | 0 | 7 | 0 | 0 |

The reporter-locator membership gate found the queried citation in 470 unique lookup records, found a different listed citation in 24, and could not parse the listing in 0. This is a saved-response diagnostic; the annotations do not label the provider's citation list independently.

The occurrence file beside this summary lists each prediction and each gold root that was not correctly decided, plus field judgments, readings, returned record values, locator membership, and gaps, with exact locator spans.
