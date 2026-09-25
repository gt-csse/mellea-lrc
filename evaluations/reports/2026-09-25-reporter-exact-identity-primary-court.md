# Identity at `reporter_root_exact_lookup`

<!-- Generated from saved counts by evaluations.render_identity_report. -->

**Score input:** `evaluations/results/2026-09-25-reporter-exact-identity-primary-court/summary.json`. This report makes no provider or model calls.

Prediction run: `stable` extraction rules; docket site hunting off; linked docket court retrieval on.

A decided root is a CORRECT_IDENTITY or WRONG_IDENTITY verdict; deferred roots abstain. Decision precision divides correct decided roots by scored decided roots. Decision recall uses every labeled gold reporter root represented by an unmasked full citation, including roots missed by extraction. Admission precision and recall count only correct-identity admissions, with all gold-correct reporter roots as the recall denominator. Unlabeled identity sets show a dash, not a zero score.

| Set | Documents | Gold reporter roots | Predicted roots | Decided | Decision precision | Decision recall | Admission precision | Admission recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| primary | 26 | 385 | 385 | 178 | 178/178 (100.0%) | 178/385 (46.2%) | 178/178 (100.0%) | 178/323 (55.1%) |
| Total | 26 | 385 | 385 | 178 | 178/178 (100.0%) | 178/385 (46.2%) | 178/178 (100.0%) | 178/323 (55.1%) |

Decision recall across all annotated root types: 178/430 (41.4%).

Deferred reporter roots: 207 (review 76, ambiguity 51, search 80).

## Field judgments

Gold field labels belong to the annotated root occurrence. A predicted root represented by a later citation is unscored for fields, even when its identity can be scored. A field comparison is scored only when its reading matches the annotated quote, span, and available normalized value; a court inferred from its reporter must have the annotated court ID. Misaligned readings are extraction or normalization errors. Gold marked `not_stated` is excluded from field accuracy and recall. An undetermined judgment is an abstention.

Annotated representative roots available for field scoring: 283 of 385 identity-labeled reporter roots.

Comparison accuracy is correct determinate judgments divided by scored determinate judgments. Judgment coverage is determinate judgments divided by aligned readings at unique exact lookups. Correct / stated gold shows this stage's field recall across the available unmasked annotated root occurrences with that field stated, including those without a unique exact lookup.

| Set | Field | Stated gold | Unique exact | Aligned reading | Comparison accuracy | Judgment coverage | Correct / stated gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary | Case name | 280 | 188 | 175 | 155/170 (91.2%) | 170/175 (97.1%) | 155/280 (55.4%) |
| primary | Court | 262 | 182 | 168 | 167/168 (99.4%) | 168/168 (100.0%) | 167/262 (63.7%) |
| primary | Date | 270 | 186 | 186 | 183/186 (98.4%) | 186/186 (100.0%) | 183/270 (67.8%) |
| Total | Case name | 280 | 188 | 175 | 155/170 (91.2%) | 170/175 (97.1%) | 155/280 (55.4%) |
| Total | Court | 262 | 182 | 168 | 167/168 (99.4%) | 168/168 (100.0%) | 167/262 (63.7%) |
| Total | Date | 270 | 186 | 186 | 183/186 (98.4%) | 186/186 (100.0%) | 183/270 (67.8%) |

Gold fields marked `not_stated`: 3 case name, 21 court, 13 date. These are not counted as agreeing or disagreeing fields.

A field can lack a unique exact lookup, have no aligned reading, or receive no determinate judgment. Unscored outputs include judgments on later representative occurrences, misaligned readings, and unmatched or unlabeled occurrences; they do not enter comparison accuracy.

| Set | Field | No unique exact | Missing reading | Misaligned reading | Wrong judgment | Undetermined | Omitted judgment | Unscored outputs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| primary | Case name | 92 | 6 | 7 | 15 | 5 | 0 | 67 |
| primary | Court | 80 | 14 | 0 | 1 | 0 | 0 | 54 |
| primary | Date | 84 | 0 | 0 | 3 | 0 | 0 | 60 |
| Total | Case name | 92 | 6 | 7 | 15 | 5 | 0 | 67 |
| Total | Court | 80 | 14 | 0 | 1 | 0 | 0 | 54 |
| Total | Date | 84 | 0 | 0 | 3 | 0 | 0 | 60 |

Of the unscored outputs, judgments on later representative occurrences account for 60 case name, 54 court, 60 date.

### Comparison directions across scored sets

`Agrees → mismatch` rejects an agreeing field; `Disagrees → match` accepts a disagreeing field. These counts use only aligned readings of the annotated root occurrence.

| Field | Agrees → match | Agrees → mismatch | Disagrees → match | Disagrees → mismatch | Agrees → undetermined | Disagrees → undetermined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Case name | 142 | 15 | 0 | 13 | 5 | 0 |
| Court | 154 | 1 | 0 | 13 | 0 | 0 |
| Date | 181 | 3 | 0 | 2 | 0 | 0 |

The reporter-locator membership gate found the queried citation in 242 unique lookup records, found a different listed citation in 12, and could not parse the listing in 0. This is a saved-response diagnostic; the annotations do not label the provider's citation list independently.

The occurrence file beside this summary lists each prediction and each gold root that was not correctly decided, plus field judgments, readings, returned record values, locator membership, and gaps, with exact locator spans.
