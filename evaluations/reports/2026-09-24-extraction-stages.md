# Incremental extraction stage evaluation

**Snapshot:** 2026-09-24. All 11 stages were rescored offline with `evaluations.score_stages.evaluate` using 66 saved `Document` artifacts in `local/root-stage-eval/documents`: 26 primary filings and 10 filings in each other set. There were no extraction or model calls. The run and occurrence JSON are Git-ignored under `local/`; this tracked report is the remotely readable snapshot.

The scorer reconstructs each checkpoint with `Document.get_stage(stage)` and counts only field readings or relationships written by that stage. Later work cannot improve an earlier stage score. A downstream item is eligible only if its parent full locator existed before that stage. These are **conditional incremental scores**, not end-to-end citation or identity scores.

For field stages, **span P** divides exact spans by scored predictions; **span R** divides exact spans by eligible annotated spans. Predictions for unlabeled fields are excluded from span P and shown in diagnostics. **Norm accuracy** uses only matched source evidence with independent normalized gold; **norm recall** uses all eligible normalized gold. A dash means no applicable denominator. Inferred courts can lack source spans and still count toward normalization. Case names and short reporter citations have no independent normalized-gold target.

## Totals at a glance

| Field stage | Eligible gold | Predictions | Span P | Span R | Norm accuracy | Norm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full reporter locators (rule) (66) | 2190 | 2193 | 2186/2193 (99.7%) | 2186/2190 (99.8%) | 997/997 (100.0%) | 997/998 (99.9%) |
| Docket locators (rule) (66) | 195 | 97 | 96/97 (99.0%) | 96/195 (49.2%) | 52/52 (100.0%) | 52/124 (41.9%) |
| Docket locators (site hunting) (66) | 99 | 99 | 99/99 (100.0%) | 99/99 (100.0%) | 71/72 (98.6%) | 71/72 (98.6%) |
| Docket entries (66) | 27 | 27 | 27/27 (100.0%) | 27/27 (100.0%) | 27/27 (100.0%) | 27/27 (100.0%) |
| Case names (66) | 2306 | 2173 | 1882/2170 (86.7%) | 1882/2306 (81.6%) | — | — |
| Courts (66) | 2168 | 2099 | 1583/1587 (99.7%) | 1583/1600 (98.9%) | 2072/2085 (99.4%) | 2072/2168 (95.6%) |
| Dates (66) | 2189 | 2188 | 2173/2178 (99.8%) | 2173/2189 (99.3%) | 2173/2173 (100.0%) | 2173/2189 (99.3%) |
| Pin cites (66) | 1758 | 1763 | 1746/1756 (99.4%) | 1746/1758 (99.3%) | 1696/1746 (97.1%) | 1696/1758 (96.5%) |
| Short reporter citations (66) | 565 | 580 | 530/580 (91.4%) | 530/565 (93.8%) | — | — |

| Relationship stage | Eligible full locators | Exact groups / gold groups | Correct links / predicted links | Correct links / gold links |
| --- | ---: | ---: | ---: | ---: |
| Colocation (66) | 2381/2385 | 190/193 (98.4%) | 238/239 (99.6%) | 238/241 (98.8%) |
| Root formation (66) | 2381/2385 | 2036/2053 (99.2%) | 442/442 (100.0%) | 442/534 (82.8%) |

Colocation groups require at least two locators. Root groups include singletons, so the 2036/2053 exact-root-group rate obscures the more consequential 442/534 root-link recall. Both relationship stages score 2381 annotated full locators already present of 2385 total. Eight root assignments have no matching annotated locator; link scores concern matched locators.

## Diagnostic counts

A wrong-span prediction also leaves the corresponding gold field missed, so columns are not disjoint. Normalization failures count all unnormalizable predictions; normalization mismatches count only matched evidence with normalized gold.

| Stage | Missed eligible gold | Wrong span | Normalization failures | Normalization mismatches | Unlabeled field | Unmatched locator |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full reporter locators (rule) | 4 | 0 | 4 | 0 | 0 | 7 |
| Docket locators (rule) | 99 | 0 | 0 | 0 | 0 | 1 |
| Docket locators (site hunting) | 0 | 0 | 0 | 1 | 0 | 0 |
| Docket entries | 0 | 0 | 0 | 0 | 0 | 0 |
| Case names | 424 | 286 | 0 | 0 | 3 | 2 |
| Courts | 83 | 3 | 18 | 13 | 10 | 1 |
| Dates | 16 | 4 | 0 | 0 | 10 | 1 |
| Pin cites | 12 | 8 | 9 | 50 | 7 | 2 |
| Short reporter citations | 35 | 0 | 0 | 0 | 0 | 50 |

Colocation has 3 missed gold groups, 1 extra predicted group, 3 missed gold links, and 1 incorrect predicted link. Root formation has 17 missed gold groups, 56 extra predicted groups, 92 missed gold links, and no incorrect predicted links. The 99 gold docket locators missed by the rule stage are the eligibility set for site hunting, not misses after hunting.

## New docket-entry annotations

Ten previously unlabeled predictions were reviewed and annotated: six in primary, two in reliable-high-profile, and two in reliable-low-profile. They are source-anchored `D.I.`, `Dkt.`, and `ECF No.` references. Gold entries rose from 17 to 27; all 27 saved predictions now match both source span and normalized entry number. None remains unlabeled. The additions are recorded in the [annotation update manifest](../annotation_updates/2026-09-24-docket-entries.json).

## Results by stage and set

Total rows pool occurrences across sets rather than averaging set percentages.

### Full reporter locators (rule)

Eligibility: All unmasked annotated full reporter locators.

| Set (documents) | Eligible gold | Predictions | Span P | Span R | Norm accuracy | Norm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary (26) | 480 | 480 | 480/480 (100.0%) | 480/480 (100.0%) | 283/283 (100.0%) | 283/283 (100.0%) |
| hallucination-set-1 (10) | 376 | 377 | 376/377 (99.7%) | 376/376 (100.0%) | 282/282 (100.0%) | 282/282 (100.0%) |
| hallucination-set-2 (10) | 379 | 378 | 378/378 (100.0%) | 378/379 (99.7%) | 294/294 (100.0%) | 294/295 (99.7%) |
| reliable-high-profile (10) | 403 | 405 | 402/405 (99.3%) | 402/403 (99.8%) | 54/54 (100.0%) | 54/54 (100.0%) |
| reliable-low-profile (10) | 552 | 553 | 550/553 (99.5%) | 550/552 (99.6%) | 84/84 (100.0%) | 84/84 (100.0%) |
| **Total** (66) | 2190 | 2193 | 2186/2193 (99.7%) | 2186/2190 (99.8%) | 997/997 (100.0%) | 997/998 (99.9%) |

### Docket locators (rule)

Eligibility: All unmasked annotated docket locators.

| Set (documents) | Eligible gold | Predictions | Span P | Span R | Norm accuracy | Norm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary (26) | 54 | 19 | 18/19 (94.7%) | 18/54 (33.3%) | 12/12 (100.0%) | 12/44 (27.3%) |
| hallucination-set-1 (10) | 32 | 32 | 32/32 (100.0%) | 32/32 (100.0%) | 16/16 (100.0%) | 16/16 (100.0%) |
| hallucination-set-2 (10) | 9 | 1 | 1/1 (100.0%) | 1/9 (11.1%) | — | 0/8 (0.0%) |
| reliable-high-profile (10) | 34 | 19 | 19/19 (100.0%) | 19/34 (55.9%) | 11/11 (100.0%) | 11/19 (57.9%) |
| reliable-low-profile (10) | 66 | 26 | 26/26 (100.0%) | 26/66 (39.4%) | 13/13 (100.0%) | 13/37 (35.1%) |
| **Total** (66) | 195 | 97 | 96/97 (99.0%) | 96/195 (49.2%) | 52/52 (100.0%) | 52/124 (41.9%) |

### Docket locators (site hunting)

Eligibility: Annotated docket locators not already found by the rule stage.

| Set (documents) | Eligible gold | Predictions | Span P | Span R | Norm accuracy | Norm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary (26) | 36 | 36 | 36/36 (100.0%) | 36/36 (100.0%) | 31/32 (96.9%) | 31/32 (96.9%) |
| hallucination-set-1 (10) | 0 | 0 | — | — | — | — |
| hallucination-set-2 (10) | 8 | 8 | 8/8 (100.0%) | 8/8 (100.0%) | 8/8 (100.0%) | 8/8 (100.0%) |
| reliable-high-profile (10) | 15 | 15 | 15/15 (100.0%) | 15/15 (100.0%) | 8/8 (100.0%) | 8/8 (100.0%) |
| reliable-low-profile (10) | 40 | 40 | 40/40 (100.0%) | 40/40 (100.0%) | 24/24 (100.0%) | 24/24 (100.0%) |
| **Total** (66) | 99 | 99 | 99/99 (100.0%) | 99/99 (100.0%) | 71/72 (98.6%) | 71/72 (98.6%) |

### Docket entries

Eligibility: Annotated entries whose docket locator existed before this stage.

| Set (documents) | Eligible gold | Predictions | Span P | Span R | Norm accuracy | Norm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary (26) | 6 | 6 | 6/6 (100.0%) | 6/6 (100.0%) | 6/6 (100.0%) | 6/6 (100.0%) |
| hallucination-set-1 (10) | 17 | 17 | 17/17 (100.0%) | 17/17 (100.0%) | 17/17 (100.0%) | 17/17 (100.0%) |
| hallucination-set-2 (10) | 0 | 0 | — | — | — | — |
| reliable-high-profile (10) | 2 | 2 | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) |
| reliable-low-profile (10) | 2 | 2 | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) |
| **Total** (66) | 27 | 27 | 27/27 (100.0%) | 27/27 (100.0%) | 27/27 (100.0%) | 27/27 (100.0%) |

### Colocation

Eligibility: Annotated full locators present before colocation.

| Set (documents) | Eligible full locators | Exact groups / gold groups | Correct links / predicted links | Correct links / gold links |
| --- | ---: | ---: | ---: | ---: |
| primary (26) | 534/534 | 43/43 (100.0%) | 47/47 (100.0%) | 47/47 (100.0%) |
| hallucination-set-1 (10) | 408/408 | 38/39 (97.4%) | 42/43 (97.7%) | 42/43 (97.7%) |
| hallucination-set-2 (10) | 387/388 | 22/23 (95.7%) | 42/42 (100.0%) | 42/43 (97.7%) |
| reliable-high-profile (10) | 436/437 | 20/20 (100.0%) | 20/20 (100.0%) | 20/20 (100.0%) |
| reliable-low-profile (10) | 616/618 | 67/68 (98.5%) | 87/87 (100.0%) | 87/88 (98.9%) |
| **Total** (66) | 2381/2385 | 190/193 (98.4%) | 238/239 (99.6%) | 238/241 (98.8%) |

### Case names

Eligibility: Annotated case names whose full locator existed before this stage.

| Set (documents) | Eligible gold | Predictions | Span P | Span R | Norm accuracy | Norm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary (26) | 528 | 500 | 446/499 (89.4%) | 446/528 (84.5%) | — | — |
| hallucination-set-1 (10) | 390 | 362 | 317/362 (87.6%) | 317/390 (81.3%) | — | — |
| hallucination-set-2 (10) | 383 | 364 | 333/364 (91.5%) | 333/383 (86.9%) | — | — |
| reliable-high-profile (10) | 415 | 403 | 351/401 (87.5%) | 351/415 (84.6%) | — | — |
| reliable-low-profile (10) | 590 | 544 | 435/544 (80.0%) | 435/590 (73.7%) | — | — |
| **Total** (66) | 2306 | 2173 | 1882/2170 (86.7%) | 1882/2306 (81.6%) | — | — |

### Courts

Eligibility: Annotated courts whose full locator existed before this stage.

| Set (documents) | Eligible gold | Predictions | Span P | Span R | Norm accuracy | Norm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary (26) | 465 | 433 | 364/364 (100.0%) | 364/369 (98.6%) | 430/431 (99.8%) | 430/465 (92.5%) |
| hallucination-set-1 (10) | 338 | 327 | 226/226 (100.0%) | 226/226 (100.0%) | 325/326 (99.7%) | 325/338 (96.2%) |
| hallucination-set-2 (10) | 363 | 357 | 243/246 (98.8%) | 243/246 (98.8%) | 338/348 (97.1%) | 338/363 (93.1%) |
| reliable-high-profile (10) | 420 | 417 | 261/262 (99.6%) | 261/264 (98.9%) | 414/415 (99.8%) | 414/420 (98.6%) |
| reliable-low-profile (10) | 582 | 565 | 489/489 (100.0%) | 489/495 (98.8%) | 565/565 (100.0%) | 565/582 (97.1%) |
| **Total** (66) | 2168 | 2099 | 1583/1587 (99.7%) | 1583/1600 (98.9%) | 2072/2085 (99.4%) | 2072/2168 (95.6%) |

### Dates

Eligibility: Annotated dates whose full locator existed before this stage.

| Set (documents) | Eligible gold | Predictions | Span P | Span R | Norm accuracy | Norm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary (26) | 478 | 475 | 473/473 (100.0%) | 473/478 (99.0%) | 473/473 (100.0%) | 473/478 (99.0%) |
| hallucination-set-1 (10) | 341 | 343 | 341/341 (100.0%) | 341/341 (100.0%) | 341/341 (100.0%) | 341/341 (100.0%) |
| hallucination-set-2 (10) | 365 | 370 | 361/364 (99.2%) | 361/365 (98.9%) | 361/361 (100.0%) | 361/365 (98.9%) |
| reliable-high-profile (10) | 424 | 424 | 422/424 (99.5%) | 422/424 (99.5%) | 422/422 (100.0%) | 422/424 (99.5%) |
| reliable-low-profile (10) | 581 | 576 | 576/576 (100.0%) | 576/581 (99.1%) | 576/576 (100.0%) | 576/581 (99.1%) |
| **Total** (66) | 2189 | 2188 | 2173/2178 (99.8%) | 2173/2189 (99.3%) | 2173/2173 (100.0%) | 2173/2189 (99.3%) |

### Pin cites

Eligibility: Annotated pin cites whose full locator existed before this stage.

| Set (documents) | Eligible gold | Predictions | Span P | Span R | Norm accuracy | Norm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary (26) | 338 | 338 | 337/338 (99.7%) | 337/338 (99.7%) | 337/337 (100.0%) | 337/338 (99.7%) |
| hallucination-set-1 (10) | 275 | 281 | 274/275 (99.6%) | 274/275 (99.6%) | 273/274 (99.6%) | 273/275 (99.3%) |
| hallucination-set-2 (10) | 293 | 291 | 291/291 (100.0%) | 291/293 (99.3%) | 284/291 (97.6%) | 284/293 (96.9%) |
| reliable-high-profile (10) | 351 | 352 | 345/351 (98.3%) | 345/351 (98.3%) | 318/345 (92.2%) | 318/351 (90.6%) |
| reliable-low-profile (10) | 501 | 501 | 499/501 (99.6%) | 499/501 (99.6%) | 484/499 (97.0%) | 484/501 (96.6%) |
| **Total** (66) | 1758 | 1763 | 1746/1756 (99.4%) | 1746/1758 (99.3%) | 1696/1746 (97.1%) | 1696/1758 (96.5%) |

### Root formation

Eligibility: Annotated full locators present before root formation.

| Set (documents) | Eligible full locators | Exact groups / gold groups | Correct links / predicted links | Correct links / gold links |
| --- | ---: | ---: | ---: | ---: |
| primary (26) | 534/534 | 426/430 (99.1%) | 163/163 (100.0%) | 163/172 (94.8%) |
| hallucination-set-1 (10) | 408/408 | 332/335 (99.1%) | 76/76 (100.0%) | 76/145 (52.4%) |
| hallucination-set-2 (10) | 387/388 | 301/302 (99.7%) | 124/124 (100.0%) | 124/125 (99.2%) |
| reliable-high-profile (10) | 436/437 | 405/412 (98.3%) | 20/20 (100.0%) | 20/31 (64.5%) |
| reliable-low-profile (10) | 616/618 | 572/574 (99.7%) | 59/59 (100.0%) | 59/61 (96.7%) |
| **Total** (66) | 2381/2385 | 2036/2053 (99.2%) | 442/442 (100.0%) | 442/534 (82.8%) |

### Short reporter citations

Eligibility: All unmasked annotated short reporter citations.

| Set (documents) | Eligible gold | Predictions | Span P | Span R | Norm accuracy | Norm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary (26) | 40 | 41 | 34/41 (82.9%) | 34/40 (85.0%) | — | — |
| hallucination-set-1 (10) | 50 | 54 | 50/54 (92.6%) | 50/50 (100.0%) | — | — |
| hallucination-set-2 (10) | 59 | 67 | 58/67 (86.6%) | 58/59 (98.3%) | — | — |
| reliable-high-profile (10) | 235 | 236 | 211/236 (89.4%) | 211/235 (89.8%) | — | — |
| reliable-low-profile (10) | 181 | 182 | 177/182 (97.3%) | 177/181 (97.8%) | — | — |
| **Total** (66) | 565 | 580 | 530/580 (91.4%) | 530/565 (93.8%) | — | — |

## Representative remaining errors

- **Case-name boundary:** primary filing 002 has a quoted case name `Under Norton v. Shelby County`; gold begins at `Norton`. The extra introductory word makes the source span wrong.
- **Colocation across pagination:** in reliable-low-profile filing `68138504_37.txt`, a footnote separates annotated parallel reporter spans; the proximity rule leaves them apart.
- **Root equivalence:** in primary filing 005, two *Perez v. Sunbeam Prods., Inc.* cites use `21-CV-01915-PAB-KAS` and `21-CV-01915-PABKAS`; the root stage separates them.
- **Court reading:** `D.N.J. Mar.31` in hallucination-set-2 filing `69412014_46.txt` includes date text and cannot normalize as a court.
- **Docket normalization:** site hunting copies `CIV  16-0318  JB\SCY` from primary filing 006, while the annotated identifier is `CIV 16-0318 JB/SCY`. This is the one normalized docket-hunting disagreement.
- **Pin-cite normalization:** matched bare numbers such as `74950` are read as page pins where annotated normalization is empty; 50 of 1746 matched pin-cite readings disagree on normalized value.

This snapshot should be regenerated if annotations or saved Documents change. Detailed occurrence traces remain in the ignored local artifacts.
