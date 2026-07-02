# Ambiguous Resolution

A reporter lookup returns **HTTP 300** when the citation is valid but matches
more than one CourtListener cluster. Validation pulls the candidates back;
deciding which one the citation refers to is an opinion, so that is delegated to
assessment. See also [Validation Model Development](./Validation%20Model%20Development.md).

## What ambiguity actually is (real data)

We ran our test corpus through the deployed cl-access lookup (45 unique locators:
32 found, 9 not-found, **4 ambiguous ≈ 9%**). Every ambiguous case was the **same
underlying case returned as two duplicate clusters** — not the reporter-ambiguity
or "different cases on one page" the API docs describe:

| Locator | Cluster A | Cluster B | Cause |
|---|---|---|---|
| 16 F.3d 1083 | `Robinson v. Missouri Pacific Railroad` | `Lewis R. Robinson v. Missouri Pacific Railroad Company` | short vs full name, same case/date |
| 784 F.2d 1040 | `Perrin v. Anderson` | `Perrin v. Anderson` | identical |
| 405 F.3d 764 | `Bains LLC v. ARCO Products Co.` | `…, Div. of Atlantic Richfield Co.` | same case, date off-by-one |
| 215 F.3d 1140 | `No. 98-2228` (junk stub) | `Schwartz v. American College of Emergency Physicians` | stub vs real |

**docket_id analysis** (why we now keep it on every lookup):

- Duplicate clusters have **different** `docket_id`s in all 4 pairs → docket is *not*
  a within-pair de-dup signal, but confirms they are independent ingestions.
- `docket_id` was **never null** (0/32 found) and is present even on the junk stub.
- It **repeats across parallel-cite citations of one case** → a strong *cross-citation*
  identity key, kept for future use.

So the discriminators for "same case" are name subsumption/equality, same
`date_filed`, and stub detection (`case_name` like `No. 1234`, empty
`case_name_full`, `citation_count == 0`) — all comparisons, hence assessment's job.

## Shipped

**Validation — retrieve only.** Ambiguous candidates are returned as `CitationMatch`
records; `docket_id` is now a structured field on all lookups (was buried in
`extra_data`). No collapse, no opinion.

**Assessment — found-branch per candidate.** The pipeline was restructured into
per-`(citation, candidate)` jobs, all delegating to the same `assess_found_citation`:

- Found → one candidate → `AssessedCitationAssessment`.
- Ambiguous → one candidate per cluster → `AmbiguousCitationAssessment`, holding a
  `CandidateAssessment(match, result)` per cluster (order preserved).
- **Defensive gate**: more than `MAX_AMBIGUOUS_CANDIDATES` (5) → `gated=True`, no
  enumeration, reason recorded. Empty-candidate lookups short-circuit the same way.
- Court for a candidate uses its own `match.court_id` (usually absent); no
  per-candidate docket resolution runs, so court stays best-effort here.

**Frontend.** Ambiguous citations are assessment-eligible and selectable. The
right-column panel shows the selected candidate's full field-by-field trace,
**synced to the bibliographic candidate switcher** — flipping the CourtListener
candidate flips its assessment. Gated/empty shows a "not enumerated" note.

## Deferred / future direction

- **Drawing a single conclusion** across candidates (or prefiltering duplicates
  before assessment) is intentionally not implemented — each candidate is surfaced
  independently until we have enough assessed data to choose a rule. Given the
  data above, a likely first rule is *collapse duplicates* (name subsumption +
  same date + stub detection) → promote to a single effective `found`, with true
  multi-case ambiguity (different names AND dates) as the rare fallback.
