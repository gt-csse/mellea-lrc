# Retrieval Route Research Corpus [in progress]

## Status and purpose

This is an immediate research fixture, not a source-routing recommendation.
It records cases from the 2026-07-03 retrieval snapshots that can be used to
compare proposed not-found recovery routes. Route preference, ordering, and
implementation remain undecided until the experiments are complete.

The research must answer three different questions:

1. **Case recommendation:** which retrieved case or cases should be presented as
   plausible matches for the cited authority?
2. **Locator recovery:** which volume/reporter/page, slip, neutral, WL, LEXIS,
   or parallel locators can be recovered for each candidate identity?
3. **User choice:** what evidence and alternatives must be displayed so a user
   can select a case without relying on an unexplained system verdict?

These questions require related evidence but have different success criteria.
A route that retrieves the correct case may not recover its reporter locator. A
route that recovers a locator from a citing opinion may not provide the target
opinion. A broad candidate list may support user choice while being unsuitable
for an automatic recommendation.

## Snapshot scope

The corpus comes from cleanly overwritten snapshots for `bookmarked.txt` and
`1.txt` through `10.txt` under `local/snapshots/<document>/`. Preprocessing,
extraction, and retrieval ran; assessment did not run.

The snapshots contain 423 citation retrieval records. CourtListener began
returning HTTP 429 during the later documents. Throttled records are preserved
as transport observations, not treated as retrieval misses.

Research fixtures refer to `<document>/<citation_id>`, which is stable within
this snapshot set. Repeated citations are deliberately retained where they can
test consistency, but aggregate evaluation must also deduplicate by normalized
input and by retrieved candidate identity.

## Candidate route inventory

No order is assigned to these routes yet:

- CourtListener `type=o` cluster metadata search;
- CourtListener opinion-text search and citation-echo extraction;
- CourtListener `type=r` RECAP case search;
- CourtListener `type=rd` filing-document search;
- authenticated RECAP document metadata, including `is_free_on_pacer` and
  `is_available`;
- GovInfo USCOURTS metadata and opinion packages;
- issuing-court opinion and case-information surfaces;
- CAP or other applicable structured historical collections;
- compliant legal verticals or licensed sources;
- constrained and unconstrained web discovery.

Each experiment must preserve raw route results. Combining evidence is a later
operation and must not erase which source supplied which fact.

## Primary research cohorts

### A. One-corpus, low-count candidates

These test whether a narrow search result actually contains the correct case
and whether the other corpus adds anything.

| Fixture | Extracted locator and name | Opinion clusters | RECAP cases | Research value |
| --- | --- | ---: | ---: | --- |
| `8/cite-0001` | `444 F. Supp. 3d 593`, *Oconner v. Agilant Solutions* | 0 | 1 | RECAP-only federal candidate; repeated four times in document 8. |
| `3/cite-0029` | `2011 WL 1833007`, *Doe v. Penzato* | 0 | 1 | RECAP-only WL candidate. |
| `3/cite-0038` | `2022 WL 3447983`, *Doe v. Northrop Grumman* | 0 | 1 | Recent RECAP-only WL candidate. |
| `9/cite-0001` | `180 A.D.3d 83`, *Breest v. Haggis* | 1 | 0 | State opinion-cluster-only candidate; repeated in `9/cite-0023`. |
| `3/cite-0036` | `2017 WL 4460441`, *Doe v. Maricopa County Community College District* | 2 | 0 | Small state/federal-name cluster set with no RECAP case result. |

Questions:

- Does the low count contain the intended case or merely a lexical neighbor?
- Does the candidate expose the original locator, a parallel locator, or only
  identity metadata?
- Can the candidate document be retrieved, and from which source?
- Are repeated citations handled consistently?

### B. Cross-corpus candidate sets

These test identity linking when both corpora return plausible but structurally
different objects.

| Fixture | Extracted locator and name | Opinion clusters | RECAP cases | Research value |
| --- | --- | ---: | ---: | --- |
| `2/cite-0023` | `2021 WL 3081160`, *E.E.O.C. v. Maricopa County Community College District* | 4 | 1 | Small cluster set plus one docket candidate. |
| `3/cite-0020` | `2023 WL 3568691`, *Doe v. Amazon.com* | 1 | 19 | One opinion cluster versus many RECAP cases. |
| `4/cite-0043` | `2021 WL 7540814`, *Pizzo v. City of Chandler* | 2 | 1 | Small result sets in both corpora. |
| `bookmarked/cite-0019` | `400 F. Supp. 3d 1122`, *Peterson v. Nelnet Diversified Solutions* | 0 | 8 | Proceeding family used in the prior cross-corpus investigation; repeated across documents. |
| `3/cite-0039` | `2021 WL 392929`, *Doe v. Hobart & William Smith Colleges* | 0 | 4 | Multiple RECAP dockets without an opinion-cluster result. |

Questions:

- Can a RECAP docket be linked to an opinion cluster using explicit IDs, or
  only by caption, court, date, and docket number?
- Are multiple results versions, appeals, related proceedings, or unrelated
  same-name cases?
- Which identity should be recommended, and which related identities should be
  shown separately to the user?
- Can later citing opinions recover a missing reporter locator after the target
  cluster is absent?

### C. Broad and ambiguous queries

These test whether the current one-anchor-per-party query masks the useful
candidate inside an unreviewable count.

| Fixture | Query | Opinion clusters | RECAP cases |
| --- | --- | ---: | ---: |
| `1/cite-0007` | `caseName:(United AND Begay)` | 59 | 1,322 |
| `1/cite-0036` | `caseName:(Estate AND City)` | 1,196 | 648 |
| `4/cite-0009` | `caseName:(Johnson AND City)` | 1,534 | 1,916 |
| `5/cite-0020` | `caseName:(Dixon AND Cty)` | 66 | 200 |
| `6/cite-0013` | `caseName:(Lynch AND New)` | 121 | 58 |

Questions:

- Which additional grounded fields reduce the set without excluding the target:
  court, year, docket number, complete party tokens, locator text, or context?
- Does candidate retrieval remain useful when counts are large but only a small
  ranked page is inspected?
- What evidence supports stopping rather than presenting a misleading top hit?

### D. Locator-family coverage

These test whether route performance depends on the locator family.

- Reporter citations missing from exact lookup: `400 F. Supp. 3d 1122`,
  `444 F. Supp. 3d 593`, `180 A.D.3d 83`, `39 N.Y.S.3d 580`, and
  `574 U.S. 10`.
- WL citations: `2019 WL 1085179`, `2021 WL 3081160`, `2023 WL 3568691`,
  `2011 WL 1833007`, `2021 WL 7540814`, and others in the corpus.
- LEXIS citations: `2021 U.S. Dist. LEXIS 147975` and
  `2018 U.S. Dist. LEXIS 176841`.
- Slip-opinion citation: `2019 NY Slip Op 50388`.

Questions:

- Is failure caused by corpus absence, unsupported reporter families, malformed
  extraction, or missing citation enrichment?
- Which routes recover the same locator, and which recover only parallel or
  later-assigned locators?
- Can a locator be corroborated by independent citing opinions while the target
  opinion remains absent?

### E. Missing or partial case names

Fixtures such as `bookmarked/cite-0002`, `bookmarked/cite-0018`,
`1/cite-0025`, `1/cite-0051`, `5/cite-0054`, and `10/cite-0004` did not run the
case-name probe because extraction lacked both parties.

Questions:

- Can locator text, court/year, local context, docket numbers, or cited-case
  context produce a grounded search without inventing parties?
- Should a route request case-name re-extraction before external retrieval?
- What must be shown to the user when the retrieval input itself is incomplete?

### F. Extraction-defect controls

Some extracted names are visibly malformed, including the table fragment in
`6/cite-0013`, generic `Corp. v. Son Fish Sauce USA Corp.` in `10/cite-0005`,
and party strings containing `No.` plus a docket number.

These are controls for separating retrieval failure from extraction failure.
Experiments must retain the original extraction, any grounded re-extraction,
and the exact query generated from each. Retrieval must not silently rewrite the
citation.

### G. Throttling controls

Documents 8–10 contain successful lookups, partial case-name probes, fully
failed case-name probes, and citation lookups with HTTP 429. Examples include
`8/cite-0003`, `9/cite-0003`, and `10/cite-0001`.

Do not use these records to compare source recall. They test transport-state
representation, retry policy, resumability, and whether incomplete experiments
remain distinguishable from zero-result searches. Re-run them only after the
access window resets, preserving the original trace.

## Request observations — batch 1 (2026-07-03)

These are observations, not route preferences. Requests used CourtListener's
v4 search endpoint directly and retained corpus, query, count, stable IDs, and
candidate metadata. All eight requests in this batch returned HTTP 200; no 429
was encountered. Markup in returned captions is omitted below for readability.

| Fixture | Corpus and query | Observed result | Immediate research value |
| --- | --- | --- | --- |
| `8/cite-0001` | `r`, `caseName:(Oconner AND Agilant)` | One result: docket `7574496`, `OConner v. Agilant Solutions, Inc.`, `1:18-cv-06937`, `nysd`, PACER case `498465` | A low-count positive control with a compact candidate set. |
| `9/cite-0001` | `o`, `caseName:(Breest AND Haggis)` | One result: cluster `4690547`, docket `16628038`, `Breest v. Haggis`, `nyappdiv`, `2019 NY Slip Op 9398` | The extracted locator is `180 A.D.3d 83`; test whether a parallel reporter locator can be recovered after identity resolution. |
| `2/cite-0023` | `o`, `caseName:(EEOC AND Maricopa)` | Four historical results. Clusters `8497513` and `8497514` appear duplicate-like; cluster `8628489` is another Ninth Circuit matter; cluster `437188` is the 1984 matter reported at `736 F.2d 510`. | None matches the cited 2021 District of Arizona case, so a nonzero count masks lexical false positives and possible duplicate clusters. |
| `2/cite-0023` | `r`, same query | One result: docket `4747255`, `EEOC v. Maricopa, County of`, `2:02-cv-01874`, `azd`, PACER case `23664` | The sole result is still not target docket `CV-20-01788`; count one is not identity confirmation. |
| `4/cite-0043` | `o`, `caseName:(Pizzo AND City)` | Two Third Circuit `Ditullio v. Pizzo / City of Philadelphia` results: clusters `560719` and `575937`. | Both conflict with the cited Arizona `Pizzo v. City of Chandler`, so party-token overlap alone creates high-confidence-looking noise. |
| `4/cite-0043` | `r`, same query | One result: docket `47377911`, `Pizzo v. The City of New York`, `1:09-cv-09389`, `nysd`, PACER case `354673`. | The opinion and RECAP corpora fail differently; court and fuller defendant identity are needed before recommending any candidate. |
| `3/cite-0020` | `o`, `caseName:(Doe AND Amazon)` | One unrelated 2025 New Hampshire result: cluster `10698429`, docket `71597521`, `2025 DNH 041`. | A single opinion hit is not evidence that the cited 2023 Western District of Washington opinion has been found. |
| `3/cite-0020` | `r`, same query | Nineteen results. First is docket `67420269`, `Doe v. Amazon Incorporated`, `2:23-cv-00910`, `azd`; the set also contains many Amazon-as-plaintiff matters. | The cited input is `2023 WL 3568691`, `No. 22-cv-1231`, `wawd`. Court, year, party orientation, and docket comparison must govern candidate evaluation rather than count or top rank. |

Batch-level observations:

- HTTP 200 plus a numeric count describes a completed search, not a retrieved
  identity match.
- Corpus searches can return disjoint false-positive families for the same
  generated query.
- A unique result can still be wrong; a larger result set can still contain the
  target below the inspected window.
- Opinion-cluster identity and locator recovery are distinct operations. The
  Breest result is suitable for testing later-assigned parallel locators.
- Duplicate-looking clusters must remain separate until their document,
  sub-opinion, and version relationships are inspected.

Continuation point: begin with `3/cite-0029` in RECAP using
`caseName:(Doe AND Penzato)`, then `3/cite-0038` in RECAP using
`caseName:(Doe AND Northrop)`. Stop on the first HTTP 429 and record the request
that received it without interpreting it as a zero-result search.

## Three evaluation scorecards

### 1. Case recommendation

For every fixture, record:

- retrieved candidate identities and stable source IDs;
- court, docket number, decision date, caption, procedural level, and document
  type;
- whether the target appears in the inspected result window;
- duplicate, version, appeal, and related-proceeding relationships;
- evidence supporting or contradicting each candidate;
- whether recommendation is unique, multiple, or abstained.

Measure correct-candidate recall, false recommendation rate, correct-candidate
rank, abstention quality, and duplicate-collapse errors. Do not treat source rank
or BM25 as recommendation confidence.

### 2. Locator recovery

For each candidate identity, record every locator separately:

- original extracted locator;
- reporter and parallel reporter citations;
- slip or neutral citation;
- WL and LEXIS identifiers;
- docket and PACER document identifiers;
- the source and exact surrounding text that supplied each locator;
- corroborating observations after deduplicating repeated source records.

Measure exact-locator recovery, valid-parallel-locator recovery, false linkage,
and provenance completeness. A citation echo must remain bound to its cited name
and court/year parenthetical; an official slip opinion may establish identity
without containing its later reporter locator.

### 3. User-choice packet

Evaluate whether the evidence can be rendered as a neutral choice set containing:

- candidate caption and normalized short name;
- court, date, docket number, and procedural relationship;
- all recovered locators with source attribution;
- opinion/document availability and direct source links;
- why the candidate was retrieved;
- conflicts, missing fields, and unresolved relationships;
- alternative candidates and explicit search limitations.

Measure whether a reviewer can distinguish candidates without hidden source
knowledge, whether contradictory evidence is visible, and whether unavailable
or throttled routes are represented rather than omitted.

## Experiment record

Each route execution should append a record shaped conceptually as:

```text
fixture_id
route_id and source class
input evidence used
exact request/query
started/finished timestamps
transport status and cost
raw result reference
candidates and stable identifiers
locators with local context
availability facts
cross-source relations asserted and their evidence
errors, truncation, pagination, and unattempted follow-ups
```

Derived recommendation, locator-recovery, and user-choice evaluations must
reference these immutable observations rather than replacing them.

## Immediate research sequence

This sequence organizes measurement; it does not assert source preference:

1. Deduplicate the fixtures while retaining every document occurrence.
2. Establish a manually reviewed identity and locator answer key for a small,
   balanced subset from cohorts A–F.
3. Run each applicable route independently against that subset.
4. Evaluate each route on the three scorecards.
5. Test combinations only after individual route behavior is understood.
6. Expand to the remaining fixtures and re-run throttled controls separately.
7. Select route eligibility and ordering only from the accumulated results.
8. Remove deprecated, redundant, or losing route designs from the development
   documentation rather than preserving an obsolete menu of alternatives.

No backend implementation should be inferred from this document before step 7.
