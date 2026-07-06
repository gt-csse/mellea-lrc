---
tags: [courtlistener, courts, classification, coverage, search, cl_jurisdiction]
status: active
created: 2026-07-02
---

# CourtListener Jurisdiction Classification

## Terminology and classification target

The useful term for routing citations and queries is **`cl_jurisdiction`**. We must classify the court that issued the cited decision or owns the retrieved docket—not an entire real-world dispute. One dispute can produce separate trial and appellate dockets, and an appellate court can hear an original proceeding without becoming a trial court.

We faithfully represent the CourtListener (`cl`) way. CourtListener already models federal bankruptcy courts and panels, federal and state special courts, military, tribal, territorial, international, attorney-general, and committee records precisely. We retain CourtListener's exact fields instead of coercing every record into arbitrary human-readable labels (like "federal appellate" or "state trial").

> **Data source note.** The `system` / `type` / `jurisdiction` triple used by `cl_jurisdiction` is not a CourtListener API field. It comes from the Free Law Project [`courts-db`](https://github.com/freelawproject/courts-db) package (the same data eyecite uses for parenthetical lookup). The courts-db slugs happen to coincide with CourtListener court IDs, so the snapshot works for our routing purposes, but the authoritative source for the classification is `courts-db`. The `cl_jurisdiction` term itself is a project-local name for the `(system, type, jurisdiction)` triple.

## Recommended model

Use CourtListener's `Court` record as the knowledge model whenever a court slug or name resolves. The API gives us the canonical slug, full and short names, and the exact `cl_jurisdiction` properties (`system`, `type`, `jurisdiction`). Preserve those upstream values instead of building a parallel court ontology prematurely.

```text
court slug/name
    -> CourtListener /courts lookup
    -> canonical court identity + cl_jurisdiction properties
    -> retrieval prior for post-not-found search
```

The `cl_jurisdiction` is a routing prior, not a claim that citation lookup or either CourtListener corpus supports the court:

```text
court recognized
!= citation lookup supported
!= case-law available
!= docket available
!= docket entries available
!= document available
```

Raw CourtListener identity and `cl_jurisdiction` properties remain the knowledge. Special, bankruptcy, military, tribal, territorial, and malformed values remain explicit. 

If neither exact slug nor exact/unique-prefix name lookup resolves a Court record, delegate court identification to general court reasoning. That fallback must preserve uncertainty and must not manufacture a CourtListener slug.

## CourtListener's `cl_jurisdiction`

CourtListener's `Court` record is the primary structured source. It provides three core properties that together form the `cl_jurisdiction`:

1. **`system`**: The overarching authority (e.g., `federal`, `state`, `special`, `tribal`, `military`, `territorial`, `international`).
2. **`type`**: The institutional role of the court (e.g., `appellate`, `trial`).
3. **`jurisdiction`**: The state or territory code where applicable (e.g., `C.A.`, `N.Y.`). For federal courts, this is usually null.

This faithful representation avoids assumptions about "rank" (e.g. New York's Supreme Court is a trial court, but `system=state`, `type=trial`, `jurisdiction=N.Y.` makes this unambiguous).

## Human-readable recognition patterns

Patterns should produce evidence with a confidence level, not silently become truth. Prefer explicit court identity over reporter-family inference.

### Reporter patterns

- `U.S.`, `S. Ct.`, and `L. Ed.` identify SCOTUS and therefore `system=federal`, `type=appellate`.
- `F.2d`, `F.3d`, and `F.4th` indicate federal appellate publication, but the parenthetical or retrieved court is still required to identify the circuit.
- `F. Supp.`, `F. Supp. 2d`, and `F. Supp. 3d` indicate federal district-court publication and therefore normally `system=federal`, `type=trial`.
- Official state and regional reporters are not sufficient by themselves to distinguish a state supreme court from an intermediate appellate court, and some historically include other courts.
- `B.R.` and `M.J.` span multiple types and must not determine `cl_jurisdiction` alone.

Reporter-family inference is broader and weaker than the exclusive reporter-to-court inference documented in [Reporter-to-Court Inference](Reporter%20Court%20Inference.md).

### Docket and document patterns

Federal document headers and PACER identifiers are useful corroboration:

- `…-cv-…` and `…-cr-…` usually identify federal district civil and criminal dockets.
- `…-bk-…` identifies bankruptcy, not a general federal trial case.
- Federal appellate numbers commonly resemble `YY-NNNN`, but this shape is too generic to use without a known appellate court.
- A header naming the court is stronger than the docket-number shape.

### Evidence precedence

1. CourtListener court record resolved directly from a slug.
2. CourtListener court record resolved uniquely from an exact normalized name.
3. Explicit court extracted from a parenthetical or document header, then resolved through the Court API.
4. Exclusive reporter-to-court mapping, followed by Court API resolution.
5. Reporter-family evidence plus a compatible parenthetical.
6. Docket-number or case-caption heuristics as corroboration only.
7. Otherwise delegate to general court reasoning or return `unknown`.

Every inference should preserve `source`, `confidence`, and the raw evidence so later corrections do not erase provenance.

## CourtListener availability by `cl_jurisdiction`

CourtListener has two principal, overlapping data legs relevant here. They share the `Docket` model and identifier namespace but have different children, sources, timeliness, and missingness:

```text
Case law:       Docket -> OpinionCluster -> Opinion
PACER / RECAP:  Docket -> DocketEntry -> RECAPDocument
```

| Derived `cl_jurisdiction` context | Case-law leg | PACER/RECAP leg | Practical search strategy |
|---|---|---|---|
| `system=federal`, `type=appellate` | Strong for published/collected opinions. | Federal appellate dockets and filings may appear, but coverage is contribution-dependent. | Search citation/case law first for cited decisions; use RECAP separately for docket documents. |
| `system=federal`, `type=trial` | Selected reported opinions and orders. Not a complete docket corpus. | Strongest docket metadata leg. CourtListener regularly gathers basic metadata for new district cases. | Search RECAP for cases/filings; search case law for published or collected opinions. |
| `system=state` | Strongest state category, especially published historical case law. | PACER/RECAP does not cover state courts. | Search case law only; do not interpret absent entries as an absent case. |
| `system=special`/`tribal`/`territorial` | Varies by tribunal and source. | Varies widely. | Route by specialization instead of general assumptions. |

CourtListener itself instructs users to search case law and federal filings in separate databases. Cross-corpus fallback is therefore a deliberate strategy, not pagination over one universal index.

## Timeliness and lag

- CAP is a historical book-digitization source, not a live feed. CourtListener has incorporated and normalized it, then supplements it with other sources.
- For new federal district and bankruptcy matters, CourtListener reports that it regularly scrapes free basic PACER metadata shortly after filing.
- RECAP docket entries and PDFs remain non-random. They arrive through browser users, email contributions, special scrapers, RSS feeds, bulk projects, and fetch APIs. A current docket may exist while a desired filing does not.
- PACER opinions or orders marked by clerks are downloaded nightly, but not every dispositive document is necessarily marked correctly.

Availability should be modeled as an observation with `checked_at`, corpus, query, and result—not as a permanent property of the case.

## Development implication

Court classification remains a small deterministic enrichment step:

1. Resolve a known slug directly with `/courts/{slug}/`.
2. Resolve a name with exact case-insensitive matching; accept a prefix query only when it returns one credible court.
3. Preserve canonical name, slug, raw `cl_jurisdiction` category, lookup method, and lookup time.
4. Use `cl_jurisdiction` only as a prior for selecting likely retrieval paths.
5. Delegate unresolved names and conflicting human-readable evidence to general court reasoning.

The workflow consuming this knowledge after citation lookup fails is specified in [Not-Found Retrieval Agent](../development/retrieval/not-found-retrieval-agent%20%5Bin%20progress%5D.md).

## Future Roadmap: Taxonomy Utilization

Looking ahead, the `courts_db_classification` will unlock three major downstream search and reasoning capabilities:

### 1. Coverage Confidence and Search Routing
The taxonomy will inform whether a correctly parsed citation/locator is expected to be indexed by CourtListener and via which specific endpoint (e.g., citation lookup / Cluster vs. RECAP / Docket). This determines our confidence: if a case isn't found where the taxonomy strongly suggests it should be, we can confidently state it does not exist in the dataset or gracefully delegate to a broader external web search.

### 2. Semantic Inference and Anomaly Detection
The structured taxonomy enables cross-jurisdictional conflict detection during semantic inference. For example, if a state-level trial court is found citing an unrelated state's appellate or trial court, the system can flag this as an anomaly or conflict. This will add a layer of legal logic validation beyond string matching.

### 3. Direct Reporter-to-Taxonomy Mapping
Currently, the pipeline attempts to resolve `reporter -> court -> courts_db_classification`. Because `eyecite` mappings often attempt to recover one exact court from a reporter, this path is brittle. Future implementations will establish a direct `reporter -> courts_db_classification` path (via rule-based mappings or LLM inference). Even when the exact court is ambiguous, providing the overarching system/jurisdiction/type from the reporter alone will significantly narrow the downstream search scope.

## Primary sources

- [CourtListener available jurisdictions](https://www.courtlistener.com/help/api/jurisdictions/)
- [CourtListener case-law API hierarchy](https://wiki.free.law/c/courtlistener/help/api/rest/v4/case-law)
- [CourtListener case-law coverage](https://www.courtlistener.com/help/coverage/opinions/)
- [CourtListener RECAP coverage and ingestion sources](https://www.courtlistener.com/help/coverage/recap/)
- [CourtListener guidance on separate search databases](https://wiki.free.law/c/courtlistener/help/search/i-cant-find-something-when-i-search-courtlistener-help)
- [Free Law Project comparison of CAP and CourtListener data](https://wiki.free.law/c/courtlistener/help/general/how-does-the-data-in-harvards-caselaw-access-project-compare-to-courtlisteners-case-law-database)
- [Free Law Project Courts Database](https://free.law/projects/courts-db)
