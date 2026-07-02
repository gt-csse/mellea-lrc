# Not-Found Retrieval Agent

## Goal and boundary

Every complete volume/reporter/page locator first receives exact CourtListener
citation lookup. It is cheap and deterministic and is never bypassed based on
expected coverage.

A 404 ends exact locator validation but does not establish that the authority
is false. The reporter may be unsupported, a field may be wrong, the decision
may be too recent for permanent citation ingestion, the court may be poorly
covered, or the case may exist under a parallel locator.

The proposed **not-found retrieval agent** performs bounded, iterative
exploration after that 404. It retrieves and synthesizes candidates; it does
not decide whether the citation is real or hallucinated. Candidate comparison
remains assessment's responsibility.

```text
complete locator -> exact citation lookup
                    |-> found/ambiguous: existing path
                    `-> not_found: retrieval-agent task
```

## Knowledge dependencies

The agent consumes the knowledge base instead of recreating court and coverage
assumptions in its prompt:

- [Court Level Classification](../knowledge/Court%20Level%20Classification.md):
  Court API resolution, raw jurisdiction categories, human-readable fallback
  signals, coverage priors, and timeliness.
- [CourtListener Search API](../knowledge/CourtListener%20Search%20API.md):
  separate corpora, query behavior, and the shared docket hierarchy.
- [Data Source](../knowledge/Data%20Source.md): CAP, PACER/RECAP, direct court
  scraping, and availability gaps.
- [Reporter-to-Court Inference](../knowledge/Reporter%20Court%20Inference.md):
  exclusive reporter mappings and deliberately ambiguous reporters.
- [Not-Found Candidate Search](./Not%20Found%20Candidate%20Search.md): current
  one-shot, count-only behavior that this design supersedes.

## Inputs

The task contains immutable structured evidence:

- citation ID, matched text, parties, locator fields, year, and extracted court;
- bounded local document context and source metadata;
- reporter-to-court inference, if present;
- Court API result: canonical slug, full/short name, raw jurisdiction category,
  resolution method, and timestamp;
- exact citation-lookup request and 404 response;
- searches already attempted, including the existing case-name count trace;
- exploration budget and enabled retrieval tools.

If CourtListener does not recognize the court, record that fact and permit
general court reasoning. Never fabricate a CourtListener slug.

## Exploration paths

The agent chooses, orders, and may revisit paths based on intermediate results.

### Locator-oriented search

Search the failed locator as text. This remains useful when citation lookup
does not support the reporter or court, a state-trial decision appears only on
external sites, or results expose a parallel citation. Probes may use quoted
and normalized locators, reporter aliases, and locator plus one party name.

### CourtListener case-name and corpus search

Search `type=o` using a fielded case-name phrase and retain actual candidates,
not only the count. Inspect court, date, docket, citations, and ranking metadata;
case names are non-unique and BM25 rank does not establish identity.

For plausible federal proceedings, the agent may search RECAP cases and
documents. State opinions should not be expected in RECAP merely because their
case-law records have CourtListener docket IDs.

### General and court-specific search

Search the wider web or an available court-specific source using combinations
of case name, locator, canonical court name, year, docket number, and distinctive
local context. This path has a higher prior for state-trial cases and weakly
covered courts, but remains available for every jurisdiction category.

### Evidence-driven reformulation

Intermediate results may justify searches for parallel or corrected citations,
abbreviated parties, alternate reporter spellings, corrected fields, a newly
discovered docket number, or a slip/neutral citation. Distinguish reformulation
supported by retrieved evidence from unconstrained guessing.

## Role of CourtListener jurisdiction

Court recognition simplifies planning. The Court API supplies canonical court
identity and jurisdiction category without requiring our own ontology. The
category changes expected search yield; it neither suppresses exact citation
lookup nor asserts content availability.

- `State Trial`: after 404, quickly consider general and state-specific search;
  CourtListener recall is low, not zero.
- `State Appellate`/`State Supreme`: prioritize CourtListener case law and do
  not expect RECAP entries.
- `Federal District`: consider RECAP case/document search alongside case law.
- `Federal Appellate`: prioritize case law, using RECAP separately for docket
  material.
- Special, bankruptcy, military, tribal, territorial, malformed, or missing
  categories: preserve the raw value and let the agent research/select a route.

## Iterative deliberation

This is not a one-shot classifier. A high-capacity agent should:

1. state competing explanations for the 404;
2. select high-information probes within budget;
3. inspect returned candidates and local context;
4. update hypotheses and launch evidence-supported follow-up probes;
5. stop when a grounded candidate set exists, paths are exhausted, or further
   searches have low expected value.

Independent probes may run in parallel; follow-up probes depend on their
results. The framework enforces tool, request, result-count, token, and elapsed
time budgets.

## Output contract

The agent returns retrieval state, not a truth verdict:

```text
candidate_found
multiple_candidates
not_found_after_exploration
insufficient_evidence
search_unavailable
```

The artifact preserves hypotheses, exact queries, sources/corpora, timestamps,
stable candidate identifiers, supporting evidence, rejected candidates and
reasons, unresolved conflicts, stopping reason, and unused plausible paths.

`not_found_after_exploration` means only that bounded retrieval found nothing.
It must not become `False` or hallucinated without separate assessment.
Retrieval also never silently mutates the original eyecite citation.

## Assessment handoff

Assessment receives the candidate set, original citation, local context, and
complete exploration trace. It decides whether a candidate corresponds to the
cited authority and may propose field corrections.

## Delivery plan

1. Replace the count-only trace with candidate-bearing artifacts while
   preserving backward-compatible status/error reporting.
2. Add Court API enrichment for known court slug/name and retain raw
   jurisdiction category.
3. Define the agent task, allowed tools, budgets, output schema, and stopping
   states before model prompts.
4. Implement CourtListener case-law and RECAP probes as deterministic tools.
5. Add general-search and court-specific adapters with source attribution.
6. Implement iterative orchestration and candidate deduplication without
   treating docket ID as universal identity.
7. Add the non-opinionated assessment handoff.
8. Evaluate stratified 404s: corrupted locators, parallel reporters, recent
   cases, federal/state and trial/appellate courts, special courts, genuine
   hallucinations, and temporary search failures.

Measure candidate recall, correct-candidate rank, queries and latency per task,
coverage by jurisdiction category, abstention quality, and false-link rate.
