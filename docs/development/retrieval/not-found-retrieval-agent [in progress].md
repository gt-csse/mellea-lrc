# Not-Found Retrieval Agent [in progress]

## Goal and boundary

Every complete volume/reporter/page locator first receives exact CourtListener
citation lookup. It is cheap and deterministic and is never bypassed based on
expected coverage.

A 404 ends exact locator retrieval but does not establish that the authority
is false. The reporter may be unsupported, a field may be wrong, the decision
may be too recent for permanent citation ingestion, the court may be poorly
covered, the case may exist under a parallel locator, or CourtListener may have
captured its docket without creating the corresponding opinion cluster.

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

- [Court Level Classification](../../knowledge/Court%20Level%20Classification%20%5Bin%20progress%5D.md):
  Court API resolution, raw jurisdiction categories, human-readable fallback
  signals, coverage priors, and timeliness.
- [CourtListener Search API](../../knowledge/CourtListener%20Search%20API.md):
  separate corpora, query behavior, and the shared docket hierarchy.
- [Data Source](../../knowledge/Data%20Source.md): CAP, PACER/RECAP, direct court
  scraping, and availability gaps.
- [Reporter-to-Court Inference](../../knowledge/Reporter%20Court%20Inference.md):
  exclusive reporter mappings and deliberately ambiguous reporters.
- [Reporter Jurisdiction Inference](./reporter-jurisdiction-inference%20%5Bin%20progress%5D.md):
  formal non-opinionated reporter constraints, of which exact-court inference
  is one special case.
- [Not-Found Candidate Search](./not-found-candidate-search.md): current
  one-shot, count-only baseline retained while this design is investigated.
- [Not-Found Retrieval Sources](../../knowledge/Not-Found%20Retrieval%20Sources%20%5Bin%20progress%5D.md):
  structured, official, legal-vertical, and open-web source inventory.
- [Retrieval Route Research Corpus](./retrieval-route-research-corpus%20%5Bin%20progress%5D.md):
  snapshot fixtures, neutral route experiments, and separate scorecards for
  case recommendation, locator recovery, and user-choice evidence.

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

## Exploration policy

This design has not selected a preferred recovery route. Investigation must
measure coverage, precision, availability, cost, and latency before source
ordering or automatic escalation policy is fixed. Current route families are:

```text
CourtListener refinement
RECAP docket/document recovery
CourtListener citation-echo recovery
GovInfo USCOURTS recovery
issuing-court official source
legal vertical or licensed source
constrained/open-web discovery
```

These are investigation candidates, not a ranking. Every experiment records
which routes were applicable, attempted, unavailable, or deliberately omitted.
Pure web search remains separately identifiable so evaluation can determine
whether structured routes make it unnecessary and where it still adds recall.

### CourtListener refinement

Search `type=o` using locator text, reporter aliases, fielded case-name phrases,
party tokens, court/year constraints, docket number, neutral citation, and
distinctive local phrases. Retain actual candidates and inspect court, date,
docket, parallel citations, ranking metadata, and cited-by relationships. Case
names are non-unique and BM25 or semantic rank does not establish identity.

Treat `type=o` results as clusters with nested individual opinions. Keep the
cluster ID, docket ID, sibling/opinion IDs, case name, citations, source, status,
and nested opinion types distinct. Do not describe the count as a count of
individual opinions.

For plausible federal proceedings, search RECAP cases, dockets, and filing
documents. State opinions should not be expected in RECAP merely because their
case-law records have CourtListener docket IDs.

### Cross-corpus gap handling

CourtListener case law is continuously updated and can ingest new court-source
clusters within hours. Age is therefore a useful hypothesis, not a sufficient
explanation for a miss. A years-old published opinion that fails independent
citation, case-name, docket-number, docket-ID, court/date, and distinctive-text
probes should be represented as a probable coverage or linkage gap—not “still
pending ingestion” and not a false citation.

When RECAP finds a plausible federal case but `type=o` finds no cluster:

- retain each RECAP docket candidate and its exact case name;
- inspect `type=rd` by docket ID for opinion/order/judgment records;
- record document availability, entry date, PACER document ID, description,
  and stable RECAP identifiers;
- probe `type=o` independently with discovered stable keys;
- expose GovInfo, issuing-court, citation-echo, legal-vertical, and web routes as
  independent experiments rather than assuming which should run next.

Never synthesize a `cluster_id` from a RECAP document or docket. A RECAP event
described as an opinion is evidence for retrieval planning, not evidence that a
CourtListener `OpinionCluster` exists. Preserve the explicit relation state:
`linked`, `not_linked`, `link_unavailable`, or `not_checked`.

### Citation-echo recovery

An absent target cluster can still be cited in the text of clusters that are
present. Search CourtListener opinion text using grounded case-name variants,
then parse citations near the matching name. This is different from searching
the target's `caseName` metadata.

The *Peterson* investigation demonstrated the route:

```text
caseName:(Peterson AND Nelnet)       -> 0 target clusters
"Peterson v. Nelnet" in opinion text -> later citing clusters
                                       -> 15 F.4th 1033
```

Five returned clusters reproduced the reporter locator. After deduplicating a
duplicate Third Circuit representation, the evidence included decisions from
the Second, Third, and Ninth Circuits. This establishes feasibility for one
case, not a preferred default or a measured general success rate.

Citation-echo candidates require contextual identity checks. A parsed locator
must remain attached to the surrounding cited case name and any court/year
parenthetical. Multiple citing clusters are corroboration observations, not
automatic proof, and duplicate clusters or repeated versions must not be counted
as independent evidence.

### GovInfo USCOURTS recovery

GovInfo's USCOURTS collection is a structured government corpus maintained by
GPO and AOUSC. It supports court, case-number, party, and full-text search, and
offers a documented API, feeds, sitemaps, stable package identifiers, metadata,
and PDFs. The *Peterson* opinion is package `USCOURTS-ca10-19-01348`, despite
being absent from CourtListener case law.

GovInfo can establish official opinion identity and supply document content. A
slip opinion may still omit the reporter citation assigned later, so GovInfo and
citation-echo recovery solve different parts of the problem. Investigation must
measure GovInfo coverage by court, level, and year before assigning it a routing
position.

### Other structured and official sources

Candidate sources include CAP for original historical metadata/parallel
citations, PACER/PCL for federal docket recovery, and the resolved issuing
court's official opinion or case-information system. GovInfo is described
separately above.
For each source, record whether access is an API, feed, sitemap, predictable
identifier, interactive search, or page retrieval. This is evaluation metadata;
the design does not yet choose among them.

### Legal vertical and web candidates

Candidate routes include a compliant legal vertical such as Google Scholar Case
law, an explicitly configured licensed research service, constrained web probes
(`site:` official domain, exact locator/name, court, year, docket, distinctive
phrase), and unconstrained search. Keep their observations separate so the
investigation can measure incremental recall and avoid silently treating search
rank as authority.

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

- `State Trial`: CourtListener recall is expected to be lower, not zero.
- `State Appellate`/`State Supreme`: RECAP is generally inapplicable.
- `Federal District`: both case-law and RECAP records may exist.
- `Federal Appellate`: case-law, RECAP, GovInfo, and court-source records may
  each exist independently.
- Special, bankruptcy, military, tribal, territorial, malformed, or missing
  categories: preserve the raw value and let the agent research/select a route.

## Iterative deliberation

This is not a one-shot classifier. A high-capacity agent should:

1. state competing explanations for the 404;
2. produce an experiment plan from applicable routes without a preset source
   preference;
3. inspect returned candidates and local context;
4. update hypotheses and launch evidence-supported follow-up probes;
5. record why each follow-up route was selected without treating the current
   experimental route list as a settled preference;
6. stop when a grounded candidate set exists, paths are exhausted, or further
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

The artifact preserves hypotheses, route plan, source class, exact queries and
filters, sources/corpora, timestamps, stable candidate identifiers, supporting
evidence, rejected candidates and reasons, unresolved conflicts, escalation
justifications, stopping reason, and unused plausible paths.

`not_found_after_exploration` means only that bounded retrieval found nothing.
It must not become `False` or hallucinated without separate assessment.
Retrieval also never silently mutates the original eyecite citation.

## Assessment handoff

Assessment receives the candidate set, original citation, local context, and
complete exploration trace. It decides whether a candidate corresponds to the
cited authority and may propose field corrections.

## Delivery plan

1. Define a common experiment trace for route applicability, requests,
   responses, candidates, costs, latency, and failures.
2. Build read-only research probes for CourtListener metadata search,
   CourtListener citation echo, RECAP case/document search, GovInfo, issuing
   courts, legal verticals, and web discovery.
3. Evaluate stratified 404s: corrupted locators, parallel reporters, recent
   cases, federal/state and trial/appellate courts, special courts, genuine
   hallucinations, and temporary search failures.
4. Measure route-level and combined recall, precision, correct-candidate rank,
   deduplication behavior, availability, cost, and latency.
5. Decide route eligibility and ordering only from the investigation results;
   remove losing or redundant route designs rather than retaining deprecated
   alternatives in the implementation plan.
6. After that decision, specify candidate-bearing artifacts, cross-corpus
   relation states, budgets, stopping states, and the non-opinionated assessment
   handoff.
7. Implement only the selected deterministic tools and orchestration.

Measure candidate recall, correct-candidate rank, queries and latency per task,
coverage by jurisdiction category, abstention quality, and false-link rate.
