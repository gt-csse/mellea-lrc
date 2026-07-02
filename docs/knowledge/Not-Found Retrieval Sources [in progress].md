---
tags: [retrieval, case-law, courtlistener, govinfo, pacer, search]
status: research
created: 2026-07-02
---

# Not-Found Retrieval Sources

This document inventories retrieval routes for a complete case citation that
CourtListener's exact citation lookup does not resolve. The goal is candidate
recall with traceable evidence, not a truth verdict.

Pure open-web search is the final fallback. We prefer a source with a legal
corpus, structured metadata, stable identifiers, a known jurisdiction, or
official provenance whenever one is available.

## Source hierarchy

Use the narrowest reliable source that could answer the question:

1. refine the citation and search CourtListener's appropriate corpus;
2. query another structured case-law or government collection;
3. query the issuing court's official publication surface;
4. use a legal vertical index to discover a source copy;
5. use commercial legal research only through an authorized integration;
6. use pure open-web search only after recording why the preceding routes are
   unavailable, out of scope, or exhausted.

This is a routing hierarchy, not a requirement to call every source. A known
state court should route directly to that court's archive after structured
case-law searches; a federal docket number should route to RECAP/PACER rather
than an unrelated state archive.

## 1. Exhaust CourtListener intelligently

An exact-locator 404 only closes `volume + reporter + page` lookup. It does not
close CourtListener search.

Search the case-law corpus with progressively weaker, recorded probes:

- the quoted and normalized locator, including reporter aliases;
- the complete case name as a `caseName` phrase;
- both party names, then stable distinctive party tokens;
- case name constrained by `court_id` and a bounded `dateFiled` range;
- docket number or neutral citation when the source document supplies one;
- one party plus locator, year, judge, or a distinctive quoted phrase;
- returned parallel citations, cluster IDs, docket IDs, and cited-by edges.

Run keyword search first. CourtListener supports Boolean, phrase, fielded,
wildcard, fuzzy/proximity, and range syntax across its corpora. Semantic search
is useful only for issue or passage recovery; it is not an identity resolver.
See the official [advanced search operators](https://www.courtlistener.com/help/search-operators/)
and [Citegeist behavior](https://www.courtlistener.com/help/citegeist/).

Route by corpus:

- `o`: published and explicitly requested unpublished case-law opinions;
- `r`/`d`: federal RECAP cases and docket metadata;
- `rd`: federal filing documents whose text or description may expose the
  authority, docket number, or parallel citation;
- `oa`: oral-argument metadata only when it provides a useful identity bridge.

Every search hit is a candidate. BM25 rank, semantic similarity, and a matching
case name do not establish identity.

## 2. Structured alternatives

### Caselaw Access Project

Harvard's Caselaw Access Project (CAP) provides structured metadata and full
text for more than 6.5 million published state and federal decisions across
U.S. history, with API and bulk access. Its data is already incorporated into
CourtListener, so CAP is not normally an independent coverage expansion.
Direct CAP access is still useful for:

- querying reporter/citation metadata in its original shape;
- recovering parallel citations or OCR text hidden by normalization;
- diagnosing CourtListener ingestion/indexing gaps or temporary outages; and
- retrieving a stable historical source record.

CAP should not be counted as independent corroboration when the CourtListener
record derives from CAP. See Harvard's [CAP project description](https://lil.law.harvard.edu/our-work/caselaw-access-project/)
and [open-data transition](https://lil.law.harvard.edu/blog/2024/03/26/transitions-for-the-caselaw-access-project/).

### GovInfo United States Courts Opinions

GovInfo's `USCOURTS` collection contains authenticated opinions from selected
federal appellate, district, bankruptcy, and national courts, generally from
2004 onward. It supports fielded search, predictable package identifiers,
machine-readable metadata, PDFs, and an API search service. This is a preferred
source for a federal opinion absent from CourtListener because GPO receives the
files from the Administrative Office of the U.S. Courts and preserves chain of
custody.

Search by court code, case number, party/title terms, date, and quoted locator.
Use the [USCOURTS collection guide](https://www.govinfo.gov/help/uscourts) and
[GovInfo developer/API resources](https://www.govinfo.gov/developers).

### RECAP, PACER, and PACER Case Locator

For federal matters, first search RECAP case and filing corpora. If the docket
or document is absent, PACER is the authoritative next route:

- PACER Case Locator searches a nationwide index by case type, number, title,
  party, and court type;
- the court-specific CM/ECF system can expose a docket or document not mirrored
  into RECAP; and
- a docket entry, opinion attachment, or related filing may reveal the neutral,
  slip, parallel, or corrected citation.

PACER requires an account and may incur fees, so it must be an explicit,
budgeted adapter rather than an implicit model tool. See the official
[PACER service](https://pacer.uscourts.gov/) and
[CM/ECF court lookup](https://pacer.uscourts.gov/file-case/court-cmecf-lookup).

## 3. Issuing-court sources

When a court is known or can be resolved, prefer its official site over a broad
search engine. CourtListener's `Court` record can provide the court identity and
homepage from which an adapter can discover:

- opinion and order search forms;
- slip-opinion, advance-sheet, and recent-decision pages;
- court-specific APIs, RSS/Atom feeds, sitemaps, and predictable PDF paths;
- docket calendars or case-information portals; and
- historical opinion archives maintained by the court, state judiciary, or
  official law library.

Adapters should be court- or platform-specific. They must record the official
domain, query fields, result URL, publication date, and retrieved file hash.
HTML scraping is acceptable only when the official source offers no structured
interface and its terms and robots policy permit it.

Jurisdiction routing matters:

- state trial: state judiciary case portal, county/court clerk, then official
  opinion/order archive if the court publishes decisions;
- state appellate/supreme: statewide opinion search, slip opinions, and official
  reporter/advance-sheet archive;
- federal trial/bankruptcy: RECAP, GovInfo, PACER/PCL, then the court site;
- federal appellate/special: GovInfo and the court's opinion archive before
  PACER document retrieval;
- U.S. Supreme Court: the Court's official opinions and orders surfaces;
- tribal, territorial, military, administrative, or special bodies: the body's
  own publication repository or the responsible government's official archive.

## 4. Legal vertical discovery

A legal vertical search is preferable to pure web search because it constrains
the corpus and often exposes jurisdiction, date, citation, cited-by, and version
relationships.

Google Scholar's Case law mode covers published U.S. state appellate and supreme
court opinions since 1950, several federal court classes since 1923, and U.S.
Supreme Court opinions since 1791. It supports court/date restriction, cited-by,
related-case, and version discovery. It does not provide bulk API access and
explicitly asks automated users to respect `robots.txt`, so use it as a manual
or compliant interactive adapter—not a scraper. See the official
[Google Scholar search and coverage help](https://scholar.google.com/intl/us/scholar/help.html).

Authorized Westlaw, Lexis, Bloomberg Law, Fastcase, vLex, or institutional
library integrations can also be high-value legal verticals. They are optional
capabilities: the agent must never assume credentials, automate a human-only
license, or reproduce proprietary editorial material.

## 5. Pure open-web search: last resort

Open-web search is enabled only when the exploration trace shows that applicable
structured, official, and legal-vertical routes were attempted or could not be
used. “CourtListener returned 404” is not sufficient justification.

Start constrained and relax deliberately:

1. exact quoted locator plus quoted case name;
2. exact locator plus court or one distinctive party;
3. quoted case name plus court and year/date range;
4. docket number plus party or court;
5. `site:` search over the resolved official court/judiciary domain;
6. known document tokens such as judge, neutral citation, or distinctive phrase;
7. reporter aliases, parallel-citation clues, and evidence-supported field
   corrections;
8. unconstrained web search only after the constrained queries fail.

Prefer results in this order: official court/government copy, institutional law
library or reporter scan, stable nonprofit legal archive, then secondary legal
publisher. Blogs, briefs, news, and generated summaries may provide discovery
clues but are not candidate authority records by themselves.

## Search-plan requirements

Before executing probes, produce a route plan containing:

- competing explanations for the exact-lookup miss;
- normalized and alternate locator forms;
- resolved court and court-level evidence;
- applicable sources in priority order and why each applies;
- query variants and expected discriminators;
- credential, fee, robots, rate-limit, and access constraints; and
- stopping and escalation conditions.

Each probe records its source class (`courtlistener`, `structured_archive`,
`official_court`, `legal_vertical`, `open_web`), exact query, filters, timestamp,
result count, candidates retained/rejected, and next inference. The trace must
make it possible to audit why open-web search was necessary.

## Remaining research

We still need a maintained registry of official state, territorial, tribal,
military, and special-court search surfaces and the platforms they share. That
registry should be built from resolved CourtListener court records and verified
official domains, then tested against a stratified set of known coverage gaps.
