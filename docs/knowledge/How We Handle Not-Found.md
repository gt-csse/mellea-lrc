---
tags: [retrieval, case-law, courtlistener, not-found, jurisdiction-inference, search]
status: active
created: 2026-07-06
---

# How We Handle "Not Found"

This is the canonical not-found document. It covers the flow we follow after a complete volume/reporter/page locator returns nothing from CourtListener's exact citation lookup, the sources we route through, and what the jurisdiction inference layer does.

A 404 ends exact locator retrieval. It does not mean the authority is false. The reporter may be unsupported, a field may be wrong, the decision may be too recent for permanent citation ingestion, the court may be poorly covered, the case may exist under a parallel locator, or CourtListener may have captured its docket without creating the opinion cluster. The not-found flow retrieves and synthesizes candidates. Deciding whether a candidate is the cited authority is assessment's job.

## The two primary approaches

After exact citation lookup fails, we have two families of search left:

1. **CourtListener search** — refined, court-aware probes against CourtListener's case-law (`type=o`), RECAP (`type=r`/`rd`/`d`), and dockets endpoints. This is the only family that gives us stable identifiers (`cluster_id`, `docket_id`), structured metadata, and direct evidence about CourtListener coverage gaps.
2. **Broader web search** — constrained open-web probes (`site:` official court domains, official judiciary archives, government publication surfaces), falling back to unconstrained open web only after the constrained routes are exhausted or inapplicable.

CourtListener search is always tried first when it is even plausibly applicable. Pure open-web search is a last resort and must always be traceable: the trace records why the structured routes were inapplicable, attempted but failed, or exhausted.

The move from "CourtListener search" to "broader web search" is the central design moment in the not-found flow, and most of the rest of this document exists to justify when that move should happen.

## A 404 on the exact locator does not mean "unavailable" for courtlistener

A 404 on `volume + reporter + page` is an answer about one specific lookup against one specific endpoint. It is not an answer about the case. The case may still exist in CourtListener in a form the locator cannot reach. So we keep searching CourtListener with progressively weaker, recorded probes.

Three historical/maintenance reasons this happens.

### Reason 1: the case is recent and the opinion has not been ingested yet

CourtListener's case-law corpus is continuously updated. New opinions from contributing courts can be ingested within hours of release, but not every opinion is ingested at the same speed, and the ingestion pipeline is not synchronized with opinion publication. A 404 on a locator from a recent decision may simply mean "not yet ingested." Age is a useful hypothesis, not a sufficient one.

### Reason 2: the opinion is captured by RECAP but not by the case-law corpus

This is the most important gap to know about, and it is the reason we keep trying CourtListener even when the locator is hopeless.

CourtListener has two distinct data legs, both rooted in a shared `Docket` record:

```text
Case law:    Docket -> OpinionCluster -> Opinion
RECAP:       Docket -> DocketEntry   -> RECAPDocument
```

These two legs are maintained by different pipelines, with different timeliness and different coverage.

The case-law leg is built by scraping court websites, direct court contributions, and historical digitization projects (including Harvard's Caselaw Access Project). It is authoritative for the *opinion cluster* and the opinion text.

The RECAP leg is built by user contributions through the RECAP browser extension, special scrapers, email contributions, RSS, and the PACER fetch API. It is authoritative for the *docket and its filing record*, and is the most complete metadata leg for federal district and bankruptcy matters.

Because the two legs are maintained by **different methodologies**, their coverage does not move together. A case may have a complete docket in RECAP — including a docket entry that explicitly records the opinion — and yet have no `OpinionCluster` on the case-law side. The opinion is "in" CourtListener as a docket event, but not as an opinion cluster that the locator can resolve.

**Concrete example: *Peterson v. Nelnet Diversified Solutions*, 15 F.4th 1033 (10th Cir. 2021).**

- The Tenth Circuit published the opinion on 2021-10-08.
- CourtListener contains two RECAP appellate dockets for the case, each described as `Case termination for opinion`, but the document is unavailable and the docket has no `cluster_id`.
- The CourtListener citation resolver for `15 F.4th 1033` returns 404.
- Independent keyword probes against `type=o` — the locator, both appellate docket numbers, both docket IDs, the case-name intersection, and a distinctive text phrase — all return zero.
- The official Tenth Circuit PDF is still available on the court's own site at
  [ca10.uscourts.gov](https://www.ca10.uscourts.gov/sites/ca10/files/opinions/010110588582.pdf).

This is not a recent-ingestion lag (the opinion is years old) and not a format quirk (eyecite recognizes the locator). The best explanation is that RECAP captured the opinion-related docket event but the case-law ingestion pipeline never created or indexed the `OpinionCluster`. The case is in CourtListener in the docket sense but not in the cluster sense.

This is exactly why the dockets endpoint stays a first-class target for not-found search even when the locator is the only thing we have.

### Reason 3: the docket is captured but the case is too old or too marginal to be in clusters

The reverse of the previous gap: a historical state appellate opinion may have a `Docket` record on the case-law side but no `DocketEntry` records on the RECAP side. CourtListener creates dockets as part of case-law ingestion, but the docket population process does not guarantee that entries are carried over. The case is "in" CourtListener as a cluster and as a docket, but the docket is not a useful search target for it.

The point: the case-law and RECAP legs are two different corpora. Presence in one does not imply presence in the other. Searching one without searching the other is incomplete; searching both is what "exhaust CourtListener" actually means.

## The locator problem in the dockets interface

Once we know the dockets endpoint might know about a case the case-law leg does not, the next problem is how to *find* that docket without a locator.

The dockets endpoint does not accept a volume/reporter/page locator. It is keyed by `docket_id` (a numeric CourtListener internal id) and accepts filters for `docketNumber`, `court`, `court_id`, `dateFiled` range, `caseName`, and party tokens. The locator is a property of the *opinion*, and the docket is the case — the case lives one level up from the opinion.

In the case-law hierarchy `Docket -> OpinionCluster -> Opinion`, the locator attaches to the `Opinion` and is mirrored on the `OpinionCluster` (as parallel citations). The `Docket` does not have a locator. So when we search `type=o` for a locator, we are searching the *right* level. When we get nothing, switching to the dockets endpoint means switching from a level that has a locator to a level that does not, and we have to formulate the query without the locator.

This is the central design tension in the not-found flow: the most direct CourtListener search is keyed on the thing the lookup just told us is useless, and the endpoint we should try next is keyed on things we do not yet have.

The practical workarounds, in increasing order of effort:

- use the extracted case name and the court as a docket filter, hoping a docket exists with the same name;
- use both party names and a bounded `dateFiled` range derived from the citation year;
- use the docket number when the source document supplies one;
- use distinctive party tokens or a known document token (judge, neutral citation, distinctive phrase).

None of these is a locator. All of them are noisier and more ambiguous than a locator search would have been. The dockets endpoint is necessary because the case-law leg has gaps; it is harder to use because it is not locator-indexed.

## Case-name search is necessary but unreliable

Because the dockets endpoint is not locator-indexed, the next most natural query is by case name. Case name is the most distinctive textual feature of a citation that does not require a locator, and most search interfaces (CourtListener included) accept it as a filter.

Case-name search is also a known reliability hazard. Two examples that bracket the range:

- **Narrow case names.** A case like *Marquette v. Township of Marquette* returns a small handful of clearly matching candidates. A case-name search constrained by court and year usually surfaces the right one.
- **Generic case names.** A case like *Smith v. Smith* or *In re Smith* returns so many candidates across decades and jurisdictions that no case-name search can be considered reliable evidence of identity, even with court and year constraints.

The defining feature is *distinctiveness*. The number of candidates returned for an unconstrained case-name search is itself evidence: when it is small, the case name is doing real work; when it is large, the case name is not telling us anything and we are just enumerating candidates.

The most important cue by far is **court**. If we have a court (a real CourtListener slug, not a guess), we can constrain the case-name search to that court and the candidate set drops dramatically. State trial dockets in California are common, but California Supreme Court dockets with a given case name are far rarer. Court is the single most powerful constraint we have for narrowing a case-name probe.

This is also where most not-found flows break in practice. Without a court, a case-name probe is little better than a party-name probe. The reliability of a case-name search is dominated by the quality of the court constraint, not by the case-name string itself.

## Why the jurisdiction inference layer exists

The not-found flow needs a court. We often do not have one.

Eyecite's `FullCaseCitation.court` field is populated by two completely separate mechanisms:

1. **Parenthetical lookup is the primary mechanism.** For everything except SCOTUS, eyecite uses `get_court_by_paren()` from the `courts-db` package. This is a pure dictionary lookup: the parenthetical string (e.g. `2d Cir.`, `D.D.C.`, `9th Cir.`) is matched against `courts-db.citation_string` and the matching `courts-db.id` becomes the court. If the citation has no parenthetical, or the parenthetical is not in `courts-db`, no court is set.

2. **SCOTUS reporter heuristic is the exception.** Only for SCOTUS does eyecite set the court from the reporter itself. In `Reporter.__post_init__()`, eyecite checks whether `cite_type == "federal"` and `"supreme" in self.name.lower()`, and if so marks the reporter as `is_scotus = True`. `CaseCitation.guess_court()` then sets `metadata.court = "scotus"`.

These are the only two paths. **There is no other direct reporter-to-court translation in eyecite.** For every non-SCOTUS citation, if the parenthetical is missing, malformed, or unrecognized, eyecite leaves `court` empty. The reporter itself does not tell eyecite anything about the court.

When eyecite leaves `court` empty, the not-found flow is missing its single most powerful constraint. A case-name search without a court constraint is unreliable. The jurisdiction inference layer was developed to fill that gap.

## What the jurisdiction inference layer actually does

The initial purpose was narrow: derive a court when eyecite fails to extract one because of parenthetical formatting, missing parenthetical, or an unrecognized parenthetical. It has since grown into a much more general layer that runs entirely on local data, with no network calls, no LLM, and no rate limits. The place of this layer in the pipeline is:

```text
Preprocessing -> Extraction -> Jurisdiction Inference -> Retrieval -> Assessment
```

The layer is the natural home for a set of inferences that share a property: they take structured input from the citation and return structured evidence that downstream retrieval and assessment can consume. It has five components.

### (1) Reporter-to-court translation (the "exhaustive singleton" projection)

When a reporter publishes decisions from exactly one court, the reporter itself is the court. Eyecite's SCOTUS heuristic does this for `U.S.` and `S. Ct.`, but not for any other court. The jurisdiction inference layer extends this idea to other exclusive reporters.

The current registry covers a small set of single-court reporters where publication scope is empirically one court: `L. Ed.`, `L. Ed. 2d`, `U.S. LEXIS`, `T.C.`, `B.T.A.`, `Fed. Cl.`, `Cl. Ct.`, `Ct. Int'l Trade`, `Cust. Ct.`, `C.C.P.A.`, `Vet. App.`, `M.S.P.R.`, `C.M.A.`. The full list and admission rules are in [Reporter-to-Court Inference](Reporter%20Court%20Inference.md).

The general mechanism is not a SCOTUS exception; it is a project-level contribution. The Free Law Project databases (`reporters-db`, `courts-db`) do not provide a general reporter-to-court-slug mapping, and the manually curated bridge in `mlz_to_cl_map.json` covers only 17 MLZ strings. A curated, publication-scope-verified `reporter -> court_slug` table for non-SCOTUS reporters is one of the things this project ships.

### (2) Court-level inference

For reporters that cover a bounded set of courts but more than one, we cannot pick one court, but we can pick the court level. CourtListener's `Court` records carry `system` and `type` properties (e.g. `system=federal, type=appellate`) that we can use as a coarse filter.

The level matters operationally because it tells us *which leg* of CourtListener to trust:

- `system=federal, type=appellate` — case-law leg is strongest for published/collected opinions; RECAP may have appellate dockets but coverage is contribution-dependent.
- `system=federal, type=trial` — RECAP leg is strongest for docket metadata; case-law leg has selected reported opinions but is not a complete docket corpus.
- `system=state` — case-law leg only; RECAP does not cover state courts.

The level-inference table is the project-side mirror of the CL `Court` record's level properties. The retrieval planner consumes it to pick which `type=` corpus to search and to set the prior for what we expect to find.

### (3) Reporter-jurisdiction inference (with MLZ translation)

For reporters that cover a wide set of courts, the next-best signal is the set of jurisdictions the reporter can possibly cover. `reporters-db` carries an `mlz_jurisdiction` field for each reporter, listing MLZ jurisdiction strings associated with that reporter historically.

MLZ is not a US-jurisdiction taxonomy. It is the Modern Legal encyclopedia taxonomy, and its jurisdiction strings are not directly mappable to CourtListener slugs. The 17-entry `mlz_to_cl_map.json` bridge is the minimum needed to translate the most common MLZ strings into CourtListener slugs. The rest of the translation is a gap this project contributes: a curated MLZ-to-CL map that is broad enough to support reporter-jurisdiction inference across the US legal system.

Once the MLZ-to-CL translation is in place, the reporter-jurisdiction inference tells us, for a given reporter, the *set* of courts the cited case could plausibly belong to. This is weaker than an exact court, but it is a strong availability cue:

- A reporter that covers a small set of federal appellate courts gives us a tight availability bound even without the parenthetical.
- A reporter that covers lots of courts but where all of them are federal appellate tells us the case should be in our dockets even if we do not know the specific court.
- A reporter that covers lots of state supreme courts across several states tells us we need a state filter and a year filter, not a court filter.

This is the reason the `ReporterJurisdictionInference` representation in the codebase is more general than the exact-court projection: the model records a court set, a court class, and a jurisdiction, and the downstream code intersects compatible evidence.

### (4) Reporter-court consistency check

Given (1), (2), and (3), we can check whether a citation's extracted court is consistent with its reporter. The check is straightforward:

- If the reporter is in the exhaustive-singleton set and the extracted court is not the singleton, the court is wrong or the citation is wrong.
- If the reporter covers a bounded set and the extracted court is outside the set, the same.
- If the reporter covers a court class and the extracted court is in a different class, the same.

A consistency failure is strong evidence that the citation is false or mis-parsed. It is not *proof* — the extracted court could itself be wrong — but it is a strong signal. A false-positive consistency check is a much more dangerous failure mode than a false-negative one, so the check should fire only on clear-cut cases, not on borderline ones.

This is not a retrieval aid directly, but it is an assessment aid: the not-found flow should not invest significant effort in retrieving a case that fails a basic internal consistency check. The check is cheap (local) and filters out the worst false cases before any I/O.

### (5) Foundation for future semantic analysis (third-layer assessment)

The jurisdiction inference layer also serves as a foundation for downstream semantic reasoning that the project plans but does not yet implement. The rough idea:

- The document under analysis is in some jurisdiction (the one that issued it).
- The citations in the document should mostly be in compatible jurisdictions.
- A citation that resolves to a state trial court in a different state from the document's own jurisdiction, for example, is suspicious. A state trial decision from California cited as authority in a New York state trial decision is, in most legal contexts, not a relevant authority. It is more likely a mis-parsed case name, a misread reporter, or a hallucinated citation.

This kind of cross-jurisdictional anomaly detection belongs to the third-layer assessment, not the not-found retrieval flow. But the structured representation in the jurisdiction inference layer — court set, court class, jurisdiction — is the substrate that the assessment layer will need.

## Putting the flow together

After exact citation lookup returns 404, the not-found flow proceeds roughly as follows:

1. **State competing explanations for the 404.** Is the case too recent, captured only in RECAP, covered by a different leg, the victim of a malformed parenthetical, or simply a false citation? Each explanation points to a different next step.
2. **Run jurisdiction inference on the citation.** The reporter, the extracted court, and the year together give us a court, a court class, a jurisdiction set, and a consistency check. The result is the planning substrate for the rest of the flow.
3. **Search CourtListener with a court constraint.** If we have a court, the case-name search becomes usable. Try `type=o` first, then `type=r`/`rd`/`d` for federal matters, then the dockets endpoint with the case name and the court.
4. **If CourtListener is silent, fall back to the dockets endpoint without a court constraint.** This is the legitimate response to the case-law/RECAP coverage gap above. The case-name search is unreliable here, but it is the only signal we have at this level.
5. **Fall back to broader web search** — official court sites, government publication surfaces (GovInfo for federal appellate), legal verticals, and ultimately unconstrained open web. Each step must be traceable: the trace records why the prior step was inapplicable, attempted but failed, or exhausted.

The jurisdiction inference layer is not a single call that resolves a not-found case. It is a set of local inferences that make every later step more reliable: a court lets us constrain a case-name search; a court class lets us pick a CourtListener leg; a jurisdiction set lets us pick a web source; a consistency check tells us when to stop.

The single most important thing the layer does, in practice, is **make case-name search usable by giving it a court constraint**. Without that constraint, case-name search is not retrieval; it is enumeration.

## Source hierarchy

Use the narrowest reliable source that could answer the question:

1. refine the citation and search CourtListener's appropriate corpus;
2. query another structured case-law or government collection;
3. query the issuing court's official publication surface;
4. use a legal vertical index to discover a source copy;
5. use commercial legal research only through an authorized integration;
6. use pure open-web search only after recording why the preceding routes are
   unavailable, out of scope, or exhausted.

This is a routing hierarchy, not a requirement to call every source. A known state court should route directly to that court's archive after structured case-law searches; a federal docket number should route to RECAP/PACER rather than an unrelated state archive.

### 1. Exhaust CourtListener intelligently

An exact-locator 404 only closes `volume + reporter + page` lookup. It does not close CourtListener search.

Search the case-law corpus with progressively weaker, recorded probes:

- the quoted and normalized locator, including reporter aliases;
- the complete case name as a `caseName` phrase;
- both party names, then stable distinctive party tokens;
- case name constrained by `court_id` and a bounded `dateFiled` range;
- docket number or neutral citation when the source document supplies one;
- one party plus locator, year, judge, or a distinctive quoted phrase;
- returned parallel citations, cluster IDs, docket IDs, and cited-by edges.

Run keyword search first. CourtListener supports Boolean, phrase, fielded, wildcard, fuzzy/proximity, and range syntax across its corpora. Semantic search is useful only for issue or passage recovery; it is not an identity resolver. See the official [advanced search operators](https://www.courtlistener.com/help/search-operators/) and [Citegeist behavior](https://www.courtlistener.com/help/citegeist/).

Route by corpus:

- `o`: published and explicitly requested unpublished case-law opinions;
- `r`/`d`: federal RECAP cases and docket metadata;
- `rd`: federal filing documents whose text or description may expose the authority, docket number, or parallel citation;
- `oa`: oral-argument metadata only when it provides a useful identity bridge.

Every search hit is a candidate. BM25 rank, semantic similarity, and a matching case name do not establish identity.

### 2. Structured alternatives

**Caselaw Access Project.** Harvard's Caselaw Access Project (CAP) provides structured metadata and full text for more than 6.5 million published state and federal decisions across U.S. history, with API and bulk access. Its data is already incorporated into CourtListener, so CAP is not normally an independent coverage expansion. Direct CAP access is still useful for:

- querying reporter/citation metadata in its original shape;
- recovering parallel citations or OCR text hidden by normalization;
- diagnosing CourtListener ingestion/indexing gaps or temporary outages; and
- retrieving a stable historical source record.

CAP should not be counted as independent corroboration when the CourtListener record derives from CAP. See Harvard's [CAP project description](https://lil.law.harvard.edu/our-work/caselaw-access-project/) and [open-data transition](https://lil.law.harvard.edu/blog/2024/03/26/transitions-for-the-caselaw-access-project/).

**GovInfo United States Courts Opinions.** GovInfo's `USCOURTS` collection contains authenticated opinions from selected federal appellate, district, bankruptcy, and national courts, generally from 2004 onward. It supports fielded search, predictable package identifiers, machine-readable metadata, PDFs, and an API search service. This is a preferred source for a federal opinion absent from CourtListener because GPO receives the files from the Administrative Office of the U.S. Courts and preserves chain of custody.

Search by court code, case number, party/title terms, date, and quoted locator. Use the [USCOURTS collection guide](https://www.govinfo.gov/help/uscourts) and [GovInfo developer/API resources](https://www.govinfo.gov/developers).

**RECAP, PACER, and PACER Case Locator.** For federal matters, first search RECAP case and filing corpora. If the docket or document is absent, PACER is the authoritative next route:

- PACER Case Locator searches a nationwide index by case type, number, title, party, and court type;
- the court-specific CM/ECF system can expose a docket or document not mirrored into RECAP; and
- a docket entry, opinion attachment, or related filing may reveal the neutral, slip, parallel, or corrected citation.

PACER requires an account and may incur fees, so it must be an explicit, budgeted adapter rather than an implicit model tool. See the official [PACER service](https://pacer.uscourts.gov/) and [CM/ECF court lookup](https://pacer.uscourts.gov/file-case/court-cmecf-lookup).

### 3. Issuing-court sources

When a court is known or can be resolved, prefer its official site over a broad search engine. CourtListener's `Court` record can provide the court identity and homepage from which an adapter can discover:

- opinion and order search forms;
- slip-opinion, advance-sheet, and recent-decision pages;
- court-specific APIs, RSS/Atom feeds, sitemaps, and predictable PDF paths;
- docket calendars or case-information portals; and
- historical opinion archives maintained by the court, state judiciary, or official law library.

Adapters should be court- or platform-specific. They must record the official domain, query fields, result URL, publication date, and retrieved file hash. HTML scraping is acceptable only when the official source offers no structured interface and its terms and robots policy permit it.

Jurisdiction routing matters:

- state trial: state judiciary case portal, county/court clerk, then official opinion/order archive if the court publishes decisions;
- state appellate/supreme: statewide opinion search, slip opinions, and official reporter/advance-sheet archive;
- federal trial/bankruptcy: RECAP, GovInfo, PACER/PCL, then the court site;
- federal appellate/special: GovInfo and the court's opinion archive before PACER document retrieval;
- U.S. Supreme Court: the Court's official opinions and orders surfaces;
- tribal, territorial, military, administrative, or special bodies: the body's own publication repository or the responsible government's official archive.

### 4. Legal vertical discovery

A legal vertical search is preferable to pure web search because it constrains the corpus and often exposes jurisdiction, date, citation, cited-by, and version relationships.

Google Scholar's Case law mode covers published U.S. state appellate and supreme court opinions since 1950, several federal court classes since 1923, and U.S. Supreme Court opinions since 1791. It supports court/date restriction, cited-by, related-case, and version discovery. It does not provide bulk API access and explicitly asks automated users to respect `robots.txt`, so use it as a manual or compliant interactive adapter—not a scraper. See the official [Google Scholar search and coverage help](https://scholar.google.com/intl/us/scholar/help.html).

Authorized Westlaw, Lexis, Bloomberg Law, Fastcase, vLex, or institutional library integrations can also be high-value legal verticals. They are optional capabilities: the agent must never assume credentials, automate a human-only license, or reproduce proprietary editorial material.

### 5. Pure open-web search is the last resort

Open-web search is enabled only when the exploration trace shows that applicable structured, official, and legal-vertical routes were attempted or could not be used. "CourtListener returned 404" is not sufficient justification.

Start constrained and relax deliberately:

1. exact quoted locator plus quoted case name;
2. exact locator plus court or one distinctive party;
3. quoted case name plus court and year/date range;
4. docket number plus party or court;
5. `site:` search over the resolved official court/judiciary domain;
6. known document tokens such as judge, neutral citation, or distinctive phrase;
7. reporter aliases, parallel-citation clues, and evidence-supported field corrections;
8. unconstrained web search only after the constrained queries fail.

Prefer results in this order: official court/government copy, institutional law library or reporter scan, stable nonprofit legal archive, then secondary legal publisher. Blogs, briefs, news, and generated summaries may provide discovery clues but are not candidate authority records by themselves.

## Search-plan requirements

Before executing probes, produce a route plan containing:

- competing explanations for the exact-lookup miss;
- normalized and alternate locator forms;
- resolved court and court-level evidence;
- applicable sources in priority order and why each applies;
- query variants and expected discriminators;
- credential, fee, robots, rate-limit, and access constraints; and
- stopping and escalation conditions.

Each probe records its source class (`courtlistener`, `structured_archive`, `official_court`, `legal_vertical`, `open_web`), exact query, filters, timestamp, result count, candidates retained/rejected, and next inference. The trace must make it possible to audit why open-web search was necessary.

## Output contract

The agent returns retrieval state, not a truth verdict:

```text
candidate_found
multiple_candidates
not_found_after_exploration
insufficient_evidence
search_unavailable
```

`not_found_after_exploration` means only that bounded retrieval found nothing. It must not become `False` or hallucinated without separate assessment. Retrieval also never silently mutates the original eyecite citation.

## Related documents

- [Reporter-to-Court Inference](Reporter%20Court%20Inference.md) — the exact-court projection of the broader jurisdiction inference, including the admission rule and the current registry.
- [Court Level Classification](Court%20Level%20Classification%20%5Bin%20progress%5D.md) — the `cl_jurisdiction` model and the level-aware availability priors.
- [CourtListener Search API](CourtListener%20Search%20API.md) — endpoint behavior, separate corpora, and the shared docket hierarchy that makes the case-law/RECAP gap matter.
- [Reporter Jurisdiction Inference](../development/retrieval/reporter-jurisdiction-inference%20%5Bin%20progress%5D.md) — the design of the broader `ReporterJurisdictionInference` representation and its invariants.
- [Not-Found Retrieval Agent](../development/retrieval/not-found-retrieval-agent%20%5Bin%20progress%5D.md) — the agent that consumes the inference and the source hierarchy to produce candidates.
