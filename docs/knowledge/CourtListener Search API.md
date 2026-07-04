---
tags: [courtlistener, search, api, semantic-search, case-law]
status: active
created: 2026-06-30
source: https://wiki.free.law/c/courtlistener/help/api/rest/v4/search
---

# CourtListener Search API

This document summarizes the CourtListener v4 Search API behavior relevant to
`mellea-lrc`. The authoritative upstream documentation is the
[Free Law Project Search API reference](https://wiki.free.law/c/courtlistener/help/api/rest/v4/search).

CourtListener search is backed by its search engine rather than its relational
API. Search results are relevance-ranked candidates, not exact database lookup
results. Use `POST /citation-lookup` for authoritative volume/reporter/page
lookup and use search to discover or rank candidates.

## Unified CourtListener Case Hierarchy

CourtListener uses `Docket` as a shared top-level case record across several
corpora. A docket is not evidence that a case came from PACER or RECAP, and a
docket does not imply that docket entries exist.

The child hierarchy depends on the corpus:

```text
PACER / RECAP:   Docket -> DocketEntry -> RECAPDocument
Case law:        Docket -> OpinionCluster -> Opinion
Oral arguments:  Docket -> Audio
```

Free Law Project documents this directly: in PACER, dockets connect entries,
parties, and attorneys; in case law, dockets sit above opinion clusters; and in
oral-argument data, dockets sit above audio records. See the
[official Case Law API documentation](https://wiki.free.law/c/courtlistener/help/api/rest/v4/case-law).

Harvard's Caselaw Access Project data was incorporated into CourtListener rather
than maintained as a separate lookup system. During ingestion, Free Law Project
corrected and normalized CAP fields, including mapping Harvard's plain-text
court values into CourtListener's verified `Court` records. See Free Law
Project's [CAP and CourtListener data comparison](https://wiki.free.law/c/courtlistener/help/general/how-does-the-data-in-harvards-caselaw-access-project-compare-to-courtlisteners-case-law-database).

### The case-law corpus is live, not a CAP-era snapshot

CAP is one historical source, not the endpoint of CourtListener's case-law
collection. Free Law Project says it continuously collects decisions as courts
release them, has added more than 1.75 million records through scrapers, and has
supported direct court publishing since 2020. Its search `source` values make
the provenance distinction explicit: `U` is Harvard CAP, `C` is a court
website, and `D` is direct court input. See the official
[case-law coverage statement](https://www.courtlistener.com/help/coverage/opinions/)
and [search-field reference](https://www.courtlistener.com/help/search-operators/).

Live verification on 2026-07-03 found Tenth Circuit `type=o` clusters dated as
recently as 2026-07-01, with `source=C`. The same search returned 99 Tenth
Circuit clusters filed in 2026, 230 in 2025, 170 in 2024, and 119 in 2021.
CourtListener separately says real-time case-law alerts normally arrive within
about an hour of court publication. These facts rule out a corpus frozen at the
Harvard era and show that ordinary successful ingestion can be near-real-time.

Continuous ingestion does not imply complete ingestion. A court opinion may be
present as RECAP docket metadata or a filing record without ever becoming an
`OpinionCluster`. The reverse is also possible: historical and state case-law
clusters may have dockets but no docket entries. Corpus existence must therefore
be tested per layer, not inferred across the shared docket hierarchy.

#### Observed gap: *Peterson v. Nelnet Diversified Solutions*

The published Tenth Circuit opinion at `15 F.4th 1033` illustrates a persistent
cross-corpus gap rather than normal lag:

- the issuing court published the opinion on 2021-10-08;
- RECAP contains appellate dockets `59714552` and `67593265`, both named
  `Peterson v. Nelnet Diversified Solutions`;
- each docket has an `rd` record described as `Case termination for opinion`,
  but the document is unavailable and has no `cluster_id`;
- CourtListener's citation resolver for `15 F.4th 1033` returns 404;
- `type=o` returns zero for the citation, both appellate docket numbers, both
  docket IDs, the party-name intersection, and a distinctive text phrase;
- the official Tenth Circuit PDF remains available at the
  [court source](https://www.ca10.uscourts.gov/sites/ca10/files/opinions/010110588582.pdf).

Because independent cluster metadata and content probes all miss, this is not
merely delayed reporter-citation enrichment or an unexpectedly formatted case
name. The best current explanation is that RECAP captured an opinion-related
docket event while case-law ingestion failed to create or index the cluster.
The evidence does not establish which upstream ingestion step failed.

Consequences for `mellea-lrc`:

- State appellate opinions can have a valid `docket_id` and no docket entries.
- `/dockets/{docket_id}` remains the preferred source for `court_id` for both
  federal and state case-law matches.
- The presence or absence of `DocketEntry` records must not control this path.
- `/search?q=cluster_id:<cluster_id>&type=o` is an operational fallback when the
  docket request or expected docket fields fail, not a state-case fallback.
- A search fallback must require exact equality with the known cluster ID and
  must use keyword mode.

The shared docket hierarchy does not make coverage uniform across court
levels. See [Court Level Classification](Court%20Level%20Classification%20%5Bin%20progress%5D.md) for
the classification model and level-aware corpus routing.

Live verification on 2026-06-30 found that state supreme and appellate CAP
dockets for Myers, Redhair, Watkins, and Ford all returned `court_id` from the
docket endpoint while returning zero docket entries. A RECAP federal comparison
docket returned docket entries as expected.

## Upstream Endpoint

```text
GET https://www.courtlistener.com/api/rest/v4/search/
```

The upstream API uses the same query parameters as the CourtListener web search.
The primary parameters are:

| Parameter | Meaning |
| --- | --- |
| `q` | Search text, including supported fielded and Boolean operators. |
| `type` | Corpus to search. Defaults to case law (`o`). |
| `semantic=true` | Enables semantic search. Supported only for case law. |
| `cursor` | Cursor returned by the API for pagination. |
| `order_by` | Upstream result ordering. |
| `highlight=on` | Enables highlighted snippets. |

CourtListener also accepts a `POST` containing a precomputed 768-dimensional
embedding. That option avoids sending the original semantic query text to
CourtListener, but it requires their compatible fine-tuned embedding model.

## Search Types

| Type | Upstream corpus |
| --- | --- |
| `o` | Case-law opinion clusters with nested individual opinions. The response `count` is a cluster count. |
| `r` | Federal RECAP cases with up to three matching nested documents. |
| `rd` | Federal PACER filing documents without docket metadata. |
| `d` | Federal PACER cases without filing metadata. |
| `p` | Judges. |
| `oa` | Oral-argument recordings. |

Our wrapper currently supports `o`, `r`, `rd`, and `d` only.

## Keyword and Semantic Search

Keyword search is the upstream default. It supports fielded queries and
advanced operators. Many search fields use camel case, for example:

```text
caseName:(Johnson v. City of Shelby)
court_id:scotus
dateFiled:[2014-01-01 TO 2014-12-31]
```

Semantic search is enabled with `semantic=true`. It is intended for natural
language descriptions of legal facts or issues. It is not an exact case-name or
citation resolver. Quoted terms in a semantic query enable hybrid lexical and
semantic retrieval.

Semantic search is available only for `type=o`. Keyword search supports deep
pagination; semantic search is intended to return the most relevant candidates.

## Response Shape

The upstream response contains:

```json
{
  "count": 1,
  "next": null,
  "previous": null,
  "results": []
}
```

Case-law results can include:

- `cluster_id`
- `caseName` and `caseNameFull`
- `citation`, a list of parallel citations identifying the opinion
- `court` and `court_id`
- `dateFiled`
- `docketNumber` and `docket_id`
- `absolute_url`
- nested opinions and snippets
- `meta.score.bm25`, the Elasticsearch keyword-relevance score

The `citation` list is distinct from an opinion's `cites` field. `citation`
contains parallel reporter identifiers for the result; `cites` identifies other
opinions cited by an opinion.

The top-level unit is an opinion cluster even though CourtListener labels the
search surface “Opinions” or “Case Law.” For example, the merits result for
*Roe v. Wade* is cluster `108713`; its nested `opinions` include distinct lead,
concurrence, dissent, and combined-opinion IDs. `caseName:(Roe AND Wade)` returns
multiple clusters, including the merits decision and related procedural orders,
not one row per judicial writing.

## Result and Coverage Caveats

- Case-law search returns published opinions by default. Other statuses must be
  requested explicitly.
- Search results can match query text inside an opinion, not only metadata.
  A quoted citation can therefore return opinions that cite the target case.
- For `type=d` and `type=r`, counts above 2,000 use an approximate cardinality
  aggregation with documented error of approximately plus or minus six percent.
- Upstream search results are cached for ten minutes.
- BM25 is a ranking score, not a probability or a retrieval confidence value.
- A candidate should not be treated as an exact citation match unless its
  returned `citation` list contains the normalized locator.
- A zero `type=o` count means no matching indexed cluster. It does not establish
  that the opinion is unreal, unpublished, or absent from the issuing court.
- A RECAP docket or opinion-related filing does not establish that a linked
  opinion cluster exists.

## `mellea-lrc` Wrapper

Our CourtListener access service exposes:

```text
GET /search?q=<query>&type=<o|r|rd|d>&semantic=<true|false>&cursor=<cursor>
```

Supported wrapper parameters:

| Parameter | Required | Default |
| --- | --- | --- |
| `q` | Yes | None |
| `type` | Yes | None |
| `cursor` | No | None |
| `semantic` | No | `false` |

The wrapper returns normalized result fields plus the original upstream payload
under `raw`. At present, case-law normalization exposes the cluster ID, case
name, court ID, filing date, URL, snippet, and resource URI. Parallel citations,
docket metadata, and ranking scores remain available only under `raw`.

The wrapper does not currently expose upstream `order_by`, `highlight`, `p`,
`oa`, or embedding-based `POST` search.

## Citation-Recovery Guidance

For citation retrieval and recovery:

1. Use `/citation-lookup` first for the exact volume/reporter/page triad.
2. If lookup returns `404`, search by extracted case name using keyword mode.
3. Add extracted court and year constraints when available.
4. Inspect each candidate's parallel `citation` list from `raw`.
5. Treat name-only agreement as a candidate, not as a recovered citation.
6. Reserve semantic search for natural-language legal-issue queries or a
   deliberately evaluated fallback; do not use it as the default identity
   search.

## Related Upstream Documentation

- [Legal Search API](https://wiki.free.law/c/courtlistener/help/api/rest/v4/search)
- [Advanced search operators](https://www.courtlistener.com/help/search-operators/)
- [Citegeist ranking and semantic search](https://www.courtlistener.com/help/citegeist/)
- [Case-law coverage](https://www.courtlistener.com/help/coverage/opinions/)
