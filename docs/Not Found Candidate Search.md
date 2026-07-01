# Not-Found Candidate Search — CourtListener Behavior & Design Notes

When a reporter lookup (`volume/reporter/page`) 404s, the plan is to run a
**case-name search** against CourtListener to surface candidate opinions, attach
them to `NotFoundCitationValidation`, and let a later assessment stage decide
whether any candidate is really the cited case.

Before building, we verified how CourtListener's opinion search (`type=o`)
actually behaves. It is **not** an exact-match lookup. Findings below.

## What CourtListener opinion search actually is

- **It is BM25 relevance full-text search** (Elasticsearch / the "Citegeist
  Relevancy Engine"), not an exact filter. A query like `q="Brown v. Board"`
  returns *relevance-ranked* opinions — many of which are **not** the exact
  case — ordered by score, highest first.
  Source: https://wiki.free.law/c/courtlistener/help/api/rest/v4/search
- **Every result carries its BM25 score**, nested as:
  ```json
  "meta": { "score": { "bm25": 2.1369965 } }
  ```
- **`count`** in the response is the **total number of relevance hits** for the
  query — i.e. it is *not* a count of exact case-name matches. (Confirmed in our
  own client test fixtures: the CL response includes a top-level `"count"`.)
- **Pagination is cursor-based** (`next`/`previous` → `cursor`), so "keep a
  pointer" = store `next_cursor` and never walk the pages.

### Result fields available for `type=o`

`absolute_url`, `caseName`, `caseNameFull`, `citation` (array of official
reporter cites), `citeCount`, `cluster_id`, `court`, `court_citation_string`,
`dateFiled`, `docketNumber`, `docket_id`, `snippet`, `status`, and a nested
`opinions` array. Our normalizer (`_normalize_search_result`, `type=o` branch)
currently keeps only `cluster_id`, `case_name`, `court_id`, `date_filed`,
`absolute_url`, `snippet` — it **drops `citation`, `meta.score.bm25`, and
`count`**, all of which we now want.

## Is a true (exact) case-name search possible? No.

Short answer: **CourtListener has no exact case-name lookup anywhere in its API.**
The relevance Search API is the *only* way to query by case name at all.

The database-backed REST endpoints (`/clusters/`, `/dockets/`) use
`rest_framework_filters` filtersets, and **`case_name` is simply not a
filterable field on any of them.** Verified against CourtListener source
(`cl/search/filters.py`, `main`):

- `OpinionClusterFilter.Meta.fields` — filterable keys are `id`, the various
  `date_*`, `scdb_*`, `citation_count`, `precedential_status`, `blocked`,
  `date_blocked`, plus related filters (`docket`, `panel`, `citations`,
  `sub_opinions`, `source`). **No `case_name`.**
- `DocketFilter.Meta.fields` — `docket_number`, `docket_number_core`,
  `nature_of_suit`, dates, `source`, … **No `case_name`.**
- `CitationFilter.Meta.fields` — `volume`, `reporter`, `page`, `type`, all
  `exact`. (So you *can* exact-filter clusters by citation via
  `citations__volume=…&citations__reporter=…&citations__page=…` — but that is
  the same locator that already 404'd, not a case-name path.)

Also note the REST list endpoints (`/clusters/`, `/dockets/`) **require an API
token** — anonymous requests get `401`. Only `/search/` is usable anonymously.

**Consequence:** there is no server-side exact case-name match. The best
achievable is:

1. Relevance search `q=caseName:"A v. B"` (BM25, ranked) to *retrieve* candidates, then
2. apply our **own** equality check (`case_names_equivalent` /
   `normalize_case_name`) to the returned page to *decide* which candidates are
   truly the same name.

Step 1 is retrieval (validation); step 2 is comparison (assessment). CL cannot
do the exactness for us — we do it on what the relevance search surfaces, which
is inherently page-bounded (we only ever see the top hits we fetched).

## Constraining toward the case name

Free-text `q="A v. B"` is the loosest option. Better:

- **Fielded + phrase:** `q=caseName:"Plaintiff v. Defendant"`.
  - `caseName` is the case-name field (camelCase form is accepted in `q`).
  - Double quotes make it a **phrase search** — contiguous tokens, no stemming
    or synonym expansion.
  - Source: https://www.courtlistener.com/help/search-operators/
- Default boolean operator is **AND**, so `caseName:(Roe v. Wade)` (no quotes)
  requires all three tokens but in any position; the quoted form is tighter.
- Default ordering is **`order_by=score desc`** (relevance). Other values exist
  (`dateFiled`, `citeCount`, …) but relevance is what we want.

**Caveat:** even `caseName:"A v. B"` is a *phrase contains* match, not string
equality. It will still match longer names that contain the phrase and is
subject to CL's tokenization. So the search **narrows** but does **not**
guarantee exactness.

## Design implications

1. **Validation cannot report "# of exact matches."** `count` is the number of
   *relevance hits*, not exact matches. If we surface a number from validation,
   it must be labeled as "candidate hits for this case-name query," not "exact
   matches." Deciding exactness is a **comparison**, which belongs to
   assessment (via the existing `normalize_case_name` / `case_names_equivalent`
   equality applied to each candidate's `caseName`).

2. **Store the ranked DTOs verbatim, with the score.** Because CL already ranks
   by relevance, the top-of-page candidates are the most likely case. Keep
   `meta.score.bm25` per candidate so a downstream stage (or a human reviewer)
   can see confidence, and keep `count` + `next_cursor` as the pointer.

3. **Query construction:** use `q=caseName:"<build_extracted_case_name>"` with
   the default `score desc` ordering.

4. **Skip gate unchanged:** only search when both parties were extracted
   (`plaintiff` and `defendant`) — a real "A v. B". Single-party / empty case
   names produce noise. (Gating on *our* extraction side is correct regardless
   of CL's own naming conventions like `In re …` / `Ex parte …`.)

## Final decision — report the count only

We deliberately do **not** store, rank, or normalize the individual candidates.
Two reasons: (1) many real citations are only *semantically* equivalent, so a
string/normalized exact-name filter would wrongly discard true matches — and
that judgement is a comparison that belongs to assessment, not validation; and
(2) the reviewer-facing value is simply "how many CourtListener cases share this
case name," which is CL's `count`. So validation attaches just the count.

### Shipped shape (`validation/types.py`)

```python
class CaseNameSearchStatus(str, Enum):
    SEARCHED = "searched"
    SKIPPED_NO_CASE_NAME = "skipped_no_case_name"
    SKIPPED_PARTIAL_CASE_NAME = "skipped_partial_case_name"   # only one party
    SEARCH_UNAVAILABLE = "search_unavailable"                 # client can't search
    SEARCH_FAILED = "search_failed"
    NOT_ATTEMPTED = "not_attempted"                           # default / older payloads

@dataclass(frozen=True, slots=True)
class CaseNameSearchTrace:
    status: CaseNameSearchStatus = CaseNameSearchStatus.NOT_ATTEMPTED
    query: str | None = None            # caseName:"A v. B"
    case_count: int | None = None       # CL 'count' — matching opinions, NOT exact matches
    error_message: str | None = None
```

Attached as `candidate_search: CaseNameSearchTrace` on
`NotFoundCitationValidation`, mirroring how `court_resolution` hangs off the
found variant.

### Wiring

- `validation/not_found_search.py` — `search_case_name_candidates(citation,
  client)`: skip gate (both parties), `q=caseName:"A v. B"`, reads `count`.
- `CourtListenerClient.search_opinions` / `CourtListenerAccessClient
  .search_opinions` — `type=o` search; `search()` now surfaces `count`.
- Serialization: `CaseNameSearchTracePayload` on the not-found payload.
- Frontend: the validation trace's second step becomes **Case-name search**
  for a not-found cite, reporting "Cases found: N".

The candidate DTOs, bm25 score, pagination cursor, and any exact-name filtering
are intentionally **out of scope** here — if assessment later wants to decide
*which* candidate is the cited case, it re-queries and compares there.
