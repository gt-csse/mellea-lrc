# Not-Found Candidate Search

When a reporter lookup 404s, the case may still be real under a different locator. Retrieval performs retrieval only: it records what CourtListener returned and expresses no opinion about whether any result is the cited case.

Two decisions are intentionally separate.

The current search gate requires both parties. Missing, partial, or malformed
party extraction can therefore block retrieval before assessment's existing
re-extraction fallback runs. The proposed stage-independent repair boundary is
documented in
[Shared Re-extraction Workflow](../../architecture/shared-reextraction-workflow.md).

## 1. Case-name query engineering

The current reasonable default is one fielded token intersection plus the
extracted court when one is available, such as
`caseName:(Peterson AND Nelnet) AND court_id:ca10`. `caseName:` keeps body-text
mentions from dominating, while one meaningful anchor per party avoids
requiring exact punctuation, abbreviations, entity suffixes, or a contiguous
full name. Citations without an extracted court retain the case-name-only query.

This query is a replaceable engineering choice, not a claim that the resulting
cases match the citation. Fuzzy tokens and alternative expansions may be tested
later as separate probes; they are not silently mixed into today's default.

## 2. Default search-path implementation

For every eligible not-found citation, retrieval sends the same engineered
query to both default CourtListener paths:

- opinion-cluster search (`type=o`);
- RECAP search (`type=r`).

Both searches always run when supported. Their HTTP status, cache outcome,
count, and error are traced independently. Counts are never added together: the
corpora may overlap, and retrieval does not deduplicate, rank, compare, or
interpret results. Later design work can decide how assessment should consume
these two observations.

## What we can and can't get from CourtListener

- **Opinion-cluster search (`type=o`) is BM25 relevance full-text**, not exact match. `q="Brown v. Board"` returns ranked cluster hits, many not the cited case. Each result carries `meta.score.bm25`, nested individual opinions, and the response carries a total cluster `count`.
- **There is no exact case-name lookup anywhere in the API.** The DB-backed REST filtersets (`OpinionClusterFilter`, `DocketFilter`) don't expose `case_name` at all (verified in `cl/search/filters.py`); only citation (`volume/reporter/page`) is exact-filterable — the same locator that already 404'd. Those list endpoints also require a token; only `/search` is anonymous.
- Both opinion and RECAP search are relevance retrieval, not exact case-name lookup.
- Empirically, exact case names are **non-unique**: `Smith v. Jones` → 100+ distinct cases, `Roe v. Wade` → several clusters (merits, cert denials). The name yields a *set*, not one case, and top bm25 is not reliably the exact one.

## Shipped decision — report corpus-scoped counts only

We attach the two search observations, no candidates and no normalizer. A count
means only that the corresponding CourtListener path returned that many hits.

`CaseNameSearchTrace` on `NotFoundCitationRetrieval` (mirrors `court_resolution` on the found variant):

```python
class CaseNameSearchStatus(str, Enum):
    SEARCHED, PARTIAL, SKIPPED_NO_CASE_NAME, SKIPPED_PARTIAL_CASE_NAME,
    SEARCH_UNAVAILABLE, SEARCH_FAILED, NOT_ATTEMPTED

@dataclass(frozen=True, slots=True)
class CaseNameSearchProbe:
    corpus: Literal["o", "r"]
    status: CaseNameSearchStatus
    request_trace: CourtListenerRequestTrace
    case_count: int | None = None     # corpus-scoped hits; type=o counts clusters

@dataclass(frozen=True, slots=True)
class CaseNameSearchTrace:
    status: CaseNameSearchStatus = NOT_ATTEMPTED
    query: str | None = None
    probes: tuple[CaseNameSearchProbe, ...] = ()
```

The serialized retrieval contract exposes common request metadata only as
`request_trace` (`http_status`, `cache`, `key`, and `error_message`). Citation
lookup, each search probe, and docket-based court resolution use this same shape.

Each probe state is determined from transport status before the response count:

- `searched`: HTTP 200 with a non-negative integer `count` (including zero);
- `search_failed`: non-200 HTTP status or no HTTP response.

The parent trace is `searched` when both probes succeed, `partial` when one
succeeds, and `search_failed` when neither succeeds (unless both paths are
explicitly unavailable).

Within each probe, `case_count` never determines whether the request succeeded.
It is populated only for `searched`; all other states preserve `http_status`
and diagnostics with `case_count=None`. Current cl-access responses expose the
count at the top level. Older deployed responses omit that normalized field but
preserve the same value under `raw.count`; the parser supports both shapes.

Wiring:
- `retrieval/not_found_search.py` — skip gate (**both parties** required;
  single/no party is noise), select one meaningful alphanumeric anchor from each
  party, query `caseName:(A AND B) AND court_id:<court>` when a court is
  available (otherwise case name only), and read `count`.
- `search_opinions` and `search_recap` on both clients, with each method preserving its own response.
- Frontend: the retrieval trace's second step is **Case-name search**, showing opinion and RECAP outcomes separately.

CourtListener also supports fuzzy tokens such as `Peterson~1`. We do not apply
fuzziness to the first probe: edit-distance matching increases noise and does
not reliably expand legal abbreviations (`Sols.` → `Solutions`). A bounded fuzzy
probe belongs in the candidate-bearing retrieval workflow, where its results can
be inspected independently.

Out of scope (defer to assessment if needed): candidate DTOs, bm25 score,
pagination, and any exact-name filtering.

## Planned replacement

The count-only fallback above remains the description of shipped behavior. The
next design is not merely a better ranking of the same query. It is a bounded,
iterative retrieval agent that can inspect case-name candidates, retain the
failed locator as search evidence, use CourtListener jurisdiction as a coverage
prior, search appropriate CourtListener corpora, and escalate to general or
court-specific sources.

See [Not-Found Retrieval Agent](./not-found-retrieval-agent%20%5Bin%20progress%5D.md) for inputs,
tools, deliberation, the non-opinionated output contract, assessment boundary,
and delivery plan.
