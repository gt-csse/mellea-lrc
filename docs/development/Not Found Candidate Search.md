# Not-Found Candidate Search

When a reporter lookup 404s, the case may still be real under a different locator. Validation runs one CourtListener case-name search and reports **how many** opinions match — retrieval only, no comparison (deciding *which* candidate is the cited case belongs to assessment).

## What we can and can't get from CourtListener

- **Opinion search (`type=o`) is BM25 relevance full-text**, not exact match. `q="Brown v. Board"` returns ranked hits, many not the cited case. Each result carries `meta.score.bm25`, and the response carries a total `count`.
- **There is no exact case-name lookup anywhere in the API.** The DB-backed REST filtersets (`OpinionClusterFilter`, `DocketFilter`) don't expose `case_name` at all (verified in `cl/search/filters.py`); only citation (`volume/reporter/page`) is exact-filterable — the same locator that already 404'd. Those list endpoints also require a token; only `/search` is anonymous.
- Best case-name lever is a fielded phrase: `q=caseName:"A v. B"` (contiguous, no stemming) — narrows, doesn't guarantee equality.
- Empirically, exact case names are **non-unique**: `Smith v. Jones` → 100+ distinct cases, `Roe v. Wade` → several clusters (merits, cert denials). The name yields a *set*, not one case, and top bm25 is not reliably the exact one.

## Shipped decision — report the count only

We attach just the count, no candidates, no normalizer. Reasons: (1) many real matches are only *semantically* equivalent, so exact-name filtering is lossy and is a comparison that belongs to assessment; (2) the reviewer-facing value is simply "N CourtListener cases share this name" = CL's `count`.

`CaseNameSearchTrace` on `NotFoundCitationValidation` (mirrors `court_resolution` on the found variant):

```python
class CaseNameSearchStatus(str, Enum):
    SEARCHED, SKIPPED_NO_CASE_NAME, SKIPPED_PARTIAL_CASE_NAME,
    SEARCH_UNAVAILABLE, SEARCH_FAILED, NOT_ATTEMPTED

@dataclass(frozen=True, slots=True)
class CaseNameSearchTrace:
    status: CaseNameSearchStatus = NOT_ATTEMPTED
    query: str | None = None          # caseName:"A v. B"
    case_count: int | None = None     # CL 'count' — matches, NOT exact matches
    error_message: str | None = None
```

Wiring:
- `validation/not_found_search.py` — skip gate (**both parties** required; single/no party is noise), query `caseName:"A v. B"`, read `count`.
- `search_opinions` on both clients; `search()` now surfaces `count`. Guarded by `hasattr` so partial clients degrade to `SEARCH_UNAVAILABLE`.
- Frontend: the validation trace's second step is **Case-name search** for a not-found cite, showing "Cases found: N".

Out of scope (defer to assessment if needed): candidate DTOs, bm25 score, pagination, and any exact-name filtering.

## Future direction (ordered — B depends on A)

**A. Rank candidates into high-probability dockets.** Move past a raw count to identifying *which* opinions/dockets are likely the cited case. Explore a programmatic signal — a third-party relevance score (e.g. CL's `meta.score.bm25`) or our own enhanced implementation — that turns "N cases share this name" into a meaningful, ranked suggestion. This is the prerequisite for any real not-found suggestion.

**B. Re-extract the case name, then search — gated on A.** For not-found cites, re-extract the case name (as assessment already does for case-name reassessment) and re-run the search on the cleaned name. This only pays off once A proves the case-name search actually yields useful suggestions; without A it just feeds a better query into an output nobody can act on. **Do not start B until A is meaningful.**
