# Not-Found Candidate Search

When a reporter lookup 404s, the case may still be real under a different locator. Retrieval performs retrieval only: it records what CourtListener returned and expresses no opinion about whether any result is the cited case.

Candidate-description, proceeding, document, and locator evidence must remain
distinct. WL/database locators also require a different verification route from
reporter coordinates. See
[Decision Publication, Access, and Locators](../../knowledge/Decision%20Publication,%20Access,%20and%20Locators.md).

In particular, **state-trial + WL** is a bounded shallow-search route when no
licensed WL backend is configured. After exact CourtListener, one constrained
case-law/official-court probe, and at most one exact web probe, retrieval should
stop and recommend Westlaw for full locator validation. It should not spend
model tokens iterating over fragmented state trial sources, and it must not
turn that coverage limitation into an opinion that the citation is incorrect.

Two decisions are intentionally separate.

The current search gate requires both parties. Missing, partial, or malformed
party extraction can therefore block retrieval before assessment's existing
re-extraction fallback runs. The proposed stage-independent repair boundary is
documented in
[Shared Re-extraction Workflow](../../architecture/shared-reextraction-workflow.md).

## 1. Case-name query engineering

Before a not-found query is sent, the asynchronous path runs one **search
preparation** attempt. Extraction carries an additive
`ExtractedCitation.asserted_decision_date` when eyecite bound a complete
month/day/year to the citation; this is transitional data until the canonical
citation model owns date components. Preparation then examines copied parties
and the date together, enforcing string grounding and citation-relative
position rules. A date must be written after the locator and within that
citation's full span. Only after that preparation is accepted does a second,
bounded LLM call provide plain normalized party search terms. It cannot supply
CourtListener query syntax; retrieval constructs one fielded query from those
terms and the court. A future deliberation node may add bounded follow-up
attempts after inspecting results. This version makes exactly one.

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
count, bounded candidate summaries, and error are traced independently. Counts
are never added together: the corpora may overlap. Retrieval does not claim
that a candidate is the cited authority.

## 3. Bounded RECAP docket evidence expansion

A RECAP candidate with a stable `docket_id` is expanded as retrieval evidence,
not accepted as an identity match. Before expansion, both engineered party
anchors must occur as tokens in the returned case name; otherwise the candidate
is retained as `skipped_party_mismatch` without spending docket requests. An
eligible expansion normally has two deterministic calls:

1. `/dockets/{docket_id}` obtains canonical proceeding metadata: case name,
   court, docket number, filing/termination dates, assigned and referred judges,
   nature of suit, cause, and jurisdiction type.
2. RECAP search is constrained to that exact `docket_id` and the asserted
   decision date. If the copied citation includes a full month/day/year, the
   query uses that exact date. Otherwise it uses the cited calendar year. The
   query also requests decisional terminology (`memorandum opinion`, `opinion`,
   `decision`, `judgment`, or `order`). If this exact search returns no
   decisional document, one pay-as-needed fallback searches a ±1-year window.
   Exact date/year therefore remains the high-precision first request and a
   mismatch costs at most one additional request.

The full date is consumed only when accepted by search preparation. Eyecite can
supply a citation-bound extraction hint; preparation may correct it from copied
text, and deterministic validation proves the accepted ISO date is present in
the bounded citation parenthetical. Retrieval has no regex/date fallback. A
candidate docket opened more than one year after the cited year is recorded as
`skipped_after_cited_year` without further HTTP requests. The tolerance
recognizes that docket opening date, decision date, and asserted citation year
are different metadata while still pruning obviously later same-name
proceedings.

Returned RECAP documents are bounded to eight and ranked by:

1. exact equality with a recovered full citation date, when available;
2. specificity of the decisional description cue;
3. distance from the cited year;
4. whether a downloadable document is available.

Every ranked item retains the docket/document IDs, filing date, descriptions,
page count, PACER document ID, availability, URL, detected cues, and year
distance. Generic orders can therefore be inspected without being confused
with stronger `Memorandum Opinion and Order` evidence.

The citation graph represents each candidate as an honest parallel branch:

```text
RECAP corpus probe
  -> docket metadata (candidate n)
  -> decisional documents (candidate n)
```

Candidate results depend on the terminal document-evidence branches. Failure or
absence of document evidence does not rewrite the original search result and
does not verify or reject a locator.

## What we can and can't get from CourtListener

- **Opinion-cluster search (`type=o`) is BM25 relevance full-text**, not exact match. `q="Brown v. Board"` returns ranked cluster hits, many not the cited case. Each result carries `meta.score.bm25`, nested individual opinions, and the response carries a total cluster `count`.
- **There is no exact case-name lookup anywhere in the API.** The DB-backed REST filtersets (`OpinionClusterFilter`, `DocketFilter`) don't expose `case_name` at all (verified in `cl/search/filters.py`); only citation (`volume/reporter/page`) is exact-filterable — the same locator that already 404'd. Those list endpoints also require a token; only `/search` is anonymous.
- Both opinion and RECAP search are relevance retrieval, not exact case-name lookup.
- Empirically, exact case names are **non-unique**: `Smith v. Jones` → 100+ distinct cases, `Roe v. Wade` → several clusters (merits, cert denials). The name yields a *set*, not one case, and top bm25 is not reliably the exact one.

## Shipped retrieval state

We attach the two search observations and at most five normalized candidate
summaries per corpus. A count still means only that the corresponding
CourtListener path returned that many hits; the bounded summaries do not imply
that the remaining results were examined.

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
    candidates: tuple[CaseNameSearchCandidate, ...] = ()  # at most five

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
  available (otherwise case name only), read `count`, retain five candidates,
  and expand eligible RECAP dockets.
- `search_opinions` and `search_recap` on both clients, with each method preserving its own response.
- Frontend: the retrieval trace's second step is **Case-name search**, showing opinion and RECAP outcomes separately.

CourtListener also supports fuzzy tokens such as `Peterson~1`. We do not apply
fuzziness to the first probe: edit-distance matching increases noise and does
not reliably expand legal abbreviations (`Sols.` → `Solutions`). A bounded fuzzy
probe belongs in the candidate-bearing retrieval workflow, where its results can
be inspected independently.

Out of scope (defer to a later deliberation/assessment layer): accepting a
proceeding or document as the cited authority, validating a WL locator,
downloading and interpreting every surfaced PDF, bm25 score interpretation,
pagination beyond the bounded search response, and exact-name filtering.

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
