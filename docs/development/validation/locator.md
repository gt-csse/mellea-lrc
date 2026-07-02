# Locator Lookup

The validation lookup key for a full case citation is the three-field locator:

| Field | Example | Role |
|---|---|---|
| `volume` | `531` | CourtListener citation query input |
| `reporter` | `U.S.` | CourtListener citation query input |
| `page` | `98` | CourtListener citation query input |

Together these form `531 U.S. 98`. Case name, year, court, and pin cite do not
participate in this exact lookup. Comparing those fields with a retrieved record
is [assessment](../assessment/index.md), not validation.

## Outcomes

`validation/pipeline.py` maps the upstream response to one typed result:

| Condition | Validation status | Meaning |
|---|---|---|
| non-case citation | `skipped` | this validator does not handle the citation type |
| incomplete locator | `invalid` | required retrieval input is absent |
| HTTP 200 | `found` | one match was retrieved |
| HTTP 300 | `ambiguous` | multiple matches were retrieved |
| HTTP 404 | `not_found` | this lookup retrieved no match |
| HTTP 400 | `lookup_failed` | the source rejected the query |
| HTTP 429 | `throttled` | the source asked the client to retry later |
| any other failure | `lookup_failed` | transport, parsing, server, or malformed-response failure |

These are retrieval states, not truth labels. In particular, `found` is not a
claim that the extracted citation is correct, and `not_found` is not a claim
that it is false.

## Implementation contract

The lookup preserves `lookup_status`, `lookup_cache`, `lookup_key`, typed
`CitationMatch` values where available, failure detail, and upstream
`extra_data`. A malformed HTTP 200 with no matches becomes `lookup_failed`
rather than inventing a record.

Implementations live in `validation/pipeline.py`, with source access under
`courtlistener/`. Tests should assert retrieval state and provenance without
introducing assessment conclusions.
