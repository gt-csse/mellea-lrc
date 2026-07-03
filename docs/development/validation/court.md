# Court Retrieval

For every candidate returned by a `found` or `ambiguous` locator, validation
attempts to resolve the court associated with that CourtListener record. This
is retrieval enrichment only.

`CourtResolutionTrace` records:

- the retrieved `courtlistener_court_id`;
- whether it came from the cluster payload or a docket lookup;
- the docket identifier and URL used;
- whether the per-document docket cache was used; and
- any lookup failure.

Validation never compares this value with eyecite's extracted `court`, infers
whether they agree, or changes the citation. A missing court remains missing.
The comparison and the optional exclusive-reporter fallback are documented in
[Court Assessment](../assessment/court.md).

Court resolution is implemented in `validation/court_resolution.py`. Both found
and ambiguous validation use
`RetrievedCandidate(candidate_id, record, court_resolution)`; found owns one and
ambiguous owns a tuple.
The document pipeline owns a per-run docket cache so candidates or citations
sharing a docket do not trigger duplicate GET requests.
