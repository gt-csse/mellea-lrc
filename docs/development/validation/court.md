# Court Retrieval

For a `found` locator, validation attempts to resolve the court associated with
the retrieved CourtListener record. This is retrieval enrichment only.

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

Court resolution is implemented in `validation/court_resolution.py`. The
document pipeline owns a per-run docket cache so citations sharing a docket do
not trigger duplicate GET requests.
