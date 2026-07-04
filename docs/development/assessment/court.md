# Court Assessment

Court assessment compares eyecite's extracted court slug with the
`courtlistener_court_id` retrieved during retrieval.

| Condition | Status |
|---|---|
| either value is absent | `missing` |
| slugs are equal | `exact_match` |
| slugs differ | `mismatch` |

## Reporter inference follow-up

When the initial result is `missing`, assessment may infer the citation-side
court if the reporter publishes decisions from exactly one court. This is a
field-local extraction fallback, not retrieval enrichment: the initial result,
reporter, pre-inference value, and reassessment are all retained in
`CourtAssessmentRun`.

Inference can run when the CourtListener court is also missing. In that case the
final comparison remains `missing`; inference alone does not create agreement
or disagreement. Reporters serving multiple courts are deliberately excluded.

Implementation: `assessment/fields/court/{assess,inference}.py` and
`assessment/types/court.py`. Mapping policy and coverage notes live in
[Reporter-to-Court Inference](../../knowledge/Reporter%20Court%20Inference.md).
The exact-court fallback is the exhaustive-singleton projection of the broader
[Reporter Jurisdiction Inference](../retrieval/reporter-jurisdiction-inference%20%5Bin%20progress%5D.md)
design; court assessment should not absorb its multi-court retrieval uses.
