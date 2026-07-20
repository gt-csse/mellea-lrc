# CourtListener client

`CourtListenerClient` provides direct CourtListener API access for exact citation lookup and search,
with typed results.

Configure one API token.

```shell
export COURTLISTENER_API_TOKEN="..."
```

```python
from mellea_lrc.courtlistener import CourtListenerClient

client = CourtListenerClient()
lookup = client.lookup_citation("347", "U.S.", "483")
search = client.search("Brown v. Board of Education", "o")
```

`lookup` is an immutable `CourtListenerCitationLookup`. Its `records` tuple contains
`CourtListenerCitationRecord` values. A `200` status identifies one match, `300` preserves every
ambiguous candidate returned by CourtListener, and an explicit `404` result preserves a not-found
lookup. Responses without exactly one result are rejected as invalid upstream responses.

`search` supports CourtListener's `o` (opinions), `r` (RECAP cases), `rd` (RECAP documents), and
`d` (dockets) corpora. It returns an immutable `CourtListenerSearchResult`, including the total
`count`, immutable result records, and pagination cursors. Pass `semantic=True` to opt into
CourtListener semantic search.

CourtListener JSON is validated at the package boundary by transport-only Pydantic payloads:

- `CourtListenerCitationLookupResponsePayload` validates the outer one-result response list.
- `CourtListenerCitationLookupResultPayload` validates that response's citation result.
- `CourtListenerCitationLookupRecordPayload` validates one item from its `clusters` collection.

The payloads are converted immediately into the immutable domain objects above.

`CourtListenerConfig` accepts an explicit `base_url` and token tuple when
environment-based configuration is undesirable. The default base URL is
`https://www.courtlistener.com/api/rest/v4/`.

API and transport failures raise `CourtListenerError`. Its structured fields distinguish upstream
authentication failures, rate limits, timeouts, bad requests, missing resources, invalid JSON, and
other upstream errors.
