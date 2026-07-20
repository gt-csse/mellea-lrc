# CourtListener client

`CourtListenerClient` provides direct CourtListener API access. Its first supported operation is the
exact citation-lookup endpoint, with typed results and token rotation after upstream rate limits.

Configure one or more API tokens. Numbered variables are supported so the client can try the next
token after an upstream `429` response.

```shell
export COURTLISTENER_API_TOKEN="..."
export COURTLISTENER_API_TOKEN_2="..."
```

```python
from mellea_lrc.courtlistener import CourtListenerClient

client = CourtListenerClient()
lookup = client.lookup_citation("347", "U.S.", "483")
```

`lookup` is an immutable `CourtListenerCitationLookup`. Its `records` tuple contains
`CourtListenerCitationRecord` values. A `200` status identifies one match, `300` preserves every
ambiguous candidate returned by CourtListener, and an explicit `404` result preserves a not-found
lookup. Responses without exactly one result are rejected as invalid upstream responses.

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
