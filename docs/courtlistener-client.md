# CourtListener clients

`mellea_lrc.courtlistener` provides two interchangeable ways to retrieve CourtListener data:

- `CourtListenerClient` calls CourtListener's REST API directly.
- `CourtListenerAccessClient` calls an already-deployed HTTP access service.

This package is client-side infrastructure. It does not deploy or configure an access service.

## Direct API access

Configure one or more API tokens. Numbered variables are supported because the client can rotate to
the next token after an upstream `429` response.

```shell
export COURTLISTENER_API_TOKEN="..."
export COURTLISTENER_API_TOKEN_2="..."
```

```python
from mellea_lrc.courtlistener import CourtListenerClient

client = CourtListenerClient()
docket = client.get_docket(4214664)
opinions = client.search_opinions("Brown v. Board of Education")
citation = client.lookup_citation("347", "U.S.", "483")
```

`CourtListenerConfig` also accepts an explicit `base_url` and token tuple. The default base URL is
CourtListener REST API v4. Client-side rate limits are configurable with
`COURTLISTENER_RATE_LIMIT_*` variables or `CourtListenerRateLimitConfig`.

The direct client uses `NullCache` by default. Any implementation of `CacheStore` can be injected.
`R2Cache` is available when the optional dependency is installed:

```shell
uv sync --group courtlistener-r2
```

## Existing access service

Set `CL_ACCESS_URL` to the base URL of a compatible deployed service:

```shell
export CL_ACCESS_URL="https://your-courtlistener-access.example"
```

```python
from mellea_lrc.courtlistener import CourtListenerAccessClient

client = CourtListenerAccessClient()
citation = client.lookup_citation("347", "U.S.", "483")
docket = client.get_docket(4214664)
opinions = client.search_opinions("Brown v. Board of Education")
```

The remote client expects these routes:

| Operation | Method and route |
| --- | --- |
| Citation lookup | `POST /citation-lookup` |
| Docket | `GET /dockets/{id}` |
| Opinion cluster | `GET /clusters/{id}` |
| Docket entries | `GET /docket-entries/search` |
| Opinion and RECAP search | `GET /search` |

The service implementation and deployment mechanism are intentionally outside this module. As long
as a deployment honors the route and JSON contracts above, callers do not need to know how it is
hosted.

## Stable boundary behavior

Citation lookup responses from both clients pass through the same strict Pydantic boundary and are
returned as immutable `CourtListenerCitationLookup` records. Unknown upstream JSON fields are retained
in immutable `extra_data`, allowing CourtListener to add fields without discarding information.

Direct API failures raise `CourtListenerError`, whose `failure_type`, upstream status, retryability,
and public serialization are stable. The remote client preserves structured service error responses
for retrieval orchestration to inspect.
