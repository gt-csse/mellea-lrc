---
tags: [courtlistener, api, client, rate-limits]
status: active
---

# CourtListener client

`CourtListenerClient` is the only thing in the project that talks to the
network. Validation goes through it for every citation it checks, so its token
and its quota are what decide whether a run finishes.

It wraps four v4 endpoints, returns immutable typed objects, and raises one
error type carrying enough structure to decide whether a retry is worth it.

---

## Getting an API token

CourtListener requires a token for the endpoints used here — citation lookup
answers `401` without one.

1. Register at [courtlistener.com/sign-in](https://www.courtlistener.com/sign-in/)
   (free; it is a Free Law Project service).
2. Open your profile's
   [API tokens page](https://www.courtlistener.com/profile/api-tokens/) and copy
   the token.
3. Put it in `.env` as `COURTLISTENER_API_TOKEN`.

```bash
COURTLISTENER_API_TOKEN=your_token_here
```

Nothing loads `.env` implicitly. Either run through `uv run --env-file .env` or
export the variable yourself.

`COURTLISTENER_BASE_URL` is read too, defaulting to
`https://www.courtlistener.com/api/rest/v4/`. Point it at a cache or a proxy
without touching the code.

---

## Rate limits

**A free-tier token will not carry a real workload.** Validation makes several
calls per citation — a lookup, often a docket, and a page retrieval — so a
corpus of a few dozen filings runs to thousands of requests. Measured on the
26-filing benchmark as of 10 August 2026, a free-tier quota was exhausted before
the first document finished.

A `429` surfaces as `CourtListenerError` with `failure_type="api_limit"` and
`retryable=True`. Nothing in this client retries or backs off on your behalf;
that policy is left to the caller.

Two ways through:

**Ask for a higher quota.** CourtListener will raise the limit on request —
their maintainers are approachable, and the project is a non-profit that expects
research use. This is the simplest route if the work is ongoing.

**Put a cache in front.** A thin layer that stores responses by request and
replays them is enough, because a run repeats the same lookups many times over.
Point `COURTLISTENER_BASE_URL` at it. If you cache, fill it for the whole corpus
before trusting a final artifact: rerun until nothing new is fetched, since each
pass gets further before the quota stops it.

Expect a full validation run over the benchmark to take **up to two hours**,
dominated by HTTP latency and model inference rather than by local work.

---

## Usage

```python
from mellea_lrc.courtlistener import CourtListenerClient

client = CourtListenerClient()
```

`CourtListenerClient()` reads its configuration from the environment. Pass a
`CourtListenerConfig` to set the base URL or token explicitly, and a
`requests.Session` to control connection reuse:

```python
from mellea_lrc.courtlistener import CourtListenerConfig

client = CourtListenerClient(CourtListenerConfig(token="...", base_url="http://localhost:8000/api/rest/v4/"))
```

### Citation lookup

Resolve one reporter citation. Volume, reporter and page are separate
arguments — the three fields that identify a case in a reporter system.

```python
lookup = client.lookup_citation("347", "U.S.", "483")

for cluster in lookup.clusters:
    print(cluster.case_name, cluster.date_filed, cluster.cluster_id)
```

`lookup` is a `CourtListenerCitationLookup` with `citation`, `status`,
`clusters`, and `error_message`. `clusters` is a tuple of
`CourtListenerOpinionCluster` — a citation can resolve to more than one, and an
unresolvable one gives an empty tuple rather than an error.

Each cluster carries `cluster_id`, `case_name`, `date_filed`, `court`,
`court_id`, `docket_id`, `opinion_url`, its own `citations`, and
`sub_opinion_ids`. `cluster.year` is the filing year, derived from `date_filed`.

### Opinions

`sub_opinion_ids` is how you reach the text:

```python
opinion = client.get_opinion(cluster.sub_opinion_ids[0])
opinion.html_with_citations
```

A `CourtListenerOpinion` carries `opinion_id`, `cluster_id`, `opinion_type`,
`html_with_citations`, and `ordering_key`. The text arrives as HTML with
citations marked up; there is no plain-text field, so strip the markup yourself
if you need one.

### Search

For citations that do not resolve exactly.

```python
result = client.search("Brown v. Board of Education", "o")
result.count, result.results, result.next_cursor
```

`search_type` is one of `o` (opinions), `r` (RECAP), `rd` (RECAP documents), or
`d` (dockets); anything else raises `ValueError` before a request is made. Pass
`cursor=result.next_cursor` to page, and `semantic=True` to opt into semantic
search.

`results` is a tuple of raw mappings — deliberately not normalised, because the
shape differs per search type.

### Dockets

```python
docket = client.get_docket("12345")
docket.court_id, docket.case_name
```

`courtlistener_docket_url(absolute_url)` is a separate helper: CourtListener
returns docket links as site-relative paths, and this joins one onto the public
origin. It takes the path, not a docket id, and returns `None` when given
`None`.

---

## Errors

Every failure is a `CourtListenerError`. It is a `RuntimeError`, so it will not
be caught by accident, and it carries:

| attribute | what it tells you |
|---|---|
| `failure_type` | the categories below |
| `upstream_status_code` | the HTTP status, when there was a response |
| `retryable` | whether trying again could plausibly help |
| `url` | the request that failed |
| `upstream_detail` | the parsed error body, or the first 500 characters of it |

| `failure_type` | when | retryable |
|---|---|:--:|
| `api_limit` | `429` — quota exhausted | yes |
| `upstream_auth` | `401` / `403` — missing or rejected token | no |
| `upstream_not_found` | `404` | no |
| `upstream_bad_request` | other `4xx` | no |
| `upstream_error` | `5xx` | yes |
| `upstream_timeout` | no response within 45s | yes |
| `upstream_request_error` | connection failed before a response | yes |
| `upstream_invalid_json` | response body was not JSON | yes |
| `upstream_invalid_response` | JSON parsed, but not the expected shape | no |

The last one is worth separating from the rest: it means CourtListener answered
successfully with something this client does not recognise, which is a contract
change rather than a transient fault. `upstream_detail` holds the validation
errors.

---

## Substituting your own

`CourtListenerServiceClient` is a `Protocol` with the four methods above.
Anything satisfying it can be passed wherever the real client is expected —
`validate_document(document, client=...)` takes one. That is the seam for a
cache, a recorded fixture, or a stub in tests, and it is why the client is
injected rather than constructed inside the validation pipeline.
