---
tags: [validation, courtlistener, citation-identity]
status: active
---

# Validation

The active validation checkpoint settles identity for full reporter locators.
It preserves every lookup candidate and every model attempt in a serializable
node trace. It does not currently decide lookup misses, docket locators, leaf
citations, or pinpoint support.

## Running it

```python
import asyncio
from mellea_lrc.extraction import extract_from_raw_document
from mellea_lrc.validation import validate_document_identity

identity = asyncio.run(validate_document_identity(document))
for citation in identity.citations:
    print(citation.citation_id, citation.identity_resolution)
```

`validate_document` currently calls this same checkpoint. Both functions accept
an injectable CourtListener client and Mellea session:

```python
validate_document_identity(document, client=my_client, session=my_session)
```

The document context sent to Mellea is target-only: neighbouring locators are
masked without moving source offsets. Colocation remains an extraction-level
hook; it is never shown as an object or instruction to the model.

## Current route

Only a `FullCaseCitation` with volume, reporter, and page enters exact lookup.
All other citation types get a terminal `deferred` identity node. In particular,
a `DocketCitation` is deferred without a lookup until docket-number-only search
and its disambiguation contract are settled.

```text
full reporter locator
├── exact lookup: one candidate
│   └── field checks → candidate summary
│       ├── exactly one confirmed match → resolved
│       └── zero confirmed matches → grounded model choice → resolved | no_match | deferred
├── exact lookup: 2–19 candidates
│   └── field checks for every candidate → candidate summary
│       ├── exactly one confirmed match → resolved
│       └── zero or multiple confirmed matches → grounded model choice → resolved | no_match | deferred
├── exact lookup: no candidate, failure, or incomplete locator → deferred
└── exact lookup: at least 20 candidates → deferred
```

A candidate summary is evidence, not an identity decision. It retains every
reviewed candidate whether its assessment is `match`, `partial_match`, or
`mismatch`. The raw exact-lookup node retains the complete CourtListener result
set even when review is deferred at the 20-candidate limit.

For a zero-match or ambiguous bounded summary, Mellea sees the local citation
context and the complete candidate list. It must reparse the filing's stated
case name, court, and date; select one candidate by index; or return
`no_match`. It cannot change the reporter locator. The resulting
`MelleaLocatorCandidateChoiceNode` records the reparsed fields, decision,
rationale, and complete IVR repair trace. The terminal
`LocatorIdentityResolutionNode` points directly to the summary or model-choice
node that supports its result.

Reading an opinion could provide a more refined tie-breaker. That belongs to a
later opinion-reading stage and is deliberately not coupled to this identity
checkpoint.

## Reading a result

Each `CitationValidation` is an ordered tuple of nodes. Every node has a stable
`node_id`, `status`, typed `outcome`, explicit dependencies, and explanatory
messages. Nodes that use Mellea also retain the model attempts, provider
request/response projection, and requirement failures.

| terminal outcome | meaning |
|---|---|
| `resolved` | a candidate assessment was selected, deterministically or by grounded model choice |
| `no_match` | the model completed its reparse and found that no reviewed candidate represents the citation |
| `deferred` | the current checkpoint deliberately did not decide the locator |

`deferred` is not a negative result. It records a scope boundary: no candidate
was discarded, and later stages can start directly from the serialized document
and trace.

## Deferred work

The following stages require their own contracts before being re-enabled:

- case-name search after an exact reporter lookup miss;
- docket-number-only lookup, which must work without a stated court;
- leaf attribution and leaf case-name consistency;
- reporter-page retrieval, citing-proposition extraction, and pinpoint support.
