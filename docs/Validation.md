---
tags: [validation, courtlistener, citation-identity]
status: active
---

# Validation

Validation currently has one independent, resumable stage: **root identity**. It works only on `FullCaseCitation` roots with a reporter locator. Docket lookup, lookup-miss search, leaf growth, and pinpoint work are separate stages that have not been admitted to this route.

## API

```python
import asyncio
from pathlib import Path

from mellea_lrc.api import (
    Document,
    find_docket_locators,
    find_full_reporter_locators,
    resolve_colocations,
    full_reporter_locator_identity,
)

document = Document.from_source(Path("filing.pdf"))
document = find_full_reporter_locators(document)
document = find_docket_locators(document)
document = resolve_colocations(document)
document = asyncio.run(full_reporter_locator_identity(document))
payload = document.serialize()
document = Document.from_serialized(payload)
```

`full_reporter_locator_identity` preserves every citation in source order and returns the same `Document` type. It writes the lookup and candidate evidence to the root citation's `trace`, writes a selected archive result to `found`, writes its pointer to `authority_id`, and writes the terminal state to `judgements[IDENTITY]`. Docket locators and every other citation type remain untouched.

`mellea_lrc.api` is the sole compositional import: it exposes locator readers, the optional docket-hunting plugin, co-location, field readers, and each admitted validation stage. `Document.from_source(...)` starts a locator document from a string or a `Path`; callers with an existing preprocessing result use `Document.from_preprocessed(...)` instead. `Document.serialize()` returns a JSON-ready mapping and `Document.from_serialized(payload)` restores it. Python reserves `from`, so the constructor cannot be named `Document.from(...)`. A completed identity result is safe to pass back to `full_reporter_locator_identity`: its explicit identity judgement prevents a second lookup. A partial root-identity trace is rejected so one logical lookup cannot be recorded twice.

## Current route

```text
full reporter locator
├── exact lookup: one candidate
│   └── field checks → candidate summary → resolved | model selection
├── exact lookup: 2–19 candidates
│   └── field checks for every candidate → candidate summary → resolved | model selection
├── exact lookup: no candidate
│   └── deferred_to_search
└── exact lookup: 20+ candidates, incomplete locator, or lookup failure
    └── deferred_to_future_implementation
```

`deferred_to_search` is a positive handoff: exact locator lookup found no candidate, so the later search stage should work from this document. `deferred_to_future_implementation` means there is no admitted route yet. In particular, an exact result set of 20 or more candidates is preserved on the lookup node but not truncated or sent to the model.

For a bounded candidate set whose deterministic field checks leave zero or multiple matches, the model receives the target-only local context and all reviewed candidates. It reparses the stated case name, court, and date; selects one candidate or emits `no_match`; and cannot change the reporter locator. Colocation is never sent to the model and is not a model-level operation.

A later cross-locator reconciliation stage may use stored colocation along with independently resolved reporter and docket results. Keeping that decision later preserves the potential benefit without making reporter identity depend on an unfinished docket contract.

## Provenance of a model reparse

Every `CitationRecord` has `stated_fields_reparsed_by_model: bool`. It becomes `True` when a grounded model successfully completes a local reparse of stated identity fields. It does not mean that a field was corrected or that identity was resolved. The corresponding `MelleaLocatorCandidateChoiceNode`, including all IVR attempts and repair feedback, remains in the citation's trace.

## Next stages

The following are deliberately separate:

- reporter lookup-miss search, consuming roots judged `deferred_to_search`;
- docket-number-only lookup and disambiguation, consuming untouched `DocketCitation` records, including citations without a court;
- cross-locator reconciliation after both identity routes have evidence;
- leaf attribution and leaf case-name consistency;
- reporter-page retrieval, proposition extraction, and pinpoint support.
