---
tags: [validation, courtlistener, citation-identity]
status: active
---

# Validation

Validation currently has one independent, resumable checkpoint: **full reporter-locator identity**. It works only on `FullCaseCitation` roots with a reporter locator. Docket lookup, lookup-miss search, leaf growth, and pinpoint work are separate stages that have not been admitted to this checkpoint.

## API

```python
import asyncio
from pathlib import Path

from mellea_lrc.api import (
    Document,
    ValidatedDocument,
    find_docket_locators,
    find_full_reporter_locators,
    resolve_colocations,
    run_full_reporter_locator_identity,
    start_full_reporter_locator_identity,
)

document = Document.from_source(Path("filing.pdf"))
document = find_full_reporter_locators(document)
document = find_docket_locators(document)
document = resolve_colocations(document)
checkpoint = start_full_reporter_locator_identity(document)
checkpoint = asyncio.run(run_full_reporter_locator_identity(checkpoint))
payload = checkpoint.serialize()
checkpoint = ValidatedDocument.from_serialized(payload)
```

`start_full_reporter_locator_identity` preserves every active citation in source order. `run_full_reporter_locator_identity` returns the same `ValidatedDocument` type. It reads and writes nodes only for full reporter locators; docket locators and every other citation type remain present with an empty validation progression.

`mellea_lrc.api` is the sole compositional import: it exposes locator readers, the optional docket-hunting plugin, co-location, field readers, and each admitted validation checkpoint. `Document.from_source(...)` starts a locator document from a string or a `Path`; callers with an existing preprocessing result use `Document.from_preprocessed(...)` instead. `Document.serialize()` and `ValidatedDocument.serialize()` return JSON-ready mappings. Their paired constructors are `Document.from_serialized(payload)` and `ValidatedDocument.from_serialized(payload)`; Python reserves `from`, so the constructor cannot be named `Document.from(...)`. A completed checkpoint is safe to pass back to `run_full_reporter_locator_identity`: completed reporter progressions are retained without another lookup. A partial reporter progression is rejected so one logical lookup cannot be recorded twice.

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

`deferred_to_search` is a positive handoff: exact locator lookup found no candidate, so the later search stage should work from this artifact. `deferred_to_future_implementation` means there is no admitted route yet. In particular, an exact result set of 20 or more candidates is preserved on the lookup node but not truncated or sent to the model.

For a bounded candidate set whose deterministic field checks leave zero or multiple matches, the model receives the target-only local context and all reviewed candidates. It reparses the stated case name, court, and date; selects one candidate or emits `no_match`; and cannot change the reporter locator. Colocation is never sent to the model and is not a model-level operation.

A later cross-locator reconciliation stage may use stored colocation along with independently resolved reporter and docket results. Keeping that decision later preserves the potential benefit without making reporter identity depend on an unfinished docket contract.

## Provenance of a model reparse

Every `CitationRecord` has `stated_fields_reparsed_by_model: bool`. It becomes `True` when a grounded model successfully completes a local reparse of stated identity fields. It does not mean that a field was corrected or that identity was resolved. The corresponding `MelleaLocatorCandidateChoiceNode`, including all IVR attempts and repair feedback, remains in the same serialized checkpoint.

## Next checkpoints

The following are deliberately separate:

- reporter lookup-miss search, consuming progressions marked `deferred_to_search`;
- docket-number-only lookup and disambiguation, consuming untouched `DocketCitation` progressions, including citations without a court;
- cross-locator reconciliation after both identity routes have evidence;
- leaf attribution and leaf case-name consistency;
- reporter-page retrieval, proposition extraction, and pinpoint support.
