---
tags: [validation, courtlistener, citation-identity]
status: active
---

# Validation

Validation currently has one independent, resumable checkpoint: **full reporter-locator identity**. It works only on `FullCaseCitation` roots with a reporter locator. Docket lookup, lookup-miss search, leaf growth, and pinpoint work are separate stages that have not been admitted to this checkpoint.

## API

```python
import asyncio

from mellea_lrc.validation import (
    initialize_full_reporter_locator_identity,
    run_full_reporter_locator_identity,
)

checkpoint = initialize_full_reporter_locator_identity(document)
checkpoint = asyncio.run(run_full_reporter_locator_identity(checkpoint))
```

`initialize_full_reporter_locator_identity` preserves every active citation in source order. `run_full_reporter_locator_identity` returns the same `ValidatedDocument` type. It reads and writes nodes only for full reporter locators; docket locators and every other citation type remain present with an empty validation progression.

The checkpoint round-trips through `serialize_validated_document` and `deserialize_validated_document`. A completed checkpoint is safe to pass back to `run_full_reporter_locator_identity`: completed reporter progressions are retained without another lookup. A partial reporter progression is rejected so one logical lookup cannot be recorded twice.

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
