# Document Artifact Model

Status: accepted

This document-stage model remains canonical for the current development cycle.
A possible future per-citation orchestration model is recorded, but explicitly
deferred, in
[Per-Citation Execution Model](./per-citation-execution-model.md).

## Context

The document pipeline is monotonic: preprocessing establishes text and provenance,
extraction adds citations, retrieval adds lookup results, and assessment adds
cross-reference judgments and re-extraction history. The previous top-level result
types represented that lifecycle inconsistently: `PreprocessedDocument` was wrapped
by `DocumentExtraction`, while later stages copied the preprocessing, citation, and
retrieval fields into new peer classes.

That structure obscured the fact that every later artifact is also a valid earlier
artifact, duplicated forwarding properties, and forced adapters to reconstruct
earlier result objects for serialization and presentation.

## Decision

Document artifacts use an immutable, additive inheritance chain:

```text
DocumentBase
└── PreprocessedDocument
    └── ExtractedDocument
        └── RetrievedDocument
            └── AssessedDocument
```

The classes contain data and read-only queries only. Pipeline behavior remains in
pure transition functions:

```text
run_preprocessing(source)          -> PreprocessedDocument
run_extraction(preprocessed)       -> ExtractedDocument
run_retrieval(extracted)          -> RetrievedDocument
run_assessment(retrieved)          -> AssessedDocument
```

External operations such as document conversion, CourtListener lookup, and LLM
calls do not become methods on artifact classes.

### Field ownership

Real stage outputs remain direct fields. Metadata records contain provenance and
execution context only; they do not hide text, citations, retrievals, or
assessments.

- `DocumentBase` owns `source_metadata`, containing source identity and provenance
  (`path`, original `format`, source `header`, and explicit source `extra_data`).
- `PreprocessedDocument` adds the real `text` output and
  `preprocessing_metadata`, containing preprocessing backend provenance.
- `ExtractedDocument` adds the real `citations` output and
  `extraction_metadata`, containing extraction backend provenance.
- `RetrievedDocument` adds one real `retrievals` outcome for every citation and
  `retrieval_metadata`, containing lookup client mode and source.
- `AssessedDocument` adds one real citation-assessment execution state per citation,
  plus `assessment_metadata`, containing assessment-run execution provenance.
  Completed citation assessments own their field results and case-name follow-up.

Each later artifact inherits all earlier fields. A transition copies references to
immutable earlier-stage values and allocates only the fields introduced by the new
stage.

### Invariants

- Preprocessed text is non-empty.
- Citation identifiers are non-empty and unique within a document.
- Citation spans are contained by the preprocessed text.
- `resolves_to` refers to another citation in the same document.
- Retrieval identifiers exactly match extracted citation identifiers.
- Assessment identifiers exactly match extracted citation identifiers.
- Assessment execution states form a tagged union: `waiting`, `skipped`,
  `assessed`, or `failed`. Each state carries only the data valid for that state.
- Citation identity exists only on the document-level assessment record. Nested
  case-name and year results do not duplicate it.
- A re-extracted case name contains only its case-name value and required
  document-local `case_name_span`; it is not represented as a modified citation.
- Re-extraction is append-only: original extracted citations are never mutated or
  replaced.
- Semantic external data is converted into typed immutable domain records before
  entering an artifact. CourtListener responses use `CourtListenerCitationRecord`;
  retrieval wraps each retrieved record in `RetrievedCandidate` with a stable
  `candidate_id` and candidate-scoped provenance. Lookup
  diagnostics use `RetrievalFailureDetail`, and assessment conversations use
  `ChatTurn`.
- Unknown external fields are preserved only in an explicitly named `ExtraData`
  field. `ExtraData` is defensively copied and deeply frozen, so domain records
  cannot be mutated indirectly through retained input dictionaries or nested lists.
- Found and ambiguous retrieval share the same `RetrievedCandidate` type.
  Assessment refers to it by `candidate_id` rather than duplicating the external
  record. Array position is presentation order, not identity.

### External boundaries

Untrusted CourtListener payloads are retrieved by strict Pydantic transport models
and then converted into plain immutable dataclasses. Transport models do not become
the document domain model. They reject type coercion while accepting newly added
upstream fields; those unknown fields are collected into `extra_data` instead of
being discarded or mixed with modeled fields. LLM structured outputs use the same
boundary-retrieval principle.

Internally initiated stage objects remain dataclasses. This keeps external parsing,
schema adaptation, and domain invariants as separate responsibilities.

Mellea assessment calls use the project-owned direct IVR wrapper in
`mellea_lrc.llm.ivr`. Domain modules provide compact instructions and explicit
Pydantic validation as the first requirement; Mellea owns retry/repair, while the
domain layer owns parsing and invariants. Structured-output or requirement
failures propagate into explicit assessment failure states. Re-extraction uses
the same IVR path with an additional deterministic grounding requirement against
the document context.

Assessment initialization marks eligible, found full-case citations as `waiting`
and all ineligible citations as `skipped` with a structured reason. Execution moves
each `waiting` record to either `assessed` with a substantive result or `failed`
with an error.

| Execution state | State-specific payload |
|---|---|
| `waiting` | Citation identity only |
| `skipped` | Structured reason and message |
| `assessed` | Required case-name run and year assessment result |
| `failed` | Required error detail |

These are execution states. Domain conclusions such as `semantic_match` or
`mismatch` remain inside the substantive result and cannot be confused with whether
assessment execution succeeded.

Case-name conclusions are likewise conclusion-only: `exact_match`,
`semantic_match`, `not_semantic_match`, or `unassessable`. Workflow markers such
as “needs assessment” and re-extraction failure are not case-name conclusions.
Work that has not run remains `waiting`; re-extraction and reassessment failures
are represented by their corresponding reassessment execution states. Legacy
snapshots may still deserialize `different_case` and `irregular_form`, but the
active validation path no longer emits those lawyer-facing subclassifications.

Case-name reassessment is field-local and nested within a completed citation
assessment. The initial assessment and follow-up outcome therefore cannot become
detached from each other or from the owning citation record.

| Case-name follow-up state | State-specific payload |
|---|---|
| `not_required` | No additional payload |
| `reassessed` | Required grounded `reextracted_case_name` and substantive result |
| `reextraction_failed` | Required error detail; no grounded case name exists |
| `reassessment_failed` | Required grounded `reextracted_case_name` and error detail |

Skipped and failed citation assessments have no synthetic case-name follow-up;
their document-level state already determines that field work did not run. A
successful reassessment contains only the new case-name conclusion and does not
copy the unchanged year assessment. `assessment_complete` is derived and is true
exactly when no citation assessment remains `waiting`.

`assessment_metadata` records the effective Mellea concurrency when applicable.
It does not persist a `mellea_calls` counter; that redundant ordinal was removed
because it was easy to misread during concurrent debugging.

### Identity

Citation identifiers are deterministic within an exact extracted artifact and are
assigned by citation order. Re-running extraction on identical text therefore
produces stable identifiers. They are document-local identifiers, not globally
stable legal-citation identifiers.

### Serialization

Serialized artifacts retain explicit top-level stage fields and require:

- `schema_version`
- `artifact_type`

Unversioned artifacts, previous schema versions, and mismatched artifact types are
rejected. Deserialization validates artifact invariants rather than silently
constructing contradictory stage objects. Schema version 19 uses one nested
`request_trace` contract for CourtListener HTTP status, cache outcome, request key,
and error metadata. Schema changes always increment the version; deserializers do
not adapt previous versions.

Every public artifact deserializer first validates a strict Pydantic transport DTO
configured with `extra="forbid"` and no type coercion. Citation kinds and assessment
execution states use discriminated transport unions, so state-inappropriate fields
are rejected at the boundary. The retrieved DTO is then converted into immutable
domain dataclasses; Pydantic models never enter the document inheritance chain.

Serialized artifacts contain source-of-truth domain data only. Counts, completion
flags, and status summaries are derived by consumers and are not duplicated in the
artifact schema.

### Breaking-change policy

The project is still pre-stability, so this refactor is an intentional clean break.
Only the canonical names are exported:

- `PreprocessedDocument`
- `ExtractedDocument`
- `RetrievedDocument`
- `AssessedDocument`

Superseded domain type names are not exported. Artifact readers require an explicit
supported schema version and do not infer stages from payload keys.

## Alternatives considered

### Nested composition

Each stage could wrap the previous stage. This makes boundaries explicit but
creates progressively nested access and still requires forwarding or flattening at
most API boundaries. It also does not reflect the desired substitutability of later
artifacts for earlier read-only consumers.

### One document with optional stage fields

A single class with optional citations, retrievals, and assessments was rejected
because it permits many invalid combinations and makes stage requirements runtime
conventions instead of type-level contracts.

## Consequences

Later artifacts can be consumed safely by earlier-stage read-only functions. Stage
transitions allocate new objects and preserve prior tuples and records. Constructors
and deserializers become responsible for enforcing cross-stage consistency.
Pydantic transport models validate external payloads before conversion, while
serializers produce ordinary JSON objects without exposing mutable references to
the artifact's internal values. Python and serialized artifact consumers migrate
directly to the canonical contract while the project remains pre-stability.
