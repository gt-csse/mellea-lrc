# Document Artifact Model

Status: accepted

## Context

The document pipeline is monotonic: preprocessing establishes text and provenance,
extraction adds citations, validation adds lookup results, and assessment adds
cross-reference judgments and re-extraction history. The previous top-level result
types represented that lifecycle inconsistently: `PreprocessedDocument` was wrapped
by `DocumentExtraction`, while later stages copied the preprocessing, citation, and
validation fields into new peer classes.

That structure obscured the fact that every later artifact is also a valid earlier
artifact, duplicated forwarding properties, and forced adapters to reconstruct
earlier result objects for serialization and presentation.

## Decision

Document artifacts use an immutable, additive inheritance chain:

```text
DocumentBase
└── PreprocessedDocument
    └── ExtractedDocument
        └── ValidatedDocument
            └── AssessedDocument
```

The classes contain data and read-only queries only. Pipeline behavior remains in
pure transition functions:

```text
run_preprocessing(source)          -> PreprocessedDocument
run_extraction(preprocessed)       -> ExtractedDocument
run_validation(extracted)          -> ValidatedDocument
run_assessment(validated)          -> AssessedDocument
```

External operations such as document conversion, CourtListener lookup, and LLM
calls do not become methods on artifact classes.

### Field ownership

Real stage outputs remain direct fields. Metadata records contain provenance and
execution context only; they do not hide text, citations, validations, or
assessments.

- `DocumentBase` owns `source_metadata`, containing source identity and provenance
  (`path`, original `format`, source `header`, and source-specific `extras`).
- `PreprocessedDocument` adds the real `text` output and
  `preprocessing_metadata`, containing preprocessing backend provenance.
- `ExtractedDocument` adds the real `citations` output and
  `extraction_metadata`, containing extraction backend provenance.
- `ValidatedDocument` adds one real `validations` outcome for every citation and
  `validation_metadata`, containing lookup client mode and source.
- `AssessedDocument` adds real assessment execution states, modified-extraction
  and reassessment history, plus `assessment_metadata`, containing assessment-run
  execution provenance.

Each later artifact inherits all earlier fields. A transition copies references to
immutable earlier-stage values and allocates only the fields introduced by the new
stage.

### Invariants

- Preprocessed text is non-empty.
- Citation identifiers are non-empty and unique within a document.
- Citation spans are contained by the preprocessed text.
- `resolves_to` refers to another citation in the same document.
- Validation identifiers exactly match extracted citation identifiers.
- Assessment identifiers exactly match extracted citation identifiers.
- Assessment execution states form a tagged union: `waiting`, `skipped`,
  `assessed`, or `failed`. Each state carries only the data valid for that state.
- Modified-citation and reassessment identifiers refer to citations in the same
  document and are unique within their respective collections.
- Nested case-name and year assessments use the same identifier as their parent
  citation assessment.
- Re-extraction is append-only: original extracted citations are never mutated or
  replaced.

Assessment initialization marks eligible, found full-case citations as `waiting`
and all ineligible citations as `skipped` with a structured reason. Execution moves
each `waiting` record to either `assessed` with a substantive result or `failed`
with an error. `assessment_complete` is derived: it is true exactly when no record
remains `waiting`.

| Execution state | State-specific payload |
|---|---|
| `waiting` | Citation identity only |
| `skipped` | Structured reason and message |
| `assessed` | Required case-name and year assessment result |
| `failed` | Required error detail |

These are execution states. Domain conclusions such as `semantic_match` or
`mismatch` remain inside the substantive result and cannot be confused with whether
assessment execution succeeded.

### Identity

Citation identifiers are deterministic within an exact extracted artifact and are
assigned by citation order. Re-running extraction on identical text therefore
produces stable identifiers. They are document-local identifiers, not globally
stable legal-citation identifiers.

### Serialization

Serialized artifacts remain flat for interoperability and require:

- `schema_version`
- `artifact_type`

Unversioned artifacts and mismatched artifact types are rejected. Deserialization
validates the artifact invariants rather than silently constructing contradictory
stage objects. Schema version 2 exposes the same stage ownership explicitly with
`source_metadata`, `preprocessing_metadata`, `extraction_metadata`,
`validation_metadata`, and `assessment_metadata` keys as applicable. Version 1 is
not accepted.

### Breaking-change policy

The project is still pre-stability, so this refactor is an intentional clean break.
Only the canonical names are exported:

- `PreprocessedDocument`
- `ExtractedDocument`
- `ValidatedDocument`
- `AssessedDocument`

The previous `DocumentExtraction`, `DocumentValidation`, and `DocumentAssessment`
types are removed without compatibility adapters. Artifact readers require the
current explicit schema version and do not infer stages from payload keys.

## Alternatives considered

### Nested composition

Each stage could wrap the previous stage. This makes boundaries explicit but
creates progressively nested access and still requires forwarding or flattening at
most API boundaries. It also does not reflect the desired substitutability of later
artifacts for earlier read-only consumers.

### One document with optional stage fields

A single class with optional citations, validations, and assessments was rejected
because it permits many invalid combinations and makes stage requirements runtime
conventions instead of type-level contracts.

## Consequences

Later artifacts can be consumed safely by earlier-stage read-only functions. Stage
transitions allocate new objects and preserve prior tuples and records. Constructors
and deserializers become responsible for enforcing cross-stage consistency. Python
and serialized artifact consumers migrate directly to the canonical contract while
the project remains pre-stability.
