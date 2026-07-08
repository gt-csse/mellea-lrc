# Shared Re-extraction Workflow

Status: proposed; documentation only

## Decision

Re-extraction should become a shared, stage-independent evidence-repair
capability. Retrieval and assessment may request it for different reasons, but
neither stage should own the mechanism or silently mutate the original
extraction.

The current assessment-only placement no longer matches the workflow:

- retrieval needs two usable parties before it can perform a bounded case-name
  search;
- bookmarked cases show that malformed, partial, and missing case names prevent
  useful retrieval before assessment can begin;
- assessment also needs re-extraction when retrieved evidence shows that the
  initial case name may be wrong or irregular.

This is one grounded recovery operation with multiple consumers, not two
unrelated re-extraction implementations.

## Separation of responsibilities

The shared capability answers only:

> What citation field value is actually present in this local document
> context, and where is it?

It receives document text, an absolute context window, the original extracted
field value (possibly absent), and a reason for the request. It returns a typed
attempt containing:

- status (`accepted`, `empty`, `invalid`, or `failed`);
- the faithfully copied field value when accepted;
- absolute document offsets;
- trigger/reason provenance;
- model/validation diagnostics needed for reproducibility.

The proposal must be grounded in the source text. It must not manufacture a
canonical value or copy a value from CourtListener. The original eyecite
extraction remains immutable.

Each consumer retains its own policy:

- **Retrieval** decides whether the initial evidence is sufficient to form a
  search, whether re-extraction is worth attempting, and which accepted value
  becomes search input. It records search observations without deciding that a
  candidate is the cited case.
- **Assessment** decides whether comparison evidence warrants re-extraction and
  reassesses an accepted value against a particular retrieved candidate.

Thus re-extraction produces evidence; retrieval searches with evidence; and
assessment interprets evidence.

## Proposed flow

```text
original extraction
        |
        +--> retrieval sufficiency gate
        |        |
        |        +-- sufficient --------------------> candidate search
        |        |
        |        +-- missing/partial/malformed
        |                 |
        |                 +--> shared re-extraction
        |                          |
        |                          +-- accepted ----> candidate search
        |                          +-- otherwise ---> typed no-search outcome
        |
        +--> found candidate(s) --> assessment comparison
                                     |
                                     +-- comparison sufficient --> conclusion
                                     |
                                     +-- repair warranted
                                              |
                                              +--> shared re-extraction
                                                       |
                                                       +--> reassessment
```

## Reuse and attempt history

A successful retrieval-stage attempt should be available to assessment. The
assessment stage should not spend another model call recovering the same field
from the same context unless it can state a materially different request.

Re-extraction therefore needs an append-only attempt history rather than one
assessment-specific `followup` slot. An attempt identity should be based on the
document/citation, field, context span, original value, and request purpose—not
on a retrieved candidate's proposed answer. Consumers may reference an earlier
attempt and record how they used it.

Multiple attempts remain possible when evidence genuinely changes, for
example:

- retrieval recovers both parties from a narrow citation window;
- assessment later requests a wider window because the candidate comparison
  reveals that the first window captured a short form;
- a deterministic extractor and a model-backed extractor produce separate,
  auditable attempts.

No attempt replaces the source extraction or erases an earlier attempt.

## Candidate evidence and anchoring

Assessment may know a candidate case name when it requests repair; retrieval
often does not. A generalized API must not require candidate text.

Candidate information may explain the trigger (for example, “initial name did
not match candidate”), but it should not be presented as the desired output.
Validation remains source-grounding plus field-shape validation. Candidate
agreement is a later assessment operation.

This keeps retrieval-stage recovery neutral and reduces the risk that
assessment-stage recovery merely copies CourtListener's wording.

## Package direction

The mechanism and its domain types should eventually move out of
`assessment/fields/case_name/`. The exact package name is deferred until the
contract is designed; plausible homes are a top-level `reextraction/` package
or an extraction-revision package beside `extraction/`.

The shared layer should contain:

- context-window and absolute-span grounding;
- field proposal/result and attempt-provenance types;
- deterministic and model-backed recovery implementations;
- validation and attempt identity.

It should not contain CourtListener query construction, candidate ranking, or
assessment verdicts.

## Migration constraints

When implemented:

1. Preserve the current immutable extraction artifact.
2. Introduce the shared attempt contract before moving behavior.
3. Adapt assessment to consume that contract without changing its verdict
   semantics.
4. Add retrieval-stage triggers for missing, partial, and demonstrably
   malformed case names.
5. Reuse retrieval attempts during assessment where context and purpose are
   equivalent.
6. Serialize attempt provenance explicitly and update the artifact schema.
7. Use bookmarked cases to evaluate trigger precision, recovered grounding,
   search narrowing, and avoided duplicate model calls.

## Questions intentionally left open

- Which deterministic checks should run before a model-backed attempt?
- Is malformed-name detection part of extraction quality assessment or the
  retrieval sufficiency gate?
- Should an accepted revision live on a citation-level evidence ledger or be
  referenced from each consuming stage?
- How wide should the initial and widened context windows be?
- Which request purposes are equivalent enough to reuse one attempt?

These are contract and evaluation questions. They should be resolved against
the bookmark corpus before runtime behavior changes.
