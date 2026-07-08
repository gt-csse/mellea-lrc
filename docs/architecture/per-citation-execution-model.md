# Per-Citation Execution Model

Status: deferred direction; not for the current development cycle

## Motivation

The current pipeline advances the whole document through additive stages:
extraction, inference, retrieval, and assessment. This is useful for examining
one stage at a time and produces clear document snapshots.

As citation reasoning grows, however, a small citation-local capability can
force a new document-wide stage, artifact layer, serializer change, API path,
and snapshot boundary. That cost will become disproportionate when one citation
may invoke many conditional model nodes while another needs only deterministic
work.

The longer-term direction is therefore to treat each extracted citation as an
independent reasoning unit after extraction.

## Proposed shape

```text
preprocess document
        |
extract all citations
        |
shared document prefix + document inference
        |
        +--> citation A reasoning graph
        |       extraction repair? -> retrieval -> identity -> later evaluation
        |
        +--> citation B reasoning graph
        |       deterministic lookup -> identity
        |
        +--> citation C reasoning graph
                alternate retrieval -> repair -> identity
        |
aggregate citation records into a document view
```

Each citation graph develops its own append-only reasoning chain. Node selection
depends on that citation's state and evidence rather than on a requirement that
every citation pass through every document-wide phase.

This would make it easier to add a narrowly scoped model operation, retry,
repair, retrieval leg, or deliberation step without inventing another global
document stage.

## Shared versus citation-local state

The model does not imply isolated citation prompts with no document context.
Citation graphs should fork from the shared full-document prefix and grounded
document inference described in
[Prefix-Cached Document Reasoning](./prefix-cached-document-reasoning.md).

Shared immutable state includes:

- source and preprocessed text;
- extraction output and citation identities;
- grounded document inference;
- model/prompt contract versions.

Citation-local state includes:

- extraction or re-extraction attempts;
- retrieval probes and candidates;
- authority-identity reasoning;
- candidate-specific assessment;
- later appropriateness deliberation;
- node execution provenance and failures.

The document becomes an aggregate/read model over independent citation records,
not the unit that dictates every reasoning transition.

## Why this is deferred

The current cycle still benefits from strict document-stage snapshots:

- they make extraction, inference, retrieval, and assessment easy to inspect in
  isolation;
- they expose contract changes clearly;
- the bookmark corpus is still developing the evidence needed to design
  citation-local transitions;
- changing storage and orchestration now would combine architectural migration
  with active retrieval research.

Accordingly, the current artifact inheritance chain and snapshot structure
remain canonical for this cycle. New work should continue to fit those stages
unless doing so would distort the domain boundary; this document is not
authorization to begin incremental migration.

## Questions for a later design cycle

- What is the stable identity and persistence unit for a citation reasoning
  chain?
- Is the chain an event log, a typed node DAG, or both?
- How are candidate fan-out and joins represented?
- Which citation nodes may execute concurrently?
- How does a document aggregate express partial completion?
- How are old per-stage snapshots migrated or retained as projections?
- Which node outputs affect shared document inference, if any, without making
  sibling chains order-dependent?

These questions should be answered only after the current snapshot-based cycle
has produced enough varied bookmarked cases.
