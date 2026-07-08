# Prefix-Cached Document Reasoning

Status: proposed; documentation only

## Decision direction

Citation-level model work should share one stable document prefix instead of
receiving independently constructed local-context prompts. The prefix contains
the full preprocessed document, followed by one grounded document inference.
Each citation node forks from that common prefix.

```text
stable instructions
        + full preprocessed document
        + grounded document-inference request
        + materialized document inference
        = shared prefix
                    |
                    +--> citation node A
                    +--> citation node B
                    +--> citation node C
```

The full document preserves evidence that a summary may omit. The inference
provides a consistent interpretation shared by all citation nodes. Prefix/KV
caching makes repeated attention over the unchanged prefix substantially more
efficient than rebuilding and prefilling it for every citation.

## Why both the document and an inference belong in the prefix

A document inference alone is lossy. It is useful for stable facts such as the
document type, issuing or target court, authoring party, procedural posture,
claims, and major argumentative structure, but it cannot anticipate every
citation-specific question.

The full document alone preserves evidence but allows each citation call to
independently reinterpret the document. That can produce inconsistent views of
the same party role, issue, or procedural posture.

The shared prefix therefore keeps both:

1. the source text as the ultimate evidence;
2. a grounded, reusable interpretation that regulates later reasoning.

Citation nodes may inspect the source when the inference is insufficient, but
they should treat the same materialized inference as their common starting
point.

## Node-level execution model

The orchestration graph should control the boundary exactly:

1. Preprocess and freeze the document text.
2. Construct a canonical instruction-and-document prefix.
3. Run document inference once.
4. Validate and materialize its structured result with source spans.
5. Append that result to the canonical prefix.
6. Fork one child node per citation or citation/candidate pair.
7. Add only node-specific evidence and instructions after the fork.

The materialized inference is a pipeline artifact, not an implicit model memory.
It must be serializable, inspectable, and reusable when a run is resumed.

This graph shape also makes concurrency safe: sibling citation nodes read the
same immutable prefix and cannot alter one another's context.

In a later cycle, these child nodes may become independently persisted
per-citation reasoning graphs. That direction is documented in
[Per-Citation Execution Model](./per-citation-execution-model.md). The current
cycle retains document-stage artifacts and snapshots.

## Cache contract

Provider prompt caching is an optimization beneath the semantic contract. A run
must remain correct when the cache is cold or unavailable.

Effective KV/prefix reuse generally requires:

- identical model and model configuration;
- byte/token-identical prefix ordering and content;
- stable instructions before all citation-specific material;
- no timestamps, random IDs, citation indices, or other volatile data in the
  shared prefix;
- provider/session routing that can reuse the cached prefix;
- citation-specific prompts appended only after the shared boundary.

The node layer should compute a prefix identity from the model configuration,
prompt-contract version, exact preprocessed text, and exact materialized
document inference. That identity is provenance and a cache key hint; it is not
proof that a provider retained its KV state.

Prefix caching reduces repeated prefill latency and cost where supported. It
does not remove the document from the model's context window, guarantee cache
retention, or make an oversized document fit. Very large documents still need a
separate long-context or hierarchical strategy.

## Grounded document inference

Every inferred fact should carry supporting absolute spans into the
preprocessed document. The first contract should remain small and stable:

- document genre and authoring role;
- issuing court or target court and jurisdiction;
- procedural posture;
- primary claims/issues;
- requested relief;
- major sections or argument structure.

Unknown and conflicting evidence must remain explicit. The inference should not
decide whether any cited authority is real, matched, or appropriately used.

The inference contract needs its own version. Changing that version changes the
shared prefix identity and intentionally invalidates reuse.

## Relationship to other layers

- Extraction identifies citation text and fields.
- Shared re-extraction repairs grounded citation evidence when needed.
- Retrieval finds external candidate records and decisional documents.
- Assessment determines whether retrieved evidence identifies the cited
  authority.
- Appropriateness evaluation, proposed separately, determines whether the
  authority supports its use in this document.

Document inference supplies common context to these consumers but does not
collapse their conclusions into one verdict.

## Evaluation questions

Before implementation, measure on the bookmark corpus and larger documents:

- cached versus uncached prefill latency and token cost;
- prefix-cache hit rate under concurrent citation fan-out;
- consistency of document-level facts across citation nodes;
- whether full-document access improves decisions over local windows alone;
- whether the inference anchors later nodes incorrectly when its evidence is
  weak;
- context-window headroom left for candidate documents and deliberation.

The architecture is successful only if it improves efficiency without hiding
source evidence or coupling correctness to a cache hit.
