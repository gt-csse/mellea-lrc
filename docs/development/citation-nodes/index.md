# Citation-node execution model

The citation-node model is the new execution substrate for citation-level
reasoning. It starts from existing document artifacts, but it does not mutate
them.

## Why this exists

The older stage-oriented shape is useful for batch inspection:

```text
PreprocessedDocument → ExtractedDocument → later document stages
```

That shape becomes awkward when a small new operation applies to only one
citation. Each new concern tends to become another document-wide layer, and
every layer has to align every citation again.

The citation-node model changes the internal unit of work:

```text
ExtractedDocument
  → CitationNodeDocument
  → CitationNode(...)
  → CitationStep(...)
```

Document artifacts remain stable inputs and projections. Citation nodes carry
execution state and trace.

Operations are intentionally small:

```python
updated_node = operation.run(node)
```

The internal runner can apply an operation to one node or repeat the same
operation independently for every node. This gives us a controlled, typed graph
substrate without adopting LangGraph before the domain state model is settled.

The first slice also includes JSON projection helpers. The current
`citation_nodes.json` artifact can be loaded by the frontend as an
extraction-level review snapshot. Longer term, the node artifact should replace
the document-wide `InferredDocument`, `RetrievedDocument`, and `AssessedDocument`
snapshots by accumulating those operations as per-citation steps.

## Design constraints

- `ExtractedDocument` is not changed by this model.
- Each extracted citation becomes one independent `CitationNode`.
- Node transitions append `CitationStep` records instead of rewriting prior
  stages.
- Step data is JSON-shaped and immutable so traces can become strict snapshots.
- The first implementation is a small typed runner substrate, not LangGraph.
- Steps may carry `step_id`, `depends_on`, and `lane`. These fields are light
  graph metadata: they let the frontend render chains now and leave room for
  fork/join or parallel corpus probes later.

## Near-term path

The first retrieval operation should be exact locator lookup. Later operations
can be added as node steps:

```text
ready
  → exact_lookup
  → fallback_decision
  → case_name_preparation
  → candidate_query
  → corpus_probe(opinions) + corpus_probe(RECAP)
  → candidate_results
  → identity_evaluation
  → proposition_support
```

Exact lookup remains first because it is cheap and gives strong locator-based
evidence. LLM-assisted preparation belongs only after exact lookup cannot find a
candidate.

For the async/e2e path, `case_name_preparation` is now a mandatory LLM-backed
gate before candidate search. Its trace should be read literally:

- `original_case_name` records the parser's prior plaintiff/defendant
  reconstruction when available.
- `llm_status` records the preparation status (`accepted`, `empty`, `failed`,
  or `legacy_deterministic` for the old sync path).
- `llm_classification` records how the model classified the local window.
- plaintiff and defendant must be grounded in the local context before the
  locator.
- `prepared_case_name` is constructed from the validated parties; the model's
  free-form case-name string is not trusted as an independent source.
- `candidate_query` records the query actually sent to CourtListener search.
- each `corpus_probe` records one corpus-specific search outcome, with `lane`
  set to the CourtListener corpus (`o` for opinions, `r` for RECAP).
- `candidate_results` aggregates surfaced summaries only; it does not assert
  identity or proposition support.

This distinction matters: candidate search can tell us that a plausible case
was surfaced, but identity and locator correctness remain later deliberation
nodes.

The synchronous `run_retrieval` API still has a legacy deterministic
preparation fallback so existing deterministic tests and callers remain stable.
Frontend API routes and snapshot regeneration use the async retrieval path,
where preparation is mandatory before not-found candidate search.

## What is deliberately out of scope for this first slice

- Reporter/dataclass migration.
- LangGraph orchestration.
- Proposition-support deliberation.
- Frontend redesign.
- Replacing document-level snapshots immediately.

Those can layer on after the node substrate is stable.

## Snapshot transition

For this branch, `snapshot_corpus.py` still writes the old stage snapshots for
compatibility:

```text
preprocessed.json
extraction.json
citation_nodes.json
inferred.json
retrieval.json
assessment.json
```

This is a transition shape, not the final model. The intended destination is:

```text
preprocessed/extracted source artifact
  → citation_nodes.json with exact lookup, search, identity, and assessment
    represented as node steps
```

That lets the frontend and tests inspect one evolving citation-level trace
instead of loading separate document-wide artifacts for every stage after
extraction.
