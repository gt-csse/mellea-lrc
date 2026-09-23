---
tags: [record, artifact, extraction, validation]
status: active
---

# Document and citation operations

Every public stage takes a `Document` and returns a `Document`. The document
contains the original text, citation records, document-level findings, and the
names of completed passes. `Document` is a Pydantic model. Save any stage result
with `document.model_dump(mode="json")`; restore it with
`Document.model_validate(payload)` and continue at the next stage. For a JSON
string, use `document.model_dump_json()` and `Document.model_validate_json(raw)`.
The current checkpoint schema is 21. A checkpoint is self-contained; no
earlier artifact or custom document codec is needed to resume.

## Citation state

`CitationRecord` has one current `fields` object. Its class fixes the
`CitationKind` at creation. A case name, court, date, pin cite, or locator read
from the filing lives in `fields`; there are no parallel original and corrected
field sets. `operations` is the ordered, append-only history of its creation
and every durable change. Each event records its resulting value and supporting
node ID. The prior value follows from the preceding events; `field_updates`
derives a field-only before/after view when needed. A
caller can read a current value with `record.get_field(CitationField.CASE_NAME)`;
typed conveniences such as `record.case_name` use that same history-backed
read. The `fields` projection is checked against the operations when a
checkpoint is loaded, so these views cannot disagree.

Other durable state answers different questions:

| State | Meaning |
| --- | --- |
| `root_id`, `resolves_to`, `colocation_id` | Filing-internal relationships and locator grouping |
| `root_link_node_id` | Evidence for the current root attachment |
| `found`, `authority_id` | Retrieved identity and authority |
| `judgements` | Current answer to each `Question`, with its deciding node |
| `trace` | Readings, retrievals, and model decisions supporting the above |
| `operations` | Ordered changes to the current state, each linked to a trace node |

The record keeps a `created_by` pointer to the node that classified the
citation. A root points to its own citation ID; a leaf points to its root.
Withdrawal appends a root-link operation that reattaches the citation to the
reserved virtual head `__withdrawn__`. The prior attachment remains in the
operation history, and `root_link_node_id` points directly to the evidence for
the current one. A withdrawn citation remains in `document.citations`; only
`document.active_citations` excludes it. Withdrawing a formed root moves its
leaves to the same head in one document transition.

## Node and operation

A `Node` records what was read or retrieved, who did it, its result, and any
stage-owned details. It is evidence, not the state change itself. One model
review can justify no change, one field update, several field updates, a
resolution, and a judgement without inventing extra model calls. Every
durable change appends a typed `CitationOperation` to the record; a no-change
reading remains in `trace` without an operation.
Source re-extraction follows that rule too: a model's copied text is grounded
to a source span, and an admitted correction uses `update_field` or
`update_fields`. Its typed review node remains in the trace as the operation's
evidence. A temporary validation result is never a substitute for the current
field value.
Node details are JSON value trees; stage-owned objects must be projected into
those trees before insertion so native checkpoints can restore them exactly.

The stage-neutral mutation API in `mellea_lrc.model.operations` begins with:

```python
record = create_citation(citation_id, CitationKind.FULL_CASE, reading_node)
record = update_fields(record, reading_node, parsed_fields, reason="Initial reading")
record = assign_root(record, record.citation_id, reading_node)
```

`create_citation` initializes only kind and evidence. The first locator read is
an `update_fields` operation, like any later correction. A leaf receives its
root through `assign_root` before it enters a `Document`. `withdraw_citation`
reattaches an individual record to the virtual head; `withdraw_subtree` moves
a formed root and its leaves together. Field names are
`CitationField` enum members, and a batch of field changes is validated before
any of them is written.

`Reads.DOCUMENT` nodes may update what the filing says. `Reads.RECORD` nodes
may settle what an archive found. A document-level rejected site is a finding
with a document-level node, not an invented citation. `observe_document`
records that node by returning a revised `Document`. CitationRecord's internal
mutation methods are private; stages use the operation functions for durable
field, graph, resolution, judgement, and withdrawal changes.

## Stage boundaries

The public composition boundary is `mellea_lrc.api`. The root path is:

```python
document = Document.from_source(source)
document = await grow_roots(document, hunt_dockets=True)
document = await validate_roots_identity(document)
document = await grow_leaves(document)
document = await hunt_leaf_case_names(document)      # optional
document = await validate_pin_cite_sites(document)   # optional
```

`grow_roots` runs locator discovery and optional docket hunting before one
colocation pass, then reads case names, courts, dates, and pin cites, and forms
roots. The named identity stages can also be called individually and serialized
after each step. `grow_leaves` runs after root formation; identity validation
is optional for structural evaluation. Each stage advertises its own pass name
so an empty result can be distinguished from a stage that never ran.

Colocation is a low-level proximity aid for field reading and checking. It
does not decide that neighboring identifiers name one authority, and model
prompts do not expose it as an operation.

## Checkpoint independence

The native Pydantic dump saves current state, the full citation operation log,
its evidence, and the document-level event stream. Document creation is the
first document event; later source-metadata and unread-name changes append
typed events. A checkpoint after any small stage can resume without prior
checkpoint files. Native reload replays the citation and document histories
and rejects a current projection that disagrees with them.

`Document.evolve(...)`, public stages, and `@serialize()` enforce a cumulative
transition: earlier citation IDs, operations, trace nodes, document nodes,
findings, passes, and document events cannot be removed or rewritten. The
document text and its coordinate system also remain fixed. Public stages work
on independent snapshots, so an earlier in-memory `Document` remains usable
after a later stage returns. Native dumps verify the same history before
writing; the checkpoint is a complete save, not a delta against an earlier
file.
