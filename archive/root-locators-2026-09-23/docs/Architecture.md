# Source layout and public boundaries

The repository is organized by responsibility, with `mellea_lrc.api` as the
single outer composition surface. A caller passes a `Document` to a named
stage, receives a `Document`, and may serialize it before continuing.

| Package | Responsibility |
| --- | --- |
| `model/` | `Document`, preprocessing contracts, citation kinds and fields, record state, nodes, append-only citation operation history, stage-neutral operations, locator projections, and shared quote grounding |
| `preprocessing/` | Convert source files to indexed text without changing citation semantics |
| `extraction/reading/` | Read locator-adjacent case names, courts, dates, and pin cites from the filing |
| `extraction/structure/` | Derive colocations, roots, and leaf attachments from admitted citations |
| `extraction/adjudication/` | Optional model-assisted site hunting and extraction reviews |
| `validation/root_identity/` | Resolve reporter and docket roots, including body corroboration and optional open-web review |
| `validation/search/` | Retrieve CourtListener and GovInfo metadata candidates for unresolved roots |
| `validation/candidates/` | Evaluate, select, and track retrieved candidates |
| `validation/field_checks/`, `validation/pinpoint_retrieval/` | Compare fields and check page-level claims |
| `validation/types.py`, `validation/execution.py`, `validation/pipeline.py` | Typed validation contracts and orchestration |
| `courtlistener/`, `govinfo/` | Service clients and response models |
| `llm/` | Model sessions, IVR execution, and grounded text matching |
| `serialization/` | Stage artifact persistence and the separate validated-output format; `Document` uses native Pydantic checkpoint methods |
| `evaluations/`, `scripts/` | Experiments and explicit execution sequences; neither defines citation state |

`Document`, `CitationRecord`, and their operations are in `model/`. The model
owns state and pure projections; preprocessing, extraction, and validation own
the stages that read or change it. `Document.from_source` and its serialization
methods are convenience adapters at the outer boundary, with local imports to
avoid reversing package dependencies at import time. A validation node and an
extraction node use the same operation contract.

The public stage signatures are `Document -> Document` (async where retrieval
or a model is involved). For example:

```python
document = Document.from_source(source)
document = find_full_reporter_locators(document)
document = find_docket_locators(document)
document = await hunt_docket_locators(document)  # optional
document = resolve_colocations(document)
document = resolve_case_names(document)
document = resolve_courts(document)
document = resolve_dates(document)
document = resolve_pin_cites(document)
document = form_roots(document)
```

`grow_roots(document, hunt_dockets=True)` composes those extraction stages in
that order. The identity stages remain individually callable; the explicit
sequence in `scripts/run_root_identity_pipeline.py` decorates each stage with
`@serialize()` for reviewable artifacts. Root identity finishes before
`grow_leaves(document)`. Optional `hunt_leaf_case_names(document)` then reviews
unread bare names, and `validate_pin_cite_sites(document)` reviews uncertain
page claims on the complete root-and-leaf set. These are separate checkpoints;
neither is part of deterministic leaf growth. Open-web search remains a
separate optional stage.

A node reports a reading or decision; an operation materializes its effect.
`create_citation` sets only citation kind, `update_field`/`update_fields` write
the first or later filing-field values, and `withdraw_citation` retains a
rejected record. Judgements, retrieved resolutions, authority links, and
observations also use stage-neutral operations. Root and antecedent links use
separate operations. A single node may support several operations, each with
its own typed resulting-value event and evidence pointer in the serializable
record. Withdrawal is another root-link operation, attaching a root and its
leaves to the reserved virtual head while retaining their prior attachments.
Document-level creation and later mutable readings have their own typed event
stream. Earlier values can be replayed from prior events. Colocation
is an internal rule-derived link; it is not an instruction or operation offered
to a model. The public API snapshots its input at each stage boundary and
checks that earlier history is a prefix of the result, so an earlier in-memory
`Document` remains an independent checkpoint.

See [Document.md](Document.md) for record invariants and checkpoint behavior.
