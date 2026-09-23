# Extraction

The extraction layer finds full citation locators, reads nearby fields, and forms roots. Its public entry point is [`mellea_lrc.api`](../src/mellea_lrc/api.py). Each stage accepts and returns a `Document`, so a caller can save or inspect the result between stages.

```python
from pathlib import Path

from mellea_lrc.api import Document, grow_roots, stable

document = Document.from_source(Path("filing.pdf"))
document = grow_roots(document, rules=stable())
after_dockets = document.get_stage("docket_locators")
```

`get_stage` returns the exact `Document` produced when that stage committed. The requested stage must appear in `document.stage_runs`. The returned document has the stage-run prefix through that stage, along with the citations and field values present then.

`Document.from_source(Path(...))` preprocesses a file; `Document.from_source("text")` treats a string as the document's text. To choose preprocessing rules, turn the result of `preprocess` into a `Document` before running the stages:

```python
from mellea_lrc.api import Document, grow_roots, preprocess

preprocessed = preprocess(Path("filing.pdf"), rules=[])
document = Document.from_preprocessed(preprocessed)
document = grow_roots(document)
```

`grow_roots` is synchronous and runs the full first pass from a `Document`.

The stages can also be called individually:

```python
from mellea_lrc.api import (
    find_docket_locators,
    find_full_reporter_locators,
    form_roots,
    resolve_case_names,
    resolve_colocations,
    resolve_courts,
    resolve_dates,
    resolve_pin_cites,
)

document = find_full_reporter_locators(document)
document = find_docket_locators(document)
document = resolve_colocations(document)
document = resolve_case_names(document)
document = resolve_courts(document)
document = resolve_dates(document)
document = resolve_pin_cites(document)
document = form_roots(document)
```

The reporter stage uses eyecite's full case citation spans. The docket stage finds explicitly labelled federal CM/ECF numbers such as `Case No. 1:24-cv-00123`; it keeps courtless dockets. Colocation groups nearby locators to set context-reading boundaries. A group does not establish that its members cite the same case. Pin cites are read beside each individual locator. `ExtractionRules` controls colocation distance and context windows; `stable()` supplies the defaults.

`document.full_locators` contains every detected occurrence, and `document.roots` contains the canonical occurrences after root formation. Exact reporter keys can deduplicate repeated reporter locators. Docket keys require both a docket number and a court; a courtless docket remains its own root. `document.colocations` is rebuilt from the citations' colocation histories. `document.stage_runs` lists committed stages in order, including stages that found nothing, so resumed extraction does not repeat them. A stage commits its citation changes together when it completes.

To reconstruct a stage, `get_stage` keeps each citation's nodes and field entries through that stage and omits citations created later. A citation need not have a node for the requested stage: for example, a reporter citation untouched by `docket_locators` keeps its latest earlier node and field entries. This preserves the complete document state at that stage even when the stage changed only other citations.

The [`model/citations/`](../src/mellea_lrc/model/citations/) package defines the citation state. Each reporter or docket locator constructor creates a citation with its first node and initial field entries. Later, `citation.record(stage)` starts one decision. Its named `with_case_name`, `with_court`, `with_date`, `with_pin_cite`, `with_colocation`, and `with_root` methods can record several distinct field changes under that one node. `citation.record(stage).withdraw()` moves the citation to the withdrawn root while retaining its history. The concrete subclasses own their locator readings and expose `locator_span`.

For example, given a case name, court, and their spans in `document.text`, one review can record both readings together:

```python
citation = document.full_locators[0]
revised = (
    citation.record("review")
    .with_case_name(document.text, name, name_span)
    .with_court(document.text, court, court_span)
)
document = document.replace_citation(revised)
```

The two new field entries point to the same node. A later review starts another node with `revised.record("review")`.

Each citation field is an append-only tuple of `FieldUpdate` entries carrying a value, node ID, and optional source span. Literal readings such as a case name or pin cite require a span that exactly matches the document text; normalized fields, such as a court, retain their supporting span when available. The last entry gives the current value; `latest(log)` returns `None` for an empty log. An empty log means no value has been read, while an entry whose value is `None` records an explicit clearing. Every entry points to a node on the same citation. There is no separate current-value copy or operation list. To see how a field changed:

```python
from mellea_lrc.model.citations import latest

citation = document.full_locators[0]
current_name = latest(citation.case_name)
nodes = {node.id: node for node in citation.nodes}
for update in citation.case_name:
    print(update.value, update.span, nodes[update.node_id].stage)
```

This gives the recorded values, source locations, and stages in order for that field. Nodes show which stage made a change; they do not record a reason or a wall-clock timestamp. A citation-wide timeline can be reconstructed from its node order and field entries when needed.

The field logs are the serialized state. The `kind` discriminator restores the concrete reporter or docket type through Pydantic's native JSON round trip:

```python
checkpoint_path = Path("extraction.json")
checkpoint_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
document = Document.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
```

Root identity validation, search, and leaves are later layers; this first pass does not make an identity judgment.
