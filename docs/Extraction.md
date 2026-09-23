# Extraction

The active extraction layer reads full citation locators and their nearby fields, then forms roots. Its public boundary is [`mellea_lrc.api`](../src/mellea_lrc/api.py). Every individual stage accepts and returns a `Document`, so a caller can save or inspect the result after any stage.

```python
from pathlib import Path

from mellea_lrc.api import Document, grow_roots, stable

document = Document.from_source(Path("filing.pdf"))
document = await grow_roots(document, rules=stable())
```

`Document.from_source(Path(...))` preprocesses a file. `Document.from_source("text")` treats a string as the document's text. If preprocessing needs selected layout rules, use `preprocess(source, rules=...)` and pass its `PreprocessedDocument` to `start_extraction`:

```python
from mellea_lrc.api import preprocess, start_extraction

preprocessed = preprocess(Path("filing.pdf"), rules=[])
document = start_extraction(preprocessed)
```

For the default preprocessing and extraction rules together, `await extract(source)` is the one-call convenience API.

The first pass can also be called one stage at a time:

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

The reporter stage uses eyecite's full case citation spans. The docket stage reads explicitly labelled federal CM/ECF numbers, such as `Case No. 1:24-cv-00123`; it leaves courtless dockets intact. Other docket conventions belong to a later hunting stage. Colocation runs once after locator discovery. It groups nearby locator occurrences to bound case-name reading before the group and court/date reading after it. A group helps parse context; it does not prove that its members cite the same case. Pin cites are read next to each individual locator.

`document.full_locators` contains every detected occurrence with its exact text span. `document.roots` contains canonical occurrences after root formation. Exact reporter keys can deduplicate repeated reporter locators; docket keys require both a docket number and a court. A courtless docket remains its own root until later identity work can resolve it. `ExtractionRules` controls colocation distance and the context windows; `stable()` supplies the default rules.

The state models are independent of any extraction stage. [`model/document.py`](../src/mellea_lrc/model/document.py) defines `Document`. The [`model/citations/`](../src/mellea_lrc/model/citations/) package defines `FullCitation` for fields shared by full citations. `FullReporterCitation` owns `locator_span`, `locator_text`, and reporter fields (`volume`, `reporter`, `page`); `FullDocketCitation` owns `locator_span`, `locator_text`, and docket fields (`docket_number`, `docket_entry`, `docket_entry_span`). The `Full` names scope these types to full reporter and docket citations.

Each mutable citation field is an append-only tuple of typed `FieldUpdate(value, node_id)` entries. Its current value is the last entry's value (`latest(log)` also handles an empty log); an empty log means the field has not been read, while an entry with `value=None` records an explicit clearing. Each citation stores its own `nodes`, and one node may explain updates to several fields. The field logs are the serialized state, with no separate current-value copy or operation log. The stable `kind` discriminator preserves the concrete citation subtype through a Pydantic JSON round trip. `Document` holds citations, colocations, and completed stages. Save and restore it with Pydantic's native JSON methods:

```python
checkpoint = Path("extraction.json")
checkpoint.write_text(document.model_dump_json(indent=2), encoding="utf-8")
document = Document.model_validate_json(checkpoint.read_text(encoding="utf-8"))
```

`grow_roots` is currently an async composition of the rule-based stages. Its `hunt_dockets=True` option raises `NotImplementedError` until model-backed hunting is rebuilt. Root identity validation, search, and leaves are later layers; this first pass does not issue an identity judgment.
