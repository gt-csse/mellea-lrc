# Extraction

The active extraction pipeline ends at root formation. It accepts a `Document` and returns a `Document`; preprocessing is a separate step. The public functions are in [`mellea_lrc.api`](../src/mellea_lrc/api.py).

```python
from mellea_lrc.api import Document, grow_roots, stable

document = Document.from_source("See 347 U.S. 483 (1954).")
document = grow_roots(document, rules=stable())
after_dockets = document.get_stage("docket_locators")
```

`grow_roots` is synchronous. It runs reporter locator discovery, docket locator discovery, colocation, case-name/court/date/pin-cite reading, and root formation in that order. Each stage can also be called separately through the same API. Colocation sets parsing boundaries; it does not establish case identity.

`Document` contains the preprocessed text and provenance, citation histories, and an ordered `stage_runs` tuple. Each parsed field is an append-only log of domain-specific entries carrying an exact `quote`, its source `span`, normalization status, and a citation-local `node_id`. A successful field contains a typed normalized value. A failed field keeps the quote, span, and error with `normalizable=False`; its `get_normalized()` method raises instead of returning `None`. The field class owns normalization and checks successful values against the quote. An inferred court has no quote or span; root and colocation assignments use separate relationship entries. `get_stage(stage)` reconstructs exactly the document returned by that completed stage, including citations untouched in that stage; it raises `KeyError` if the stage has not run.

`CaseNameField` parses an adversarial name into plaintiff and defendant, or stores the subject of an *In re* or *Ex parte* name. `FullReporterLocator` keeps eyecite's `Reporter` object alongside normalized volume, edition, and page in one locator field. `FullDocketLocator` keeps the written docket number opaque; equivalence is a later identity question. `CourtField`, `DateField`, and `PinCiteField` likewise contain their own typed normalized values and post-validation. Court labels are compared as tokens against `courts-db` citation labels; ordinal variants and district abbreviations derived from that database are accepted only when they identify one court. A reader leaves a field log empty when it finds no field. If normalization fails, it retains the reading as unnormalizable so later stages can review it. Root formation leaves such locators unmerged until their identity can be resolved. The current pin reader recognizes only adjacent numeric page and star-page pins, so other pin formats await a later reader.

```python
saved = document.model_dump_json()
restored = Document.model_validate_json(saved)
assert restored.get_stage("docket_locators") == after_dockets
```

Identity validation, search, and leaf growth are later layers and are not active in this pipeline.
