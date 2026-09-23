# Extraction

The active extraction pipeline ends at root formation. It accepts a `Document` and returns a `Document`; preprocessing is a separate step. The public functions are in [`mellea_lrc.api`](../src/mellea_lrc/api.py).

```python
from mellea_lrc.api import Document, grow_roots, stable

document = Document.from_source("See 347 U.S. 483 (1954).")
document = grow_roots(document, rules=stable())
after_dockets = document.get_stage("docket_locators")
```

`grow_roots` is synchronous. It runs reporter locator discovery, docket locator discovery, colocation, case-name/court/date/pin-cite reading, and root formation in that order. Each stage can also be called separately through the same API. Colocation sets parsing boundaries; it does not establish case identity.

`Document` contains the preprocessed text and provenance, citation histories, and an ordered `stage_runs` tuple. A citation's fields are append-only logs whose entries point to citation-local nodes. `get_stage(stage)` reconstructs exactly the document returned by that completed stage, including citations untouched in that stage; it raises `KeyError` if the stage has not run.

```python
saved = document.model_dump_json()
restored = Document.model_validate_json(saved)
assert restored.get_stage("docket_locators") == after_dockets
```

Identity validation, search, and leaf growth are later layers and are not active in this pipeline.
