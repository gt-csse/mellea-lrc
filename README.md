# mellea-lrc

The active package provides preprocessing and a rule-based first extraction pass through root formation. Identity validation, search, and leaves are not part of this active pipeline yet.

```python
from pathlib import Path

from mellea_lrc.api import Document, grow_roots, stable

document = Document.from_source(Path("filing.pdf"))
document = grow_roots(document, rules=stable())
print(document.full_locators, document.roots)
```

`grow_roots` composes independent `Document -> Document` stages. The document retains citation histories and completed stage runs; `document.get_stage("docket_locators")` restores that stage's result. See [extraction](docs/Extraction.md) and [preprocessing](docs/Preprocessing.md).

A `Path` reads a file; a `str` is the document text itself. Install the optional Docling backend for PDFs and other formatted files with `uv sync --group preprocessing`. Plain text works with the base package. Run the active tests with `uv run pytest`.
