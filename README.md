# mellea-lrc

This branch is rebuilding the citation pipeline from a small, inspectable base. The active package provides preprocessing and a rule-based first extraction pass. The previous extraction, validation, provider, and evaluation implementations are preserved in [`archive/root-locators-2026-09-23/`](archive/root-locators-2026-09-23/) for reference; they are outside the installable package.

```python
from pathlib import Path

from mellea_lrc.api import Document, grow_roots, stable

document = Document.from_source(Path("filing.pdf"))
document = await grow_roots(document, rules=stable())
print(document.full_locators, document.roots)
```

`grow_roots` composes independent `Document -> Document` stages: reporter locators, docket locators, colocation, contextual fields, then root formation. Each full locator is an occurrence; roots deduplicate only supported exact identifiers. The document retains stage history and can be saved and restored as Pydantic JSON. See [the extraction API](docs/Extraction.md) and [the preprocessing contract](docs/Preprocessing.md).

A `Path` reads a file; a `str` is the document text itself. Install the optional Docling backend for PDFs and other formatted files with `uv sync --group preprocessing`. Plain text works with the base package. Run the active tests with `uv run pytest`.
