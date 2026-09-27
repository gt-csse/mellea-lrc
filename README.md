# mellea-lrc

The active package provides preprocessing, extraction through root formation, and independently callable root-lookup validation stages. Docket site hunting is optional and runs before colocation. Text-body corroboration, open search, and leaf growth are later work.

```python
from pathlib import Path
import asyncio

from mellea_lrc.api import Document, grow_roots, stable

document = Document.from_source(Path("filing.pdf"))
document = asyncio.run(grow_roots(document, rules=stable()))
print(document.full_locators, document.roots)
```

`grow_roots` is async so it can optionally review docket sites with the configured model: pass `hunt_dockets=True` to enable that stage. The document retains citation histories, site reviews, and completed stage runs; after hunting, `document.get_stage("3_docket_locator_site_hunting")` restores that stage's result. See [extraction](docs/Extraction.md), [validation](docs/Validation.md), and [preprocessing](docs/Preprocessing.md).

A `Path` reads a file; a `str` is the document text itself. Install the optional Docling backend for PDFs and other formatted files with `uv sync --group preprocessing`. Plain text works with the base package. Run the active tests with `uv run pytest`.
