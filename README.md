# mellea-lrc

The active package provides preprocessing and a first extraction pass through root formation. Docket site hunting is optional and runs before colocation. Identity validation, search, and leaves are not part of this active pipeline yet.

```python
from pathlib import Path
import asyncio

from mellea_lrc.api import Document, grow_roots, stable

document = Document.from_source(Path("filing.pdf"))
document = asyncio.run(grow_roots(document, rules=stable()))
print(document.full_locators, document.roots)
```

`grow_roots` is async so it can optionally review docket sites with the configured model: pass `hunt_dockets=True` to enable that stage. The document retains citation histories, site reviews, and completed stage runs; after hunting, `document.get_stage("docket_locator_site_hunting")` restores that stage's result. See [extraction](docs/Extraction.md) and [preprocessing](docs/Preprocessing.md).

A `Path` reads a file; a `str` is the document text itself. Install the optional Docling backend for PDFs and other formatted files with `uv sync --group preprocessing`. Plain text works with the base package. Run the active tests with `uv run pytest`.
