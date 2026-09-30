# mellea-lrc

The package has three workflows: `grow_roots`, `validate_roots`, and `grow_leaves`. Each takes and returns a `Document`, and each consists of independently callable stages. Locator-body review and intended-case discovery belong to root validation. Open-web search is not implemented.

The source layers are:

```text
src/mellea_lrc/
  workflows/       grow_roots, validate_roots, grow_leaves
  extraction/      citation discovery, field reading, and root/leaf assignment stages
  validation/      retrieval and judgment stages, with local review services
  providers/       courtlistener/ and govinfo/ clients and response models
  model/           Document, citations, typed fields, and append-only histories
  preprocessing/   source loading and text preparation
  matching/        fuzzy matching and grounded source quotes
  llm/             model configuration and the reusable IVR wrapper
```

Workflows are defined by the user. A group of stages does not introduce another workflow. `evaluations/` produces one Markdown report and its JSON representation per workflow, including that workflow's stage tables.

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
