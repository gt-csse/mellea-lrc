# mellea-lrc

This branch is rebuilding the citation pipeline from a small, inspectable base. The active package currently provides **preprocessing only**. The previous extraction, validation, provider, and evaluation code is preserved in [`archive/root-locators-2026-09-23/`](archive/root-locators-2026-09-23/) for reference; it is outside the installable package.

```python
from pathlib import Path

from mellea_lrc.preprocessing import preprocess

filing = preprocess(Path("filing.pdf"))
print(filing.text)
```

A `Path` reads a file. A `str` is the document text itself. Preprocessing returns a `PreprocessedDocument` with source metadata, the resulting text, and the layout rules that ran. Its spans refer to character offsets in that text. See [the preprocessing contract](docs/Preprocessing.md).

Install the optional Docling backend for PDFs and other formatted files with `uv sync --group preprocessing`. Plain text works with the base package. Run the active tests with `uv run pytest`.
