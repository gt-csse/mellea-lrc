# Preprocessing

`preprocess(source, rules=None)` produces a `PreprocessedDocument`. It is the only active pipeline stage while extraction is rebuilt.

```python
from pathlib import Path

from mellea_lrc.preprocessing import Rule, preprocess

from_file = preprocess(Path("filing.pdf"))
from_text = preprocess("See a citation in this filing.")
without_layout_rules = preprocess(Path("filing.pdf"), rules=[])
```

A `Path` names a file; a `str` is text content. `.txt` files retain their text exactly. PDFs, DOCX, PPTX, XLSX, HTML, and Markdown files use the optional Docling backend (`uv sync --group preprocessing`). Unsupported path suffixes raise `ValueError`.

The returned object contains `text`, `source_metadata`, `preprocessing_metadata`, and `index_spans`. `preprocessing_metadata` records the backend, its version when available, and the rules that ran. `index_spans` marks text regions recognized as a table of authorities. Empty `index_spans` does not prove that no index exists.

The default rules handle margin line numbers, repeated page furniture, docket stamps, tables as text, and tables of authorities. Pass a sequence of `Rule` values to choose exactly which rules run; pass `[]` to use the converter's reading without these project rules. Layout rules require page structure, so they are not applied to plain text. Different rule selections may produce different text and character offsets.

`PreprocessedDocument` is a Pydantic model and can be saved and restored with `model_dump(mode="json")` and `PreprocessedDocument.model_validate(...)`.
