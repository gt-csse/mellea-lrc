# Preprocessing

`preprocess(source, rules=None)` produces a `PreprocessedDocument`. It is the entry point before the active [extraction stages](Extraction.md).

```python
from pathlib import Path

from mellea_lrc.preprocessing import Rule, preprocess

from_file = preprocess(Path("filing.pdf"))
from_text = preprocess("See a citation in this filing.")
without_layout_rules = preprocess(Path("filing.pdf"), rules=[])
```

A `Path` names a file; a `str` is text content. `.txt` files retain their text exactly, including CRLF and CR line endings. PDFs, DOCX, PPTX, XLSX, HTML, and Markdown files also use the optional Docling backend; `uv sync --group preprocessing` installs both in this repository. Unsupported path suffixes raise `ValueError`.

The returned object contains `text`, `source_metadata`, `preprocessing_metadata`, and `index_spans`. `preprocessing_metadata` records the backend, its version when available, and the rules that ran. `index_spans` marks text regions recognized as a table of authorities. Empty `index_spans` does not prove that no index exists.

Evaluation datasets retain rendering metadata separately from their frozen TXT
files. The evaluation loader verifies the text against that rendering record and
restores a native `PreprocessedDocument`, including its index spans; reading the
TXT alone cannot restore layout metadata. The current Docling detector treats
`document_index` tables as authority indexes, so its recorded spans can also
include a contents table. The loader preserves those source-derived spans.

`Document.add_citation` tags an occurrence when its complete creation span is
inside one of these components. The citation exposes `is_toa()`; content
substages use that method rather than inspecting regions or tag enums. A partial
boundary overlap does not tag the occurrence, and a leaf does not inherit its
root's tag.

The default rules handle margin line numbers, repeated page furniture, docket stamps, tables as text, and tables of authorities. Pass a sequence of `Rule` values to choose exactly which rules run; pass `[]` to use the converter's reading without these project rules. Layout rules require page structure, so they are not applied to plain text. Different rule selections may produce different text and character offsets. `TABLE_AS_TEXT` disables cell reconstruction for PDF conversion; other formats use their default Docling converters.

Margin line-number handling belongs entirely to preprocessing. The Docling rule
requires a counting numeric column at the page edge. It reclassifies those items
as furniture before text export and offset measurement, preserving body lists
and reporter page numbers. Items without geometry remain untouched. Matching and grounding operate on the
supplied text and do not infer or remove line numbers from it.

`PreprocessedDocument` is a Pydantic model and can be saved and restored with `model_dump(mode="json")` and `PreprocessedDocument.model_validate(...)`.
