"""Load plain text files into canonical preprocessing types.

A text file is its text. Nothing is stripped from the front of it, so an offset
into the file is an offset into the document, and there is one coordinate system
rather than one per source.

This project used to write conversion provenance -- the source PDF, the Docling
version -- as a header above a `--- Plain text ---` marker, and every reader had
to know to skip it. Court records arrive with enough furniture of their own; the
provenance now sits beside the text in a `renderings.json` or `documents.json`,
where reading it is a choice rather than an obligation. Nothing else writes that
marker: an archive hands over the document's own first page, so there is no
preamble to take out and no rule that would find one.

The layout rules do not apply here. Every one of them reads the page -- where an
item sits, whether its neighbours repeat -- and a `.txt` file carries no
geometry, so a document made this way records that no rule ran.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mellea_lrc.core.documents import SourceFormat, SourceMetadata
from mellea_lrc.preprocessing.types import (
    DEFAULT_RULES,
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
    Rule,
)


def preprocess_plain_text(
    path: Path | str,
    *,
    rules: Sequence[Rule] = DEFAULT_RULES,
) -> PreprocessedDocument:
    """Load a `.txt` file as a preprocessed document.

    ``rules`` is accepted so that one list serves every format, and is
    not applied: see the note above.
    """
    source_path = Path(path)
    return preprocess_plain_text_from_string(
        source_path.read_text(encoding="utf-8"),
        source_path=str(source_path),
        rules=rules,
    )


def preprocess_plain_text_from_string(
    text: str,
    *,
    source_path: str | None = None,
    rules: Sequence[Rule] = DEFAULT_RULES,
) -> PreprocessedDocument:
    """Wrap raw text in a preprocessed document without reading a file."""
    del rules
    return PreprocessedDocument(
        source_metadata=SourceMetadata(
            path=source_path,
            format=SourceFormat.TEXT,
        ),
        text=text,
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend.PLAIN_TEXT,
        ),
    )
