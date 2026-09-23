"""Read plain text without adding a header or changing its offsets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mellea_lrc.model.documents import SourceFormat, SourceMetadata
from mellea_lrc.model.preprocessed import (
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
