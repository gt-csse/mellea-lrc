"""Preprocess a file path or text with an explicit set of layout rules."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mellea_lrc.model.preprocessed_document import DEFAULT_RULES, PreprocessedDocument, Rule
from mellea_lrc.preprocessing.docling import is_docling_supported_format, preprocess_with_docling
from mellea_lrc.preprocessing.plain_text import preprocess_plain_text, preprocess_plain_text_from_string


def preprocess(
    source: Path | str,
    rules: Sequence[Rule] | None = None,
) -> PreprocessedDocument:
    """Read a document, or wrap text already in hand, under the given rules.

    A `Path` is opened and converted by the backend its suffix calls for; a
    `str` is the document's text. `rules` defaults to every rule.
    """
    applied = DEFAULT_RULES if rules is None else tuple(rules)
    if isinstance(source, str):
        return preprocess_plain_text_from_string(source, rules=applied)
    if source.suffix.lower() == ".txt":
        return preprocess_plain_text(source, rules=applied)
    if not is_docling_supported_format(source):
        msg = f"Unsupported document format: {source.suffix or '<none>'}"
        raise ValueError(msg)
    return preprocess_with_docling(source, rules=applied)
