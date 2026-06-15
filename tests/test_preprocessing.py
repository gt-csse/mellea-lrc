"""Tests for preprocessing."""

from mellea_lrc.preprocessing import (
    PreprocessingBackend,
    SourceFormat,
    preprocess_plain_text_from_string,
    split_plain_text_file,
)


def test_split_plain_text_file_splits_recap_header() -> None:
    raw = "Case: Example\n\n--- Plain text ---\nBody text here."
    header, body = split_plain_text_file(raw)
    assert header == "Case: Example"
    assert body == "Body text here."


def test_preprocess_plain_text_from_string_wraps_text() -> None:
    document = preprocess_plain_text_from_string("Hello world.", source_path="sample.txt")
    assert document.text == "Hello world."
    assert document.metadata.source_path == "sample.txt"
    assert document.metadata.backend == PreprocessingBackend.PLAIN_TEXT
    assert document.metadata.source_format == SourceFormat.TEXT
