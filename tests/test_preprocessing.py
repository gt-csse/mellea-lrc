"""Tests for preprocessing."""

import sys
import types

import pytest

from mellea_lrc.core import SourceMetadata
from mellea_lrc.preprocessing import (
    DocumentBase,
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
    SourceFormat,
    is_docling_supported_format,
    preprocess,
    preprocess_plain_text_from_string,
    preprocess_with_docling,
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
    assert isinstance(document, DocumentBase)
    assert document.source_metadata.path == "sample.txt"
    assert document.preprocessing_metadata.backend == PreprocessingBackend.PLAIN_TEXT
    assert document.source_metadata.format == SourceFormat.TEXT


def test_is_docling_supported_format_checks_supported_suffixes() -> None:
    assert is_docling_supported_format("sample.pdf")
    assert is_docling_supported_format("sample.docx")
    assert not is_docling_supported_format("sample.csv")


def test_preprocess_rejects_unsupported_format() -> None:
    with pytest.raises(ValueError, match=r"Unsupported document format: \.csv"):
        preprocess("sample.csv")


def test_preprocess_rejects_path_without_suffix() -> None:
    with pytest.raises(ValueError, match="Unsupported document format: <none>"):
        preprocess("sample")


def test_preprocess_with_docling_exports_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, str | bool] = {}

    class FakeDocument:
        def export_to_text(self) -> str:
            calls["export_to_text"] = True
            return "Plain text"

        def export_to_markdown(self, **_kwargs: object) -> str:
            raise AssertionError("Expected Docling preprocessing to export plain text")

    class FakeResult:
        document = FakeDocument()

    class FakeConverter:
        def convert(self, path: str) -> FakeResult:
            calls["path"] = path
            return FakeResult()

    fake_docling = types.ModuleType("docling")
    fake_converter_module = types.ModuleType("docling.document_converter")
    fake_converter_module.DocumentConverter = FakeConverter
    monkeypatch.setitem(sys.modules, "docling", fake_docling)
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_converter_module)

    document = preprocess_with_docling("sample.pdf")

    assert document.text == "Plain text"
    assert document.source_metadata.format == SourceFormat.PDF
    assert document.preprocessing_metadata.backend == PreprocessingBackend.DOCLING
    assert calls == {"path": "sample.pdf", "export_to_text": True}


def test_preprocessed_document_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        PreprocessedDocument(
            source_metadata=SourceMetadata(),
            text="",
            preprocessing_metadata=PreprocessingMetadata(),
        )
