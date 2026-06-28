"""Tests for preprocessing."""

import sys
import types
from pathlib import Path

import pytest

from mellea_lrc.preprocessing import (
    PreprocessingBackend,
    SourceFormat,
    is_docling_supported_format,
    run_preprocessing,
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
    assert document.source_metadata.path == "sample.txt"
    assert document.preprocessing_metadata.backend == PreprocessingBackend.PLAIN_TEXT
    assert document.source_metadata.format == SourceFormat.TEXT


def test_is_docling_supported_format_checks_supported_suffixes() -> None:
    assert is_docling_supported_format("sample.pdf")
    assert is_docling_supported_format("sample.docx")
    assert not is_docling_supported_format("sample.csv")


def test_preprocess_rejects_unsupported_format() -> None:
    with pytest.raises(ValueError, match=r"Unsupported document format: \.csv"):
        run_preprocessing("sample.csv")


def test_preprocess_rejects_path_without_suffix() -> None:
    with pytest.raises(ValueError, match="Unsupported document format: <none>"):
        run_preprocessing("sample")


def test_preprocess_with_docling_exports_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeDocument:
        def export_to_text(self) -> str:
            calls["export_to_text"] = True
            return "Plain text"

        def export_to_markdown(self, **_kwargs: object) -> str:
            raise AssertionError("Expected Docling preprocessing to export plain text")

    class FakeResult:
        document = FakeDocument()

    class FakeConverter:
        def __init__(self, **kwargs: object) -> None:
            calls["converter_kwargs"] = kwargs

        def convert(self, path: str) -> FakeResult:
            calls["path"] = path
            return FakeResult()

    class FakePdfPipelineOptions:
        def __init__(self) -> None:
            self.do_ocr = False
            self.ocr_options = None

    class FakeTesseractCliOcrOptions:
        def __init__(self, *, lang: list[str]) -> None:
            self.lang = lang

    class FakePdfFormatOption:
        def __init__(self, *, pipeline_options: FakePdfPipelineOptions) -> None:
            self.pipeline_options = pipeline_options

    fake_docling = types.ModuleType("docling")
    fake_base_models_module = types.ModuleType("docling.datamodel.base_models")
    fake_base_models_module.InputFormat = types.SimpleNamespace(PDF="pdf")
    fake_pipeline_options_module = types.ModuleType("docling.datamodel.pipeline_options")
    fake_pipeline_options_module.PdfPipelineOptions = FakePdfPipelineOptions
    fake_pipeline_options_module.TesseractCliOcrOptions = FakeTesseractCliOcrOptions
    fake_converter_module = types.ModuleType("docling.document_converter")
    fake_converter_module.DocumentConverter = FakeConverter
    fake_converter_module.PdfFormatOption = FakePdfFormatOption
    monkeypatch.setitem(sys.modules, "docling", fake_docling)
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", fake_base_models_module)
    monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", fake_pipeline_options_module)
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_converter_module)

    document = preprocess_with_docling("sample.pdf")

    assert document.text == "Plain text"
    assert document.source_metadata.format == SourceFormat.PDF
    assert document.preprocessing_metadata.backend == PreprocessingBackend.DOCLING
    assert calls["path"] == "sample.pdf"
    assert calls["export_to_text"] is True
    format_options = calls["converter_kwargs"]["format_options"]  # type: ignore[index]
    pdf_options = format_options["pdf"].pipeline_options  # type: ignore[index]
    assert pdf_options.do_ocr is True
    assert pdf_options.ocr_options.lang == ["eng"]





@pytest.mark.heavy
def test_preprocess_with_real_docling_pdf_exports_plain_text(tmp_path: Path) -> None:
    pytest.importorskip("docling", reason="Install with `uv sync --group preprocessing`.")
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(_minimal_text_pdf("Brown v. Board, 347 U.S. 483."))

    document = preprocess_with_docling(pdf_path)

    assert "Brown v. Board" in document.text
    assert "347 U.S. 483" in document.text
    assert document.source_metadata.path == str(pdf_path)
    assert document.source_metadata.format == SourceFormat.PDF
    assert document.preprocessing_metadata.backend == PreprocessingBackend.DOCLING


def _minimal_text_pdf(text: str) -> bytes:
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(_pdf_text_stream(escaped_text))} >>\nstream\n".encode()
        + _pdf_text_stream(escaped_text)
        + b"\nendstream",
    ]
    return _pdf_document(objects)


def _pdf_text_stream(escaped_text: str) -> bytes:
    return f"BT /F1 18 Tf 72 720 Td ({escaped_text}) Tj ET".encode()


def _pdf_document(objects: list[bytes]) -> bytes:
    chunks = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")

    xref_offset = sum(len(chunk) for chunk in chunks)
    chunks.append(f"xref\n0 {len(objects) + 1}\n".encode())
    chunks.append(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        chunks.append(f"{offset:010d} 00000 n \n".encode())
    chunks.append(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return b"".join(chunks)
