"""Tests for preprocessing."""

import sys
import types
from pathlib import Path

import pytest

from mellea_lrc.core import SourceMetadata
from mellea_lrc.preprocessing import (
    DEFAULT_RULES,
    DocumentBase,
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
    Rule,
    SourceFormat,
    preprocess,
)
from mellea_lrc.preprocessing.docket_stamp import looks_like_a_stamp
from mellea_lrc.preprocessing.docling import is_docling_supported_format, preprocess_with_docling
from mellea_lrc.preprocessing.filing_metadata import (
    FilingMetadataKind,
    filing_metadata_manifest,
    mask_filing_metadata,
    restore_filing_metadata,
)


def test_a_text_file_is_its_text() -> None:
    """Nothing is stripped from the front, so a file offset is a document offset."""
    raw = "Case: Example\n\n--- Plain text ---\nBody text here."

    document = preprocess(raw)

    assert document.text == raw
    assert document.preprocessing_metadata.rules == (Rule.FILING_METADATA,)


def test_text_in_hand_needs_no_file() -> None:
    document = preprocess("Hello world.")
    assert document.text == "Hello world."
    assert isinstance(document, DocumentBase)
    assert document.source_metadata.path is None
    assert document.preprocessing_metadata.backend == PreprocessingBackend.PLAIN_TEXT
    assert document.source_metadata.format == SourceFormat.TEXT


def test_is_docling_supported_format_checks_supported_suffixes() -> None:
    assert is_docling_supported_format("sample.pdf")
    assert is_docling_supported_format("sample.docx")
    assert not is_docling_supported_format("sample.csv")


def test_preprocess_rejects_unsupported_format() -> None:
    with pytest.raises(ValueError, match=r"Unsupported document format: \.csv"):
        preprocess(Path("sample.csv"))


def test_preprocess_rejects_path_without_suffix() -> None:
    with pytest.raises(ValueError, match="Unsupported document format: <none>"):
        preprocess(Path("sample"))


def test_a_string_is_content_and_a_path_is_a_location() -> None:
    """The argument's type says what it is, as it does for `extract`."""
    document = preprocess("sample.csv")

    assert document.text == "sample.csv"
    assert document.source_metadata.path is None


def test_preprocess_with_docling_exports_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, str | bool] = {}

    class FakeDocument:
        # The index locator walks these, so they have to exist. Empty here:
        # this test is about which export is called.
        tables: tuple[object, ...] = ()

        def export_to_text(self) -> str:
            calls["export_to_text"] = True
            return "Plain text"

        def export_to_markdown(self, **_kwargs: object) -> str:
            raise AssertionError("Expected Docling preprocessing to export plain text")

    class FakeResult:
        document = FakeDocument()

    class FakePipelineOptions:
        do_table_structure = True

    class FakeConverter:
        def __init__(self, format_options: dict[object, object] | None = None) -> None:
            (option,) = (format_options or {}).values()
            # A table is read as one block of text, not rebuilt as a grid.
            calls["do_table_structure"] = option.pipeline_options.do_table_structure

        def convert(self, path: str) -> FakeResult:
            calls["path"] = path
            return FakeResult()

    class FakeFormatOption:
        def __init__(self, pipeline_options: object) -> None:
            self.pipeline_options = pipeline_options

    fake_docling = types.ModuleType("docling")
    fake_converter_module = types.ModuleType("docling.document_converter")
    fake_converter_module.DocumentConverter = FakeConverter
    fake_converter_module.PdfFormatOption = FakeFormatOption
    fake_models = types.ModuleType("docling.datamodel.base_models")
    fake_models.InputFormat = types.SimpleNamespace(PDF="pdf")
    fake_options = types.ModuleType("docling.datamodel.pipeline_options")
    fake_options.PdfPipelineOptions = FakePipelineOptions
    monkeypatch.setitem(sys.modules, "docling", fake_docling)
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_converter_module)
    monkeypatch.setitem(sys.modules, "docling.datamodel", types.ModuleType("docling.datamodel"))
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", fake_models)
    monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", fake_options)

    # No layout rules: this test is about which export is called, and the rules
    # need a real Docling document to walk.
    document = preprocess_with_docling("sample.pdf", rules=())

    assert document.text == "Plain text"
    assert document.source_metadata.format == SourceFormat.PDF
    assert document.preprocessing_metadata.backend == PreprocessingBackend.DOCLING
    # No rules: the converter's own reading, table structure included.
    assert calls == {"path": "sample.pdf", "export_to_text": True, "do_table_structure": True}
    assert document.index_spans == ()


def test_docling_runs_the_rules_it_was_given_and_records_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every rule in the list runs, and the result says which did.

    A rule that is exported but never reached does nothing, and a document
    rendered without it is a different coordinate space than one rendered with
    it. The record is what tells the two apart.
    """

    class FakeDocument:
        # The rules walk these, and the index locator walks the tables. Empty
        # here: this test is about which rules run, not about what they find.
        texts: tuple[object, ...] = ()
        tables: tuple[object, ...] = ()

        def iterate_items(self, **_kwargs: object) -> tuple[object, ...]:
            return ()

        def export_to_text(self) -> str:
            return "Plain text"

    class FakeResult:
        document = FakeDocument()

    class FakePipelineOptions:
        do_table_structure = True

    class FakeFormatOption:
        def __init__(self, pipeline_options: object) -> None:
            self.pipeline_options = pipeline_options

    class FakeConverter:
        def __init__(self, format_options: dict[object, object] | None = None) -> None:
            del format_options

        def convert(self, path: str) -> FakeResult:
            return FakeResult()

    fake_docling = types.ModuleType("docling")
    fake_converter_module = types.ModuleType("docling.document_converter")
    fake_converter_module.DocumentConverter = FakeConverter
    fake_converter_module.PdfFormatOption = FakeFormatOption
    fake_models = types.ModuleType("docling.datamodel.base_models")
    fake_models.InputFormat = types.SimpleNamespace(PDF="pdf")
    fake_options = types.ModuleType("docling.datamodel.pipeline_options")
    fake_options.PdfPipelineOptions = FakePipelineOptions
    monkeypatch.setitem(sys.modules, "docling", fake_docling)
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_converter_module)
    monkeypatch.setitem(sys.modules, "docling.datamodel", types.ModuleType("docling.datamodel"))
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", fake_models)
    monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", fake_options)

    document = preprocess_with_docling("sample.pdf")

    assert document.preprocessing_metadata.rules == DEFAULT_RULES

    kept = preprocess_with_docling("sample.pdf", rules=())
    assert kept.preprocessing_metadata.rules == ()


def test_preprocessed_document_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        PreprocessedDocument(
            source_metadata=SourceMetadata(),
            text="",
            preprocessing_metadata=PreprocessingMetadata(),
        )


def test_a_filing_stamp_is_recognised_whatever_court_printed_it() -> None:
    """The gate is loose on purpose: no one court's wording is required."""
    assert looks_like_a_stamp("Case 2:25-cv-01295-GMS     Document 1     Filed 04/18/25     Page 6 of 32")
    assert looks_like_a_stamp(
        "Case No. 1:24-cv-00814-PAB-SBP   Document 77   filed 10/27/25   USDC Colorado   pg 1 of 9"
    )
    assert looks_like_a_stamp("Case: 1:24-cv-00074-SA-DAS Doc #: 79-1 Filed: 12/19/25 1 of 3 PageID #: 513")


def test_prose_is_not_a_filing_stamp() -> None:
    """A sentence that mentions a case and a page is still a sentence."""
    assert not looks_like_a_stamp("In that case the court reached page 12 of the opinion before saying so.")
    assert not looks_like_a_stamp("See Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).")
    assert not looks_like_a_stamp("")


def test_filing_metadata_mask_is_offset_preserving_and_reversible() -> None:
    """Only the opening caption and recurrent ECF furniture are unavailable to extraction."""
    raw = (
        "IN THE UNITED STATES DISTRICT COURT\n"
        "FOR THE DISTRICT OF EXAMPLE\n\n"
        "ALICE SMITH, Plaintiff,\n"
        "v.\n"
        "BOB JONES, Defendant.\n"
        "Case No. 2:25-cv-00804\n\n"
        "The related matter is Case No. 1:24-cv-00077. Counsel is State Bar No. 123456.\n"
        "See ECF No. 303 and Exhibit 2; Permit No. 99-123 remains active.\n\n"
        "Case 2:25-cv-00804 Document 303 Filed 03/17/26 Page 1 of 16\n"
        "Argument citing Smith v. Jones, No. 1:24-cv-00077 (D. Example 2024).\n"
        "Case 2:25-cv-00804 Document 303 Filed 03/17/26 Page 15 of 16\n"
    )

    masked = mask_filing_metadata(raw)

    assert len(masked.text) == len(raw)
    assert masked.text[
        raw.index("2:25-cv-00804") : raw.index("2:25-cv-00804") + len("2:25-cv-00804")
    ] == " " * len("2:25-cv-00804")
    assert "Case No. 1:24-cv-00077" in masked.text
    assert "State Bar No. 123456" in masked.text
    assert "ECF No. 303" in masked.text
    assert "Exhibit 2" in masked.text
    assert "Permit No. 99-123" in masked.text
    assert [removal.kind for removal in masked.removals] == [
        FilingMetadataKind.CAPTION_DOCKET,
        FilingMetadataKind.ECF_STAMP,
        FilingMetadataKind.ECF_STAMP,
    ]
    assert [removal.text for removal in masked.removals[1:]] == [
        "Case 2:25-cv-00804 Document 303 Filed 03/17/26 Page 1 of 16",
        "Case 2:25-cv-00804 Document 303 Filed 03/17/26 Page 15 of 16",
    ]
    assert restore_filing_metadata(masked.text, masked.removals).encode("utf-8") == raw.encode("utf-8")


def test_a_complete_ecf_stamp_masks_without_masking_a_courtless_docket() -> None:
    raw = "Case 2:25-cv-00804 Document 303 Filed 03/17/26 Page 15 of 16\nCase No. 1:24-cv-00077"

    masked = mask_filing_metadata(raw)

    assert masked.text[: raw.index("\n")] == " " * raw.index("\n")
    assert masked.text.endswith("Case No. 1:24-cv-00077")
    assert [removal.kind for removal in masked.removals] == [FilingMetadataKind.ECF_STAMP]


def test_a_caption_label_needs_a_docket_number_and_allows_a_colon() -> None:
    raw = (
        "IN THE UNITED STATES DISTRICT COURT\n"
        "Civil Action No:\n\n2:25-cv-02623-SHL-atc\n\n"
        "The case law does not change the analysis."
    )

    masked = mask_filing_metadata(raw)

    assert "2:25-cv-02623-SHL-atc" not in masked.text
    assert "case law" in masked.text


def test_filing_metadata_manifest_has_hashes_and_reversible_removals() -> None:
    raw = "Case 2:25-cv-00804 Document 303 Filed 03/17/26 Page 15 of 16"

    manifest = filing_metadata_manifest("filing.txt", raw)
    masked = mask_filing_metadata(raw)

    assert manifest["source_path"] == "filing.txt"
    assert manifest["text_length"] == len(raw)
    assert manifest["original_utf8_sha256"] != manifest["masked_utf8_sha256"]
    assert manifest["removals"] == [
        {
            "kind": "ecf_page_stamp",
            "start": 0,
            "end": len(raw),
            "text": raw,
        }
    ]
    assert restore_filing_metadata(masked.text, masked.removals).encode("utf-8") == raw.encode("utf-8")


def test_declining_the_table_rule_leaves_the_converter_to_rebuild_the_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`TABLE_AS_TEXT` is the only rule decided before the conversion runs.

    It is still a rule: name it and a table is read in the order the page reads
    it, leave it out and the converter divides the region into cells, which is
    what put a case name in a different cell from its own citation.
    """
    seen: dict[str, bool] = {}

    class FakeDocument:
        texts: tuple[object, ...] = ()
        tables: tuple[object, ...] = ()

        def iterate_items(self, **_kwargs: object) -> tuple[object, ...]:
            return ()

        def export_to_text(self) -> str:
            return "Plain text"

    class FakeResult:
        document = FakeDocument()

    class FakePipelineOptions:
        do_table_structure = True

    class FakeFormatOption:
        def __init__(self, pipeline_options: object) -> None:
            self.pipeline_options = pipeline_options

    class FakeConverter:
        def __init__(self, format_options: dict[object, object] | None = None) -> None:
            (option,) = (format_options or {}).values()
            seen["do_table_structure"] = option.pipeline_options.do_table_structure

        def convert(self, path: str) -> FakeResult:
            return FakeResult()

    fake_docling = types.ModuleType("docling")
    fake_converter_module = types.ModuleType("docling.document_converter")
    fake_converter_module.DocumentConverter = FakeConverter
    fake_converter_module.PdfFormatOption = FakeFormatOption
    fake_models = types.ModuleType("docling.datamodel.base_models")
    fake_models.InputFormat = types.SimpleNamespace(PDF="pdf")
    fake_options = types.ModuleType("docling.datamodel.pipeline_options")
    fake_options.PdfPipelineOptions = FakePipelineOptions
    monkeypatch.setitem(sys.modules, "docling", fake_docling)
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_converter_module)
    monkeypatch.setitem(sys.modules, "docling.datamodel", types.ModuleType("docling.datamodel"))
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", fake_models)
    monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", fake_options)

    preprocess_with_docling("sample.pdf", rules=(Rule.TABLE_AS_TEXT,))
    assert seen["do_table_structure"] is False

    preprocess_with_docling("sample.pdf", rules=(Rule.DOCKET_STAMP,))
    assert seen["do_table_structure"] is True
