"""Docling-backed preprocessing from raw Layer 3 documents."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from mellea_lrc.model.preprocessed_document import (
    DEFAULT_RULES,
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
    Rule,
)
from mellea_lrc.model.source import SourceFormat, SourceMetadata
from mellea_lrc.model.span import Span
from mellea_lrc.preprocessing.docket_stamp import reclassify_docket_stamps
from mellea_lrc.preprocessing.document_index import index_table_spans
from mellea_lrc.preprocessing.margin_line_numbers import reclassify_margin_line_numbers
from mellea_lrc.preprocessing.repeated_furniture import reclassify_repeated_furniture

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

_SOURCE_FORMAT_BY_SUFFIX = {
    ".pdf": SourceFormat.PDF,
    ".docx": SourceFormat.DOCX,
    ".pptx": SourceFormat.PPTX,
    ".xlsx": SourceFormat.XLSX,
    ".html": SourceFormat.HTML,
    ".htm": SourceFormat.HTML,
    ".md": SourceFormat.MARKDOWN,
}


def is_docling_supported_format(path: Path | str) -> bool:
    """Return True when Docling supports the path suffix."""
    return Path(path).suffix.lower() in _SOURCE_FORMAT_BY_SUFFIX


def _docling_version() -> str | None:
    try:
        return version("docling")
    except PackageNotFoundError:
        return None


def _source_format(path: Path) -> SourceFormat:
    return _SOURCE_FORMAT_BY_SUFFIX.get(path.suffix.lower(), SourceFormat.UNKNOWN)


def _reads_tables_as_text(rules: Sequence[Rule]) -> bool:
    """Whether the converter should leave a table as text rather than rebuild it.

    This rule is decided before the conversion runs, because it is the only one
    that changes what the converter does rather than what is done with what it
    produced. It is still a rule: named, declinable, and recorded on the result
    like the rest.
    """
    return Rule.TABLE_AS_TEXT in rules


def _apply_rules(document: DoclingDocument, rules: Sequence[Rule]) -> tuple[Span, ...]:
    """Run each rule against the converted document, in the order given.

    Returns the regions `TABLE_OF_AUTHORITIES` marked.
    """
    index_spans: tuple[Span, ...] = ()
    for rule in rules:
        if rule is Rule.MARGIN_LINE_NUMBERS:
            reclassify_margin_line_numbers(document)
        elif rule is Rule.REPEATED_FURNITURE:
            reclassify_repeated_furniture(document)
        elif rule is Rule.DOCKET_STAMP:
            reclassify_docket_stamps(document)
        elif rule is Rule.TABLE_AS_TEXT:
            pass  # Decided before the conversion; see `_reads_tables_as_text`.
        elif rule is Rule.TABLE_OF_AUTHORITIES:
            index_spans = index_table_spans(document)
        else:
            msg = f"Unknown layout rule: {rule}"
            raise ValueError(msg)
    return index_spans


def preprocess_with_docling(
    path: Path | str,
    *,
    rules: Sequence[Rule] = DEFAULT_RULES,
) -> PreprocessedDocument:
    """Convert a file to text, applying the selected layout rules before export.

    ``TABLE_AS_TEXT`` disables cell reconstruction. The other rules classify
    page furniture or mark index regions. The returned text is the span space.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        msg = (
            "Docling is required for raw document preprocessing. Install with: uv sync --group preprocessing"
        )
        raise ImportError(msg) from exc

    source_path = Path(path)
    # A table is read as one block of text rather than as a grid. See the note
    # on `do_table_structure` below.
    options = PdfPipelineOptions()
    options.do_table_structure = not _reads_tables_as_text(rules)
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    result = converter.convert(str(source_path))
    applied = tuple(rules)
    index_spans = _apply_rules(result.document, applied)
    text = result.document.export_to_text()  # Ensure to normalize all characters to Unicode TODO

    return PreprocessedDocument(
        source_metadata=SourceMetadata(
            path=str(source_path),
            format=_source_format(source_path),
        ),
        text=text,
        index_spans=index_spans,
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend.DOCLING,
            backend_version=_docling_version(),
            rules=applied,
        ),
    )
