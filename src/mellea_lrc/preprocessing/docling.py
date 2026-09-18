"""Docling-backed preprocessing from raw Layer 3 documents."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from mellea_lrc.core.documents import SourceFormat, SourceMetadata
from mellea_lrc.core.spans import Span
from mellea_lrc.preprocessing.docket_stamp import reclassify_docket_stamps
from mellea_lrc.preprocessing.document_index import index_table_spans
from mellea_lrc.preprocessing.margin_line_numbers import reclassify_margin_line_numbers
from mellea_lrc.preprocessing.repeated_furniture import reclassify_repeated_furniture
from mellea_lrc.preprocessing.types import (
    DEFAULT_RULES,
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
    Rule,
)

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
    """Convert a raw document to plain text using Docling.

    ``rules`` says which page furniture to take out before the text is
    written. Docling reads all of it correctly and files some of it under the
    body layer, where it survives into the text and lands wherever the page
    broke -- a margin number inside a citation, the `9` of "Page 3 of 9" read
    together with the date after it.

    All three run by default. None of it is the document's text, and a rendering
    that interleaves it into a citation is wrong about the document. Pass a
    shorter list to keep some of it, or an empty one to keep all of it.

    Each rule moves the offsets of everything after it, so two renderings made
    under different lists are different coordinate spaces. Which ran is recorded
    on the result rather than assumed.

    ## Tables are read, not rebuilt

    ``do_table_structure`` is off, so docling's layout model still finds a table
    and still labels it -- a table of authorities still comes back as
    ``document_index`` -- but its structure model does not divide it into cells.
    The region is written out as one block, in the order the page reads.

    A brief's table of authorities is not a table. It is indented lines with dot
    leaders, which the layout model classifies as one because of the alignment,
    and dividing it into cells does two kinds of damage. It interleaves cell
    separators into the text, so seven citations on this corpus carry a `|`
    inside their own span -- characters the filing does not contain. And it
    assigns lines to the wrong cells, which reorders them: document 021's
    `Loos v. Lowe's` and `796 F. Supp. 2d 1013, 1023` come out with the page
    before its own reporter and the reporter beside the *next* case's name, a
    citation no relaxation can read because the parts are out of order rather
    than merely separated.

    Reading the region as text instead is worth, over the 26 filings of
    `false-citation-bench`: pipes 915 to 80, one authority recovered that
    appeared nowhere else in its filing, none lost, and 24 fewer case names left
    with no locator beside them. Where a document has no table it changes
    nothing at all.

    What is given up is the column structure of the genuine tables -- a table of
    evidence, a list of proceedings. Nothing here reads columns, and a citation
    inside one is read in the same order a person would read it.
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
