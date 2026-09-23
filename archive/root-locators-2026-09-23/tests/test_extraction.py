"""Tests for citation extraction."""

from pathlib import Path

import pytest

from mellea_lrc.extraction.eyecite_extractor import extract_citations, extract_from_plain_text
from mellea_lrc.model.citations import CitationKind, FullCaseCitation, FullLawCitation, placed
from mellea_lrc.model.document import Document
from mellea_lrc.model.extraction_metadata import ExtractionMetadata
from mellea_lrc.model.record import CitationRecord
from mellea_lrc.model.spans import Span
from mellea_lrc.preprocessing import PreprocessedDocument, preprocess
from tests.record_fixtures import read_citation

SAMPLE_TEXT = (
    "Under Norton v. Shelby County, 118 U.S. 425, 442 (1886), an unconstitutional "
    "act confers no rights. See also Fed. R. Civ. P. 72(a) and 28 U.S.C. § 636(b)(1)(A)."
)


def test_extract_from_plain_text_carries_the_preprocessing_through() -> None:
    reference = preprocess(SAMPLE_TEXT)
    result = extract_from_plain_text(SAMPLE_TEXT)
    assert isinstance(result, PreprocessedDocument)
    assert result.source_metadata == reference.source_metadata
    assert result.preprocessing_metadata == reference.preprocessing_metadata
    assert result.text == SAMPLE_TEXT
    assert result.citations


def test_extraction_takes_what_preprocessing_produced(tmp_path: Path) -> None:
    """The stage's signature: the preceding stage's output in, citations out."""
    path = tmp_path / "filing.txt"
    path.write_text(SAMPLE_TEXT, encoding="utf-8")

    from_disk = extract_citations(preprocess(path))

    assert from_disk.text == SAMPLE_TEXT
    assert {item.fields.kind for item in from_disk.citations} == {
        item.fields.kind for item in extract_from_plain_text(SAMPLE_TEXT).citations
    }


def test_extract_from_plain_text_returns_canonical_types() -> None:
    result = extract_from_plain_text(SAMPLE_TEXT)
    kinds = {item.fields.kind for item in result.citations}

    assert CitationKind.FULL_CASE in kinds
    assert CitationKind.FULL_LAW in kinds
    assert len(result.full_citations) >= 2

    full_case = next(item for item in result.citations if isinstance(item.fields, FullCaseCitation))
    assert full_case.fields.defendant == "Shelby County"
    assert full_case.fields.volume == "118"
    assert full_case.fields.reporter.as_written == "U.S."
    assert SAMPLE_TEXT[full_case.locator_span.start : full_case.locator_span.end] == "118 U.S. 425"
    assert full_case.full_span.start < full_case.locator_span.start
    assert full_case.resolves_to is None

    full_law = next(item for item in result.citations if isinstance(item.fields, FullLawCitation))
    assert full_law.fields.volume == "28"
    assert full_law.fields.reporter.as_written == "U.S.C."


def test_document_rejects_duplicate_citation_ids() -> None:
    preprocessed = preprocess("347 U.S. 483")
    citation = read_citation(
        citation_id="cite-1",
        fields=placed(
            FullCaseCitation(volume="347", reporter="U.S.", page="483"),
            span=Span(0, len(preprocessed.text)),
            locator_span=Span(0, len(preprocessed.text)),
            matched_text=preprocessed.text,
        ),
    )

    with pytest.raises(ValueError, match="must be unique"):
        Document(
            source_metadata=preprocessed.source_metadata,
            text=preprocessed.text,
            preprocessing_metadata=preprocessed.preprocessing_metadata,
            citations=(citation, citation),
            extraction_metadata=ExtractionMetadata(),
        )


def test_document_rejects_span_outside_text() -> None:
    preprocessed = preprocess("347 U.S. 483")
    citation = read_citation(
        citation_id="cite-1",
        fields=placed(
            FullCaseCitation(volume="347", reporter="U.S.", page="483"),
            span=Span(0, len(preprocessed.text) + 1),
            locator_span=Span(0, len(preprocessed.text) + 1),
            matched_text=preprocessed.text,
        ),
    )

    with pytest.raises(ValueError, match="span exceeds"):
        Document(
            source_metadata=preprocessed.source_metadata,
            text=preprocessed.text,
            preprocessing_metadata=preprocessed.preprocessing_metadata,
            citations=(citation,),
            extraction_metadata=ExtractionMetadata(),
        )


def test_extract_recovers_citation_broken_by_repeated_whitespace() -> None:
    # Docling PDF extraction leaves runs of repeated spaces (justified-text
    # artifacts) that eyecite's literal single spaces break on outright. The
    # shipped relaxation matches them where they are, so the text is never
    # rewritten and the span needs no remapping.
    text = (
        "The court in Cracker Barrel Old  Country  Store,  Inc.  v.  Epperson ,  "
        "284  S.W.3d  303,  312 (Tenn. 2009) held as much."
    )
    result = extract_from_plain_text(text)

    assert len(result.citations) == 1
    citation = result.citations[0]
    assert isinstance(citation.fields, FullCaseCitation)
    assert citation.fields.volume == "284"
    assert citation.fields.reporter.as_written == "S.W.3d"
    assert text[citation.locator_span.start : citation.locator_span.end] == "284  S.W.3d  303"


def test_a_citation_id_is_decided_by_the_citation(tmp_path: Path) -> None:
    """Read the same text twice and the ids are the same, so downstream keys hold."""
    once = extract_from_plain_text(SAMPLE_TEXT)
    twice = extract_from_plain_text(SAMPLE_TEXT)

    assert [item.citation_id for item in once.citations] == [item.citation_id for item in twice.citations]
    assert all(item.citation_id for item in once.citations)


def test_a_citation_id_changes_when_the_text_does() -> None:
    """A different offset is a different citation, and must not keep the old key."""
    moved = extract_from_plain_text("A preface. " + SAMPLE_TEXT)
    original = {item.citation_id for item in extract_from_plain_text(SAMPLE_TEXT).citations}

    assert original.isdisjoint({item.citation_id for item in moved.citations})
