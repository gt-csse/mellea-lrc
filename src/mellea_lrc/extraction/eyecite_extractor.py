"""Eyecite-backed citation extraction into canonical core representations."""

from __future__ import annotations

import contextlib
import uuid
from typing import cast

from eyecite import get_citations, resolve_citations
from eyecite.models import (
    CitationBase,
    Resource,
)
from eyecite.models import (
    FullCaseCitation as EyeciteFullCaseCitation,
)
from eyecite.models import (
    FullJournalCitation as EyeciteFullJournalCitation,
)
from eyecite.models import (
    FullLawCitation as EyeciteFullLawCitation,
)
from eyecite.models import (
    IdCitation as EyeciteIdCitation,
)
from eyecite.models import (
    ReferenceCitation as EyeciteReferenceCitation,
)
from eyecite.models import (
    ShortCaseCitation as EyeciteShortCaseCitation,
)
from eyecite.models import (
    SupraCitation as EyeciteSupraCitation,
)
from eyecite.models import (
    UnknownCitation as EyeciteUnknownCitation,
)

from mellea_lrc.core.citations import (
    CanonicalCitation,
    CitationDate,
    DocketCitation,
    FullCaseCitation,
    FullJournalCitation,
    FullLawCitation,
    IdCitation,
    ReferenceCitation,
    Reporter,
    ShortCaseCitation,
    SupraCitation,
    UnknownCitation,
)
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.reading.dockets import DOCKET_GROUP, with_dockets
from mellea_lrc.extraction.reading.pin_cites import relaxed_pin_cites
from mellea_lrc.extraction.reading.relaxation import Relaxation, tokenizer_for
from mellea_lrc.extraction.stages import refine
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.preprocessing.types import PreprocessedDocument

EYECITE_CITATION_TYPES = frozenset(
    {
        EyeciteFullCaseCitation,
        EyeciteFullLawCitation,
        EyeciteFullJournalCitation,
        EyeciteShortCaseCitation,
        EyeciteSupraCitation,
        EyeciteIdCitation,
        EyeciteReferenceCitation,
        EyeciteUnknownCitation,
    }
)


def _reporter(citation: CitationBase) -> Reporter | None:
    """The reporter this citation names, with what reporters-db knows about it.

    `corrected_reporter` and `edition_guess` are eyecite's, and this project
    used to discard both and keep the raw spelling -- so two spellings of one
    reporter compared as two reporters.
    """
    as_written = citation.groups.get("reporter")
    if not as_written:
        return None
    edition = getattr(citation, "edition_guess", None)
    reporter = getattr(edition, "reporter", None)
    return Reporter(
        as_written=as_written,
        short_name=citation.corrected_reporter(),
        name=getattr(reporter, "name", None),
        cite_type=getattr(reporter, "cite_type", None),
        is_scotus=bool(getattr(reporter, "is_scotus", False)),
    )


def _date(citation: CitationBase) -> CitationDate | None:
    """The decision date the citation states, or None when it states none.

    eyecite parses the month and day of a full date and this project used to
    drop both, keeping only the year.
    """
    metadata = citation.metadata
    year = getattr(metadata, "year", None)
    if not year:
        return None
    return CitationDate(
        year=str(year),
        month=getattr(metadata, "month", None),
        day=getattr(metadata, "day", None),
    )


def _to_docket(citation: EyeciteFullCaseCitation) -> DocketCitation:
    """Read back the docket a custom extractor wrote into a case citation.

    eyecite has one shape for a case citation and reaches it through a reporter
    edition, so a docket arrives here wearing volume, reporter and page. The
    docket number and its court were carried along in the token's groups; this
    unpacks them and drops the borrowed clothes.
    """
    return DocketCitation(
        plaintiff=citation.metadata.plaintiff,
        defendant=citation.metadata.defendant,
        docket_number=citation.groups.get(DOCKET_GROUP),
        court=citation.groups.get("court"),
        court_name=citation.groups.get("court_name"),
        court_text=citation.groups.get("court_text"),
        pin_cite=citation.metadata.pin_cite,
        date=_date(citation),
        parenthetical=citation.metadata.parenthetical,
    )


def _to_full_case(citation: EyeciteFullCaseCitation) -> FullCaseCitation:
    return FullCaseCitation(
        plaintiff=citation.metadata.plaintiff,
        defendant=citation.metadata.defendant,
        volume=citation.groups.get("volume"),
        reporter=_reporter(citation),
        page=citation.groups.get("page"),
        pin_cite=citation.metadata.pin_cite,
        extra=citation.metadata.extra,
        date=_date(citation),
        court=citation.metadata.court,
        parenthetical=citation.metadata.parenthetical,
    )


def _to_full_law(citation: EyeciteFullLawCitation) -> FullLawCitation:
    return FullLawCitation(
        volume=citation.groups.get("title"),
        reporter=_reporter(citation),
        page=citation.groups.get("section"),
        pin_cite=citation.metadata.pin_cite,
        date=_date(citation),
        publisher=citation.metadata.publisher,
        parenthetical=citation.metadata.parenthetical,
    )


def _to_full_journal(citation: EyeciteFullJournalCitation) -> FullJournalCitation:
    return FullJournalCitation(
        volume=citation.groups.get("volume"),
        reporter=_reporter(citation),
        page=citation.groups.get("page"),
        pin_cite=citation.metadata.pin_cite,
        date=_date(citation),
        parenthetical=citation.metadata.parenthetical,
    )


def _to_short_case(citation: EyeciteShortCaseCitation) -> ShortCaseCitation:
    return ShortCaseCitation(
        volume=citation.groups.get("volume"),
        reporter=_reporter(citation),
        page=citation.groups.get("page"),
        pin_cite=citation.metadata.pin_cite,
        court=citation.metadata.court,
        parenthetical=citation.metadata.parenthetical,
    )


def _to_supra(citation: EyeciteSupraCitation) -> SupraCitation:
    return SupraCitation(
        pin_cite=citation.metadata.pin_cite,
        parenthetical=citation.metadata.parenthetical,
    )


def _to_id(citation: EyeciteIdCitation) -> IdCitation:
    return IdCitation(
        pin_cite=citation.metadata.pin_cite,
        parenthetical=citation.metadata.parenthetical,
    )


def _to_reference(citation: EyeciteReferenceCitation) -> ReferenceCitation:
    return ReferenceCitation(
        plaintiff=citation.metadata.plaintiff,
        defendant=citation.metadata.defendant,
    )


def _to_unknown(_citation: EyeciteUnknownCitation) -> UnknownCitation:
    return UnknownCitation()


def to_canonical(citation: CitationBase) -> CanonicalCitation:
    """Convert one eyecite citation into this project's canonical representation.

    Public because the adjudication layer re-reads a confirmed candidate through
    eyecite and needs this same conversion rather than a second one that would
    drift from it.
    """
    # Tested before the case branch, not instead of it: a docket citation *is*
    # an eyecite full case citation, and only the group written by the docket
    # extractor tells the two apart.
    if DOCKET_GROUP in citation.groups and isinstance(citation, EyeciteFullCaseCitation):
        return _to_docket(citation)
    if isinstance(citation, EyeciteFullCaseCitation):
        return _to_full_case(citation)
    if isinstance(citation, EyeciteFullLawCitation):
        return _to_full_law(citation)
    if isinstance(citation, EyeciteFullJournalCitation):
        return _to_full_journal(citation)
    if isinstance(citation, EyeciteShortCaseCitation):
        return _to_short_case(citation)
    if isinstance(citation, EyeciteSupraCitation):
        return _to_supra(citation)
    if isinstance(citation, EyeciteIdCitation):
        return _to_id(citation)
    if isinstance(citation, EyeciteReferenceCitation):
        return _to_reference(citation)
    if isinstance(citation, EyeciteUnknownCitation):
        return _to_unknown(citation)
    msg = f"Unknown citation type: {type(citation).__name__}"
    raise TypeError(msg)


def _assign_citation_ids(
    citations: list[CitationBase],
) -> list[tuple[CitationBase, str]]:
    citation_ids: list[tuple[CitationBase, str]] = []
    for citation in citations:
        if type(citation) not in EYECITE_CITATION_TYPES:
            msg = (
                f"Unknown citation type: {type(citation).__name__}. "
                "All citation types must be handled explicitly."
            )
            raise ValueError(msg)
        citation_ids.append((citation, str(uuid.uuid4())[:8]))
    return citation_ids


def _build_antecedent_map(
    resolutions: dict[Resource, list[CitationBase]],
    citation_ids: list[tuple[CitationBase, str]],
) -> dict[str, str]:
    """Map reference citation ids to their resolved full citation id."""
    citation_to_id = {id(citation): citation_id for citation, citation_id in citation_ids}
    antecedent_map: dict[str, str] = {}
    for grouped in resolutions.values():
        full_citation_id = citation_to_id[id(grouped[0])]
        for reference in grouped[1:]:
            reference_id = citation_to_id[id(reference)]
            antecedent_map[reference_id] = full_citation_id
    return antecedent_map


def _extract_from_text(
    preprocessed: PreprocessedDocument,
    *,
    relaxation: Relaxation = Relaxation.BOUNDED,
) -> ExtractedDocument:
    """Extract canonical citations from a preprocessed document.

    The text is tokenized as it stands. Nothing is rewritten before parsing and
    no span is remapped afterwards, so every offset indexes straight into
    ``preprocessed.text``: how much separator damage a citation may carry is
    entirely a property of ``relaxation``, and of nothing else.
    """
    text = preprocessed.text
    # A relaxed level reads pin cites tolerantly as well as reporter joins: the
    # same literal single space breaks both, and losing a pin cite loses the
    # page a filing argues from. NONE is left strict so it stays eyecite exactly
    # as published, which is what the evaluation baseline means by the name.
    # See :mod:`mellea_lrc.extraction.reading.pin_cites`.
    with contextlib.ExitStack() as stack:
        if relaxation is not Relaxation.NONE:
            stack.enter_context(relaxed_pin_cites())
        eyecite_citations = get_citations(text, tokenizer=with_dockets(tokenizer_for(relaxation)))
    resolutions = cast(
        dict[Resource, list[CitationBase]],
        resolve_citations(eyecite_citations),
    )
    citation_ids = _assign_citation_ids(eyecite_citations)
    antecedent_map = _build_antecedent_map(resolutions, citation_ids)

    extracted: list[ExtractedCitation] = []
    for eyecite_citation, citation_id in citation_ids:
        span_start, span_end = eyecite_citation.full_span()
        locator_start, locator_end = eyecite_citation.span()
        extracted.append(
            ExtractedCitation(
                citation_id=citation_id,
                span=Span(start=span_start, end=span_end),
                locator_span=Span(start=locator_start, end=locator_end),
                matched_text=eyecite_citation.matched_text(),
                citation=to_canonical(eyecite_citation),
                resolves_to=antecedent_map.get(citation_id),
            )
        )

    # The passes over the citation list are ordered, and one reads what another
    # writes. See :mod:`mellea_lrc.extraction.stages` for the sequence and the
    # constraint behind it.
    return ExtractedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=refine(text, extracted),
        extraction_metadata=ExtractionMetadata(relaxation=relaxation),
    )


def extract_from_plain_text(
    text: str,
    *,
    source_path: str | None = None,
    relaxation: Relaxation = Relaxation.BOUNDED,
) -> ExtractedDocument:
    """Extract citations from Layer 2 plain text.

    Spans index into ``text`` as given, so a caller that already holds the text
    can map results straight back onto it.

    ``relaxation`` chooses how much separator damage a citation may carry and
    still be found; see :class:`~mellea_lrc.extraction.reading.relaxation.Relaxation`.
    """
    preprocessed = preprocess(text)
    return _extract_from_text(preprocessed, relaxation=relaxation)
