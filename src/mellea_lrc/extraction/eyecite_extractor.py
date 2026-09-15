"""Eyecite-backed citation extraction into canonical core representations."""

from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import replace
from functools import cache, lru_cache
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
from reporters_db import REPORTERS

from mellea_lrc.core.case_names import CaseName
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
    citation_kind,
    is_leaf,
)
from mellea_lrc.core.findings import Finding, FindingKind
from mellea_lrc.core.pin_cites import PinCite
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.identity import citation_id as citation_id_for
from mellea_lrc.extraction.reading.case_names import locate_case_name
from mellea_lrc.extraction.reading.courts import court_from_reporter
from mellea_lrc.extraction.reading.dockets import DOCKET_GROUP, with_dockets
from mellea_lrc.extraction.reading.pin_cite_spans import locate_pin_cite
from mellea_lrc.extraction.reading.pin_cites import relaxed_pin_cites, strip_connector
from mellea_lrc.extraction.reading.relaxation import Relaxation, tokenizer_for
from mellea_lrc.extraction.reading.unread_names import unread_case_names
from mellea_lrc.extraction.stages import refine
from mellea_lrc.extraction.structure.attachment import Attachment, root_for
from mellea_lrc.extraction.structure.withdrawal import withdraw_leaves_of_withdrawn_roots
from mellea_lrc.extraction.types import CitationRecord, Document, ExtractionMetadata
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

    The edition is taken only when the abbreviation names exactly one. eyecite
    would break a tie by year, which fails in both directions -- see
    :class:`~mellea_lrc.core.citations.Reporter` -- so an ambiguous abbreviation
    is recorded as ambiguous and left for a reviewer.
    """
    as_written = citation.groups.get("reporter")
    if not as_written:
        return None
    candidates = list(getattr(citation, "exact_editions", ()) or getattr(citation, "variation_editions", ()))
    reporters = [getattr(edition, "reporter", None) for edition in candidates]
    names = tuple(getattr(reporter, "name", "") or "" for reporter in reporters)

    def agreed(attribute: str, default: object) -> object:
        """The value every candidate gives, or the default when they differ."""
        values = {getattr(reporter, attribute, None) for reporter in reporters if reporter}
        return values.pop() if len(values) == 1 else default

    if len(candidates) == 1:
        edition, reporter = candidates[0], reporters[0]
        return Reporter(
            as_written=as_written,
            short_name=edition.short_name,
            name=getattr(reporter, "name", None),
            cite_type=getattr(reporter, "cite_type", None),
            is_scotus=bool(getattr(reporter, "is_scotus", False)),
            editions=names,
        )
    return Reporter(
        as_written=as_written,
        short_name=None,
        name=None,
        cite_type=agreed("cite_type", None),
        is_scotus=bool(agreed("is_scotus", False)),
        editions=names,
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
        pin_cite=strip_connector(citation.metadata.pin_cite),
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
        pin_cite=strip_connector(citation.metadata.pin_cite),
        extra=citation.metadata.extra,
        date=_date(citation),
        court=_court(citation),
        parenthetical=citation.metadata.parenthetical,
        antecedent=citation.metadata.antecedent_guess,
    )


def _court(citation: CitationBase) -> str | None:
    """The court, from what the filing writes or from the reporter it cites.

    eyecite reads a court out of the parenthetical. Where a filing writes none,
    the reporter can still name one -- `556 U.S. 662 (2009)` is the Supreme
    Court's, `7 Kan. 2d 1` the Kansas Supreme Court's -- and a reader knows it
    without being told. See
    :func:`~mellea_lrc.extraction.reading.courts.court_from_reporter`, which
    names a court only where the reporter is one court's.
    """
    written = citation.metadata.court
    if written:
        return written
    edition = citation.groups.get("reporter") if hasattr(citation, "groups") else None
    entry = _reporter_entry(edition)
    return court_from_reporter(
        edition,
        cite_type=entry.get("cite_type") if entry else None,
        name=entry.get("name") if entry else None,
    )


@cache
def _reporter_entry(edition: str | None) -> dict | None:
    """What reporters-db knows about the reporter this edition belongs to."""
    if not edition:
        return None
    for entries in REPORTERS.values():
        for entry in entries:
            if edition in entry["editions"]:
                return entry
    return None


def _to_full_law(citation: EyeciteFullLawCitation) -> FullLawCitation:
    return FullLawCitation(
        volume=citation.groups.get("title"),
        reporter=_reporter(citation),
        page=citation.groups.get("section"),
        pin_cite=strip_connector(citation.metadata.pin_cite),
        date=_date(citation),
        publisher=citation.metadata.publisher,
        parenthetical=citation.metadata.parenthetical,
    )


def _to_full_journal(citation: EyeciteFullJournalCitation) -> FullJournalCitation:
    return FullJournalCitation(
        volume=citation.groups.get("volume"),
        reporter=_reporter(citation),
        page=citation.groups.get("page"),
        pin_cite=strip_connector(citation.metadata.pin_cite),
        date=_date(citation),
        parenthetical=citation.metadata.parenthetical,
    )


def _case_name(
    text: str,
    eyecite_citation: CitationBase,
    locator_span: Span,
    floor: int,
    canonical: CanonicalCitation,
) -> CaseName | None:
    """The name this citation is written under, with the parties eyecite read.

    The span is located in the document; the parties are eyecite's own parse,
    carried here so a later reading of the name replaces all of it at once
    rather than leaving a repaired span beside a stale party.
    """
    span = locate_case_name(text, eyecite_citation, locator_span, floor=floor)
    if span is None:
        return None
    return CaseName(
        span=span,
        text=text[span.start : span.end],
        plaintiff=getattr(canonical, "plaintiff", None),
        defendant=getattr(canonical, "defendant", None),
    )


def _locator_bounds(citation: CitationBase) -> tuple[int, int]:
    """Where the citation's identifier sits in the document.

    `span()` for a short form runs to the end of the page, because a short
    form's page is part of its identifier -- `695 F.Supp.2d at 1154`. When the
    pin cite pattern after the page fails, the span stops at `at ` instead and
    the page falls outside it, which leaves a locator that identifies nothing
    and a page with nowhere to point. `span_with_pincite()` covers the page in
    both cases, so it is what a short form is measured by.

    Every other kind keeps `span()`. A full citation's span is the identifier
    alone and `span_with_pincite()` would pull the pin cite into it, which is
    the distinction `locator_span` exists to make.
    """
    if isinstance(citation, EyeciteShortCaseCitation):
        return citation.span_with_pincite()
    return citation.span()


def _to_short_case(citation: EyeciteShortCaseCitation) -> ShortCaseCitation:
    # A short form's page *is* its pin cite -- `550 U.S. at 570` claims page 570
    # and identifies no page of its own -- and eyecite says so on the path that
    # works, reading the page out of the locator and passing it back through
    # `extract_pin_cite`. When the pattern after the page fails, that step is
    # skipped and `metadata.pin_cite` is `None` while `groups["page"]` still
    # holds the page: `645 B.R. at 184 (quoting …)` and `550 U.S. at 570 n.4`
    # both come back with no pin cite at all. The page is taken directly there.
    #
    # What the page cannot recover is the rest of a range: eyecite parses
    # `570-71 (quoting …)` to `page="570"` and the `-71` is in no parse, so the
    # claim narrows to its first page. That is a loss already taken, not one
    # made here.
    return ShortCaseCitation(
        volume=citation.groups.get("volume"),
        reporter=_reporter(citation),
        page=citation.groups.get("page"),
        pin_cite=strip_connector(citation.metadata.pin_cite) or citation.groups.get("page"),
        court=citation.metadata.court,
        date=_date(citation),
        parenthetical=citation.metadata.parenthetical,
        antecedent=citation.metadata.antecedent_guess,
    )


def _to_supra(citation: EyeciteSupraCitation) -> SupraCitation:
    return SupraCitation(
        volume=citation.metadata.volume,
        pin_cite=strip_connector(citation.metadata.pin_cite),
        parenthetical=citation.metadata.parenthetical,
        antecedent=citation.metadata.antecedent_guess,
    )


def _to_id(citation: EyeciteIdCitation) -> IdCitation:
    return IdCitation(
        pin_cite=strip_connector(citation.metadata.pin_cite),
        parenthetical=citation.metadata.parenthetical,
    )


def _to_reference(citation: EyeciteReferenceCitation) -> ReferenceCitation:
    return ReferenceCitation(
        plaintiff=citation.metadata.plaintiff,
        defendant=citation.metadata.defendant,
        pin_cite=strip_connector(citation.metadata.pin_cite),
        parenthetical=citation.metadata.parenthetical,
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
        start, end = citation.span()
        citation_ids.append((citation, citation_id_for(Span(start=start, end=end), citation.matched_text())))
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


def extract_citations(
    preprocessed: PreprocessedDocument,
    *,
    relaxation: Relaxation = Relaxation.FULL,
    with_leaves: bool = False,
    attach: Attachment = Attachment.STATED,
) -> Document:
    """Extract canonical citations from a preprocessed document.

    This is the extraction stage's entry point. Preprocessing is a stage of its
    own and runs first: converting a PDF, deciding which page furniture is not
    the document's text, settling the coordinate space every span will index.
    Extraction takes what that produced and reads citations out of it.

    The text is tokenized as it stands. Nothing is rewritten before parsing and
    no span is remapped afterwards, so every offset indexes straight into
    ``preprocessed.text``: how much separator damage a citation may carry is
    entirely a property of ``relaxation``, and of nothing else.

    **Roots only, unless ``with_leaves``.** A leaf's meaning is which root it
    points at, and attaching one means matching a case name -- against names
    that are whatever the parser made of them here. The leaves are grown
    afterwards, over the roots validation admitted, by
    :func:`~mellea_lrc.extraction.grow_leaves`. ``with_leaves`` does both in one
    call and is what a caller with no validation stage in the loop wants; it is
    the same leaf pass, given every root instead of the admitted ones. See
    `docs/Extraction.md`, "Roots first, leaves after validation".
    """
    document, leaves = _read(preprocessed, relaxation)
    return _with_leaves(document, leaves, attach) if with_leaves else document


def _read(
    preprocessed: PreprocessedDocument, relaxation: Relaxation
) -> tuple[Document, list[tuple[str, CanonicalCitation, str | None]]]:
    """One read of the document: the roots as a document, and the leaves apart.

    Both growths go through here, so they read the text exactly the same way and
    the only difference between them is which roots the leaves are offered.
    """

    text = preprocessed.text
    # A relaxed level reads pin cites tolerantly as well as reporter joins: the
    # same literal single space breaks both, and losing a pin cite loses the
    # page a filing argues from. NONE is left strict so it stays eyecite exactly
    # as published, which is what the evaluation baseline means by the name.
    # See :mod:`mellea_lrc.extraction.reading.pin_cites`.
    with contextlib.ExitStack() as stack:
        if relaxation is not Relaxation.NONE:
            stack.enter_context(relaxed_pin_cites(relaxation))
        eyecite_citations = get_citations(text, tokenizer=with_dockets(tokenizer_for(relaxation)))
        # Resolution is inside the block because it reads pin cites too: it
        # tests an `Id.`'s page against the citation it would attach to, with a
        # pattern that counts spaces. Leaving it outside read the page
        # tolerantly and then threw the attribution away.
        resolutions = cast(
            dict[Resource, list[CitationBase]],
            resolve_citations(eyecite_citations),
        )
    citation_ids = _assign_citation_ids(eyecite_citations)
    antecedent_map = _build_antecedent_map(resolutions, citation_ids)

    extracted: list[CitationRecord] = []
    leaves: list[tuple[str, CanonicalCitation, str | None]] = []
    # Where the citation before this one stopped, so a name cannot open inside it.
    name_floor = 0
    for eyecite_citation, citation_id in citation_ids:
        span_start, span_end = eyecite_citation.full_span()
        locator_start, locator_end = _locator_bounds(eyecite_citation)
        full_span = Span(start=span_start, end=span_end)
        locator_span = Span(start=locator_start, end=locator_end)
        canonical = to_canonical(eyecite_citation)
        # The citation carries where it is written. `pin_cite` arrives from the
        # parse as the string the filing wrote; it becomes a `PinCite` here,
        # with the position located against the document and the pages read
        # from the same characters, so the three cannot disagree.
        pin_cite_span = locate_pin_cite(text, canonical, locator_span=locator_span, full_span=full_span)
        written = getattr(canonical, "pin_cite", None)
        canonical = dataclasses.replace(
            canonical,
            span=full_span,
            locator_span=locator_span,
            matched_text=eyecite_citation.matched_text(),
            case_name=_case_name(text, eyecite_citation, locator_span, name_floor, canonical),
            **(
                {"pin_cite": PinCite.read(written, pin_cite_span)}
                if isinstance(written, str) and written
                else {}
            ),
        )
        # **Roots only.** A leaf's meaning is which root it points at, and
        # attaching one means matching a case name -- against names that are
        # whatever the parser made of them at this point. The leaves are grown
        # after validation has admitted the roots and settled their names. So
        # nothing leaf-shaped is emitted here, not even a span with the kind
        # left off: that would be a leaf-shaped hole, and the invariant
        # `CitationRecord` enforces would be a convention again. See
        # `docs/Extraction.md`, "Roots first, leaves after validation".
        if is_leaf(canonical):
            leaves.append((citation_id, canonical, antecedent_map.get(citation_id)))
        else:
            extracted.append(
                CitationRecord(
                    citation_id=citation_id,
                    source=canonical,
                    resolves_to=antecedent_map.get(citation_id),
                )
            )
        name_floor = max(name_floor, eyecite_citation.full_span()[1])

    # The passes over the citation list are ordered, and one reads what another
    # writes. See :mod:`mellea_lrc.extraction.stages` for the sequence and the
    # constraint behind it.
    refined = refine(text, extracted)
    document = Document(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=refined,
        unread_case_names=unread_case_names(text, refined),
        passes=("roots",),
        extraction_metadata=ExtractionMetadata(relaxation=relaxation),
    )
    return document, leaves


def _with_leaves(
    document: Document,
    leaves: list[tuple[str, CanonicalCitation, str | None]],
    attach: Attachment = Attachment.STATED,
) -> Document:
    """Attach each leaf to a root the document holds, or drop it.

    **Decided from `stated`**, by
    :func:`~mellea_lrc.extraction.structure.attachment.root_for`, and not from
    eyecite's own resolution. eyecite decides while parsing, against the party
    names it read; a root it named `Cnty.` is a root no short form can find, and
    the same root named properly is one that `Huri , 804 F.3d at 833` reaches.
    Attaching against the parse would throw away every correction a reader or
    validation has since made, which is the whole reason the leaves are grown in
    a pass of their own. eyecite's answer is kept beside ours in `resolves_to`,
    so the two can be compared.

    A leaf `root_for` cannot decide is **not built, and is reported**.
    `CitationRecord` refuses a leaf with no root, and a leaf attached to a guess
    is worse than a leaf that is not there -- but a leaf with no root to reach is
    a finding either way: the root is in the document and was not read, or there
    is no root and the filing cites a case it never gives in full. So it goes to
    `findings` rather than to nothing. See :mod:`mellea_lrc.core.findings`.

    Each leaf is settled before the next is read, because an `Id.` means the
    authority of the citation before it and that citation is often another leaf.

    `attach` chooses which of the two decides. `Attachment.EYECITE` takes
    eyecite's answer instead of ours and is the baseline the second growth is
    measured against; it changes nothing else, so a difference between the two
    runs is the attachment and only the attachment.
    """
    # A document with no roots is not a document with no leaves: every leaf in
    # it is one that reached nothing, and that is exactly what a filing citing
    # only short forms looks like. So the loop runs either way and the findings
    # are the whole of what comes out.
    roots = [record for record in document.citations if not is_leaf(record.stated)]
    by_id = {record.citation_id: record for record in roots}
    settled = sorted(roots, key=lambda record: record.full_span.start)
    grown: list[CitationRecord] = []
    ungrown: list[Finding] = []
    for citation_id, canonical, antecedent in sorted(leaves, key=lambda item: item[1].span.start):
        if attach is Attachment.EYECITE:
            root_id = antecedent if antecedent in by_id else None
        else:
            before = [r for r in settled if r.full_span.start < canonical.span.start]
            root_id = root_for(canonical, roots, before=before)
        if root_id is None:
            ungrown.append(
                Finding(
                    kind=FindingKind.UNGROWN_LEAF,
                    stage="extraction",
                    made_by="mellea_lrc.extraction.structure.attachment",
                    message=(
                        f"{citation_kind(canonical).value} reaching no root this document "
                        "holds, so no leaf was built"
                    ),
                    span=canonical.span,
                    citation=canonical,
                )
            )
            continue
        leaf = CitationRecord(
            citation_id=citation_id, source=canonical, root_id=root_id, resolves_to=antecedent
        )
        grown.append(leaf)
        settled = sorted([*settled, leaf], key=lambda record: record.full_span.start)
    citations = sorted((*document.citations, *grown), key=lambda record: record.full_span.start)
    grown_document = dataclasses.replace(
        document,
        citations=tuple(citations),
        unread_case_names=unread_case_names(document.text, citations),
        findings=(*document.findings, *ungrown),
        passes=_after("leaves", document),
    )
    # A leaf grown onto a root that has since been withdrawn is withdrawn with
    # it, keeping the `root_id` that says which root took it. Attaching first
    # and sweeping after is one rule rather than two: the same sweep runs when a
    # root is withdrawn after its leaves already exist.
    withdraw_leaves_of_withdrawn_roots(grown_document)
    return grown_document


def _after(name: str, document: Document) -> tuple[str, ...]:
    """The passes with this one added, and not added twice.

    Growing the leaves over a document that already has them is one pass that
    ran again, not two passes.
    """
    return document.passes if name in document.passes else (*document.passes, name)


def grow_leaves(
    document: Document,
    *,
    relaxation: Relaxation | None = None,
    attach: Attachment = Attachment.STATED,
) -> Document:
    """Attach every leaf the document's text writes to a root the document holds.

    The second growth. The roots came from :func:`extract_citations` and have
    since been through validation's identity stage, so each one that survived is
    a root with an authority behind it and a case name that was checked rather
    than parsed -- which is what a leaf needs, because reaching a root means
    matching a name.

    **The roots the document holds are kept exactly as they are**, with whatever
    validation wrote on them: a citation's identifier is a hash of its span and
    the characters at it, so the re-read produces the same ids and the roots
    already here are the ones the leaves point at. A leaf whose root the document
    does not hold -- removed as unidentifiable, or never found -- is dropped
    rather than recorded pointing nowhere.
    """
    level = relaxation or document.extraction_metadata.relaxation
    preprocessed = PreprocessedDocument(
        source_metadata=document.source_metadata,
        text=document.text,
        preprocessing_metadata=document.preprocessing_metadata,
    )
    _, leaves = _read(preprocessed, level)
    known = {record.citation_id for record in document.citations}
    return _with_leaves(document, [leaf for leaf in leaves if leaf[0] not in known], attach)


def extract_from_plain_text(
    text: str,
    *,
    source_path: str | None = None,
    relaxation: Relaxation = Relaxation.FULL,
    with_leaves: bool = False,
    attach: Attachment = Attachment.STATED,
) -> Document:
    """Extract citations from Layer 2 plain text.

    Spans index into ``text`` as given, so a caller that already holds the text
    can map results straight back onto it.

    ``relaxation`` chooses how much separator damage a citation may carry and
    still be found; see :class:`~mellea_lrc.extraction.reading.relaxation.Relaxation`.
    """
    preprocessed = preprocess(text)
    if source_path is not None:
        preprocessed = replace(
            preprocessed,
            source_metadata=replace(preprocessed.source_metadata, path=source_path),
        )
    return extract_citations(
        preprocessed, relaxation=relaxation, with_leaves=with_leaves, attach=attach
    )
