"""Tests for grouping a document's citations under the authority they cite."""

from __future__ import annotations

import pytest

from mellea_lrc.extraction.structure.citation_tree import build_citation_tree
from mellea_lrc.model.citations import (
    DocketCitation,
    FullCaseCitation,
    FullLawCitation,
    IdCitation,
    ShortCaseCitation,
    placed,
)
from mellea_lrc.model.document import Document
from mellea_lrc.model.extraction_metadata import ExtractionMetadata
from mellea_lrc.model.pin_cites import PinCite
from mellea_lrc.model.record import CitationRecord
from mellea_lrc.model.spans import Span
from mellea_lrc.preprocessing import preprocess
from tests.record_fixtures import read_citation


def _document(*citations: CitationRecord, text: str = "x" * 400) -> Document:
    source = preprocess(text)
    return Document(
        source_metadata=source.source_metadata,
        preprocessing_metadata=source.preprocessing_metadata,
        text=source.text,
        citations=citations,
        extraction_metadata=ExtractionMetadata(),
    )


def _full(citation_id: str, page: str, pin: str | None, start: int) -> CitationRecord:
    return read_citation(
        citation_id=citation_id,
        root_id=citation_id,
        fields=placed(
            FullCaseCitation(
                volume="550", reporter="U.S.", page=page, pin_cite=PinCite.read(pin) if pin else None
            ),
            span=Span(start, start + 12),
            locator_span=Span(start, start + 12),
            matched_text=f"550 U.S. {page}",
        ),
    )


def _short(citation_id: str, pin: str, resolves_to: str, start: int) -> CitationRecord:
    return read_citation(
        citation_id=citation_id,
        resolves_to=resolves_to,
        root_id=resolves_to,
        fields=placed(
            ShortCaseCitation(volume="550", reporter="U.S.", page=pin, pin_cite=PinCite.read(f"at {pin}")),
            span=Span(start, start + 12),
            locator_span=Span(start, start + 12),
            matched_text=f"550 U.S. at {pin}",
        ),
    )


def _id(
    citation_id: str, pin: str, resolves_to: str, start: int, *, root_id: str | None = None
) -> CitationRecord:
    return read_citation(
        citation_id=citation_id,
        resolves_to=resolves_to,
        root_id=root_id or resolves_to,
        fields=placed(
            IdCitation(pin_cite=PinCite.read(f"at {pin}")),
            span=Span(start, start + 8),
            locator_span=Span(start, start + 8),
            matched_text=f"Id. at {pin}",
        ),
    )


def test_every_reference_gathers_under_the_authority_it_cites() -> None:
    """A brief cites a case once in full and returns to it; all of that is one authority."""
    document = _document(
        _full("c1", "544", "555", 0),
        _short("c2", "563", "c1", 100),
        _id("c3", "570", "c2", 200, root_id="c1"),
    )

    tree = build_citation_tree(document)

    (authority,) = tree.roots
    assert authority.root_id == "c1"
    assert [o.citation_id for o in authority.occurrences] == ["c1", "c2", "c3"]
    assert tree.unattributed == ()


def test_resolution_is_followed_through_a_short_form() -> None:
    """`Id.` points at the short form before it, not at the full citation."""
    document = _document(
        _full("c1", "544", None, 0),
        _short("c2", "563", "c1", 100),
        _id("c3", "570", "c2", 200, root_id="c1"),
    )

    (authority,) = build_citation_tree(document).roots
    depths = {o.citation_id: o.depth for o in authority.occurrences}

    assert depths == {"c1": 0, "c2": 1, "c3": 2}


def test_each_reference_keeps_the_page_it_names() -> None:
    """This is the point: one authority, several separate claims about separate pages.

    Validating only the full citation checks the first of these and skips the
    rest, which is most of what the brief actually asserts about the case.
    """
    document = _document(
        _full("c1", "544", "555", 0),
        _short("c2", "563", "c1", 100),
        _id("c3", "570", "c2", 200, root_id="c1"),
    )

    (authority,) = build_citation_tree(document).roots

    assert authority.pin_cites == ("555", "at 563", "at 570")
    assert build_citation_tree(document).pinpoint_claim_count == 3


def test_one_page_cited_twice_is_one_claim() -> None:
    """Returning to the same page does not create a second thing to verify."""
    document = _document(_full("c1", "544", "555", 0), _short("c2", "555", "c1", 100))

    (authority,) = build_citation_tree(document).roots

    assert authority.pin_cites == ("555", "at 555")


def test_a_leaf_with_no_root_cannot_be_built_at_all() -> None:
    """What `unattributed` used to hold, the type now refuses.

    Two tests stood here: an `Id.` that resolved to nothing, and a short form
    for a case the filing never gives in full -- the corpus's `Rosenblatt v.
    Baer, 383 U.S. at 85`, quoted inside another case's parenthetical. Both
    described a leaf attached to nothing, and the tree reported them so that a
    claim would not be checked against the wrong page.

    A leaf is now built from a root or it is not built, so "attached to nothing"
    and "not there" are one state and the tree never sees the first. The filing
    still writes those characters and the ground truth still records them -- as
    a `nonconforming_citation`, which is what a case the document never
    identifies is. See `docs/Extraction.md`, "Roots first, leaves after
    validation".
    """
    with pytest.raises(ValueError, match="states no root"):
        _document(
            read_citation(
                citation_id="c9",
                fields=placed(
                    ShortCaseCitation(
                        volume="383", reporter="U.S.", page="85", pin_cite=PinCite.read("at 85")
                    ),
                    span=Span(0, 14),
                    locator_span=Span(0, 14),
                    matched_text="383 U.S. at 85",
                ),
            )
        )


def test_a_resolution_cycle_cannot_enter_a_document() -> None:
    """A cycle of leaves has no self-root and cannot become a checkpoint."""
    first = _short("c1", "563", "c2", 0)
    second = _short("c2", "570", "c1", 100)

    with pytest.raises(ValueError, match="missing or non-root"):
        _document(first, second)


def test_a_dangling_antecedent_cannot_reach_the_tree_at_all() -> None:
    """The document type refuses one, so the tree never has to decide about it.

    Worth pinning here rather than assuming: the tree still handles a missing
    antecedent defensively, but this is why that path is unreachable through an
    `Document` and why no citation can be attributed to an authority
    that was never extracted.
    """
    with pytest.raises(ValueError, match="invalid resolves_to"):
        _document(_full("c1", "544", None, 100), _id("c3", "570", "gone", 0, root_id="c1"))


def _law(citation_id: str, start: int, resolves_to: str | None = None) -> CitationRecord:
    return read_citation(
        citation_id=citation_id,
        root_id=citation_id,
        resolves_to=resolves_to,
        fields=placed(
            FullLawCitation(reporter="U.S.C."),
            span=Span(start, start + 14),
            locator_span=Span(start, start + 14),
            matched_text="28 U.S.C. § 636",
        ),
    )


def test_a_statute_is_out_of_scope_rather_than_unattributed() -> None:
    """The two look alike in a count and mean opposite things.

    A statute has no case authority to belong to, so declining it is correct
    behaviour. A case citation that could not be traced to its full form is a
    failure worth looking at. Reporting them together turns a 1-in-894 failure
    rate on this corpus into an apparent 30%, and buries the one case that
    actually needs reading.
    """
    document = _document(_full("c1", "544", None, 0), _law("s1", 200))

    tree = build_citation_tree(document)

    assert [c.citation_id for c in tree.out_of_scope] == ["s1"]
    assert tree.unattributed == ()


def test_an_id_standing_in_for_a_statute_is_out_of_scope_too() -> None:
    """`Id.` carries no reporter, so what it refers to is whatever it resolved to."""
    document = _document(_law("s1", 0), _id("c2", "637", "s1", 200))

    tree = build_citation_tree(document)

    assert {c.citation_id for c in tree.out_of_scope} == {"s1", "c2"}
    assert tree.unattributed == ()


def _docket(citation_id: str, start: int, resolves_to: str | None = None) -> CitationRecord:
    return read_citation(
        citation_id=citation_id,
        root_id=citation_id,
        resolves_to=resolves_to,
        fields=placed(
            DocketCitation(defendant="Chen Zhi", docket_number="1:25-cr-00312-RPK", court="nyed"),
            span=Span(start, start + 21),
            locator_span=Span(start, start + 21),
            matched_text="No. 1:25-cr-00312-RPK",
        ),
    )


def test_a_docket_can_stand_at_the_head_of_a_chain() -> None:
    """A case cited by docket is an authority, not a citation of some other kind.

    Some cases are cited by docket and by nothing else -- too recent or too
    minor for a reporter -- and that is the population where a fabricated
    citation is hardest to catch. If the tree cannot root on one, every return
    visit to it is stranded and every claim those visits make goes unchecked.
    """
    document = _document(_docket("d1", 0), _id("c2", "34", "d1", 200))

    tree = build_citation_tree(document)

    (authority,) = tree.roots
    assert authority.root_id == "d1"
    assert authority.pin_cites == ("at 34",)
    assert tree.unattributed == ()


def test_a_docket_is_never_out_of_scope() -> None:
    """What a docket number names is a case, so there is nothing to send out of scope.

    `out_of_scope` is for positive evidence that a citation names something
    other than a case -- a statute, a journal article. A docket is the opposite
    of that evidence.
    """
    tree = build_citation_tree(_document(_docket("d1", 0)))

    assert tree.out_of_scope == ()
    assert [a.root_id for a in tree.roots] == ["d1"]
