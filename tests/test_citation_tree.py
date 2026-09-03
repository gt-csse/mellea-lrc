"""Tests for grouping a document's citations under the authority they cite."""

from __future__ import annotations

import pytest

from mellea_lrc.core.citations import (
    DocketCitation,
    FullCaseCitation,
    FullLawCitation,
    IdCitation,
    ShortCaseCitation,
)
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import ExtractedCitation
from mellea_lrc.extraction.structure.citation_tree import build_citation_tree
from mellea_lrc.extraction.types import ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing.plain_text import preprocess_plain_text_from_string


def _document(*citations: ExtractedCitation, text: str = "x" * 400) -> ExtractedDocument:
    source = preprocess_plain_text_from_string(text)
    return ExtractedDocument(
        source_metadata=source.source_metadata,
        preprocessing_metadata=source.preprocessing_metadata,
        text=source.text,
        citations=citations,
        extraction_metadata=ExtractionMetadata(),
    )


def _full(citation_id: str, page: str, pin: str | None, start: int) -> ExtractedCitation:
    return ExtractedCitation(
        citation_id=citation_id,
        full_span=Span(start, start + 12),
        locator_span=Span(start, start + 12),
        matched_text=f"550 U.S. {page}",
        citation=FullCaseCitation(volume="550", reporter="U.S.", page=page, pin_cite=pin),
    )


def _short(citation_id: str, pin: str, resolves_to: str, start: int) -> ExtractedCitation:
    return ExtractedCitation(
        citation_id=citation_id,
        full_span=Span(start, start + 12),
        locator_span=Span(start, start + 12),
        matched_text=f"550 U.S. at {pin}",
        citation=ShortCaseCitation(volume="550", reporter="U.S.", page=pin, pin_cite=f"at {pin}"),
        resolves_to=resolves_to,
    )


def _id(citation_id: str, pin: str, resolves_to: str, start: int) -> ExtractedCitation:
    return ExtractedCitation(
        citation_id=citation_id,
        full_span=Span(start, start + 8),
        locator_span=Span(start, start + 8),
        matched_text=f"Id. at {pin}",
        citation=IdCitation(pin_cite=f"at {pin}"),
        resolves_to=resolves_to,
    )


def test_every_reference_gathers_under_the_authority_it_cites() -> None:
    """A brief cites a case once in full and returns to it; all of that is one authority."""
    document = _document(
        _full("c1", "544", "555", 0),
        _short("c2", "563", "c1", 100),
        _id("c3", "570", "c2", 200),
    )

    tree = build_citation_tree(document)

    (authority,) = tree.authorities
    assert authority.authority_id == "c1"
    assert [o.citation_id for o in authority.occurrences] == ["c1", "c2", "c3"]
    assert tree.unattributed == ()


def test_resolution_is_followed_through_a_short_form() -> None:
    """`Id.` points at the short form before it, not at the full citation."""
    document = _document(
        _full("c1", "544", None, 0), _short("c2", "563", "c1", 100), _id("c3", "570", "c2", 200)
    )

    (authority,) = build_citation_tree(document).authorities
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
        _id("c3", "570", "c2", 200),
    )

    (authority,) = build_citation_tree(document).authorities

    assert authority.pin_cites == ("555", "at 563", "at 570")
    assert build_citation_tree(document).pinpoint_claim_count == 3


def test_one_page_cited_twice_is_one_claim() -> None:
    """Returning to the same page does not create a second thing to verify."""
    document = _document(_full("c1", "544", "555", 0), _short("c2", "555", "c1", 100))

    (authority,) = build_citation_tree(document).authorities

    assert authority.pin_cites == ("555", "at 555")


def test_an_unresolved_reference_is_reported_not_guessed() -> None:
    """Attaching a claim to the wrong authority checks it against the wrong page."""
    document = _document(_full("c1", "544", None, 0), _id("c9", "570", None, 200))

    tree = build_citation_tree(document)

    assert [c.citation_id for c in tree.unattributed] == ["c9"]
    assert tree.occurrence_count == 1


def test_a_resolution_cycle_does_not_hang_or_attribute() -> None:
    """A cycle is a pathology; it must not loop and must not produce an authority."""
    first = _short("c1", "563", "c2", 0)
    second = _short("c2", "570", "c1", 100)

    tree = build_citation_tree(_document(first, second))

    assert {c.citation_id for c in tree.unattributed} == {"c1", "c2"}
    assert tree.authorities == ()


def test_a_dangling_antecedent_cannot_reach_the_tree_at_all() -> None:
    """The document type refuses one, so the tree never has to decide about it.

    Worth pinning here rather than assuming: the tree still handles a missing
    antecedent defensively, but this is why that path is unreachable through an
    `ExtractedDocument` and why no citation can be attributed to an authority
    that was never extracted.
    """
    with pytest.raises(ValueError, match="invalid resolves_to"):
        _document(_id("c3", "570", "gone", 0))


def _law(citation_id: str, start: int, resolves_to: str | None = None) -> ExtractedCitation:
    return ExtractedCitation(
        citation_id=citation_id,
        full_span=Span(start, start + 14),
        locator_span=Span(start, start + 14),
        matched_text="28 U.S.C. § 636",
        citation=FullLawCitation(reporter="U.S.C."),
        resolves_to=resolves_to,
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


def test_a_short_form_with_no_antecedent_is_a_real_failure() -> None:
    """A short form carries a volume and reporter, so it is a case on its own evidence.

    This is the corpus's single unattributed citation: `Rosenblatt v. Baer, 383
    U.S. at 85`, quoted inside another case's parenthetical and never given in
    full. Nothing can be verified about it, and it must not be silently folded
    in with the statutes.
    """
    orphan = ExtractedCitation(
        citation_id="c9",
        full_span=Span(0, 14),
        locator_span=Span(0, 14),
        matched_text="383 U.S. at 85",
        citation=ShortCaseCitation(volume="383", reporter="U.S.", page="85", pin_cite="at 85"),
    )

    tree = build_citation_tree(_document(orphan))

    assert [c.citation_id for c in tree.unattributed] == ["c9"]
    assert tree.out_of_scope == ()


def _docket(citation_id: str, start: int, resolves_to: str | None = None) -> ExtractedCitation:
    return ExtractedCitation(
        citation_id=citation_id,
        full_span=Span(start, start + 21),
        locator_span=Span(start, start + 21),
        matched_text="No. 1:25-cr-00312-RPK",
        citation=DocketCitation(defendant="Chen Zhi", docket_number="1:25-cr-00312-RPK", court="nyed"),
        resolves_to=resolves_to,
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

    (authority,) = tree.authorities
    assert authority.authority_id == "d1"
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
    assert [a.authority_id for a in tree.authorities] == ["d1"]
