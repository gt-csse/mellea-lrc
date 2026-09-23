"""Tests for the deterministic parts of case-name adjudication.

The model call is not exercised here. What is tested is the generator that
proposes the sites, the lists a reader chooses from, and the grounding that
turns a quote back into a document offset -- everything that decides whether an
answer can enter the record.
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.model.extraction_metadata import Relaxation
from mellea_lrc.extraction.eyecite_extractor import extract_from_plain_text
from mellea_lrc.extraction.adjudication.candidates.case_name_sites import case_name_sites
from mellea_lrc.extraction.adjudication.review.case_name import (
    _ground,
    _same_case,
    neighbours,
    parties_read_as,
    roots,
    trim_sentence_period,
)
from mellea_lrc.extraction.adjudication.types import CandidateKind


def _extract(text: str):
    # Eyecite writes overlap diagnostics to stdout on some inputs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(text, relaxation=Relaxation.FULL)


class TestSites:
    """What the generator proposes, over the residue of a full mask."""

    def test_a_bare_name_with_no_citation_is_a_site(self) -> None:
        text = (
            "Doe v. Skyline Automobiles Inc. , 375 F. Supp. 3d 401 (S.D.N.Y. 2019). "
            "Courts have granted anonymity in similar circumstances (see Doe v. Rose)."
        )
        document = _extract(text)
        sites = list(case_name_sites(document))

        assert [text[s.span.start : s.span.end] for s in sites] == ["Doe v. Rose"]
        assert sites[0].kind is CandidateKind.CASE_NAME

    def test_a_name_inside_a_citation_that_was_read_is_not_a_site(self) -> None:
        """Masking is by full span, so a name eyecite reached is already gone."""
        document = _extract("Ashcroft v. Iqbal , 556 U.S. 662, 678 (2009).")

        assert list(case_name_sites(document)) == []

    def test_the_window_reaches_past_the_name(self) -> None:
        text = "x" * 500 + " In re Flint Water Cases or Myers v. City of Centerville " + "y" * 500
        document = _extract(text)
        site = next(iter(case_name_sites(document)))

        assert site.window.start < site.span.start
        assert site.window.end > site.span.end


class TestTheListsAReaderChoosesFrom:
    """A reader never returns an offset: it picks a citation by number."""

    def test_neighbours_are_the_citations_inside_the_window(self) -> None:
        text = (
            "Ashcroft v. Iqbal , 556 U.S. 662, 678 (2009). In Loos v. Lowe's , for example, "
            "the plaintiff alleged isolated incidents. 796 F. Supp. 2d 1013, 1023 (D. Ariz. 2011)."
        )
        document = _extract(text)
        site = next(iter(case_name_sites(document)))
        nearby = neighbours(document, site.window)

        assert [text[c.locator_span.start : c.locator_span.end] for c in nearby] == [
            "556 U.S. 662",
            "796 F. Supp. 2d 1013",
        ]

    def test_roots_are_offered_from_the_whole_document(self) -> None:
        """Rule 10.9 puts no distance between a short form and its full citation."""
        text = (
            "Doe v. Rose , 2016 WL 9137645, at 3 (C.D. Cal. 2016). "
            + "filler " * 200
            + "Courts have granted anonymity (see Doe v. Rose)."
        )
        document = _extract(text)
        site = next(iter(case_name_sites(document)))

        assert site.window.start > 0
        assert any(c.locator_span.start < site.window.start for c in roots(document))


class TestSameCase:
    """The check that catches an index off by one."""

    def test_a_shortened_name_still_shares_a_party(self) -> None:
        assert _same_case("Doe v. Skyline", "Doe Skyline Automobiles Inc.")
        assert _same_case("Loos v. Lowe's", "Loos Lowe's")

    def test_two_different_cases_share_nothing_that_identifies_them(self) -> None:
        assert not _same_case("Doe v. Skyline", "Ashcroft Iqbal")

    def test_a_name_made_only_of_common_words_is_not_refused(self) -> None:
        """`United States v. United States` carries no identity to compare."""
        assert _same_case("United States", "Ashcroft Iqbal")


class TestGrounding:
    """A quote is resolved against collapsed text, then mapped back."""

    def test_maps_a_name_through_collapsed_whitespace(self) -> None:
        window = "In  Loos  v.  Lowe's , for example, the plaintiff alleged"
        collapsed = " ".join(window.split())
        grounded = _ground(window, collapsed, 2000, "Loos v. Lowe's")

        assert grounded is not None
        span, text, _ = grounded
        assert window[span.start - 2000 : span.end - 2000] == text
        assert text.replace(" ", "") == "Loosv.Lowe's"

    def test_a_quote_that_is_not_there_does_not_resolve(self) -> None:
        window = "In Loos v. Lowe's , for example"

        assert _ground(window, window, 0, "Doe v. Amazon.com") is None


class TestTheSentencePeriod:
    """A case name quoted at the end of a sentence takes the period with it."""

    def test_a_whole_word_before_the_period_ends_the_sentence(self) -> None:
        assert trim_sentence_period("Murray v. Nationwide.") == "Murray v. Nationwide"
        assert trim_sentence_period("Loos v. Lowe's.") == "Loos v. Lowe's"

    def test_an_abbreviation_keeps_its_own_period(self) -> None:
        for name in (
            "Romandette v. Weetabix Co.",
            "In re Giftcraft Ltd.",
            "Jimerson v. Tetlin Native Corp.",
            "Bell Atl. Bus. Sys. Servs., Inc.",
            "Andrade Gutierrez Engenharia S.A.",
        ):
            assert trim_sentence_period(name) == name

    def test_a_name_with_no_trailing_period_is_untouched(self) -> None:
        assert trim_sentence_period("Doe v. Skyline") == "Doe v. Skyline"


class TestTheParties:
    """A patched name carries its parties, checked against the quote."""

    def test_a_two_party_name_splits_at_the_versus(self) -> None:
        assert parties_read_as("Bell Atl. Corp. v. Twombly", "Bell Atl. Corp.", "Twombly")

    def test_a_case_with_no_adverse_party_has_only_a_defendant(self) -> None:
        """eyecite's own convention, and the opening words belong to neither."""
        assert parties_read_as("In re Giftcraft Ltd.", None, "Giftcraft Ltd.")
        assert parties_read_as("Ex parte Young", None, "Young")
        assert parties_read_as("Hassan", None, "Hassan")

    def test_the_opening_words_are_not_a_party(self) -> None:
        assert not parties_read_as("In re Giftcraft Ltd.", "In re", "Giftcraft Ltd.")

    def test_damage_in_the_quote_is_repaired_in_the_parties(self) -> None:
        """The quote keeps the document's characters; the parties do not."""
        assert parties_read_as(
            "World Wide Ass ' n of Specialty Programs v. Pure, Inc.",
            "World Wide Ass'n of Specialty Programs",
            "Pure, Inc.",
        )
        assert parties_read_as("Romandette v. Weetabix Co .", "Romandette", "Weetabix Co.")

    def test_a_party_that_is_not_in_the_name_is_refused(self) -> None:
        assert not parties_read_as("Doe v. Skyline", "Doe", "Amazon.com")

    def test_a_name_with_no_defendant_is_refused(self) -> None:
        assert not parties_read_as("Doe v. Skyline", "Doe", None)
