"""Tests for the deterministic parts of site adjudication.

The model call itself is not exercised here. What is tested is everything that
guards it: the shapes rejected before grounding, and the mapping that turns a
quote against collapsed text back into a document offset.
"""

from mellea_lrc.extraction.adjudication.review.locator import _ground, implausible_locator


class TestImplausibleLocator:
    """Shapes a locator cannot have, whatever the model claims."""

    def test_a_bare_number_is_a_pin_cite_not_a_locator(self) -> None:
        assert implausible_locator("442")
        assert implausible_locator("1086")

    def test_a_run_of_counting_numbers_is_pdf_margin_numbering(self) -> None:
        """The true citation is 214 F.3d 1058; the run is page furniture."""
        assert implausible_locator("214 F.3d 1 2 3 4 5 6 7 8 9 10")

    def test_ordinary_locators_survive(self) -> None:
        for text in ("550 U.S. 544", "284 S.W.3d 303", "16 F. 3d 1083", "550 US 544"):
            assert not implausible_locator(text)

    def test_damaged_locators_survive(self) -> None:
        """Damage is what this layer exists to recover; it must not be filtered."""
        for text in ("2016 WL1448829", "WL9137645", "455 US. 363", "58  N.Y .2d  916"):
            assert not implausible_locator(text)

    def test_a_locator_with_a_pin_cite_still_survives(self) -> None:
        """Three numeric tokens is a locator plus a pin cite, not line numbering."""
        assert not implausible_locator("937 S.W.2d 796, 800")


class TestGrounding:
    """A quote is resolved against collapsed text, then mapped back."""

    def test_maps_a_span_through_collapsed_whitespace(self) -> None:
        window = "See also White v. McBride ,  937  S.W.2d  796,  800  (Tenn.  1996)"
        collapsed = " ".join(window.split())
        grounded = _ground(window, collapsed, 1000, "937 S.W.2d 796")
        assert grounded is not None
        span, text, method = grounded
        assert window[span.start - 1000 : span.end - 1000] == text
        assert text.replace(" ", "") == "937S.W.2d796"
        assert method == "exact"

    def test_offsets_are_absolute_in_the_document(self) -> None:
        window = "Doe v. Colgate Univ. , 2016 WL1448829, at *2"
        grounded = _ground(window, window, 5000, "2016 WL1448829")
        assert grounded is not None
        span, _, _ = grounded
        assert span.start == 5000 + window.index("2016 WL1448829")

    def test_an_ungroundable_quote_resolves_to_nothing(self) -> None:
        """A hallucinated locator must fail rather than yield a plausible span."""
        window = "Doe v. Colgate Univ. , 2016 WL1448829, at *2"
        assert _ground(window, window, 0, "999 F.3d 12345") is None


def test_a_scanner_confusion_is_repairable_but_a_wrong_page_is_not() -> None:
    """The rules read damage between a citation's parts; this is damage inside one.

    `833 F.2d l83` is a citation only a reader recovers, and the reader may only
    swap a character for the digit it is shaped like -- never for a different
    number, and never by adding a word, which changes the length no folding
    hides.
    """
    from mellea_lrc.extraction.adjudication.review.locator import _repairable

    assert _repairable("833 F.2d l83") == _repairable("833" + "F.2d" + "183")
    assert _repairable("32O N.C. 1") == _repairable("320" + "N.C." + "1")
    assert _repairable("833 F.2d l83") != _repairable("833" + "F.2d" + "999")


def test_a_short_form_is_refused_however_it_is_grounded() -> None:
    """`at 1071` is a pin cite, and a locator quoting one resolves to the wrong case."""
    from mellea_lrc.extraction.adjudication.review.locator import (
        _Locator,
        _grounded_more_than_a_locator,
    )

    short = _Locator(text="214  F.3d  at 1071", volume="214", reporter="F.3d", page="1071")
    repaired = _Locator(text="833 F.2d l83", volume="833", reporter="F.2d", page="183")

    assert _grounded_more_than_a_locator(short.text, short)
    assert not _grounded_more_than_a_locator(repaired.text, repaired)
