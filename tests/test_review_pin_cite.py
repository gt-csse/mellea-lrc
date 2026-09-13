"""What the pin-cite reviewer will and will not take from a reader.

The reader is asked one thing -- whether the characters after a citation are a
page of it -- and everything that follows is computed from the document. So the
tests are about what happens to an answer after it arrives: it is quoted or it
is refused, it is near its own citation or it is refused, and the pages come
from the characters rather than from the reader.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from mellea_lrc.core.pin_cites import PinCiteKind
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.adjudication import pin_cite_sites
from mellea_lrc.extraction.adjudication.review.pin_cite import (
    REACH,
    Reading,
    _Answer,
    _find,
    _validate_quote,
)

TEXT = (
    "Haines v. Kerner, 404 U.S. 519, 520 (1972). In Fuller v. Quire, 916 F.2d 358, 360 "
    "(6th Cir. 1990), the court implied excusable neglect could apply."
)


class _Output:
    def __init__(self, value: str) -> None:
        self.value = value


class _Context:
    """The one method the validators use."""

    def __init__(self, answer: _Answer) -> None:
        self._answer = answer

    def last_output(self) -> _Output:
        return _Output(self._answer.model_dump_json())


def _answer(**fields: object) -> _Answer:
    return _Answer.model_validate({"reason": "because", **fields})


def _site():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(TEXT, relaxation=Relaxation.FULL)
    sites = list(pin_cite_sites(document))
    assert len(sites) == 1
    return document, sites[0]


def test_the_site_names_the_citation_it_corrects() -> None:
    document, site = _site()
    about = {record.citation_id for record in document.citations}

    assert site.about in about


@pytest.mark.parametrize(
    ("quote", "valid"),
    [
        ("360", True),
        ("3 60", True),  # the spacing may differ; the digits may not
        ("361", False),  # not in the window
        ("", False),  # `states_a_page` has to quote something
    ],
)
def test_a_page_claim_has_to_be_in_the_window(quote: str, valid: bool) -> None:
    _, site = _site()
    window = TEXT[site.window.start : site.window.end]
    answer = _answer(reading=Reading.CLAIMS_A_PAGE, pin_cite=quote)

    assert _validate_quote(_Context(answer), window).as_bool() is valid


def test_no_page_claim_quotes_nothing() -> None:
    _, site = _site()
    window = TEXT[site.window.start : site.window.end]

    assert _validate_quote(_Context(_answer(reading=Reading.NO_PAGE_CLAIM, pin_cite="")), window).as_bool()
    assert not _validate_quote(
        _Context(_answer(reading=Reading.NO_PAGE_CLAIM, pin_cite="360")), window
    ).as_bool()


def test_a_quote_is_found_through_the_converter_s_spacing() -> None:
    """A reader asked to copy `1053 -54` returns `1053-54` often enough that
    the digits have to be the window's and the spaces need not be."""
    found = _find("at 1053 -54 (D. Kan.)", "1053-54")

    assert found is not None
    assert found.group() == "1053 -54"


def test_written_but_no_page_keeps_the_characters_and_claims_nothing() -> None:
    """`192 F.3d 742, 74950` is a page claim the filing made and no page
    numbering reaches. Recording nothing would say it claims no page; recording
    74,950 would state a page it does not claim."""
    from mellea_lrc.core.pin_cites import PinCite

    claim = PinCite(span=Span(start=0, end=5), text="74950", pages=())

    assert claim.text == "74950"
    assert claim.pages == ()


def test_a_quote_far_from_its_own_citation_is_refused() -> None:
    """`REACH` is what keeps an answer against the citation it is about rather
    than against a number elsewhere in the window that reads the same."""
    assert REACH > 0
