"""What the pin-cite generator proposes, and what it leaves alone.

It proposes; it never decides. So the tests are about coverage on one side --
every shape of damaged or truncated page claim reaches a reader -- and about
silence on the other, because a generator that fires on every citation has told
the reader nothing.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.adjudication import pin_cite_sites
from mellea_lrc.extraction.adjudication.types import CandidateKind


def _sites(text: str) -> list[str]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
    found = list(pin_cite_sites(document))
    assert all(site.kind is CandidateKind.PIN_CITE for site in found)
    return [site.note for site in found]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The converter drops the hyphen and `749-50` arrives as one number.
        ("Milwaukee Police Ass'n v. Jones , 192 F.3d 742, 74950 (7th Cir. 1999).", "damage"),
        # A margin line number landed between the page and the pin cite, and
        # nothing after it parses as one.
        ("In re Motors Liquidation Co., 957 F.3d 357, 23 - 361-62 (2d Cir. 2020).", "states one"),
        # A colon is not a terminator eyecite's pattern accepts.
        ("Tavoulareas v. Piro , 763 F.2d 1472, 1478: ' no single piece", "states one"),
        # A year parenthetical standing before a citation costs it its pin cite
        # and its party names, which is eyecite's own defect and not this
        # project's -- so the generator has to reach it from the text.
        (
            "Haines v. Kerner, 404 U.S. 519, 520 (1972). Huri v. Office of the Chief Judge , "
            "804 F.3d 826, 833-34 (7th Cir. 2015); x",
            "states one",
        ),
    ],
)
def test_a_page_the_rules_are_unsure_of_reaches_a_reader(text: str, expected: str) -> None:
    assert any(expected in note for note in _sites(text))


@pytest.mark.parametrize(
    "text",
    [
        "See Ashcroft v. Iqbal , 556 U.S. 662, 678 (2009). The court",
        "Bell Atl. Corp. v. Twombly , 550 U.S. 544, 570 (2007); x",
        # One case, two reporters, one position: the number after the page is a
        # volume, and a citation was read starting there.
        "McDonnell Douglas Corp. v. Green, 411 U.S. 792, 93 S.Ct. 1817 (1973). x",
        # A database citation writes a document number where a first page
        # belongs, so its star page is not a page before the first one.
        "Turner v. Murphy Oil USA, Inc., No. 05-4206, 2006 WL 1984362, at *1 (E.D. La. 2006). x",
        # `§ 1681e(b)` is a subsection, not a page of a case.
        "the reasonableness of procedures under § 1681e(b) is normally a question",
    ],
)
def test_an_ordinary_citation_proposes_nothing(text: str) -> None:
    assert _sites(text) == []


def test_a_citation_nobody_read_is_not_this_generator_s_business() -> None:
    """`Planned Parenthood Minn., N.D., S.D. at 732` is a case name and a page
    and no citation at all, and finding it means searching the document for a
    name. Until validation has resolved the roots, the names in the record are
    whatever eyecite's parser made of them, so that search is a search for a
    guess -- it comes back after the roots are resolved."""
    text = (
        "Planned Parenthood Minn., N.D., S.D. v. Rounds, 530 F.3d 724, 732 (8th Cir. 2008). "
        "Plaintiffs must show a likelihood of success through a heavy and compelling weight "
        "of evidence. Planned Parenthood Minn., N.D., S.D. at 732. Plaintiffs fail to meet it."
    )

    assert _sites(text) == []
