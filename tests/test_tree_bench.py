"""Tests for the pin cite the tree bench records as ground truth.

This reader decides labels, so its mistakes do not show up as a worse score --
they show up as a bench that is wrong and agrees with itself. The two failures
worth guarding are inventing a pin cite that is not there, and dropping one the
document states.
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.core.citations import CitationKind
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from scripts.build_tree_bench import pin_cite_limits, pin_cite_written


def _pin(text: str, kind: CitationKind = CitationKind.FULL_CASE) -> str | None:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
    citation = next(item for item in document.citations if item.citation.kind is kind)
    limits = pin_cite_limits(document.citations, len(text))
    written = pin_cite_written(text, citation, limits[citation.citation_id])
    return None if written is None else written[0]


def test_a_stated_page_is_recorded() -> None:
    assert _pin("Bell Atl. Corp. v. Twombly, 550 U.S. 544, 570 (2007).") == "570"


def test_a_citation_with_no_pin_cite_records_none() -> None:
    assert _pin("Doe v. Roe, 403 F. Supp. 1199 (E.D. Va. 1975).") is None


def test_the_pin_cite_is_kept_verbatim() -> None:
    """Damage is part of the label. Normalisation is a separate, testable step."""
    assert _pin("Doe v. Roe, 899 F.3d 988, 998 -1003 (9th Cir. 2018).") == "998 -1003"


def test_a_parallel_citation_is_not_read_as_a_page() -> None:
    """`88` is the volume of the second reporter, not page 88 of the first."""
    assert _pin("Time, Inc. v. Hill, 390 U.S. 727, 88 S.Ct. 1323 (1968).") is None


def test_a_pin_cite_before_a_parallel_citation_survives() -> None:
    """Only the volume is dropped; the page the filing does state is kept."""
    assert _pin("Time, Inc. v. Hill, 390 U.S. 727, 731, 88 S.Ct. 1323 (1968).") == "731"


def test_a_short_form_states_its_page_inside_its_own_span() -> None:
    assert _pin("See Iqbal , 556 U.S. at 678.", CitationKind.SHORT_CASE) == "at 678"


def test_a_bare_id_records_no_pin_cite() -> None:
    """Its page is inherited, and inheritance is computed rather than annotated.

    Recording the antecedent's page here would bake a tree decision into the
    label and leave nothing to measure the tree against.
    """
    assert _pin("Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009). Id.", CitationKind.ID) is None


def test_an_id_that_states_a_page_records_it() -> None:
    text = "Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009). Id. at 686."

    assert _pin(text, CitationKind.ID) == "at 686"


def test_a_section_is_not_a_page() -> None:
    """`Id. § 1231` returns to a statute; reading it as a page invents a claim."""
    text = "Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009). Id. § 1231(g)(1)."

    assert _pin(text, CitationKind.ID) is None
