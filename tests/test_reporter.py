"""Tests for the reporter a citation names, as written and as canonical.

A filing writes one reporter several ways and extraction adds more, so two
spellings of one reporter used to compare as two reporters. What the document
wrote still has to survive: this project records what was written and decides
separately what it means.
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.core.citations import FullCaseCitation, FullLawCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text


def _first(text: str, kind: type = FullCaseCitation):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
    return next(item.citation for item in document.citations if isinstance(item.citation, kind))


def test_the_spelling_the_document_used_is_kept() -> None:
    citation = _first("Doe v. Roe, 695 F.Supp.2d 1149 (D. Colo. 2010).")

    assert citation.reporter.as_written == "F.Supp.2d"


def test_two_spellings_of_one_reporter_share_a_canonical_name() -> None:
    """`F.Supp.2d` and `F. Supp. 2d` are one reporter and used to be two."""
    tight = _first("Doe v. Roe, 695 F.Supp.2d 1149 (D. Colo. 2010).")
    spaced = _first("Doe v. Roe, 695 F. Supp. 2d 1149 (D. Colo. 2010).")

    assert tight.reporter.as_written != spaced.reporter.as_written
    assert tight.reporter.canonical == spaced.reporter.canonical == "F. Supp. 2d"


def test_an_abbreviation_variant_is_reconciled_too() -> None:
    """`Fed. Appx.` is the filer's choice, not converter damage.

    No amount of whitespace repair would reconcile it with `F. App'x`; only the
    reporter database knows they are the same.
    """
    citation = _first("United States v. Rucker, 188 Fed. Appx. 772, 778 (10th Cir. 2006).")

    assert citation.reporter.as_written == "Fed. Appx."
    assert citation.reporter.canonical == "F. App'x"


def test_a_statute_says_so_in_its_cite_type() -> None:
    """A sourced answer to "is this a case", in place of guessing from the name."""
    citation = _first("See 28 U.S.C. § 1927 for the standard.", FullLawCitation)

    assert citation.reporter.cite_type == "leg_statute"


def test_a_supreme_court_reporter_says_so() -> None:
    citation = _first("Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).")

    assert citation.reporter.is_scotus
    assert citation.reporter.cite_type == "federal"


def test_the_reporter_reads_back_as_the_document_wrote_it() -> None:
    """Anything rendering a citation for a person should show what was written."""
    citation = _first("Doe v. Roe, 695 F.Supp.2d 1149 (D. Colo. 2010).")

    assert str(citation.reporter) == "F.Supp.2d"


def test_an_ambiguous_abbreviation_is_left_undecided() -> None:
    """`5 Cranch 137` is United States Reports and also District of Columbia.

    eyecite breaks that tie by year, which fails both ways on this one reporter:
    Marbury's 1803 is inside both ranges and yields no edition at all, while a
    year inside only one yields a confident answer naming the wrong court.
    """
    citation = _first("Marbury v. Madison, 5 Cranch 137 (1803).")

    assert citation.reporter.short_name is None
    assert len(citation.reporter.editions) == 2
    assert citation.reporter.canonical == "Cranch"


def test_the_tie_is_not_broken_by_the_year() -> None:
    """A year inside only one range must not decide it either."""
    inside_both = _first("Marbury v. Madison, 5 Cranch 137 (1803).")
    inside_one = _first("Doe v. Roe, 5 Cranch 137 (1830).")

    assert inside_both.reporter.editions == inside_one.reporter.editions
    assert inside_one.reporter.short_name is None


def test_what_the_candidates_agree_on_is_still_recorded() -> None:
    """Undecided is not the same as unknown: agreement survives, difference does not."""
    citation = _first("Marbury v. Madison, 5 Cranch 137 (1803).")

    # The two Cranch reporters are different courts, so neither is asserted.
    assert citation.reporter.cite_type is None
    assert citation.reporter.is_scotus is False


def test_an_unambiguous_reporter_keeps_everything() -> None:
    citation = _first("Doe v. Roe, 695 F.Supp.2d 1149 (D. Colo. 2010).")

    assert citation.reporter.short_name == "F. Supp. 2d"
    assert citation.reporter.cite_type == "federal"
    assert citation.reporter.editions == ("Federal Supplement",)
