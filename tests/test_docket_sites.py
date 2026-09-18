"""Candidate generation for docket identifiers outside federal CM/ECF syntax."""

from __future__ import annotations

from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import grow_roots, stable
from mellea_lrc.extraction.adjudication.candidates.docket_sites import suspected_dockets
from mellea_lrc.preprocessing import preprocess


def test_hunting_proposes_a_labelled_state_docket_the_first_pass_leaves_open() -> None:
    text = "The related proceeding is Case  No.  035547/2021."
    document = grow_roots(preprocess(text), rules=stable())

    (site,) = suspected_dockets(document)

    assert site.locator_span == Span(start=26, end=48)
    assert site.locator_text == "Case  No.  035547/2021"
    assert site.docket_number == "035547/2021"


def test_hunting_keeps_an_opaque_full_locator_without_deciding_its_fields() -> None:
    text = "See Civ. A. No. CIV 11-0107 JB/KBM (D.N.M. Mar. 28, 2013)."
    document = grow_roots(preprocess(text), rules=stable())

    (site,) = suspected_dockets(document)

    assert site.locator_text == "Civ. A. No. CIV 11-0107 JB/KBM"
    assert site.docket_number == "CIV 11-0107 JB/KBM"
    assert not hasattr(site, "courts")


def test_hunting_keeps_periods_inside_a_local_docket_locator() -> None:
    document = grow_roots(preprocess("Doe v. Townes, No. 19 Civ. 8034 (S.D.N.Y. 2020)."), rules=stable())

    (site,) = suspected_dockets(document)

    assert site.locator_text == "No. 19 Civ. 8034"
    assert site.docket_number == "19 Civ. 8034"


def test_hunting_relaxes_whitespace_between_a_label_and_its_period() -> None:
    text = "Koulkina v. City of New York, No . 06 Civ. 11357, 2009 WL 2103627."
    document = grow_roots(preprocess(text), rules=stable())

    (site,) = suspected_dockets(document)

    assert site.locator_text == "No . 06 Civ. 11357"
    assert site.docket_number == "06 Civ. 11357"


def test_hunting_does_not_repeat_a_docket_the_cmecf_reader_already_found() -> None:
    document = grow_roots(preprocess("Case No. 1:24-cv-00123"), rules=stable())

    assert suspected_dockets(document) == ()
