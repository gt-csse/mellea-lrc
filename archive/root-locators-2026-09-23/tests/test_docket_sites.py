"""Candidate generation for docket identifiers outside federal CM/ECF syntax."""

from __future__ import annotations

from mellea_lrc.model.spans import Span
from mellea_lrc.extraction.eyecite_extractor import grow_roots
from mellea_lrc.extraction.rules import stable
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


def test_hunting_keeps_an_internal_period_in_an_opaque_docket_label() -> None:
    text = "Webb v. Ocwen, No. CIV.A. 11-00732-KD-M, 2012 WL 5906729."
    document = grow_roots(preprocess(text), rules=stable())

    (site,) = suspected_dockets(document)

    assert site.locator_text == "No. CIV.A. 11-00732-KD-M"
    assert site.docket_number == "CIV.A. 11-00732-KD-M"


def test_hunting_recovers_the_whole_slashed_number_after_a_partial_cmecf_match() -> None:
    text = "Long v. Benson, No. 4:08-cv-26-RH/WCS, 2008 WL 4571904."
    document = grow_roots(preprocess(text), rules=stable())

    sites = suspected_dockets(document)

    assert not any(
        record.fields.docket_number == "4:08-cv-26-RH"
        for record in document.citations
        if hasattr(record.fields, "docket_number")
    )
    assert any(site.locator_text == "No. 4:08-cv-26-RH/WCS" for site in sites)


def test_hunting_proposes_an_unlabelled_identifier_beside_a_reporter() -> None:
    text = "See Smith v. Jones, 19-6658, 2021 WL 346418 (C.D. Cal. 2021)."
    document = grow_roots(preprocess(text), rules=stable())

    assert any(
        site.locator_text == "19-6658" and site.docket_number == "19-6658"
        for site in suspected_dockets(document)
    )


def test_hunting_does_not_propose_an_ordinary_reporter_pinpoint_as_a_docket() -> None:
    text = "See Smith v. Jones, 426 F. Supp. 3d 151, 153-54 (D. Ariz. 2021)."
    document = grow_roots(preprocess(text), rules=stable())

    assert not any(site.locator_text == "153-54" for site in suspected_dockets(document))


def test_hunting_proposes_an_unlabelled_compound_identifier_in_citation_context() -> None:
    text = "Smith v. Jones, 1:21-cv-04370, at *11 (N.D. Ill. Sept. 30, 2023)."
    document = grow_roots(preprocess(text), rules=stable())

    assert any(site.locator_text == "1:21-cv-04370" for site in suspected_dockets(document))


def test_hunting_proposes_a_compound_identifier_with_spacing_damage() -> None:
    text = "Student Doe v. Agency, 4:25-cv-  00175, (D. Ariz. 2025)."
    document = grow_roots(preprocess(text), rules=stable())

    assert any(site.locator_text == "4:25-cv-  00175" for site in suspected_dockets(document))


def test_hunting_does_not_scan_unrelated_compound_numbers() -> None:
    text = "The file was saved on 2025-09-21 and records 16:19-17:2."
    document = grow_roots(preprocess(text), rules=stable())

    assert suspected_dockets(document) == ()
