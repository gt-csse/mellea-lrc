"""The exact lookup name rule recognizes written parties, not loose similarity."""

import pytest

from mellea_lrc.model.citations.fields.case_name import CaseName
from mellea_lrc.validation._support.party_names import compare_case_names


@pytest.mark.parametrize(
    ("source", "retrieved"),
    [
        ("Bell Atl. Corp. v. Twombly", "Bell Atlantic Corporation v. Twombly"),
        ("N.Y. Elec. Co. v. United States", "New York Electric Company v. U.S."),
        ("U.S. v. W. Va. Power Co.", "United States v. West Virginia Power Company"),
        ("Smith & Sons v. Jones", "Smith and Sons v. Jones"),
    ],
)
def test_both_parties_match_with_database_abbreviations(source: str, retrieved: str) -> None:
    result = compare_case_names(CaseName.from_quote(source), retrieved)
    assert result.plaintiff_present is True
    assert result.defendant_present is True
    assert result.qualifies is True


def test_one_party_missing_does_not_qualify() -> None:
    result = compare_case_names(
        CaseName.from_quote("Bell Atl. Corp. v. Twombly"), "Bell Atlantic Corporation v. Jones"
    )
    assert result.plaintiff_present is True
    assert result.defendant_present is False
    assert result.qualifies is False


def test_party_must_be_whole_tokens() -> None:
    result = compare_case_names(CaseName.from_quote("Bell v. Twombly"), "Bellamy v. Twombly")
    assert result.plaintiff_present is False
    assert result.qualifies is False


def test_two_parties_cannot_reuse_one_occurrence() -> None:
    result = compare_case_names(CaseName.from_quote("Smith v. Smith"), "In re Smith")
    assert result.plaintiff_present is True
    assert result.defendant_present is True
    assert result.qualifies is False


def test_shared_ambiguous_abbreviation_does_not_equate_distinct_full_words() -> None:
    result = compare_case_names(
        CaseName.from_quote("Electric Corp. v. Smith"), "Electronic Corporation v. Smith"
    )
    assert result.plaintiff_present is False
    assert result.defendant_present is True
    assert result.qualifies is False


@pytest.mark.parametrize(
    ("source", "retrieved", "qualifies"),
    [
        ("In re Bell Atl. Corp.", "In re Bell Atlantic Corporation", True),
        ("Ex parte N.Y. Elec. Co.", "Ex parte New York Electric Company", True),
        ("In re Bell Atl. Corp.", "Bell Atlantic Corporation v. Twombly", False),
    ],
)
def test_subject_case_requires_subject_and_same_form(source: str, retrieved: str, qualifies: bool) -> None:
    result = compare_case_names(CaseName.from_quote(source), retrieved)
    assert result.subject_present is qualifies
    assert result.qualifies is qualifies
