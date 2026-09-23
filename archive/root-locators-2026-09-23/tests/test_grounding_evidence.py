"""Grounding policies only return canonical source evidence."""

from __future__ import annotations

import pytest

from mellea_lrc.model.fuzziness import FuzzinessOption, FuzzinessType
from mellea_lrc.llm.grounding import (
    EvidenceCandidate,
    GroundingEvidence,
    _without_margin_line_numbers,
    fuzzy_match,
)


def _evidence() -> GroundingEvidence[str]:
    return GroundingEvidence(
        (
            EvidenceCandidate(text="No. 21-381", value="source-one"),
            EvidenceCandidate(text="No. 23-381", value="source-two"),
        )
    )


@pytest.mark.parametrize(
    "fallback",
    [FuzzinessType.WHITESPACE_RELAXATION, FuzzinessType.EDIT_DISTANCE],
)
def test_perfect_match_cannot_silently_enable_a_fallback(fallback: FuzzinessType) -> None:
    with pytest.raises(ValueError, match="PERFECT_MATCH cannot coexist"):
        FuzzinessOption(types=frozenset({FuzzinessType.PERFECT_MATCH, fallback}))


def test_whitespace_relaxation_returns_the_canonical_source_candidate() -> None:
    match = _evidence().resolve("No.   21-381", FuzzinessOption.whitespace_relaxation())

    assert match is not None
    assert match.candidate.value == "source-one"
    assert match.candidate.text == "No. 21-381"
    assert match.match_type is FuzzinessType.WHITESPACE_RELAXATION


def test_edit_distance_composes_after_whitespace_relaxation() -> None:
    evidence = GroundingEvidence((EvidenceCandidate(text="No. 21-381", value="source"),))
    fuzziness = FuzzinessOption.edit_distance(similarity_percent=99)

    whitespace = evidence.resolve("No.   21-381", fuzziness)
    one_edit = evidence.resolve("No. 21-38l", fuzziness)

    assert whitespace is not None
    assert whitespace.match_type is FuzzinessType.WHITESPACE_RELAXATION
    assert one_edit is not None
    assert one_edit.match_type is FuzzinessType.EDIT_DISTANCE
    assert one_edit.edits == 1


def test_similarity_below_100_has_a_one_edit_floor_for_short_strings() -> None:
    matches = fuzzy_match(
        "ABD",
        (EvidenceCandidate(text="ABC", value="source"),),
        FuzzinessOption.edit_distance(similarity_percent=99, whitespace_relaxation=False),
    )

    assert len(matches) == 1
    assert matches[0].edits == 1


def test_absolute_edit_limit_of_zero_stays_exact() -> None:
    evidence = GroundingEvidence((EvidenceCandidate(text="ABC", value="source"),))

    assert evidence.resolve("ABD", FuzzinessOption.edit_distance(maximum_edits=0)) is None


def test_fragment_grounding_returns_actual_source_span() -> None:
    source = "Before Smith v. Jones, 1:24-cv-08760. After"
    evidence = GroundingEvidence((EvidenceCandidate(text=source, value="later-filing"),))

    grounded = evidence.find_fragment(
        "Smith v. Jones, 1:24-cv-0876O",
        FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True),
    )

    assert grounded is not None
    assert grounded.candidate.value == "later-filing"
    assert source[grounded.start : grounded.end] == grounded.text
    assert grounded.text == "Smith v. Jones, 1:24-cv-08760"
    assert grounded.match_type is FuzzinessType.EDIT_DISTANCE


def test_fragment_grounding_can_ignore_sequential_pdf_margin_numbers() -> None:
    quote = "Smith v. Jones, No. 13-cv-04115-WHO, 2016 WL 1019669 (N.D. Cal. 2016)."
    source = (
        "                 3   Some preceding text.\n\n"
        "                 4   More preceding text.\n\n"
        "                 5   See Smith v. Jones, No. 13-cv-\n\n"
        "                 6   04115-WHO, 2016 WL 1019669 (N.D. Cal. 2016).\n\n"
        "                 7         Subsequent text."
    )
    evidence = GroundingEvidence((EvidenceCandidate(text=source, value="later-filing"),))
    policy = FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True)

    assert evidence.find_fragment(quote, policy) is None
    grounded = evidence.find_fragment(quote, policy, line_number_aware=True)

    assert grounded is not None
    assert grounded.similarity_percent >= 90
    assert source[grounded.start : grounded.end] == grounded.text
    assert "\n\n                 6   04115-WHO, 2016 WL 1019669" in grounded.text
    assert "6   04115-WHO" not in grounded.normalized_text
    assert "04115-WHO, 2016 WL 1019669" in grounded.normalized_text


def test_margin_view_preserves_isolated_and_citation_numbers() -> None:
    source = "                 6   No. 13-cv-04115-WHO, 2016 WL 1019669"

    view, offsets = _without_margin_line_numbers(source)

    assert view == source
    assert offsets == ()


def test_margin_column_stays_aligned_when_line_number_gains_a_digit() -> None:
    quote = "Smith v. Jones, No. 13-cv-04115-WHO, 2016 WL 1019669 (N.D. Cal. 2016)."
    contents = (
        "Preceding sentence.",
        "Another sentence.",
        "See Smith v. Jones, No. 13-cv-04115-",
        "WHO, 2016 WL 1019669 (N.D. Cal. 2016).",
        "Subsequent sentence.",
        "More text.",
        "Final text.",
    )
    source = "\n\n".join(
        f"{' ' * (36 - len(str(number)))}{number}   {content}"
        for number, content in zip(range(6, 13), contents, strict=True)
    )
    policy = FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True)
    evidence = GroundingEvidence((EvidenceCandidate(text=source, value="later-filing"),))

    grounded = evidence.find_fragment(quote, policy, line_number_aware=True)

    assert grounded is not None
    assert grounded.similarity_percent >= 90
    assert source[grounded.start : grounded.end] == grounded.text
    assert "04115-\n\n" in grounded.text
    assert "9   WHO, 2016 WL 1019669" in grounded.text
