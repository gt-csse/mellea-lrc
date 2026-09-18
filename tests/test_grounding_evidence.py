"""Grounding policies only return canonical source evidence."""

from __future__ import annotations

import pytest

from mellea_lrc.llm.grounding import (
    EvidenceCandidate,
    FuzzinessOption,
    FuzzinessType,
    GroundingEvidence,
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
