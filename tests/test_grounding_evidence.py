"""Grounding policies only return canonical source evidence."""

from __future__ import annotations

import pytest

from mellea_lrc.matching.fuzziness import FuzzinessOption, FuzzinessType
from mellea_lrc.matching.grounding import (
    EvidenceCandidate,
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


def test_resolve_by_condition_keeps_caller_owned_candidate_selection() -> None:
    chosen = _evidence().resolve_by_condition(lambda candidate: candidate.value == "source-two")

    assert chosen is not None
    assert chosen.text == "No. 23-381"


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


@pytest.mark.parametrize("types", [frozenset({"unknown"}), frozenset({"edit_distance"})])
def test_fuzziness_rejects_untyped_match_operations(types) -> None:
    with pytest.raises(ValueError, match="FuzzinessType"):
        FuzzinessOption(types=types)


@pytest.mark.parametrize("maximum_edits", [-1, True, 1.5, float("nan"), float("inf"), "1"])
def test_fuzziness_rejects_invalid_absolute_edit_limits(maximum_edits) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        FuzzinessOption.edit_distance(maximum_edits=maximum_edits)


@pytest.mark.parametrize("similarity", [0, 101, True, float("nan"), float("inf"), "99"])
def test_fuzziness_rejects_invalid_similarity_thresholds(similarity) -> None:
    with pytest.raises(ValueError, match="similarity_percent"):
        FuzzinessOption.edit_distance(similarity_percent=similarity)


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


def test_fragment_grounding_combines_whitespace_and_one_character_repair() -> None:
    source = "Before Case No.  1:24-cv-08760. After"
    evidence = GroundingEvidence((EvidenceCandidate(text=source, value="filing"),))

    grounded = evidence.find_fragment(
        "Case No. 1:24-cv-0876O",
        FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True),
    )

    assert grounded is not None
    assert grounded.text == "Case No.  1:24-cv-08760"
    assert source[grounded.start : grounded.end] == grounded.text
    assert grounded.match_type is FuzzinessType.EDIT_DISTANCE


@pytest.mark.parametrize("expanded_gaps", (6, 8))
def test_fragment_grounding_preserves_complete_whitespace_only_quotation(expanded_gaps: int) -> None:
    quote = (
        '"A source passage carries all of its stated conditions, qualifications, and limits. '
        "An accurate quotation must preserve the beginning and end of the selected wording, "
        "even when line breaks and repeated spaces differ from the proposal. Matching an entire "
        "passage without changing any non-whitespace character provides stronger evidence than "
        'accepting a shorter fragment that clips letters or punctuation." (Citation omitted.)'
    )
    canonical = quote.replace(" ", "   ", expanded_gaps).replace("An accurate", "An\n\naccurate")
    before = "Earlier paragraph.\n\n"
    source = before + canonical + "\n\nLater paragraph."
    evidence = GroundingEvidence((EvidenceCandidate(text=source, value="filing"),))

    grounded = evidence.find_fragment(
        quote,
        FuzzinessOption.edit_distance(similarity_percent=98, whitespace_relaxation=True),
    )

    assert grounded is not None
    assert grounded.text == canonical
    assert (grounded.start, grounded.end) == (len(before), len(before) + len(canonical))
    assert source[grounded.start : grounded.end] == canonical
    assert grounded.match_type is FuzzinessType.WHITESPACE_RELAXATION
    assert grounded.edits == 0
    assert grounded.similarity_percent == 100


@pytest.mark.parametrize(
    "policy",
    [
        FuzzinessOption.perfect_match(),
        FuzzinessOption.whitespace_relaxation(),
        FuzzinessOption.edit_distance(similarity_percent=98, whitespace_relaxation=True),
    ],
)
def test_literal_numbered_source_is_grounded_without_layout_rejection(policy) -> None:
    quote = "\n".join(f"            {number}  Source paragraph {number}." for number in range(1, 6))
    before = "Earlier text.\n"
    source = before + quote + "\nLater text."
    evidence = GroundingEvidence((EvidenceCandidate(text=source, value="source"),))

    grounded = evidence.find_fragment(quote, policy)

    assert grounded is not None
    assert grounded.text == quote
    assert (grounded.start, grounded.end) == (len(before), len(before) + len(quote))
    assert source[grounded.start : grounded.end] == quote
    assert grounded.match_type is FuzzinessType.PERFECT_MATCH
    assert grounded.edits == 0


def test_whitespace_relaxation_does_not_silently_remove_source_digits() -> None:
    source = "The selected\n            6  source passage."
    evidence = GroundingEvidence((EvidenceCandidate(text=source, value="source"),))

    assert (
        evidence.find_fragment("The selected source passage.", FuzzinessOption.whitespace_relaxation())
        is None
    )
