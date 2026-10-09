"""Post-root docket review changes citation attachments from a complete partition."""

from __future__ import annotations

import asyncio

import pytest

from mellea_lrc.api import Document, grow_roots
from mellea_lrc.extraction.docket_root_llm_reassignment import (
    SUBSTAGE,
    docket_root_llm_reassignment,
)
from mellea_lrc.extraction.docket_root_llm_reassignment.context import DocketRootReviewContext
from mellea_lrc.extraction.docket_root_llm_reassignment.reviewer import DocketRootReviewOutcome
from mellea_lrc.model.citations import latest
from mellea_lrc.model.citations.docket_root_llm_reassignment import DocketRootPartition


def _rooted(*numbers: str) -> Document:
    source = " ".join(f"No. {number}." for number in numbers)
    return asyncio.run(grow_roots(Document.from_source(source))).get_substage(
        "grow_roots.root_formation.rule"
    )


def _rooted_with_repeated_second_number() -> Document:
    source = (
        "Smith v. Jones, Case No. 1:24-cv-00123 (D. Ariz. 2024). "
        "Smith v. Jones, Case No. 1:24-cv-00124 (D. Ariz. 2024). "
        "Smith v. Jones, Case No. 1:24-cv-00124 (D. Ariz. 2024). "
        "State v. Lee, Case No. 1:24-cr-00125 (D. Ariz. 2024)."
    )
    result = asyncio.run(grow_roots(Document.from_source(source)))
    first, second, _, third = result.full_locators
    assert [latest(citation.root_id) for citation in result.full_locators] == [
        first.id,
        second.id,
        second.id,
        third.id,
    ]
    return result.get_substage("grow_roots.root_formation.rule")


def test_pairwise_boundary_forms_one_review_group_through_a_bridge() -> None:
    # The first and second numbers are exactly 40% similar under the project's
    # edit-distance policy, as are the second and third. First to third is 30%.
    before = _rooted("12-ab-3456", "17-bb-6915", "65-yb-4943")
    contexts: list[DocketRootReviewContext] = []

    async def reviewer(context: DocketRootReviewContext) -> DocketRootPartition:
        contexts.append(context)
        return DocketRootPartition(groups=((0, 1), (2,)), reason="The third is a different case.")

    after = asyncio.run(docket_root_llm_reassignment(before, reviewer=reviewer))

    assert len(contexts) == 1
    assert [candidate.citation_id for candidate in contexts[0].candidates] == [
        citation.id for citation in before.roots
    ]
    assert [candidate.docket_number for candidate in contexts[0].candidates] == [
        "12-ab-3456",
        "17-bb-6915",
        "65-yb-4943",
    ]
    first, second, third = after.full_locators
    assert [latest(citation.root_id) for citation in after.full_locators] == [
        first.id,
        first.id,
        third.id,
    ]
    assert second.id != first.id
    assert after.roots == (first, third)


def test_merge_updates_every_occurrence_attached_to_the_losing_root() -> None:
    before = _rooted_with_repeated_second_number()
    first, second, repeated_second, third = before.full_locators
    seen: list[DocketRootReviewContext] = []

    async def reviewer(context: DocketRootReviewContext) -> DocketRootPartition:
        seen.append(context)
        # Group order is the model's classification, not a choice of root ID.
        return DocketRootPartition(groups=((1, 0), (2,)), reason="The first two identify one case.")

    after = asyncio.run(docket_root_llm_reassignment(before, reviewer=reviewer))

    assert len(seen) == 1
    assert [candidate.citation_id for candidate in seen[0].candidates] == [
        first.id,
        second.id,
        third.id,
    ]
    assert [
        candidate.case_name and candidate.case_name.endswith(name)
        for candidate, name in zip(
            seen[0].candidates,
            ("Smith v. Jones", "Smith v. Jones", "State v. Lee"),
            strict=True,
        )
    ] == [True, True, True]
    assert all(candidate.court and candidate.date for candidate in seen[0].candidates)
    assert [latest(citation.root_id) for citation in after.full_locators] == [
        first.id,
        first.id,
        first.id,
        third.id,
    ]
    for original, updated in zip(before.full_locators, after.full_locators, strict=True):
        assert updated.id == original.id
        assert updated.nodes[: len(original.nodes)] == original.nodes
        assert updated.root_id[: len(original.root_id)] == original.root_id
    assert len(after.full_locators[1].root_id) == len(second.root_id) + 1
    assert len(after.full_locators[2].root_id) == len(repeated_second.root_id) + 1
    review = after.full_locators[0].docket_root_reviews[-1]
    assert review.candidate_ids == (first.id, second.id, third.id)
    assert review.decision is not None
    assert review.decision.groups == ((1, 0), (2,))
    assert after.get_substage("grow_roots.root_formation.rule") == before
    assert after.get_substage(SUBSTAGE) == after
    assert Document.model_validate_json(after.model_dump_json()) == after


@pytest.mark.parametrize("numbers", [("12-ab-3456",), ("12-ab-3456", "66-wa-1449")])
def test_singletons_and_below_threshold_pairs_skip_review_but_complete_stage(
    numbers: tuple[str, ...],
) -> None:
    before = _rooted(*numbers)

    async def reviewer(_context: DocketRootReviewContext) -> DocketRootPartition:
        pytest.fail("No candidate pair reaches the 40% docket-number threshold")

    after = asyncio.run(docket_root_llm_reassignment(before, reviewer=reviewer))

    assert after.citations == before.citations
    assert after.substage_runs == (*before.substage_runs, SUBSTAGE)
    assert after.get_substage("grow_roots.root_formation.rule") == before
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_separate_model_groups_keep_distinct_roots_and_save_review() -> None:
    before = _rooted("1:24-cv-00123", "1:24-cv-00124")

    async def reviewer(_context: DocketRootReviewContext) -> DocketRootPartition:
        return DocketRootPartition(groups=((0,), (1,)), reason="Different proceedings.")

    after = asyncio.run(docket_root_llm_reassignment(before, reviewer=reviewer))

    assert [latest(citation.root_id) for citation in after.full_locators] == [
        citation.id for citation in before.full_locators
    ]
    assert after.roots == after.full_locators
    assert after.full_locators[0].docket_root_reviews
    assert after.get_substage("grow_roots.root_formation.rule") == before
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_grow_roots_can_opt_in_to_the_post_root_review() -> None:
    async def reviewer(_context: DocketRootReviewContext) -> DocketRootPartition:
        return DocketRootPartition(groups=((0, 1),), reason="The two writings identify one case.")

    source = "No. 12-ab-3456. Later, No. 17-bb-6915."
    document = asyncio.run(
        grow_roots(
            Document.from_source(source),
            review_docket_roots=True,
            docket_root_reviewer=reviewer,
        )
    )

    assert document.substage_runs[-2:] == ("grow_roots.root_formation.rule", SUBSTAGE)
    assert len(document.roots) == 1


def test_failed_review_keeps_roots_and_records_the_failure() -> None:
    before = _rooted("1:24-cv-00123", "1:24-cv-00124")

    async def reviewer(_context: DocketRootReviewContext) -> DocketRootReviewOutcome:
        return DocketRootReviewOutcome(decision=None, failure_reason="Could not decide")

    after = asyncio.run(docket_root_llm_reassignment(before, reviewer=reviewer))

    assert [latest(citation.root_id) for citation in after.full_locators] == [
        citation.id for citation in before.full_locators
    ]
    review = after.full_locators[0].docket_root_reviews[-1]
    assert review.decision is None
    assert review.failure_reason == "Could not decide"
    assert after.get_substage("grow_roots.root_formation.rule") == before
    assert Document.model_validate_json(after.model_dump_json()) == after


@pytest.mark.parametrize("groups", [((0,),), ((0, 2),)])
def test_incomplete_or_unknown_partition_cannot_reassign_roots(
    groups: tuple[tuple[int, ...], ...],
) -> None:
    before = _rooted("1:24-cv-00123", "1:24-cv-00124")

    async def reviewer(_context: DocketRootReviewContext) -> DocketRootPartition:
        return DocketRootPartition(groups=groups, reason="Malformed partition")

    with pytest.raises(ValueError, match="each candidate index"):
        asyncio.run(docket_root_llm_reassignment(before, reviewer=reviewer))
    assert all(latest(citation.root_id) == citation.id for citation in before.full_locators)
