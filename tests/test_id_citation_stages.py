"""Id. creation reads pins before its independent, sequential attribution stage."""

from __future__ import annotations

import asyncio

import pytest

from mellea_lrc.api import (
    Document,
    attribute_id_citations,
    attribute_reference_citations,
    find_id_citations,
    find_reference_citations,
    grow_leaves,
    grow_roots,
)
from mellea_lrc.extraction.id_attribution import STAGE as ATTRIBUTION_STAGE
from mellea_lrc.extraction.id_citations import STAGE as CREATION_STAGE
from mellea_lrc.extraction.leaf_attribution_review.reviewer import LeafReviewContext, LeafReviewOutcome
from mellea_lrc.model.citations import (
    AttributionResult,
    IdCitation,
    LeafReviewDecision,
    ReferenceCitation,
    SupraCitation,
    latest,
)
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID

ROOT_TEXT = "Alpha v. Beta, 100 F.3d 1 (2000). "


def _roots(text: str) -> Document:
    return asyncio.run(grow_roots(Document.from_source(text)))


def _ids(document: Document) -> list[IdCitation]:
    return [citation for citation in document.short_citations if isinstance(citation, IdCitation)]


@pytest.mark.parametrize(
    "tail, keyword, pin_quote",
    [
        ("Id .", "id", None),
        ("Ibid  .", "ibid", None),
        ("Id\n. at\n 14.", "id", "14"),
        ("Id .at14.", "id", "14"),
        ("Id at 14.", "id", "14"),
        ("Ibid\n at\n ¶ 14.", "ibid", "¶ 14"),
    ],
)
def test_marker_and_pin_joins_share_whitespace_relaxation_and_original_offsets(
    tail: str, keyword: str, pin_quote: str | None
) -> None:
    source = ROOT_TEXT + tail
    roots = _roots(source)
    discovered = find_id_citations(roots)
    citation = _ids(discovered)[0]
    field = citation.id_reference[-1]

    assert field.get_normalized().keyword == keyword
    assert field.span.start == len(ROOT_TEXT)
    assert field.quote == source[field.span.start : field.span.end]
    assert (citation.pin_cite[-1].quote if citation.pin_cite else None) == pin_quote
    attributed = asyncio.run(attribute_id_citations(discovered, review=False))
    restored = Document.model_validate_json(attributed.model_dump_json())
    assert restored.get_stage(CREATION_STAGE) == discovered
    assert restored.get_stage("10_roots") == roots.get_stage("10_roots")


@pytest.mark.parametrize("tail", ["The ID is 14.", "Ibid refers to that term.", "identity at 14.", "Idat14."])
def test_marker_relaxation_does_not_turn_ordinary_words_into_id_citations(tail: str) -> None:
    assert _ids(find_id_citations(_roots(ROOT_TEXT + tail))) == []


@pytest.mark.parametrize("tail", ["Id.5", "Id . 5", "Ibid .\n5 A footnote label."])
def test_unlabelled_numbers_never_extend_the_id_site(tail: str) -> None:
    citation = _ids(find_id_citations(_roots(ROOT_TEXT + tail)))[0]
    assert citation.id_reference[-1].quote == tail[: tail.index(".") + 1]
    assert citation.pin_cite is None


def test_creation_reads_a_pin_on_its_native_node_and_attribution_preserves_the_checkpoint() -> None:
    text = ROOT_TEXT + "See id. at ¶ 12 - 14,\n ¶ 16."
    roots = _roots(text)
    created = find_id_citations(roots)
    citation = _ids(created)[0]

    assert CREATION_STAGE == "32_id_citations"
    assert ATTRIBUTION_STAGE == "33_id_attribution"
    assert created.stage_runs[-1] == CREATION_STAGE
    assert len(citation.nodes) == 1
    assert citation.root_id == citation.attributions == citation.reviews == ()
    assert citation.pin_cite is not None
    pin = citation.pin_cite[-1]
    assert pin.node_id == citation.id_reference[-1].node_id == citation.nodes[0].id
    assert pin.quote == text[pin.span.start : pin.span.end] == "¶ 12 - 14,\n ¶ 16"
    assert [(value.first, value.last, value.kind.value) for value in pin.get_normalized()] == [
        (12, 14, "paragraph"),
        (16, 16, "paragraph"),
    ]
    assert Document.model_validate_json(created.model_dump_json()) == created

    attributed = asyncio.run(attribute_id_citations(created, review=False))
    restored = Document.model_validate_json(attributed.model_dump_json())

    assert restored.get_stage(CREATION_STAGE) == created
    assert restored.get_stage(ATTRIBUTION_STAGE) == attributed
    assert _ids(restored)[0].pin_cite == citation.pin_cite
    assert latest(_ids(restored)[0].root_id) == roots.roots[0].id


@pytest.mark.parametrize("tail", ["Id.", "Id.\n6 A footnote label."])
def test_creation_retains_a_pin_absence_outcome_without_consuming_a_footnote_label(tail: str) -> None:
    created = find_id_citations(_roots(ROOT_TEXT + tail))
    citation = _ids(created)[0]

    assert citation.id_reference[-1].quote == "Id."
    assert citation.pin_cite is None
    assert citation.get_pin_cite() is None
    assert citation.root_id == ()
    assert len(citation.nodes) == 1
    assert Document.model_validate_json(created.model_dump_json()).get_stage(CREATION_STAGE) == created


@pytest.mark.parametrize("next_authority", ["200 F.3d 2 (2001).", "42 U.S.C. § 1983."])
def test_creation_stops_a_pin_before_the_next_recognized_source(next_authority: str) -> None:
    created = find_id_citations(_roots(ROOT_TEXT + "Id. at 3, " + next_authority))
    citations = _ids(created)

    assert len(citations) == 1
    assert citations[0].id_reference[-1].quote == "Id. at 3"
    assert citations[0].pin_cite[-1].quote == "3"
    assert [(pin.first, pin.last) for pin in citations[0].get_pin_cite()] == [(3, 3)]


def test_attribution_requires_creation_and_both_stages_reject_duplicate_runs() -> None:
    roots = _roots(ROOT_TEXT + "Id. at 3.")
    with pytest.raises(ValueError, match=r"(?i)create.*id"):
        asyncio.run(attribute_id_citations(roots, review=False))

    created = find_id_citations(roots)
    with pytest.raises(ValueError, match="already completed"):
        find_id_citations(created)
    with pytest.raises(KeyError, match="Stage has not run"):
        created.get_stage(ATTRIBUTION_STAGE)

    attributed = asyncio.run(attribute_id_citations(created, review=False))
    with pytest.raises(ValueError, match="already completed"):
        asyncio.run(attribute_id_citations(attributed, review=False))


def test_rule_attribution_preserves_a_noncase_barrier_after_a_pinned_id() -> None:
    text = ROOT_TEXT + "Id. at 3, 42 U.S.C. § 1983. Id. at 4."
    created = find_id_citations(_roots(text))
    attributed = asyncio.run(attribute_id_citations(created, review=False))
    first, second = _ids(attributed)

    assert first.pin_cite[-1].quote == "3"
    assert latest(first.root_id) == attributed.roots[0].id
    assert latest(second.root_id) == WITHDRAWN_ROOT_ID
    assert second.attributions[-1].result is AttributionResult.UNRESOLVED
    assert second.reviews == ()
    restored = Document.model_validate_json(attributed.model_dump_json())
    assert restored.get_stage(CREATION_STAGE) == created
    assert all(citation.root_id == () for citation in _ids(restored.get_stage(CREATION_STAGE)))
    assert latest(_ids(restored.get_stage(ATTRIBUTION_STAGE))[1].root_id) == WITHDRAWN_ROOT_ID


@pytest.mark.parametrize("supra", ["Alpha, supra, at 3.", "Alpha,42supra,at3."])
def test_an_unrepresented_supra_blocks_early_id_attribution_and_later_stages_preserve_it(supra: str) -> None:
    text = ROOT_TEXT + f"Gamma v. Delta, 200 F.3d 2 (2001). {supra} Id. at 4."
    document = asyncio.run(grow_leaves(_roots(text), review_leaves=False))
    checkpoint = document.get_stage(ATTRIBUTION_STAGE)
    early_id = _ids(checkpoint)[0]

    assert latest(early_id.root_id) == WITHDRAWN_ROOT_ID
    assert early_id.attributions[-1].result is AttributionResult.UNRESOLVED
    assert not any(isinstance(citation, SupraCitation) for citation in checkpoint.short_citations)
    supra = next(citation for citation in document.short_citations if isinstance(citation, SupraCitation))
    alpha = next(root for root in document.roots if root.get_case_name().plaintiff == "Alpha")
    assert latest(supra.root_id) == alpha.id
    assert _ids(document)[0] == early_id


def test_a_forward_named_reference_does_not_give_id_a_future_root() -> None:
    text = "See Young at 670. Id. at 671. Young v. State, 501 F.3d 660 (2007)."
    document = find_reference_citations(_roots(text))
    document = asyncio.run(attribute_reference_citations(document, review=False))
    reference = next(
        citation for citation in document.short_citations if isinstance(citation, ReferenceCitation)
    )
    assert latest(reference.root_id) == document.roots[0].id

    attributed = asyncio.run(attribute_id_citations(find_id_citations(document), review=False))
    citation = _ids(attributed)[0]

    assert citation.site_span.end < attributed.roots[0].site_span.start
    assert latest(citation.root_id) == WITHDRAWN_ROOT_ID
    assert citation.attributions[-1].result is AttributionResult.UNRESOLVED


def test_review_checks_every_id_including_rule_attached_chains_on_separate_decision_nodes() -> None:
    created = find_id_citations(_roots(ROOT_TEXT + "Id. at 3. Id. at 4."))
    contexts: list[LeafReviewContext] = []

    async def accept(context: LeafReviewContext) -> LeafReviewDecision:
        contexts.append(context)
        return LeafReviewDecision(is_citation=True, root_index=0, reason="Supported preceding source")

    attributed = asyncio.run(attribute_id_citations(created, reviewer=accept))

    assert len(contexts) == 2
    for citation in _ids(attributed):
        assert citation.attributions[0].result is AttributionResult.ATTACHED
        assert citation.attributions[-1].result is AttributionResult.ATTACHED
        assert len([node for node in citation.nodes if node.stage == ATTRIBUTION_STAGE]) == 2
        assert citation.reviews[-1].node_id != citation.attributions[0].node_id
        assert citation.pin_cite[-1].node_id == citation.nodes[0].id
    assert Document.model_validate_json(attributed.model_dump_json()).get_stage(CREATION_STAGE) == created


@pytest.mark.parametrize("first_result", [AttributionResult.REJECTED, AttributionResult.UNRESOLVED])
def test_dummy_id_outcome_blocks_the_next_rule_link_even_if_its_review_fails(
    first_result: AttributionResult,
) -> None:
    created = find_id_citations(_roots(ROOT_TEXT + "Id. at 3. Id. at 4."))
    contexts: list[LeafReviewContext] = []

    async def unresolved_or_rejected_then_fail(
        context: LeafReviewContext,
    ) -> LeafReviewDecision | LeafReviewOutcome:
        contexts.append(context)
        if len(contexts) == 1:
            return LeafReviewDecision(
                is_citation=first_result is AttributionResult.UNRESOLVED,
                root_index=None,
                reason="No supported case antecedent",
            )
        return LeafReviewOutcome(None, failure_reason="Synthetic review failure")

    attributed = asyncio.run(attribute_id_citations(created, reviewer=unresolved_or_rejected_then_fail))
    first, second = _ids(attributed)

    assert latest(first.root_id) == WITHDRAWN_ROOT_ID
    assert first.attributions[0].result is AttributionResult.ATTACHED
    assert first.attributions[-1].result is first_result
    assert first.reviews[-1].decision.is_citation is (first_result is AttributionResult.UNRESOLVED)
    assert first.reviews[-1].decision.root_index is None
    assert [link.value for link in first.root_id] == [attributed.roots[0].id, WITHDRAWN_ROOT_ID]
    assert first.root_id[0].node_id != first.root_id[-1].node_id
    assert second.attributions[-1].result is AttributionResult.UNRESOLVED
    assert latest(second.root_id) == WITHDRAWN_ROOT_ID
    assert [link.value for link in second.root_id] == [WITHDRAWN_ROOT_ID]
    assert second.reviews[-1].failure_reason == "Synthetic review failure"
    prior_id = contexts[1].antecedents[-1]
    assert prior_id["kind"] == "id"
    assert prior_id["root_index"] is None
    assert prior_id["attribution"] is first_result
    assert prior_id["model_reviewed"] is True
    restored = Document.model_validate_json(attributed.model_dump_json())
    assert restored.get_stage(CREATION_STAGE) == created
    assert all(citation.root_id == () for citation in _ids(restored.get_stage(CREATION_STAGE)))
    assert restored.get_stage(ATTRIBUTION_STAGE) == attributed
    assert [link.value for link in _ids(restored.get_stage(ATTRIBUTION_STAGE))[0].root_id] == [
        attributed.roots[0].id,
        WITHDRAWN_ROOT_ID,
    ]
