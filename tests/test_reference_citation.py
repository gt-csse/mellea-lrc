"""Name-and-pin reference creation and attribution have independent checkpoints."""

from __future__ import annotations

import asyncio

import pytest

from mellea_lrc.api import (
    Document,
    attribute_reference_citations,
    attribute_supra_citations_rule,
    find_id_citations,
    find_reference_citations,
    find_supra_citations,
    grow_roots,
    resolve_supra_case_names,
    resolve_supra_pin_cites,
)
from mellea_lrc.extraction.leaf_attribution_review.reviewer import IvrLeafReviewer, LeafReviewOutcome
from mellea_lrc.extraction.supra_attribution_llm import review_supra_attributions
from mellea_lrc.model import Span
from mellea_lrc.model.citations import AttributionResult, LeafReviewDecision, ReferenceCitation
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID
from mellea_lrc.model.ivr import IvrAttempt, IvrRequirementAttempt, IvrRun

CREATION_STAGE = "30_reference_citations"
ATTRIBUTION_STAGE = "31_reference_attribution"
AMBIGUOUS = "Smith v. Jones, 347 U.S. 483 (1954). Smith v. Brown, 348 U.S. 500 (1955). See Smith at 510."


@pytest.fixture(autouse=True)
def prohibit_live_reference_review(monkeypatch) -> None:
    def unexpected_reviewer(*_args, **_kwargs):
        pytest.fail("Reference tests must never construct a live leaf reviewer")

    monkeypatch.setattr(IvrLeafReviewer, "from_profile", classmethod(unexpected_reviewer))


def _roots(source: str) -> Document:
    return asyncio.run(grow_roots(Document.from_source(source)))


def _reference(document: Document) -> ReferenceCitation:
    return next(citation for citation in document.short_citations if isinstance(citation, ReferenceCitation))


def _trace(*, success: bool) -> IvrRun:
    return IvrRun(
        success=success,
        selected_attempt=1,
        attempts=(
            IvrAttempt(
                output='{"is_citation": true, "root_index": 9, "reason": "invalid choice"}',
                requirements=(
                    IvrRequirementAttempt(
                        description="candidate index", passed=False, reason="index out of range", score=None
                    ),
                ),
                request=[{"role": "user", "content": "choose a source authority"}],
                response={"usage": {"output_tokens": 11}},
            ),
            IvrAttempt(
                output='{"is_citation": true, "root_index": 0, "reason": "source supports it"}',
                requirements=(
                    IvrRequirementAttempt(description="candidate index", passed=True, reason=None, score=1.0),
                ),
                request=[
                    {"role": "user", "content": "choose a source authority"},
                    {"role": "assistant", "content": "invalid choice"},
                    {"role": "user", "content": "repair: index out of range"},
                ],
                response={"finish_reason": "stop", "usage": {"output_tokens": 7}},
            ),
        ),
        backend="test-backend",
        model="test-model",
        model_options={"temperature": 0},
        instruction="choose a source authority",
        prefix="source filing",
        grounding_context={"site": "Smith"},
        user_variables={"kind": "reference"},
        output_schema={"type": "object", "required": ["is_citation", "root_index", "reason"]},
    )


@pytest.mark.parametrize("include_pin", [False, True])
def test_reference_constructor_retains_name_anchor_and_creation_readings(include_pin: bool) -> None:
    source = "Smith at 495 - 97, 501 n.2"
    pin_span = Span(9, len(source)) if include_pin else None
    reference = ReferenceCitation.from_source(
        source=source, span=Span(0, 5), stage=CREATION_STAGE, pin_span=pin_span
    )

    assert reference.site_span == Span(0, 5)
    assert reference.reference_name == reference.case_name
    assert reference.case_name[-1].quote == "Smith"
    assert len(reference.nodes) == 1
    assert reference.reference_name[-1].node_id == reference.nodes[0].id
    assert reference.root_id == reference.attributions == reference.reviews == ()
    if include_pin:
        pin = reference.pin_cite[-1]
        assert pin.span == pin_span
        assert pin.quote == "495 - 97, 501 n.2"
        assert pin.node_id == reference.nodes[0].id
        assert [(value.first, value.last, value.footnote) for value in pin.get_normalized()] == [
            (495, 497, None),
            (501, 501, "2"),
        ]
    else:
        assert reference.pin_cite is None
    reference.validate_source(source)
    assert ReferenceCitation.model_validate_json(reference.model_dump_json()) == reference


def test_reference_creation_retains_a_failed_pin_normalization() -> None:
    source = "Smith v. Jones, 347 U.S. 483 (1954). See Smith at 495-490."
    created = find_reference_citations(_roots(source))
    reference = _reference(created)

    assert reference.pin_cite[-1].quote == "495-490"
    assert reference.pin_cite[-1].normalizable is False
    assert reference.pin_cite[-1].normalization_error
    assert reference.pin_cite[-1].node_id == reference.nodes[0].id
    with pytest.raises(ValueError, match="not normalizable"):
        reference.pin_cite[-1].get_normalized()
    assert Document.model_validate_json(created.model_dump_json()) == created


def test_reference_creation_is_independent_and_a_unique_future_name_attaches_without_review() -> None:
    source = "See Iqbal at 670. Iqbal v. Ashcroft, 556 U.S. 662 (2009)."
    roots = _roots(source)
    created = find_reference_citations(roots)
    reference = _reference(created)

    assert reference.reference_name[-1].quote == reference.case_name[-1].quote == "Iqbal"
    assert reference.pin_cite[-1].quote == "670"
    assert reference.root_id == reference.attributions == reference.reviews == ()
    assert len(reference.nodes) == 1

    async def unexpected_review(_context):
        pytest.fail("A unique root name must skip semantic review")

    attributed = asyncio.run(attribute_reference_citations(created, reviewer=unexpected_review))
    restored = Document.model_validate_json(attributed.model_dump_json())
    attached = _reference(restored)

    assert attached.root_id[-1].value == roots.roots[0].id
    assert attached.attributions[-1].result is AttributionResult.ATTACHED
    assert attached.attributions[-1].candidate_root_ids == (roots.roots[0].id,)
    assert tuple(node.stage for node in attached.nodes) == (CREATION_STAGE, ATTRIBUTION_STAGE)
    assert attached.attributions[-1].node_id == attached.nodes[-1].id
    assert attached.reviews == ()
    assert attached.reference_name == reference.reference_name
    assert attached.case_name == reference.case_name
    assert attached.pin_cite == reference.pin_cite
    assert restored.get_stage("10_roots") == roots
    assert restored.get_stage(CREATION_STAGE) == created
    assert restored.get_stage(ATTRIBUTION_STAGE) == attributed
    attached.validate_source(source)


@pytest.mark.parametrize(
    "source", ["No authorities here.", "Smith v. Jones, 347 U.S. 483. See Smith at 495."]
)
def test_reference_stages_require_roots_and_creation_and_reject_repeats(source: str) -> None:
    with pytest.raises(ValueError, match="roots"):
        find_reference_citations(Document.from_source(source))
    roots = _roots(source)
    with pytest.raises(ValueError, match="Create reference"):
        asyncio.run(attribute_reference_citations(roots, review=False))
    created = find_reference_citations(roots)
    with pytest.raises(ValueError, match="already completed"):
        find_reference_citations(created)
    attributed = asyncio.run(attribute_reference_citations(created, review=False))

    assert attributed.stage_runs[-2:] == (CREATION_STAGE, ATTRIBUTION_STAGE)
    assert attributed.get_stage(CREATION_STAGE) == created
    with pytest.raises(ValueError, match="already completed"):
        asyncio.run(attribute_reference_citations(attributed, review=False))


def test_ambiguous_reference_stays_unresolved_without_review() -> None:
    created = find_reference_citations(_roots(AMBIGUOUS))
    attributed = asyncio.run(attribute_reference_citations(created, review=False))
    reference = _reference(attributed)

    assert len(created.roots) == 2
    assert set(reference.attributions[-1].candidate_root_ids) == {root.id for root in created.roots}
    assert reference.attributions[-1].result is AttributionResult.UNRESOLVED
    assert reference.root_id[-1].value == WITHDRAWN_ROOT_ID
    assert reference.reviews == ()
    assert reference.next_stage is None
    assert reference.case_name == _reference(created).case_name
    assert reference.pin_cite == _reference(created).pin_cite


def test_ambiguous_reference_review_records_both_decisions_and_complete_trace() -> None:
    created = find_reference_citations(_roots(AMBIGUOUS))
    trace = _trace(success=True)
    calls = 0

    async def choose_brown(context):
        nonlocal calls
        calls += 1
        assert context.kind == "reference"
        assert context.quote == "Smith"
        assert "<SITE>Smith</SITE> at 510" in context.context
        assert set(context.candidate_root_ids) == {root.id for root in created.roots}
        index = next(
            index
            for index, candidate in enumerate(context.candidates)
            if candidate["case_name"] == "Smith v. Brown"
        )
        return LeafReviewOutcome(
            LeafReviewDecision(is_citation=True, root_index=index, reason="Source supports Brown"), run=trace
        )

    attributed = asyncio.run(attribute_reference_citations(created, reviewer=choose_brown))
    restored = Document.model_validate_json(attributed.model_dump_json())
    reference = _reference(restored)
    brown = next(root for root in created.roots if root.get_case_name().defendant == "Brown")

    assert calls == 1
    assert reference.root_id[-1].value == brown.id
    assert [record.result for record in reference.attributions] == [
        AttributionResult.UNRESOLVED,
        AttributionResult.ATTACHED,
    ]
    assert len(reference.nodes) == 3
    assert reference.attributions[0].node_id == reference.nodes[1].id
    assert reference.attributions[-1].node_id == reference.reviews[-1].node_id == reference.nodes[2].id
    assert [entry.value for entry in reference.root_id] == [WITHDRAWN_ROOT_ID, brown.id]
    assert reference.reviews[-1].ivr == trace
    assert reference.reviews[-1].ivr.attempts[1].request[-1]["content"] == "repair: index out of range"
    assert restored.get_stage(CREATION_STAGE) == created
    assert restored.get_stage(CREATION_STAGE).short_citations[-1].reviews == ()
    assert restored == attributed


@pytest.mark.parametrize("outcome_kind", ["failure", "reject", "uncertain", "invalid_index"])
def test_ambiguous_reference_preserves_failed_rejected_and_unresolved_review_outcomes(
    outcome_kind: str,
) -> None:
    created = find_reference_citations(_roots(AMBIGUOUS))
    trace = _trace(success=outcome_kind != "failure")

    async def decide(_context):
        if outcome_kind == "failure":
            return LeafReviewOutcome(None, run=trace, failure_reason="Synthetic review failure")
        return LeafReviewOutcome(
            LeafReviewDecision(
                is_citation=outcome_kind != "reject",
                root_index=9 if outcome_kind == "invalid_index" else None,
                reason="No supported authority",
            ),
            run=trace,
        )

    attributed = asyncio.run(attribute_reference_citations(created, reviewer=decide))
    restored = Document.model_validate_json(attributed.model_dump_json())
    reference = _reference(restored)

    assert reference.reviews[-1].ivr == trace
    assert len(reference.attributions[-1].candidate_root_ids) == 2
    assert reference.next_stage is None
    if outcome_kind in {"failure", "invalid_index"}:
        assert reference.root_id[-1].value == WITHDRAWN_ROOT_ID
        assert reference.attributions[-1].result is AttributionResult.UNRESOLVED
        assert reference.reviews[-1].decision is None
        assert reference.reviews[-1].failure_reason
    else:
        assert reference.root_id[-1].value == WITHDRAWN_ROOT_ID
        assert reference.attributions[-1].result is (
            AttributionResult.REJECTED if outcome_kind == "reject" else AttributionResult.UNRESOLVED
        )
        assert reference.reviews[-1].decision.is_citation is (outcome_kind != "reject")
    assert reference.case_name == _reference(created).case_name
    assert reference.pin_cite == _reference(created).pin_cite
    assert restored.get_stage(CREATION_STAGE) == created


@pytest.mark.parametrize("outcome_kind", ["failure", "reject", "uncertain"])
def test_reference_with_no_matching_root_retains_review_and_trace_without_inventing_candidates(
    outcome_kind: str,
) -> None:
    source = "See Unknown at 495."
    roots = _roots(source)
    reference = ReferenceCitation.from_source(
        source=source, span=Span(4, 11), pin_span=Span(15, 18), stage=CREATION_STAGE
    )
    created = roots.add_citation(reference).complete(CREATION_STAGE)
    calls = 0
    trace = _trace(success=outcome_kind != "failure")

    async def unsupported(context):
        nonlocal calls
        calls += 1
        assert context.candidate_root_ids == context.candidates == ()
        assert context.kind == "reference"
        if outcome_kind == "failure":
            return LeafReviewOutcome(None, run=trace, failure_reason="Synthetic review failure")
        return LeafReviewOutcome(
            LeafReviewDecision(
                is_citation=outcome_kind != "reject", root_index=None, reason="No source authority available"
            ),
            run=trace,
        )

    attributed = asyncio.run(attribute_reference_citations(created, reviewer=unsupported))
    restored = Document.model_validate_json(attributed.model_dump_json())
    updated = _reference(restored)

    assert calls == 1
    assert updated.attributions[-1].candidate_root_ids == ()
    assert updated.reviews[-1].candidate_root_ids == ()
    assert updated.reviews[-1].ivr == trace
    if outcome_kind == "failure":
        assert updated.root_id[-1].value == WITHDRAWN_ROOT_ID
        assert updated.reviews[-1].failure_reason == "Synthetic review failure"
    else:
        assert updated.root_id[-1].value == WITHDRAWN_ROOT_ID
    assert updated.attributions[-1].result is (
        AttributionResult.REJECTED if outcome_kind == "reject" else AttributionResult.UNRESOLVED
    )
    assert updated.case_name == reference.case_name
    assert updated.pin_cite == reference.pin_cite
    assert restored.get_stage(CREATION_STAGE) == created


@pytest.mark.parametrize("outcome_kind", ["rule_only", "attach", "reject", "failure"])
def test_later_shared_stages_preserve_reference_fields_attachment_and_review(outcome_kind: str) -> None:
    created = find_reference_citations(_roots(AMBIGUOUS))

    async def decide(_context):
        if outcome_kind == "failure":
            return LeafReviewOutcome(
                None, run=_trace(success=False), failure_reason="Synthetic review failure"
            )
        return LeafReviewDecision(
            is_citation=outcome_kind != "reject",
            root_index=0 if outcome_kind == "attach" else None,
            reason="Reviewed source context",
        )

    document = asyncio.run(
        attribute_reference_citations(created, review=outcome_kind != "rule_only", reviewer=decide)
    )
    settled = _reference(document)

    for stage in (
        find_id_citations,
        find_supra_citations,
        resolve_supra_case_names,
        resolve_supra_pin_cites,
        attribute_supra_citations_rule,
    ):
        document = stage(document)
        assert _reference(document) == settled

    async def unexpected_review(_context):
        pytest.fail("A reference attribution must never be reviewed again by the general leaf stage")

    document = asyncio.run(review_supra_attributions(document, reviewer=unexpected_review))
    assert _reference(document) == settled
    restored = Document.model_validate_json(document.model_dump_json())
    assert _reference(restored) == settled
    assert restored.get_stage(ATTRIBUTION_STAGE).short_citations[-1] == settled
