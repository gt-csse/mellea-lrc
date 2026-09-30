"""Behavioral tests for staged leaf citation extraction and attribution."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mellea_lrc.api import Document, grow_roots
from mellea_lrc.extraction.id_attribution_llm import review_id_attributions
from mellea_lrc.extraction.leaf_attribution_llm.reviewer import LeafReviewContext, LeafReviewOutcome
from mellea_lrc.model.citations import (
    AttributionResult,
    CaseNameField,
    IdCitation,
    LeafReviewDecision,
    ReferenceCitation,
    ShortReporterCitation,
    SupraCitation,
)
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID
from mellea_lrc.model.ivr import IvrAttempt, IvrRequirementAttempt, IvrRun
from mellea_lrc.model.span import Span
from mellea_lrc.workflows.grow_leaves import grow_leaves


def rooted(tmp_path: Path, text: str) -> Document:
    source = tmp_path / "leaf-test.txt"
    source.write_text(text)
    return asyncio.run(grow_roots(Document.from_source(source)))


def test_workflow_grows_leaves_after_roots_without_validation(tmp_path: Path) -> None:
    text = "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 347 U.S. at 495."
    roots = rooted(tmp_path, text)

    result = asyncio.run(grow_leaves(roots, review_leaves=False))

    assert result.roots == roots.roots
    assert len(result.short_reporters) == 1
    short = result.short_reporters[0]
    assert isinstance(short, ShortReporterCitation)
    assert short.root_id[-1].value == roots.roots[0].id
    assert short.case_name[-1].quote == "Smith"
    assert short.pin_cite[-1].quote == "495"
    assert "35_leaf_attribution_llm" not in result.stage_runs


def test_short_reporter_rule_rejects_shared_volume_or_name_disagreement(tmp_path: Path) -> None:
    from mellea_lrc.extraction.leaf_attribution_rule import attribute_leaves_rule
    from mellea_lrc.extraction.leaf_case_names import resolve_leaf_case_names
    from mellea_lrc.extraction.short_reporter_locator import find_short_reporter_citations

    text = (
        "Smith v. Jones, 347 U.S. 483 (1954). "
        "Brown v. Green, 347 U.S. 500 (1954). See Wrong, 347 U.S. at 495."
    )
    document = rooted(tmp_path, text)
    document = find_short_reporter_citations(document)
    document = resolve_leaf_case_names(document)
    document = attribute_leaves_rule(document)

    short = document.short_reporters[0]
    assert short.root_id == ()
    assert short.attributions[-1].result is AttributionResult.UNRESOLVED
    assert len(short.attributions[-1].candidate_root_ids) == 2


def test_short_reporter_name_disambiguates_shared_reporter(tmp_path: Path) -> None:
    text = (
        "Smith v. Jones, 347 U.S. 483 (1954). "
        "Brown v. Green, 347 U.S. 500 (1954). See Brown, 347 U.S. at 510."
    )
    roots = rooted(tmp_path, text)
    result = asyncio.run(grow_leaves(roots, review_leaves=False))

    short = result.short_reporters[0]
    brown = next(root for root in roots.roots if root.get_case_name().plaintiff == "Brown")
    assert short.root_id[-1].value == brown.id
    assert set(short.attributions[-1].candidate_root_ids) == {root.id for root in roots.roots}


def test_id_chain_attaches_to_previous_leaf_and_stops_at_noncase_barrier(tmp_path: Path) -> None:
    text = (
        "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 347 U.S. at 494. "
        "Id. at 495. Id. at 496. 42 U.S.C. § 1983. Id. at 497."
    )
    document = rooted(tmp_path, text)
    document = asyncio.run(grow_leaves(document, review_leaves=False))
    ids = [citation for citation in document.short_citations if isinstance(citation, IdCitation)]

    assert len(ids) == 3
    assert all(citation.root_id[-1].value == document.roots[0].id for citation in ids[:2])
    assert ids[2].root_id == ()
    assert [citation.id_reference[-1].quote for citation in ids] == ["Id. at 495", "Id. at 496", "Id. at 497"]


@pytest.mark.xfail(strict=True, reason="eyecite does not emit Fed. R. Civ. P. as a chronology barrier")
def test_id_after_civil_rule_is_unresolved(tmp_path: Path) -> None:
    text = "Smith v. Jones, 347 U.S. 483 (1954). Fed. R. Civ. P. 12(b)(6). Id. at 497."
    document = asyncio.run(grow_leaves(rooted(tmp_path, text), review_leaves=False))
    id_citation = next(citation for citation in document.short_citations if isinstance(citation, IdCitation))
    assert id_citation.root_id == ()


def test_supra_uses_only_preceding_roots(tmp_path: Path) -> None:
    text = "Smith v. Jones, 347 U.S. 483 (1954). Smith, supra, at 495."
    document = asyncio.run(grow_leaves(rooted(tmp_path, text), review_leaves=False))

    supra = next(citation for citation in document.short_citations if isinstance(citation, SupraCitation))
    assert supra.case_name[-1].quote == "Smith"
    assert supra.root_id[-1].value == document.roots[0].id
    assert supra.pin_cite[-1].quote == "495"


def test_short_leaf_reads_multiline_names_and_pin_targets(tmp_path: Path) -> None:
    text = "Smith v. Jones, 347 U.S. 483 (1954). See Smith,\n 347 U.S. at\n 495 - 97, 501 n.2."
    document = asyncio.run(grow_leaves(rooted(tmp_path, text), review_leaves=False))

    short = document.short_reporters[0]
    assert short.case_name[-1].quote == "Smith"
    assert short.short_locator[-1].quote == "347 U.S. at\n 495 - 97, 501 n.2"
    assert text[short.site_span.start : short.site_span.end] == short.short_locator[-1].quote
    assert short.short_locator[-1].get_normalized().pin_page == "495"
    assert short.pin_cite[-1].quote == "495 - 97, 501 n.2"
    targets = short.pin_cite[-1].get_normalized()
    assert [(item.first, item.last, item.footnote) for item in targets] == [(495, 497, None), (501, 501, "2")]


def test_id_citation_reads_multiline_paragraph_pin_lists_with_exact_spans(tmp_path: Path) -> None:
    text = "Smith v. Jones, 347 U.S. 483 (1954). See id. at ¶ 12 - 14,\n ¶ 16."
    document = asyncio.run(grow_leaves(rooted(tmp_path, text), review_leaves=False))
    id_citation = next(citation for citation in document.short_citations if isinstance(citation, IdCitation))

    assert id_citation.id_reference[-1].quote == text[id_citation.site_span.start : id_citation.site_span.end]
    assert id_citation.pin_cite[-1].quote == "¶ 12 - 14,\n ¶ 16"
    normalized = id_citation.pin_cite[-1].get_normalized()
    assert [(pin.first, pin.last, pin.kind.value) for pin in normalized] == [
        (12, 14, "paragraph"),
        (16, 16, "paragraph"),
    ]
    restored = Document.model_validate_json(document.model_dump_json())
    restored_id = next(citation for citation in restored.short_citations if isinstance(citation, IdCitation))
    restored_id.validate_source(text)
    assert restored_id == id_citation


def test_id_discovery_does_not_consume_a_bare_following_footnote_label(tmp_path: Path) -> None:
    text = "Smith v. Jones, 347 U.S. 483 (1954). Id.\n6 A short footnote label."
    document = asyncio.run(grow_leaves(rooted(tmp_path, text), review_leaves=False))
    id_citation = next(c for c in document.short_citations if isinstance(c, IdCitation))

    assert text[id_citation.site_span.start : id_citation.site_span.end] == "Id."
    assert not id_citation.pin_cite


def test_name_only_reference_can_resolve_to_a_later_full_root(tmp_path: Path) -> None:
    text = "See Iqbal. Iqbal v. Ashcroft, 556 U.S. 662 (2009)."
    document = rooted(tmp_path, text)

    async def select_the_future_authority(context):
        assert context.kind == "reference"
        assert len(context.candidate_root_ids) == 1
        assert context.candidates[0]["case_name"] == "Iqbal v. Ashcroft"
        return LeafReviewDecision(is_citation=True, root_index=0, reason="Later root matches the reference")

    result = asyncio.run(grow_leaves(document, reviewer=select_the_future_authority))
    reference = next(c for c in result.short_citations if isinstance(c, ReferenceCitation))

    assert reference.root_id[-1].value == result.roots[0].id
    assert reference.reviews[-1].decision.is_citation


def test_corporate_punctuation_alias_resolves_the_full_reporter_root(tmp_path: Path) -> None:
    text = (
        "Acme Holdings, Inc. v. Capital Partners LLC, 123 F.3d 400 (2000). "
        "See Acme Holdings, Inc., 123 F.3d at 410."
    )
    document = asyncio.run(grow_leaves(rooted(tmp_path, text), review_leaves=False))
    short = document.short_reporters[0]

    assert short.case_name[-1].quote == "Acme Holdings, Inc."
    assert short.root_id[-1].value == document.roots[0].id


def test_full_citation_constituents_and_court_label_are_masked_from_reference_proposals(
    tmp_path: Path,
) -> None:
    text = "D.C. v. California, 123 F.3d 400 (D.C. Cir. 2000). See D.C."
    document = asyncio.run(grow_leaves(rooted(tmp_path, text), review_leaves=False))
    root = document.roots[0]
    references = [c for c in document.short_citations if isinstance(c, ReferenceCitation)]

    assert len(references) == 1
    assert text[references[0].site_span.start : references[0].site_span.end] == "D.C."
    assert references[0].site_span.start == text.rindex("D.C.")
    for name in type(root).model_fields:
        value = getattr(root, name)
        if isinstance(value, tuple):
            for field in value:
                if hasattr(field, "span") and field.span is not None:
                    assert not references[0].site_span.overlaps(field.span)

    context = LeafReviewContext.from_document(document, references[0])
    assert "<SITE>D.C.</SITE>" in context.context
    assert "The words inside <SITE> must themselves function as a reference" in context.prefix


def test_parenthesized_full_citation_name_is_not_reproposed_as_a_reference(tmp_path: Path) -> None:
    from mellea_lrc.extraction.reference_citations import find_reference_citations

    text = "Smith v. Jones (123 U.S. 45 (2000))."
    document = find_reference_citations(rooted(tmp_path, text))

    assert document.full_locators[0].case_name[-1].quote == "Smith v. Jones"
    assert not [
        citation
        for citation in document.short_citations
        if isinstance(citation, ReferenceCitation) and citation.reference_name[-1].quote == "Jones"
    ]


@pytest.mark.parametrize("outcome_kind", ["reject", "uncertain", "failure"])
def test_id_semantic_review_reject_uncertain_and_failure(tmp_path: Path, outcome_kind: str) -> None:
    text = "Smith v. Jones, 347 U.S. 483 (1954). Id. at 495."
    document = rooted(tmp_path, text)

    async def reviewer(_context):
        if outcome_kind == "reject":
            return LeafReviewDecision(is_citation=False, root_index=None, reason="Not an authority citation")
        if outcome_kind == "uncertain":
            return LeafReviewDecision(is_citation=True, root_index=None, reason="No supported root")
        return LeafReviewOutcome(None, failure_reason="Synthetic reviewer failure")

    result = asyncio.run(grow_leaves(document, reviewer=reviewer))
    id_citation = next(c for c in result.short_citations if isinstance(c, IdCitation))
    review = id_citation.reviews[-1]

    assert result.stage_runs[-1] == "37_id_attribution_llm"
    if outcome_kind == "reject":
        assert review.decision is not None and not review.decision.is_citation
        assert id_citation.root_id[-1].value == WITHDRAWN_ROOT_ID
    elif outcome_kind == "uncertain":
        assert review.decision is not None and review.decision.is_citation
        assert id_citation.root_id[-1].value is None
    else:
        assert review.failure_reason == "Synthetic reviewer failure"
        # A failed review preserves the preceding rule attribution.
        assert id_citation.root_id[-1].value == result.roots[0].id


def test_id_review_checkpoint_restores_stage_36_and_only_scores_new_review_links(tmp_path: Path) -> None:
    text = "Smith v. Jones, 347 U.S. 483 (1954). Id. at 495. Id. at 496."
    before_review = asyncio.run(grow_leaves(rooted(tmp_path, text), review_leaves=False))
    calls = 0

    async def accept_then_fail(_context):
        nonlocal calls
        calls += 1
        if calls == 1:
            return LeafReviewDecision(is_citation=True, root_index=0, reason="Matches preceding case")
        return LeafReviewOutcome(None, failure_reason="Synthetic reviewer failure")

    after_review = asyncio.run(review_id_attributions(before_review, reviewer=accept_then_fail))
    assert after_review.get_stage("36_id_attribution") == before_review
    assert after_review.get_stage("37_id_attribution_llm").stage_runs[-1] == "37_id_attribution_llm"


def test_id_review_persists_complete_ivr_trace_and_checkpoint_removes_it(tmp_path: Path) -> None:
    text = "Smith v. Jones, 347 U.S. 483 (1954). Id. at 495."
    before_review = asyncio.run(grow_leaves(rooted(tmp_path, text), review_leaves=False))
    trace = IvrRun(
        success=True,
        selected_attempt=1,
        attempts=(
            IvrAttempt(
                output='{"is_citation": true, "root_index": 9, "reason": "bad choice"}',
                requirements=(
                    IvrRequirementAttempt(
                        description="candidate index",
                        passed=False,
                        reason="index is out of range",
                        score=None,
                    ),
                ),
                request=[{"role": "user", "content": "first prompt"}],
                response={"finish_reason": "stop", "usage": {"output_tokens": 11}},
            ),
            IvrAttempt(
                output='{"is_citation": true, "root_index": 0, "reason": "matches"}',
                requirements=(
                    IvrRequirementAttempt(description="candidate index", passed=True, reason=None, score=1.0),
                ),
                request=[
                    {"role": "user", "content": "first prompt"},
                    {"role": "assistant", "content": "bad choice"},
                    {"role": "user", "content": "repair: index is out of range"},
                ],
                response={"finish_reason": "stop", "usage": {"output_tokens": 7}},
            ),
        ),
        backend="test-backend",
        model="test-model",
        model_options={"temperature": 0},
        instruction="choose an antecedent",
        prefix="prior case references",
        grounding_context={"site": "Id. at 495"},
        user_variables={"citation_id": "id:1"},
        output_schema={"type": "object", "required": ["is_citation", "root_index", "reason"]},
    )

    async def reviewer(_context):
        return LeafReviewOutcome(
            LeafReviewDecision(is_citation=True, root_index=0, reason="matches"), run=trace
        )

    reviewed = asyncio.run(review_id_attributions(before_review, reviewer=reviewer))
    restored = Document.model_validate_json(reviewed.model_dump_json())
    id_citation = next(c for c in restored.short_citations if isinstance(c, IdCitation))
    persisted = id_citation.reviews[-1].ivr

    assert persisted == trace
    assert persisted.attempts[1].request[-1]["content"] == "repair: index is out of range"
    assert restored.get_stage("36_id_attribution") == before_review
    assert restored.get_stage("37_id_attribution_llm").short_citations[-1].reviews[-1].ivr == trace


def test_root_repeated_occurrence_is_a_distinct_leaf_not_a_new_root(tmp_path: Path) -> None:
    text = "Smith v. Jones, 347 U.S. 483 (1954). Smith v. Jones, 347 U.S. 483 (1954)."
    roots = rooted(tmp_path, text)
    result = asyncio.run(grow_leaves(roots, review_leaves=False))

    assert len(result.roots) == len(roots.roots) == 1
    assert len(result.full_locators) == 2
    assert result.full_locators[1].root_id[-1].value == roots.roots[0].id


def test_leaf_stage_checkpoints_round_trip_and_cannot_repeat(tmp_path: Path) -> None:
    from mellea_lrc.extraction.short_reporter_locator import find_short_reporter_citations

    text = "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 347 U.S. at 495."
    roots = rooted(tmp_path, text)
    leaves = find_short_reporter_citations(roots)

    assert leaves.get_stage("10_roots") == roots
    assert leaves.get_stage("28_short_reporter_citations") == leaves
    assert Document.model_validate_json(leaves.model_dump_json()) == leaves
    with pytest.raises(ValueError, match="already completed"):
        find_short_reporter_citations(leaves)


def test_leaf_workflow_requires_only_roots_and_review_index_is_checked(tmp_path: Path) -> None:
    text = (
        "Smith v. Jones, 347 U.S. 483 (1954). "
        "Brown v. Green, 347 U.S. 500 (1954). See Unknown, 347 U.S. at 510."
    )
    document = rooted(tmp_path, text)

    # A document returned from grow_roots has no validation decisions or findings.
    assert not hasattr(document.roots[0], "validation")

    async def invalid_index(_context):
        return LeafReviewDecision(is_citation=True, root_index=8, reason="invalid index")

    result = asyncio.run(grow_leaves(document, reviewer=invalid_index))
    short = result.short_reporters[0]
    assert short.reviews[-1].failure_reason is not None
    assert short.root_id == ()


def test_reference_proposals_do_not_treat_ordinary_prose_as_citation(tmp_path: Path) -> None:
    text = (
        "Smith v. Jones, 347 U.S. 483 (1954). "
        "The Smith family lives nearby; this sentence discusses a person, not the case."
    )

    async def reject_ordinary_mention(_context):
        return LeafReviewDecision(is_citation=False, root_index=None, reason="ordinary party mention")

    document = asyncio.run(grow_leaves(rooted(tmp_path, text), reviewer=reject_ordinary_mention))

    reference = next(
        citation for citation in document.short_citations if isinstance(citation, ReferenceCitation)
    )
    assert reference.attributions[-1].result is AttributionResult.REJECTED
    assert reference.root_id[-1].value == WITHDRAWN_ROOT_ID


@pytest.mark.parametrize(
    ("citation_type", "source", "span", "field"),
    [
        (IdCitation, "Id.", Span(0, 3), "id_reference"),
        (SupraCitation, "Smith, supra", Span(0, 12), "supra_reference"),
        (ReferenceCitation, "Smith", Span(0, 5), "reference_name"),
    ],
)
def test_leaf_types_reject_empty_required_source_fields(
    citation_type, source: str, span: Span, field: str
) -> None:
    citation = citation_type.from_source(source=source, span=span, stage="31_reference_citations")
    with pytest.raises(ValueError):
        citation_type.model_validate({**citation.model_dump(mode="python"), field: ()})


@pytest.mark.parametrize(
    ("citation_type", "source", "span"),
    [
        (IdCitation, "Id.", Span(0, 3)),
        (SupraCitation, "Smith, supra", Span(0, 12)),
        (ReferenceCitation, "Smith", Span(0, 5)),
    ],
)
def test_leaf_types_reject_fields_attached_to_foreign_nodes(citation_type, source: str, span: Span) -> None:
    citation = citation_type.from_source(source=source, span=span, stage="31_reference_citations")
    citation = citation.record("foreign_field")
    foreign = CaseNameField.from_source(source, span, node_id="foreign:node")
    with pytest.raises(ValueError, match="missing node"):
        citation_type.model_validate(
            {
                **citation.model_dump(mode="python"),
                "case_name": (foreign,),
            }
        )
