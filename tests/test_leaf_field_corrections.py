"""Saved-validation leaf rereading without retrieval or live model requests."""

from __future__ import annotations

import asyncio
import json

import pytest

from mellea_lrc.api import correct_leaf_fields
from mellea_lrc.extraction.leaf_field_corrections import SUBSTAGE
from mellea_lrc.extraction.leaf_field_corrections.reviewer import (
    IvrLeafFieldCorrectionReviewer,
    LeafFieldCorrectionOutcome,
)
from mellea_lrc.model.citations import (
    CaseName,
    CaseNameKind,
    FullReporterCitation,
    IdCitation,
    IdentityVerdict,
    LeafReviewDecision,
    ShortReporterCitation,
)
from mellea_lrc.model.citations.leaf_field_correction import (
    LeafFieldCorrectionDecision,
    LeafFieldCorrectionProposal,
)
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactCandidateDocket,
    ReporterExactDocket,
    ReporterExactLookup,
    ReporterExactLookupOutcome,
    ReporterExactLookupQuery,
)
from mellea_lrc.model.document import Document
from mellea_lrc.model.ivr import IvrAttempt, IvrRequirementAttempt, IvrRun
from mellea_lrc.model.preprocessed_document import PreprocessingMetadata
from mellea_lrc.model.source import SourceMetadata
from mellea_lrc.model.span import Span
from mellea_lrc.providers.courtlistener.models import (
    CourtListenerCitationLookup,
    CourtListenerCluster,
    CourtListenerDocket,
)
from mellea_lrc.workflows.grow_leaves import grow_leaves

ROOT_TEXT = "Smith v. Jones, 347 U.S. 483 (1954). "


def span(text, quote, *, after=0):
    start = text.index(quote, after)
    return Span(start, start + len(quote))


def rooted(tail, *, validate=True):
    text = ROOT_TEXT + tail
    document = Document(
        text=text, source_metadata=SourceMetadata(), preprocessing_metadata=PreprocessingMetadata()
    )
    root = FullReporterCitation.from_locator(
        citation_id="root", substage="1_reporter_locator", source=text, span=span(text, "347 U.S. 483")
    )
    document = document.add_citation(root).complete_substage("1_reporter_locator")
    root = (
        root.record("grow_roots.root_formation.rule")
        .with_root(root.id)
        .with_case_name(text, span(text, "Smith v. Jones"))
    )
    document = document.replace_citation(root).complete_substage("grow_roots.root_formation.rule")
    if validate:
        root = root.record("11_reporter_lookup")
        root = root.with_reporter_exact_lookup(
            ReporterExactLookup(
                node_id=root.nodes[-1].id,
                outcome=ReporterExactLookupOutcome.UNIQUE,
                query=ReporterExactLookupQuery(volume=347, edition="U.S.", page="483"),
                response=CourtListenerCitationLookup(
                    citation="347 U.S. 483",
                    status=200,
                    clusters=[
                        CourtListenerCluster(
                            id="42", case_name="Smith v. Jones", date_filed="1954-05-17", court_id="scotus"
                        )
                    ],
                    found_pages=["495"],
                ),
            )
        )
        document = document.replace_citation(root).complete_substage("11_reporter_lookup")
        root = root.record("14_root_validation").with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
        document = document.replace_citation(root).complete_substage("14_root_validation")
    return document


def short(document, quote="347 U.S. at 999", *, name=None, pin=None, citation_id="leaf", after=None):
    after = len(ROOT_TEXT) if after is None else after
    leaf = (
        ShortReporterCitation.from_short_locator(
            citation_id=citation_id,
            substage="grow_leaves.short_reporter_citations.discovery",
            source=document.text,
            span=span(document.text, quote, after=after),
            pin_cite_span=span(document.text, pin, after=after) if pin is not None else None,
        )
        .record("grow_leaves.short_reporter_citations.discovery")
        .with_root("root")
    )
    if name is not None:
        leaf = leaf.with_case_name(document.text, span(document.text, name, after=after))
    return document.add_citation(leaf)


def decision(context, replacements=None, *, normalized=None):
    replacements = replacements or {}
    return LeafFieldCorrectionDecision(
        fields=tuple(
            LeafFieldCorrectionProposal(
                field=window.field,
                propose_replacement=window.field in replacements,
                quote=replacements.get(window.field),
                normalized=normalized
                if window.field == "case_name" and window.field in replacements
                else None,
                reason="Reread this occurrence"
                if window.field in replacements
                else "Written reading is already correct",
            )
            for window in context.windows
        ),
        reason="Correct the source extraction, preserving the written citation",
    )


def trace(output, *, success=True):
    return IvrRun(
        success=success,
        selected_attempt=1,
        attempts=(
            IvrAttempt(
                output='{"fields": []}',
                requirements=(
                    IvrRequirementAttempt(
                        description="local grounding",
                        passed=False,
                        reason="Quote belongs to the root",
                        score=0,
                    ),
                ),
                request=[{"role": "user", "content": "first source reading"}],
                response={"finish_reason": "stop"},
            ),
            IvrAttempt(
                output=output,
                requirements=(
                    IvrRequirementAttempt(
                        description="local grounding",
                        passed=success,
                        reason=None if success else "still unsupported",
                        score=1 if success else 0,
                    ),
                ),
                request=[{"role": "user", "content": "repair: Quote belongs to the root"}],
                response={"finish_reason": "stop"},
            ),
        ),
        backend="fake",
        model="fake-model",
        model_options={},
        instruction="reread fields",
        prefix="saved root",
        grounding_context={},
        user_variables={},
        output_schema={},
    )


def test_missing_leaf_name_and_pin_append_grounded_histories_and_checkpoint():
    before = short(rooted("See Smith, 347 U.S. at 999.")).complete_substage(
        "grow_leaves.short_reporter_citations.discovery"
    )
    contexts = []

    async def reviewer(context):
        contexts.append(context)
        return decision(context, {"case_name": "Smith", "pin_cite": "999"})

    after = asyncio.run(correct_leaf_fields(before, reviewer=reviewer))
    leaf = after.short_reporters[0]
    assert leaf.case_name[-1].quote == "Smith"
    assert leaf.pin_cite[-1].quote == "999"
    assert leaf.site_span == before.short_reporters[0].site_span
    assert leaf.short_locator == before.short_reporters[0].short_locator
    assert leaf.root_id == before.short_reporters[0].root_id
    assert after.roots == before.roots
    review = leaf.leaf_field_correction_reviews[-1]
    assert all(item.applied for item in review.grounded)
    assert {item.field for item in review.grounded} == {"case_name", "pin_cite"}
    lookup_ref = next(ref for ref in review.evidence_refs if ref.history == "reporter_exact_lookup")
    assert lookup_ref.selected_candidate_index == 0
    assert any(ref.history == "identity_judgments" for ref in review.evidence_refs)
    assert "found_pages" not in json.dumps(contexts[0].validation_evidence)
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored == after
    assert restored.get_substage("grow_leaves.short_reporter_citations.discovery") == before
    assert restored.get_substage(SUBSTAGE) == after


def test_attached_full_repeat_rereads_its_own_name_court_date_and_pin():
    document = rooted("See Smith v. Jones, 347 U.S. 483, 999 (2d Cir. 1955).")
    start = len(ROOT_TEXT)
    leaf = (
        FullReporterCitation.from_locator(
            citation_id="repeat",
            substage="grow_leaves.short_reporter_citations.discovery",
            source=document.text,
            span=span(document.text, "347 U.S. 483", after=start),
        )
        .record("grow_leaves.short_reporter_citations.discovery")
        .with_root("root")
    )
    leaf = leaf.with_case_name(document.text, span(document.text, "See Smith v. Jones", after=start))
    leaf = leaf.with_pin_cite(document.text, span(document.text, "999", after=start))
    before = document.add_citation(leaf).complete_substage("grow_leaves.short_reporter_citations.discovery")

    async def reviewer(context):
        assert context.kind == "reporter"
        return decision(context, {"case_name": "Smith v. Jones", "court": "2d Cir.", "date": "1955"})

    after = asyncio.run(correct_leaf_fields(before, reviewer=reviewer))
    repeat = next(item for item in after.leaves if item.id == "repeat")
    assert repeat.case_name[0].quote == "See Smith v. Jones"
    assert repeat.case_name[-1].quote == "Smith v. Jones"
    assert repeat.court[-1].quote == "2d Cir."
    assert repeat.date[-1].quote == "1955"
    assert repeat.pin_cite == leaf.pin_cite
    assert repeat.locator == leaf.locator
    assert after.roots == before.roots


@pytest.mark.parametrize("quote", ["Smith v. Jones", "Opinion-only caption", "Neighbor v. Else"])
def test_root_neighbor_and_source_opinion_quotes_cannot_replace_leaf_fields(quote):
    document = rooted("See Smith, 347 U.S. at 999. Neighbor v. Else, 500 U.S. 10 (1999).")
    document = short(document, name="Smith", pin="999")
    other = (
        FullReporterCitation.from_locator(
            citation_id="neighbor",
            substage="grow_leaves.short_reporter_citations.discovery",
            source=document.text,
            span=span(document.text, "500 U.S. 10"),
        )
        .record("grow_leaves.short_reporter_citations.discovery")
        .with_case_name(document.text, span(document.text, "Neighbor v. Else"))
    )
    before = document.add_citation(other).complete_substage("grow_leaves.short_reporter_citations.discovery")

    async def reviewer(context):
        return decision(context, {"case_name": quote})

    after = asyncio.run(correct_leaf_fields(before, reviewer=reviewer))
    leaf = after.short_reporters[0]
    assert leaf.case_name == before.short_reporters[0].case_name
    assert leaf.leaf_field_correction_reviews[-1].decision is None
    assert "bounded source context" in leaf.leaf_field_correction_reviews[-1].failure_reason


@pytest.mark.parametrize("quote", ["998", "495", "347", "1954"])
def test_printed_wrong_pin_cannot_be_replaced_by_canonical_page_volume_or_year(quote):
    before = short(rooted("See Smith, 347 U.S. at 999."), name="Smith", pin="999").complete_substage(
        "grow_leaves.short_reporter_citations.discovery"
    )

    async def reviewer(context):
        return decision(context, {"pin_cite": quote})

    after = asyncio.run(correct_leaf_fields(before, reviewer=reviewer))
    leaf = after.short_reporters[0]
    assert leaf.pin_cite == before.short_reporters[0].pin_cite
    assert leaf.get_pin_cite()[0].first == 999
    assert leaf.leaf_field_correction_reviews[-1].failure_reason is not None


def test_fuzzy_name_grounding_keeps_actual_source_bytes_and_evidence():
    before = short(rooted("See Smith, 347 U.S. at 999."), pin="999").complete_substage(
        "grow_leaves.short_reporter_citations.discovery"
    )

    async def reviewer(context):
        return decision(context, {"case_name": "Smth"})

    after = asyncio.run(correct_leaf_fields(before, reviewer=reviewer))
    leaf = after.short_reporters[0]
    assert leaf.case_name[-1].quote == "Smith"
    grounded = leaf.leaf_field_correction_reviews[-1].grounded[0]
    assert grounded.match_type.value == "edit_distance"
    assert grounded.edits == 1
    assert grounded.quote == "Smith"


def test_same_span_case_name_normalization_is_a_new_grounded_reading():
    before = short(
        rooted("See Smith vs. Jones, 347 U.S. at 999."), name="Smith vs. Jones", pin="999"
    ).complete_substage("grow_leaves.short_reporter_citations.discovery")
    normalized = CaseName(kind=CaseNameKind.ADVERSARIAL, plaintiff="Smith", defendant="Jones")

    async def reviewer(context):
        return decision(context, {"case_name": "Smith vs. Jones"}, normalized=normalized)

    after = asyncio.run(correct_leaf_fields(before, reviewer=reviewer))
    leaf = after.short_reporters[0]
    assert len(leaf.case_name) == 2
    assert leaf.case_name[-1].span == leaf.case_name[0].span
    assert leaf.case_name[-1].get_normalized() == normalized
    assert leaf.case_name[-1].normalized_by == "model"
    assert leaf.leaf_field_correction_reviews[-1].grounded[0].applied


@pytest.mark.parametrize("failure", ["no_change", "ivr_failed", "transport", "no_decision"])
def test_failure_and_no_change_are_persisted_without_new_field_readings(failure):
    before = short(rooted("See Smith, 347 U.S. at 999."), name="Smith", pin="999").complete_substage(
        "grow_leaves.short_reporter_citations.discovery"
    )

    async def reviewer(context):
        answer = decision(context)
        if failure == "ivr_failed":
            return LeafFieldCorrectionOutcome(answer, run=trace(answer.model_dump_json(), success=False))
        if failure == "transport":
            raise RuntimeError("synthetic transport failure")
        if failure == "no_decision":
            return LeafFieldCorrectionOutcome(None)
        return LeafFieldCorrectionOutcome(answer, run=trace(answer.model_dump_json()))

    after = asyncio.run(correct_leaf_fields(before, reviewer=reviewer))
    leaf = after.short_reporters[0]
    assert leaf.case_name == before.short_reporters[0].case_name
    assert leaf.pin_cite == before.short_reporters[0].pin_cite
    review = leaf.leaf_field_correction_reviews[-1]
    assert (review.decision is not None) == (failure == "no_change")
    assert (review.failure_reason is None) == (failure == "no_change")
    assert not review.grounded
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored == after
    if review.ivr is not None:
        assert review.ivr.attempts[1].request[-1]["content"] == "repair: Quote belongs to the root"


@pytest.mark.parametrize("validation, review", [(False, True), (True, False)])
def test_no_validation_or_rule_only_checkpoint_never_calls_reviewer(validation, review):
    before = short(
        rooted("See Smith, 347 U.S. at 999.", validate=validation), name="Smith", pin="999"
    ).complete_substage("grow_leaves.short_reporter_citations.discovery")

    async def forbidden(_context):
        pytest.fail("No leaf correction model call is authorized by this input")

    after = asyncio.run(correct_leaf_fields(before, review=review, reviewer=forbidden))
    assert after.citations == before.citations
    assert after.substage_runs[-1] == SUBSTAGE


def test_grow_leaves_without_validate_roots_finishes_with_empty_correction_checkpoint():
    before = rooted("See Smith, 347 U.S. at 999.", validate=False)

    async def attach(_context):
        return LeafReviewDecision(is_citation=True, root_index=0, reason="Printed short name matches")

    async def forbidden(_context):
        pytest.fail("Extraction-only roots provide no validation assistance")

    after = asyncio.run(grow_leaves(before, reviewer=attach, correction_reviewer=forbidden))
    assert after.substage_runs[-1] == SUBSTAGE
    assert after.short_reporters[0].pin_cite[-1].quote == "999"
    assert not after.short_reporters[0].leaf_field_correction_reviews


def test_stage_and_native_evidence_reference_guards():
    before = short(rooted("See Smith, 347 U.S. at 999."), name="Smith", pin="999").complete_substage(
        "grow_leaves.short_reporter_citations.discovery"
    )

    async def reviewer(context):
        return decision(context)

    after = asyncio.run(correct_leaf_fields(before, reviewer=reviewer))
    with pytest.raises(ValueError, match="already completed"):
        asyncio.run(correct_leaf_fields(after, reviewer=reviewer))
    data = json.loads(after.model_dump_json())
    ref = next(
        item
        for item in data["citations"][1]["leaf_field_correction_reviews"][0]["evidence_refs"]
        if item["history"] == "reporter_exact_lookup"
    )
    ref["selected_candidate_index"] = 9
    with pytest.raises(ValueError, match="missing selected candidate"):
        Document.model_validate(data)
    with pytest.raises(ValueError, match="before resolving"):
        asyncio.run(
            correct_leaf_fields(
                before.complete_substage("validate_pincite.citation_preparation.page_resolution"),
                reviewer=reviewer,
            )
        )


@pytest.mark.parametrize("ambiguous", [False, True])
@pytest.mark.parametrize("available", [False, True])
def test_linked_reporter_docket_response_reaches_leaf_reviewer(ambiguous, available):
    document = rooted("See Smith, 347 U.S. at 999.")
    root = document.roots[0]
    data = root.model_dump(mode="json")
    lookup = data["reporter_exact_lookup"]
    lookup["response"]["clusters"][0].update(docket_id="40", court_id=None)
    response = CourtListenerDocket(id="40", court_id="nysd") if available else None
    if ambiguous:
        lookup["outcome"] = ReporterExactLookupOutcome.AMBIGUOUS.value
        lookup["response"]["clusters"].append(CourtListenerCluster(id="43").model_dump(mode="json"))
        history = "reporter_exact_candidate_dockets"
        record = ReporterExactCandidateDocket(
            node_id=root.nodes[-1].id, docket_id="40", response=response, candidate_index=0
        )
        data[history] = [record.model_dump(mode="json")]
    else:
        history = "reporter_exact_docket"
        record = ReporterExactDocket(node_id=root.nodes[-1].id, docket_id="40", response=response)
        data[history] = record.model_dump(mode="json")
    saved = document.model_dump(mode="json")
    saved["citations"][0] = data
    document = Document.model_validate(saved)
    before = short(document, name="Smith", pin="999").complete_substage(
        "grow_leaves.short_reporter_citations.discovery"
    )
    before = Document.model_validate_json(before.model_dump_json())
    seen = []

    async def reviewer(context):
        evidence = next(
            item for item in context.validation_evidence if item["reference"]["history"] == history
        )
        seen.append(evidence)
        assert evidence["record"]["candidate_index"] == (0 if ambiguous else None)
        assert evidence["record"]["docket"] == (
            {"id": "40", "court": None, "court_id": "nysd"} if available else None
        )
        return decision(context)

    after = asyncio.run(correct_leaf_fields(before, reviewer=reviewer))
    assert len(seen) == 1
    review = after.leaves[0].leaf_field_correction_reviews[-1]
    assert review.decision is not None and review.failure_reason is None
    assert any(reference.history == history for reference in review.evidence_refs)
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_ivr_reviewer_cacheable_prefix_precedes_occurrence_specific_context(monkeypatch):
    document = rooted("See Smith, 347 U.S. at 999. See Smith, 347 U.S. at 998.")
    document = short(document, name="Smith", pin="999")
    document = short(
        document,
        quote="347 U.S. at 998",
        name="Smith",
        pin="998",
        citation_id="leaf2",
        after=document.text.index("See Smith", len(ROOT_TEXT) + 5),
    )
    before = document.complete_substage("grow_leaves.short_reporter_citations.discovery")
    specs = []

    async def fake_run(_session, spec, **_kwargs):
        specs.append(spec)
        fields = json.loads(spec.user_variables["windows"])
        answer = LeafFieldCorrectionDecision(
            fields=tuple(
                LeafFieldCorrectionProposal(
                    field=item["field"],
                    propose_replacement=False,
                    quote=None,
                    reason="Correctly extracted",
                )
                for item in fields
            ),
            reason="No change",
        )
        return trace(answer.model_dump_json())

    monkeypatch.setattr("mellea_lrc.extraction.leaf_field_corrections.reviewer.run_instruct_ivr", fake_run)
    service = IvrLeafFieldCorrectionReviewer(session=object(), model_options={}, max_attempts=3)
    after = asyncio.run(correct_leaf_fields(before, reviewer=service))
    assert len(specs) == 2
    assert specs[0].prefix == specs[1].prefix
    assert "Saved validation evidence for attached root root" in specs[0].prefix
    assert "999" not in specs[0].prefix
    assert '"found_pages"' not in specs[0].prefix
    assert specs[0].user_variables["citation_id"] != specs[1].user_variables["citation_id"]
    assert all(item.leaf_field_correction_reviews[-1].ivr.success for item in after.leaves)


def test_short_reporter_colocation_shares_its_own_preceding_name_window():
    document = rooted("See Smith, 347 U.S. at 999, 347 U.S. at 998.")
    document = short(document, citation_id="leaf1", pin="999")
    first = (
        document.short_reporters[0]
        .record("grow_leaves.short_reporter_citations.discovery")
        .with_colocation("local-group")
    )
    document = document.replace_citation(first)
    document = short(
        document,
        quote="347 U.S. at 998",
        citation_id="leaf2",
        after=document.text.index("999") + 3,
        pin="998",
    )
    second = (
        document.short_reporters[1]
        .record("grow_leaves.short_reporter_citations.discovery")
        .with_colocation("local-group")
    )
    before = document.replace_citation(second).complete_substage(
        "grow_leaves.short_reporter_citations.discovery"
    )

    async def reviewer(context):
        return decision(context, {"case_name": "Smith"})

    after = asyncio.run(correct_leaf_fields(before, reviewer=reviewer))
    assert all(item.case_name[-1].quote == "Smith" for item in after.short_reporters)
    assert all(item.leaf_field_correction_reviews[-1].decision is not None for item in after.short_reporters)


def test_correction_precision_reuses_field_targets_skips_failures_and_missing_gold(monkeypatch):
    from evaluations import grow_leaves as evaluation
    from evaluations.score_types import Precision

    before = short(rooted("See Smith, 347 U.S. at 999.")).complete_substage(
        "grow_leaves.short_reporter_citations.discovery"
    )

    async def no_change(context):
        return decision(context)

    after = asyncio.run(correct_leaf_fields(before, reviewer=no_change))
    leaf = after.short_reporters[0]
    row = {"pin_cite": {"source": {"kind": "not_stated"}, "normalization": {"kind": "unavailable"}}}
    monkeypatch.setattr(
        evaluation, "annotations_by_site", lambda _document: {evaluation.citation_site(leaf): row}
    )
    score = evaluation.score_leaf_field_corrections(after)
    assert score.metrics["case_name_span"] == Precision(0, 1)
    assert score.metrics["case_name_normalization"] == Precision(0, 1)
    assert score.metrics["pin_cite_span"] == Precision(1, 1)
    assert score.metrics["pin_cite_normalization"] == Precision(1, 1)

    async def failed(_context):
        return LeafFieldCorrectionOutcome(None, failure_reason="No accepted field decision")

    failed_document = asyncio.run(correct_leaf_fields(before, reviewer=failed))
    assert evaluation.score_leaf_field_corrections(failed_document).metrics == {}


def test_correction_precision_handles_inferred_court_without_a_source_span(monkeypatch):
    from evaluations import grow_leaves as evaluation
    from evaluations.score_types import Precision

    document = rooted("Smith v. Jones, 347 U.S. 483, 999 (1954).")
    leaf = (
        FullReporterCitation.from_locator(
            citation_id="repeat",
            substage="grow_leaves.short_reporter_citations.discovery",
            source=document.text,
            span=span(document.text, "347 U.S. 483", after=len(ROOT_TEXT)),
        )
        .record("grow_leaves.short_reporter_citations.discovery")
        .with_root("root")
        .with_inferred_court("scotus")
    )
    before = document.add_citation(leaf).complete_substage("grow_leaves.short_reporter_citations.discovery")

    async def no_change(context):
        return decision(context)

    after = asyncio.run(correct_leaf_fields(before, reviewer=no_change))
    row = {
        "court": {
            "source": {"kind": "quoted", "start": 0, "end": 1},
            "normalization": {
                "kind": "value",
                "value": leaf.court[-1].get_normalized().model_dump(mode="json"),
            },
        }
    }
    monkeypatch.setattr(
        evaluation, "annotations_by_site", lambda _document: {evaluation.citation_site(leaf): row}
    )
    score = evaluation.score_leaf_field_corrections(after)
    assert score.metrics["court_span"] == Precision(0, 1)
    assert score.metrics["court_normalization"] == Precision(1, 1)


def test_same_quote_replacement_records_grounding_without_duplicate_field_reading():
    before = short(rooted("See Smith, 347 U.S. at 999."), name="Smith", pin="999").complete_substage(
        "grow_leaves.short_reporter_citations.discovery"
    )

    async def reviewer(context):
        return decision(context, {"case_name": "Smith", "pin_cite": "999"})

    after = asyncio.run(correct_leaf_fields(before, reviewer=reviewer))
    leaf = after.short_reporters[0]
    assert leaf.case_name == before.short_reporters[0].case_name
    assert leaf.pin_cite == before.short_reporters[0].pin_cite
    assert all(not item.applied for item in leaf.leaf_field_correction_reviews[-1].grounded)


def test_retrieval_only_root_does_not_trigger_validation_assistance():
    document = rooted("See Smith, 347 U.S. at 999.").get_substage("11_reporter_lookup")
    before = short(document, name="Smith", pin="999").complete_substage(
        "grow_leaves.short_reporter_citations.discovery"
    )

    async def forbidden(_context):
        pytest.fail("Root lookup has not been reviewed")

    after = asyncio.run(correct_leaf_fields(before, reviewer=forbidden))
    assert after.citations == before.citations


def test_overextended_pin_is_corrected_from_its_own_source_and_old_reading_survives():
    document = rooted("See Smith, 347 U.S. at 999 (1954).")
    before = short(document, name="Smith", pin="999 (1954)").complete_substage(
        "grow_leaves.short_reporter_citations.discovery"
    )

    async def reviewer(context):
        return decision(context, {"pin_cite": "999"})

    after = asyncio.run(correct_leaf_fields(before, reviewer=reviewer))
    leaf = after.short_reporters[0]
    assert [item.quote for item in leaf.pin_cite] == ["999 (1954)", "999"]
    assert leaf.pin_cite[-1].normalizable
    assert leaf.short_locator == before.short_reporters[0].short_locator
    assert after.get_substage("grow_leaves.short_reporter_citations.discovery") == before


def test_colocated_neighbor_reporter_volume_is_outside_each_leaf_pin_window():
    document = rooted("See Smith, 347 U.S. at 999, 347 U.S. at 998.")
    document = short(document, citation_id="leaf1", pin="999")
    first = (
        document.short_reporters[0]
        .record("grow_leaves.short_reporter_citations.discovery")
        .with_colocation("local-group")
    )
    document = document.replace_citation(first)
    document = short(
        document,
        quote="347 U.S. at 998",
        citation_id="leaf2",
        after=document.text.index("999") + 3,
        pin="998",
    )
    second = (
        document.short_reporters[1]
        .record("grow_leaves.short_reporter_citations.discovery")
        .with_colocation("local-group")
    )
    before = document.replace_citation(second).complete_substage(
        "grow_leaves.short_reporter_citations.discovery"
    )

    async def reviewer(context):
        return decision(context, {"pin_cite": "347"})

    after = asyncio.run(correct_leaf_fields(before, reviewer=reviewer))
    assert all(item.leaf_field_correction_reviews[-1].failure_reason for item in after.short_reporters)
    assert [item.pin_cite for item in after.short_reporters] == [
        item.pin_cite for item in before.short_reporters
    ]


@pytest.mark.parametrize("kind", ["short", "id"])
@pytest.mark.parametrize(
    "marker, pin", [("at999", "999"), ("at\n999", "999"), ("at\n¶12", "¶12"), ("at*3", "*3")]
)
def test_shared_relaxed_at_marker_bounds_zero_space_and_newline_pin_readings(kind, marker, pin):
    quote = f"347 U.S. {marker}" if kind == "short" else f"Id. {marker}"
    document = rooted("See Smith, " + quote + ".")
    if kind == "short":
        document = short(document, quote=quote, name="Smith")
    else:
        leaf = (
            IdCitation.from_source(
                source=document.text,
                span=span(document.text, quote),
                substage="grow_leaves.short_reporter_citations.discovery",
            )
            .record("grow_leaves.short_reporter_citations.discovery")
            .with_root("root")
        )
        document = document.add_citation(leaf)
    before = document.complete_substage("grow_leaves.short_reporter_citations.discovery")

    async def reviewer(context):
        return decision(context, {"pin_cite": pin})

    after = asyncio.run(correct_leaf_fields(before, reviewer=reviewer))
    original, corrected = before.leaves[0], after.leaves[0]
    assert corrected.pin_cite[-1].quote == pin
    assert corrected.pin_cite[-1].normalizable
    assert corrected.leaf_field_correction_reviews[-1].decision is not None
    assert corrected.site_span == original.site_span
    field = "short_locator" if kind == "short" else "id_reference"
    assert getattr(corrected, field) == getattr(original, field)
