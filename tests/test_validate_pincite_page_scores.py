"""Explicit page correctness and recovered-page agreement have separate gold."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluations import validate_pincite as evaluation
from mellea_lrc.model import Document, Span
from mellea_lrc.model.citations.fields.pin_cite import PinCiteKind, PinCiteTarget
from mellea_lrc.model.citations.reporter_pinpoint import (
    OpinionEvidenceQuote,
    OpinionReviewScope,
    OpinionSupportResult,
    ReporterCitationSupportReview,
    ReporterOpinionEvidence,
    ReporterPinpointJudgment,
    ReporterPinpointVerdict,
    ReporterSupportDecision,
)
from tests.test_validate_pincite_semantic_scores import OPINION, _document

SUBSTAGES = (evaluation.FULL_OPINION_SUBSTAGE, evaluation.JUDGMENT_SUBSTAGE)


def _page(first, last=None, *, kind=PinCiteKind.PAGE, footnote=None):
    return PinCiteTarget(first=first, last=first if last is None else last, kind=kind, footnote=footnote)


def _finding(label="CORRECT_PINCITE", *, correct=True, pagination=True, found=()):
    return {
        "label": label,
        "pagination_available": pagination,
        "correct_page": correct,
        "found_pages": [target.model_dump(mode="json") for target in found],
    }


def _native_findings(document, **findings):
    source = Path(document.source_path)
    annotation = source.parent.parent / "documents" / f"{source.stem}.jsonl"
    rows = [json.loads(line) for line in annotation.read_text().splitlines()]
    for row in rows[1:]:
        if row["id"] in findings:
            row["validation"]["pincite"] = findings[row["id"]]
    annotation.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return document


def _begin(document, substage):
    if evaluation.PAGE_SUPPORT_SUBSTAGE not in document.substage_runs:
        document = document.complete_substage(evaluation.PAGE_SUPPORT_SUBSTAGE)
    if (
        substage == evaluation.JUDGMENT_SUBSTAGE
        and evaluation.FULL_OPINION_SUBSTAGE not in document.substage_runs
    ):
        document = document.complete_substage(evaluation.FULL_OPINION_SUBSTAGE)
    return document


def _prediction(
    document,
    substage,
    citation_id,
    correct,
    *,
    pagination=True,
    found=(),
    result=OpinionSupportResult.SUPPORTED,
    verdict=ReporterPinpointVerdict.CORRECT_PINCITE,
):
    citation = next(c for c in document.citations if c.id == citation_id).record(substage)
    if substage == evaluation.FULL_OPINION_SUBSTAGE:
        quotes = ()
        indices = ()
        if result in {OpinionSupportResult.SUPPORTED, OpinionSupportResult.CONTRADICTED}:
            root_id = citation.reporter_pinpoint_evidence[0].root_id
            citation = citation.with_reporter_opinion_evidence(
                ReporterOpinionEvidence(
                    node_id=citation.nodes[-1].id,
                    root_id=root_id,
                    opinion_id="20",
                    quote=OPINION,
                    span=Span(start=0, end=len(OPINION)),
                )
            )
            quotes = (OpinionEvidenceQuote(opinion_id="20", quote=OPINION),)
            indices = (len(citation.reporter_opinion_evidence) - 1,)
        citation = citation.with_reporter_support_review(
            ReporterCitationSupportReview(
                node_id=citation.nodes[-1].id,
                evidence_index=0,
                scope=OpinionReviewScope.FULL_OPINION,
                decision=ReporterSupportDecision(
                    result=result,
                    evidence=quotes,
                    pagination_available=pagination,
                    correct_page=correct,
                    found_pages=found,
                    reason="Evaluate the written target separately from attributed support.",
                ),
                opinion_evidence_indices=indices,
            )
        )
    else:
        citation = citation.with_reporter_pinpoint_judgment(
            ReporterPinpointJudgment(
                node_id=citation.nodes[-1].id,
                evidence_index=0,
                review_index=None,
                verdict=verdict,
                pagination_available=pagination,
                correct_page=correct,
                found_pages=found,
                reason="Page location is independent of the support verdict.",
            )
        )
    return document.replace_citation(citation)


def _score(document, substage):
    if substage == evaluation.FULL_OPINION_SUBSTAGE:
        return evaluation.score_reporter_citation_full_opinion_review(document)
    return evaluation.score_reporter_citation_pinpoint_judgment(document)


@pytest.mark.parametrize("substage", SUBSTAGES)
def test_page_target_agreement_is_independent_of_support_correctness(tmp_path, substage):
    document = _native_findings(
        _document(tmp_path),
        g0=_finding(),
        g3=_finding("WRONG_PINCITE", correct=False, found=(_page(561),)),
    )
    document = _begin(document, substage)
    document = _prediction(document, substage, "c0", True)
    document = _prediction(document, substage, "c3", False).complete_substage(substage)

    score = _score(document, substage)

    assert score.page_precision["total"].correct == 2
    assert score.page_precision["total"].predicted == 2
    assert score.judgments["total"].correct == 1  # The supported leaf disagrees with gold support.
    assert score.page_precision["reporter_roots"].correct == 1
    assert score.page_precision["reporter_leaves"].correct == 1


@pytest.mark.parametrize("substage", SUBSTAGES)
def test_content_approval_does_not_override_native_page_negative(tmp_path, substage):
    document = _native_findings(
        _document(tmp_path),
        g0=_finding(correct=False, found=(_page(571),)),
    )
    document = _prediction(_begin(document, substage), substage, "c0", True).complete_substage(substage)

    score = _score(document, substage)

    assert score.judgments["total"].correct == 1
    assert score.page_precision["total"].predicted == 1
    assert score.page_precision["total"].correct == 0


@pytest.mark.parametrize("substage", SUBSTAGES)
def test_unknown_native_placement_and_unlocated_predictions_are_separate(tmp_path, substage):
    document = _native_findings(
        _document(tmp_path),
        g0=_finding(correct=None, pagination=False),
        g3=_finding(),
    )
    document = _begin(document, substage)
    document = _prediction(document, substage, "c0", True)
    document = _prediction(document, substage, "c3", None).complete_substage(substage)

    precision = _score(document, substage).page_precision["total"]

    assert precision.correct == precision.predicted == 0
    assert precision.unlocated == 1
    assert precision.unscored == 1


@pytest.mark.parametrize("substage", SUBSTAGES)
def test_wrong_support_without_caveat_does_not_imply_wrong_page(tmp_path, substage):
    document = _prediction(_begin(_document(tmp_path), substage), substage, "c3", False).complete_substage(
        substage
    )

    precision = _score(document, substage).page_precision["total"]

    assert precision.correct == precision.predicted == precision.unlocated == 0
    assert precision.unscored == 1


def test_full_review_scores_page_booleans_independently_of_content_results(tmp_path):
    substage = evaluation.FULL_OPINION_SUBSTAGE
    document = _begin(_document(tmp_path), substage)
    document = _prediction(document, substage, "c0", False, result=OpinionSupportResult.CONTRADICTED)
    document = _prediction(document, substage, "c3", False, result=OpinionSupportResult.NOT_FOUND)
    document = _prediction(document, substage, "c2", None, result=OpinionSupportResult.UNAVAILABLE)
    precision = _score(document.complete_substage(substage), substage).page_precision["total"]

    assert precision.correct == precision.unlocated == 0
    assert precision.predicted == 1 and precision.unscored == 2


def test_final_scores_page_booleans_independently_of_content_verdicts(tmp_path):
    substage = evaluation.JUDGMENT_SUBSTAGE
    document = _begin(_document(tmp_path), substage)
    document = _prediction(document, substage, "c0", False, verdict=ReporterPinpointVerdict.WRONG_PINCITE)
    document = _prediction(document, substage, "c3", None, verdict=ReporterPinpointVerdict.UNDETERMINED)
    precision = _score(document.complete_substage(substage), substage).page_precision["total"]

    assert precision.correct == precision.unlocated == 0
    assert precision.predicted == precision.unscored == 1


@pytest.mark.parametrize("substage", SUBSTAGES)
def test_gold_root_identity_excludes_family_and_skipped_or_unannotated_predictions(tmp_path, substage):
    document = _begin(_document(tmp_path, wrong_root_identity=True), substage)
    for citation_id in ("c0", "c3", "c4", "c5"):
        document = _prediction(document, substage, citation_id, True)
    precision = _score(document.complete_substage(substage), substage).page_precision["total"]

    assert precision.correct == precision.predicted == precision.unlocated == 0
    assert precision.unscored == 4


@pytest.mark.parametrize("substage", SUBSTAGES)
def test_wrong_root_attachment_cannot_receive_page_credit(tmp_path, substage):
    document = _native_findings(_document(tmp_path), g3=_finding())
    document = _begin(document, substage)
    citation = next(c for c in document.citations if c.id == "c3").record(substage).with_root("c2")
    document = document.replace_citation(citation)
    document = _prediction(document, substage, "c3", True).complete_substage(substage)

    precision = _score(document, substage).page_precision["total"]

    assert precision.predicted == 1
    assert precision.correct == 0


def test_page_scores_roundtrip_aggregate_and_preserve_stage_checkpoints(tmp_path):
    full_stage, final_stage = SUBSTAGES
    document = _native_findings(_document(tmp_path), g0=_finding(correct=False, found=(_page(571),)))
    document = _begin(document, full_stage)
    document = _prediction(document, full_stage, "c0", False, found=(_page(571),)).complete_substage(
        full_stage
    )
    full_score = _score(document, full_stage)
    document = _prediction(document, final_stage, "c0", True, found=(_page(570),)).complete_substage(
        final_stage
    )
    restored = Document.model_validate_json(document.model_dump_json())

    assert _score(restored, full_stage) == full_score
    final_score = _score(restored, final_stage)
    assert full_score.page_precision["total"].correct == 1
    assert final_score.page_precision["total"].correct == 0
    assert final_score.as_dict()["page_precision"]["total"]["predicted"] == 1
    assert full_score.found_page_precision["total"].correct == 1
    assert final_score.found_page_precision["total"].correct == 0
    assert final_score.found_page_precision["total"].predicted == 1
    doubled = full_score + full_score
    assert doubled.page_precision["total"].correct == doubled.page_precision["total"].predicted == 2
    assert doubled.as_dict()["page_precision"]["total"]["precision"] == 1
    assert (
        doubled.found_page_precision["total"].correct == doubled.found_page_precision["total"].predicted == 2
    )
    assert doubled.as_dict()["found_page_precision"]["total"]["reference_agreement"] == 1

    workflow = evaluation.score_validate_pincite(restored)
    assert workflow.page_precision == final_score.page_precision
    assert workflow.found_page_precision == final_score.found_page_precision
    assert (workflow + workflow).found_page_precision["total"].predicted == 2
    assert workflow.as_dict()["page_precision"]["total"]["correct"] == 0
    assert "Page precision" in evaluation.render_reporter_citation_full_opinion_review(full_score)
    assert "Page precision" in evaluation.render_reporter_citation_pinpoint_judgment(final_score)
    assert "Page precision" in evaluation.render_validate_pincite(workflow)
    for report in (
        evaluation.render_reporter_citation_full_opinion_review(full_score),
        evaluation.render_reporter_citation_pinpoint_judgment(final_score),
        evaluation.render_validate_pincite(workflow),
    ):
        assert "Found-page agreement" in report
        assert "positive-reference agreement, not exhaustive factual precision" in report
        assert "Unlisted or partly uncovered alternatives remain unscored" in report
    page_score = evaluation.score_reporter_citation_pinpoint_page_review(restored)
    assert "page_precision" not in page_score.as_dict()
    assert "found_page_precision" not in page_score.as_dict()


@pytest.mark.parametrize("result", list(OpinionSupportResult))
def test_full_review_scores_explicit_page_boolean_for_every_content_result(tmp_path, result):
    substage = evaluation.FULL_OPINION_SUBSTAGE
    document = _prediction(
        _begin(_document(tmp_path), substage), substage, "c0", True, result=result
    ).complete_substage(substage)
    precision = _score(document, substage).page_precision["total"]
    assert precision.correct == precision.predicted == 1


@pytest.mark.parametrize("verdict", list(ReporterPinpointVerdict))
def test_final_scores_explicit_page_boolean_for_every_content_verdict(tmp_path, verdict):
    substage = evaluation.JUDGMENT_SUBSTAGE
    document = _prediction(
        _begin(_document(tmp_path), substage), substage, "c0", True, verdict=verdict
    ).complete_substage(substage)
    precision = _score(document, substage).page_precision["total"]
    assert precision.correct == precision.predicted == 1


@pytest.mark.parametrize("substage", SUBSTAGES)
@pytest.mark.parametrize(
    "native",
    [
        {"label": "CORRECT_PINCITE"},
        {"label": "WRONG_PINCITE"},
        _finding(correct="false"),
        _finding(correct=0),
    ],
)
def test_no_explicit_native_boolean_means_unscored_not_an_inferred_negative(tmp_path, substage, native):
    document = _native_findings(_document(tmp_path), g0=native)
    document = _prediction(_begin(document, substage), substage, "c0", False).complete_substage(substage)
    precision = _score(document, substage).page_precision["total"]
    assert precision.correct == precision.predicted == precision.unlocated == 0
    assert precision.unscored == 1


@pytest.mark.parametrize("substage", SUBSTAGES)
@pytest.mark.parametrize("found", [(_page(571),), (_page(571, 572),), (_page(571), _page(572))])
def test_found_pages_agree_with_positive_typed_ranges_without_provider_id_matching(tmp_path, substage, found):
    document = _native_findings(_document(tmp_path), g0=_finding(correct=False, found=(_page(571, 572),)))
    document = _prediction(_begin(document, substage), substage, "c0", False, found=found).complete_substage(
        substage
    )
    score = _score(document, substage)
    assert score.page_precision["total"].correct == 1
    agreement = score.found_page_precision["total"]
    assert agreement.correct == agreement.predicted == 1
    # Native source CAP ID9999 differs from the saved CourtListener evidence ID20.
    assert agreement.as_dict()["reference_agreement"] == 1
    assert "precision" not in agreement.as_dict()


@pytest.mark.parametrize("substage", SUBSTAGES)
@pytest.mark.parametrize(
    "found",
    [
        (_page(573),),
        (_page(571, 573),),
        (_page(571), _page(573)),
        (_page(571, kind=PinCiteKind.PARAGRAPH),),
        (_page(571, footnote="2"),),
    ],
)
def test_unlisted_partial_or_different_kind_alternatives_are_unscored(tmp_path, substage, found):
    document = _native_findings(_document(tmp_path), g0=_finding(correct=False, found=(_page(571, 572),)))
    document = _prediction(_begin(document, substage), substage, "c0", False, found=found).complete_substage(
        substage
    )
    agreement = _score(document, substage).found_page_precision["total"]
    assert agreement.correct == agreement.predicted == agreement.unlocated == 0
    assert agreement.unscored == 1


@pytest.mark.parametrize("substage", SUBSTAGES)
def test_found_footnote_and_star_targets_preserve_typed_agreement(tmp_path, substage):
    references = (_page(8, 9, kind=PinCiteKind.STAR, footnote="3"),)
    document = _native_findings(_document(tmp_path), g0=_finding(correct=False, found=references))
    document = _prediction(
        _begin(document, substage), substage, "c0", False, found=references
    ).complete_substage(substage)
    agreement = _score(document, substage).found_page_precision["total"]
    assert agreement.correct == agreement.predicted == 1


@pytest.mark.parametrize("substage", SUBSTAGES)
def test_native_page_negative_can_rule_out_a_recovered_written_target(tmp_path, substage):
    document = _native_findings(_document(tmp_path), g0=_finding(correct=False, found=(_page(571),)))
    document = _prediction(
        _begin(document, substage), substage, "c0", False, found=(_page(570),)
    ).complete_substage(substage)
    score = _score(document, substage)
    assert score.page_precision["total"].correct == 1
    assert score.found_page_precision["total"].predicted == 1
    assert score.found_page_precision["total"].correct == 0


@pytest.mark.parametrize("substage", SUBSTAGES)
def test_empty_found_pages_and_unknown_positive_references_are_separate(tmp_path, substage):
    document = _native_findings(
        _document(tmp_path), g0=_finding(found=(_page(570),)), g3=_finding(correct=None, pagination=False)
    )
    document = _prediction(_begin(document, substage), substage, "c0", None, pagination=False)
    document = _prediction(document, substage, "c3", False, found=(_page(561),)).complete_substage(substage)
    agreement = _score(document, substage).found_page_precision["total"]
    assert agreement.correct == agreement.predicted == 0
    assert agreement.unlocated == agreement.unscored == 1


@pytest.mark.parametrize("substage", SUBSTAGES)
def test_unnormalizable_written_target_does_not_abort_positive_found_page_agreement(tmp_path, substage):
    document = _native_findings(_document(tmp_path), g0=_finding(correct=False, found=(_page(571),)))
    root = document.roots[0].record("test_failed_pin_reading")
    failed = root.pin_cite[-1].model_copy(
        update={
            "node_id": root.nodes[-1].id,
            "normalizable": False,
            "unchecked_normalized": None,
            "normalization_error": "Earlier reader failed",
        }
    )
    document = document.replace_citation(
        root.model_copy(update={"pin_cite": (*root.pin_cite, failed)})
    ).complete_substage("test_failed_pin_reading")
    document = _prediction(
        _begin(document, substage), substage, "c0", False, found=(_page(571),)
    ).complete_substage(substage)
    agreement = _score(document, substage).found_page_precision["total"]
    assert agreement.correct == agreement.predicted == 1
