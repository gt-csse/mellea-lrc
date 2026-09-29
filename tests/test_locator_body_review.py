"""Locator occurrences receive one context-aware review across saved providers."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluations import validate_roots as evaluation
from mellea_lrc.model.citations import FullDocketCitation, FullReporterCitation
from mellea_lrc.model.citations.body_evidence import (
    BodyCitationTreatment,
    BodyCorroborationDecision,
    BodySearch,
    BodySource,
)
from mellea_lrc.model.citations.judgments import IdentityBasis, IdentityVerdict
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.validation.body_search.common import make_body_evidences
from mellea_lrc.validation.locator_body_review import STAGE, review_locator_body_evidence
from mellea_lrc.validation.locator_body_review.reviewer import BodyCorroborationContext, _diverse_evidence

SOURCE = "Smith v. Jones, No. 05-4206 (2d Cir. 2007)."
BODY = "The court discussed Smith v. Jones, No. 05-4206 (2d Cir. 2007), in its analysis."
STAGES = (
    (BodySource.COURTLISTENER_OPINION, "20_courtlistener_opinion_locator_body_search"),
    (BodySource.COURTLISTENER_RECAP, "21_courtlistener_recap_locator_body_search"),
    (BodySource.GOVINFO_OPINION, "22_govinfo_opinion_locator_body_search"),
)


def _document(
    *,
    include_evidence: bool = True,
    body: str = BODY,
    source_input: Path | str = SOURCE,
    include_validation_history: bool = False,
) -> Document:
    number_start = SOURCE.index("05-4206")
    root = FullDocketCitation.from_locator(
        citation_id="docket:0",
        stage="sites",
        source=SOURCE,
        span=Span(SOURCE.index("No."), number_start + len("05-4206")),
        number_span=Span(number_start, number_start + len("05-4206")),
    )
    document = Document.from_source(source_input).add_citation(root).complete("sites")
    root = root.record("fields")
    root = root.with_case_name(SOURCE, Span(0, len("Smith v. Jones")))
    root = root.with_court(SOURCE, Span(SOURCE.index("2d Cir."), SOURCE.index("2d Cir.") + 7))
    root = root.with_date(SOURCE, Span(SOURCE.index("2007"), SOURCE.index("2007") + 4))
    document = document.replace_citation(root).complete("fields")
    document = document.replace_citation(root.record("roots").with_root(root.id)).complete("roots")
    if include_validation_history:
        for stage in evaluation.WORKFLOW_STAGES:
            document = document.complete(stage)
    for source, stage in STAGES:
        root = document.roots[0].record(stage)
        evidence = (
            make_body_evidences(
                body_id=f"{source.value}:1",
                parent_id=None,
                url="https://example.test/source",
                issued_on=date(2006, 1, 1),
                date_basis="opinion.date_filed",
                metadata={"id": 1},
                body_text=body,
                locator="05-4206",
                source_text=SOURCE,
            )
            if include_evidence and source is BodySource.GOVINFO_OPINION
            else ()
        )
        root = root.with_body_search(
            BodySearch(node_id=root.nodes[-1].id, source=source, retrospective_date=None, evidence=evidence)
        )
        document = document.replace_citation(root).complete(stage)
    return document


def _annotated_source(
    tmp_path: Path,
    *,
    source: str,
    locator: str,
    kind: str,
    labels: dict[str, str],
) -> Path:
    dataset = tmp_path / "primary"
    source_dir = dataset / "documents_txt"
    annotation_dir = dataset / "documents"
    source_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    source_path = source_dir / "example.txt"
    source_path.write_text(source, encoding="utf-8")
    start = source.index(locator)
    rows = [
        {
            "unit": "header",
            "dataset": "primary",
            "document": source_path.name,
            "text": {
                "path": "primary/documents_txt/example.txt",
                "length": len(source),
                "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            },
        },
        {
            "unit": "citation",
            "id": "example-o01",
            "is_root": True,
            "kind": kind,
            "locator": {"source": {"kind": "quoted", "start": start, "end": start + len(locator)}},
            "validation": {
                "identity": {
                    "label": "WRONG_IDENTITY" if "disagrees" in labels.values() else "CORRECT_IDENTITY",
                    "fields": {field: {"label": label} for field, label in labels.items()},
                }
            },
        },
    ]
    (annotation_dir / "example.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return source_path


def test_body_review_keeps_field_scores_and_marks_final_checkpoint(tmp_path: Path) -> None:
    source_path = _annotated_source(
        tmp_path,
        source=SOURCE,
        locator="No. 05-4206",
        kind="DocketCitation",
        labels={"case_name": "agrees", "court": "disagrees", "date": "agrees"},
    )
    ready = _document(source_input=source_path, include_validation_history=True)
    prior = evaluation.score_validate_roots(ready.get_stage(evaluation.WORKFLOW_STAGES[-1]))
    assert all(score == evaluation.FieldScore(0, 0, 1) for score in prior.fields.values())

    reviewed = asyncio.run(
        review_locator_body_evidence(
            ready,
            reviewer=FakeReviewer(_decision(court_result="mismatch", verdict=IdentityVerdict.WRONG_IDENTITY)),
        )
    )
    score = evaluation.score_validate_roots(Document.model_validate_json(reviewed.model_dump_json()))
    assert reviewed.roots[0].identity_judgments[-1].verdict is IdentityVerdict.WRONG_IDENTITY
    assert score.stages == prior.stages
    assert score.fields == prior.fields
    assert score.body_review is not None
    assert score.body_review.verdict_counts == {"wrong_identity": 1}
    assert score.identity == evaluation.IdentityScore(1, 1, 1, 0)
    assert score.identity_with_partial == score.identity
    assert score.checkpoint == evaluation.LOCATOR_BODY_REVIEW
    assert score.as_dict()["checkpoint"] == evaluation.LOCATOR_BODY_REVIEW
    report = evaluation.render_validate_roots(score)
    assert "Checkpoint: 23_locator_body_review completed" in report
    assert "## 23_locator_body_review" in report
    assert "| wrong_identity | 1 |" in report
    assert "## Root identity after locator-body review" in report
    assert "printed citation comparisons have no corresponding field identity gold" in report

    declined = asyncio.run(
        review_locator_body_evidence(
            _document(source_input=source_path, include_evidence=False, include_validation_history=True)
        )
    )
    assert evaluation.score_validate_roots(declined).fields == prior.fields
    assert evaluation.score_validate_roots(declined).body_review.verdict_counts == {}


def test_body_identity_score_counts_an_incorrect_third_party_verdict(tmp_path: Path) -> None:
    source_path = _annotated_source(
        tmp_path,
        source=SOURCE,
        locator="No. 05-4206",
        kind="DocketCitation",
        labels={"case_name": "agrees", "court": "agrees", "date": "agrees"},
    )
    ready = _document(source_input=source_path, include_validation_history=True)
    reviewed = asyncio.run(
        review_locator_body_evidence(
            ready,
            reviewer=FakeReviewer(_decision(court_result="mismatch", verdict=IdentityVerdict.WRONG_IDENTITY)),
        )
    )
    score = evaluation.score_validate_roots(reviewed)
    assert score.body_review is not None
    assert score.body_review.verdict_counts == {"wrong_identity": 1}
    assert score.identity == evaluation.IdentityScore(0, 1, 1, 0)
    assert score.identity_with_partial == score.identity


def test_disputed_third_party_quote_does_not_become_field_identity_gold(tmp_path: Path) -> None:
    source_path = _annotated_source(
        tmp_path,
        source=SOURCE,
        locator="No. 05-4206",
        kind="DocketCitation",
        labels={"case_name": "disagrees", "court": "disagrees", "date": "disagrees"},
    )
    body = "This citation is fabricated: Smith v. Jones, No. 05-4206 (2d Cir. 2007)."
    ready = _document(source_input=source_path, body=body, include_validation_history=True)
    decision = _decision(
        treatment=BodyCitationTreatment.EXPLICITLY_DISPUTES,
        context_quote="This citation is fabricated",
        verdict=IdentityVerdict.WRONG_IDENTITY,
    )
    reviewed = asyncio.run(review_locator_body_evidence(ready, reviewer=FakeReviewer(decision)))
    assert reviewed.roots[0].identity_judgments[-1].verdict is IdentityVerdict.WRONG_IDENTITY
    assert (
        evaluation.score_validate_roots(reviewed).stages
        == evaluation.score_validate_roots(ready.get_stage(evaluation.WORKFLOW_STAGES[-1])).stages
    )
    assert evaluation.score_validate_roots(reviewed).fields == {
        field: evaluation.FieldScore(0, 0, 1) for field in evaluation.FIELDS
    }


def _decision(
    *,
    source: BodySource = BodySource.GOVINFO_OPINION,
    case_name_result: str = "match",
    court_result: str = "match",
    date_result: str = "match",
    quote: str | None = None,
    third_party_locator: str = "05-4206",
    third_party_case_name: str = "Smith v. Jones",
    third_party_date: str = "2007",
    treatment: BodyCitationTreatment = BodyCitationTreatment.CITES_AS_AUTHORITY,
    context_quote: str | None = None,
    verdict: IdentityVerdict = IdentityVerdict.CORRECT_IDENTITY,
) -> BodyCorroborationDecision:
    return BodyCorroborationDecision.model_validate(
        {
            "source": source.value,
            "evidence_index": 0,
            "citation_quote": quote or "Smith v. Jones, No. 05-4206 (2d Cir. 2007)",
            "treatment": treatment.value,
            "context_quote": context_quote,
            "filing": {
                "locator": "05-4206",
                "case_name": "Smith v. Jones",
                "normalized_case_name": {
                    "kind": "adversarial",
                    "plaintiff": "Smith",
                    "defendant": "Jones",
                    "subject": None,
                },
                "court": "2d Cir.",
                "date": "2007",
            },
            "third_party": {
                "locator": third_party_locator,
                "case_name": third_party_case_name,
                "court": "2d Cir.",
                "date": third_party_date,
            },
            "comparisons": {
                field: {
                    "result": (
                        case_name_result
                        if field == "case_name"
                        else court_result
                        if field == "court"
                        else date_result
                        if field == "date"
                        else "match"
                    ),
                    "reason": "Compared both citations.",
                }
                for field in ("locator", "case_name", "court", "date")
            },
            "identity_verdict": verdict.value,
            "reason": "The third-party filing cites the same docket and named case.",
        }
    )


def _declined_decision() -> BodyCorroborationDecision:
    return BodyCorroborationDecision(
        source=None,
        evidence_index=None,
        citation_quote=None,
        context_quote=None,
        treatment=None,
        filing=None,
        third_party=None,
        comparisons=None,
        identity_verdict=None,
        reason="No excerpt contains a reviewable citation at the locator.",
    )


class FakeReviewer:
    def __init__(self, decision: BodyCorroborationDecision) -> None:
        self.decision = decision
        self.contexts: list[object] = []

    async def __call__(self, context: object) -> BodyCorroborationDecision:
        self.contexts.append(context)
        return self.decision


def _assert_routed_without_judgment(reviewed: Document) -> None:
    root = reviewed.roots[0]
    assert root.identity_judgments == ()
    assert root.next_stage == "case_name_body_discovery"
    assert root.body_reviews[-1].decision is None or root.body_reviews[-1].decision.source is None
    assert Document.model_validate_json(reviewed.model_dump_json()) == reviewed


def test_one_judgment_can_select_govinfo_after_other_provider_searches() -> None:
    assert STAGE == "23_locator_body_review"
    document = _document()
    reviewer = FakeReviewer(_decision())
    reviewed = asyncio.run(review_locator_body_evidence(document, reviewer=reviewer))
    root = reviewed.roots[0]
    assert len(reviewer.contexts) == 1
    assert len(root.body_searches) == 3
    assert root.body_reviews[-1].decision.source is BodySource.GOVINFO_OPINION
    assert root.body_reviews[-1].grounded_quote == "Smith v. Jones, No. 05-4206 (2d Cir. 2007)"
    assert root.identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
    assert root.identity_judgments[-1].basis is IdentityBasis.THIRD_PARTY
    assert reviewed.get_stage(STAGES[-1][1]) == document
    assert Document.model_validate_json(reviewed.model_dump_json()) == reviewed


def test_model_can_issue_wrong_identity_after_field_mismatch() -> None:
    reviewed = asyncio.run(
        review_locator_body_evidence(
            _document(),
            reviewer=FakeReviewer(_decision(court_result="mismatch", verdict=IdentityVerdict.WRONG_IDENTITY)),
        )
    )
    assert reviewed.roots[0].identity_judgments[-1].verdict is IdentityVerdict.WRONG_IDENTITY
    assert reviewed.roots[0].body_reviews[-1].decision.comparisons.court.result.value == "mismatch"


def test_docket_case_can_be_supported_without_admitting_a_different_order_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = "The court discussed Smith v. Jones, No. 05-4206 (2d Cir. 2008), in its analysis."
    source_path = _annotated_source(
        tmp_path,
        source=SOURCE,
        locator="No. 05-4206",
        kind="DocketCitation",
        labels={"case_name": "agrees", "court": "agrees", "date": "agrees"},
    )
    decision = _decision(
        quote="Smith v. Jones, No. 05-4206 (2d Cir. 2008)",
        third_party_date="2008",
        date_result="mismatch",
        verdict=IdentityVerdict.PARTIALLY_CORROBORATED,
    )
    reviewed = asyncio.run(
        review_locator_body_evidence(
            _document(body=body, source_input=source_path, include_validation_history=True),
            reviewer=FakeReviewer(decision),
        )
    )
    root = reviewed.roots[0]
    assert root.identity_judgments[-1].verdict is IdentityVerdict.PARTIALLY_CORROBORATED
    assert root.next_stage == "case_name_body_discovery"
    assert root.body_reviews[-1].decision.comparisons.date.result.value == "mismatch"
    assert evaluation.score_locator_body_review(reviewed).verdict_counts == {
        "partially_corroborated": 1
    }
    # A qualified body judgment must retract even a prior full-citation admission.
    monkeypatch.setattr(evaluation, "_final_field_label", lambda *_: "agrees")
    assert evaluation.score_validate_roots(
        reviewed.get_stage(evaluation.WORKFLOW_STAGES[-1])
    ).identity == evaluation.IdentityScore(1, 1, 1, 0)
    body_score = evaluation.score_validate_roots(reviewed)
    assert body_score.identity == evaluation.IdentityScore(0, 0, 1, 1)
    assert body_score.identity_with_partial == evaluation.IdentityScore(1, 1, 1, 0)
    assert "| partially_corroborated | 1 |" in evaluation.render_validate_roots(body_score)
    assert Document.model_validate_json(reviewed.model_dump_json()) == reviewed


def test_qualified_docket_verdict_cannot_be_used_for_reporter_root() -> None:
    source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."
    locator = "550 U.S. 544"
    root = FullReporterCitation.from_locator(
        citation_id="reporter:0",
        stage="sites",
        source=source,
        span=Span(source.index(locator), source.index(locator) + len(locator)),
    )
    context = BodyCorroborationContext.from_document(
        Document.from_source(source).add_citation(root).complete("sites"), root
    )
    decision = _decision(verdict=IdentityVerdict.PARTIALLY_CORROBORATED)
    assert "only available for docket" in context.validation_error(decision)


def test_unqualified_admission_rejects_mismatch_or_mention() -> None:
    with pytest.raises(ValidationError, match="conflicting printed field"):
        _decision(court_result="mismatch")
    with pytest.raises(ValidationError, match="mere mention"):
        _decision(treatment=BodyCitationTreatment.MENTIONS_ONLY)
    with pytest.raises(ValidationError, match="field conflict or explicit challenge"):
        _decision(
            treatment=BodyCitationTreatment.MENTIONS_ONLY,
            verdict=IdentityVerdict.WRONG_IDENTITY,
        )


def test_docket_number_alone_cannot_confirm_a_full_citation() -> None:
    candidate = _decision().model_dump(mode="python")
    for side in ("filing", "third_party"):
        for field in ("case_name", "court", "date"):
            candidate[side][field] = None
    candidate["filing"]["normalized_case_name"] = None
    for field in ("case_name", "court", "date"):
        candidate["comparisons"][field]["result"] = "unavailable"
    with pytest.raises(ValidationError, match="locator and case-name support"):
        BodyCorroborationDecision.model_validate(candidate)


def test_docket_decision_date_must_be_supported_for_unqualified_admission() -> None:
    decision = _decision()
    changed = decision.model_copy(
        update={
            "comparisons": decision.comparisons.model_copy(
                update={
                    "date": decision.comparisons.date.model_copy(
                        update={"result": evaluation.MatchResult.UNAVAILABLE}
                    )
                }
            )
        }
    )
    context = BodyCorroborationContext.from_document(_document(), _document().roots[0])
    assert "date is not corroborated" in context.validation_error(changed)


def test_incorrect_model_match_label_cannot_hide_different_written_years() -> None:
    body = "The court discussed Smith v. Jones, No. 05-4206 (2d Cir. 2008), in its analysis."
    decision = _decision(
        quote="Smith v. Jones, No. 05-4206 (2d Cir. 2008)",
        third_party_date="2008",
        date_result="match",
        verdict=IdentityVerdict.CORRECT_IDENTITY,
    )
    reviewed = asyncio.run(
        review_locator_body_evidence(_document(body=body), reviewer=FakeReviewer(decision))
    )
    root = reviewed.roots[0]
    _assert_routed_without_judgment(reviewed)
    assert "written decision years differ" in root.body_reviews[-1].failure_reason


def test_uniform_grounding_threshold_rejects_excess_ocr_contamination() -> None:
    body = (
        "The court discussed RCHFU, LLC v. Marriott Vacations\n14 Worldwide Corp., No. 05-4206 "
        "(2d Cir. 2007), in its analysis."
    )
    decision = _decision(
        quote="RCHFU, LLC v. Marriott Vacations Worldwide Corp., No. 05-4206 (2d Cir. 2007)",
        third_party_case_name="RCHFU, LLC v. Marriott Vacations Worldwide Corp.",
        case_name_result="mismatch",
        verdict=IdentityVerdict.WRONG_IDENTITY,
    )
    reviewed = asyncio.run(
        review_locator_body_evidence(_document(body=body), reviewer=FakeReviewer(decision))
    )
    _assert_routed_without_judgment(reviewed)
    assert "98% similarity" in reviewed.roots[0].body_reviews[-1].failure_reason


def test_name_only_body_cannot_substantiate_the_locator_review() -> None:
    body = "A later brief cited Smith v. Jones, No. 05-CV-4206 (2d Cir. 2007)."
    document = _document(body=body)
    reviewer = FakeReviewer(_decision())
    reviewed = asyncio.run(review_locator_body_evidence(document, reviewer=reviewer))
    assert reviewer.contexts == []
    _assert_routed_without_judgment(reviewed)


def test_explicitly_disputed_locator_produces_negative_identity_with_matching_fields() -> None:
    challenge = "The court found that this citation was fabricated."
    document = _document(body=f"{BODY} {challenge}")
    decision = _decision(
        treatment=BodyCitationTreatment.EXPLICITLY_DISPUTES,
        context_quote=challenge,
        verdict=IdentityVerdict.WRONG_IDENTITY,
    )
    reviewed = asyncio.run(review_locator_body_evidence(document, reviewer=FakeReviewer(decision)))
    root = reviewed.roots[0]
    assert root.body_reviews[-1].decision.treatment is BodyCitationTreatment.EXPLICITLY_DISPUTES
    assert root.body_reviews[-1].grounded_context == challenge
    assert root.body_reviews[-1].context_span is not None
    assert root.identity_judgments[-1].verdict is IdentityVerdict.WRONG_IDENTITY
    assert root.body_reviews[-1].decision.comparisons.case_name.result.value == "match"
    assert reviewed.get_stage(STAGES[-1][1]) == document
    assert Document.model_validate_json(reviewed.model_dump_json()) == reviewed


def test_disputed_treatment_requires_a_grounded_context_quote() -> None:
    document = _document(body=f"{BODY} The court found that this citation was fabricated.")
    decision = _decision(
        treatment=BodyCitationTreatment.EXPLICITLY_DISPUTES,
        context_quote="The court found this citation was verified.",
        verdict=IdentityVerdict.WRONG_IDENTITY,
    )
    reviewed = asyncio.run(review_locator_body_evidence(document, reviewer=FakeReviewer(decision)))
    assert reviewed.roots[0].body_reviews[-1].decision is None
    _assert_routed_without_judgment(reviewed)


def test_saved_context_span_must_still_match_the_fetched_excerpt() -> None:
    challenge = "The court found that this citation was fabricated."
    document = _document(body=f"{BODY} {challenge}")
    reviewed = asyncio.run(
        review_locator_body_evidence(
            document,
            reviewer=FakeReviewer(
                _decision(
                    treatment=BodyCitationTreatment.EXPLICITLY_DISPUTES,
                    context_quote=challenge,
                    verdict=IdentityVerdict.WRONG_IDENTITY,
                )
            ),
        )
    )
    saved = reviewed.model_dump(mode="json")
    saved["citations"][0]["body_reviews"][-1]["context_span"] = {"start": 0, "end": len(challenge)}
    with pytest.raises(ValidationError, match="context must match"):
        Document.model_validate(saved)


def test_merely_mentioned_locator_records_uncertainty_and_routes_to_discovery() -> None:
    reviewed = asyncio.run(
        review_locator_body_evidence(
            _document(),
            reviewer=FakeReviewer(
                _decision(
                    treatment=BodyCitationTreatment.MENTIONS_ONLY,
                    verdict=IdentityVerdict.UNDETERMINED,
                )
            ),
        )
    )
    root = reviewed.roots[0]
    assert root.identity_judgments[-1].verdict is IdentityVerdict.UNDETERMINED
    assert root.next_stage == "case_name_body_discovery"
    assert root.body_reviews[-1].decision.identity_verdict is IdentityVerdict.UNDETERMINED
    assert Document.model_validate_json(reviewed.model_dump_json()) == reviewed


def test_ungrounded_quote_cannot_create_a_verdict() -> None:
    decision = _decision(quote="Smith v. Jones")
    reviewed = asyncio.run(review_locator_body_evidence(_document(), reviewer=FakeReviewer(decision)))
    root = reviewed.roots[0]
    assert root.body_reviews[-1].decision is None
    assert "anchor" in root.body_reviews[-1].failure_reason
    _assert_routed_without_judgment(reviewed)


def test_unquoted_third_party_field_cannot_create_a_verdict() -> None:
    decision = _decision()
    fabricated = decision.model_copy(
        update={"third_party": decision.third_party.model_copy(update={"case_name": "Another v. Case"})}
    )
    reviewed = asyncio.run(review_locator_body_evidence(_document(), reviewer=FakeReviewer(fabricated)))
    _assert_routed_without_judgment(reviewed)
    assert "case_name" in reviewed.roots[0].body_reviews[-1].failure_reason


def test_docket_number_digit_must_match_the_grounded_citation() -> None:
    decision = _decision(third_party_locator="05-4208")
    reviewed = asyncio.run(review_locator_body_evidence(_document(), reviewer=FakeReviewer(decision)))
    _assert_routed_without_judgment(reviewed)
    assert "must occur inside" in reviewed.roots[0].body_reviews[-1].failure_reason


@pytest.mark.parametrize("field,replacement", [("court", "3d Cir."), ("date", "2008")])
def test_other_numeric_field_must_match_the_grounded_citation(field: str, replacement: str) -> None:
    decision = _decision()
    fabricated = decision.model_copy(
        update={"third_party": decision.third_party.model_copy(update={field: replacement})}
    )
    reviewed = asyncio.run(review_locator_body_evidence(_document(), reviewer=FakeReviewer(fabricated)))
    _assert_routed_without_judgment(reviewed)
    assert field in reviewed.roots[0].body_reviews[-1].failure_reason


def test_no_fetched_body_skips_model_and_routes_without_identity_judgment() -> None:
    document = _document(include_evidence=False)
    reviewed = asyncio.run(review_locator_body_evidence(document))
    root = reviewed.roots[0]
    assert root.body_reviews[-1].decision.source is None
    assert root.body_reviews[-1].decision.identity_verdict is None
    _assert_routed_without_judgment(reviewed)
    assert json.loads(reviewed.model_dump_json())["stage_runs"][-1] == STAGE
    with pytest.raises(ValueError, match="already completed"):
        asyncio.run(review_locator_body_evidence(reviewed))


def test_routing_without_body_evidence_preserves_an_earlier_identity_judgment() -> None:
    document = _document(include_evidence=False)
    document = document.replace_citation(
        document.roots[0]
        .record("pre_body_identity")
        .with_identity_judgment(IdentityVerdict.UNDETERMINED)
        .with_route(STAGE)
    ).complete("pre_body_identity")
    prior_root = document.roots[0]
    reviewed = asyncio.run(review_locator_body_evidence(document))
    root = reviewed.roots[0]
    assert root.identity_judgments == prior_root.identity_judgments
    assert root.next_stage == "case_name_body_discovery"
    assert root.body_reviews[-1].decision.identity_verdict is None
    assert Document.model_validate_json(reviewed.model_dump_json()) == reviewed


def test_model_declines_all_fetched_citations_without_issuing_a_verdict() -> None:
    reviewer = FakeReviewer(_declined_decision())
    reviewed = asyncio.run(review_locator_body_evidence(_document(), reviewer=reviewer))
    assert len(reviewer.contexts) == 1
    assert reviewed.roots[0].body_reviews[-1].decision == reviewer.decision
    _assert_routed_without_judgment(reviewed)


def test_declined_citation_cannot_claim_an_identity_verdict() -> None:
    candidate = _declined_decision().model_dump(mode="python")
    candidate["identity_verdict"] = IdentityVerdict.UNDETERMINED
    with pytest.raises(ValidationError, match="Only a selected body citation"):
        BodyCorroborationDecision.model_validate(candidate)


def test_review_completes_when_no_root_needs_body_evidence() -> None:
    document = Document.from_source("No citations here.").complete("10_roots")
    reviewed = asyncio.run(review_locator_body_evidence(document))
    assert reviewed.stage_runs[-1] == STAGE
    assert reviewed.citations == ()


def test_one_fetched_body_preserves_three_separate_occurrences() -> None:
    repeated = " // ".join((BODY, BODY, BODY, BODY))
    found = make_body_evidences(
        body_id="opinion:1",
        parent_id=None,
        url=None,
        issued_on=None,
        date_basis=None,
        metadata={},
        body_text=repeated,
        locator="05-4206",
        source_text=SOURCE,
    )
    assert len(found) == 3
    assert len({item.source_offset + item.anchor_span.start for item in found}) == 3
    assert all(item.anchor_kind == "locator" for item in found)


def test_locator_excerpt_keeps_a_governing_heading_beyond_the_short_window() -> None:
    heading = "The following citations are fictitious and do not identify real decisions."
    body = f"{heading}\n\n{'Explanatory material in the table. ' * 15}\n{BODY}"
    found = make_body_evidences(
        body_id="opinion:heading",
        parent_id=None,
        url=None,
        issued_on=None,
        date_basis=None,
        metadata={},
        body_text=body,
        locator="05-4206",
        source_text=SOURCE,
    )
    locator_occurrences = [item for item in found if item.anchor_kind == "locator"]
    assert locator_occurrences
    assert all(heading in item.excerpt for item in locator_occurrences)


def test_source_filing_with_page_furniture_is_not_independent_evidence() -> None:
    found = make_body_evidences(
        body_id="filing-copy",
        parent_id=None,
        url=None,
        issued_on=None,
        date_basis=None,
        metadata={},
        body_text="Page 1\n" + SOURCE,
        locator="05-4206",
        source_text=SOURCE,
    )
    assert found == ()


def test_one_source_copy_passage_excludes_all_occurrences_from_its_body() -> None:
    continuation = "The parties filed exhibits and memoranda concerning service. " * 24
    source = SOURCE + continuation
    copied_body = "Page 7 of 12\n" + source + ("Unrelated appendix material. " * 45) + BODY
    copied = make_body_evidences(
        body_id="copied-filing",
        parent_id=None,
        url=None,
        issued_on=None,
        date_basis=None,
        metadata={},
        body_text=copied_body,
        locator="05-4206",
        source_text=source,
    )
    independent = make_body_evidences(
        body_id="independent-opinion",
        parent_id=None,
        url=None,
        issued_on=None,
        date_basis=None,
        metadata={},
        body_text=BODY,
        locator="05-4206",
        source_text=source,
    )
    assert len(copied) == 2
    assert independent

    root = FullDocketCitation.from_locator(
        citation_id="docket:copy",
        stage="sites",
        source=source,
        span=Span(source.index("No."), source.index("05-4206") + len("05-4206")),
        number_span=Span(source.index("05-4206"), source.index("05-4206") + len("05-4206")),
    )
    document = Document.from_source(source).add_citation(root).complete("sites")
    root = root.record("copy-review")
    root = root.with_body_search(
        BodySearch(
            node_id=root.nodes[-1].id,
            source=BodySource.COURTLISTENER_RECAP,
            retrospective_date=None,
            evidence=(*copied, *independent),
        )
    )
    context = BodyCorroborationContext.from_document(document.replace_citation(root), root)
    assert {item.body_id for _, _, item in context.evidence} == {"independent-opinion"}


def test_saved_body_evidence_cannot_violate_retrospective_cutoff() -> None:
    evidence = _document().roots[0].body_searches[-1].evidence
    with pytest.raises(ValidationError, match="retrospective cutoff"):
        BodySearch(
            node_id="root:node:1",
            source=BodySource.GOVINFO_OPINION,
            retrospective_date=date(2005, 1, 1),
            evidence=evidence,
        )


def test_reporter_root_uses_same_body_review_without_changing_its_locator() -> None:
    source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."
    locator = "550 U.S. 544"
    body = "A later opinion cited Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."
    root = FullReporterCitation.from_locator(
        citation_id="reporter:0",
        stage="sites",
        source=source,
        span=Span(source.index(locator), source.index(locator) + len(locator)),
    )
    document = Document.from_source(source).add_citation(root).complete("sites")
    root = root.record("fields")
    root = root.with_case_name(source, Span(0, len("Bell Atl. Corp. v. Twombly")))
    root = root.with_date(source, Span(source.index("2007"), source.index("2007") + 4))
    document = document.replace_citation(root).complete("fields")
    document = document.replace_citation(root.record("roots").with_root(root.id)).complete("roots")
    root = document.roots[0].record("22_govinfo_opinion_locator_body_search")
    found = make_body_evidences(
        body_id="opinion:2",
        parent_id=None,
        url=None,
        issued_on=date(2008, 1, 1),
        date_basis="opinion.date_filed",
        metadata={},
        body_text=body,
        locator=locator,
        source_text=source,
    )
    root = root.with_body_search(
        BodySearch(
            node_id=root.nodes[-1].id,
            source=BodySource.GOVINFO_OPINION,
            retrospective_date=None,
            evidence=found,
        )
    )
    document = document.replace_citation(root).complete("22_govinfo_opinion_locator_body_search")
    decision = BodyCorroborationDecision.model_validate(
        {
            "source": "govinfo_opinion",
            "evidence_index": 0,
            "citation_quote": "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)",
            "treatment": "cites_as_authority",
            "context_quote": None,
            "filing": {
                "locator": locator,
                "case_name": "Bell Atl. Corp. v. Twombly",
                "normalized_case_name": {
                    "kind": "adversarial",
                    "plaintiff": "Bell Atl. Corp.",
                    "defendant": "Twombly",
                    "subject": None,
                },
                "court": None,
                "date": "2007",
            },
            "third_party": {
                "locator": locator,
                "case_name": "Bell Atl. Corp. v. Twombly",
                "court": None,
                "date": "2007",
            },
            "comparisons": {
                field: {
                    "result": "unavailable" if field == "court" else "match",
                    "reason": "Compared the written citation fields.",
                }
                for field in ("locator", "case_name", "court", "date")
            },
            "identity_verdict": IdentityVerdict.CORRECT_IDENTITY.value,
            "reason": "The third-party opinion cites the same reported authority.",
        }
    )
    reviewed = asyncio.run(review_locator_body_evidence(document, reviewer=FakeReviewer(decision)))
    assert reviewed.roots[0].identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
    assert len(reviewed.roots[0].locator) == 1
    fabricated = decision.model_copy(
        update={"third_party": decision.third_party.model_copy(update={"locator": "550 U.S. 545"})}
    )
    rejected = asyncio.run(review_locator_body_evidence(document, reviewer=FakeReviewer(fabricated)))
    _assert_routed_without_judgment(rejected)
    assert "must occur inside" in rejected.roots[0].body_reviews[-1].failure_reason


def test_review_shows_distinct_documents_before_repeated_citations() -> None:
    items = tuple(
        item
        for index in range(8)
        for item in make_body_evidences(
            body_id=f"opinion:{index}",
            parent_id=None,
            url=None,
            issued_on=None,
            date_basis=None,
            metadata={},
            body_text=" // ".join((BODY, BODY, BODY)),
            locator="05-4206",
            source_text=SOURCE,
        )
    )
    selected = _diverse_evidence(BodySource.COURTLISTENER_OPINION, items)
    assert len(selected) == 6
    assert len({item.body_id for _, _, item in selected}) == 6


def test_reviewer_context_serializes_anchor_offsets_for_the_model() -> None:
    document = _document()
    context = BodyCorroborationContext.from_document(document, document.roots[0])
    payload = json.loads(context.prompt_evidence())
    assert payload[0]["anchor_span"] == {"start": 40, "end": 47}
