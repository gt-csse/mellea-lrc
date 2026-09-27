"""Validation scores use saved judgments and explicit root-level gold outcomes."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import pytest

from evaluations import validate_roots as evaluation
from mellea_lrc.api import (
    Document,
    grow_roots,
    reporter_root_lookup,
    reporter_root_lookup_ambiguous,
    reporter_root_lookup_ambiguous_llm,
    reporter_root_lookup_unique_llm,
)
from mellea_lrc.courtlistener import CourtListenerCitationLookup
from mellea_lrc.model import FullDocketCitation, FullReporterCitation, Span
from mellea_lrc.model.citations.docket_lookup import (
    DocketLookup,
    DocketLookupAttempt,
    DocketLookupCandidate,
    DocketLookupReview,
    DocketLookupReviewDecision,
)
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterAmbiguousReviewDecision,
    ReporterUniqueReviewDecision,
)

STAGES = (
    "12_reporter_root_lookup",
    "13_reporter_root_lookup_ambiguous",
    "14_reporter_root_lookup_unique_llm",
    "15_reporter_root_lookup_ambiguous_llm",
    "17_docket_root_lookup_review",
)
WORKFLOW_STAGES = (*STAGES[:-1], "16_docket_root_lookup", STAGES[-1])
SCORERS = {stage: f"score_{stage.split('_', 1)[1]}" for stage in STAGES}
RENDERERS = {stage: f"render_{stage.split('_', 1)[1]}" for stage in STAGES}
FIELDS = {"case_name", "court", "date"}
SOURCE = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007). Gamma v. Delta, No. 1:24-cv-08705 (S.D.N.Y. 2024)."
SOURCE_WITHOUT_REPORTER_DATE = (
    "Bell Atl. Corp. v. Twombly, 550 U.S. 544. Gamma v. Delta, No. 1:24-cv-08705 (S.D.N.Y. 2024)."
)
SOURCE_WITHOUT_DOCKET_DATE = (
    "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007). Gamma v. Delta, No. 1:24-cv-08705 (S.D.N.Y.)."
)
NORMALIZED_NAME = {
    "kind": "adversarial",
    "plaintiff": "Bell Atl. Corp.",
    "defendant": "Twombly",
    "subject": None,
}


def _quoted(source: str, quote: str, value: object) -> dict[str, object]:
    start = source.index(quote)
    return {
        "source": {"kind": "quoted", "start": start, "end": start + len(quote), "quote": quote},
        "normalization": {"kind": "value", "value": value},
    }


def _not_stated() -> dict[str, object]:
    return {"source": {"kind": "not_stated"}, "normalization": {"kind": "unavailable"}}


def _not_applicable() -> dict[str, object]:
    return {
        "source": {"kind": "not_applicable"},
        "normalization": {"kind": "not_applicable"},
    }


def _identity(*, date_stated: bool = True) -> dict[str, object]:
    return {
        "identity": {
            "label": "CORRECT_IDENTITY",
            "fields": {
                "case_name": {"label": "agrees"},
                "court": {"label": "agrees"},
                "date": {"label": "agrees" if date_stated else "not_stated"},
            },
        }
    }


def _gold_rows(source: str) -> list[dict[str, object]]:
    reporter_date_stated = "(2007)" in source
    docket_date_stated = "S.D.N.Y. 2024" in source
    return [
        {
            "unit": "citation",
            "id": "example-o01",
            "is_root": True,
            "root_id": "example-o01",
            "kind": "FullCaseCitation",
            "locator": _quoted(
                source,
                "550 U.S. 544",
                {"kind": "reporter", "volume": "550", "reporter": "U.S.", "page": "544"},
            ),
            "case_name": _quoted(source, "Bell Atl. Corp. v. Twombly", NORMALIZED_NAME),
            "court": {
                "source": {"kind": "inferred", "basis": "reporter"},
                "normalization": {
                    "kind": "value",
                    "value": {"id": "scotus", "name": "Supreme Court of the United States"},
                },
            },
            "date": (
                _quoted(source, "2007", {"normalized": "2007", "precision": "year"})
                if reporter_date_stated
                else _not_stated()
            ),
            "pin_cite": _not_stated(),
            "docket_entry": _not_applicable(),
            "validation": _identity(date_stated=reporter_date_stated),
        },
        {
            "unit": "citation",
            "id": "example-o02",
            "is_root": True,
            "root_id": "example-o02",
            "kind": "DocketCitation",
            "locator": _quoted(
                source,
                "No. 1:24-cv-08705",
                {"kind": "docket", "docket_number": "1:24-cv-08705"},
            ),
            "case_name": _quoted(
                source,
                "Gamma v. Delta",
                {"kind": "adversarial", "plaintiff": "Gamma", "defendant": "Delta", "subject": None},
            ),
            "court": _quoted(
                source,
                "S.D.N.Y.",
                {"id": "nysd", "name": "District Court, S.D. New York"},
            ),
            "date": (
                _quoted(source, "2024", {"normalized": "2024", "precision": "year"})
                if docket_date_stated
                else _not_stated()
            ),
            "pin_cite": _not_stated(),
            "docket_entry": _not_stated(),
            "validation": _identity(date_stated=docket_date_stated),
        },
    ]


def _write_source(tmp_path: Path, source: str = SOURCE) -> Path:
    set_dir = tmp_path / "primary"
    source_dir = set_dir / "documents_txt"
    annotation_dir = set_dir / "documents"
    source_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    source_path = source_dir / "example.txt"
    source_path.write_text(source, encoding="utf-8")
    header = {
        "unit": "header",
        "dataset": "primary",
        "document": source_path.name,
        "text": {
            "path": "primary/documents_txt/example.txt",
            "length": len(source),
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        },
    }
    rows = (header, *_gold_rows(source))
    (annotation_dir / "example.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return source_path


def _roots(tmp_path: Path, source: str = SOURCE, *, docket_stages: bool = True) -> Document:
    document = asyncio.run(grow_roots(Document.from_source(_write_source(tmp_path, source))))
    assert len(document.roots) == 2
    if docket_stages:
        document = document.complete("16_docket_root_lookup").complete("17_docket_root_lookup_review")
    return document


def _cluster(identifier: int, name: str, *, full_name: str | None = None) -> dict[str, object]:
    return {
        "id": identifier,
        "caseName": name,
        "caseNameFull": full_name,
        "court_id": "scotus",
        "dateFiled": "2007-05-21",
        "citations": [{"volume": 550, "reporter": "U.S.", "page": "544"}],
    }


class FakeLookupClient:
    def __init__(self, *clusters: dict[str, object]) -> None:
        self.response = CourtListenerCitationLookup.model_validate(
            {"citation": "550 U.S. 544", "status": 200, "clusters": list(clusters)}
        )

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        assert (volume, reporter, page) == ("550", "U.S.", "544")
        return self.response

    def get_docket(self, docket_id: str) -> None:
        pytest.fail(f"No linked docket is needed in this fixture: {docket_id}")


class FakeReviewer:
    def __init__(self, decision: ReporterUniqueReviewDecision | ReporterAmbiguousReviewDecision) -> None:
        self.decision = decision

    async def __call__(
        self, context: object
    ) -> ReporterUniqueReviewDecision | ReporterAmbiguousReviewDecision:
        return self.decision


def _reviewed_docket(
    tmp_path: Path,
    *,
    source: str = SOURCE,
    selected: bool = True,
    failed: bool = False,
    case_name: str = "match",
    court: str = "match",
    date: str = "match",
) -> Document:
    document = _roots(tmp_path, source, docket_stages=False)
    return _add_docket_review(
        document,
        selected=selected,
        failed=failed,
        case_name=case_name,
        court=court,
        date=date,
    )


def _add_docket_review(
    document: Document,
    *,
    selected: bool = True,
    failed: bool = False,
    case_name: str = "match",
    court: str = "match",
    date: str = "match",
) -> Document:
    root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
    number_span = root.locator[-1].number_span
    docket_number = document.text[number_span.start : number_span.end]
    lookup_node = root.record("16_docket_root_lookup")
    result = {"cluster_id": 999999, "docketNumber": docket_number}
    lookup = DocketLookup(
        node_id=lookup_node.nodes[-1].id,
        attempts=(
            DocketLookupAttempt(
                source_type="o",
                query=f"docketNumber:({docket_number})",
                pages=({"results": [result]},),
            ),
        ),
        candidates=(
            DocketLookupCandidate(
                source_type="o",
                record_id="999999",
                attempt_index=0,
                page_index=0,
                result_index=0,
                docket_number=docket_number,
                docket_similarity=100,
            ),
        ),
        shortlisted_candidate_indices=(0,),
    )
    document = document.replace_citation(lookup_node.with_docket_lookup(lookup)).complete(
        "16_docket_root_lookup"
    )
    root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
    review_node = root.record("17_docket_root_lookup_review")
    if failed:
        review = DocketLookupReview(node_id=review_node.nodes[-1].id, failure_reason="Model unavailable")
    else:
        results = {
            "docket_number": "match",
            "case_name": case_name,
            "court": court,
            "date": date,
        }
        if not selected:
            results = dict.fromkeys(results, "undetermined")
        assessments = {
            field: {
                "propose_replacement": False,
                "quote": None,
                "result": result,
                "reason": "Compared with the saved record.",
            }
            for field, result in results.items()
        }
        assessments["case_name"]["normalized"] = (
            root.case_name[-1].get_normalized().model_dump(mode="python")
            if root.case_name and root.case_name[-1].normalizable
            else None
        )
        decision = DocketLookupReviewDecision.model_validate(
            {
                "selected_candidate_index": 0 if selected else None,
                **assessments,
                "reason": "Reviewed the shortlist.",
            }
        )
        review = DocketLookupReview(node_id=review_node.nodes[-1].id, decision=decision)
    return document.replace_citation(review_node.with_docket_lookup_review(review)).complete(
        "17_docket_root_lookup_review"
    )


def _model_fields() -> dict[str, object]:
    return {
        "case_name": {
            "propose_replacement": False,
            "quote": None,
            "normalized": NORMALIZED_NAME,
            "result": "match",
            "reason": "The filing names the candidate's case.",
        },
        "court": {
            "propose_replacement": False,
            "quote": None,
            "result": "match",
            "reason": "The inferred court matches.",
        },
        "date": {
            "propose_replacement": False,
            "quote": None,
            "result": "match",
            "reason": "The written year matches.",
        },
        "reason": "The saved candidate matches the citation.",
    }


def _finish_unrouted_stages(document: Document) -> Document:
    document = reporter_root_lookup_ambiguous(document)
    document = asyncio.run(reporter_root_lookup_unique_llm(document))
    return asyncio.run(reporter_root_lookup_ambiguous_llm(document))


def _complete_reporter_stages(document: Document) -> Document:
    for stage in STAGES[:-1]:
        document = document.complete(stage)
    return document


def test_validate_roots_composes_stages_in_execution_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = importlib.import_module("mellea_lrc.workflows.validate_roots")
    initial = Document.from_source(_write_source(tmp_path))
    called: list[str] = []

    def run_sync(stage: str):
        def run(document: Document) -> Document:
            assert document.stage_runs == (*initial.stage_runs, *called)
            called.append(stage)
            return document.complete(stage)

        return run

    def run_async(stage: str):
        async def run(document: Document) -> Document:
            assert document.stage_runs == (*initial.stage_runs, *called)
            called.append(stage)
            return document.complete(stage)

        return run

    for stage in WORKFLOW_STAGES:
        runner = run_async if stage.endswith(("_review", "_llm")) else run_sync
        monkeypatch.setattr(workflow, stage.split("_", 1)[1], runner(stage))

    result = asyncio.run(workflow.validate_roots(initial))
    assert tuple(called) == WORKFLOW_STAGES
    assert result.stage_runs == (*initial.stage_runs, *WORKFLOW_STAGES)


def test_final_score_uses_docket_review_as_latest_checkpoint(tmp_path: Path) -> None:
    roots = _roots(tmp_path, docket_stages=False)
    client = FakeLookupClient(
        _cluster(1, "Bell Atlantic Corporation v. Twombly", full_name="Bell Atlantic Corporation v. Twombly")
    )
    reporter_lookup = reporter_root_lookup(roots, client=client)
    reviewed_reporter = _finish_unrouted_stages(reporter_lookup)
    final = _add_docket_review(reviewed_reporter)

    assert final.stage_runs[-6:] == WORKFLOW_STAGES
    score = evaluation.score_validate_roots(final)
    assert tuple(stage.stage for stage in score.stages) == STAGES
    assert score == evaluation.score_validate_roots(final.get_stage("17_docket_root_lookup_review"))


@pytest.mark.parametrize("name", (*SCORERS.values(), "score_validate_roots"))
def test_public_scorers_take_only_a_document(name: str) -> None:
    parameters = tuple(inspect.signature(getattr(evaluation, name)).parameters.values())
    assert len(parameters) == 1
    assert parameters[0].name == "document"
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters[0].default is inspect.Parameter.empty


def test_docket_review_scores_selected_fields_by_locator_and_preserves_stage_boundary(
    tmp_path: Path,
) -> None:
    reviewed = _reviewed_docket(tmp_path, court="mismatch", date="undetermined")
    docket = next(root for root in reviewed.roots if isinstance(root, FullDocketCitation))
    assert docket.docket_lookup_review is not None
    assert docket.docket_lookup_review.decision is not None
    assert docket.docket_lookup_review.decision.selected_candidate_index == 0
    assert docket.docket_lookup.candidates[0].record_id == "999999"
    assert docket.id != "example-o02"

    final = _complete_reporter_stages(reviewed)
    restored = Document.model_validate_json(final.model_dump_json())
    stage = evaluation.score_docket_root_lookup_review(final)
    assert stage == evaluation.score_docket_root_lookup_review(reviewed)
    assert stage == evaluation.score_docket_root_lookup_review(
        final.get_stage("17_docket_root_lookup_review")
    )
    assert stage == evaluation.score_docket_root_lookup_review(restored)
    assert stage.metrics == {
        "case_name": evaluation.Precision(1, 1),
        "court": evaluation.Precision(0, 1),
        "date": evaluation.Precision(0, 1),
    }
    assert "## 17_docket_root_lookup_review" in evaluation.render_docket_root_lookup_review(stage)
    workflow = evaluation.score_validate_roots(final)
    assert tuple(score.stage for score in workflow.stages) == STAGES
    assert workflow.fields == {
        "case_name": evaluation.FieldScore(1, 1, 2),
        "court": evaluation.FieldScore(0, 1, 2),
        "date": evaluation.FieldScore(0, 1, 2),
    }
    report = evaluation.render_validate_roots(workflow)
    assert report.index("## 15_reporter_root_lookup_ambiguous_llm") < report.index(
        "## 17_docket_root_lookup_review"
    )
    assert "## 16_docket_root_lookup\n" not in report


def test_docket_undetermined_without_source_date_counts_as_not_stated(tmp_path: Path) -> None:
    reviewed = _reviewed_docket(tmp_path, source=SOURCE_WITHOUT_DOCKET_DATE, date="undetermined")
    docket = next(root for root in reviewed.roots if isinstance(root, FullDocketCitation))
    assert not docket.date
    assert evaluation.score_docket_root_lookup_review(reviewed).metrics["date"] == (
        evaluation.Precision(1, 1)
    )
    final = _complete_reporter_stages(reviewed)
    assert evaluation.score_validate_roots(final).fields["date"] == evaluation.FieldScore(1, 1, 2)


@pytest.mark.parametrize("failed", (False, True))
def test_docket_review_without_selection_makes_no_prediction(tmp_path: Path, failed: bool) -> None:
    reviewed = _reviewed_docket(tmp_path, selected=False, failed=failed)
    assert evaluation.score_docket_root_lookup_review(reviewed).metrics == {
        field: evaluation.Precision(0, 0) for field in FIELDS
    }
    final = _complete_reporter_stages(reviewed)
    assert evaluation.score_validate_roots(final).fields == {
        field: evaluation.FieldScore(0, 0, 2) for field in FIELDS
    }


def test_rule_ambiguity_scores_only_the_selected_candidate_and_keeps_stage_boundaries(
    tmp_path: Path,
) -> None:
    client = FakeLookupClient(
        _cluster(1, "Jones v. Smith", full_name="Jones v. Smith"),
        _cluster(2, "Bell Atlantic Corporation v. Twombly", full_name="Bell Atlantic Corporation v. Twombly"),
    )
    lookup = reporter_root_lookup(_roots(tmp_path), client=client)
    ambiguous = reporter_root_lookup_ambiguous(lookup, client=client)
    root = next(root for root in ambiguous.roots if isinstance(root, FullReporterCitation))
    assert root.reporter_exact_ambiguity_resolution is not None
    assert root.reporter_exact_ambiguity_resolution.selected_candidate_index == 1
    assert [(judgment.candidate_index, judgment.result) for judgment in root.case_name_judgments] == [
        (0, MatchResult.MISMATCH),
        (1, MatchResult.MATCH),
    ]

    final = asyncio.run(reporter_root_lookup_unique_llm(ambiguous))
    final = asyncio.run(reporter_root_lookup_ambiguous_llm(final))
    restored = Document.model_validate_json(final.model_dump_json())
    assert restored == final
    for stage, scorer_name in SCORERS.items():
        scorer = getattr(evaluation, scorer_name)
        assert scorer(final) == scorer(final.get_stage(stage)) == scorer(restored)
        assert set(scorer(final).metrics) == FIELDS
        rendered = getattr(evaluation, RENDERERS[stage])(scorer(final))
        assert stage in rendered
    assert evaluation.score_reporter_root_lookup(lookup).metrics == {
        field: evaluation.Precision(0, 0) for field in FIELDS
    }
    assert evaluation.score_reporter_root_lookup_ambiguous(final).metrics == {
        field: evaluation.Precision(1, 1) for field in FIELDS
    }
    assert evaluation.score_reporter_root_lookup_ambiguous_llm(final).metrics == {
        field: evaluation.Precision(0, 0) for field in FIELDS
    }
    assert evaluation.score_reporter_root_lookup_unique_llm(final).metrics == {
        field: evaluation.Precision(0, 0) for field in FIELDS
    }
    score = evaluation.score_validate_roots(final)
    assert tuple(stage.stage for stage in score.stages) == STAGES
    assert set(score.fields) == FIELDS
    assert set(score.as_dict()) == {"stages", "fields"}
    assert score.fields == {field: evaluation.FieldScore(1, 1, 2) for field in FIELDS}
    report = evaluation.render_validate_roots(score)
    assert "| Field | Precision | Recall |" in report
    assert "identity" not in score.as_dict()["fields"]


def test_unique_undetermined_is_an_explicit_incorrect_judgment(tmp_path: Path) -> None:
    client = FakeLookupClient(_cluster(1, "Bell Atlantic Corporation v. Twombly"))
    lookup = reporter_root_lookup(_roots(tmp_path), client=client)
    root = next(root for root in lookup.roots if isinstance(root, FullReporterCitation))
    assert root.case_name_judgments[-1].result is MatchResult.UNDETERMINED
    assert evaluation.score_reporter_root_lookup(lookup).metrics == {
        "case_name": evaluation.Precision(0, 1),
        "court": evaluation.Precision(1, 1),
        "date": evaluation.Precision(1, 1),
    }

    decision = ReporterUniqueReviewDecision.model_validate(_model_fields())
    final = reporter_root_lookup_ambiguous(lookup)
    final = asyncio.run(reporter_root_lookup_unique_llm(final, reviewer=FakeReviewer(decision)))
    final = asyncio.run(reporter_root_lookup_ambiguous_llm(final))
    assert evaluation.score_reporter_root_lookup(final) == evaluation.score_reporter_root_lookup(lookup)
    assert evaluation.score_reporter_root_lookup_unique_llm(final).metrics == {
        field: evaluation.Precision(1, 1) for field in FIELDS
    }
    assert evaluation.score_validate_roots(final).fields == {
        field: evaluation.FieldScore(1, 1, 2) for field in FIELDS
    }


def test_ambiguous_model_review_scores_only_its_new_selected_judgments(tmp_path: Path) -> None:
    client = FakeLookupClient(
        _cluster(1, "Jones v. Smith", full_name="Jones v. Smith"),
        _cluster(2, "Bell Atlantic Corporation v. Twombly"),
    )
    lookup = reporter_root_lookup(_roots(tmp_path), client=client)
    ambiguous = reporter_root_lookup_ambiguous(lookup, client=client)
    root = next(root for root in ambiguous.roots if isinstance(root, FullReporterCitation))
    assert root.reporter_exact_ambiguity_resolution is not None
    assert root.reporter_exact_ambiguity_resolution.selected_candidate_index is None
    decision = ReporterAmbiguousReviewDecision.model_validate(
        {"selected_candidate_index": 1, **_model_fields()}
    )
    before_review = asyncio.run(reporter_root_lookup_unique_llm(ambiguous))
    final = asyncio.run(reporter_root_lookup_ambiguous_llm(before_review, reviewer=FakeReviewer(decision)))
    assert evaluation.score_reporter_root_lookup_ambiguous(final).metrics == {
        field: evaluation.Precision(0, 0) for field in FIELDS
    }
    assert evaluation.score_reporter_root_lookup_ambiguous_llm(final).metrics == {
        field: evaluation.Precision(1, 1) for field in FIELDS
    }
    assert evaluation.score_validate_roots(final).fields == {
        field: evaluation.FieldScore(1, 1, 2) for field in FIELDS
    }


def test_not_stated_is_a_final_outcome_only_for_a_processed_root(tmp_path: Path) -> None:
    client = FakeLookupClient(
        _cluster(1, "Bell Atlantic Corporation v. Twombly", full_name="Bell Atlantic Corporation v. Twombly")
    )
    roots = _roots(tmp_path, SOURCE_WITHOUT_REPORTER_DATE)
    lookup = reporter_root_lookup(roots, client=client)
    final = _finish_unrouted_stages(lookup)
    reporter = next(root for root in final.roots if isinstance(root, FullReporterCitation))
    assert not reporter.date
    assert reporter.date_judgments == ()
    assert evaluation.score_reporter_root_lookup(final).metrics["date"] == evaluation.Precision(0, 0)
    assert evaluation.score_validate_roots(final).fields == {
        field: evaluation.FieldScore(1, 1, 2) for field in FIELDS
    }
    assert all(
        not root.identity_judgments for root in final.roots if not isinstance(root, FullReporterCitation)
    )


def test_lookup_miss_does_not_predict_an_absent_citation_field(tmp_path: Path) -> None:
    lookup = reporter_root_lookup(_roots(tmp_path, SOURCE_WITHOUT_REPORTER_DATE), client=FakeLookupClient())
    reporter = next(root for root in lookup.roots if isinstance(root, FullReporterCitation))
    assert any(node.stage == "12_reporter_root_lookup" for node in reporter.nodes)
    assert not reporter.date
    assert reporter.case_name_judgments == reporter.court_judgments == reporter.date_judgments == ()

    final = _finish_unrouted_stages(lookup)
    assert evaluation.score_reporter_root_lookup(final).metrics == {
        field: evaluation.Precision(0, 0) for field in FIELDS
    }
    assert evaluation.score_validate_roots(final).fields == {
        field: evaluation.FieldScore(0, 0, 2) for field in FIELDS
    }


def test_explicit_undetermined_for_an_absent_field_agrees_with_not_stated_gold(
    tmp_path: Path,
) -> None:
    client = FakeLookupClient(_cluster(1, "Bell Atlantic Corporation v. Twombly"))
    lookup = reporter_root_lookup(_roots(tmp_path, SOURCE_WITHOUT_REPORTER_DATE), client=client)
    decision = ReporterUniqueReviewDecision.model_validate(
        {
            **_model_fields(),
            "date": {
                "propose_replacement": False,
                "quote": None,
                "result": "undetermined",
                "reason": "The filing states no date for this reporter root.",
            },
        }
    )
    before_review = reporter_root_lookup_ambiguous(lookup)
    reviewed = asyncio.run(reporter_root_lookup_unique_llm(before_review, reviewer=FakeReviewer(decision)))
    reviewed = asyncio.run(reporter_root_lookup_ambiguous_llm(reviewed))
    reporter = next(root for root in reviewed.roots if isinstance(root, FullReporterCitation))
    assert reporter.date_judgments[-1].result is MatchResult.UNDETERMINED
    assert reporter.date_judgments[-1].reading_index is None
    assert evaluation.score_reporter_root_lookup_unique_llm(reviewed).metrics["date"] == (
        evaluation.Precision(1, 1)
    )
    assert evaluation.score_validate_roots(reviewed).fields["date"] == evaluation.FieldScore(1, 1, 2)


def test_primary_sized_gold_keeps_all_440_roots_in_each_recall_denominator(tmp_path: Path) -> None:
    set_dir = tmp_path / "primary"
    source_dir = set_dir / "documents_txt"
    annotation_dir = set_dir / "documents"
    source_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    source_path = source_dir / "many.txt"
    locators = [f"{index} U.S. {index}" for index in range(1, 394)]
    locators.extend(f"No. 1:24-cv-{index:05d}" for index in range(1, 48))
    source = "\n".join(locators)
    source_path.write_text(source, encoding="utf-8")
    document = Document.from_source(source_path)
    assert document.text == source

    header = {
        "unit": "header",
        "dataset": "primary",
        "document": source_path.name,
        "text": {
            "path": "primary/documents_txt/many.txt",
            "length": len(source),
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        },
    }
    rows: list[dict[str, object]] = [header]
    start = 0
    for index, locator in enumerate(locators):
        reporter = index < 393
        value: dict[str, object] = (
            {
                "kind": "reporter",
                "volume": str(index + 1),
                "reporter": "U.S.",
                "page": str(index + 1),
            }
            if reporter
            else {"kind": "docket", "docket_number": locator.removeprefix("No. ")}
        )
        rows.append(
            {
                "unit": "citation",
                "id": f"many-o{index + 1:03d}",
                "is_root": True,
                "kind": "FullCaseCitation" if reporter else "DocketCitation",
                "locator": {
                    "source": {
                        "kind": "quoted",
                        "start": start,
                        "end": start + len(locator),
                        "quote": locator,
                    },
                    "normalization": {"kind": "value", "value": value},
                },
                "validation": {
                    "identity": {
                        "label": "CORRECT_IDENTITY",
                        "fields": {field: {"label": "not_stated"} for field in FIELDS},
                    }
                },
            }
        )
        start += len(locator) + 1
    assert len(rows) == 441
    (annotation_dir / "many.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    docket_locator = locators[393]
    docket_start = source.index(docket_locator)
    docket = FullDocketCitation.from_locator(
        citation_id="generated-docket-root",
        stage="test_sites",
        source=source,
        span=Span(docket_start, docket_start + len(docket_locator)),
        number_span=Span(docket_start + len("No. "), docket_start + len(docket_locator)),
    )
    document = document.add_citation(docket).complete("test_sites")
    document = document.replace_citation(docket.record("10_roots").with_root(docket.id)).complete("10_roots")
    document = _add_docket_review(
        document, case_name="undetermined", court="undetermined", date="undetermined"
    )
    document = _complete_reporter_stages(document)

    score = evaluation.score_validate_roots(document)
    assert score.fields == {field: evaluation.FieldScore(1, 1, 440) for field in FIELDS}
    assert all(score.fields[field].as_dict()["recall"] == 1 / 440 for field in FIELDS)


def test_missing_stage_checkpoint_raises(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    with pytest.raises(KeyError, match="Stage has not run"):
        evaluation.score_reporter_root_lookup(roots)
    unvalidated = Document.from_source(roots.source_path)
    with pytest.raises(KeyError, match="Stage has not run"):
        evaluation.score_docket_root_lookup_review(unvalidated)
