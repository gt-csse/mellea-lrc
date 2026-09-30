"""Validation scores use saved judgments and explicit root-level gold outcomes."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
from datetime import date
from pathlib import Path

import pytest

from evaluations import validate_roots as evaluation
from mellea_lrc.api import (
    Document,
    grow_roots,
    reporter_root_lookup_ambiguous_llm_judgment,
    reporter_root_lookup_ambiguous_rule_judgment,
    reporter_root_lookup_cluster_retrieval,
    reporter_root_lookup_docket_retrieval,
    reporter_root_lookup_unique_llm_judgment,
    reporter_root_lookup_unique_rule_judgment,
)
from mellea_lrc.providers.courtlistener import CourtListenerCitationLookup
from mellea_lrc.model import FullDocketCitation, FullReporterCitation, Span
from mellea_lrc.model.citations.body_evidence import (
    BodyCorroborationDecision,
    BodyCorroborationReview,
    BodySearch,
    BodySearchAttempt,
    BodySource,
)
from mellea_lrc.model.citations.docket_lookup import (
    DocketLookup,
    DocketLookupAttempt,
    DocketLookupCandidate,
    DocketLookupReview,
    DocketLookupReviewDecision,
)
from mellea_lrc.model.citations.field_body_evidence import (
    FieldBodySearch,
    IntendedCaseConfidence,
    IntendedCaseDecision,
    IntendedCaseReview,
)
from mellea_lrc.model.citations.govinfo_lookup import (
    GovInfoDocketLookup,
    GovInfoDocketReview,
    GovInfoLookupAttempt,
    GovInfoLookupCandidate,
)
from mellea_lrc.model.citations.judgments import IdentityBasis, IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterAmbiguousReviewDecision,
    ReporterUniqueReviewDecision,
)
from mellea_lrc.validation.body_search.common import make_body_evidences

STAGES = (
    "13.1_reporter_root_lookup_unique_rule_judgment",
    "13.2_reporter_root_lookup_ambiguous_rule_judgment",
    "14_reporter_root_lookup_unique_llm_judgment",
    "15_reporter_root_lookup_ambiguous_llm_judgment",
    "17_docket_root_lookup_courtlistener_llm_review",
    "19_docket_root_lookup_govinfo_llm_review",
)
WORKFLOW_STAGES = (
    "12.1_reporter_root_lookup_cluster_retrieval",
    "12.2_reporter_root_lookup_docket_retrieval",
    STAGES[0],
    STAGES[1],
    *STAGES[2:-2],
    "16_docket_root_lookup_courtlistener_retrieval",
    "17_docket_root_lookup_courtlistener_llm_review",
    "18_docket_root_lookup_govinfo_retrieval",
    STAGES[-1],
)
SCORERS = {stage: f"score_{stage.split('_', 1)[1]}" for stage in STAGES}
RETRIEVAL_SCORERS = {stage: f"score_{stage.split('_', 1)[1]}" for stage in evaluation.RETRIEVAL_STAGES}
RENDERERS = {stage: name.replace("score_", "render_") for stage, name in SCORERS.items()}
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
        for stage in (
            "16_docket_root_lookup_courtlistener_retrieval",
            "17_docket_root_lookup_courtlistener_llm_review",
            "18_docket_root_lookup_govinfo_retrieval",
            "19_docket_root_lookup_govinfo_llm_review",
        ):
            document = document.complete(stage)
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
    lookup_node = root.record("16_docket_root_lookup_courtlistener_retrieval")
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
        "16_docket_root_lookup_courtlistener_retrieval"
    )
    root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
    review_node = root.record("17_docket_root_lookup_courtlistener_llm_review")
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
            results = dict.fromkeys(results, "unavailable")
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
        "17_docket_root_lookup_courtlistener_llm_review"
    )


def _add_govinfo_review(
    document: Document,
    *,
    selected: bool = True,
    case_name: str = "match",
    court: str = "mismatch",
    date: str = "unavailable",
) -> Document:
    root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
    number_span = root.locator[-1].number_span
    docket_number = document.text[number_span.start : number_span.end]
    lookup_node = root.record("18_docket_root_lookup_govinfo_retrieval")
    result = {
        "packageId": "USCOURTS-nysd-1_24-cv-08705",
        "granuleId": "USCOURTS-nysd-1_24-cv-08705-0",
        "caseNumber": docket_number,
    }
    lookup = GovInfoDocketLookup(
        node_id=lookup_node.nodes[-1].id,
        attempts=(GovInfoLookupAttempt(query="docketNumber:1:24-cv-08705", pages=({"results": [result]},)),),
        candidates=(
            GovInfoLookupCandidate(
                attempt_index=0,
                page_index=0,
                result_index=0,
                package_id=result["packageId"],
                granule_id=result["granuleId"],
                court_code="nysd",
                docket_number=docket_number,
                docket_similarity=100,
            ),
        ),
        shortlisted_candidate_indices=(0,),
    )
    document = document.replace_citation(lookup_node.with_govinfo_docket_lookup(lookup)).complete(
        "18_docket_root_lookup_govinfo_retrieval"
    )
    root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
    review_node = root.record("19_docket_root_lookup_govinfo_llm_review")
    assessments = {
        field: {
            "propose_replacement": False,
            "quote": None,
            "result": result,
            "reason": "Compared with the selected GovInfo package.",
        }
        for field, result in {
            "docket_number": "match",
            "case_name": case_name,
            "court": court,
            "date": date,
        }.items()
    }
    if not selected:
        for assessment in assessments.values():
            assessment["result"] = "unavailable"
    assessments["case_name"]["normalized"] = NORMALIZED_NAME
    decision = DocketLookupReviewDecision.model_validate(
        {
            "selected_candidate_index": 0 if selected else None,
            **assessments,
            "reason": "Reviewed the GovInfo shortlist.",
        }
    )
    review = GovInfoDocketReview(node_id=review_node.nodes[-1].id, decision=decision)
    return document.replace_citation(review_node.with_govinfo_docket_review(review)).complete(
        "19_docket_root_lookup_govinfo_llm_review"
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
    document = reporter_root_lookup_docket_retrieval(document)
    document = reporter_root_lookup_unique_rule_judgment(document)
    document = reporter_root_lookup_ambiguous_rule_judgment(document)
    document = asyncio.run(reporter_root_lookup_unique_llm_judgment(document))
    return asyncio.run(reporter_root_lookup_ambiguous_llm_judgment(document))


def _complete_reporter_stages(document: Document) -> Document:
    for stage in WORKFLOW_STAGES[:6]:
        document = document.complete(stage)
    for stage in ("18_docket_root_lookup_govinfo_retrieval", "19_docket_root_lookup_govinfo_llm_review"):
        if stage not in document.stage_runs:
            document = document.complete(stage)
    return document


@pytest.mark.parametrize("with_client", [False, True])
def test_validate_roots_composes_stages_in_execution_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_client: bool
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
        runner = run_async if stage.startswith(("14_", "15_", "17_", "19_")) else run_sync
        monkeypatch.setattr(workflow, stage.split("_", 1)[1], runner(stage))

    body_stages = (
        "20_locator_body_courtlistener_opinion_retrieval",
        "21_locator_body_courtlistener_recap_retrieval",
        "22_locator_body_govinfo_opinion_retrieval",
        "23_locator_body_llm_judgment",
    )
    cutoff = date(2024, 1, 1)
    selected_client = object() if with_client else None

    async def run_body(
        document: Document, *, retrospective_date: date | None, courtlistener_client: object = None
    ) -> Document:
        assert retrospective_date == cutoff
        assert courtlistener_client is selected_client
        for stage in body_stages:
            assert document.stage_runs == (*initial.stage_runs, *called)
            called.append(stage)
            document = document.complete(stage)
        return document

    monkeypatch.setattr(workflow, "_validate_locator_bodies", run_body)
    client_kwargs = {"courtlistener_client": selected_client} if selected_client is not None else {}
    result = asyncio.run(workflow.validate_roots(initial, retrospective_date=cutoff, **client_kwargs))
    assert tuple(called) == (*WORKFLOW_STAGES, *body_stages)
    assert result.stage_runs == (*initial.stage_runs, *WORKFLOW_STAGES, *body_stages)


@pytest.mark.parametrize("completed_fields", range(5))
def test_validation_resumes_its_optional_intended_case_stages(
    monkeypatch: pytest.MonkeyPatch, completed_fields: int
) -> None:
    workflow = importlib.import_module("mellea_lrc.workflows.validate_roots")
    fields = evaluation.INTENDED_CASE_STAGES
    document = Document.from_source("source")
    for stage in (*WORKFLOW_STAGES, *evaluation.LOCATOR_BODY_STAGES, *fields[:completed_fields]):
        document = document.complete(stage)
    cutoff = date(2024, 1, 1)
    client = object()
    calls: list[str] = []
    checkpoints: list[Document] = []

    def retrieve(stage: str, uses_client: bool):
        def run(saved: Document, *, retrospective_date: date | None, **kwargs: object) -> Document:
            assert retrospective_date == cutoff
            assert kwargs == ({"client": client} if uses_client else {})
            calls.append(stage)
            return saved.complete(stage)

        return run

    for stage in fields[:3]:
        monkeypatch.setattr(workflow, stage.split("_", 1)[1], retrieve(stage, "courtlistener" in stage))

    async def review(saved: Document) -> Document:
        calls.append(fields[-1])
        return saved.complete(fields[-1])

    monkeypatch.setattr(workflow, "intended_case_llm_selection", review)
    result = asyncio.run(
        workflow.validate_roots(
            document,
            retrospective_date=cutoff,
            courtlistener_client=client,
            search_other_fields=True,
            checkpoint=checkpoints.append,
        )
    )
    assert tuple(calls) == fields[completed_fields:]
    assert tuple(saved.stage_runs[-1] for saved in checkpoints) == fields[completed_fields:]
    assert result.stage_runs == (*WORKFLOW_STAGES, *evaluation.LOCATOR_BODY_STAGES, *fields)
    if document.stage_runs:
        assert result.get_stage(document.stage_runs[-1]) == document


def test_final_score_uses_docket_review_as_latest_checkpoint(tmp_path: Path) -> None:
    roots = _roots(tmp_path, docket_stages=False)
    client = FakeLookupClient(
        _cluster(1, "Bell Atlantic Corporation v. Twombly", full_name="Bell Atlantic Corporation v. Twombly")
    )
    reporter_lookup = reporter_root_lookup_cluster_retrieval(roots, client=client)
    reviewed_reporter = _finish_unrouted_stages(reporter_lookup)
    final = _add_docket_review(reviewed_reporter)
    final = final.complete("18_docket_root_lookup_govinfo_retrieval").complete(
        "19_docket_root_lookup_govinfo_llm_review"
    )

    assert final.stage_runs[-len(WORKFLOW_STAGES) :] == WORKFLOW_STAGES
    score = evaluation.score_validate_roots(final)
    assert tuple(stage.stage for stage in score.stages) == STAGES
    assert score == evaluation.score_validate_roots(
        final.get_stage("19_docket_root_lookup_govinfo_llm_review")
    )


@pytest.mark.parametrize(
    "name",
    (
        *SCORERS.values(),
        *RETRIEVAL_SCORERS.values(),
        "score_locator_body_llm_judgment",
        "score_validate_roots",
    ),
)
def test_public_scorers_take_only_a_document(name: str) -> None:
    parameters = tuple(inspect.signature(getattr(evaluation, name)).parameters.values())
    assert len(parameters) == 1
    assert parameters[0].name == "document"
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters[0].default is inspect.Parameter.empty
    assert parameters[0].annotation == "Document"


def test_retrieval_stage_scorers_are_named_and_mapped() -> None:
    assert not hasattr(evaluation, "score_retrieval_stage")
    assert tuple(evaluation.RETRIEVAL_STAGE_SCORERS) == tuple(
        (stage, getattr(evaluation, name)) for stage, name in RETRIEVAL_SCORERS.items()
    )


def test_docket_review_scores_selected_fields_by_locator_and_preserves_stage_boundary(
    tmp_path: Path,
) -> None:
    reviewed = _reviewed_docket(tmp_path, court="mismatch", date="unavailable")
    docket = next(root for root in reviewed.roots if isinstance(root, FullDocketCitation))
    assert docket.docket_lookup_review is not None
    assert docket.docket_lookup_review.decision is not None
    assert docket.docket_lookup_review.decision.selected_candidate_index == 0
    assert docket.docket_lookup.candidates[0].record_id == "999999"
    assert docket.id != "example-o02"

    final = _complete_reporter_stages(reviewed)
    restored = Document.model_validate_json(final.model_dump_json())
    stage = evaluation.score_docket_root_lookup_courtlistener_llm_review(final)
    assert stage == evaluation.score_docket_root_lookup_courtlistener_llm_review(reviewed)
    assert stage == evaluation.score_docket_root_lookup_courtlistener_llm_review(
        final.get_stage("17_docket_root_lookup_courtlistener_llm_review")
    )
    assert stage == evaluation.score_docket_root_lookup_courtlistener_llm_review(restored)
    assert stage.metrics == {
        "case_name": evaluation.Precision(1, 1),
        "court": evaluation.Precision(0, 1),
        "date": evaluation.Precision(0, 1),
    }
    assert f"## {STAGES[-2]}" in evaluation.render_docket_root_lookup_courtlistener_llm_review(stage)
    workflow = evaluation.score_validate_roots(final)
    assert tuple(score.stage for score in workflow.stages) == STAGES
    assert workflow.fields == {
        "case_name": evaluation.FieldScore(1, 1, 2),
        "court": evaluation.FieldScore(0, 1, 2),
        "date": evaluation.FieldScore(0, 1, 2),
    }
    assert workflow.identity == evaluation.IdentityScore(0, 1, 2, 0)
    report = evaluation.render_validate_roots(workflow)
    numbered_headings = tuple(
        line.removeprefix("## ")
        for line in report.splitlines()
        if line.startswith("## ") and line[3:4].isdigit()
    )
    assert numbered_headings == WORKFLOW_STAGES
    assert workflow.as_dict()["stage_order"] == list(WORKFLOW_STAGES)
    assert ("## 16_docket_root_lookup_courtlistener_retrieval\n\n| Metric | Coverage |") in report


def test_govinfo_review_scores_selected_candidate_and_supplies_final_docket_labels(
    tmp_path: Path,
) -> None:
    reviewed = _reviewed_docket(tmp_path)
    final = _add_govinfo_review(reviewed)
    docket = next(root for root in final.roots if isinstance(root, FullDocketCitation))
    assert docket.govinfo_docket_review is not None
    assert docket.govinfo_docket_review.decision is not None
    assert docket.govinfo_docket_review.decision.selected_candidate_index == 0

    stage = evaluation.score_docket_root_lookup_govinfo_llm_review(final)
    assert stage.metrics == {
        "case_name": evaluation.Precision(1, 1),
        "court": evaluation.Precision(0, 1),
        "date": evaluation.Precision(0, 1),
    }
    assert f"## {STAGES[-1]}" in evaluation.render_docket_root_lookup_govinfo_llm_review(stage)
    assert evaluation._final_docket_field_label(docket, "case_name") == "agrees"
    assert evaluation._final_docket_field_label(docket, "court") == "disagrees"
    assert evaluation._final_docket_field_label(docket, "date") == "unavailable"


def test_govinfo_lookup_stage_makes_no_judgments(tmp_path: Path) -> None:
    reviewed = _reviewed_docket(tmp_path)
    looked_up = _add_govinfo_review(reviewed, selected=False)
    assert evaluation.score_docket_root_lookup_govinfo_llm_review(looked_up).metrics == {
        field: evaluation.Precision(0, 0) for field in FIELDS
    }
    docket = next(root for root in looked_up.roots if isinstance(root, FullDocketCitation))
    assert evaluation._final_docket_field_label(docket, "court") == "agrees"


def test_docket_unavailable_without_source_date_matches_not_stated_gold(tmp_path: Path) -> None:
    reviewed = _reviewed_docket(tmp_path, source=SOURCE_WITHOUT_DOCKET_DATE, date="unavailable")
    docket = next(root for root in reviewed.roots if isinstance(root, FullDocketCitation))
    assert not docket.date
    assert evaluation.score_docket_root_lookup_courtlistener_llm_review(reviewed).metrics["date"] == (
        evaluation.Precision(1, 1)
    )
    final = _complete_reporter_stages(reviewed)
    assert evaluation.score_validate_roots(final).fields["date"] == evaluation.FieldScore(1, 1, 2)


def test_docket_unavailable_with_source_date_does_not_match_gold(tmp_path: Path) -> None:
    reviewed = _reviewed_docket(tmp_path, date="unavailable")
    assert evaluation.score_docket_root_lookup_courtlistener_llm_review(reviewed).metrics["date"] == (
        evaluation.Precision(0, 1)
    )
    final = _complete_reporter_stages(reviewed)
    assert evaluation.score_validate_roots(final).fields["date"] == evaluation.FieldScore(0, 1, 2)


@pytest.mark.parametrize("failed", (False, True))
def test_docket_review_without_selection_makes_no_prediction(tmp_path: Path, failed: bool) -> None:
    reviewed = _reviewed_docket(tmp_path, selected=False, failed=failed)
    assert evaluation.score_docket_root_lookup_courtlistener_llm_review(reviewed).metrics == {
        field: evaluation.Precision(0, 0) for field in FIELDS
    }
    final = _complete_reporter_stages(reviewed)
    assert evaluation.score_validate_roots(final).fields == {
        field: evaluation.FieldScore(0, 0, 2) for field in FIELDS
    }
    assert evaluation.score_validate_roots(final).identity == evaluation.IdentityScore(0, 0, 2, 0)


def test_lookup_identity_abstains_when_case_name_is_unavailable(tmp_path: Path) -> None:
    reviewed = _reviewed_docket(tmp_path, case_name="unavailable", court="match", date="match")
    final = _complete_reporter_stages(reviewed)
    assert evaluation.score_validate_roots(final).identity == evaluation.IdentityScore(0, 0, 2, 1)


def test_lookup_identity_ignores_unavailable_court_and_date(tmp_path: Path) -> None:
    reviewed = _reviewed_docket(tmp_path, case_name="match", court="unavailable", date="unavailable")
    final = _complete_reporter_stages(reviewed)
    assert evaluation.score_validate_roots(final).identity == evaluation.IdentityScore(1, 1, 2, 0)


def test_rule_ambiguity_scores_only_the_selected_candidate_and_keeps_stage_boundaries(
    tmp_path: Path,
) -> None:
    client = FakeLookupClient(
        _cluster(1, "Jones v. Smith", full_name="Jones v. Smith"),
        _cluster(2, "Bell Atlantic Corporation v. Twombly", full_name="Bell Atlantic Corporation v. Twombly"),
    )
    lookup = reporter_root_lookup_cluster_retrieval(_roots(tmp_path), client=client)
    dockets = reporter_root_lookup_docket_retrieval(lookup, client=client)
    unique_review = reporter_root_lookup_unique_rule_judgment(dockets)
    ambiguous = reporter_root_lookup_ambiguous_rule_judgment(unique_review)
    root = next(root for root in ambiguous.roots if isinstance(root, FullReporterCitation))
    assert root.reporter_exact_ambiguity_resolution is not None
    assert root.reporter_exact_ambiguity_resolution.selected_candidate_index == 1
    assert [(judgment.candidate_index, judgment.result) for judgment in root.case_name_judgments] == [
        (0, MatchResult.MISMATCH),
        (1, MatchResult.MATCH),
    ]

    final = asyncio.run(reporter_root_lookup_unique_llm_judgment(ambiguous))
    final = asyncio.run(reporter_root_lookup_ambiguous_llm_judgment(final))
    restored = Document.model_validate_json(final.model_dump_json())
    assert restored == final
    for stage, scorer_name in SCORERS.items():
        scorer = getattr(evaluation, scorer_name)
        assert scorer(final) == scorer(final.get_stage(stage)) == scorer(restored)
        assert set(scorer(final).metrics) == FIELDS
        rendered = getattr(evaluation, RENDERERS[stage])(scorer(final))
        assert stage in rendered
    assert evaluation.score_reporter_root_lookup_unique_rule_judgment(unique_review).metrics == {
        field: evaluation.Precision(0, 0) for field in FIELDS
    }
    assert evaluation.score_reporter_root_lookup_ambiguous_rule_judgment(final).metrics == {
        field: evaluation.Precision(1, 1) for field in FIELDS
    }
    assert evaluation.score_reporter_root_lookup_ambiguous_llm_judgment(final).metrics == {
        field: evaluation.Precision(0, 0) for field in FIELDS
    }
    assert evaluation.score_reporter_root_lookup_unique_llm_judgment(final).metrics == {
        field: evaluation.Precision(0, 0) for field in FIELDS
    }
    score = evaluation.score_validate_roots(final)
    assert tuple(stage.stage for stage in score.stages) == STAGES
    assert set(score.fields) == FIELDS
    assert set(score.as_dict()) == {"stage_order", "stages", "retrieval_stages", "fields", "identity"}
    assert score.fields == {field: evaluation.FieldScore(1, 1, 2) for field in FIELDS}
    report = evaluation.render_validate_roots(score)
    assert "## Root field judgments through stage 19 (before open search)" in report
    assert "| Field | Precision | Recall |" in report
    assert "identity" not in score.as_dict()["fields"]


@pytest.mark.parametrize(
    "labels",
    (
        {"case_name": "disagrees", "court": "unavailable", "date": "not_stated"},
        {"case_name": "unavailable", "court": "disagrees", "date": "unavailable"},
        {"case_name": "not_stated", "court": "unavailable", "date": "disagrees"},
    ),
)
def test_identity_value_mismatch_takes_precedence_over_unavailable_case_name(
    labels: dict[str, str],
) -> None:
    assert evaluation._identity_value(labels) == "WRONG_IDENTITY"


@pytest.mark.parametrize("case_name", ("unavailable", "not_stated"))
def test_identity_value_needs_a_case_name_match(case_name: str) -> None:
    assert (
        evaluation._identity_value({"case_name": case_name, "court": "agrees", "date": "unavailable"})
        == "UNDETERMINED"
    )


@pytest.mark.parametrize(
    ("court", "date"),
    (("unavailable", "agrees"), ("agrees", "not_stated"), ("unavailable", "unavailable")),
)
def test_identity_value_ignores_unavailable_court_and_date(court: str, date: str) -> None:
    assert (
        evaluation._identity_value({"case_name": "agrees", "court": court, "date": date})
        == "CORRECT_IDENTITY"
    )


@pytest.mark.parametrize("missing_field", ("case_name", "court", "date"))
def test_identity_value_requires_all_three_fields(missing_field: str) -> None:
    labels = {"case_name": "agrees", "court": "agrees", "date": "agrees"}
    del labels[missing_field]
    with pytest.raises(ValueError):
        evaluation._identity_value(labels)


def test_unique_unavailable_is_an_explicit_incorrect_judgment(tmp_path: Path) -> None:
    client = FakeLookupClient(_cluster(1, "Bell Atlantic Corporation v. Twombly"))
    lookup = reporter_root_lookup_cluster_retrieval(_roots(tmp_path), client=client)
    root = next(root for root in lookup.roots if isinstance(root, FullReporterCitation))
    assert root.case_name_judgments == root.court_judgments == root.date_judgments == ()
    dockets = reporter_root_lookup_docket_retrieval(lookup)
    reviewed = reporter_root_lookup_unique_rule_judgment(dockets)
    root = next(root for root in reviewed.roots if isinstance(root, FullReporterCitation))
    assert root.case_name_judgments[-1].result is MatchResult.UNAVAILABLE
    assert evaluation.score_reporter_root_lookup_unique_rule_judgment(reviewed).metrics == {
        "case_name": evaluation.Precision(0, 1),
        "court": evaluation.Precision(1, 1),
        "date": evaluation.Precision(1, 1),
    }

    decision = ReporterUniqueReviewDecision.model_validate(_model_fields())
    final = reporter_root_lookup_ambiguous_rule_judgment(reviewed)
    final = asyncio.run(reporter_root_lookup_unique_llm_judgment(final, reviewer=FakeReviewer(decision)))
    final = asyncio.run(reporter_root_lookup_ambiguous_llm_judgment(final))
    assert evaluation.score_reporter_root_lookup_unique_rule_judgment(
        final
    ) == evaluation.score_reporter_root_lookup_unique_rule_judgment(reviewed)
    assert evaluation.score_reporter_root_lookup_unique_llm_judgment(final).metrics == {
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
    lookup = reporter_root_lookup_cluster_retrieval(_roots(tmp_path), client=client)
    dockets = reporter_root_lookup_docket_retrieval(lookup, client=client)
    reviewed = reporter_root_lookup_unique_rule_judgment(dockets)
    ambiguous = reporter_root_lookup_ambiguous_rule_judgment(reviewed)
    root = next(root for root in ambiguous.roots if isinstance(root, FullReporterCitation))
    assert root.reporter_exact_ambiguity_resolution is not None
    assert root.reporter_exact_ambiguity_resolution.selected_candidate_index is None
    decision = ReporterAmbiguousReviewDecision.model_validate(
        {"selected_candidate_index": 1, **_model_fields()}
    )
    before_review = asyncio.run(reporter_root_lookup_unique_llm_judgment(ambiguous))
    final = asyncio.run(
        reporter_root_lookup_ambiguous_llm_judgment(before_review, reviewer=FakeReviewer(decision))
    )
    assert evaluation.score_reporter_root_lookup_ambiguous_rule_judgment(final).metrics == {
        field: evaluation.Precision(0, 0) for field in FIELDS
    }
    assert evaluation.score_reporter_root_lookup_ambiguous_llm_judgment(final).metrics == {
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
    lookup = reporter_root_lookup_cluster_retrieval(roots, client=client)
    final = _finish_unrouted_stages(lookup)
    reporter = next(root for root in final.roots if isinstance(root, FullReporterCitation))
    assert not reporter.date
    assert reporter.date_judgments == ()
    assert evaluation.score_reporter_root_lookup_unique_rule_judgment(final).metrics[
        "date"
    ] == evaluation.Precision(0, 0)
    assert evaluation.score_validate_roots(final).fields == {
        field: evaluation.FieldScore(1, 1, 2) for field in FIELDS
    }
    assert all(
        not root.identity_judgments for root in final.roots if not isinstance(root, FullReporterCitation)
    )


def test_lookup_miss_does_not_predict_an_absent_citation_field(tmp_path: Path) -> None:
    lookup = reporter_root_lookup_cluster_retrieval(
        _roots(tmp_path, SOURCE_WITHOUT_REPORTER_DATE), client=FakeLookupClient()
    )
    reporter = next(root for root in lookup.roots if isinstance(root, FullReporterCitation))
    assert any(node.stage == "12.1_reporter_root_lookup_cluster_retrieval" for node in reporter.nodes)
    assert not reporter.date
    assert reporter.case_name_judgments == reporter.court_judgments == reporter.date_judgments == ()

    final = _finish_unrouted_stages(lookup)
    assert evaluation.score_reporter_root_lookup_unique_rule_judgment(final).metrics == {
        field: evaluation.Precision(0, 0) for field in FIELDS
    }
    assert evaluation.score_validate_roots(final).fields == {
        field: evaluation.FieldScore(0, 0, 2) for field in FIELDS
    }


def test_unavailable_for_an_absent_reporter_field_agrees_with_not_stated_gold(
    tmp_path: Path,
) -> None:
    client = FakeLookupClient(_cluster(1, "Bell Atlantic Corporation v. Twombly"))
    lookup = reporter_root_lookup_cluster_retrieval(
        _roots(tmp_path, SOURCE_WITHOUT_REPORTER_DATE), client=client
    )
    decision = ReporterUniqueReviewDecision.model_validate(
        {
            **_model_fields(),
            "date": {
                "propose_replacement": False,
                "quote": None,
                "result": "unavailable",
                "reason": "The filing states no date for this reporter root.",
            },
        }
    )
    before_review = reporter_root_lookup_docket_retrieval(lookup)
    before_review = reporter_root_lookup_unique_rule_judgment(before_review)
    before_review = reporter_root_lookup_ambiguous_rule_judgment(before_review)
    reviewed = asyncio.run(
        reporter_root_lookup_unique_llm_judgment(before_review, reviewer=FakeReviewer(decision))
    )
    reviewed = asyncio.run(reporter_root_lookup_ambiguous_llm_judgment(reviewed))
    reporter = next(root for root in reviewed.roots if isinstance(root, FullReporterCitation))
    assert reporter.date_judgments[-1].result is MatchResult.UNAVAILABLE
    assert reporter.date_judgments[-1].reading_index is None
    assert evaluation.score_reporter_root_lookup_unique_llm_judgment(reviewed).metrics["date"] == (
        evaluation.Precision(1, 1)
    )
    assert evaluation.score_validate_roots(reviewed).fields["date"] == evaluation.FieldScore(1, 1, 2)


@pytest.mark.parametrize("invalid_label", ("unavailable", "undetermined"))
def test_gold_rejects_non_identity_labels(tmp_path: Path, invalid_label: str) -> None:
    source_path = _write_source(tmp_path)
    annotation = source_path.parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines()]
    rows[1]["validation"]["identity"]["fields"]["date"]["label"] = invalid_label
    annotation.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete field-level identity gold"):
        evaluation._gold_roots(Document.from_source(source_path))


def test_gold_not_stated_requires_no_source_reading(tmp_path: Path) -> None:
    source_path = _write_source(tmp_path)
    annotation = source_path.parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines()]
    rows[1]["validation"]["identity"]["fields"]["date"]["label"] = "not_stated"
    annotation.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="date gold is not_stated despite a source reading"):
        evaluation._gold_roots(Document.from_source(source_path))


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
    document = _add_docket_review(document, case_name="unavailable", court="unavailable", date="unavailable")
    document = _complete_reporter_stages(document)

    score = evaluation.score_validate_roots(document)
    assert score.fields == {field: evaluation.FieldScore(1, 1, 440) for field in FIELDS}
    assert all(score.fields[field].as_dict()["recall"] == 1 / 440 for field in FIELDS)


def test_missing_stage_checkpoint_raises(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    with pytest.raises(KeyError, match="Stage has not run"):
        evaluation.score_reporter_root_lookup_unique_rule_judgment(roots)
    unvalidated = Document.from_source(roots.source_path)
    with pytest.raises(KeyError, match="Stage has not run"):
        evaluation.score_docket_root_lookup_courtlistener_llm_review(unvalidated)


def _stage23_review(
    tmp_path: Path,
    *,
    verdict: IdentityVerdict | None = IdentityVerdict.CORRECT_IDENTITY,
    fetched: bool = True,
    no_reviewable: bool = False,
    failure: bool = False,
    next_stage: str | None = None,
) -> Document:
    document = _roots(tmp_path, docket_stages=False)
    for stage in evaluation.WORKFLOW_STAGES:
        document = document.complete(stage)
    quote = "Gamma v. Delta, No. 1:24-cv-08705 (S.D.N.Y. 2024)"
    context_quote = "The opinion cites"
    body = f"{context_quote} {quote}."
    evidence = (
        make_body_evidences(
            body_id="opinion-123",
            parent_id="docket-456",
            url="https://example.test/opinion-123",
            issued_on=date(2024, 1, 1),
            date_basis="opinion.date_filed",
            metadata={"id": 123},
            body_text=body,
            locator="1:24-cv-08705",
            source_text=document.text,
        )
        if fetched
        else ()
    )
    for stage in evaluation.LOCATOR_BODY_STAGES[:-1]:
        if stage == evaluation.LOCATOR_BODY_STAGES[0]:
            root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
            recorded = root.record(stage)
            recorded = recorded.with_body_search(
                BodySearch(
                    node_id=recorded.nodes[-1].id,
                    source=BodySource.COURTLISTENER_OPINION,
                    retrospective_date=None,
                    evidence=evidence,
                )
            )
            document = document.replace_citation(recorded)
        document = document.complete(stage)
    root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
    recorded = root.record(evaluation.LOCATOR_BODY_LLM_JUDGMENT)
    if failure:
        review = BodyCorroborationReview(
            node_id=recorded.nodes[-1].id,
            failure_reason="The model review failed.",
        )
    elif verdict is None or not fetched:
        decision = BodyCorroborationDecision.model_validate(
            {
                "source": None,
                "evidence_index": None,
                "citation_quote": None,
                "context_quote": None,
                "treatment": None,
                "filing": None,
                "third_party": None,
                "comparisons": None,
                "identity_verdict": None,
                "reason": (
                    "No fetched third-party body citation is available for comparison."
                    if no_reviewable or not fetched
                    else "No independent citation was selected."
                ),
            }
        )
        review = BodyCorroborationReview(node_id=recorded.nodes[-1].id, decision=decision)
    else:
        court_result = "mismatch" if verdict is IdentityVerdict.WRONG_IDENTITY else "match"
        decision = BodyCorroborationDecision.model_validate(
            {
                "source": BodySource.COURTLISTENER_OPINION.value,
                "evidence_index": 0,
                "citation_quote": quote,
                "context_quote": context_quote,
                "treatment": "cites_as_authority",
                "filing": {
                    "locator": "1:24-cv-08705",
                    "case_name": "Gamma v. Delta",
                    "normalized_case_name": {
                        "kind": "adversarial",
                        "plaintiff": "Gamma",
                        "defendant": "Delta",
                        "subject": None,
                    },
                    "court": "S.D.N.Y.",
                    "date": "2024",
                },
                "third_party": {
                    "locator": "1:24-cv-08705",
                    "case_name": "Gamma v. Delta",
                    "court": "N.D. Tex." if court_result == "mismatch" else "S.D.N.Y.",
                    "date": "2024",
                },
                "comparisons": {
                    field: {
                        "result": court_result if field == "court" else "match",
                        "reason": f"The printed {field} was compared.",
                    }
                    for field in ("locator", "case_name", "court", "date")
                },
                "identity_verdict": verdict.value,
                "reason": "The independent citation supports this assessment.",
            }
        )
        assert len(evidence) == 1
        excerpt = evidence[0].excerpt
        quote_start = excerpt.index(quote)
        context_start = excerpt.index(context_quote)
        review = BodyCorroborationReview(
            node_id=recorded.nodes[-1].id,
            decision=decision,
            grounded_quote=quote,
            quote_span=Span(quote_start, quote_start + len(quote)),
            quote_similarity=100,
            grounded_context=context_quote,
            context_span=Span(context_start, context_start + len(context_quote)),
            context_similarity=100,
        )
    recorded = recorded.with_body_review(review)
    if review.decision is not None and review.decision.source is not None:
        assert verdict is not None
        recorded = recorded.with_identity_judgment(verdict, basis=IdentityBasis.THIRD_PARTY)
    recorded = recorded.with_route(next_stage)
    return document.replace_citation(recorded).complete(evaluation.LOCATOR_BODY_LLM_JUDGMENT)


def _stage27_review(tmp_path: Path, outcome: str) -> Document:
    document = _stage23_review(tmp_path, verdict=None, next_stage="case_name_body_discovery")
    quote = "Gamma v. Delta"
    evidence = make_body_evidences(
        body_id="independent-opinion",
        parent_id=None,
        url=None,
        issued_on=date(2024, 1, 1),
        date_basis=None,
        metadata={},
        body_text=f"Another court cites {quote} as authority.",
        locator="Gamma",
        source_text=document.text,
        anchor_kind="case_name",
    )
    root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
    recorded = root.record(evaluation.INTENDED_CASE_STAGES[0])
    recorded = recorded.with_field_body_search(
        FieldBodySearch(
            node_id=recorded.nodes[-1].id,
            source=BodySource.COURTLISTENER_OPINION,
            retrospective_date=None,
            query_name="Gamma",
            evidence=evidence,
        )
    )
    document = document.replace_citation(recorded).complete(evaluation.INTENDED_CASE_STAGES[0])
    for stage in evaluation.INTENDED_CASE_STAGES[1:-1]:
        document = document.complete(stage)
    root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
    recorded = root.record(evaluation.INTENDED_CASE_LLM_SELECTION)
    if outcome.startswith("selected_"):
        decision = IntendedCaseDecision(
            source=BodySource.COURTLISTENER_OPINION,
            evidence_index=0,
            citation_quote=quote,
            case_name=quote,
            locator=None,
            court=None,
            date=None,
            confidence=IntendedCaseConfidence(outcome.removeprefix("selected_")),
            reason="The independent citation identifies a possible intended case.",
        )
        review = IntendedCaseReview(
            node_id=recorded.nodes[-1].id,
            decision=decision,
            grounded_quote=quote,
            quote_span=Span(evidence[0].excerpt.index(quote), evidence[0].excerpt.index(quote) + len(quote)),
            quote_similarity=100,
        )
        route = "intended_case_resolution"
    elif outcome == "declined":
        review = IntendedCaseReview(
            node_id=recorded.nodes[-1].id,
            decision=IntendedCaseDecision(
                source=None,
                evidence_index=None,
                citation_quote=None,
                case_name=None,
                locator=None,
                court=None,
                date=None,
                confidence=None,
                reason="No independent citation identifies the intended case.",
            ),
        )
        route = "open_web_search"
    else:
        assert outcome == "review_failure"
        review = IntendedCaseReview(
            node_id=recorded.nodes[-1].id,
            failure_reason="Review exhausted its attempts.",
        )
        route = "intended_case_review_retry"
    recorded = recorded.with_intended_case_review(review).with_route(route)
    return document.replace_citation(recorded).complete(evaluation.INTENDED_CASE_LLM_SELECTION)


def test_stage23_report_counts_issued_verdicts(tmp_path: Path) -> None:
    document = _stage23_review(tmp_path, verdict=IdentityVerdict.WRONG_IDENTITY)
    detail = evaluation.score_locator_body_llm_judgment(document)
    assert detail.verdict_counts == {"wrong_identity": 1}
    workflow = evaluation.score_validate_roots(document)
    assert workflow.body_review == detail
    assert workflow.as_dict()["body_review"] == {
        "stage": evaluation.LOCATOR_BODY_LLM_JUDGMENT,
        "verdict_counts": {"wrong_identity": 1},
    }
    report = evaluation.render_validate_roots(workflow)
    assert f"## {evaluation.LOCATOR_BODY_LLM_JUDGMENT}" in report
    assert "| wrong_identity | 1 |" in report
    assert "opinion-123" not in report
    numbered_headings = tuple(
        line.removeprefix("## ")
        for line in report.splitlines()
        if line.startswith("## ") and line[3:4].isdigit()
    )
    assert numbered_headings == (*evaluation.WORKFLOW_STAGES, *evaluation.LOCATOR_BODY_STAGES)
    assert workflow.as_dict()["stage_order"] == list(numbered_headings)


@pytest.mark.parametrize("outcome", ("selected_likely", "selected_possible", "declined", "review_failure"))
def test_stage27_report_lists_all_stages_without_rescoring_identity(tmp_path: Path, outcome: str) -> None:
    document = _stage27_review(tmp_path, outcome)
    before = evaluation.score_validate_roots(document.get_stage(evaluation.LOCATOR_BODY_LLM_JUDGMENT))
    score = evaluation.score_validate_roots(document)
    assert score.checkpoint == evaluation.INTENDED_CASE_LLM_SELECTION
    assert score.fields == before.fields
    assert score.identity == before.identity
    assert score.identity_with_partial == before.identity_with_partial
    assert score.body_review == before.body_review
    assert score.intended_case_outcomes == {outcome: 1}
    assert score.stage_order == (
        *evaluation.WORKFLOW_STAGES,
        *evaluation.LOCATOR_BODY_STAGES,
        *evaluation.INTENDED_CASE_STAGES,
    )
    assert score.as_dict()["intended_case_review"] == {
        "stage": evaluation.INTENDED_CASE_LLM_SELECTION,
        "outcome_counts": {outcome: 1},
    }
    report = evaluation.render_validate_roots(score)
    numbered_headings = tuple(
        line.removeprefix("## ")
        for line in report.splitlines()
        if line.startswith("## ") and line[3:4].isdigit()
    )
    assert numbered_headings == score.stage_order
    assert f"| {outcome} | 1 |" in report
    assert "Candidate outcomes only; the annotations do not label intended-case candidates." in report
    assert report.count("Citations with records / citations queried") == 10


def test_stage27_outcome_counts_combine_without_accuracy_metrics(tmp_path: Path) -> None:
    outcomes = ("selected_likely", "selected_possible", "declined", "review_failure")
    scores = [
        evaluation.score_validate_roots(_stage27_review(tmp_path / outcome, outcome)) for outcome in outcomes
    ]
    combined = scores[0]
    for score in scores[1:]:
        combined += score
    assert combined.intended_case_outcomes == dict.fromkeys(outcomes, 1)
    assert set(combined.as_dict()["intended_case_review"]) == {"stage", "outcome_counts"}


def test_validate_roots_rejects_incomplete_intended_case_workflow(tmp_path: Path) -> None:
    document = _stage23_review(tmp_path)
    document = document.complete(evaluation.INTENDED_CASE_STAGES[0])
    with pytest.raises(ValueError, match="Incomplete intended-case stages in validate_roots"):
        evaluation.score_validate_roots(document)


@pytest.mark.parametrize(
    "verdict",
    (IdentityVerdict.PARTIALLY_CORROBORATED, IdentityVerdict.UNDETERMINED),
)
def test_stage23_qualified_and_undetermined_verdicts_have_distinct_identity_scores(
    tmp_path: Path, verdict: IdentityVerdict
) -> None:
    document = _stage23_review(tmp_path, verdict=verdict, next_stage="case_name_body_discovery")
    detail = evaluation.score_locator_body_llm_judgment(document)
    assert detail.verdict_counts == {verdict.value: 1}
    assert "| " + verdict.value + " | 1 |" in evaluation.render_locator_body_llm_judgment(detail)
    root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
    assert root.next_stage == "case_name_body_discovery"
    workflow = evaluation.score_validate_roots(document)
    assert workflow.identity == evaluation.IdentityScore(0, 0, 2, 1)
    assert workflow.identity_with_partial == (
        evaluation.IdentityScore(1, 1, 2, 0)
        if verdict is IdentityVerdict.PARTIALLY_CORROBORATED
        else evaluation.IdentityScore(0, 0, 2, 1)
    )
    if verdict is IdentityVerdict.PARTIALLY_CORROBORATED:
        report = evaluation.render_validate_roots(workflow)
        assert (
            "| — (including partial: 1/1 = 100.0%) | 0/2 (0.0%) (including partial: 1/2 = 50.0%) | 1 |"
        ) in report


def test_partial_verdict_lowers_broader_precision_on_wrong_identity_gold(tmp_path: Path) -> None:
    document = _stage23_review(
        tmp_path,
        verdict=IdentityVerdict.PARTIALLY_CORROBORATED,
        next_stage="case_name_body_discovery",
    )
    annotation = tmp_path / "primary" / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines()]
    rows[-1]["validation"]["identity"]["label"] = "WRONG_IDENTITY"
    annotation.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    score = evaluation.score_validate_roots(document)
    assert score.identity == evaluation.IdentityScore(0, 0, 2, 1)
    assert score.identity_with_partial == evaluation.IdentityScore(0, 1, 2, 0)


@pytest.mark.parametrize(
    ("fetched", "failure"),
    (
        (False, False),
        (True, False),
        (True, True),
    ),
)
def test_stage23_routes_without_issuing_judgment(tmp_path: Path, fetched: bool, failure: bool) -> None:
    document = _stage23_review(
        tmp_path,
        verdict=None,
        fetched=fetched,
        failure=failure,
        next_stage="case_name_body_discovery",
    )
    detail = evaluation.score_locator_body_llm_judgment(document)
    assert detail.verdict_counts == {}
    root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
    assert root.next_stage == "case_name_body_discovery"
    assert root.body_reviews[-1].failure_reason == ("The model review failed." if failure else None)
    assert evaluation.score_validate_roots(document).body_review == detail


def test_stage23_no_reviewable_decision_allows_fetched_but_filtered_excerpts(tmp_path: Path) -> None:
    document = _stage23_review(
        tmp_path,
        verdict=None,
        no_reviewable=True,
        next_stage="case_name_body_discovery",
    )
    root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
    assert root.body_searches[0].evidence
    assert root.body_reviews[-1].decision.source is None
    assert evaluation.score_locator_body_llm_judgment(document).verdict_counts == {}


def test_stage23_evaluation_merges_verdict_counts_and_ignores_later_routes(tmp_path: Path) -> None:
    first = _stage23_review(tmp_path / "first", next_stage=None)
    second = _stage23_review(tmp_path / "second", verdict=None, next_stage="case_name_body_discovery")
    recorded = next(root for root in first.roots if isinstance(root, FullDocketCitation)).record(
        "24_followup"
    )
    first_later = first.replace_citation(recorded.with_route("unrelated_followup")).complete("24_followup")
    first_score = evaluation.score_locator_body_llm_judgment(first)
    assert evaluation.score_locator_body_llm_judgment(first_later) == first_score
    combined = first_score + evaluation.score_locator_body_llm_judgment(second)
    assert combined.verdict_counts == {"correct_identity": 1}
    assert combined.as_dict() == {
        "stage": evaluation.LOCATOR_BODY_LLM_JUDGMENT,
        "verdict_counts": {"correct_identity": 1},
    }


@pytest.mark.parametrize(
    ("missing_log", "error"),
    (
        ("identity_judgments", "matching identity judgment"),
        ("routes", "exactly one route decision"),
    ),
)
def test_stage23_evaluation_rejects_incomplete_selected_history(
    tmp_path: Path, missing_log: str, error: str
) -> None:
    document = _stage23_review(tmp_path)
    root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
    review_node = root.body_reviews[-1].node_id
    payload = document.model_dump(mode="python")
    citation = next(item for item in payload["citations"] if item["id"] == root.id)
    citation[missing_log] = tuple(item for item in citation[missing_log] if item["node_id"] != review_node)
    incomplete = Document.model_validate(payload)
    with pytest.raises(ValueError, match=error):
        evaluation.score_locator_body_llm_judgment(incomplete)


def test_stage23_evaluation_rejects_identity_judgment_on_a_route_only_review(tmp_path: Path) -> None:
    document = _stage23_review(tmp_path, verdict=None, next_stage="case_name_body_discovery")
    root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
    payload = document.model_dump(mode="python")
    citation = next(item for item in payload["citations"] if item["id"] == root.id)
    citation["identity_judgments"] = (
        *citation["identity_judgments"],
        {"node_id": root.body_reviews[-1].node_id, "verdict": "correct_identity", "basis": "third_party"},
    )
    invalid = Document.model_validate(payload)
    with pytest.raises(ValueError, match="cannot issue an identity judgment"):
        evaluation.score_locator_body_llm_judgment(invalid)


def test_reporter_cluster_retrieval_counts_only_queried_citations(tmp_path: Path) -> None:
    found = reporter_root_lookup_cluster_retrieval(
        _roots(tmp_path / "found", docket_stages=False),
        client=FakeLookupClient(_cluster(1, "Bell Atlantic Corporation v. Twombly")),
    )
    missed = reporter_root_lookup_cluster_retrieval(
        _roots(tmp_path / "missed", docket_stages=False), client=FakeLookupClient()
    )
    stage = evaluation.REPORTER_ROOT_LOOKUP_CLUSTER_RETRIEVAL
    scorer = evaluation.score_reporter_root_lookup_cluster_retrieval
    found_score = scorer(found)
    missed_score = scorer(missed)
    assert found_score == evaluation.RetrievalScore(stage, 1, 1)
    assert missed_score == evaluation.RetrievalScore(stage, 0, 1)
    assert (found_score + missed_score).as_dict()["record_coverage"]["rate"] == 0.5
    assert scorer(found.complete("later")) == evaluation.RetrievalScore(stage, 1, 1)


def test_reporter_docket_retrieval_counts_saved_requests_even_when_empty(tmp_path: Path) -> None:
    from mellea_lrc.providers.courtlistener import CourtListenerDocket

    class DocketClient:
        def __init__(self, response: CourtListenerDocket | None) -> None:
            self.response = response

        def get_docket(self, docket_id: str) -> CourtListenerDocket | None:
            assert docket_id == "10"
            return self.response

    candidate = {**_cluster(1, "Bell Atlantic Corporation v. Twombly"), "court_id": None, "docketId": 10}
    stage = evaluation.REPORTER_ROOT_LOOKUP_DOCKET_RETRIEVAL
    scorer = evaluation.score_reporter_root_lookup_docket_retrieval
    for label, response, expected in (
        ("empty", None, 0),
        ("found", CourtListenerDocket.model_validate({"id": 10, "court_id": "scotus"}), 1),
    ):
        lookup = reporter_root_lookup_cluster_retrieval(
            _roots(tmp_path / label, docket_stages=False), client=FakeLookupClient(candidate)
        )
        retrieved = reporter_root_lookup_docket_retrieval(lookup, client=DocketClient(response))
        assert scorer(retrieved) == evaluation.RetrievalScore(stage, expected, 1)
    no_link = reporter_root_lookup_cluster_retrieval(
        _roots(tmp_path / "no_link", docket_stages=False),
        client=FakeLookupClient(_cluster(1, "Bell Atlantic Corporation v. Twombly")),
    )
    no_link = reporter_root_lookup_docket_retrieval(no_link)
    assert scorer(no_link) == evaluation.RetrievalScore(stage, 0, 0)


@pytest.mark.parametrize(
    "stage",
    (
        evaluation.DOCKET_ROOT_LOOKUP_COURTLISTENER_RETRIEVAL,
        evaluation.DOCKET_ROOT_LOOKUP_GOVINFO_RETRIEVAL,
    ),
)
def test_docket_retrieval_counts_saved_candidates_and_search_attempts(tmp_path: Path, stage: str) -> None:
    scorer = getattr(evaluation, RETRIEVAL_SCORERS[stage])
    found = (
        _reviewed_docket(tmp_path / "found")
        if stage == evaluation.DOCKET_ROOT_LOOKUP_COURTLISTENER_RETRIEVAL
        else _add_govinfo_review(_reviewed_docket(tmp_path / "found"))
    )
    assert scorer(found) == evaluation.RetrievalScore(stage, 1, 1)

    empty = _roots(tmp_path / "empty", docket_stages=False)
    root = next(root for root in empty.roots if isinstance(root, FullDocketCitation))
    recorded = root.record(stage)
    if stage == evaluation.DOCKET_ROOT_LOOKUP_COURTLISTENER_RETRIEVAL:
        recorded = recorded.with_docket_lookup(
            DocketLookup(
                node_id=recorded.nodes[-1].id,
                attempts=(DocketLookupAttempt(source_type="d", query="1:24-cv-08705"),),
            )
        )
    else:
        recorded = recorded.with_govinfo_docket_lookup(
            GovInfoDocketLookup(
                node_id=recorded.nodes[-1].id,
                attempts=(GovInfoLookupAttempt(query="1:24-cv-08705"),),
            )
        )
    empty = empty.replace_citation(recorded).complete(stage)
    assert scorer(empty) == evaluation.RetrievalScore(stage, 0, 1)
    skipped = _roots(tmp_path / "skipped", docket_stages=False).complete(stage)
    assert scorer(skipped) == evaluation.RetrievalScore(stage, 0, 0)


@pytest.mark.parametrize(
    ("stage", "source", "field_stage"),
    (
        (evaluation.LOCATOR_BODY_STAGES[0], BodySource.COURTLISTENER_OPINION, False),
        (evaluation.LOCATOR_BODY_STAGES[1], BodySource.COURTLISTENER_RECAP, False),
        (evaluation.LOCATOR_BODY_STAGES[2], BodySource.GOVINFO_OPINION, False),
        (evaluation.INTENDED_CASE_STAGES[0], BodySource.COURTLISTENER_OPINION, True),
        (evaluation.INTENDED_CASE_STAGES[1], BodySource.COURTLISTENER_RECAP, True),
        (evaluation.INTENDED_CASE_STAGES[2], BodySource.GOVINFO_OPINION, True),
    ),
)
def test_body_retrieval_counts_queried_citations_with_reviewable_excerpts(
    tmp_path: Path, stage: str, source: BodySource, field_stage: bool
) -> None:
    scorer = getattr(evaluation, RETRIEVAL_SCORERS[stage])

    def saved_search(label: str, *, queried: bool, found: bool) -> Document:
        document = _roots(tmp_path / label, docket_stages=False)
        root = next(root for root in document.roots if isinstance(root, FullDocketCitation))
        recorded = root.record(stage)
        anchor = "Gamma" if field_stage else "1:24-cv-08705"
        evidence = (
            make_body_evidences(
                body_id="other-opinion",
                parent_id=None,
                url=None,
                issued_on=None,
                date_basis=None,
                metadata={},
                body_text=f"Another court discusses {anchor} in this separate opinion.",
                locator=anchor,
                source_text=document.text,
                anchor_kind="case_name" if field_stage else "locator",
            )
            if found
            else ()
        )
        attempts = (BodySearchAttempt(query=f'"{anchor}"'),) if queried else ()
        if field_stage:
            recorded = recorded.with_field_body_search(
                FieldBodySearch(
                    node_id=recorded.nodes[-1].id,
                    source=source,
                    retrospective_date=None,
                    query_name=anchor,
                    attempts=attempts,
                    evidence=evidence,
                )
            )
        else:
            recorded = recorded.with_body_search(
                BodySearch(
                    node_id=recorded.nodes[-1].id,
                    source=source,
                    retrospective_date=None,
                    attempts=attempts,
                    evidence=evidence,
                )
            )
        return document.replace_citation(recorded).complete(stage)

    found = saved_search("found", queried=True, found=True)
    missed = saved_search("missed", queried=True, found=False)
    skipped = saved_search("skipped", queried=False, found=False)
    assert scorer(found) == evaluation.RetrievalScore(stage, 1, 1)
    assert scorer(missed) == evaluation.RetrievalScore(stage, 0, 1)
    assert scorer(skipped) == evaluation.RetrievalScore(stage, 0, 0)


def test_workflow_retrieval_scores_render_one_metric_per_stage(tmp_path: Path) -> None:
    lookup = reporter_root_lookup_cluster_retrieval(
        _roots(tmp_path),
        client=FakeLookupClient(_cluster(1, "Bell Atlantic Corporation v. Twombly")),
    )
    document = reporter_root_lookup_docket_retrieval(lookup)
    document = reporter_root_lookup_unique_rule_judgment(document)
    document = reporter_root_lookup_ambiguous_rule_judgment(document)
    document = document.complete(STAGES[2]).complete(STAGES[3])
    score = evaluation.score_validate_roots(document)
    assert tuple(item.stage for item in score.retrieval_stages) == evaluation.RETRIEVAL_STAGES[:4]
    assert score.retrieval_stages[0] == evaluation.RetrievalScore(
        evaluation.REPORTER_ROOT_LOOKUP_CLUSTER_RETRIEVAL, 1, 1
    )
    assert score.as_dict()["retrieval_stages"][0]["record_coverage"] == {
        "citations_with_records": 1,
        "citations_queried": 1,
        "rate": 1.0,
    }
    report = evaluation.render_validate_roots(score)
    assert report.count("Citations with records / citations queried") == 4
    assert "| Citations with records / citations queried | 1/1 (100.0%) |" in report
    assert "| Citations with records / citations queried | 0/0 |" in report
