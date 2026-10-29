"""Provider usage observations follow native histories, independent of accuracy."""

from __future__ import annotations

import json

import pytest

from evaluations.llm_usage import (
    LlmUsageReport,
    ReviewTotals,
    UsageTotals,
    render_llm_usage,
    score_llm_usage_run,
    summarize_llm_usage,
)
from mellea_lrc.llm.config import PROFILE_OPTION
from mellea_lrc.model import Document, FullDocketCitation, Span
from mellea_lrc.model.citations.docket_root_llm_reassignment import DocketRootReview
from mellea_lrc.model.ivr import (
    SCHEMA_REQUIREMENT_DESCRIPTION,
    IvrAttempt,
    IvrFailureKind,
    IvrRequirementAttempt,
    IvrRun,
    IvrTransportError,
)
from mellea_lrc.model.site_review import SiteReview

DISCOVERY = "grow_roots.discover_root_locators.find_docket_locators"
HUNTING = "grow_roots.discover_root_locators.hunt_docket_locators"
REASSIGNMENT = "grow_roots.form_root_groups.reassign_docket_roots_llm"
BODY_REVIEW = "validate_roots_citation_identity.corroborate_locator_bodies.review_locator_bodies_llm"


def _run(
    *usages, model="example-model", endpoint="https://example.test/v1", success=False, elapsed_seconds=None
) -> IvrRun:
    return IvrRun(
        success=success,
        selected_attempt=len(usages) - 1,
        attempts=tuple(
            IvrAttempt(output="{}", requirements=(), response={"usage": usage} if usage is not None else None)
            for usage in usages
        ),
        backend="OpenAIBackend",
        model=model,
        model_options={PROFILE_OPTION: {"endpoint": endpoint}},
        instruction="Review this site.",
        prefix=None,
        grounding_context={},
        user_variables={},
        output_schema=None,
        elapsed_seconds=elapsed_seconds,
    )


def _usage(input_tokens, output_tokens, *, cached=None, reasoning=None):
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "prompt_tokens_details": None if cached is None else {"cached_tokens": cached},
        "completion_tokens_details": None if reasoning is None else {"reasoning_tokens": reasoning},
    }


def _site(document, run, *, substage=HUNTING, outcome="declined"):
    return document.add_site_review(
        SiteReview(
            substage=substage,
            candidate_span=Span(start=0, end=3),
            candidate_text=document.text[:3],
            outcome=outcome,
            reason="A retained site review.",
            ivr=run,
        )
    )


def _dockets() -> Document:
    document = Document.from_source("No. 24-123; No. 24-124")
    for index, quote in enumerate(("No. 24-123", "No. 24-124")):
        start = document.text.index(quote)
        document = document.add_citation(
            FullDocketCitation.from_locator(
                citation_id=f"docket-{index}",
                substage=DISCOVERY,
                source=document.text,
                span=Span(start=start, end=start + len(quote)),
                number_span=Span(start=start + 4, end=start + len(quote)),
            )
        )
    return document.complete_substage(DISCOVERY)


def test_repairs_and_failed_reviews_count_every_attempt_once():
    document = _dockets()
    citation = document.citations[0].record(REASSIGNMENT)
    citation = citation.with_docket_root_review(
        DocketRootReview(
            node_id=citation.nodes[-1].id,
            candidate_ids=tuple(item.id for item in document.citations),
            ivr=_run(_usage(100, 20), _usage(200, 30, reasoning=10), None),
            failure_reason="The final attempt timed out.",
        )
    )
    document = document.replace_citation(citation).complete_substage(REASSIGNMENT)

    report = summarize_llm_usage(document)

    assert len(report.rows) == 1
    row = report.rows[0]
    assert (row.substage, row.model, row.endpoint) == (
        REASSIGNMENT,
        "example-model",
        "https://example.test/v1",
    )
    assert report.total.attempts == 3
    assert report.total.usage_reported_attempts == 2
    assert report.total.input_tokens == 300
    assert report.total.output_tokens == 50
    assert report.total.total_tokens == 350
    assert report.total.reasoning_output_tokens == 10
    assert report.total.cached_input_tokens is None
    assert report.total.cache_rate is None


def test_cache_rate_uses_only_attempts_with_both_counters():
    document = _site(
        Document.from_source("No. 24-123"),
        _run(
            _usage(100, 10, cached=60),
            _usage(900, 10),
            _usage(300, 10, cached=0),
            {"prompt_tokens_details": {"cached_tokens": 500}},
            None,
        ),
    )

    total = summarize_llm_usage(document).total

    assert total.input_tokens == 1300
    assert total.cached_input_tokens == 560
    assert total.cache_rate_input_tokens == 400
    assert total.cache_rate_cached_input_tokens == 60
    assert total.cache_reported_attempts == 3
    assert total.cache_rate_attempts == 2
    assert total.usage_reported_attempts == 4
    assert total.attempts == 5
    assert total.cache_rate == 0.15


def test_reported_zero_is_preserved_including_zero_cache_rate():
    document = _site(Document.from_source("No. 24-123"), _run(_usage(100, 0, cached=0, reasoning=0)))
    report = summarize_llm_usage(document)

    assert report.total.output_tokens == 0
    assert report.total.cached_input_tokens == 0
    assert report.total.reasoning_output_tokens == 0
    assert report.total.cache_rate == 0.0
    rendered = render_llm_usage(report)
    assert "| 100 | 0 | 0 | 1/1 | 0.0% |" in rendered


def test_zero_input_does_not_invent_cache_rate():
    total = UsageTotals.from_attempt(_run(_usage(0, 0, cached=0)).attempts[0])
    assert total.input_tokens == 0
    assert total.cache_rate is None


def test_cached_only_counter_is_reported_but_cannot_define_cache_rate():
    total = UsageTotals.from_attempt(_run({"prompt_tokens_details": {"cached_tokens": 40}}).attempts[0])

    assert total.cached_input_tokens == 40
    assert total.cache_reported_attempts == 1
    assert total.cache_rate_attempts == 0
    assert total.cache_rate_input_tokens is None
    assert total.cache_rate is None


@pytest.mark.parametrize("outcome", ("declined", "failed"))
def test_site_outcomes_keep_usage_even_when_no_citation_exists(outcome):
    document = _site(Document.from_source("No. 24-123"), _run(_usage(10, 2)), outcome=outcome)
    report = summarize_llm_usage(document)
    assert report.total.attempts == 1
    assert report.total.total_tokens == 12
    assert not document.citations


def test_accepted_site_is_counted_once_with_its_created_citation():
    document = Document.from_source("No. 24-123")
    span = Span(start=0, end=len(document.text))
    document = document.add_citation(
        FullDocketCitation.from_locator(
            citation_id="found-docket",
            substage=HUNTING,
            source=document.text,
            span=span,
            number_span=Span(start=4, end=span.end),
        )
    )
    document = document.add_site_review(
        SiteReview(
            substage=HUNTING,
            candidate_span=span,
            candidate_text=document.text,
            outcome="accepted",
            reason="A source-grounded docket number.",
            citation_id="found-docket",
            ivr=_run(_usage(10, 2)),
        )
    )

    report = summarize_llm_usage(document)

    assert report.total.attempts == 1
    assert report.total.total_tokens == 12


def test_partial_usage_does_not_estimate_missing_totals():
    document = _site(
        Document.from_source("No. 24-123"),
        _run(
            {"completion_tokens": 12, "prompt_tokens_details": {"cache_write_tokens": 20}},
            {"completion_tokens": 0},
        ),
    )
    total = summarize_llm_usage(document).total

    assert total.attempts == 2
    assert total.usage_reported_attempts == 2
    assert total.input_tokens is None
    assert total.output_tokens == 12
    assert total.total_tokens is None
    assert total.cache_write_input_tokens == 20
    assert total.cached_input_tokens is None


def test_workflow_filter_and_addition_keep_model_endpoint_groups_in_catalog_order():
    first = _site(Document.from_source("No. 24-123"), _run(_usage(10, 2)), substage=BODY_REVIEW)
    first = first.complete_substage(BODY_REVIEW)
    first = _site(first, _run(_usage(30, 4)), substage=HUNTING)
    second = _site(
        Document.from_source("No. 24-124"),
        _run(_usage(20, 3), endpoint="https://other.test/v1"),
    )

    report = summarize_llm_usage(first) + summarize_llm_usage(second)
    roots = report.for_workflow("grow_roots")

    assert [row.substage for row in report.rows] == [HUNTING, HUNTING, BODY_REVIEW]
    assert len(roots.rows) == 2
    assert roots.total.input_tokens == 50
    assert roots.total.output_tokens == 7
    assert isinstance(roots.total.attempts, int)
    assert isinstance(roots.total.usage_reported_attempts, int)
    assert isinstance(roots.as_dict()["total"]["input_tokens"], int)
    assert report.for_workflow("validate_roots_citation_identity").total.input_tokens == 10
    assert report.for_workflow("grow_leaves").rows == ()
    with pytest.raises(KeyError):
        report.for_workflow("invented_workflow")
    rendered = render_llm_usage(roots)
    assert "https://example.test/v1" in rendered
    assert "https://other.test/v1" in rendered


def test_later_native_document_includes_previous_usage_once_and_checkpoint_scopes_history():
    document = _dockets()
    document = _site(document, _run(_usage(100, 20))).complete_substage(HUNTING)
    citation = document.citations[0].record(REASSIGNMENT)
    citation = citation.with_docket_root_review(
        DocketRootReview(
            node_id=citation.nodes[-1].id,
            candidate_ids=tuple(item.id for item in document.citations),
            ivr=_run(_usage(200, 30)),
            failure_reason="The generated decision was not accepted.",
        )
    )
    document = document.replace_citation(citation).complete_substage(REASSIGNMENT)
    document = Document.model_validate_json(document.model_dump_json())

    early = summarize_llm_usage(document.get_substage(HUNTING))
    final = summarize_llm_usage(document)

    assert early.total.input_tokens == 100
    assert final.total.input_tokens == 300
    assert final.total.attempts == 2
    assert [row.substage for row in final.rows] == [HUNTING, REASSIGNMENT]


def test_missing_usage_and_empty_report_stay_unknown_in_json_and_markdown():
    report = summarize_llm_usage(_site(Document.from_source("No. 24-123"), _run(None)))
    serialized = report.as_dict()

    assert serialized["total"]["attempts"] == 1
    assert serialized["total"]["usage_reported_attempts"] == 0
    assert serialized["total"]["input_tokens"] is None
    assert serialized["total"]["cache_rate"] is None
    rendered = render_llm_usage(report)
    assert "| 0/1 | Unavailable | Unavailable | Unavailable | 0/1 | Unavailable |" in rendered
    assert "reported subtotals" in rendered
    assert "already included" in rendered
    empty = LlmUsageReport()
    assert (empty + report).as_dict() == report.as_dict()
    assert empty.total.input_tokens is None
    assert empty.total.attempts == 0


def _attempt(
    *,
    output="{}",
    usage=None,
    model=None,
    failed_requirement=None,
    finish_reason="stop",
    transport_error=None,
):
    return IvrAttempt(
        output=output,
        requirements=()
        if failed_requirement is None
        else (
            IvrRequirementAttempt(
                description=failed_requirement, passed=False, reason="Failed check.", score=None
            ),
        ),
        response=None
        if transport_error is not None
        else {
            "model": model,
            "usage": usage,
            "choices": [{"finish_reason": finish_reason}],
        },
        transport_error=transport_error,
    )


def test_mixed_review_outcomes_repair_recovery_events_and_unknown_causes():
    first_pass = _run(_usage(100, 10, cached=0), success=True, elapsed_seconds=0.5)
    repaired = _run(None, None, success=True, elapsed_seconds=2.5).model_copy(
        update={
            "attempts": (
                _attempt(
                    output='{"broken":',
                    usage=_usage(50, 20, cached=10),
                    failed_requirement=SCHEMA_REQUIREMENT_DESCRIPTION,
                    finish_reason="length",
                ),
                _attempt(usage=_usage(70, 15, cached=0)),
            ),
        }
    )
    failed = _run(None, None, success=False).model_copy(
        update={
            "attempts": (
                _attempt(usage={"completion_tokens": 12}, failed_requirement="Quote must be grounded."),
                _attempt(
                    output="",
                    transport_error=IvrTransportError(
                        kind=IvrFailureKind.HTTP_TIMEOUT,
                        error_type="APITimeoutError",
                        message="Timed out.",
                    ),
                ),
            ),
        }
    )
    unknown = _run(None)
    documents = [
        _site(Document.from_source("No. 24-123"), run) for run in (first_pass, repaired, failed, unknown)
    ]
    report = sum((summarize_llm_usage(document) for document in documents), LlmUsageReport())
    reviews, tokens = report.review_total, report.total

    assert reviews.runs == 4
    assert reviews.successful_runs == reviews.failed_runs == 2
    assert reviews.failure_rate == 0.5
    assert reviews.first_attempt_successful_runs == 1
    assert reviews.first_attempt_success_rate == 0.25
    assert reviews.repaired_runs == 2
    assert reviews.repair_rate == 0.5
    assert reviews.recovered_after_repair_runs == 1
    assert reviews.recovery_rate == 0.5
    assert reviews.repair_attempts == 2
    assert reviews.attempt_failure_events == {
        "invalid_json": 1,
        "output_truncated": 1,
        "requirement_validation": 1,
        "http_timeout": 1,
    }
    assert reviews.attempts_with_failure_events == 3
    assert reviews.final_failure_events == {"http_timeout": 1}
    assert reviews.unclassified_failed_runs == 1
    assert reviews.failed_requirements == {
        SCHEMA_REQUIREMENT_DESCRIPTION: 1,
        "Quote must be grounded.": 1,
    }
    assert reviews.elapsed_seconds == 3.0
    assert reviews.elapsed_reported_runs == 2
    assert tokens.attempts == 6
    assert tokens.input_tokens == 220
    assert tokens.output_tokens == 57
    assert tokens.total_tokens == 265  # Partial-output counter is not silently added to a provider total.
    assert tokens.counter_coverage == {
        "input_tokens": 3,
        "output_tokens": 4,
        "total_tokens": 3,
        "cached_input_tokens": 3,
        "cache_write_input_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    assert (tokens.cache_hit_attempts, tokens.cache_miss_attempts, tokens.cache_unreported_attempts) == (
        1,
        2,
        3,
    )
    assert tokens.cache_hit_rate == 1 / 3
    assert tokens.cache_rate == 10 / 220
    rendered = render_llm_usage(report)
    assert "Categories overlap" in rendered
    assert "Unclassified terminal failure" in rendered
    assert "2/4 | 3.000" in rendered


@pytest.mark.parametrize("status,event", ((429, "http_429"), (500, "http_5xx"), (503, "http_5xx")))
def test_http_status_subsets_are_observed_events_without_fabricated_json_or_usage(status, event):
    run = _run(None).model_copy(
        update={
            "attempts": (
                _attempt(
                    output="",
                    transport_error=IvrTransportError(
                        kind=IvrFailureKind.HTTP_STATUS,
                        error_type="APIStatusError",
                        message="Provider failure.",
                        status_code=status,
                    ),
                ),
            )
        }
    )
    report = summarize_llm_usage(_site(Document.from_source("No. 24-123"), run))

    assert report.review_total.failed_runs == 1
    assert report.review_total.attempt_failure_events == {"http_status": 1, event: 1}
    assert report.review_total.final_failure_events == {"http_status": 1, event: 1}
    assert report.review_total.unclassified_failed_runs == 0
    assert report.total.usage_reported_attempts == 0
    assert report.total.input_tokens is None


def test_requested_and_returned_models_keep_per_attempt_subtotals_configuration_and_unknowns():
    run = _run(None, None, None, model="glm-5", success=True).model_copy(
        update={
            "attempts": (
                _attempt(usage=_usage(100, 20, cached=10), model="Inferact/GLM-5.3-NVFP4"),
                _attempt(usage=_usage(200, 30, cached=0), model="provider/GLM-5"),
                _attempt(usage=_usage(300, 40)),
            ),
            "model_options": {
                PROFILE_OPTION: {"name": "NRP_GLM", "endpoint": "https://example.test/v1", "max_attempts": 3},
                "temperature": 0,
                "max_tokens": 8000,
            },
        }
    )
    report = summarize_llm_usage(_site(Document.from_source("No. 24-123"), run))
    doubled = report + report
    row = doubled.rows[0]

    assert row.model == "glm-5"
    assert [(item.model, item.totals.attempts, item.totals.input_tokens) for item in row.returned_models] == [
        (None, 2, 600),
        ("Inferact/GLM-5.3-NVFP4", 2, 200),
        ("provider/GLM-5", 2, 400),
    ]
    assert sum(item.totals.attempts for item in row.returned_models) == doubled.total.attempts == 6
    assert len(row.configurations) == 1
    serialized = doubled.as_dict()
    assert serialized["rows"][0]["returned_models"][1]["model"] == "Inferact/GLM-5.3-NVFP4"
    assert serialized["rows"][0]["configurations"][0]["model_options"][PROFILE_OPTION]["name"] == "NRP_GLM"
    rendered = render_llm_usage(doubled)
    assert "glm-5" in rendered and "Inferact/GLM-5.3-NVFP4" in rendered
    assert "provider/GLM-5" in rendered and "NRP_GLM" in rendered
    assert "https://example.test/v1" in rendered and "8000" in rendered


def _save_run(directory, document, *, status="complete", missing_document=False):
    (directory / "documents").mkdir()
    (directory / "documents" / "example.txt.json").write_text(document.model_dump_json())
    filings = ["example.txt", "missing.txt"] if missing_document else ["example.txt"]
    (directory / "run.json").write_text(
        json.dumps(
            {
                "status": status,
                "set": "primary",
                "filings": filings,
                "annotation_sha256": {"example.txt": "intentionally-stale-no-annotation-file-exists"},
            }
        )
    )


def test_resource_only_report_uses_saved_traces_without_annotations_or_accuracy_scoring(tmp_path):
    document = _site(
        Document.from_source("No. 24-123"), _run(_usage(100, 20), success=True)
    ).complete_substage(HUNTING)
    document = _site(document, _run(_usage(200, 30)), substage=BODY_REVIEW)
    _save_run(tmp_path, document)

    report = score_llm_usage_run(tmp_path)
    saved = json.loads((tmp_path / "llm_usage.json").read_text())

    assert report.total.input_tokens == saved["total"]["input_tokens"] == 300
    assert saved["total"]["reviews"]["runs"] == 2
    assert saved["total"]["reviews"]["failed_runs"] == 1
    assert saved["workflows"]["grow_roots"]["total"]["input_tokens"] == 100
    assert saved["workflows"]["validate_roots_citation_identity"]["total"]["input_tokens"] == 200
    assert saved["workflows"]["grow_leaves"]["total"]["attempts"] == 0
    assert saved["saved_documents"] == saved["expected_documents"] == 1
    assert {path.name for path in tmp_path.glob("*.md")} == {"llm_usage.md"}
    rendered = (tmp_path / "llm_usage.md").read_text()
    assert "## Workflow totals" in rendered
    assert "**Total** | 2 | 1/2 (50.0%)" in rendered


def test_resource_only_report_marks_partial_saved_document_coverage(tmp_path):
    document = _site(Document.from_source("No. 24-123"), _run(_usage(10, 2)))
    _save_run(tmp_path, document, status="running", missing_document=True)
    report = score_llm_usage_run(tmp_path)
    saved = json.loads((tmp_path / "llm_usage.json").read_text())

    assert report.total.attempts == 1
    assert (saved["run_status"], saved["saved_documents"], saved["expected_documents"]) == ("running", 1, 2)
    assert "Saved Documents: 1/2" in (tmp_path / "llm_usage.md").read_text()


def test_resource_only_report_rejects_missing_completed_or_invalid_documents(tmp_path):
    _save_run(tmp_path, Document.from_source("No. 24-123"), missing_document=True)
    with pytest.raises(ValueError, match="do not match"):
        score_llm_usage_run(tmp_path)
    (tmp_path / "documents" / "missing.txt.json").write_text('{"obsolete_format":true}')
    with pytest.raises(ValueError):
        score_llm_usage_run(tmp_path)
    assert not (tmp_path / "llm_usage.md").exists()


def test_empty_review_totals_have_unknown_rates_and_duration():
    total = ReviewTotals()
    assert total.failure_rate is None
    assert total.first_attempt_success_rate is None
    assert total.repair_rate is None
    assert total.recovery_rate is None
    assert total.elapsed_seconds is None


def test_existing_four_workflow_reports_include_scoped_telemetry_and_one_run_total(tmp_path, monkeypatch):
    from evaluations import score_run as scoring
    from evaluations.score_types import Precision

    document = _site(
        Document.from_source("No. 24-123"), _run(_usage(100, 20), success=True)
    ).complete_substage(HUNTING)
    document = _site(document, _run(_usage(200, 30)), substage=BODY_REVIEW).complete_substage(BODY_REVIEW)
    _save_run(tmp_path, document)
    run_record = json.loads((tmp_path / "run.json").read_text())
    del run_record["annotation_sha256"]
    (tmp_path / "run.json").write_text(json.dumps(run_record))
    names = tuple(scoring._WORKFLOWS)
    calls = []

    def scorer(saved):
        assert saved == document
        calls.append(saved)
        return Precision(2, 3)

    monkeypatch.setattr(
        scoring,
        "_WORKFLOWS",
        {name: (scorer, lambda score, *, set_name: "Unchanged accuracy table: 2/3.\n") for name in names},
    )

    rendered = scoring.score_run(tmp_path, names)

    assert tuple(rendered) == names
    assert len(calls) == 4
    for name in names:
        score = json.loads((tmp_path / f"{name}.json").read_text())
        assert (score["correct"], score["total"], score["precision"]) == (2, 3, 2 / 3)
        assert "Unchanged accuracy table: 2/3." in rendered[name]
        assert "## LLM review outcomes" in rendered[name]
        assert score["llm_usage"]["total"]["attempts"] == int(
            name in ("grow_roots", "validate_roots_citation_identity")
        )
    total = json.loads((tmp_path / "llm_usage.json").read_text())
    assert total["total"]["attempts"] == total["total"]["reviews"]["runs"] == 2
    assert total["total"]["input_tokens"] == 300
