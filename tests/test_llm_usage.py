"""Provider usage observations follow native histories, independent of accuracy."""

from __future__ import annotations

import pytest

from evaluations.llm_usage import LlmUsageReport, UsageTotals, render_llm_usage, summarize_llm_usage
from mellea_lrc.llm.config import PROFILE_OPTION
from mellea_lrc.model import Document, FullDocketCitation, Span
from mellea_lrc.model.citations.docket_root_llm_reassignment import DocketRootReview
from mellea_lrc.model.ivr import IvrAttempt, IvrRun
from mellea_lrc.model.site_review import SiteReview

DISCOVERY = "grow_roots.discover_root_locators.find_docket_locators"
HUNTING = "grow_roots.discover_root_locators.hunt_docket_locators"
REASSIGNMENT = "grow_roots.form_root_groups.reassign_docket_roots_llm"
BODY_REVIEW = "validate_roots_citation_identity.corroborate_locator_bodies.review_locator_bodies_llm"


def _run(*usages, model="example-model", endpoint="https://example.test/v1") -> IvrRun:
    return IvrRun(
        success=False,
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
