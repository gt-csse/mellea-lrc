"""CourtListener body stages use fetched text, preserve trace, and enforce dates."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date
from typing import Any, Literal

import pytest

from mellea_lrc.api import Document, grow_roots
from mellea_lrc.providers.courtlistener import CourtListenerHTTPError, CourtListenerTransportError
from mellea_lrc.providers.courtlistener.models import CourtListenerSearchPage
from mellea_lrc.model.citations.body_evidence import BodySource
from mellea_lrc.validation.body_search.common import roots_for_body_search
from mellea_lrc.validation.body_search.locator_body_courtlistener_opinion_retrieval import (
    STAGE as OPINION_STAGE,
)
from mellea_lrc.validation.body_search.locator_body_courtlistener_opinion_retrieval import (
    locator_body_courtlistener_opinion_retrieval,
)
from mellea_lrc.validation.body_search.locator_body_courtlistener_recap_retrieval import STAGE as RECAP_STAGE
from mellea_lrc.validation.body_search.locator_body_courtlistener_recap_retrieval import (
    locator_body_courtlistener_recap_retrieval,
)


class FakeBodyClient:
    def __init__(
        self,
        search: Callable[[str, Literal["o", "rd"], str | None], CourtListenerSearchPage],
        *,
        opinions: dict[str, dict[str, Any] | Exception | None] | None = None,
        recap_documents: dict[str, dict[str, Any] | Exception | None] | None = None,
    ) -> None:
        self.respond_search = search
        self.opinions = opinions or {}
        self.recap_documents = recap_documents or {}
        self.search_calls: list[tuple[str, str, str | None]] = []
        self.opinion_calls: list[str] = []
        self.recap_calls: list[str] = []

    def search(
        self, q: str, search_type: Literal["o", "rd"], cursor: str | None = None
    ) -> CourtListenerSearchPage:
        self.search_calls.append((q, search_type, cursor))
        return self.respond_search(q, search_type, cursor)

    def get_opinion(self, opinion_id: str) -> dict[str, Any] | None:
        self.opinion_calls.append(opinion_id)
        result = self.opinions[opinion_id]
        if isinstance(result, Exception):
            raise result
        return result

    def get_recap_document(self, recap_document_id: str) -> dict[str, Any] | None:
        self.recap_calls.append(recap_document_id)
        result = self.recap_documents[recap_document_id]
        if isinstance(result, Exception):
            raise result
        return result


def _rooted(source: str = "Brown v. Board of Education, 347 U.S. 483 (1954).") -> Document:
    return asyncio.run(grow_roots(Document.from_source(source)))


def _page(*results: dict[str, Any], next_url: str | None = None) -> CourtListenerSearchPage:
    return CourtListenerSearchPage.model_validate(
        {
            "count": len(results),
            "next": next_url,
            "previous": None,
            "results": list(results),
            "upstream_marker": {"preserve": True},
        }
    )


def test_queued_field_identity_root_waits_for_aggregation_before_body_search() -> None:
    before = _rooted()
    route = before.roots[0].record("reporter_review")
    queued = before.replace_citation(route.with_route("fields_aggregated_identity")).complete(
        "reporter_review"
    )
    assert roots_for_body_search(queued) == ()
    assert queued.roots[0].identity_judgments == ()

    client = FakeBodyClient(lambda *_args: pytest.fail("Queued root must not be searched"))
    skipped = locator_body_courtlistener_opinion_retrieval(queued, client=client)
    assert client.search_calls == []
    assert skipped.roots[0].body_searches == ()

    aggregate = queued.roots[0].record("fields_aggregated_identity")
    ready = queued.replace_citation(aggregate.with_route(OPINION_STAGE)).complete(
        "fields_aggregated_identity"
    )
    assert roots_for_body_search(ready) == ready.roots
    assert ready.roots[0].identity_judgments == ()
    assert [item.value for item in ready.roots[0].routes] == ["fields_aggregated_identity", OPINION_STAGE]


def test_opinion_stage_uses_nested_opinion_id_and_multiple_full_body_occurrences() -> None:
    assert OPINION_STAGE == "20_locator_body_courtlistener_opinion_retrieval"
    before = _rooted()
    first = _page(
        {
            "cluster_id": 900,
            "id": 900,
            "dateFiled": "1960-01-01",
            "opinions": [{"id": 901}],
            "absolute_url": "/opinion/900/later-case/",
            "snippet": "347 U.S. 483",
        }
    )
    text = "A later court cited 347 U.S. 483. It also discussed 347 U.S. 483 in detail."
    client = FakeBodyClient(
        lambda *_args: first,
        opinions={"901": {"id": 901, "plain_text": text, "absolute_url": "/api/opinions/901/"}},
    )

    after = locator_body_courtlistener_opinion_retrieval(
        before, client=client, retrospective_date=date(1970, 1, 1)
    )

    assert client.search_calls == [('"347 U.S. 483"', "o", None)]
    assert client.opinion_calls == ["901"]
    assert after.stage_runs == (*before.stage_runs, OPINION_STAGE)
    assert after.get_stage("10_roots") == before
    search = after.roots[0].body_searches[0]
    assert search.source is BodySource.COURTLISTENER_OPINION
    assert search.node_id == after.roots[0].nodes[-1].id
    assert search.attempts[0].pages == (first.raw_json,)
    assert len(search.evidence) == 2
    assert [item.body_id for item in search.evidence] == ["901", "901"]
    assert all(item.issued_on == date(1960, 1, 1) for item in search.evidence)
    assert all(item.date_basis == "search.dateFiled" for item in search.evidence)
    assert all(item.parent_id == "900" for item in search.evidence)
    assert all(item.metadata["text_field"] == "plain_text" for item in search.evidence)
    assert all(item.anchor_kind == "locator" for item in search.evidence)
    assert all(
        item.excerpt[item.anchor_span.start : item.anchor_span.end] == "347 U.S. 483"
        for item in search.evidence
    )
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_opinion_stage_never_uses_cluster_id_as_opinion_id() -> None:
    before = _rooted()
    client = FakeBodyClient(lambda *_args: _page({"cluster_id": 900, "dateFiled": "1960-01-01"}))

    after = locator_body_courtlistener_opinion_retrieval(before, client=client)

    assert client.opinion_calls == []
    search = after.roots[0].body_searches[0]
    assert search.evidence == ()
    assert any(
        failure.failure_type == "missing_body_id" and failure.item_id == "900" for failure in search.failures
    )
    assert len(search.attempts) == 1


def test_opinion_detail_must_belong_to_search_cluster() -> None:
    client = FakeBodyClient(
        lambda *_args: _page({"cluster_id": 900, "opinions": [{"id": 901}]}),
        opinions={
            "901": {
                "id": 901,
                "cluster": "https://www.courtlistener.com/api/rest/v4/clusters/999/",
                "plain_text": "The later court cited 347 U.S. 483.",
            }
        },
    )

    after = locator_body_courtlistener_opinion_retrieval(_rooted("347 U.S. 483."), client=client)

    search = after.roots[0].body_searches[0]
    assert search.evidence == ()
    assert search.failures[0].failure_type == "parent_id_mismatch"


def test_detail_must_identify_the_fetched_opinion() -> None:
    client = FakeBodyClient(
        lambda *_args: _page({"cluster_id": 900, "opinions": [{"id": 901}]}),
        opinions={"901": {"plain_text": "The later court cited 347 U.S. 483."}},
    )

    after = locator_body_courtlistener_opinion_retrieval(_rooted("347 U.S. 483."), client=client)

    search = after.roots[0].body_searches[0]
    assert search.evidence == ()
    assert search.failures[0].failure_type == "record_id_mismatch"


def test_recap_stage_uses_document_id_and_entry_date_not_case_date() -> None:
    assert RECAP_STAGE == "21_locator_body_courtlistener_recap_retrieval"
    before = _rooted()
    client = FakeBodyClient(
        lambda *_args: _page(
            {
                "id": 44,
                "docket_id": 88,
                "dateFiled": "1950-01-01",
                "entry_date_filed": "1975-01-01",
                "snippet": "347 U.S. 483",
            }
        ),
        recap_documents={"44": {"id": 44, "plain_text": "This filing cites 347 U.S. 483."}},
    )

    after = locator_body_courtlistener_recap_retrieval(
        before, client=client, retrospective_date=date(1970, 1, 1)
    )

    assert client.recap_calls == ["44"]
    assert client.opinion_calls == []
    search = after.roots[0].body_searches[0]
    assert search.source is BodySource.COURTLISTENER_RECAP
    assert search.evidence == ()
    assert any(failure.failure_type == "ineligible_issue_date" for failure in search.failures)
    assert after.stage_runs[-1] == RECAP_STAGE


def test_multi_opinion_cluster_date_cannot_certify_an_individual_opinion_cutoff() -> None:
    client = FakeBodyClient(
        lambda *_args: _page(
            {
                "cluster_id": 900,
                "dateFiled": "1960-01-01",
                "opinions": [{"id": 901}, {"id": 902}],
            }
        ),
        opinions={
            "901": {"id": 901, "plain_text": "A later court cited 347 U.S. 483."},
            "902": {"id": 902, "plain_text": "A separate order cited 347 U.S. 483."},
        },
    )
    result = locator_body_courtlistener_opinion_retrieval(
        _rooted(), client=client, retrospective_date=date(1970, 1, 1)
    )
    search = result.roots[0].body_searches[-1]
    assert search.evidence == ()
    assert [failure.failure_type for failure in search.failures] == [
        "ineligible_issue_date",
        "ineligible_issue_date",
    ]


def test_recap_stage_requires_fetched_full_text_and_exact_item_date() -> None:
    before = _rooted()
    first = _page(
        {
            "id": 44,
            "docket_id": 88,
            "dateFiled": "1950-01-01",
            "entry_date_filed": "1960-01-01",
            "snippet": "347 U.S. 483",
        }
    )
    client = FakeBodyClient(
        lambda *_args: first,
        recap_documents={"44": {"id": 44, "plain_text": "This filing cites 347 U.S. 483."}},
    )

    after = locator_body_courtlistener_recap_retrieval(
        before, client=client, retrospective_date=date(1970, 1, 1)
    )

    assert client.search_calls == [('"347 U.S. 483"', "rd", None)]
    search = after.roots[0].body_searches[0]
    assert search.attempts[0].pages == (first.raw_json,)
    assert len(search.evidence) == 1
    assert search.evidence[0].body_id == "44"
    assert search.evidence[0].parent_id == "88"
    assert search.evidence[0].date_basis == "search.entry_date_filed"
    assert search.evidence[0].issued_on == date(1960, 1, 1)
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_search_snippet_never_becomes_evidence_without_fetched_locator() -> None:
    before = _rooted("347 U.S. 483.")
    client = FakeBodyClient(
        lambda *_args: _page({"id": 44, "entry_date_filed": "1960-01-01", "snippet": "347 U.S. 483"}),
        recap_documents={"44": {"id": 44, "plain_text": "No cited case appears in this body."}},
    )

    after = locator_body_courtlistener_recap_retrieval(before, client=client)

    search = after.roots[0].body_searches[0]
    assert search.evidence == ()
    assert [failure.failure_type for failure in search.failures] == ["no_grounded_anchor"]


def test_locator_search_hit_with_only_a_case_name_does_not_become_evidence() -> None:
    def respond(query: str, _kind: Literal["o", "rd"], _cursor: str | None) -> CourtListenerSearchPage:
        assert query == '"347 U.S. 483"'
        return _page({"cluster_id": 900, "opinions": [{"id": 901}], "dateFiled": "1960-01-01"})

    client = FakeBodyClient(
        respond,
        opinions={
            "901": {
                "id": 901,
                "plain_text": "The later case cites Brown v. Board of Education, 349 U.S. 294.",
            }
        },
    )

    after = locator_body_courtlistener_opinion_retrieval(_rooted(), client=client)

    assert [call[0] for call in client.search_calls] == ['"347 U.S. 483"']
    search = after.roots[0].body_searches[0]
    assert len(search.attempts) == 1
    assert search.evidence == ()
    assert any(failure.failure_type == "no_grounded_anchor" for failure in search.failures)


def test_opinion_html_full_text_is_read_when_plain_text_is_empty() -> None:
    client = FakeBodyClient(
        lambda *_args: _page({"cluster_id": 900, "opinions": [{"id": 901}], "dateFiled": "1960-01-01"}),
        opinions={
            "901": {
                "id": 901,
                "plain_text": "",
                "html_lawbox": "<p>A later court cited <b>347 U.S. 483</b> in its opinion.</p>",
            }
        },
    )

    after = locator_body_courtlistener_opinion_retrieval(_rooted("347 U.S. 483."), client=client)

    evidence = after.roots[0].body_searches[0].evidence
    assert len(evidence) == 1
    assert evidence[0].metadata["text_field"] == "html_lawbox"
    assert "347 U.S. 483" in evidence[0].excerpt


def test_retrospective_run_excludes_undated_item_even_with_body_match() -> None:
    client = FakeBodyClient(
        lambda *_args: _page({"id": 44, "dateFiled": "1950-01-01"}),
        recap_documents={"44": {"id": 44, "plain_text": "A later filing cites 347 U.S. 483."}},
    )

    after = locator_body_courtlistener_recap_retrieval(
        _rooted("347 U.S. 483."), client=client, retrospective_date=date(1970, 1, 1)
    )

    search = after.roots[0].body_searches[0]
    assert search.evidence == ()
    assert search.failures[0].failure_type == "ineligible_issue_date"
    assert "no exact filing date" in search.failures[0].message


def test_detail_fetch_budget_records_truncation() -> None:
    hits = tuple({"id": index, "entry_date_filed": "1960-01-01"} for index in range(1, 10))
    client = FakeBodyClient(
        lambda *_args: _page(*hits),
        recap_documents={
            str(index): {"id": index, "plain_text": "No matching locator here."} for index in range(1, 10)
        },
    )

    after = locator_body_courtlistener_recap_retrieval(_rooted("347 U.S. 483."), client=client)

    search = after.roots[0].body_searches[0]
    assert client.recap_calls == [str(index) for index in range(1, 9)]
    assert search.attempts[0].pages[0]["results"] == list(hits)
    assert search.attempts[0].failure.failure_type == "candidate_limit_reached"


def test_search_hit_budget_records_truncation_without_fetching() -> None:
    hits = tuple({"cluster_id": index} for index in range(1, 42))
    client = FakeBodyClient(lambda *_args: _page(*hits))

    after = locator_body_courtlistener_opinion_retrieval(_rooted("347 U.S. 483."), client=client)

    search = after.roots[0].body_searches[0]
    assert len(search.attempts[0].pages[0]["results"]) == 41
    assert len(search.failures) == 40
    assert search.attempts[0].failure.failure_type == "hit_limit_reached"
    assert client.opinion_calls == []


def test_detail_failure_retains_status_url_and_message(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", sleep_calls.append)
    error = CourtListenerHTTPError(
        "CourtListener opinion lookup returned HTTP 429",
        failure_type="http_error",
        upstream_status_code=429,
        url="https://proxy.example/api/rest/v4/opinions/901/",
        upstream_detail="retry later",
    )
    client = FakeBodyClient(
        lambda *_args: _page({"cluster_id": 900, "opinions": [{"id": 901}]}),
        opinions={"901": error},
    )

    after = locator_body_courtlistener_opinion_retrieval(_rooted("347 U.S. 483."), client=client)

    failure = after.roots[0].body_searches[0].failures[0]
    assert (failure.failure_type, failure.status_code, failure.item_id) == ("http_error", 429, "901")
    assert "proxy.example" in failure.message
    assert "retry later" in failure.message
    assert client.opinion_calls == ["901"] * 3
    assert sleep_calls == [2.0, 4.0]


@pytest.mark.parametrize("retry_after_seconds", [301, 17000])
def test_long_proxy_quota_hint_does_not_retry(
    monkeypatch: pytest.MonkeyPatch, retry_after_seconds: int
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", sleep_calls.append)
    error = CourtListenerHTTPError(
        "CourtListener proxy returned HTTP 429",
        failure_type="http_error",
        upstream_status_code=429,
        upstream_detail=f'{{"retry_after_seconds": {retry_after_seconds}}}',
    )
    client = FakeBodyClient(
        lambda *_args: _page({"cluster_id": 900, "opinions": [{"id": 901}]}),
        opinions={"901": error},
    )

    after = locator_body_courtlistener_opinion_retrieval(_rooted("347 U.S. 483."), client=client)

    assert after.roots[0].body_searches[0].failures[0].status_code == 429
    assert client.opinion_calls == ["901"]
    assert sleep_calls == []


def test_unhinted_short_429_recovers_on_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", sleep_calls.append)
    error = CourtListenerHTTPError(
        "CourtListener lookup returned HTTP 429",
        failure_type="http_error",
        upstream_status_code=429,
        upstream_detail='{"detail":"Request was throttled."}',
    )

    class RetryOnceBodyClient(FakeBodyClient):
        def get_opinion(self, opinion_id: str) -> dict[str, Any] | None:
            if not self.opinion_calls:
                self.opinion_calls.append(opinion_id)
                raise error
            return super().get_opinion(opinion_id)

    client = RetryOnceBodyClient(
        lambda *_args: _page({"cluster_id": 900, "opinions": [{"id": 901}]}),
        opinions={"901": {"id": 901, "plain_text": "A later court cited 347 U.S. 483."}},
    )

    after = locator_body_courtlistener_opinion_retrieval(_rooted("347 U.S. 483."), client=client)

    assert client.opinion_calls == ["901", "901"]
    assert sleep_calls == [2.0]
    assert after.roots[0].body_searches[0].failures == ()


@pytest.mark.parametrize("failed_step", ["search", "detail"])
@pytest.mark.parametrize("failure_kind", ["transport", "server"])
def test_transient_provider_failure_recovers_without_rerunning_document(
    monkeypatch: pytest.MonkeyPatch, failed_step: str, failure_kind: str
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", sleep_calls.append)
    error = (
        CourtListenerTransportError("read timed out", failure_type="transport_error")
        if failure_kind == "transport"
        else CourtListenerHTTPError("server unavailable", failure_type="http_error", upstream_status_code=503)
    )
    search_attempts = 0

    def respond_search(*_args: object) -> CourtListenerSearchPage:
        nonlocal search_attempts
        search_attempts += 1
        if failed_step == "search" and search_attempts == 1:
            raise error
        return _page({"cluster_id": 900, "opinions": [{"id": 901}]})

    class RetryDetailClient(FakeBodyClient):
        def get_opinion(self, opinion_id: str) -> dict[str, Any] | None:
            if failed_step == "detail" and not self.opinion_calls:
                self.opinion_calls.append(opinion_id)
                raise error
            return super().get_opinion(opinion_id)

    client = RetryDetailClient(
        respond_search,
        opinions={"901": {"id": 901, "plain_text": "A later court cited 347 U.S. 483."}},
    )
    after = locator_body_courtlistener_opinion_retrieval(_rooted("347 U.S. 483."), client=client)
    assert sleep_calls == [2.0]
    assert after.roots[0].body_searches[0].failures == ()
    assert client.search_calls == [('"347 U.S. 483"', "o", None)] * (2 if failed_step == "search" else 1)
    assert client.opinion_calls == ["901"] * (2 if failed_step == "detail" else 1)


@pytest.mark.parametrize("rate_limited_step", ["search", "detail"])
@pytest.mark.parametrize("retry_after_seconds", [0.25, 63, 195])
def test_proxy_429_retry_hint_recovers_grounded_opinion_evidence(
    monkeypatch: pytest.MonkeyPatch, rate_limited_step: str, retry_after_seconds: float
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", sleep_calls.append)
    page = _page({"cluster_id": 900, "opinions": [{"id": 901}], "dateFiled": "1960-01-01"})
    error = CourtListenerHTTPError(
        "CourtListener proxy returned HTTP 429",
        failure_type="http_error",
        upstream_status_code=429,
        url="https://proxy.example/api/rest/v4/",
        upstream_detail=f'{{"retry_after_seconds": {retry_after_seconds}}}',
    )

    class RetryOnceBodyClient(FakeBodyClient):
        def search(
            self, q: str, search_type: Literal["o", "rd"], cursor: str | None = None
        ) -> CourtListenerSearchPage:
            if rate_limited_step == "search" and not self.search_calls:
                self.search_calls.append((q, search_type, cursor))
                raise error
            return super().search(q, search_type, cursor)

        def get_opinion(self, opinion_id: str) -> dict[str, Any] | None:
            if rate_limited_step == "detail" and not self.opinion_calls:
                self.opinion_calls.append(opinion_id)
                raise error
            return super().get_opinion(opinion_id)

    client = RetryOnceBodyClient(
        lambda *_args: page,
        opinions={"901": {"id": 901, "plain_text": "A later court cited 347 U.S. 483."}},
    )

    after = locator_body_courtlistener_opinion_retrieval(
        _rooted("347 U.S. 483."), client=client, retrospective_date=date(1970, 1, 1)
    )

    query = ('"347 U.S. 483"', "o", None)
    assert client.search_calls == [query] * (2 if rate_limited_step == "search" else 1)
    assert client.opinion_calls == ["901"] * (2 if rate_limited_step == "detail" else 1)
    assert sleep_calls == [retry_after_seconds + 0.25]
    search = after.roots[0].body_searches[0]
    assert search.attempts[0].pages == (page.raw_json,)
    assert search.attempts[0].failure is None
    assert search.failures == ()
    assert len(search.evidence) == 1
    evidence = search.evidence[0]
    assert evidence.body_id == "901"
    assert evidence.excerpt[evidence.anchor_span.start : evidence.anchor_span.end] == "347 U.S. 483"


def test_pagination_failure_retains_prior_raw_page() -> None:
    first = _page(next_url="https://www.courtlistener.com/api/rest/v4/search/?cursor=next%2B%2F%3D")

    def respond(_query: str, _kind: Literal["o", "rd"], cursor: str | None) -> CourtListenerSearchPage:
        if cursor is None:
            return first
        raise CourtListenerHTTPError(
            "CourtListener search returned HTTP 500",
            failure_type="http_error",
            upstream_status_code=500,
        )

    client = FakeBodyClient(respond)

    after = locator_body_courtlistener_opinion_retrieval(_rooted("347 U.S. 483."), client=client)

    attempt = after.roots[0].body_searches[0].attempts[0]
    assert attempt.pages == (first.raw_json,)
    assert (attempt.failure.failure_type, attempt.failure.status_code) == ("http_error", 500)
    assert client.search_calls[:2] == [
        ('"347 U.S. 483"', "o", None),
        ('"347 U.S. 483"', "o", "next+/="),
    ]


def test_both_sources_append_independently_and_reject_repeat() -> None:
    client = FakeBodyClient(lambda *_args: _page())
    before = _rooted("347 U.S. 483.")

    after_opinion = locator_body_courtlistener_opinion_retrieval(before, client=client)
    after_recap = locator_body_courtlistener_recap_retrieval(after_opinion, client=client)

    assert [item.source for item in after_recap.roots[0].body_searches] == [
        BodySource.COURTLISTENER_OPINION,
        BodySource.COURTLISTENER_RECAP,
    ]
    assert after_recap.stage_runs[-2:] == (OPINION_STAGE, RECAP_STAGE)
    with pytest.raises(ValueError, match="already completed"):
        locator_body_courtlistener_opinion_retrieval(after_recap, client=client)
