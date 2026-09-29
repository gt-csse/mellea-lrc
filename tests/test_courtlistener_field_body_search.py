"""Case-name body discovery stays separate from locator corroboration."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any, Literal

import pytest

from mellea_lrc.api import Document, grow_roots
from mellea_lrc.courtlistener.models import CourtListenerSearchPage
from mellea_lrc.model.citations.body_evidence import BodySource
from mellea_lrc.validation.body_search.common import field_query_name, field_query_parties
from mellea_lrc.validation.body_search.courtlistener_opinion_fields import (
    STAGE as OPINION_STAGE,
)
from mellea_lrc.validation.body_search.courtlistener_opinion_fields import (
    courtlistener_opinion_field_body_search,
)
from mellea_lrc.validation.body_search.courtlistener_recap_fields import STAGE as RECAP_STAGE
from mellea_lrc.validation.body_search.courtlistener_recap_fields import (
    courtlistener_recap_field_body_search,
)


class FakeBodyClient:
    def __init__(
        self,
        page: CourtListenerSearchPage,
        *,
        opinions: dict[str, dict[str, Any]] | None = None,
        recap_documents: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.page = page
        self.opinions = opinions or {}
        self.recap_documents = recap_documents or {}
        self.search_calls: list[tuple[str, str, str | None]] = []
        self.opinion_calls: list[str] = []
        self.recap_calls: list[str] = []

    def search(
        self, q: str, search_type: Literal["o", "rd"], cursor: str | None = None
    ) -> CourtListenerSearchPage:
        self.search_calls.append((q, search_type, cursor))
        return self.page

    def get_opinion(self, opinion_id: str) -> dict[str, Any] | None:
        self.opinion_calls.append(opinion_id)
        return self.opinions.get(opinion_id)

    def get_recap_document(self, recap_document_id: str) -> dict[str, Any] | None:
        self.recap_calls.append(recap_document_id)
        return self.recap_documents.get(recap_document_id)


def _page(*results: dict[str, Any]) -> CourtListenerSearchPage:
    return CourtListenerSearchPage.model_validate(
        {"count": len(results), "next": None, "previous": None, "results": list(results)}
    )


def _ready(source: str = "Brown v. Board of Education, 347 U.S. 483 (1954).") -> Document:
    document = asyncio.run(grow_roots(Document.from_source(source)))
    root = document.roots[0].record("23_locator_body_review")
    return document.replace_citation(root.with_route("case_name_body_discovery")).complete(
        "23_locator_body_review"
    )


def test_opinion_field_search_saves_grounded_name_and_different_locator() -> None:
    before = _ready()
    name = field_query_name(before.roots[0])
    assert name is not None
    assert field_query_parties(before.roots[0]) == ("Brown", "Education")
    page = _page({"cluster_id": 900, "opinions": [{"id": 901}], "dateFiled": "1960-01-01"})
    client = FakeBodyClient(
        page,
        opinions={
            "901": {
                "id": 901,
                "plain_text": (
                    "A later court cited Brown v. Board of Education, 349 U.S. 294, "
                    "and discussed the holding."
                ),
            }
        },
    )

    after = courtlistener_opinion_field_body_search(
        before, client=client, retrospective_date=date(1970, 1, 1)
    )

    assert after.stage_runs[-1] == OPINION_STAGE
    assert client.search_calls == [("Brown AND Education", "o", None)]
    assert client.opinion_calls == ["901"]
    assert after.roots[0].body_searches == ()
    assert after.roots[0].identity_judgments == ()
    search = after.roots[0].field_body_searches[0]
    assert search.source is BodySource.COURTLISTENER_OPINION
    assert search.query_name == name
    assert search.attempts[0].pages == (page.raw_json,)
    assert len(search.evidence) == 1
    evidence = search.evidence[0]
    assert evidence.anchor_kind == "case_name"
    assert evidence.excerpt[evidence.anchor_span.start : evidence.anchor_span.end] == name
    assert "349 U.S. 294" in evidence.excerpt
    assert "347 U.S. 483" not in evidence.excerpt
    assert after.get_stage("23_locator_body_review") == before
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_recap_search_requires_fetched_case_name_and_item_date() -> None:
    before = _ready()
    name = field_query_name(before.roots[0])
    assert name is not None
    page = _page({"id": 44, "docket_id": 88, "entry_date_filed": "1960-01-01", "snippet": name})
    client = FakeBodyClient(
        page,
        recap_documents={"44": {"id": 44, "plain_text": "This filing discusses an unrelated case."}},
    )

    after = courtlistener_recap_field_body_search(before, client=client, retrospective_date=date(1970, 1, 1))

    assert client.search_calls == [("Brown AND Education", "rd", None)]
    assert client.recap_calls == ["44"]
    search = after.roots[0].field_body_searches[0]
    assert search.attempts[0].pages == (page.raw_json,)
    assert search.evidence == ()
    assert [failure.failure_type for failure in search.failures] == ["no_grounded_anchor"]


def test_field_search_excludes_an_opinion_after_the_retrospective_cutoff() -> None:
    before = _ready()
    name = field_query_name(before.roots[0])
    assert name is not None
    client = FakeBodyClient(
        _page({"cluster_id": 900, "opinions": [{"id": 901}], "dateFiled": "1980-01-01"}),
        opinions={"901": {"id": 901, "plain_text": f"A later court cites {name}, 349 U.S. 294."}},
    )

    after = courtlistener_opinion_field_body_search(
        before, client=client, retrospective_date=date(1970, 1, 1)
    )

    search = after.roots[0].field_body_searches[0]
    assert search.evidence == ()
    assert [failure.failure_type for failure in search.failures] == ["ineligible_issue_date"]


def test_two_party_search_stops_after_four_unhelpful_details_without_broad_query() -> None:
    before = _ready()
    page = _page(
        *(
            {"cluster_id": index, "opinions": [{"id": index}], "dateFiled": "1960-01-01"}
            for index in range(1, 6)
        )
    )
    client = FakeBodyClient(
        page,
        opinions={str(index): {"id": index, "plain_text": "An unrelated opinion."} for index in range(1, 6)},
    )

    after = courtlistener_opinion_field_body_search(before, client=client)

    assert client.search_calls == [("Brown AND Education", "o", None)]
    assert client.opinion_calls == ["1", "2", "3", "4"]
    search = after.roots[0].field_body_searches[0]
    assert search.query_name == "Brown"
    assert search.evidence == ()
    assert search.attempts[0].failure.failure_type == "candidate_limit_reached"


def test_field_stages_only_process_routed_roots_and_keep_independent_records() -> None:
    before = _ready()
    client = FakeBodyClient(_page())
    opinion = courtlistener_opinion_field_body_search(before, client=client)
    recap = courtlistener_recap_field_body_search(opinion, client=client)

    assert [item.source for item in recap.roots[0].field_body_searches] == [
        BodySource.COURTLISTENER_OPINION,
        BodySource.COURTLISTENER_RECAP,
    ]
    assert recap.stage_runs[-2:] == (OPINION_STAGE, RECAP_STAGE)
    with pytest.raises(ValueError, match="already completed"):
        courtlistener_opinion_field_body_search(recap, client=client)

    unselected = asyncio.run(grow_roots(Document.from_source("Brown v. Board of Education, 347 U.S. 483.")))
    unselected = unselected.complete("23_locator_body_review")
    skipped = courtlistener_opinion_field_body_search(unselected, client=client)
    assert skipped.roots[0].field_body_searches == ()
    assert len(client.search_calls) == 2


def test_missing_case_name_records_failure_without_provider_call() -> None:
    before = _ready("347 U.S. 483.")
    client = FakeBodyClient(_page())

    after = courtlistener_opinion_field_body_search(before, client=client)

    assert client.search_calls == []
    search = after.roots[0].field_body_searches[0]
    assert search.query_name is None
    assert search.attempts == ()
    assert [failure.failure_type for failure in search.failures] == ["unsearchable_case_name"]
