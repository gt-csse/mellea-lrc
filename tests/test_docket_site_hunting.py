"""Optional docket site hunting adds source-grounded full locators before colocation."""

import asyncio

import httpx
import pytest

from mellea_lrc.extraction import (
    find_docket_locators,
    find_full_reporter_locators,
    grow_roots,
    hunt_docket_locators,
    resolve_colocations,
)
from mellea_lrc.extraction.docket_hunting import DocketSiteDecision, suspected_dockets
from mellea_lrc.llm.docket_review import DocketReviewServiceError, OpenRouterDocketReviewer
from mellea_lrc.model import Document, FullDocketCitation, FullReporterCitation, Span

STAGE = "docket_locator_site_hunting"


def _ready(source: str, *, index_spans: tuple[Span, ...] = ()) -> Document:
    document = Document.from_source(source)
    if index_spans:
        document = Document.model_validate({**document.model_dump(mode="python"), "index_spans": index_spans})
    return find_docket_locators(find_full_reporter_locators(document))


def _accept(site: object) -> DocketSiteDecision:
    return DocketSiteDecision(
        is_docket_citation=True,
        locator=site.locator_text,
        docket_number=site.docket_number,
        reason="The text cites a case docket.",
    )


def test_hunt_recomputes_mask_after_each_admission_and_adds_only_full_dockets() -> None:
    source = (
        "Doe v. Townes, No. 19 Civ. 8034 (S.D.N.Y. 2020). "
        "See Smith v. Jones, Case No. 035547/2021 (N.Y. 2021). "
        "See Doe, 347 U.S. at 495."
    )
    before = _ready(source)
    seen: list[tuple[str, str]] = []

    async def reviewer(site: object) -> DocketSiteDecision:
        seen.append((site.locator_text, site.context))
        return _accept(site)

    hunted = asyncio.run(hunt_docket_locators(before, reviewer=reviewer))

    assert [text for text, _context in seen] == ["No. 19 Civ. 8034", "Case No. 035547/2021"]
    assert "No. 19 Civ. 8034" in seen[0][1]
    assert "No. 19 Civ. 8034" in seen[1][1]
    assert suspected_dockets(hunted) == ()
    assert [type(citation) for citation in hunted.citations] == [FullDocketCitation] * 2
    assert [review.outcome for review in hunted.site_reviews] == ["accepted", "accepted"]
    assert [review.citation_id for review in hunted.site_reviews] == [
        citation.id for citation in hunted.citations
    ]
    assert hunted.short_reporters == ()
    assert "short_reporter_citations" not in hunted.stage_runs
    assert "colocations" not in hunted.stage_runs
    for citation, written_number in zip(hunted.citations, ("19 Civ. 8034", "035547/2021")):
        locator = citation.locator[-1]
        assert source[locator.span.start : locator.span.end] == locator.quote
        assert source[locator.number_span.start : locator.number_span.end] == written_number
        assert locator.get_normalized().docket_number == written_number
        assert citation.case_name == citation.court == citation.date == citation.pin_cite == ()
        assert citation.root_id == ()


@pytest.mark.parametrize(
    ("locator", "number"),
    [
        ("Case No. 035547/2022", "035547/2021"),
        ("Case No. 035547/2021", "035547/2022"),
    ],
)
def test_hunt_does_not_promote_a_decision_with_changed_source_characters(locator: str, number: str) -> None:
    before = _ready("See Case No. 035547/2021 (N.Y. 2021).")

    async def reviewer(_site: object) -> DocketSiteDecision:
        return DocketSiteDecision(
            is_docket_citation=True,
            locator=locator,
            docket_number=number,
            reason="Proposed docket.",
        )

    hunted = asyncio.run(hunt_docket_locators(before, reviewer=reviewer))

    assert hunted.citations == ()
    assert len(hunted.site_reviews) == 1
    assert hunted.site_reviews[0].outcome == "failed"
    assert hunted.site_reviews[0].citation_id is None
    assert hunted.stage_runs == (*before.stage_runs, STAGE)
    assert hunted.get_stage(STAGE) == hunted


def test_declined_site_is_not_reviewed_again_before_the_next_candidate() -> None:
    source = "No. 19 Civ. 8034; later Case No. 035547/2021."
    before = _ready(source)
    seen: list[str] = []

    async def reviewer(site: object) -> DocketSiteDecision:
        seen.append(site.locator_text)
        if len(seen) == 1:
            return DocketSiteDecision(
                is_docket_citation=False,
                locator=None,
                docket_number=None,
                reason="This site was declined.",
            )
        return _accept(site)

    hunted = asyncio.run(hunt_docket_locators(before, reviewer=reviewer))

    assert seen == ["No. 19 Civ. 8034", "Case No. 035547/2021"]
    assert [item.locator[-1].quote for item in hunted.citations] == ["Case No. 035547/2021"]
    assert [review.outcome for review in hunted.site_reviews] == ["declined", "accepted"]


def test_hunt_skips_existing_locators_and_reporter_pinpoints() -> None:
    source = "Case No. 1:24-cv-00123; Doe v. Townes, No. 19 Civ. 8034; Smith v. Jones, 347 U.S. 483, 495-97."
    before = _ready(source)
    seen: list[str] = []

    async def reviewer(site: object) -> DocketSiteDecision:
        seen.append(site.locator_text)
        return _accept(site)

    hunted = asyncio.run(hunt_docket_locators(before, reviewer=reviewer))

    assert seen == ["No. 19 Civ. 8034"]
    assert len(hunted.full_locators) == 3
    assert sum(isinstance(item, FullDocketCitation) for item in hunted.citations) == 2
    assert sum(isinstance(item, FullReporterCitation) for item in hunted.citations) == 1
    assert before.citations[0] in hunted.citations
    for left, right in zip(hunted.citations, hunted.citations[1:]):
        assert left.site_span.end <= right.site_span.start


def test_entry_reference_is_not_a_full_docket_proposal() -> None:
    source = "See (ECF No. 82) and Doe v. Townes, No. 19 Civ. 8034."
    before = _ready(source)
    proposals = suspected_dockets(before)
    assert [site.locator_text for site in proposals] == ["No. 19 Civ. 8034"]


def test_hunt_ignores_index_occurrence_and_keeps_repeated_body_sites_distinct() -> None:
    locator = "Case No. 035547/2021"
    source = f"INDEX: {locator}\nBODY: {locator}; later {locator}."
    index_start = source.index(locator)
    body_start = source.index(locator, index_start + 1)
    later_start = source.index(locator, body_start + 1)
    before = _ready(source, index_spans=(Span(0, source.index("\n")),))
    seen: list[int] = []

    async def reviewer(site: object) -> DocketSiteDecision:
        seen.append(site.locator_span.start)
        return _accept(site)

    hunted = asyncio.run(hunt_docket_locators(before, reviewer=reviewer))

    assert seen == [body_start, later_start]
    assert [item.site_span.start for item in hunted.citations] == [body_start, later_start]
    assert len({item.id for item in hunted.citations}) == 2


def test_hunt_is_idempotent_and_checkpoint_survives_later_colocation() -> None:
    source = "See Doe v. Townes, No. 19 Civ. 8034; Smith v. Jones, 347 U.S. 483."
    before = _ready(source)
    calls = 0

    async def reviewer(site: object) -> DocketSiteDecision:
        nonlocal calls
        calls += 1
        return _accept(site)

    hunted = asyncio.run(hunt_docket_locators(before, reviewer=reviewer))
    assert calls == 1
    assert asyncio.run(hunt_docket_locators(hunted, reviewer=reviewer)) == hunted
    assert calls == 1

    grouped = resolve_colocations(hunted)
    restored = Document.model_validate_json(grouped.model_dump_json())
    assert restored == grouped
    assert grouped.get_stage("docket_locators") == before
    assert grouped.get_stage(STAGE) == hunted
    assert restored.get_stage(STAGE) == hunted
    assert restored.site_reviews == grouped.site_reviews


def test_hunt_rejects_a_late_call_after_colocation() -> None:
    grouped = resolve_colocations(_ready("See No. 19 Civ. 8034."))

    async def reviewer(site: object) -> DocketSiteDecision:
        return _accept(site)

    with pytest.raises(ValueError, match="precede colocation"):
        asyncio.run(hunt_docket_locators(grouped, reviewer=reviewer))


def test_empty_hunt_commits_a_checkpoint_without_calling_reviewer() -> None:
    before = _ready("See Case No. 1:24-cv-00123.")

    async def reviewer(_site: object) -> DocketSiteDecision:
        raise AssertionError("The deterministic locator must be masked")

    hunted = asyncio.run(hunt_docket_locators(before, reviewer=reviewer))

    assert hunted.citations == before.citations
    assert hunted.site_reviews == ()
    assert hunted.stage_runs == (*before.stage_runs, STAGE)
    assert hunted.get_stage(STAGE) == hunted
    assert Document.model_validate_json(hunted.model_dump_json()) == hunted


def test_grow_roots_hunts_before_context_and_does_not_find_short_citations() -> None:
    source = "Doe v. Townes, No. 19 Civ. 8034 (S.D.N.Y. 2020). See Doe, 347 U.S. at 495."

    async def reviewer(site: object) -> DocketSiteDecision:
        return _accept(site)

    document = asyncio.run(grow_roots(Document.from_source(source), hunt_dockets=True, reviewer=reviewer))

    assert document.stage_runs[:4] == (
        "full_reporter_locators",
        "docket_locators",
        STAGE,
        "colocations",
    )
    assert document.stage_runs[-1] == "roots"
    assert len(document.full_locators) == 1
    assert document.full_locators[0].locator[-1].get_normalized().docket_number == "19 Civ. 8034"
    assert document.full_locators[0].case_name
    assert document.full_locators[0].court
    assert document.full_locators[0].date
    assert document.short_reporters == ()
    assert "short_reporter_citations" not in document.stage_runs


def test_provider_error_in_successful_http_response_aborts_hunt(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _ready("See No. 19 Civ. 8034.")
    real_client = httpx.AsyncClient

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"code": 429, "message": "Upstream rate limit"}})

    monkeypatch.setattr(
        "mellea_lrc.llm.docket_review.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    reviewer = OpenRouterDocketReviewer(api_key="test", base_url="https://example.invalid", model="test")

    with pytest.raises(DocketReviewServiceError, match="Upstream rate limit"):
        asyncio.run(hunt_docket_locators(before, reviewer=reviewer))
    assert STAGE not in before.stage_runs


def test_structured_provider_answer_is_grounded_and_traced(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _ready("See No. 19 Civ. 8034.")
    real_client = httpx.AsyncClient
    decision = DocketSiteDecision(
        is_docket_citation=True,
        locator="No. 19 Civ. 8034",
        docket_number="19 Civ. 8034",
        reason="Court-assigned case identifier.",
    )

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "choices": [{"finish_reason": "stop", "message": {"content": decision.model_dump_json()}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 40},
            },
        )

    monkeypatch.setattr(
        "mellea_lrc.llm.docket_review.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    reviewer = OpenRouterDocketReviewer(api_key="test", base_url="https://example.invalid", model="test")
    document = asyncio.run(hunt_docket_locators(before, reviewer=reviewer))

    assert document.full_locators[0].locator[-1].get_normalized().docket_number == "19 Civ. 8034"
    assert document.site_reviews[0].outcome == "accepted"
    attempt = document.site_reviews[0].attempts[0]
    assert attempt.provider_id == "response-1"
    assert attempt.finish_reason == "stop"
    assert attempt.usage == {"prompt_tokens": 120, "completion_tokens": 40}
    assert attempt.request_json and attempt.response_json
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_provider_schema_repair_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _ready("See No. 19 Civ. 8034.")
    real_client = httpx.AsyncClient
    responses = iter(
        (
            '{"is_docket_citation":true,"complete_locator":"No. 19 Civ. 8034",'
            '"docket_number":"19 Civ. 8034","reason":"Case docket."}',
            DocketSiteDecision(
                is_docket_citation=True,
                locator="No. 19 Civ. 8034",
                docket_number="19 Civ. 8034",
                reason="Case docket.",
            ).model_dump_json(),
        )
    )

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": next(responses)}}]})

    monkeypatch.setattr(
        "mellea_lrc.llm.docket_review.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    reviewer = OpenRouterDocketReviewer(api_key="test", base_url="https://example.invalid", model="test")
    document = asyncio.run(hunt_docket_locators(before, reviewer=reviewer))

    assert document.site_reviews[0].outcome == "accepted"
    assert len(document.site_reviews[0].attempts) == 2
    assert "locator" in (document.site_reviews[0].attempts[0].feedback or "")
    assert document.site_reviews[0].attempts[1].feedback is None
