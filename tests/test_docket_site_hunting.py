"""Optional docket site hunting adds source-grounded full locators before colocation."""

import asyncio
from types import SimpleNamespace

import pytest
from mellea.backends import ModelOption

from mellea_lrc.api import (
    find_docket_locators,
    find_full_reporter_locators,
    grow_roots,
    hunt_docket_locators,
    resolve_colocations,
)
from mellea_lrc.extraction._site_hunting.candidates import suspected_dockets
from mellea_lrc.extraction._site_hunting.review import DocketSiteDecision, IvrDocketReviewer
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


def test_docket_item_reference_cannot_propose_its_inner_number_as_a_case_docket() -> None:
    before = _ready("In re Holdings, No. 21-11854 (DSJ) [D.I. No. 17].")

    assert [site.locator_text for site in suspected_dockets(before)] == ["No. 21-11854"]


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


def test_hunt_rejects_a_repeat_run_and_checkpoint_survives_later_colocation() -> None:
    source = "See Doe v. Townes, No. 19 Civ. 8034; Smith v. Jones, 347 U.S. 483."
    before = _ready(source)
    calls = 0

    async def reviewer(site: object) -> DocketSiteDecision:
        nonlocal calls
        calls += 1
        return _accept(site)

    hunted = asyncio.run(hunt_docket_locators(before, reviewer=reviewer))
    assert calls == 1
    with pytest.raises(ValueError):
        asyncio.run(hunt_docket_locators(hunted, reviewer=reviewer))
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


def test_hunt_continues_from_a_serialized_rule_checkpoint() -> None:
    before = _ready("See No. 19 Civ. 8034.")
    restored = Document.model_validate_json(before.model_dump_json())

    async def reviewer(site: object) -> DocketSiteDecision:
        return _accept(site)

    hunted = asyncio.run(hunt_docket_locators(restored, reviewer=reviewer))

    assert hunted.get_stage("docket_locators") == before
    assert hunted.stage_runs[-1] == STAGE
    assert [review.outcome for review in hunted.site_reviews] == ["accepted"]
    assert Document.model_validate_json(hunted.model_dump_json()) == hunted


def test_grow_roots_hunts_before_context_and_does_not_find_short_citations() -> None:
    source = "Doe v. Townes, No. 19 Civ. 8034 (S.D.N.Y. 2020). See Doe, 347 U.S. at 495."

    async def reviewer(site: object) -> DocketSiteDecision:
        return _accept(site)

    document = asyncio.run(grow_roots(Document.from_source(source), hunt_dockets=True, reviewer=reviewer))

    assert document.stage_runs[:5] == (
        "full_reporter_locators",
        "docket_locators",
        STAGE,
        "docket_entries",
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


def _mellea_context(output: str) -> SimpleNamespace:
    return SimpleNamespace(last_output=lambda: SimpleNamespace(value=output))


def _sampled_answers(requirements: object, outputs: list[str], *, success: bool) -> SimpleNamespace:
    """Model the public Mellea trace while exercising our installed requirements."""
    checks = [
        [(requirement, requirement.validation_fn(_mellea_context(output))) for requirement in requirements]
        for output in outputs
    ]
    requests: list[dict[str, str]] = [{"role": "user", "content": "Review the proposed docket."}]
    generations = []
    for index, output in enumerate(outputs):
        generations.append(
            SimpleNamespace(
                value=output,
                _generate_log=SimpleNamespace(
                    prompt=list(requests),
                    model_output={
                        "id": f"answer-{index}",
                        "choices": [{"finish_reason": "stop"}],
                        "usage": {"completion_tokens": 25 + index},
                    },
                ),
            )
        )
        requests.extend(
            [
                {"role": "assistant", "content": output},
                {
                    "role": "user",
                    "content": "\n".join(
                        validation.reason or ""
                        for _requirement, validation in checks[index]
                        if not validation.as_bool()
                    ),
                },
            ]
        )
    return SimpleNamespace(
        success=success,
        result_index=len(outputs) - 1,
        sample_generations=generations,
        sample_validations=checks,
    )


def test_ivr_reviewer_repairs_schema_and_grounding_then_persists_the_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _ready("See No. 19 Civ. 8034.")
    accepted = DocketSiteDecision(
        is_docket_citation=True,
        locator="No. 19 Civ. 8034",
        docket_number="19 Civ. 8034",
        reason="Court-assigned case identifier.",
    )
    wrong_number = accepted.model_copy(update={"docket_number": "19 Civ. 8035"})

    def instruct(_description: str, **kwargs: object) -> SimpleNamespace:
        assert kwargs["format"] is DocketSiteDecision
        assert kwargs["strategy"].loop_budget == 3
        assert "docket locator" in kwargs["model_options"][ModelOption.SYSTEM_PROMPT]
        sample = _sampled_answers(
            kwargs["requirements"],
            [
                '{"complete_locator":"No. 19 Civ. 8034"}',
                wrong_number.model_dump_json(),
                accepted.model_dump_json(),
            ],
            success=True,
        )
        assert not sample.sample_validations[0][0][1].as_bool()
        assert "is_docket_citation" in sample.sample_validations[0][0][1].reason
        assert sample.sample_validations[0][1][1].as_bool()  # Domain check waits for valid JSON.
        assert sample.sample_validations[1][0][1].as_bool()
        assert not sample.sample_validations[1][1][1].as_bool()
        assert all(validation.as_bool() for _requirement, validation in sample.sample_validations[2])
        return sample

    monkeypatch.setattr("mellea.stdlib.functional.instruct", instruct)
    reviewer = IvrDocketReviewer(
        session=SimpleNamespace(backend=SimpleNamespace(model_id="test-model")),
        model_options={"max_tokens": 1800},
        max_attempts=3,
    )
    document = asyncio.run(hunt_docket_locators(before, reviewer=reviewer))

    assert document.full_locators[0].locator[-1].get_normalized().docket_number == "19 Civ. 8034"
    assert document.site_reviews[0].outcome == "accepted"
    run = document.site_reviews[0].ivr
    assert run is not None and run.success
    assert len(run.attempts) == 3
    assert run.attempts[0].response["id"] == "answer-0"
    assert "docket-number portion" in run.attempts[1].requirements[1].reason
    assert run.attempts[2].request[-1]["content"] == run.attempts[1].requirements[1].reason
    assert run.output == accepted.model_dump_json()
    assert document.get_stage(STAGE) == document
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_exhausted_ivr_review_is_durable_without_creating_a_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _ready("See No. 19 Civ. 8034.")

    def instruct(_description: str, **kwargs: object) -> SimpleNamespace:
        return _sampled_answers(
            kwargs["requirements"],
            ['{"complete_locator":"No. 19 Civ. 8034"}', '{"complete_locator":"No. 19 Civ. 8034"}'],
            success=False,
        )

    monkeypatch.setattr("mellea.stdlib.functional.instruct", instruct)
    reviewer = IvrDocketReviewer(
        session=SimpleNamespace(backend=SimpleNamespace(model_id="test-model")),
        model_options={"max_tokens": 1800},
    )
    document = asyncio.run(hunt_docket_locators(before, reviewer=reviewer))

    assert document.citations == ()
    assert document.site_reviews[0].outcome == "failed"
    assert "is_docket_citation" in document.site_reviews[0].reason
    assert document.site_reviews[0].ivr is not None
    assert len(document.site_reviews[0].ivr.attempts) == 2
    assert Document.model_validate_json(document.model_dump_json()) == document
