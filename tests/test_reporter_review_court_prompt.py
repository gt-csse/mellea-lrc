"""Offline checks for court evidence sent to reporter model reviews."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from types import ModuleType

from mellea_lrc.courtlistener import CourtListenerCluster, CourtListenerDocket
from mellea_lrc.llm.ivr import InstructIvrSpec
from mellea_lrc.model import Span
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.fields.court import Court
from mellea_lrc.model.ivr import IvrAttempt, IvrRun
from mellea_lrc.validation._support import reporter_ambiguous_llm, reporter_unique_llm
from mellea_lrc.validation._support.reporter_court_context import (
    inferred_reporter_court_note,
    reporter_court_context,
)


def _cluster(identifier: int, court_id: str) -> CourtListenerCluster:
    return CourtListenerCluster.model_validate(
        {
            "id": identifier,
            "caseName": "Example v. Other",
            "court_id": court_id,
            "dateFiled": "2000-01-01",
            "citations": [{"volume": 123, "reporter": "F.3d", "page": "456"}],
        }
    )


def _docket(identifier: int, court_id: str) -> CourtListenerDocket:
    return CourtListenerDocket.model_validate({"id": identifier, "court_id": court_id})


def _context_fields() -> dict[str, object]:
    return {
        "source": "Example v. Other, 123 F.3d 456 (2d Cir. 2000).",
        "locator": "123 F.3d 456",
        "case_name_window": "Example v. Other, ",
        "case_name_offset": 0,
        "following_window": " (2d Cir. 2000).",
        "following_offset": 32,
        "current_case_name": "Example v. Other",
        "current_case_name_normalized": "Example v. Other",
        "current_court": "2d Cir.",
        "current_date": "2000",
        "has_case_name": True,
        "has_court": True,
        "has_date": True,
    }


def _failed_run(spec: InstructIvrSpec) -> IvrRun:
    return IvrRun(
        success=False,
        selected_attempt=0,
        attempts=(IvrAttempt(output="", requirements=()),),
        backend="test",
        model=None,
        model_options={},
        instruction=spec.description,
        prefix=spec.prefix,
        grounding_context={},
        user_variables=dict(spec.user_variables),
        output_schema=None,
    )


def _capture_spec(monkeypatch, module: ModuleType, reviewer: object, context: object) -> InstructIvrSpec:
    captured: list[InstructIvrSpec] = []

    async def fake_run(_session: object, spec: InstructIvrSpec, **_kwargs: object) -> IvrRun:
        captured.append(spec)
        return _failed_run(spec)

    monkeypatch.setattr(module, "run_instruct_ivr", fake_run)
    asyncio.run(reviewer(context))
    assert len(captured) == 1
    return captured[0]


def _assert_expanded_courts(court_context: object, opinion_id: str, docket_id: str) -> None:
    assert isinstance(court_context, dict)
    assert "opinion_cluster" in court_context
    assert "linked_docket" in court_context
    opinion = json.dumps(court_context["opinion_cluster"], ensure_ascii=False)
    docket = json.dumps(court_context["linked_docket"], ensure_ascii=False)
    assert f'"{opinion_id}"' in opinion
    assert Court.from_id(opinion_id).name in opinion
    assert f'"{docket_id}"' in docket
    assert Court.from_id(docket_id).name in docket
    assert Court.from_id(docket_id).name not in opinion
    assert Court.from_id(opinion_id).name not in docket


def _assert_court_guidance(spec: InstructIvrSpec) -> None:
    prompt = f"{spec.prefix or ''}\n{spec.description}".lower()
    assert re.search(r"\bequivalen(?:t|ce)\b", prompt)
    assert re.search(r"\bdistricts?\b", prompt)
    assert re.search(r"\bdepartments?\b", prompt)
    assert "naming convention" in prompt or "naming variation" in prompt


def test_unique_review_sends_raw_and_expanded_opinion_and_docket_courts(monkeypatch) -> None:
    context = reporter_unique_llm.ReporterUniqueReviewContext(
        **_context_fields(), candidate=_cluster(1, "ca2"), docket=_docket(10, "nysd")
    )
    reviewer = reporter_unique_llm.IvrReporterUniqueReviewer(session=object(), model_options={})

    spec = _capture_spec(monkeypatch, reporter_unique_llm, reviewer, context)

    assert "{{court_name_context}}" in spec.description
    assert json.loads(spec.user_variables["candidate"])["court_id"] == "ca2"
    assert json.loads(spec.user_variables["docket"])["court_id"] == "nysd"
    _assert_expanded_courts(json.loads(spec.user_variables["court_name_context"]), "ca2", "nysd")
    _assert_court_guidance(spec)


def test_ambiguous_review_sends_raw_and_expanded_courts_for_each_candidate(monkeypatch) -> None:
    context = reporter_ambiguous_llm.ReporterAmbiguousReviewContext(
        **_context_fields(),
        candidates=(_cluster(1, "ca2"), _cluster(2, "ca9")),
        dockets=(_docket(10, "nysd"), _docket(20, "cacd")),
        rule_results=({"court": "unavailable"}, {"court": "unavailable"}),
        passing_candidate_indices=(),
    )
    reviewer = reporter_ambiguous_llm.IvrReporterAmbiguousReviewer(session=object(), model_options={})

    spec = _capture_spec(monkeypatch, reporter_ambiguous_llm, reviewer, context)

    assert "{{candidates}}" in spec.description
    candidates = json.loads(spec.user_variables["candidates"])
    assert [item["candidate_index"] for item in candidates] == [0, 1]
    for item, opinion_id, docket_id in zip(candidates, ("ca2", "ca9"), ("nysd", "cacd"), strict=True):
        assert item["cluster"]["court_id"] == opinion_id
        assert item["linked_docket_court"]["court_id"] == docket_id
        _assert_expanded_courts(item["court_name_context"], opinion_id, docket_id)
    _assert_court_guidance(spec)


def test_court_context_expands_known_url_without_inventing_unknown_court() -> None:
    court_url = "https://www.courtlistener.com/api/rest/v4/courts/ca2/"
    unknown_court = "Unlisted Provincial Tribunal"
    candidate = CourtListenerCluster.model_validate({"id": 1, "court": court_url})
    docket = CourtListenerDocket.model_validate({"id": 10, "court": unknown_court})

    context = reporter_court_context(candidate, docket)

    assert context == {
        "opinion_cluster": {"court": {"id": "ca2", "full_name": Court.from_id("ca2").name}},
        "linked_docket": {},
    }
    assert json.loads(candidate.model_dump_json(exclude={"raw_json"}))["court"] == court_url
    assert json.loads(docket.model_dump_json(exclude={"raw_json"}))["court"] == unknown_court
    assert unknown_court not in json.dumps(context)


def _reporter_citation(source: str) -> FullReporterCitation:
    locator = "550 U.S. 544"
    start = source.index(locator)
    return FullReporterCitation.from_locator(
        citation_id="reporter:test",
        stage="1_full_reporter_locator",
        source=source,
        span=Span(start, start + len(locator)),
    )


def test_inferred_court_note_identifies_reporter_and_latest_court_only() -> None:
    source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."
    root = _reporter_citation(source).record("7_courts").with_inferred_court("scotus")

    note = inferred_reporter_court_note(root)

    assert note is not None
    assert root.locator[-1].get_normalized().edition in note
    assert root.locator[-1].get_normalized().reporter.name in note
    assert Court.from_id("scotus").name in note
    assert "No court label was extracted near this locator" in note
    assert "Treat this as court evidence" in note
    assert "explain any conflict" in note
    assert "forbid" not in note

    explicit_source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2d Cir. 2007)."
    explicit = (
        _reporter_citation(explicit_source)
        .record("7_courts")
        .with_inferred_court("scotus")
        .record("8_explicit_court")
        .with_court(
            explicit_source,
            Span(explicit_source.index("2d Cir."), explicit_source.index("2d Cir.") + len("2d Cir.")),
        )
    )
    assert inferred_reporter_court_note(explicit) is None


def test_inferred_court_note_is_conditional_in_both_reporter_review_prompts(monkeypatch) -> None:
    source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."
    root = _reporter_citation(source).record("7_courts").with_inferred_court("scotus")
    note = inferred_reporter_court_note(root)
    assert note is not None
    contexts_and_reviewers = (
        (
            reporter_unique_llm,
            reporter_unique_llm.ReporterUniqueReviewContext(
                **_context_fields(),
                candidate=_cluster(1, "scotus"),
                docket=None,
                inferred_court_note=note,
            ),
            reporter_unique_llm.IvrReporterUniqueReviewer(session=object(), model_options={}),
        ),
        (
            reporter_ambiguous_llm,
            reporter_ambiguous_llm.ReporterAmbiguousReviewContext(
                **_context_fields(),
                candidates=(_cluster(1, "scotus"),),
                dockets=(None,),
                rule_results=({"court": "match"},),
                passing_candidate_indices=(0,),
                inferred_court_note=note,
            ),
            reporter_ambiguous_llm.IvrReporterAmbiguousReviewer(session=object(), model_options={}),
        ),
    )

    for module, context, reviewer in contexts_and_reviewers:
        spec = _capture_spec(monkeypatch, module, reviewer, context)
        assert "{{inferred_court_note}}" in spec.description
        assert spec.user_variables["inferred_court_note"] == note
        assert "unavailable" in spec.prefix.lower()
        assert "not_stated" not in spec.prefix.lower()

        without_note = replace(context, inferred_court_note=None)
        plain_spec = _capture_spec(monkeypatch, module, reviewer, without_note)
        assert "{{inferred_court_note}}" not in plain_spec.description
        assert "inferred_court_note" not in plain_spec.user_variables
