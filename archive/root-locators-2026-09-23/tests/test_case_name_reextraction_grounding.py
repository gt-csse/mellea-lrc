"""Source grounding for case names recovered during reporter validation."""

import asyncio
from types import SimpleNamespace

import pytest

from mellea_lrc.courtlistener import CourtListenerOpinionCluster
from mellea_lrc.llm.ivr import IvrAttempt, IvrRun
from mellea_lrc.model.citations import FullCaseCitation, placed
from mellea_lrc.model.spans import Span
from mellea_lrc.validation.field_checks.mellea_case_name_reextraction import (
    _validate_grounding,
    run_mellea_case_name_reextraction,
)
from mellea_lrc.validation.field_checks.source_case_name import ground_source_case_name
from mellea_lrc.validation.types import (
    CitationValidation,
    ExactLocatorLookupNode,
    LocatorLookupOutcome,
    ValidationNodeStatus,
)
from tests.record_fixtures import read_citation


def test_reporter_reextraction_rejects_a_case_name_quote_absent_from_source() -> None:
    output = (
        '{"classification":"complete_case_name","case_name_quote":"Brown v. Board",'
        '"plaintiff":"Smith","defendant":"Jones"}'
    )
    context = SimpleNamespace(last_output=lambda: SimpleNamespace(value=output))

    validation = _validate_grounding(context, "Smith v. Jones, ")

    assert validation.as_bool() is False
    assert "case_name_quote" in validation.reason


def test_repeated_case_name_quote_attaches_to_occurrence_nearest_locator() -> None:
    before_locator = "Smith v. Jones appears in prose. Smith v. Jones, "
    source_start = 41

    grounded = ground_source_case_name(
        "Smith v. Jones",
        before_locator=before_locator,
        source_start=source_start,
    )

    assert grounded is not None
    assert grounded.span == Span(
        source_start + before_locator.rfind("Smith v. Jones"),
        source_start + before_locator.rfind("Smith v. Jones") + len("Smith v. Jones"),
    )
    assert grounded.text == "Smith v. Jones"


def test_reporter_reader_preserves_a_grounded_matter_of_caption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caption = "Matter of M4 Enters., Inc."
    locator = "157 F.R.D. 477"
    text = f"{caption}, {locator}."
    record = read_citation(
        citation_id="cite-matter",
        fields=placed(
            FullCaseCitation(volume="157", reporter="F.R.D.", page="477"),
            span=Span(0, len(text)),
            locator_span=Span(text.index(locator), text.index(locator) + len(locator)),
            matched_text=locator,
        ),
    )
    lookup = ExactLocatorLookupNode(
        node_id="cite-matter:exact_locator_lookup",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorLookupOutcome.FOUND,
        locator=locator,
        cluster=CourtListenerOpinionCluster(case_name=caption),
        candidate_count=1,
    )
    output = (
        '{"classification":"complete_case_name",'
        '"case_name_quote":"Matter of M4 Enters., Inc.",'
        '"plaintiff":null,"defendant":"M4 Enters., Inc."}'
    )

    async def fake_instruct(_session: object, _spec: object, **_kwargs: object) -> IvrRun:
        return IvrRun(
            success=True,
            selected_attempt=0,
            attempts=(IvrAttempt(output=output, requirements=()),),
            backend="test",
            model="test-model",
            model_options={},
            instruction="",
            prefix=None,
            grounding_context={},
            user_variables={},
            output_schema=None,
        )

    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test-model")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test-key")
    monkeypatch.setattr(
        "mellea_lrc.validation.field_checks.mellea_case_name_reextraction.run_instruct_ivr",
        fake_instruct,
    )

    node = asyncio.run(
        run_mellea_case_name_reextraction(
            CitationValidation(citation=record).append(lookup),
            trigger=lookup,
            locator_lookup=lookup,
            document_text=text,
            session=object(),
        )
    )

    assert node.status is ValidationNodeStatus.SUCCEEDED
    assert node.case_name is not None
    assert node.case_name.text == caption
    assert node.case_name.span == Span(0, len(caption))
    assert node.case_name.plaintiff is None
    assert node.case_name.defendant == "M4 Enters., Inc."
