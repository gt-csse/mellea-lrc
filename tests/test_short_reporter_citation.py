"""Short reporter occurrences stay separate from full locator roots."""

import asyncio

import pytest
from eyecite.models import FullCaseCitation, ShortCaseCitation
from eyecite.models import IdCitation as EyeciteIdCitation
from eyecite.models import SupraCitation as EyeciteSupraCitation

from mellea_lrc.api import (
    Document,
    attribute_short_reporter_citations,
    find_short_reporter_citations,
    grow_roots,
    resolve_short_reporter_case_names,
    resolve_short_reporter_colocations,
)
from mellea_lrc.parsing.reporters import full_reporter_readings
from mellea_lrc.extraction.leaf_attribution_review.reviewer import LeafReviewOutcome
from mellea_lrc.parsing.events import events
from mellea_lrc.extraction.short_reporter_locator import STAGE as SHORT_REPORTER_STAGE
from mellea_lrc.parsing.reporters import short_reporter_readings
from mellea_lrc.model import FullReporterCitation, ShortReporterCitation, Span
from mellea_lrc.model.citations import AttributionResult, LeafReviewDecision
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID
from mellea_lrc.model.ivr import IvrAttempt, IvrRun

ATTRIBUTION_STAGE = "29_short_reporter_attribution"
COLOCATION_STAGE = "28.1_short_reporter_colocations"
CASE_NAME_STAGE = "28.2_short_reporter_case_names"


def _read_short_names(document: Document) -> Document:
    return resolve_short_reporter_case_names(resolve_short_reporter_colocations(document))


def test_stage_readers_share_eyecite_matching_for_full_and_short_kinds() -> None:
    source = "Smith v. Jones, 347 U.S. 483. See Smith, 347 U.S. at 495."
    full = full_reporter_readings(source)
    short = short_reporter_readings(source)

    assert [type(reading.citation) for reading in full + short] == [FullCaseCitation, ShortCaseCitation]
    assert [source[slice(*reading.span)] for reading in full + short] == ["347 U.S. 483", "347 U.S. at 495"]


def test_short_reporter_is_a_distinct_checkpointed_citation() -> None:
    source = "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 347 U.S. at 495 n.4."
    roots = asyncio.run(grow_roots(Document.from_source(source)))
    document = find_short_reporter_citations(roots)

    assert len(document.full_locators) == len(document.roots) == 1
    assert isinstance(document.full_locators[0], FullReporterCitation)
    assert len(document.short_reporters) == 1
    short = document.short_reporters[0]
    assert isinstance(short, ShortReporterCitation)
    assert not hasattr(short, "locator_span")
    locator_quote = "347 U.S. at 495 n.4"
    assert short.short_locator_span == Span(
        source.index(locator_quote), source.index(locator_quote) + len(locator_quote)
    )
    assert short.short_locator[-1].quote == locator_quote
    assert (
        source[short.short_locator_span.start : short.short_locator_span.end] == short.short_locator[-1].quote
    )
    assert set(type(short.short_locator[-1].get_normalized()).model_fields) == {
        "volume",
        "reporter",
        "edition",
    }
    assert short.case_name == ()
    assert short.pin_cite[-1].quote == "495 n.4"
    assert short.pin_cite[-1].get_normalized()[0].footnote == "4"
    assert len(short.nodes) == 1
    assert short.pin_cite[-1].node_id == short.nodes[0].id
    assert short.root_id == ()
    assert document.get_stage("10_roots") == roots
    assert document.get_stage(SHORT_REPORTER_STAGE) == document
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_short_reporter_discovery_is_optional_and_rejects_a_repeat_run() -> None:
    source = "See Smith, 347 U.S. at 495."
    with pytest.raises(ValueError, match="Form full roots"):
        find_short_reporter_citations(Document.from_source(source))
    before = asyncio.run(grow_roots(Document.from_source(source)))
    after = find_short_reporter_citations(before)

    assert before.citations == ()
    assert after.full_locators == after.roots == ()
    assert len(after.short_reporters) == 1
    with pytest.raises(ValueError):
        find_short_reporter_citations(after)


def test_short_reporter_can_later_attach_without_changing_its_checkpoint() -> None:
    source = "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 347 U.S. at 495."
    found = find_short_reporter_citations(asyncio.run(grow_roots(Document.from_source(source))))
    short = found.short_reporters[0]
    attached = found.replace_citation(short.record("attach_short").with_root(found.roots[0].id))
    attached = attached.complete("attach_short")

    assert attached.short_reporters[0].root_id[-1].value == found.roots[0].id
    assert attached.get_stage(SHORT_REPORTER_STAGE) == found
    assert Document.model_validate_json(attached.model_dump_json()) == attached


def test_short_reporter_creation_reads_a_complete_multiline_pin_in_one_node() -> None:
    source = "Smith v. Jones, 347 U.S. 483 (1954). See Smith,\n 347 U.S. at\n 495 - 97, 501 n.2."
    roots = asyncio.run(grow_roots(Document.from_source(source)))
    created = find_short_reporter_citations(roots)
    short = created.short_reporters[0]

    assert short.short_locator[-1].quote == "347 U.S. at\n 495 - 97, 501 n.2"
    assert short.case_name == ()
    assert short.pin_cite[-1].quote == "495 - 97, 501 n.2"
    assert [(pin.first, pin.last, pin.footnote) for pin in short.pin_cite[-1].get_normalized()] == [
        (495, 497, None),
        (501, 501, "2"),
    ]
    assert len(short.nodes) == 1
    for field in (short.short_locator[-1], short.pin_cite[-1]):
        assert field.node_id == short.nodes[0].id
        assert source[field.span.start : field.span.end] == field.quote
    assert short.root_id == short.attributions == short.reviews == ()
    assert Document.model_validate_json(created.model_dump_json()) == created


def test_short_reporter_creation_retains_a_failed_descending_pin_reading() -> None:
    source = "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 347 U.S. at 495-490."
    created = find_short_reporter_citations(asyncio.run(grow_roots(Document.from_source(source))))
    short = created.short_reporters[0]

    assert short.short_locator[-1].quote == "347 U.S. at 495-490"
    assert short.short_locator[-1].normalizable is True
    assert short.short_locator[-1].get_normalized().volume == 347
    assert short.pin_cite[-1].quote == "495-490"
    assert short.pin_cite[-1].normalizable is False
    assert short.pin_cite[-1].normalization_error
    assert short.pin_cite[-1].node_id == short.nodes[0].id
    with pytest.raises(ValueError, match="not normalizable"):
        short.pin_cite[-1].get_normalized()
    assert Document.model_validate_json(created.model_dump_json()) == created


@pytest.mark.parametrize(
    ("intervening", "event_type"),
    [
        ("Id.", EyeciteIdCitation),
        ("Smith, supra, at 494.", EyeciteSupraCitation),
    ],
)
def test_name_reading_bounds_case_name_after_leaf_events_not_yet_created(intervening, event_type) -> None:
    source = f"Smith v. Jones, 347 U.S. 483 (1954). {intervening} Unknown, 347 U.S. at 495."
    created = find_short_reporter_citations(asyncio.run(grow_roots(Document.from_source(source))))
    created = _read_short_names(created)
    short = created.short_reporters[0]
    barriers = [event for event in events(source) if isinstance(event, event_type)]

    assert barriers
    assert created.short_citations == (short,)
    assert short.case_name[-1].quote == "Unknown"
    assert all(event.span_with_pincite()[1] <= short.case_name[-1].span.start for event in barriers)
    assert short.case_name[-1].node_id == next(
        node.id for node in short.nodes if node.stage == CASE_NAME_STAGE
    )


def test_independent_short_reporter_attribution_restores_creation_after_reload() -> None:
    source = "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 347 U.S. at 495."
    roots = asyncio.run(grow_roots(Document.from_source(source)))
    created = find_short_reporter_citations(roots)

    async def unexpected_review(_context):
        pytest.fail("A unique reporter/name agreement must skip semantic review")

    attributed = asyncio.run(
        attribute_short_reporter_citations(_read_short_names(created), reviewer=unexpected_review)
    )
    restored = Document.model_validate_json(attributed.model_dump_json())
    short = restored.short_reporters[0]

    assert short.root_id[-1].value == roots.roots[0].id
    assert short.attributions[-1].result is AttributionResult.ATTACHED
    assert short.attributions[-1].node_id == next(
        node.id for node in short.nodes if node.stage == ATTRIBUTION_STAGE
    )
    assert short.reviews == ()
    named = restored.get_stage(CASE_NAME_STAGE)
    assert short.case_name == named.short_reporters[0].case_name
    assert short.pin_cite == created.short_reporters[0].pin_cite
    assert restored.get_stage("10_roots") == roots
    assert restored.get_stage(SHORT_REPORTER_STAGE) == created
    assert restored.get_stage(ATTRIBUTION_STAGE) == attributed


@pytest.mark.parametrize("source", ["No authorities here.", "See Smith, 347 U.S. at 495."])
def test_short_reporter_attribution_requires_creation_and_rejects_repeats(source: str) -> None:
    roots = asyncio.run(grow_roots(Document.from_source(source)))
    with pytest.raises(ValueError):
        asyncio.run(attribute_short_reporter_citations(roots, review=False))
    created = find_short_reporter_citations(roots)
    with pytest.raises(ValueError, match="case names"):
        asyncio.run(attribute_short_reporter_citations(created, review=False))
    attributed = asyncio.run(attribute_short_reporter_citations(_read_short_names(created), review=False))

    assert attributed.stage_runs[-4:] == (
        SHORT_REPORTER_STAGE,
        COLOCATION_STAGE,
        CASE_NAME_STAGE,
        ATTRIBUTION_STAGE,
    )
    assert attributed.get_stage(SHORT_REPORTER_STAGE) == created
    if attributed.short_reporters:
        assert attributed.short_reporters[0].root_id[-1].value == WITHDRAWN_ROOT_ID
        assert attributed.short_reporters[0].attributions[-1].candidate_root_ids == ()
    with pytest.raises(ValueError, match="already completed"):
        asyncio.run(attribute_short_reporter_citations(attributed, review=False))


@pytest.mark.parametrize(
    "source",
    [
        "347 U.S. at 495. Smith v. Jones, 347 U.S. 483 (1954).",
        "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 348 U.S. at 495.",
    ],
)
def test_short_reporter_rules_do_not_borrow_a_future_or_different_reporter(source: str) -> None:
    created = find_short_reporter_citations(asyncio.run(grow_roots(Document.from_source(source))))
    attributed = asyncio.run(attribute_short_reporter_citations(_read_short_names(created), review=False))
    short = attributed.short_reporters[0]

    assert short.root_id[-1].value == WITHDRAWN_ROOT_ID
    assert short.attributions[-1].result is AttributionResult.UNRESOLVED
    assert short.attributions[-1].candidate_root_ids == ()


def test_unnamed_short_reporter_can_use_a_unique_preceding_reporter() -> None:
    source = "Smith v. Jones, 347 U.S. 483 (1954). 347 U.S. at 495."
    roots = asyncio.run(grow_roots(Document.from_source(source)))
    created = find_short_reporter_citations(roots)
    attributed = asyncio.run(attribute_short_reporter_citations(_read_short_names(created), review=False))

    assert created.short_reporters[0].case_name == ()
    assert attributed.short_reporters[0].root_id[-1].value == roots.roots[0].id


def test_unique_reporter_does_not_override_a_disagreeing_written_name() -> None:
    source = "Smith v. Jones, 347 U.S. 483 (1954). See Wrong, 347 U.S. at 495."
    roots = asyncio.run(grow_roots(Document.from_source(source)))
    created = find_short_reporter_citations(roots)
    attributed = asyncio.run(attribute_short_reporter_citations(_read_short_names(created), review=False))
    short = attributed.short_reporters[0]

    assert short.case_name[-1].quote == "Wrong"
    assert short.attributions[-1].candidate_root_ids == (roots.roots[0].id,)
    assert short.attributions[-1].result is AttributionResult.UNRESOLVED
    assert short.root_id[-1].value == WITHDRAWN_ROOT_ID


def test_ambiguous_short_reporter_review_records_separate_rule_and_review_nodes() -> None:
    source = "Smith v. Jones, 347 U.S. 483 (1954). Brown v. Green, 347 U.S. 500 (1954). 347 U.S. at 510."
    roots = asyncio.run(grow_roots(Document.from_source(source)))
    created = find_short_reporter_citations(roots)

    async def choose_brown(context):
        assert context.kind == "short_reporter"
        assert set(context.candidate_root_ids) == {root.id for root in roots.roots}
        index = next(
            index
            for index, candidate in enumerate(context.candidates)
            if candidate["case_name"] == "Brown v. Green"
        )
        return LeafReviewDecision(is_citation=True, root_index=index, reason="Source names Brown")

    attributed = asyncio.run(
        attribute_short_reporter_citations(_read_short_names(created), reviewer=choose_brown)
    )
    short = attributed.short_reporters[0]
    brown = next(root for root in roots.roots if root.get_case_name().plaintiff == "Brown")

    assert short.root_id[-1].value == brown.id
    assert [record.result for record in short.attributions] == [
        AttributionResult.UNRESOLVED,
        AttributionResult.ATTACHED,
    ]
    assert short.attributions[0].node_id == next(
        node.id for node in short.nodes if node.stage == ATTRIBUTION_STAGE
    )
    assert short.attributions[-1].node_id == short.reviews[-1].node_id == short.nodes[-1].id
    assert [entry.value for entry in short.root_id] == [WITHDRAWN_ROOT_ID, brown.id]
    restored = Document.model_validate_json(attributed.model_dump_json())
    assert restored.get_stage(SHORT_REPORTER_STAGE) == created
    assert restored == attributed


def test_short_reporter_failed_review_retains_complete_trace_and_rule_candidates() -> None:
    source = "Smith v. Jones, 347 U.S. 483 (1954). Brown v. Green, 347 U.S. 500 (1954). 347 U.S. at 510."
    created = find_short_reporter_citations(asyncio.run(grow_roots(Document.from_source(source))))
    trace = IvrRun(
        success=False,
        selected_attempt=0,
        attempts=(
            IvrAttempt(
                output="invalid choice",
                requirements=(),
                request=[{"role": "user", "content": "choose a source authority"}],
                response={"usage": {"output_tokens": 2}},
            ),
        ),
        backend="test-backend",
        model="test-model",
        model_options={"temperature": 0},
        instruction="choose a source authority",
        prefix="source filing",
        grounding_context={"site": "347 U.S. at 510"},
        user_variables={"kind": "short_reporter"},
        output_schema={"type": "object"},
    )

    async def fail(_context):
        return LeafReviewOutcome(None, run=trace, failure_reason="Synthetic review failure")

    attributed = asyncio.run(attribute_short_reporter_citations(_read_short_names(created), reviewer=fail))
    restored = Document.model_validate_json(attributed.model_dump_json())
    short = restored.short_reporters[0]

    assert short.root_id[-1].value == WITHDRAWN_ROOT_ID
    assert short.attributions[-1].result is AttributionResult.UNRESOLVED
    assert len(short.attributions[-1].candidate_root_ids) == 2
    assert short.reviews[-1].failure_reason == "Synthetic review failure"
    assert short.reviews[-1].ivr == trace
    assert restored.get_stage(SHORT_REPORTER_STAGE) == created


@pytest.mark.parametrize("is_citation", [True, False])
def test_short_reporter_review_can_decline_an_ambiguous_attachment(is_citation: bool) -> None:
    source = "Smith v. Jones, 347 U.S. 483 (1954). Brown v. Green, 347 U.S. 500 (1954). 347 U.S. at 510."
    created = find_short_reporter_citations(asyncio.run(grow_roots(Document.from_source(source))))

    async def decline(_context):
        return LeafReviewDecision(is_citation=is_citation, root_index=None, reason="No supported authority")

    attributed = asyncio.run(attribute_short_reporter_citations(_read_short_names(created), reviewer=decline))
    short = attributed.short_reporters[0]

    assert short.root_id[-1].value == WITHDRAWN_ROOT_ID
    assert short.attributions[-1].result is (
        AttributionResult.UNRESOLVED if is_citation else AttributionResult.REJECTED
    )
