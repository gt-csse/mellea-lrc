"""Propositions preserve each filing occurrence and the IVR grounding trace."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from mellea_lrc.model import Document, Span
from mellea_lrc.model.citations.id import IdCitation
from mellea_lrc.model.citations.reporter_page_resolution import ReporterPageResolutionOutcome
from mellea_lrc.model.citations.reporter_pinpoint import PropositionDecision
from mellea_lrc.model.ivr import IvrAttempt, IvrRequirementAttempt, IvrRun
from mellea_lrc.validation.reporter_citation_page_resolution import resolve_reporter_citation_pages
from mellea_lrc.validation.reporter_citation_propositions import (
    SOURCE_STAGE,
    STAGE,
    read_reporter_citation_propositions,
)
from mellea_lrc.validation.reporter_citation_propositions import reviewer as service
from tests.test_reporter_citation_page_resolution import (
    _indexed,
    _indexed_document,
    _page,
    _retrieved,
)


def _ready(source, *, shorts=(), ids=()):
    document = _indexed_document(
        _retrieved(source, shorts=shorts, ids=ids),
        _indexed("20", _page("556"), _page("557")),
        _indexed("21"),
    )
    return resolve_reporter_citation_pages(document).complete(SOURCE_STAGE)


def _decision(*quotes):
    return PropositionDecision(
        quotes=quotes, reason="These passages preserve this occurrence's attributed use."
    )


class Reviewer:
    def __init__(self, decisions):
        self.decisions = decisions
        self.contexts = []

    async def __call__(self, context):
        self.contexts.append(context)
        return self.decisions[context.citation_id]


def _run(output, *, success=True, prefix=None, instruction="Read the attributed use", variables=None):
    return IvrRun(
        success=success,
        selected_attempt=0,
        attempts=(
            IvrAttempt(
                output=output,
                requirements=(
                    IvrRequirementAttempt(
                        description="Ground the filing quotes",
                        passed=success,
                        reason=None if success else "The proposed quote is absent from the excerpt.",
                        score=None,
                    ),
                ),
                request={"messages": [{"role": "system", "content": prefix}]},
                response={"finish_reason": "stop"},
            ),
        ),
        backend="FakeBackend",
        model="fake-model",
        model_options={"max_tokens": 3500},
        instruction=instruction,
        prefix=prefix,
        grounding_context={},
        user_variables=variables or {},
        output_schema=PropositionDecision.model_json_schema(),
    )


def test_distinct_occurrences_and_unpinned_id_keep_their_own_filing_propositions():
    first = "The pleadings must state a plausible claim."
    second = "The dissent would permit broader pleading."
    third = "Its approach emphasizes access to discovery."
    source = (
        f"{first} Alpha v. Beta, 550 U.S. 544, 556 (2007). "
        f"{second} Alpha, 550 U.S. at 557 (Justice B, dissenting). "
        f"{third} Id."
    )
    before = _ready(
        source,
        shorts=(("550 U.S. at 557", "557"),),
        ids=(("Id.", None),),
    )
    decisions = {
        citation.id: _decision(
            third if isinstance(citation, IdCitation) else first if citation in before.roots else second
        )
        for citation in before.citations
    }
    reviewer = Reviewer(decisions)

    after = asyncio.run(read_reporter_citation_propositions(before, reviewer=reviewer))

    assert len(reviewer.contexts) == 3
    for old, citation in zip(before.citations, after.citations):
        proposition = citation.reporter_propositions[-1]
        quote = decisions[citation.id].quotes[0]
        assert proposition.passages[0].quote == quote
        assert proposition.passages[0].span == Span(source.index(quote), source.index(quote) + len(quote))
        assert citation.root_id == old.root_id
        assert citation.pin_cite == old.pin_cite
        assert citation.reporter_page_resolutions == old.reporter_page_resolutions
    assert after.roots[0].identity_judgments == before.roots[0].identity_judgments
    assert after.roots[0].reporter_root_opinion_page_index == before.roots[0].reporter_root_opinion_page_index
    leaf = next(citation for citation in after.citations if isinstance(citation, IdCitation))
    assert leaf.pin_cite is None
    assert leaf.reporter_page_resolutions[-1].pin_citation_id != leaf.id
    assert leaf.reporter_propositions[-1].passages[0].quote == third


def test_whitespace_and_small_copying_error_store_original_quote_and_absolute_span():
    quote = "A complaint must state enough facts to plausibly establish an entitlement to relief."
    canonical = quote.replace("enough facts", "enough\n facts")
    source = "Unrelated filing material. " * 120 + canonical + " Alpha, 550 U.S. 544, 556 (2007)."
    before = _ready(source)
    reviewer = Reviewer({before.roots[0].id: _decision(quote.replace("complaint", "complainu"))})

    after = asyncio.run(read_reporter_citation_propositions(before, reviewer=reviewer))

    passage = after.roots[0].reporter_propositions[-1].passages[0]
    assert passage.quote == canonical
    assert passage.span == Span(source.index(canonical), source.index(canonical) + len(canonical))
    assert reviewer.contexts[0].source_offset > 0
    assert len(reviewer.contexts[0].citing_context) <= 1400 + len(reviewer.contexts[0].citation_quote) + 700


def test_negative_signal_and_explanatory_parenthetical_are_preserved_as_filing_use():
    quote = "Courts permit the procedure under ordinary circumstances."
    relation = "But see"
    explanation = "(rejecting the procedure when notice is absent)"
    source = f"{quote} {relation} Alpha, 550 U.S. 544, 556 (2007) {explanation}."
    before = _ready(source)
    reviewer = Reviewer({before.roots[0].id: _decision(quote, relation, explanation)})

    after = asyncio.run(read_reporter_citation_propositions(before, reviewer=reviewer))

    assert tuple(passage.quote for passage in after.roots[0].reporter_propositions[-1].passages) == (
        quote,
        relation,
        explanation,
    )


def test_table_of_authorities_records_no_proposition_without_initializing_llm(monkeypatch):
    source = "TABLE OF AUTHORITIES\nAlpha, 550 U.S. 544, 556 (2007) ........ 12, 18"
    before = _ready(source)
    before = Document.model_validate(
        {**before.model_dump(mode="python"), "index_spans": (Span(0, len(source)),)}
    )

    def fail(_profile):
        pytest.fail("An index occurrence must not initialize an LLM")

    monkeypatch.setattr(service.IvrReporterCitationPropositionReviewer, "from_profile", fail)
    after = asyncio.run(read_reporter_citation_propositions(before))

    record = after.roots[0].reporter_propositions[-1]
    assert record.decision.quotes == record.passages == ()
    assert "table of authorities" in record.decision.reason
    assert record.ivr is None


def test_absent_pin_skips_proposition_review():
    before = _ready("The standard governs. Alpha, 550 U.S. 544 (2007).")
    reviewer = Reviewer({})
    after = asyncio.run(read_reporter_citation_propositions(before, reviewer=reviewer))
    assert reviewer.contexts == []
    assert after.roots[0].reporter_propositions == ()
    assert after.roots[0].reporter_page_resolutions[-1].outcome is ReporterPageResolutionOutcome.NO_PIN


@pytest.mark.parametrize(
    "outcome",
    [
        ReporterPageResolutionOutcome.AMBIGUOUS,
        ReporterPageResolutionOutcome.UNLOCATED,
        ReporterPageResolutionOutcome.UNNORMALIZABLE,
    ],
)
def test_unresolved_pages_do_not_hide_the_filing_proposition(outcome):
    before = _ready("The standard governs. Alpha, 550 U.S. 544, 556 (2007).")
    root = before.roots[0].record("test_unresolved_resolution")
    resolution = root.reporter_page_resolutions[-1].model_copy(
        update={"node_id": root.nodes[-1].id, "outcome": outcome, "reason": "The source page needs review."}
    )
    before = before.replace_citation(root.with_reporter_page_resolution(resolution)).complete(
        "test_unresolved_resolution"
    )
    reviewer = Reviewer({before.roots[0].id: _decision("The standard governs.")})

    after = asyncio.run(read_reporter_citation_propositions(before, reviewer=reviewer))

    assert len(reviewer.contexts) == 1
    assert after.roots[0].reporter_propositions[-1].passages[0].quote == "The standard governs."


def test_empty_proposition_is_a_saved_reading_and_native_checkpoint_rewinds_exactly():
    before = _ready("Alpha, 550 U.S. 544, 556 (2007).")
    reviewer = Reviewer({before.roots[0].id: _decision()})
    after = asyncio.run(read_reporter_citation_propositions(before, reviewer=reviewer))
    restored = Document.model_validate_json(after.model_dump_json())

    assert restored == after
    assert restored.get_stage(SOURCE_STAGE) == before
    assert restored.get_stage(STAGE) == after
    assert restored.roots[0].reporter_propositions[-1].passages == ()
    assert restored.roots[0].reporter_propositions[-1].failure_reason is None


def test_ungrounded_custom_reviewer_decision_is_rejected_before_stage_commit():
    before = _ready("The standard governs. Alpha, 550 U.S. 544, 556 (2007).")
    reviewer = Reviewer(
        {before.roots[0].id: _decision("The opinion held that all claims must be dismissed.")}
    )

    with pytest.raises(ValueError, match="copied from the supplied filing excerpt"):
        asyncio.run(read_reporter_citation_propositions(before, reviewer=reviewer))
    assert STAGE not in before.stage_runs


def test_ivr_failed_grounding_trace_survives_serialization_and_provider_errors():
    before = _ready("The standard governs. Alpha, 550 U.S. 544, 556 (2007).")
    failed_run = _run(_decision("This is fabricated.").model_dump_json(), success=False)
    reviewer = Reviewer(
        {
            before.roots[0].id: service.ReporterCitationPropositionOutcome(
                None, run=failed_run, failure_reason="Grounding repair exhausted"
            )
        }
    )
    after = asyncio.run(read_reporter_citation_propositions(before, reviewer=reviewer))
    restored = Document.model_validate_json(after.model_dump_json())
    record = restored.roots[0].reporter_propositions[-1]
    assert record.ivr == failed_run
    assert record.passages == ()
    assert record.failure_reason == "Grounding repair exhausted"
    assert record.decision is None

    async def broken(context):
        raise RuntimeError("provider disconnected")

    after_error = asyncio.run(read_reporter_citation_propositions(before, reviewer=broken))
    assert (
        after_error.roots[0].reporter_propositions[-1].failure_reason == "RuntimeError: provider disconnected"
    )


def test_ivr_requirement_rejects_fabricated_quotes_and_preserves_context_and_repair(monkeypatch):
    quote = "The standard governs the pleading."
    before = _ready(f"{quote} Alpha, 550 U.S. 544, 556 (2007).")
    context = service.ReporterCitationPropositionContext.from_document(before, before.roots[0])
    good = _decision(quote)
    bad = _decision("The pleading establishes every legal element as a matter of law.")
    captured = {}

    async def fake_run(session, spec, *, strategy, model_options):
        captured.update(spec=spec, strategy=strategy, model_options=model_options)
        validation = spec.requirements[0].validation_fn
        invalid_schema = validation(SimpleNamespace(last_output=lambda: SimpleNamespace(value="{}")))
        assert invalid_schema.as_bool()
        invalid = validation(
            SimpleNamespace(last_output=lambda: SimpleNamespace(value=bad.model_dump_json()))
        )
        valid = validation(SimpleNamespace(last_output=lambda: SimpleNamespace(value=good.model_dump_json())))
        assert not invalid.as_bool()
        assert "copied from the supplied filing excerpt" in invalid.reason
        assert valid.as_bool()
        run = _run(
            good.model_dump_json(),
            prefix=spec.prefix,
            instruction=spec.description,
            variables=spec.user_variables,
        )
        repair = run.attempts[0].model_copy(
            update={
                "output": bad.model_dump_json(),
                "requirements": (
                    IvrRequirementAttempt(
                        description=spec.requirements[0].description,
                        passed=False,
                        reason=invalid.reason,
                        score=None,
                    ),
                ),
            }
        )
        return run.model_copy(update={"attempts": (repair, run.attempts[0]), "selected_attempt": 1})

    monkeypatch.setattr(service, "run_instruct_ivr", fake_run)
    reviewer = service.IvrReporterCitationPropositionReviewer(
        session=object(), model_options={"max_tokens": 3500}
    )
    outcome = asyncio.run(reviewer(context))

    assert outcome.decision == good
    assert outcome.run.attempts[0].output == bad.model_dump_json()
    assert outcome.run.attempts[1].output == good.model_dump_json()
    assert captured["spec"].user_variables["citing_context"] == context.citing_context
    assert captured["spec"].user_variables["source_offset"] == str(context.source_offset)
    assert captured["spec"].output_format is PropositionDecision
    assert "The standard governs" not in captured["spec"].prefix


def test_stage_guards_run_before_provider_and_reference_latest_resolution():
    before = _ready("The standard governs. Alpha, 550 U.S. 544, 556 (2007).")
    missing = before.get_stage("41_reporter_citation_page_resolution")
    reviewer = Reviewer({before.roots[0].id: _decision("The standard governs.")})
    with pytest.raises(ValueError, match="Complete reporter opinion selection"):
        asyncio.run(read_reporter_citation_propositions(missing, reviewer=reviewer))
    assert reviewer.contexts == []
    root = before.roots[0].record("test_second_resolution")
    resolution = root.reporter_page_resolutions[-1].model_copy(update={"node_id": root.nodes[-1].id})
    before = before.replace_citation(root.with_reporter_page_resolution(resolution)).complete(
        "test_second_resolution"
    )
    after = asyncio.run(read_reporter_citation_propositions(before, reviewer=reviewer))
    assert after.roots[0].reporter_propositions[-1].resolution_index == 1
    with pytest.raises(ValueError, match="already completed"):
        asyncio.run(read_reporter_citation_propositions(after, reviewer=reviewer))
