"""Scoped opinion selection stays occurrence-specific and keeps repair evidence."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from mellea_lrc.model import Document, FullReporterCitation, Span
from mellea_lrc.model.citations.fields.pin_cite import PinCiteKind
from mellea_lrc.model.citations.id import IdCitation
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.citations.reporter_opinion import (
    OpinionRetrievalOutcome,
    ReporterRootOpinionRetrieval,
    ReporterRootOpinionSource,
    RetrievedReporterOpinion,
)
from mellea_lrc.model.citations.reporter_page_resolution import (
    OpinionPageReference,
    ReporterCitationOpinionDecision,
    ReporterCitationPageResolution,
    ReporterOpinionPageChoice,
    ReporterPageCandidates,
    ReporterPageResolutionOutcome,
)
from mellea_lrc.model.citations.short_reporter import ShortReporterCitation
from mellea_lrc.model.ivr import IvrAttempt, IvrRequirementAttempt, IvrRun
from mellea_lrc.providers.courtlistener.models import CourtListenerCluster
from mellea_lrc.validation.reporter_citation_opinion_review import (
    SOURCE_STAGE,
    STAGE,
    review_reporter_citation_opinions,
)
from mellea_lrc.validation.reporter_citation_opinion_review import reviewer as service
from mellea_lrc.validation.reporter_citation_page_resolution import resolve_reporter_citation_pages
from mellea_lrc.validation.reporter_root_opinion_page_index import index_reporter_root_opinion_pages


def _span(source, quote):
    start = source.index(quote)
    return Span(start, start + len(quote))


def _fixture(
    *,
    root_outcome=ReporterPageResolutionOutcome.AMBIGUOUS,
    leaf_outcome=ReporterPageResolutionOutcome.AMBIGUOUS,
    include_id=False,
):
    source = (
        "Alpha v. Beta, 550 U.S. 544, 556-557 (2007). "
        "Later the dissent is discussed: Alpha, 550 U.S. at 556 (Justice B, dissenting)."
    )
    if include_id:
        source += " Id."
    document = Document.from_source(source)
    root = FullReporterCitation.from_locator(
        citation_id="root",
        stage="1_full_reporter_locators",
        source=source,
        span=_span(source, "550 U.S. 544"),
    )
    document = document.add_citation(root).complete("1_full_reporter_locators")
    root = root.record("10_roots").with_root(root.id).with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
    root = root.with_pin_cite(source, _span(source, "556-557"))
    document = document.replace_citation(root).complete("10_roots")
    leaf_span = _span(source, "550 U.S. at 556")
    leaf = ShortReporterCitation.from_short_locator(
        citation_id="leaf",
        stage="28_short_reporter_citations",
        source=source,
        span=leaf_span,
        pin_cite_span=Span(leaf_span.end - 3, leaf_span.end),
    )
    document = document.add_citation(leaf).complete("28_short_reporter_citations")
    leaf = leaf.record("29_short_reporter_attribution").with_root(root.id)
    document = document.replace_citation(leaf).complete("29_short_reporter_attribution")
    if include_id:
        id_citation = IdCitation.from_source(
            source=source, span=_span(source, "Id."), stage="32_id_citations"
        )
        document = document.add_citation(id_citation).complete("32_id_citations")
        id_citation = id_citation.record("33_id_attribution").with_root(root.id)
        document = document.replace_citation(id_citation).complete("33_id_attribution")
    root = root.record("39_reporter_root_opinion_retrieval")
    root = root.with_reporter_root_opinion_source(
        ReporterRootOpinionSource(
            node_id=root.nodes[-1].id,
            cluster=CourtListenerCluster.model_validate(
                {
                    "id": 1,
                    "sub_opinions": [20, 21],
                    "citations": [{"volume": 550, "reporter": "U.S.", "page": "544"}],
                }
            ),
        )
    )
    opinions = []
    for identifier, author, opinion_type, name in (
        ("20", "Justice A", "020lead", "Lead"),
        ("21", "Justice B", "040dissent", "Dissent"),
    ):
        html = (
            '<page-number label="556" volume="550" edition="U.S.">*556</page-number>'
            f"{name} first page. "
            '<page-number label="557" volume="550" edition="U.S.">*557</page-number>'
            f"{name} second page."
        )
        opinions.append(
            RetrievedReporterOpinion(
                opinion_id=identifier,
                cluster_id="1",
                outcome=OpinionRetrievalOutcome.RETRIEVED,
                response={
                    "id": identifier,
                    "cluster": "1",
                    "type": opinion_type,
                    "author_str": author,
                    "html_with_citations": html,
                },
            )
        )
    root = root.with_reporter_root_opinion_retrieval(
        ReporterRootOpinionRetrieval(
            node_id=root.nodes[-1].id,
            cluster_id="1",
            sub_opinion_ids=("20", "21"),
            opinions=tuple(opinions),
        )
    )
    document = document.replace_citation(root).complete("39_reporter_root_opinion_retrieval")
    document = index_reporter_root_opinion_pages(document)
    for citation in document.citations:
        outcome = root_outcome if citation.id == "root" else leaf_outcome
        recorded = citation.record(SOURCE_STAGE)
        requested = []
        if outcome in {ReporterPageResolutionOutcome.AMBIGUOUS, ReporterPageResolutionOutcome.RESOLVED}:
            for page_index in range(2 if citation.id == "root" else 1):
                identifiers = ("20",) if outcome is ReporterPageResolutionOutcome.RESOLVED else ("20", "21")
                requested.append(
                    ReporterPageCandidates(
                        target_index=0,
                        label=556 + page_index,
                        kind=PinCiteKind.PAGE,
                        candidates=tuple(
                            OpinionPageReference(
                                opinion_id=identifier, page_index=page_index, pagination_confirmed=True
                            )
                            for identifier in identifiers
                        ),
                    )
                )
        resolution = ReporterCitationPageResolution(
            node_id=recorded.nodes[-1].id,
            root_id="root",
            locator_citation_id="leaf" if isinstance(citation, IdCitation) else "root",
            locator_reading_index=0,
            pin_citation_id="leaf" if isinstance(citation, IdCitation) else citation.id,
            pin_reading_index=0,
            outcome=outcome,
            pages=tuple(requested),
            reason="Synthetic explicit source page candidates.",
        )
        document = document.replace_citation(recorded.with_reporter_page_resolution(resolution))
    return document.complete(SOURCE_STAGE)


def _decision(*choices):
    return ReporterCitationOpinionDecision(
        choices=tuple(
            ReporterOpinionPageChoice(page_index=page, candidate_index=candidate)
            for page, candidate in choices
        ),
        reason="The filing context identifies this representative writing.",
    )


def _run(output, *, success=True, prefix=None, reason=None):
    return IvrRun(
        success=success,
        selected_attempt=0,
        attempts=(
            IvrAttempt(
                output=output,
                requirements=(
                    IvrRequirementAttempt(
                        description="Supplied page candidates",
                        passed=success,
                        reason=reason,
                        score=None,
                    ),
                ),
                request={"messages": [{"role": "system", "content": prefix}]},
                response={"finish_reason": "stop"},
            ),
        ),
        backend="FakeBackend",
        model="fake-model",
        model_options={"max_tokens": 5000},
        instruction="Select the supplied page candidates",
        prefix=prefix,
        grounding_context={},
        user_variables={},
        output_schema=ReporterCitationOpinionDecision.model_json_schema(),
    )


class Reviewer:
    def __init__(self, decisions):
        self.decisions = decisions
        self.contexts = []

    async def __call__(self, context):
        self.contexts.append(context)
        return self.decisions[context.citation_id]


def test_root_and_leaf_select_independent_writings_and_ranges_can_cross_them():
    before = _fixture()
    reviewer = Reviewer({"root": _decision((0, 0), (1, 1)), "leaf": _decision((0, 1))})

    after = asyncio.run(review_reporter_citation_opinions(before, reviewer=reviewer))
    restored = Document.model_validate_json(after.model_dump_json())

    assert restored == after and restored.get_stage(SOURCE_STAGE) == before
    assert [context.citation_id for context in reviewer.contexts] == ["root", "leaf"]
    assert reviewer.contexts[0].prefix == reviewer.contexts[1].prefix
    assert reviewer.contexts[0].citation_span != reviewer.contexts[1].citation_span
    shared = reviewer.contexts[0].prefix
    assert "040dissent" in shared and "Justice B" in shared and "Lead first page." in shared
    assert reviewer.contexts[0].page_candidates[0]["candidates"][1]["text"].startswith("Dissent first page.")
    for original, current in zip(before.citations, after.citations, strict=True):
        assert current.nodes[:-1] == original.nodes
        assert current.nodes[-1].stage == STAGE
        assert current.reporter_opinion_reviews[-1].resolution_index == 0
        assert current.reporter_page_resolutions == original.reporter_page_resolutions
        assert current.pin_cite == original.pin_cite and current.root_id == original.root_id
    assert after.roots[0].identity_judgments == before.roots[0].identity_judgments
    assert [page.opinion_id for page in restored.roots[0].get_reporter_page_selection()] == ["20", "21"]
    assert restored.citations[1].get_reporter_page_selection()[0].opinion_id == "21"
    assert before.roots[0].get_reporter_page_selection() == (None, None)


def test_actual_page_resolver_and_inherited_id_supply_effective_readings_to_review():
    indexed = _fixture(include_id=True).get_stage("40_reporter_root_opinion_page_index")
    before = resolve_reporter_citation_pages(indexed)
    id_citation = before.citations[-1]
    reviewer = Reviewer(
        {"root": _decision((0, 0), (1, 0)), "leaf": _decision((0, 1)), id_citation.id: _decision((0, 1))}
    )

    after = asyncio.run(review_reporter_citation_opinions(before, reviewer=reviewer))

    context = reviewer.contexts[-1]
    assert context.citation_quote == "Id."
    assert context.target_readings["locator"]["quote"] == "550 U.S. at 556"
    assert context.target_readings["locator"]["normalized"] == {"volume": 550, "edition": "U.S."}
    assert context.target_readings["pin_cite"]["quote"] == "556"
    assert context.target_readings["pin_cite"]["citation_id"] == "leaf"
    assert context.target_readings["pin_cite"]["normalized"][0]["first"] == 556
    assert id_citation.pin_cite is None and after.citations[-1].pin_cite is None
    assert all(item.prefix == context.prefix for item in reviewer.contexts)
    assert after.get_stage(SOURCE_STAGE) == before


def test_review_points_to_latest_absolute_resolution_index():
    before = _fixture(root_outcome=ReporterPageResolutionOutcome.RESOLVED)
    leaf = before.citations[1].record("41.1_page_revision")
    resolution = leaf.reporter_page_resolutions[-1].model_copy(update={"node_id": leaf.nodes[-1].id})
    before = before.replace_citation(leaf.with_reporter_page_resolution(resolution)).complete(
        "41.1_page_revision"
    )
    reviewer = Reviewer({"leaf": _decision((0, 1))})

    after = asyncio.run(review_reporter_citation_opinions(before, reviewer=reviewer))

    assert after.citations[1].reporter_opinion_reviews[-1].resolution_index == 1
    assert after.get_stage("41.1_page_revision") == before


@pytest.mark.parametrize(
    "outcome",
    [
        ReporterPageResolutionOutcome.RESOLVED,
        ReporterPageResolutionOutcome.UNLOCATED,
        ReporterPageResolutionOutcome.NO_PIN,
        ReporterPageResolutionOutcome.UNNORMALIZABLE,
    ],
)
def test_nonambiguous_resolution_never_calls_reviewer(outcome):
    before = _fixture(root_outcome=outcome, leaf_outcome=outcome)
    reviewer = Reviewer({})

    after = asyncio.run(review_reporter_citation_opinions(before, reviewer=reviewer))

    assert reviewer.contexts == [] and after.citations == before.citations
    assert after.stage_runs[-1] == STAGE


def test_null_choices_preserve_unresolved_ambiguity():
    before = _fixture()
    reviewer = Reviewer({"root": _decision((0, None), (1, None)), "leaf": _decision((0, None))})

    after = asyncio.run(review_reporter_citation_opinions(before, reviewer=reviewer))

    for citation in after.citations:
        assert citation.reporter_page_resolutions[-1].outcome is ReporterPageResolutionOutcome.AMBIGUOUS
        assert all(
            choice.candidate_index is None
            for choice in citation.reporter_opinion_reviews[-1].decision.choices
        )
        assert all(page is None for page in citation.get_reporter_page_selection())


@pytest.mark.parametrize(
    "bad", [_decision((0, 0)), _decision((0, 0), (1, 0), (2, 0)), _decision((0, 2), (1, 0))]
)
def test_invalid_complete_page_or_candidate_domain_raises_at_stage_boundary(bad):
    before = _fixture()
    reviewer = Reviewer({"root": bad})

    with pytest.raises(ValueError, match="page_index"):
        asyncio.run(review_reporter_citation_opinions(before, reviewer=reviewer))

    assert STAGE not in before.stage_runs and before.roots[0].reporter_opinion_reviews == ()


def test_exhausted_ivr_failure_is_persisted_without_a_choice():
    before = _fixture(root_outcome=ReporterPageResolutionOutcome.RESOLVED)
    trace = _run("{}", success=False, reason="candidate_index is outside the supplied list")
    reviewer = Reviewer(
        {
            "leaf": service.ReporterCitationOpinionOutcome(
                None,
                run=trace,
                failure_reason=trace.failure_reason,
            )
        }
    )

    after = asyncio.run(review_reporter_citation_opinions(before, reviewer=reviewer))
    restored = Document.model_validate_json(after.model_dump_json())

    review = restored.citations[1].reporter_opinion_reviews[-1]
    assert review.ivr == trace and review.decision is None
    assert review.failure_reason == trace.failure_reason
    assert restored.get_stage(SOURCE_STAGE) == before


def test_provider_exception_is_recorded_as_a_review_failure():
    before = _fixture(root_outcome=ReporterPageResolutionOutcome.RESOLVED)

    async def fail(_context):
        raise RuntimeError("Synthetic provider failure")

    after = asyncio.run(review_reporter_citation_opinions(before, reviewer=fail))

    review = after.citations[1].reporter_opinion_reviews[-1]
    assert review.failure_reason == "RuntimeError: Synthetic provider failure" and review.decision is None


def test_domain_feedback_and_shared_ivr_prefix_keep_exact_repair_trace(monkeypatch):
    before = _fixture()
    specs = []

    async def run(_session, spec, *, strategy, model_options):
        specs.append(spec)
        assert strategy.loop_budget == 3 and model_options["max_tokens"] == 5000
        requested = json.loads(spec.user_variables["page_candidates"])
        candidate = 0 if len(specs) == 1 else 1
        valid = _decision(*((page["page_index"], candidate) for page in requested))
        bad = _decision(*((page["page_index"], 99) for page in requested))
        ctx = SimpleNamespace(last_output=lambda: SimpleNamespace(value=bad.model_dump_json()))
        failed = spec.requirements[0].validation_fn(ctx)
        assert not failed.as_bool() and "candidate_index" in failed.reason
        passed = spec.requirements[0].validation_fn(
            SimpleNamespace(last_output=lambda: SimpleNamespace(value=valid.model_dump_json()))
        )
        assert passed.as_bool()
        initial = _run(bad.model_dump_json(), success=False, prefix=spec.prefix, reason=failed.reason)
        final = _run(valid.model_dump_json(), prefix=spec.prefix)
        return final.model_copy(
            update={"attempts": (*initial.attempts, *final.attempts), "selected_attempt": 1}
        )

    monkeypatch.setattr(service, "run_instruct_ivr", run)
    reviewer = service.IvrReporterCitationOpinionReviewer(
        session=object(), model_options={"max_tokens": 5000}
    )
    after = asyncio.run(review_reporter_citation_opinions(before, reviewer=reviewer))

    assert specs[0].prefix == specs[1].prefix
    assert specs[0].user_variables["citation_span"] != specs[1].user_variables["citation_span"]
    for citation in after.citations:
        trace = citation.reporter_opinion_reviews[-1].ivr
        assert len(trace.attempts) == 2 and trace.selected_attempt == 1
        assert trace.attempts[0].requirements[0].reason.startswith("For page_index")
        assert trace.attempts[0].request["messages"][0]["content"] == specs[0].prefix


def test_stage_guards_raise_before_review_calls():
    before = _fixture()
    reviewer = Reviewer({"root": _decision((0, 0), (1, 0)), "leaf": _decision((0, 0))})
    after = asyncio.run(review_reporter_citation_opinions(before, reviewer=reviewer))

    with pytest.raises(ValueError, match="already completed"):
        asyncio.run(review_reporter_citation_opinions(after, reviewer=reviewer))
    with pytest.raises(ValueError, match="Resolve reporter citation pages"):
        asyncio.run(
            review_reporter_citation_opinions(before.get_stage("40_reporter_root_opinion_page_index"))
        )


def test_selection_getter_requires_resolution_and_review_cannot_append_twice_to_one_node():
    before = _fixture()
    with pytest.raises(ValueError, match="have not been resolved"):
        before.get_stage("40_reporter_root_opinion_page_index").roots[0].get_reporter_page_selection()
    reviewed = asyncio.run(
        review_reporter_citation_opinions(
            before,
            reviewer=Reviewer({"root": _decision((0, 0), (1, 1)), "leaf": _decision((0, 0))}),
        )
    ).roots[0]
    with pytest.raises(ValueError, match="already recorded"):
        reviewed.with_reporter_opinion_review(reviewed.reporter_opinion_reviews[-1])
