"""A unique exact reporter lookup gets one combined case-name review."""

from __future__ import annotations

import asyncio

import pytest

from mellea_lrc.api import (
    form_roots,
    lookup_full_reporter_locators_exact,
    validate_unique_full_reporter_locator_identities,
)
from mellea_lrc.courtlistener import (
    CourtListenerCitationLookup,
    CourtListenerOpinionCluster,
    CourtListenerOpinionClusterCitation,
)
from mellea_lrc.llm.ivr import IvrAttempt, IvrRun
from mellea_lrc.model.case_names import CaseName
from mellea_lrc.model.citations import CitationDate, FullCaseCitation, placed
from mellea_lrc.model.document import Document
from mellea_lrc.model.extraction_metadata import ExtractionMetadata
from mellea_lrc.model.record import Question
from mellea_lrc.model.spans import Span
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.validation.candidates.evaluation import run_locator_candidate_evaluation
from mellea_lrc.validation.field_checks.exact_case_name_check import run_exact_case_name_check
from mellea_lrc.validation.field_checks.mellea_case_name_review import (
    run_mellea_case_name_review,
)
from mellea_lrc.validation.types import (
    CitationValidation,
    ExactLocatorLookupNode,
    LocatorLookupOutcome,
    MelleaCaseNameCheckOutcome,
    MelleaCaseNameReviewNode,
    ValidationNodeStatus,
)
from tests.record_fixtures import read_citation

_LOCATOR = "139 A.D.3d 695"
_RETRIEVED_NAME = "Ramirez v. City of New York"


class _UniqueLookupClient:
    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        assert (volume, reporter, page) == ("139", "A.D.3d", "695")
        return CourtListenerCitationLookup(
            citation=_LOCATOR,
            status=200,
            clusters=(
                CourtListenerOpinionCluster(
                    cluster_id="ramirez",
                    case_name=_RETRIEVED_NAME,
                    date_filed="2016-05-12",
                    citations=(CourtListenerOpinionClusterCitation("139", "A.D.3d", "695"),),
                ),
            ),
        )


def _source_document(source_name: str, extracted_name: str) -> Document:
    text = f"{source_name}, {_LOCATOR} (2016)."
    source = preprocess(text)
    locator_start = text.index(_LOCATOR)
    extracted = CaseName(span=Span(0, len(extracted_name)), text=extracted_name)
    citation = read_citation(
        citation_id="cite-exact",
        fields=placed(
            FullCaseCitation(
                case_name=extracted,
                volume="139",
                reporter="A.D.3d",
                page="695",
                date=CitationDate(year="2016"),
            ),
            span=Span(0, len(text)),
            locator_span=Span(locator_start, locator_start + len(_LOCATOR)),
            matched_text=_LOCATOR,
        ),
    )
    return form_roots(
        Document(
            source_metadata=source.source_metadata,
            text=text,
            preprocessing_metadata=source.preprocessing_metadata,
            citations=(citation,),
            extraction_metadata=ExtractionMetadata(),
        )
    )


def _review_node(trigger, *, outcome, case_name, status=ValidationNodeStatus.SUCCEEDED):
    return MelleaCaseNameReviewNode(
        node_id=f"{trigger.node_id}:mellea_case_name_review",
        status=status,
        outcome=outcome,
        retrieved_case_name=_RETRIEVED_NAME,
        case_name=case_name,
        plaintiff=case_name.plaintiff if case_name is not None else None,
        defendant=case_name.defendant if case_name is not None else None,
        rationale="One combined source reading and comparison.",
        depends_on=(trigger.node_id,),
    )


def _run_unique(document: Document, *, monkeypatch, outcome, reviewed_name):
    client = _UniqueLookupClient()
    calls = []

    async def review(_validation, *, trigger, locator_lookup, candidate, document_text, session):
        assert locator_lookup.candidate_count == 1
        assert candidate.candidate_index == 1
        assert _LOCATOR in document_text
        calls.append(trigger)
        return _review_node(
            trigger,
            outcome=outcome,
            case_name=reviewed_name,
            status=(
                ValidationNodeStatus.FAILED
                if outcome is MelleaCaseNameCheckOutcome.FAILED
                else ValidationNodeStatus.SUCCEEDED
            ),
        )

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("Unique exact lookup must not run another model stage")

    monkeypatch.setattr("mellea_lrc.validation.execution.run_mellea_case_name_review", review)
    monkeypatch.setattr("mellea_lrc.validation.execution.run_mellea_locator_candidate_choice", forbidden)

    looked_up = asyncio.run(lookup_full_reporter_locators_exact(document, client=client))
    completed = asyncio.run(validate_unique_full_reporter_locator_identities(looked_up, client=client))
    root = completed.citations[0]
    review_nodes = [
        node for node in root.trace if node.details.get("validation_node_type") == "MelleaCaseNameReviewNode"
    ]
    assert len(calls) == 1
    assert len(review_nodes) == 1
    assert not any(
        node.details.get("validation_node_type")
        in {"MelleaLocatorCandidateChoiceNode", "MelleaCaseNameCheckNode", "MelleaCaseNameReextractionNode"}
        for node in root.trace
    )
    return completed


def test_unique_exact_reviewed_mismatch_is_no_match_despite_matching_locator_and_year(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_name = "Cadle Co. v. Ayala"
    document = _source_document(source_name, source_name)
    grounded_name = CaseName(span=Span(0, len(source_name)), text=source_name)

    completed = _run_unique(
        document,
        monkeypatch=monkeypatch,
        outcome=MelleaCaseNameCheckOutcome.MISMATCH,
        reviewed_name=grounded_name,
    )

    root = completed.citations[0]
    assert root.judgement(Question.IDENTITY).outcome == "no_match"
    assert root.found is None


def test_unique_exact_reviewed_source_correction_can_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _source_document(_RETRIEVED_NAME, "Ramirez")
    corrected = CaseName(span=Span(0, len(_RETRIEVED_NAME)), text=_RETRIEVED_NAME)

    completed = _run_unique(
        document,
        monkeypatch=monkeypatch,
        outcome=MelleaCaseNameCheckOutcome.MATCH,
        reviewed_name=corrected,
    )

    root = completed.citations[0]
    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert root.case_name == corrected
    assert root.found is not None
    assert root.found.cluster_id == "ramirez"


def test_unique_exact_failed_review_defers_without_admitting_or_rejecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _source_document("Cadle Co. v. Ayala", "Cadle Co. v. Ayala")

    completed = _run_unique(
        document,
        monkeypatch=monkeypatch,
        outcome=MelleaCaseNameCheckOutcome.FAILED,
        reviewed_name=None,
    )

    root = completed.citations[0]
    assert root.judgement(Question.IDENTITY).outcome.startswith("deferred_")
    assert root.found is None


def test_combined_review_makes_one_grounded_model_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_name = "Cadle Co. v. Ayala"
    document = _source_document(source_name, source_name)
    cluster = _UniqueLookupClient().lookup_citation("139", "A.D.3d", "695").clusters[0]
    validation = CitationValidation(citation=document.citations[0])
    lookup = ExactLocatorLookupNode(
        node_id="cite-exact:exact_locator_lookup",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorLookupOutcome.FOUND,
        locator=_LOCATOR,
        cluster=cluster,
        candidate_count=1,
    )
    validation = validation.append(lookup)
    candidate = run_locator_candidate_evaluation(
        validation,
        cluster=cluster,
        candidate_index=1,
        depends_on=(lookup.node_id,),
    )
    validation = validation.append(candidate)
    exact = run_exact_case_name_check(validation, candidate=candidate)
    validation = validation.append(exact)
    calls = []

    async def fake_instruct(_session, spec, *, strategy, model_options):
        calls.append((spec, strategy, model_options))
        return IvrRun(
            success=True,
            selected_attempt=0,
            attempts=(
                IvrAttempt(
                    output=(
                        '{"classification":"complete_case_name","case_name_quote":"Cadle Co. v. Ayala",'
                        '"plaintiff":"Cadle Co.","defendant":"Ayala","equivalent":false,'
                        '"reason":"The cited and retrieved parties differ."}'
                    ),
                    requirements=(),
                ),
            ),
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
        "mellea_lrc.validation.field_checks.mellea_case_name_review.run_instruct_ivr",
        fake_instruct,
    )

    reviewed = asyncio.run(
        run_mellea_case_name_review(
            validation,
            trigger=exact,
            locator_lookup=lookup,
            candidate=candidate,
            document_text=document.text,
            session=object(),
        )
    )

    assert reviewed.outcome is MelleaCaseNameCheckOutcome.MISMATCH
    assert reviewed.case_name == CaseName(
        span=Span(0, len(source_name)),
        text=source_name,
        plaintiff="Cadle Co.",
        defendant="Ayala",
    )
    assert len(calls) == 1
    spec, strategy, _ = calls[0]
    assert strategy.loop_budget == 1
    assert spec.grounding_context == {"local_context": document.text}
    assert spec.user_variables == {
        "locator": _LOCATOR,
        "retrieved_case_name": _RETRIEVED_NAME,
    }
