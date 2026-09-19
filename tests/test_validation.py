"""Tests for the post-extraction validation-node progression."""

import asyncio
from threading import Event
from types import SimpleNamespace

import pytest
from mellea.backends import ModelOption
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel

from mellea_lrc.core.citations import CitationDate, DocketCitation, FullCaseCitation, FullLawCitation, placed
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener import (
    CourtListenerCitationLookup,
    CourtListenerDocket,
    CourtListenerError,
    CourtListenerOpinionCluster,
    CourtListenerSearchResult,
)
from mellea_lrc.extraction import CitationRecord, Document, ExtractionMetadata
from mellea_lrc.llm.ivr import InstructIvrSpec, IvrAttempt, IvrRun, run_instruct_ivr
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.serialization import deserialize_validated_document, serialize_validated_document
from mellea_lrc.validation import (
    AggregatedFieldOutcome,
    CandidateEvaluationNode,
    CandidateEvaluationOutcome,
    CandidateEvaluationSource,
    CandidateProvenance,
    CandidateSelectionNode,
    CandidateSelectionOutcome,
    CourtCheckNode,
    ExactCaseNameCheckNode,
    ExactLocatorLookupNode,
    FieldCheckOutcome,
    LocatorCandidateAssessmentNode,
    LocatorCandidateAssessmentOutcome,
    LocatorCitationSummaryNode,
    LocatorCitationSummaryOutcome,
    LocatorIdentityResolutionNode,
    LocatorIdentityResolutionOutcome,
    LocatorLookupOutcome,
    MelleaCaseNameCheckNode,
    MelleaCaseNameCheckOutcome,
    MelleaCaseNameQueryPreparationNode,
    MelleaCaseNameQueryPreparationOutcome,
    MelleaCaseNameReextractionNode,
    MelleaCaseNameReextractionOutcome,
    MelleaCitingPropositionExtractionNode,
    MelleaLocatorCandidateChoiceNode,
    MelleaLocatorCandidateChoiceOutcome,
    MelleaPinpointCheckNode,
    OpinionSearchCandidateAssessmentNode,
    OpinionSearchNode,
    OpinionSearchOutcome,
    RecapSearchCandidateAssessmentNode,
    RecapSearchNode,
    RecapSearchOutcome,
    ReporterPageRetrievalNode,
    ReporterPageRetrievalOutcome,
    SearchCandidateAssessmentOutcome,
    SearchCitationSummaryNode,
    ValidatedDocument,
    ValidationNodeStatus,
    YearCheckNode,
    initialize_full_reporter_locator_identity,
    run_full_reporter_locator_identity,
)
from mellea_lrc.validation.aggregation.citation_summary_candidate import citation_summary_candidate
from mellea_lrc.validation.candidate_state import CandidateValidationState
from mellea_lrc.validation.case_search import run_mellea_case_name_query_preparation
from mellea_lrc.validation.execution import CitationValidationRunner
from mellea_lrc.validation.field_checks.mellea_case_name_reextraction import (
    run_mellea_case_name_reextraction,
)


class LookupClient:
    """Minimal deterministic exact-lookup client for pipeline tests."""

    def __init__(
        self,
        response: CourtListenerCitationLookup | CourtListenerError,
        docket_response: CourtListenerDocket | CourtListenerError | None = None,
        opinion_search_response: CourtListenerSearchResult | CourtListenerError | None = None,
        recap_search_response: CourtListenerSearchResult | CourtListenerError | None = None,
    ) -> None:
        self.response = response
        self.docket_response = docket_response
        self.opinion_search_response = opinion_search_response
        self.recap_search_response = recap_search_response
        self.calls: list[tuple[str, str, str]] = []
        self.docket_calls: list[str] = []
        self.search_calls: list[tuple[str, str]] = []

    def lookup_citation(
        self,
        volume: str,
        reporter: str,
        page: str,
    ) -> CourtListenerCitationLookup:
        """Record the locator and return or raise the configured outcome."""
        self.calls.append((volume, reporter, page))
        if isinstance(self.response, CourtListenerError):
            raise self.response
        return self.response

    def get_docket(self, docket_id: str) -> CourtListenerDocket:
        """Record the docket identifier and return the configured result."""
        self.docket_calls.append(docket_id)
        if isinstance(self.docket_response, CourtListenerError):
            raise self.docket_response
        if self.docket_response is None:
            msg = "No docket response configured"
            raise AssertionError(msg)
        return self.docket_response

    def search(self, query: str, search_type: str) -> CourtListenerSearchResult:
        """Record the query and return the configured search outcome."""
        self.search_calls.append((query, search_type))
        response = {
            "o": self.opinion_search_response,
            "r": self.recap_search_response,
        }.get(search_type)
        if isinstance(response, CourtListenerError):
            raise response
        if response is None:
            msg = "No search response configured"
            raise AssertionError(msg)
        return response


def _validate(
    document: Document,
    client: LookupClient,
    *,
    session: object | None = None,
) -> ValidatedDocument:
    """Run the active document-level validation entrypoint synchronously in tests."""
    checkpoint = initialize_full_reporter_locator_identity(document)
    return asyncio.run(run_full_reporter_locator_identity(checkpoint, client=client, session=session))


def _validate_identity(
    document: Document,
    client: LookupClient,
    *,
    session: object | None = None,
) -> ValidatedDocument:
    """Run the checkpoint-to-identity entrypoint synchronously in tests."""
    checkpoint = initialize_full_reporter_locator_identity(document)
    return asyncio.run(run_full_reporter_locator_identity(checkpoint, client=client, session=session))


def _document(citation: object) -> Document:
    text = "Brown v. Board, 347 U.S. 483 (1954)."
    preprocessed = preprocess(text)
    locator = "347 U.S. 483"
    start = text.index(locator)
    extracted = CitationRecord(
        citation_id="cite-0001",
        source=placed(
            citation,
            span=Span(0, len(text)),
            locator_span=Span(start, start + len(locator)),
            matched_text=locator,
        ),
    )
    return Document(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(extracted,),
        extraction_metadata=ExtractionMetadata(),
    )


def _successful_ivr(output: str) -> IvrRun:
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


def _sampling_result(output: str) -> SimpleNamespace:
    """Mellea's public sampling-result surface, without a provider call."""
    return SimpleNamespace(
        success=True,
        result_index=0,
        sample_generations=[SimpleNamespace(value=output)],
        sample_validations=[[]],
    )


def test_instruct_ivr_forwards_the_pydantic_output_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward the schema through Mellea's structured-output interface."""

    class ExpectedOutput(BaseModel):
        value: str

    calls: list[dict[str, object]] = []

    def fake_instruct(*_args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return _sampling_result('{"value":"structured"}')

    monkeypatch.setattr("mellea_lrc.llm.ivr.mfuncs.instruct", fake_instruct)
    result = asyncio.run(
        run_instruct_ivr(
            SimpleNamespace(backend=object()),
            InstructIvrSpec(description="Return a value.", output_format=ExpectedOutput),
            strategy=MultiTurnStrategy(loop_budget=1),
            model_options={},
        )
    )

    assert result.output == '{"value":"structured"}'
    assert result.success
    assert len(result.attempts) == 1
    assert calls[0]["format"] is ExpectedOutput


def test_a_prefix_is_sent_as_the_system_message_and_nowhere_else(monkeypatch) -> None:
    from mellea.backends import ModelOption

    calls: list[dict[str, object]] = []

    def fake_instruct(*_args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return _sampling_result("{}")

    monkeypatch.setattr("mellea_lrc.llm.ivr.mfuncs.instruct", fake_instruct)
    options = {"max_tokens": 10}
    for prefix in (None, "The cited text:\nthe opinion"):
        asyncio.run(
            run_instruct_ivr(
                SimpleNamespace(backend=object()),
                InstructIvrSpec(description="Return a value.", prefix=prefix),
                strategy=MultiTurnStrategy(loop_budget=1),
                model_options=options,
            )
        )
    assert ModelOption.SYSTEM_PROMPT not in calls[0]["model_options"]
    assert calls[1]["model_options"] == {
        "max_tokens": 10,
        ModelOption.SYSTEM_PROMPT: "The cited text:\nthe opinion",
    }
    assert options == {"max_tokens": 10}


def test_instruct_ivr_records_a_whole_call_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """An open provider call becomes a serializable failed IVR run."""

    def stalled_instruct(*_args: object, **_kwargs: object) -> SimpleNamespace:
        Event().wait(0.05)
        return _sampling_result("{}")

    monkeypatch.setattr("mellea_lrc.llm.ivr.mfuncs.instruct", stalled_instruct)
    result = asyncio.run(
        run_instruct_ivr(
            SimpleNamespace(backend=SimpleNamespace(model_id="test-model")),
            InstructIvrSpec(description="Return a value."),
            strategy=MultiTurnStrategy(loop_budget=1),
            model_options={ModelOption.STREAM_TIMEOUT: 0.005},
        )
    )

    assert not result.success
    assert result.output == ""
    assert result.failure_reason == "Model call exceeded the configured 0.005-second timeout."
    assert result.attempts[0].requirements[0].passed is False


def test_initialize_full_reporter_locator_identity_instances_one_progression_per_extracted_citation() -> None:
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="483"))

    validation = initialize_full_reporter_locator_identity(extracted)

    assert validation.source is extracted
    assert validation.text == extracted.text
    assert validation.citations[0].citation is extracted.citations[0]
    assert validation.citations[0].nodes == ()


def test_full_reporter_checkpoint_leaves_a_docket_progression_empty() -> None:
    text = "Brown v. Board, 347 U.S. 483 (1954); Case No. 1:24-cv-08705."
    preprocessed = preprocess(text)
    reporter_locator = "347 U.S. 483"
    docket_locator = "1:24-cv-08705"
    reporter_start = text.index(reporter_locator)
    docket_start = text.index(docket_locator)
    document = Document(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(
            CitationRecord(
                citation_id="reporter",
                source=placed(
                    FullCaseCitation(
                        plaintiff="Brown",
                        defendant="Board",
                        volume="347",
                        reporter="U.S.",
                        page="483",
                        date=CitationDate(year="1954"),
                        court="scotus",
                    ),
                    span=Span(0, text.index(";")),
                    locator_span=Span(reporter_start, reporter_start + len(reporter_locator)),
                    matched_text=reporter_locator,
                ),
            ),
            CitationRecord(
                citation_id="docket",
                source=placed(
                    DocketCitation(docket_number=docket_locator),
                    span=Span(docket_start, docket_start + len(docket_locator)),
                    locator_span=Span(docket_start, docket_start + len(docket_locator)),
                    matched_text=docket_locator,
                ),
            ),
        ),
        extraction_metadata=ExtractionMetadata(),
    )
    client = LookupClient(
        CourtListenerCitationLookup(
            citation=reporter_locator,
            status=200,
            clusters=(
                CourtListenerOpinionCluster(
                    case_name="Brown v. Board",
                    date_filed="1954-05-17",
                    court_id="scotus",
                ),
            ),
        )
    )

    checkpoint = initialize_full_reporter_locator_identity(document)
    completed = asyncio.run(run_full_reporter_locator_identity(checkpoint, client=client))
    serialized = serialize_validated_document(completed)
    resumed = asyncio.run(
        run_full_reporter_locator_identity(deserialize_validated_document(serialized), client=client)
    )

    assert client.calls == [("347", "U.S.", "483")]
    assert completed.citation_by_id("reporter").identity_resolution is not None
    assert completed.citation_by_id("docket").nodes == ()
    assert resumed == completed


def test_full_reporter_locator_runs_identity_without_pinpoint_work() -> None:
    """The active top-level runner stops at the reporter-locator identity boundary."""
    extracted = _document(
        FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter="U.S.",
            page="483",
            date=CitationDate(year="1954"),
            court="scotus",
        )
    )
    cluster = CourtListenerOpinionCluster(
        case_name="Brown v. Board",
        date_filed="1954-05-17",
        court_id="scotus",
        docket_id="84657",
    )
    client = LookupClient(
        CourtListenerCitationLookup(citation="347 U.S. 483", status=200, clusters=(cluster,)),
        docket_response=CourtListenerDocket(docket_id="84657", court_id="scotus"),
    )

    progression = _validate(extracted, client).citation_by_id("cite-0001")

    assert client.calls == [("347", "U.S.", "483")]
    assert len(progression.nodes) == 9
    lookup, candidate, exact_name, year, docket_court, court, assessment, summary, resolution = (
        progression.nodes
    )
    assert isinstance(lookup, ExactLocatorLookupNode)
    assert lookup.outcome is LocatorLookupOutcome.FOUND
    assert isinstance(candidate, CandidateEvaluationNode)
    assert candidate.record is cluster
    assert exact_name.outcome is FieldCheckOutcome.MATCH
    assert year.outcome is FieldCheckOutcome.MATCH
    assert docket_court.status is ValidationNodeStatus.SUCCEEDED
    assert client.docket_calls == ["84657"]
    assert court.outcome is FieldCheckOutcome.MATCH
    assert isinstance(assessment, LocatorCandidateAssessmentNode)
    assert assessment.outcome is LocatorCandidateAssessmentOutcome.MATCH
    assert isinstance(summary, LocatorCitationSummaryNode)
    assert summary.pinpoint_requires_review is None
    assert isinstance(resolution, LocatorIdentityResolutionNode)
    assert resolution.outcome is LocatorIdentityResolutionOutcome.RESOLVED
    assert resolution.selection_evidence_node_id == summary.node_id
    assert all(
        not isinstance(
            node,
            ReporterPageRetrievalNode | MelleaCitingPropositionExtractionNode | MelleaPinpointCheckNode,
        )
        for node in progression.nodes
    )


def test_identity_checkpoint_stops_before_reporter_page_retrieval() -> None:
    """Identity is complete before the later pinpoint phase begins."""
    extracted = _document(
        FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter="U.S.",
            page="483",
            date=CitationDate(year="1954"),
            court="scotus",
        )
    )
    cluster = CourtListenerOpinionCluster(
        case_name="Brown v. Board",
        date_filed="1954-05-17",
        court_id="scotus",
        docket_id="84657",
    )
    client = LookupClient(
        CourtListenerCitationLookup(
            citation="347 U.S. 483",
            status=200,
            clusters=(cluster,),
        ),
        docket_response=CourtListenerDocket(docket_id="84657", court_id="scotus"),
    )

    progression = _validate_identity(extracted, client).citation_by_id("cite-0001")

    assert len(progression.nodes) == 9
    assert all(
        not isinstance(
            node,
            ReporterPageRetrievalNode | MelleaCitingPropositionExtractionNode | MelleaPinpointCheckNode,
        )
        for node in progression.nodes
    )
    assert isinstance(progression.aggregation, LocatorCitationSummaryNode)
    assert progression.aggregation.candidates[0].pinpoint is None
    assert progression.aggregation.pinpoint_requires_review is None
    resolution = progression.identity_resolution
    assert resolution is not None
    assert resolution.outcome is LocatorIdentityResolutionOutcome.RESOLVED
    assert resolution.selected_candidate_index == 1
    assert resolution.selected_assessment_node_id == progression.aggregation.candidates[0].assessment_node_id


def test_found_field_checks_treat_unavailable_year_as_a_full_match() -> None:
    """An unavailable date does not downgrade a confirmed reporter identity."""
    extracted = _document(
        FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter="U.S.",
            page="483",
            date=None,
            court="scotus",
        )
    )
    cluster = CourtListenerOpinionCluster(
        case_name="Brown v. Board",
        date_filed="1954-05-17",
        court_id="scotus",
        docket_id="84657",
    )
    client = LookupClient(
        CourtListenerCitationLookup(citation="347 U.S. 483", status=200, clusters=(cluster,)),
        docket_response=CourtListenerDocket(docket_id="84657", court_id="scotus"),
    )

    _, _, exact_name, year, _, court, assessment, summary, resolution = (
        _validate(extracted, client).citations[0].nodes
    )
    assert exact_name.outcome is FieldCheckOutcome.MATCH
    assert year.outcome is FieldCheckOutcome.UNAVAILABLE
    assert court.outcome is FieldCheckOutcome.MATCH
    assert assessment.outcome is LocatorCandidateAssessmentOutcome.MATCH
    assert summary.outcome is LocatorCitationSummaryOutcome.COMPLETE
    assert resolution.outcome is LocatorIdentityResolutionOutcome.RESOLVED


def test_found_field_checks_record_mismatch_without_failing_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = _document(
        FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter="U.S.",
            page="483",
            date=CitationDate(year="1954"),
            court="ca10",
        )
    )
    client = LookupClient(
        CourtListenerCitationLookup(
            citation="347 U.S. 483",
            status=200,
            clusters=(
                CourtListenerOpinionCluster(
                    case_name="Different v. Case",
                    date_filed="1955-01-01",
                    docket_id="98765",
                ),
            ),
        ),
        docket_response=CourtListenerDocket(docket_id="98765", court_id="ca9"),
    )

    async def fake_semantic_check(
        _validation: object,
        *,
        case_name_evidence: ExactCaseNameCheckNode,
        session: object | None = None,
    ) -> MelleaCaseNameCheckNode:
        del session
        return MelleaCaseNameCheckNode(
            node_id=f"{case_name_evidence.node_id}:mellea_case_name_check",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCaseNameCheckOutcome.MATCH,
            extracted_case_name="Brown v. Board",
            retrieved_case_name="Different v. Case",
            depends_on=(case_name_evidence.node_id,),
        )

    async def fake_choice(
        validation: object,
        *,
        summary: LocatorCitationSummaryNode,
        document_text: str,
        session: object | None,
    ) -> MelleaLocatorCandidateChoiceNode:
        del validation, document_text, session
        return MelleaLocatorCandidateChoiceNode(
            node_id=f"{summary.node_id}:mellea_candidate_choice",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.NO_MATCH,
            candidate_indices=(1,),
            selected_candidate_index=None,
            reparsed_case_name=None,
            reparsed_court=None,
            reparsed_date=None,
            rationale="Retrieved candidate disagrees with every stated field.",
            depends_on=(summary.node_id,),
        )

    monkeypatch.setattr("mellea_lrc.validation.execution.run_mellea_case_name_check", fake_semantic_check)
    monkeypatch.setattr("mellea_lrc.validation.execution.run_mellea_locator_candidate_choice", fake_choice)

    (_, _, exact_name, year, _, court, semantic, assessment, summary, choice, resolution) = (
        _validate(extracted, client).citations[0].nodes
    )
    assert exact_name.outcome is FieldCheckOutcome.MISMATCH
    assert year.outcome is FieldCheckOutcome.MISMATCH
    assert semantic.outcome is MelleaCaseNameCheckOutcome.MATCH
    assert court.outcome is FieldCheckOutcome.MISMATCH
    assert assessment.outcome is LocatorCandidateAssessmentOutcome.MISMATCH
    assert summary.candidates[0].assessment_node_id == assessment.node_id
    assert choice.outcome is MelleaLocatorCandidateChoiceOutcome.NO_MATCH
    assert resolution.outcome is LocatorIdentityResolutionOutcome.NO_MATCH


def test_found_field_checks_skip_unavailable_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="483"))
    client = LookupClient(
        CourtListenerCitationLookup(
            citation="347 U.S. 483", status=200, clusters=(CourtListenerOpinionCluster(),)
        )
    )

    async def fake_choice(
        validation: object,
        *,
        summary: LocatorCitationSummaryNode,
        document_text: str,
        session: object | None,
    ) -> MelleaLocatorCandidateChoiceNode:
        del validation, document_text, session
        return MelleaLocatorCandidateChoiceNode(
            node_id=f"{summary.node_id}:mellea_candidate_choice",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.NO_MATCH,
            candidate_indices=(1,),
            selected_candidate_index=None,
            reparsed_case_name=None,
            reparsed_court=None,
            reparsed_date=None,
            rationale="The citation states no identity fields.",
            depends_on=(summary.node_id,),
        )

    monkeypatch.setattr("mellea_lrc.validation.execution.run_mellea_locator_candidate_choice", fake_choice)
    _, _, exact_name, year, _, court, assessment, summary, choice, resolution = (
        _validate(extracted, client).citations[0].nodes
    )
    assert exact_name.outcome is FieldCheckOutcome.UNAVAILABLE
    assert year.outcome is FieldCheckOutcome.UNAVAILABLE
    assert court.outcome is FieldCheckOutcome.UNAVAILABLE
    assert assessment.outcome is LocatorCandidateAssessmentOutcome.PARTIAL_MATCH
    assert summary.candidates[0].assessment_node_id == assessment.node_id
    assert choice.outcome is MelleaLocatorCandidateChoiceOutcome.NO_MATCH
    assert resolution.outcome is LocatorIdentityResolutionOutcome.NO_MATCH


def test_found_unavailable_extraction_with_retrieved_name_still_reextracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing local extraction with a retrieved name to compare against still recovers."""
    extracted = _document(FullCaseCitation(volume="503", reporter="B.R.", page="571"))
    client = LookupClient(
        CourtListenerCitationLookup(
            citation="503 B.R. 571",
            status=200,
            clusters=(
                CourtListenerOpinionCluster(
                    case_name="In re Soundview Elite, Ltd.",
                    docket_id="65785838",
                ),
            ),
        ),
        docket_response=CourtListenerDocket(docket_id="65785838", court_id="nysb"),
    )

    calls: list[object] = []

    async def fake_reextraction(
        validation: object,
        *,
        trigger: ExactCaseNameCheckNode,
        locator_lookup: ExactLocatorLookupNode,
        document_text: str,
        session: object | None,
    ) -> MelleaCaseNameReextractionNode:
        del locator_lookup, document_text, session
        assert isinstance(trigger, ExactCaseNameCheckNode)
        calls.append(trigger)
        return MelleaCaseNameReextractionNode(
            node_id=f"{trigger.node_id}:mellea_case_name_reextraction",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCaseNameReextractionOutcome.COMPLETE,
            plaintiff=None,
            defendant="Soundview Elite Ltd.",
            depends_on=(trigger.node_id,),
        )

    monkeypatch.setattr(
        "mellea_lrc.validation.execution.run_mellea_case_name_reextraction",
        fake_reextraction,
    )

    async def fake_reextracted_semantic_check(
        _validation: object,
        *,
        case_name_evidence: MelleaCaseNameReextractionNode,
        candidate: CandidateEvaluationNode | None = None,
        session: object | None = None,
    ) -> MelleaCaseNameCheckNode:
        del candidate, session
        return MelleaCaseNameCheckNode(
            node_id=f"{case_name_evidence.node_id}:mellea_case_name_check",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCaseNameCheckOutcome.MATCH,
            extracted_case_name="Soundview Elite Ltd.",
            retrieved_case_name="In re Soundview Elite, Ltd.",
            depends_on=(case_name_evidence.node_id,),
        )

    monkeypatch.setattr(
        "mellea_lrc.validation.execution.run_mellea_case_name_check",
        fake_reextracted_semantic_check,
    )

    (
        _,
        _,
        exact_case_name_check_node,
        *_rest,
    ) = _validate(extracted, client).citations[0].nodes

    assert exact_case_name_check_node.status is ValidationNodeStatus.SKIPPED
    assert exact_case_name_check_node.outcome is FieldCheckOutcome.UNAVAILABLE
    assert len(calls) == 1
    assert calls[0] is exact_case_name_check_node


def test_mellea_case_name_reextraction_uses_only_local_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ground Mellea re-extraction in local text without a retrieved case name."""
    document = _document(FullCaseCitation(volume="347", reporter="U.S.", page="483"))
    validation = initialize_full_reporter_locator_identity(document).citations[0]
    exact_locator_lookup_node = ExactLocatorLookupNode(
        node_id="cite-0001:exact_locator_lookup",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorLookupOutcome.FOUND,
        locator="347 U.S. 483",
        cluster=CourtListenerOpinionCluster(case_name="Brown v. Board of Education"),
        candidate_count=1,
    )
    validation = validation.append(exact_locator_lookup_node)
    semantic_case_name_check_node = MelleaCaseNameCheckNode(
        node_id="cite-0001:mellea_case_name_check",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=MelleaCaseNameCheckOutcome.MISMATCH,
        extracted_case_name="Brown v. Board",
        retrieved_case_name="Brown v. Board of Education",
        depends_on=(exact_locator_lookup_node.node_id,),
    )
    validation = validation.append(semantic_case_name_check_node)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test-model")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test-key")
    calls: list[object] = []

    async def fake_instruct(_session: object, spec: object, **_kwargs: object) -> IvrRun:
        calls.append(spec)
        return _successful_ivr(
            '{"classification":"complete_case_name","plaintiff":"Brown","defendant":"Board"}'
        )

    monkeypatch.setattr(
        "mellea_lrc.validation.field_checks.mellea_case_name_reextraction.run_instruct_ivr",
        fake_instruct,
    )

    node = asyncio.run(
        run_mellea_case_name_reextraction(
            validation,
            trigger=semantic_case_name_check_node,
            locator_lookup=exact_locator_lookup_node,
            document_text=document.text,
            session=object(),
        )
    )

    assert isinstance(node, MelleaCaseNameReextractionNode)
    assert node.outcome is MelleaCaseNameReextractionOutcome.COMPLETE
    assert node.plaintiff == "Brown"
    assert node.defendant == "Board"
    assert node.depends_on == (semantic_case_name_check_node.node_id,)
    spec = calls[0]
    assert spec.user_variables == {"locator": "347 U.S. 483"}
    assert spec.grounding_context.keys() == {"local_context"}
    assert "Brown v. Board" in spec.grounding_context["local_context"]
    assert spec.output_format.__name__ == "_PartyProposal"


def test_lookup_miss_defers_without_local_reextraction() -> None:
    """Lookup misses stay deferred until reporter-locator lookup design is settled."""
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="9999"))
    client = LookupClient(CourtListenerCitationLookup(citation="347 U.S. 9999", status=404, clusters=()))

    lookup, resolution = _validate(extracted, client).citations[0].nodes

    assert lookup.outcome is LocatorLookupOutcome.NOT_FOUND
    assert isinstance(resolution, LocatorIdentityResolutionNode)
    assert resolution.outcome is LocatorIdentityResolutionOutcome.DEFERRED_TO_SEARCH
    assert resolution.depends_on == (lookup.node_id,)
    assert client.search_calls == []


@pytest.mark.parametrize(
    "source",
    (CandidateEvaluationSource.OPINION_SEARCH, CandidateEvaluationSource.RECAP_SEARCH),
)
def test_search_candidate_uses_semantic_check_without_reextracting(
    monkeypatch: pytest.MonkeyPatch,
    source: CandidateEvaluationSource,
) -> None:
    """Selected search candidates share semantic fallback without re-extraction."""
    extracted = _document(
        FullCaseCitation(
            plaintiff="Brown",
            defendant="Board of Education",
            volume="347",
            reporter="U.S.",
            page="9999",
            date=CitationDate(year="1954"),
            court="scotus",
        )
    )
    validation = initialize_full_reporter_locator_identity(extracted).citations[0]
    candidate = CandidateEvaluationNode(
        node_id=f"cite-0001:{source.value}_candidate_evaluation:1",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=CandidateEvaluationOutcome.READY,
        source=source,
        candidate_index=1,
        cluster_id="123",
        case_name="Brown v. Board",
        date_filed="1954-05-17",
        court_id="scotus",
        docket_id=None,
        docket_number=None,
        record={},
        depends_on=(),
    )
    calls: list[ExactCaseNameCheckNode] = []

    async def fake_semantic_check(
        _validation: object,
        *,
        case_name_evidence: ExactCaseNameCheckNode,
        session: object | None,
    ) -> MelleaCaseNameCheckNode:
        del session
        calls.append(case_name_evidence)
        return MelleaCaseNameCheckNode(
            node_id=f"{case_name_evidence.node_id}:mellea_case_name_check",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCaseNameCheckOutcome.MATCH,
            extracted_case_name=case_name_evidence.extracted_case_name or "",
            retrieved_case_name=case_name_evidence.retrieved_case_name or "",
            depends_on=(case_name_evidence.node_id,),
        )

    monkeypatch.setattr(
        "mellea_lrc.validation.execution.run_mellea_case_name_check",
        fake_semantic_check,
    )

    progression = asyncio.run(
        CitationValidationRunner(
            client=LookupClient(
                CourtListenerCitationLookup(citation="347 U.S. 9999", status=404, clusters=())
            )
        ).run_search_candidate_validation(
            validation.append(candidate),
            candidate=candidate,
            session=object(),
            state=CandidateValidationState(),
        )
    )

    expected_types = [
        CandidateEvaluationNode,
        ExactCaseNameCheckNode,
        YearCheckNode,
        CourtCheckNode,
        MelleaCaseNameCheckNode,
    ]
    if source is CandidateEvaluationSource.OPINION_SEARCH:
        expected_types.append(OpinionSearchCandidateAssessmentNode)
    if source is CandidateEvaluationSource.RECAP_SEARCH:
        expected_types.append(RecapSearchCandidateAssessmentNode)
    assert [type(node) for node in progression.nodes] == expected_types
    assert progression.nodes[1].outcome is FieldCheckOutcome.MISMATCH
    assert progression.nodes[2].outcome is FieldCheckOutcome.MATCH
    assert progression.nodes[3].outcome is FieldCheckOutcome.MATCH
    assert calls[0].depends_on == (candidate.node_id,)
    if source in {
        CandidateEvaluationSource.OPINION_SEARCH,
        CandidateEvaluationSource.RECAP_SEARCH,
    }:
        assessment = progression.nodes[-1]
        assert isinstance(
            assessment,
            OpinionSearchCandidateAssessmentNode | RecapSearchCandidateAssessmentNode,
        )
        assert assessment.outcome is SearchCandidateAssessmentOutcome.POSSIBLE_MATCH
        assert assessment.depends_on == (
            progression.nodes[4].node_id,
            progression.nodes[2].node_id,
            progression.nodes[3].node_id,
        )
    else:
        assert progression.nodes[-1].depends_on == (calls[0].node_id,)


def test_opinion_search_candidate_assessment_requires_every_field_to_match() -> None:
    """A differing opinion year prevents the deliberately limited possible match."""
    extracted = _document(
        FullCaseCitation(
            plaintiff="Brown",
            defendant="Board of Education",
            volume="347",
            reporter="U.S.",
            page="9999",
            date=CitationDate(year="1954"),
            court="scotus",
        )
    )
    validation = initialize_full_reporter_locator_identity(extracted).citations[0]
    candidate = CandidateEvaluationNode(
        node_id="cite-0001:opinion_search_candidate_evaluation:1",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=CandidateEvaluationOutcome.READY,
        source=CandidateEvaluationSource.OPINION_SEARCH,
        candidate_index=1,
        cluster_id="123",
        case_name="Brown v. Board of Education",
        date_filed="1955-05-17",
        court_id="scotus",
        docket_id=None,
        docket_number=None,
        record={},
        depends_on=(),
    )

    progression = asyncio.run(
        CitationValidationRunner(
            client=LookupClient(
                CourtListenerCitationLookup(citation="347 U.S. 9999", status=404, clusters=())
            )
        ).run_search_candidate_validation(
            validation.append(candidate),
            candidate=candidate,
            session=None,
            state=CandidateValidationState(),
        )
    )

    assessment = progression.nodes[-1]
    assert isinstance(assessment, OpinionSearchCandidateAssessmentNode)
    assert assessment.case_name_outcome is AggregatedFieldOutcome.MATCH
    assert assessment.year_outcome is AggregatedFieldOutcome.MISMATCH
    assert assessment.court_outcome is AggregatedFieldOutcome.MATCH
    assert assessment.outcome is SearchCandidateAssessmentOutcome.MISMATCH


def test_recap_search_candidate_assessment_does_not_treat_docket_year_as_a_mismatch() -> None:
    """RECAP dates describe a docket, not necessarily the cited opinion's publication year."""
    extracted = _document(
        FullCaseCitation(
            plaintiff="Brown",
            defendant="Board of Education",
            volume="347",
            reporter="U.S.",
            page="9999",
            date=CitationDate(year="1954"),
            court="scotus",
        )
    )
    validation = initialize_full_reporter_locator_identity(extracted).citations[0]
    candidate = CandidateEvaluationNode(
        node_id="cite-0001:recap_search_candidate_evaluation:1",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=CandidateEvaluationOutcome.READY,
        source=CandidateEvaluationSource.RECAP_SEARCH,
        candidate_index=1,
        cluster_id=None,
        case_name="Brown v. Board of Education",
        date_filed="1955-05-17",
        court_id="scotus",
        docket_id="123",
        docket_number=None,
        record={},
        depends_on=(),
    )

    progression = asyncio.run(
        CitationValidationRunner(
            client=LookupClient(
                CourtListenerCitationLookup(citation="347 U.S. 9999", status=404, clusters=())
            )
        ).run_search_candidate_validation(
            validation.append(candidate),
            candidate=candidate,
            session=None,
            state=CandidateValidationState(),
        )
    )

    assessment = progression.nodes[-1]
    assert isinstance(assessment, RecapSearchCandidateAssessmentNode)
    assert assessment.year_outcome is AggregatedFieldOutcome.MISMATCH
    assert assessment.outcome is SearchCandidateAssessmentOutcome.POSSIBLE_MATCH


def test_recap_candidate_summary_exposes_canonical_docket_url() -> None:
    """Carry CourtListener's raw docket path into the terminal summary candidate."""
    validation = initialize_full_reporter_locator_identity(
        _document(FullCaseCitation(volume="347", reporter="U.S.", page="9999"))
    ).citations[0]
    candidate = CandidateEvaluationNode(
        node_id="cite-0001:recap_search_candidate_evaluation:1",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=CandidateEvaluationOutcome.READY,
        source=CandidateEvaluationSource.RECAP_SEARCH,
        candidate_index=1,
        cluster_id=None,
        case_name="Brown v. Board of Education",
        date_filed="2006-10-25",
        court_id="scotus",
        docket_id="5068645",
        docket_number=None,
        record={"docket_absolute_url": "/docket/5068645/brown-v-board-of-education/"},
        depends_on=(),
    )
    assessment = RecapSearchCandidateAssessmentNode(
        node_id=f"{candidate.node_id}:recap_search_candidate_assessment",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=SearchCandidateAssessmentOutcome.POSSIBLE_MATCH,
        candidate_index=1,
        extracted_citation="347 U.S. 9999",
        extracted_case_name="Brown v. Board of Education",
        retrieved_case_name=candidate.case_name,
        case_name_outcome=AggregatedFieldOutcome.MATCH,
        case_name_evidence="exact",
        extracted_year=None,
        retrieved_year=None,
        year_outcome=AggregatedFieldOutcome.UNAVAILABLE,
        extracted_court_id="scotus",
        retrieved_court_id="scotus",
        court_outcome=AggregatedFieldOutcome.MATCH,
        docket_id=candidate.docket_id,
        depends_on=(candidate.node_id,),
    )
    validation = validation.append(candidate).append(assessment)

    summary = citation_summary_candidate(
        validation,
        assessment,
        provenance=CandidateProvenance.RECAP,
    )

    assert summary.docket_url == "https://www.courtlistener.com/docket/5068645/brown-v-board-of-education/"


def test_mellea_case_name_query_preparation_constructs_the_courtlistener_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep CourtListener syntax in project code, not the Mellea response."""
    document = _document(FullCaseCitation(volume="347", reporter="U.S.", page="9999", court="scotus"))
    validation = initialize_full_reporter_locator_identity(document).citations[0]
    locator = ExactLocatorLookupNode(
        node_id="cite-0001:exact_locator_lookup",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorLookupOutcome.NOT_FOUND,
        locator="347 U.S. 9999",
    )
    reextraction = MelleaCaseNameReextractionNode(
        node_id="cite-0001:mellea_case_name_reextraction",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=MelleaCaseNameReextractionOutcome.COMPLETE,
        plaintiff="Brown",
        defendant="Board of Education",
        depends_on=(locator.node_id,),
    )
    validation = validation.append(locator).append(reextraction)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test-model")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test-key")
    calls: list[object] = []

    async def fake_instruct(_session: object, spec: object, **_kwargs: object) -> IvrRun:
        calls.append(spec)
        return _successful_ivr('{"query_plaintiff":"Brown","query_defendant":"Board of Education"}')

    monkeypatch.setattr(
        "mellea_lrc.validation.case_search.mellea_case_name_query_preparation.run_instruct_ivr",
        fake_instruct,
    )

    node = asyncio.run(
        run_mellea_case_name_query_preparation(
            validation,
            reextraction=reextraction,
            session=object(),
        )
    )

    assert node.status is ValidationNodeStatus.SUCCEEDED
    assert node.outcome is MelleaCaseNameQueryPreparationOutcome.PREPARED
    assert node.query == 'caseName:("Brown" AND "Board of Education") AND court_id:scotus'
    assert node.depends_on == (reextraction.node_id,)
    spec = calls[0]
    assert spec.user_variables == {"plaintiff": "Brown", "defendant": "Board of Education"}
    assert spec.grounding_context == {}
    assert spec.output_format.__name__ == "_QueryTermsProposal"


def test_ambiguous_lookup_sends_all_reviewed_candidates_to_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = _document(FullCaseCitation(volume="1", reporter="F.2d", page="2"))
    clusters = (CourtListenerOpinionCluster(), CourtListenerOpinionCluster())
    client = LookupClient(CourtListenerCitationLookup(citation="1 F.2d 2", status=300, clusters=clusters))
    calls: list[InstructIvrSpec] = []

    async def fake_instruct(_session: object, spec: InstructIvrSpec, **_kwargs: object) -> IvrRun:
        calls.append(spec)
        return _successful_ivr(
            '{"decision":"no_match","candidate_index":null,"reparsed_case_name":null,'
            '"reparsed_court":null,"reparsed_date":null,"rationale":"No stated fields."}'
        )

    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test-model")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test-key")
    monkeypatch.setattr(
        "mellea_lrc.validation.aggregation.mellea_locator_candidate_choice.run_instruct_ivr",
        fake_instruct,
    )

    progression = _validate(extracted, client, session=object()).citations[0]
    lookup, selection = progression.nodes[:2]
    summary = next(node for node in progression.nodes if isinstance(node, LocatorCitationSummaryNode))
    choice = next(node for node in progression.nodes if isinstance(node, MelleaLocatorCandidateChoiceNode))
    resolution = progression.identity_resolution

    assert lookup.outcome is LocatorLookupOutcome.AMBIGUOUS
    assert selection.outcome is CandidateSelectionOutcome.ALL_SELECTED
    assert selection.total_candidate_count == 2
    assert selection.selected_candidate_count == 2
    assert summary.candidates[0].candidate_index == 1
    assert summary.candidates[1].candidate_index == 2
    assert choice.outcome is MelleaLocatorCandidateChoiceOutcome.NO_MATCH
    assert choice.candidate_indices == (1, 2)
    assert progression.citation.stated_fields_reparsed_by_model is True
    assert resolution is not None
    assert resolution.outcome is LocatorIdentityResolutionOutcome.NO_MATCH
    assert resolution.selection_evidence_node_id == choice.node_id
    spec = calls[0]
    assert spec.grounding_context.keys() == {"local_context"}
    assert '"candidate_index":1' in spec.user_variables["reviewed_candidates_json"]
    assert '"candidate_index":2' in spec.user_variables["reviewed_candidates_json"]


def test_ambiguous_lookup_defers_at_twenty_candidates() -> None:
    extracted = _document(FullCaseCitation(volume="1", reporter="F.2d", page="2"))
    clusters = tuple(CourtListenerOpinionCluster(case_name=f"Case {index}") for index in range(20))
    client = LookupClient(CourtListenerCitationLookup(citation="1 F.2d 2", status=300, clusters=clusters))

    lookup, selection, resolution = _validate(extracted, client).citations[0].nodes
    assert lookup.outcome is LocatorLookupOutcome.AMBIGUOUS
    assert selection.outcome is CandidateSelectionOutcome.EXCEEDS_REVIEW_LIMIT
    assert selection.total_candidate_count == 20
    assert selection.selected_candidate_count == 0
    assert "meet or exceed" in selection.outcome_message
    assert resolution.outcome is LocatorIdentityResolutionOutcome.DEFERRED_TO_FUTURE_IMPLEMENTATION
    assert resolution.depends_on == (selection.node_id,)


def test_ambiguous_locator_resolves_only_one_confirmed_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = _document(
        FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter="U.S.",
            page="483",
            date=CitationDate(year="1954"),
            court="scotus",
        )
    )
    clusters = (
        CourtListenerOpinionCluster(
            cluster_id="correct",
            case_name="Brown v. Board",
            date_filed="1954-05-17",
            court_id="scotus",
        ),
        CourtListenerOpinionCluster(
            cluster_id="wrong",
            case_name="Other v. Case",
            date_filed="1954-05-17",
            court_id="scotus",
        ),
    )
    client = LookupClient(CourtListenerCitationLookup(citation="347 U.S. 483", status=300, clusters=clusters))

    async def fake_semantic_check(
        _validation: object,
        *,
        case_name_evidence: ExactCaseNameCheckNode,
        session: object | None = None,
    ) -> MelleaCaseNameCheckNode:
        del session
        return MelleaCaseNameCheckNode(
            node_id=f"{case_name_evidence.node_id}:mellea_case_name_check",
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaCaseNameCheckOutcome.FAILED,
            extracted_case_name="Brown v. Board",
            retrieved_case_name="Other v. Case",
            depends_on=(case_name_evidence.node_id,),
        )

    monkeypatch.setattr("mellea_lrc.validation.execution.run_mellea_case_name_check", fake_semantic_check)
    progression = _validate_identity(extracted, client).citations[0]

    resolution = progression.identity_resolution
    assert resolution is not None
    assert resolution.outcome is LocatorIdentityResolutionOutcome.RESOLVED
    assert resolution.selected_candidate_index == 1
    assert resolution.matching_candidate_indices == (1,)
    assert resolution.selected_assessment_node_id == next(
        node.node_id
        for node in progression.nodes
        if isinstance(node, LocatorCandidateAssessmentNode) and node.candidate_index == 1
    )


def test_ambiguous_locator_uses_model_to_select_among_confirmed_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = _document(
        FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter="U.S.",
            page="483",
            date=CitationDate(year="1954"),
            court="scotus",
        )
    )
    clusters = tuple(
        CourtListenerOpinionCluster(
            cluster_id=f"duplicate-{index}",
            case_name="Brown v. Board",
            date_filed="1954-05-17",
            court_id="scotus",
        )
        for index in range(2)
    )
    client = LookupClient(CourtListenerCitationLookup(citation="347 U.S. 483", status=300, clusters=clusters))

    async def fake_choice(
        validation: object,
        *,
        summary: LocatorCitationSummaryNode,
        document_text: str,
        session: object | None,
    ) -> MelleaLocatorCandidateChoiceNode:
        del validation, document_text, session
        return MelleaLocatorCandidateChoiceNode(
            node_id=f"{summary.node_id}:mellea_candidate_choice",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.SELECTED,
            candidate_indices=(1, 2),
            selected_candidate_index=2,
            reparsed_case_name="Brown v. Board",
            reparsed_court=None,
            reparsed_date="1954",
            rationale="Candidate 2 is the representative record.",
            depends_on=(summary.node_id,),
        )

    monkeypatch.setattr("mellea_lrc.validation.execution.run_mellea_locator_candidate_choice", fake_choice)
    progression = _validate_identity(extracted, client).citations[0]
    resolution = progression.identity_resolution
    choice = next(node for node in progression.nodes if isinstance(node, MelleaLocatorCandidateChoiceNode))

    assert choice.candidate_indices == (1, 2)
    assert resolution is not None
    assert resolution.outcome is LocatorIdentityResolutionOutcome.RESOLVED
    assert resolution.selected_candidate_index == 2
    assert resolution.matching_candidate_indices == (1, 2)
    assert resolution.selection_evidence_node_id == choice.node_id


def test_model_choice_with_court_mismatch_defers_to_future_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model choice cannot silently turn a bad stated court into an identity."""
    extracted = _document(
        FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter="U.S.",
            page="483",
            date=CitationDate(year="1954"),
            court="scotus",
        )
    )
    clusters = (
        CourtListenerOpinionCluster(
            cluster_id="wrong-court",
            case_name="Brown v. Board",
            date_filed="1954-05-17",
            court_id="ksd",
            docket_id="wrong-court-docket",
        ),
    )
    client = LookupClient(
        CourtListenerCitationLookup(citation="347 U.S. 483", status=200, clusters=clusters),
        docket_response=CourtListenerDocket(docket_id="wrong-court-docket", court_id="ksd"),
    )

    async def fake_choice(
        validation: object,
        *,
        summary: LocatorCitationSummaryNode,
        document_text: str,
        session: object | None,
    ) -> MelleaLocatorCandidateChoiceNode:
        del validation, document_text, session
        return MelleaLocatorCandidateChoiceNode(
            node_id=f"{summary.node_id}:mellea_candidate_choice",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.SELECTED,
            candidate_indices=(1,),
            selected_candidate_index=1,
            reparsed_case_name="Brown v. Board",
            reparsed_court="scotus",
            reparsed_date="1954",
            rationale="This is the only candidate.",
            depends_on=(summary.node_id,),
        )

    monkeypatch.setattr("mellea_lrc.validation.execution.run_mellea_locator_candidate_choice", fake_choice)
    progression = _validate_identity(extracted, client).citations[0]
    resolution = progression.identity_resolution

    assert resolution is not None
    assert resolution.outcome is LocatorIdentityResolutionOutcome.DEFERRED_TO_FUTURE_IMPLEMENTATION
    assert resolution.selected_candidate_index is None
    assert "court or year" in resolution.outcome_message


def test_docket_locator_is_untouched_without_service_access() -> None:
    document = _document(DocketCitation(docket_number="1:24-cv-08705"))
    client = LookupClient(CourtListenerCitationLookup(citation="unused", status=200, clusters=()))

    progression = _validate_identity(document, client).citations[0]

    assert client.calls == []
    assert progression.nodes == ()
    assert progression.identity_resolution is None


def test_non_reporter_locator_is_untouched_without_service_access() -> None:
    extracted = _document(FullLawCitation(volume="28", reporter="U.S.C.", page="636"))
    client = LookupClient(CourtListenerCitationLookup(citation="28 U.S.C. 636", status=200, clusters=()))

    progression = _validate(extracted, client).citations[0]

    assert client.calls == []
    assert progression.nodes == ()
    assert progression.identity_resolution is None


def test_service_failure_is_a_terminal_validation_node() -> None:
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="483"))
    client = LookupClient(
        CourtListenerError(
            "service unavailable",
            failure_type="upstream_error",
            retryable=True,
        )
    )

    node = _validate(extracted, client).citations[0].nodes[0]

    assert node.status is ValidationNodeStatus.FAILED
    assert node.outcome is LocatorLookupOutcome.FAILED
    assert node.error == "service unavailable"


def test_unexpected_lookup_response_raises() -> None:
    """Reject a response that violates the expected lookup contract."""
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="483"))
    client = LookupClient(CourtListenerCitationLookup(citation="347 U.S. 483", status=200, clusters=()))

    with pytest.raises(AssertionError, match="Unexpected CourtListener lookup response"):
        _validate(extracted, client)
