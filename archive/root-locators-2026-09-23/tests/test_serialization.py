"""Tests for validated-document serialization."""

import json
from dataclasses import replace

import pytest
from pydantic import TypeAdapter

from mellea_lrc.courtlistener import CourtListenerOpinionCluster, CourtListenerSearchResult
from mellea_lrc.extraction.adjudication.types import Candidate, CandidateKind, SiteReview
from mellea_lrc.llm import IvrAttempt, IvrRequirementAttempt, IvrRun
from mellea_lrc.model.case_names import CaseName
from mellea_lrc.model.citations import (
    CanonicalCitation,
    CitationDate,
    CitationField,
    CitationKind,
    DocketCitation,
    DocketEntry,
    FullCaseCitation,
    FullJournalCitation,
    FullLawCitation,
    IdCitation,
    ReferenceCitation,
    Reporter,
    ShortCaseCitation,
    SupraCitation,
    UnknownCitation,
    is_leaf,
    placed,
)
from mellea_lrc.model.document import Document
from mellea_lrc.model.evidence import EvidenceQuoteMatchMethod
from mellea_lrc.model.extraction_metadata import ExtractionMetadata
from mellea_lrc.model.findings import Finding, FindingKind
from mellea_lrc.model.operations import (
    assign_antecedent,
    assign_root,
    attribute_authority,
    create_citation,
    field_values,
    judge_citation,
    mark_extraction_reviewed,
    observe_citation,
    observe_document,
    record_colocation,
    resolve_citation,
    update_field,
    update_fields,
    withdraw_citation,
)
from mellea_lrc.model.pin_cites import PinCite
from mellea_lrc.model.preprocessed import PreprocessedDocument, Rule
from mellea_lrc.model.record import (
    WITHDRAWN_HEAD_ID,
    CitationRecord,
    DateExploration,
    Node,
    OperationKind,
    Question,
    Reads,
    Resolution,
)
from mellea_lrc.model.spans import Span
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.serialization import (
    deserialize_site_review_trace,
    deserialize_validated_document,
    serialize_site_review,
    serialize_validated_document,
)
from mellea_lrc.serialization.validated_document import SCHEMA_VERSION
from mellea_lrc.validation import (
    AggregatedFieldOutcome,
    CandidateEvaluationNode,
    CandidateEvaluationOutcome,
    CandidateEvaluationSource,
    CandidateProvenance,
    CandidateSelectionNode,
    CandidateSelectionOutcome,
    CitationSummaryAssessmentOutcome,
    CitationSummaryCandidate,
    CourtCheckNode,
    DocketCourtRetrievalNode,
    DocketCourtRetrievalOutcome,
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
    MelleaCitingPropositionExtractionOutcome,
    MelleaLocatorCandidateChoiceNode,
    MelleaLocatorCandidateChoiceOutcome,
    MelleaPinpointCheckNode,
    MelleaPinpointCheckOutcome,
    MelleaReextractedCaseNameCheckNode,
    OpinionSearchCandidateAssessmentNode,
    OpinionSearchNode,
    OpinionSearchOutcome,
    RecapSearchCandidateAssessmentNode,
    RecapSearchNode,
    RecapSearchOutcome,
    ReporterPageEvidence,
    ReporterPageRetrievalNode,
    ReporterPageRetrievalOutcome,
    SearchCandidateAssessmentOutcome,
    ValidationNodeStatus,
    YearCheckNode,
    initialize_full_reporter_locator_identity,
)


def _document_with_one_citation() -> Document:
    """Build one extracted case citation for serializer tests."""
    text = "Brown v. Board of Education, 347 U.S. 483 (1954)."
    preprocessed = preprocess(text)
    matched_text = "347 U.S. 483"
    start = text.index(matched_text)
    fields = placed(
        FullCaseCitation(
            plaintiff="Brown",
            defendant="Board of Education",
            volume="347",
            reporter=Reporter(as_written="U.S.", short_name="U.S.", is_scotus=True),
            page="483",
            date=CitationDate(year="1954"),
            court="scotus",
        ),
        span=Span(0, len(text) - 1),
        locator_span=Span(start, start + len(matched_text)),
        matched_text=matched_text,
    )
    record = create_citation(
        "cite-0001",
        CitationKind.FULL_CASE,
        Node("cite-0001:create", Reads.DOCUMENT, "extraction", "test", "classified"),
    )
    update_fields(
        record,
        Node("cite-0001:read", Reads.DOCUMENT, "extraction", "test", "read"),
        field_values(fields),
        reason="initial reading",
    )
    return Document(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(record,),
        extraction_metadata=ExtractionMetadata(),
    )


def _record_from_fields(
    citation_id: str, fields: CanonicalCitation, *, root_id: str | None = None
) -> CitationRecord:
    record = create_citation(
        citation_id,
        fields.kind,
        Node(f"{citation_id}:create", Reads.DOCUMENT, "extraction", "test", "classified"),
    )
    update_fields(
        record,
        Node(f"{citation_id}:read", Reads.DOCUMENT, "extraction", "test", "read"),
        field_values(fields),
        reason="initial reading",
    )
    if root_id is not None:
        record = assign_root(
            record,
            root_id,
            Node(f"{citation_id}:root", Reads.RECORD, "structure", "test", "attached"),
        )
    return record


def test_document_round_trip_preserves_recoverable_fields() -> None:
    """Preserve source provenance, both citation spans, and canonical fields."""
    document = _document_with_one_citation()

    payload = document.model_dump(mode="json")

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["artifact_type"] == "document"
    # Everything about the citation is written in one place, inside `citation`.
    written = payload["citations"][0]["fields"]
    assert written["span"] == {"start": 0, "end": len(document.text) - 1}
    assert written["locator_span"] == {"start": 29, "end": 41}
    # This citation states no pin cite, so it claims no pages.
    assert written["pin_cite"] is None
    assert Document.model_validate(payload) == document
    assert json.loads(json.dumps(payload)) == payload


def test_date_correction_survives_resuming_and_serializing_again() -> None:
    document = _document_with_one_citation()
    update_field(
        document.citations[0],
        Node(
            node_id="cite-0001:date-reread",
            reads=Reads.DOCUMENT,
            stage="source-reread",
            made_by="test",
            outcome="corrected",
        ),
        CitationField.DATE,
        CitationDate(year="1954", month="May", day="17"),
        reason="The filing states the full date.",
    )

    first = document.model_dump(mode="json")
    restored = Document.model_validate(first)

    assert restored.citations[0].field_updates == document.citations[0].field_updates
    assert restored.model_dump(mode="json") == first


def test_generic_get_field_and_case_name_property_read_latest_serialized_update() -> None:
    document = _document_with_one_citation()
    record = document.citations[0]
    source = document.text
    name_end = source.index(",")
    initial = CaseName(
        span=Span(0, 14),
        text=source[:14],
        plaintiff="Brown",
        defendant="Board",
    )
    corrected = CaseName(
        span=Span(0, name_end),
        text=source[:name_end],
        plaintiff="Brown",
        defendant="Board of Education",
    )
    update_field(
        record,
        Node("case-name-initial", Reads.DOCUMENT, "extraction", "test", "read"),
        CitationField.CASE_NAME,
        initial,
        reason="initial case-name reading",
    )
    update_field(
        record,
        Node("case-name-correction", Reads.DOCUMENT, "validation", "test", "corrected"),
        CitationField.CASE_NAME,
        corrected,
        reason="corrected case-name reading",
    )

    restored = Document.model_validate(json.loads(json.dumps(document.model_dump(mode="json"))))
    restored_record = restored.citations[0]
    assert restored_record.case_name == corrected
    assert restored_record.get_field(CitationField.CASE_NAME) == corrected
    assert restored_record.fields.case_name == corrected
    assert restored_record.field_updates[-1].node_id == "case-name-correction"


def test_initial_typed_field_updates_survive_native_round_trip() -> None:
    """An empty typed citation can be read in stages without losing update history."""
    document = Document.from_plain_text("Brown v. Board, 347 U.S. 483, 485 (1954).")
    created = Node("create", Reads.DOCUMENT, "extraction", "test", "classified")
    record = create_citation("cite-1", CitationKind.FULL_CASE, created)
    initial = Node("initial", Reads.DOCUMENT, "extraction", "test", "read")
    locator = "347 U.S. 483"
    start = document.text.index(locator)
    reporter = Reporter(as_written="U.S.", short_name="U.S.", is_scotus=True)
    update_fields(
        record,
        initial,
        {
            CitationField.SPAN: Span(0, len(document.text)),
            CitationField.LOCATOR_SPAN: Span(start, start + len(locator)),
            CitationField.MATCHED_TEXT: locator,
            CitationField.VOLUME: "347",
            CitationField.REPORTER: reporter,
            CitationField.PAGE: "483",
            CitationField.PIN_CITE: "485",
        },
        reason="first reading",
    )
    structured = PinCite.read("485", Span(start + len(locator) + 2, start + len(locator) + 5))
    update_field(
        record,
        Node("pin", Reads.DOCUMENT, "extraction", "test", "structured"),
        CitationField.PIN_CITE,
        structured,
        reason="parsed the pin cite",
    )
    document = document.evolve(citations=(record,))

    payload = document.model_dump(mode="json")
    restored = Document.model_validate(json.loads(json.dumps(payload)))

    assert payload["schema_version"] == 21
    assert "field_updates" not in payload["citations"][0]
    assert all("before" not in item for item in payload["citations"][0]["operations"])
    assert payload["citations"][0]["operations"][0]["after"] == CitationKind.FULL_CASE.value
    assert restored == document
    assert restored.citations[0].field_updates == record.field_updates
    assert [item.field for item in record.field_updates[:3]] == [
        CitationField.SPAN,
        CitationField.LOCATOR_SPAN,
        CitationField.MATCHED_TEXT,
    ]
    assert record.field_updates[-2].after == "485"
    assert record.field_updates[-1].after == structured


def test_every_operation_kind_survives_an_ordered_json_round_trip() -> None:
    """A checkpoint can resume after links, verdicts, lookup, review, and withdrawal."""
    document = _document_with_one_citation()
    text = document.text
    locator = "347 U.S. 483"
    locator_start = text.index(locator)
    record = create_citation(
        "cite-0002",
        CitationKind.FULL_CASE,
        Node("create-2", Reads.DOCUMENT, "extraction", "test", "created"),
    )
    update_fields(
        record,
        Node("read-2", Reads.DOCUMENT, "extraction", "test", "read"),
        {
            CitationField.SPAN: Span(0, len(text)),
            CitationField.LOCATOR_SPAN: Span(locator_start, locator_start + len(locator)),
            CitationField.MATCHED_TEXT: locator,
            CitationField.REPORTER: Reporter(as_written="U.S.", short_name="U.S.", is_scotus=True),
            CitationField.DATE: CitationDate(year="1954"),
        },
        reason="first reading",
    )
    record = assign_root(
        record,
        "cite-0001",
        Node("root-2", Reads.RECORD, "structure", "test", "linked"),
    )
    record = assign_antecedent(
        record,
        "cite-0001",
        Node("antecedent-2", Reads.RECORD, "structure", "test", "linked"),
    )
    record = record_colocation(
        record,
        "group-1",
        Node("colocation-2", Reads.RECORD, "structure", "test", "grouped"),
    )
    judge_citation(
        record,
        Node("judge-2", Reads.RECORD, "validation", "test", "resolved"),
        Question.IDENTITY,
        "resolved",
        message="Archive agrees.",
        type="third_party",
    )
    resolution = Resolution(
        cluster_id="cluster-1",
        case_name="Brown v. Board of Education",
        date_filed="1954-05-17",
        court_id="scotus",
        node_id="resolve-2",
        opinion_ids=("opinion-1",),
    )
    resolve_citation(
        record,
        Node("resolve-2", Reads.RECORD, "validation", "test", "found"),
        resolution,
    )
    attribute_authority(
        record,
        Node("authority-2", Reads.RECORD, "validation", "test", "attributed"),
        "cite-0001",
    )
    mark_extraction_reviewed(
        record,
        Node("review-2", Reads.DOCUMENT, "extraction", "test", "reviewed"),
    )
    withdraw_citation(
        record,
        Node("withdraw-2", Reads.RECORD, "validation", "test", "withdrawn"),
    )
    document = document.evolve(citations=(*document.citations, record))

    payload = json.loads(json.dumps(document.model_dump(mode="json")))
    written = payload["citations"][1]
    restored = Document.model_validate(payload)

    assert "field_updates" not in written
    assert [operation["kind"] for operation in written["operations"]] == [
        operation.kind.value for operation in record.operations
    ]
    assert {operation.kind for operation in record.operations} == set(OperationKind)
    assert "withdrawn_by" not in written
    assert written["root_id"] == WITHDRAWN_HEAD_ID
    assert written["root_link_node_id"] == "withdraw-2"
    assert written["operations"][-1] == {
        "kind": OperationKind.ROOT_LINK.value,
        "node_id": "withdraw-2",
        "after": WITHDRAWN_HEAD_ID,
    }
    assert written["operations"][0]["after"] == CitationKind.FULL_CASE.value
    assert next(operation for operation in written["operations"] if operation["kind"] == "resolution")[
        "after"
    ]["opinion_ids"] == ["opinion-1"]
    assert restored == document
    assert restored.citations[1].operations == record.operations
    assert restored.model_dump(mode="json") == payload


def test_preprocessing_rules_and_index_spans_survive_document_round_trip() -> None:
    located = _document_with_one_citation()
    preprocessed = PreprocessedDocument(
        source_metadata=located.source_metadata,
        text=located.text,
        preprocessing_metadata=replace(
            located.preprocessing_metadata,
            rules=(Rule.TABLE_AS_TEXT, Rule.TABLE_OF_AUTHORITIES),
        ),
        index_spans=(Span(0, 5), Span(8, 20)),
    )
    document = Document.from_preprocessed(preprocessed).evolve(citations=located.citations)

    payload = json.loads(json.dumps(document.model_dump(mode="json")))

    assert payload["preprocessing_metadata"]["rules"] == [
        "table_as_text",
        "table_of_authorities",
    ]
    assert payload["index_spans"] == [{"start": 0, "end": 5}, {"start": 8, "end": 20}]
    assert Document.model_validate(payload) == document


def test_checkpoint_after_one_update_resumes_with_complete_history() -> None:
    """A saved stage result is sufficient to continue; earlier files are not needed."""
    text = "Brown v. Board, 347 U.S. 483."
    document = Document.from_plain_text(text)
    record = create_citation(
        "cite-1",
        CitationKind.FULL_CASE,
        Node("create", Reads.DOCUMENT, "extraction", "test", "classified"),
    )
    locator = "347 U.S. 483"
    start = text.index(locator)
    update_fields(
        record,
        Node("read", Reads.DOCUMENT, "extraction", "test", "read"),
        {
            CitationField.SPAN: Span(start, start + len(locator)),
            CitationField.LOCATOR_SPAN: Span(start, start + len(locator)),
            CitationField.MATCHED_TEXT: locator,
        },
        reason="first locator reading",
    )
    record = assign_root(
        record,
        record.citation_id,
        Node("root", Reads.RECORD, "root_formation", "test", "attached"),
    )
    document = document.evolve(citations=(record,))

    first = Document.model_validate_json(document.model_dump_json())
    update_field(
        first.citations[0],
        Node("court", Reads.DOCUMENT, "court", "test", "read"),
        CitationField.COURT,
        "scotus",
        reason="court stated by the citation",
    )
    second = Document.model_validate_json(first.model_dump_json())
    judge_citation(
        second.citations[0],
        Node("identity", Reads.RECORD, "identity", "test", "confirmed"),
        Question.IDENTITY,
        "confirmed",
    )
    final = Document.model_validate_json(second.model_dump_json())

    assert [event.kind.value for event in final.citations[0].operations] == [
        "create",
        "field_update",
        "field_update",
        "field_update",
        "root_link",
        "field_update",
        "judgement",
    ]
    assert final.citations[0].fields.court == "scotus"
    assert final.citations[0].judgement(Question.IDENTITY).outcome == "confirmed"
    assert first.citations[0].judgement(Question.IDENTITY).outcome != "confirmed"

    final.citations[0].fields = replace(final.citations[0].fields, court="different")
    with pytest.raises(ValueError, match=r"fields\.court state disagrees with operation history"):
        Document.model_validate_json(final.model_dump_json())


def test_third_party_identity_judgement_type_survives_round_trip() -> None:
    document = _document_with_one_citation()
    record = document.citations[0]
    judge_citation(
        record,
        Node(
            node_id="cite-0001:third_party_identity",
            reads=Reads.RECORD,
            stage="identity",
            made_by="third_party_citation_check",
            outcome="corroborated",
        ),
        Question.IDENTITY,
        "resolved",
        type="third_party",
    )

    payload = document.model_dump(mode="json")
    written = payload["citations"][0]["judgements"]["identity"]
    assert written["type"] == "third_party"

    restored = Document.model_validate(json.loads(json.dumps(payload)))
    assert restored == document
    assert restored.citations[0].judgement(Question.IDENTITY).type == "third_party"
    assert restored.citations[0].judgement(Question.PINPOINT).type is None


def test_untyped_judgements_remain_untyped_in_serialized_documents() -> None:
    payload = _document_with_one_citation().model_dump(mode="json")

    assert payload["citations"][0]["judgements"]["identity"]["type"] is None
    assert Document.model_validate(payload).citations[0].judgement(Question.IDENTITY).type is None


def test_document_round_trip_preserves_an_optional_docket_entry() -> None:
    """A filed-document citation keeps both its case and entry locators."""
    text = "Doc. 75, Case No. 1:25-cv-00312-RPK"
    preprocessed = preprocess(text)
    entry_span = Span(0, len("Doc. 75"))
    locator_start = text.index("Case No.")
    document = Document(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(
            _record_from_fields(
                "docket-1",
                placed(
                    DocketCitation(
                        docket_number="1:25-cv-00312-RPK",
                        docket_entry=DocketEntry(number="75", span=entry_span),
                    ),
                    span=Span(0, len(text)),
                    locator_span=Span(locator_start, len(text)),
                    matched_text=text[locator_start:],
                ),
            ),
        ),
        extraction_metadata=ExtractionMetadata(),
    )

    payload = document.model_dump(mode="json")

    assert payload["citations"][0]["fields"]["docket_entry"] == {
        "number": "75",
        "span": {"start": 0, "end": 7},
    }
    assert Document.model_validate(payload) == document


def test_document_round_trip_preserves_a_rejected_site_review_run() -> None:
    """A failed review remains inspectable without calling the model again."""
    document = _document_with_one_citation()
    run = IvrRun(
        success=False,
        selected_attempt=1,
        attempts=(
            IvrAttempt(
                output='{"is_docket_citation":true}',
                requirements=(IvrRequirementAttempt("Return JSON.", False, "court missing", None),),
            ),
            IvrAttempt(
                output="",
                requirements=(IvrRequirementAttempt("Return JSON.", False, "empty output", None),),
            ),
        ),
        backend="OpenAIBackend",
        model="z-ai/glm-5.3-flash",
        model_options={"max_tokens": 1200},
        instruction="Decide whether the site is a docket.",
        prefix=None,
        grounding_context={},
        user_variables={"docket": "No. 25-11030"},
        output_schema={"type": "object"},
    )
    node = Node(
        node_id="docket:10-22",
        reads=Reads.DOCUMENT,
        stage="docket",
        made_by="adjudicate_docket",
        outcome="declined",
        details=serialize_site_review(
            Candidate(
                generator="suspected_dockets",
                kind=CandidateKind.DOCKET,
                span=Span(10, 22),
                window=Span(0, len(document.text)),
                note="A docket label followed by an opaque identifier.",
            ),
            SiteReview(answer=None, reason="No valid structured response.", run=run),
        ),
    )
    document = observe_document(document, node)
    object.__setattr__(
        document,
        "findings",
        (
            Finding(
                kind=FindingKind.SITE_REVIEW,
                stage="docket",
                made_by="adjudicate_docket",
                message="No valid structured response.",
                node_id=node.node_id,
                span=Span(10, 22),
            ),
        ),
    )

    recovered = Document.model_validate(document.model_dump(mode="json"))

    assert recovered.nodes == (node,)
    assert recovered.findings[0].node_id == node.node_id
    trace = deserialize_site_review_trace(recovered.nodes[0].details)
    assert trace.ivr == run
    assert trace.reason == "No valid structured response."
    assert trace.candidate.kind is CandidateKind.DOCKET


def test_date_review_trace_is_json_checkpointable() -> None:
    document = _document_with_one_citation()
    observe_citation(
        document.citations[0],
        Node(
            node_id="date-review",
            reads=Reads.RECORD,
            stage="full_reporter_exact_date_review",
            made_by="test",
            outcome="resolved",
            details={
                "date_exploration": TypeAdapter(DateExploration).dump_python(
                    DateExploration(
                        stated="1948",
                        stated_precision="year",
                        record_date_filed="1949-01-03",
                        other_dates=None,
                        phrases_by_opinion=(("123", ("Decided December 20, 1948",)),),
                        matched_phrase="Decided December 20, 1948",
                        matched_opinion_id="123",
                    ),
                    mode="json",
                )
            },
        ),
    )

    restored = Document.model_validate(json.loads(json.dumps(document.model_dump(mode="json"))))

    assert restored.citations[0].trace[-1].details["date_exploration"]["matched_opinion_id"] == "123"


def test_document_round_trip_supports_every_canonical_citation_type() -> None:
    """Keep each canonical citation shape recoverable from an extracted artifact."""
    citations = (
        FullCaseCitation(
            plaintiff="A",
            defendant="B",
            volume="1",
            reporter=Reporter(as_written="U.S.", short_name="U.S.", is_scotus=True),
            page="2",
        ),
        FullLawCitation(volume="1", reporter=Reporter(as_written="U.S.C.", short_name="U.S.C."), page="2"),
        FullJournalCitation(volume="1", reporter="Harv. L. Rev.", page="2"),
        ShortCaseCitation(
            volume="1", reporter=Reporter(as_written="U.S.", short_name="U.S.", is_scotus=True), page="2"
        ),
        SupraCitation(pin_cite="2", antecedent="A"),
        IdCitation(pin_cite="2"),
        ReferenceCitation(plaintiff="A", defendant="B", pin_cite="at 2"),
        UnknownCitation(),
    )
    source = preprocess("x" * len(citations))
    document = Document(
        source_metadata=source.source_metadata,
        text=source.text,
        preprocessing_metadata=source.preprocessing_metadata,
        citations=tuple(
            _record_from_fields(
                f"cite-{index}",
                # Every leaf reaches a root, because nothing else can exist.
                # `cite-0` is the first of the full citations, so it serves.
                root_id="cite-0" if index == 0 or is_leaf(citation) else None,
                # A pin cite is scored on its own, so its span has to survive
                # the round trip like any other offset.
                fields=placed(
                    citation,
                    span=Span(index, index + 1),
                    locator_span=Span(index, index + 1),
                    matched_text="x",
                    **(
                        {"pin_cite": PinCite.read(citation.pin_cite, Span(index, index + 1))}
                        if getattr(citation, "pin_cite", None)
                        else {}
                    ),
                ),
            )
            for index, citation in enumerate(citations)
        ),
        extraction_metadata=ExtractionMetadata(),
    )

    assert Document.model_validate(document.model_dump(mode="json")) == document


def test_serialize_validated_document_preserves_source_and_node_graph() -> None:
    """Emit one source citation and its explicit validation-node dependency."""
    text = "Brown v. Board of Education, 347 U.S. 483 (1954)."
    preprocessed = preprocess(text)
    matched_text = "347 U.S. 483"
    start = text.index(matched_text)
    extracted = Document(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(
            _record_from_fields(
                "cite-0001",
                placed(
                    FullCaseCitation(
                        plaintiff="Brown",
                        defendant="Board of Education",
                        volume="347",
                        reporter=Reporter(as_written="U.S.", short_name="U.S.", is_scotus=True),
                        page="483",
                        date=CitationDate(year="1954"),
                        court="scotus",
                    ),
                    span=Span(start, start + len(matched_text)),
                    locator_span=Span(start, start + len(matched_text)),
                    matched_text=matched_text,
                ),
            ),
        ),
        extraction_metadata=ExtractionMetadata(),
    )
    initialized = initialize_full_reporter_locator_identity(extracted)
    node = ExactLocatorLookupNode(
        node_id="cite-0001:exact_locator_lookup",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorLookupOutcome.FOUND,
        locator=matched_text,
        cluster=CourtListenerOpinionCluster(case_name="Brown v. Board of Education"),
        candidate_count=1,
    )
    validated = type(initialized)(
        source=initialized.source,
        citations=(initialized.citations[0].append(node),),
    )

    payload = serialize_validated_document(validated)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["artifact_type"] == "validated_document"
    assert payload["source"]["artifact_type"] == "document"
    assert payload["source"]["citations"][0]["fields"]["kind"] == "FullCaseCitation"
    assert payload["citations"] == [
        {
            "citation_id": "cite-0001",
            "aggregation": None,
            "nodes": [
                {
                    "node_type": "ExactLocatorLookupNode",
                    "node_id": "cite-0001:exact_locator_lookup",
                    "status": "succeeded",
                    "outcome": "found",
                    "locator": "347 U.S. 483",
                    "cluster": {
                        "cluster_id": None,
                        "opinion_url": None,
                        "case_name": "Brown v. Board of Education",
                        "date_filed": None,
                        "other_dates": None,
                        "court": None,
                        "court_id": None,
                        "docket_id": None,
                        "citations": [],
                        "sub_opinion_ids": [],
                    },
                    "candidate_clusters": [],
                    "candidate_count": 1,
                    "status_message": None,
                    "outcome_message": None,
                    "error": None,
                    "depends_on": [],
                    "retrospective_date": None,
                    "raw_candidate_count": None,
                    "excluded_candidate_count": 0,
                }
            ],
        }
    ]
    assert deserialize_validated_document(payload) == validated
    assert json.loads(json.dumps(payload)) == payload


def test_null_year_judgment_survives_validated_document_round_trip() -> None:
    """A missing date is serialized as null through field and candidate layers."""
    initialized = initialize_full_reporter_locator_identity(_document_with_one_citation())
    validation = (
        initialized.citations[0]
        .append(
            YearCheckNode(
                node_id="cite-0001:year_check",
                status=ValidationNodeStatus.SKIPPED,
                outcome=None,
                extracted_year="1954",
                retrieved_year=None,
                depends_on=(),
            )
        )
        .append(
            LocatorCandidateAssessmentNode(
                node_id="cite-0001:candidate_assessment",
                status=ValidationNodeStatus.SUCCEEDED,
                outcome=LocatorCandidateAssessmentOutcome.MATCH,
                candidate_index=1,
                extracted_citation="347 U.S. 483",
                extracted_case_name="Brown v. Board of Education",
                retrieved_case_name="Brown v. Board of Education",
                case_name_outcome=AggregatedFieldOutcome.MATCH,
                case_name_evidence="exact",
                extracted_year="1954",
                retrieved_year=None,
                year_outcome=None,
                extracted_court_id="scotus",
                retrieved_court_id="scotus",
                court_outcome=AggregatedFieldOutcome.MATCH,
                docket_id=None,
                depends_on=("cite-0001:year_check",),
            )
        )
    )
    validated = type(initialized)(source=initialized.source, citations=(validation,))
    payload = serialize_validated_document(validated)

    assert payload["citations"][0]["nodes"][0]["outcome"] is None
    assert payload["citations"][0]["nodes"][1]["year_outcome"] is None
    assert deserialize_validated_document(payload) == validated


def test_document_serialization_preserves_model_reparse_provenance() -> None:
    document = _document_with_one_citation()
    node = Node("cite-0001:review", Reads.DOCUMENT, "extraction_review", "model", "reviewed")
    mark_extraction_reviewed(document.citations[0], node)

    payload = document.model_dump(mode="json")

    assert payload["citations"][0]["extraction_reviewed_by_llm"] is True
    assert payload["citations"][0]["extraction_review_node_id"] == node.node_id
    restored = Document.model_validate(payload).citations[0]
    assert restored.extraction_reviewed_by_llm is True
    assert restored.extraction_review_node_id == node.node_id
    assert node in restored.trace


def test_serialize_validated_document_preserves_frozen_opinion_search_results() -> None:
    """Expose immutable upstream opinion results without losing their fields."""
    document = _document_with_one_citation()
    initialized = initialize_full_reporter_locator_identity(document)
    preparation = MelleaCaseNameQueryPreparationNode(
        node_id="cite-0001:mellea_case_name_query_preparation",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=MelleaCaseNameQueryPreparationOutcome.PREPARED,
        query='caseName:("Brown" AND "Board")',
        query_plaintiff="Brown",
        query_defendant="Board",
        court_id="scotus",
        depends_on=(),
    )
    response = CourtListenerSearchResult.from_payload(
        query=preparation.query,
        search_type="o",
        semantic=False,
        count=1,
        results=[{"cluster_id": 123, "caseName": "Brown v. Board", "meta": {"rank": 1}}],
        next_cursor=None,
        previous_cursor=None,
    )
    search = OpinionSearchNode(
        node_id="cite-0001:opinion_search",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=OpinionSearchOutcome.SEARCHED,
        query=response.query,
        result_count=response.count,
        results=response.results,
        next_cursor=response.next_cursor,
        depends_on=(preparation.node_id,),
    )
    validated = type(initialized)(
        source=initialized.source,
        citations=(initialized.citations[0].append(preparation).append(search),),
    )

    node = serialize_validated_document(validated)["citations"][0]["nodes"][1]

    assert node["results"] == [{"cluster_id": 123, "caseName": "Brown v. Board", "meta": {"rank": 1}}]
    assert json.loads(json.dumps(node)) == node
    assert deserialize_validated_document(serialize_validated_document(validated)) == validated


def test_validated_document_round_trip_supports_every_current_node_type() -> None:
    """Keep every explicit progression node recoverable as the graph grows."""
    initialized = initialize_full_reporter_locator_identity(_document_with_one_citation())
    citation_id = "cite-0001"
    lookup_id = f"{citation_id}:exact_locator_lookup"
    exact_id = f"{citation_id}:exact_case_name_check"
    semantic_id = f"{citation_id}:mellea_case_name_check"
    reextraction_id = f"{citation_id}:mellea_case_name_reextraction"
    preparation_id = f"{citation_id}:mellea_case_name_query_preparation"
    opinion_search_id = f"{citation_id}:opinion_search"
    selection_id = f"{citation_id}:candidate_selection"
    candidate_id = f"{citation_id}:candidate_evaluation"
    cluster = CourtListenerOpinionCluster(
        cluster_id="123",
        case_name="Brown v. Board of Education",
        date_filed="1954-05-17",
        court="Supreme Court",
        court_id="scotus",
        docket_id="42",
    )
    ivr_run = IvrRun(
        success=True,
        selected_attempt=0,
        attempts=(IvrAttempt(output='{"verdict":"match"}', requirements=()),),
        backend="OpenAIBackend",
        model="z-ai/glm-5.3-flash",
        model_options={"max_tokens": 128},
        instruction="Classify case names.",
        prefix=None,
        grounding_context={},
        user_variables={"extracted_case_name": "Brown v. Board"},
        output_schema={"type": "object"},
    )
    nodes = (
        ExactLocatorLookupNode(
            node_id=lookup_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=LocatorLookupOutcome.FOUND,
            locator="347 U.S. 483",
            cluster=cluster,
            candidate_count=1,
        ),
        ExactCaseNameCheckNode(
            node_id=exact_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=FieldCheckOutcome.MISMATCH,
            extracted_case_name="Brown v. Board",
            retrieved_case_name=cluster.case_name,
            depends_on=(lookup_id,),
        ),
        MelleaCaseNameCheckNode(
            node_id=semantic_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCaseNameCheckOutcome.MATCH,
            extracted_case_name="Brown v. Board",
            retrieved_case_name=cluster.case_name or "",
            depends_on=(exact_id,),
            run=ivr_run,
        ),
        MelleaCaseNameReextractionNode(
            node_id=reextraction_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCaseNameReextractionOutcome.COMPLETE,
            plaintiff="Brown",
            defendant="Board of Education",
            depends_on=(semantic_id,),
        ),
        MelleaCaseNameQueryPreparationNode(
            node_id=preparation_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCaseNameQueryPreparationOutcome.PREPARED,
            query="Brown Board of Education",
            query_plaintiff="Brown",
            query_defendant="Board of Education",
            court_id="scotus",
            depends_on=(reextraction_id,),
        ),
        OpinionSearchNode(
            node_id=opinion_search_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=OpinionSearchOutcome.SEARCHED,
            query="Brown Board of Education",
            result_count=1,
            results=({"cluster_id": 123, "meta": {"rank": 1}},),
            next_cursor="next",
            depends_on=(preparation_id,),
        ),
        RecapSearchNode(
            node_id=f"{citation_id}:recap_search",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=RecapSearchOutcome.SEARCHED,
            query="Brown Board of Education",
            result_count=1,
            results=({"docket_id": 42},),
            next_cursor=None,
            depends_on=(preparation_id,),
        ),
        CandidateSelectionNode(
            node_id=selection_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=CandidateSelectionOutcome.ALL_SELECTED,
            total_candidate_count=1,
            selected_candidate_count=1,
            selection_limit=20,
            depends_on=(opinion_search_id,),
        ),
        CandidateEvaluationNode(
            node_id=candidate_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=CandidateEvaluationOutcome.READY,
            source=CandidateEvaluationSource.OPINION_SEARCH,
            candidate_index=0,
            cluster_id="123",
            case_name=cluster.case_name,
            date_filed=cluster.date_filed,
            court_id=cluster.court_id,
            docket_id=cluster.docket_id,
            docket_number=None,
            record={"cluster_id": 123, "meta": {"rank": 1}},
            depends_on=(selection_id,),
        ),
        MelleaReextractedCaseNameCheckNode(
            node_id=f"{citation_id}:mellea_reextracted_case_name_check",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCaseNameCheckOutcome.MATCH,
            reextracted_case_name="Brown v. Board of Education",
            retrieved_case_name=cluster.case_name,
            depends_on=(reextraction_id,),
        ),
        DocketCourtRetrievalNode(
            node_id=f"{citation_id}:docket_court_retrieval",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=DocketCourtRetrievalOutcome.FOUND,
            docket_id="42",
            court_id="scotus",
            depends_on=(candidate_id,),
        ),
        CourtCheckNode(
            node_id=f"{citation_id}:court_check",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=FieldCheckOutcome.MATCH,
            extracted_court_id="scotus",
            retrieved_court_id="scotus",
            depends_on=(candidate_id,),
        ),
        ReporterPageRetrievalNode(
            node_id=f"{candidate_id}:reporter_page_retrieval",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=ReporterPageRetrievalOutcome.FOUND,
            cluster_id="123",
            reporter_citation="347 U.S. 483",
            pin_cite="483",
            citation_index=1,
            evidence=ReporterPageEvidence(
                opinion_id="456",
                opinion_type="020lead",
                text="The plaintiffs contend that segregated public schools are not equal.",
            ),
            depends_on=(candidate_id,),
        ),
        MelleaCitingPropositionExtractionNode(
            node_id=f"{candidate_id}:reporter_page_retrieval:mellea_citing_proposition_extraction",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCitingPropositionExtractionOutcome.IDENTIFIED,
            context_span=Span(0, 100),
            reasoning="The surrounding sentence attributes the standard to this citation.",
            proposition="Segregated public schools are inherently unequal.",
            proposition_span=Span(10, 60),
            proposition_match_method=EvidenceQuoteMatchMethod.EXACT,
            proposition_match_score=1.0,
            depends_on=(f"{candidate_id}:reporter_page_retrieval",),
        ),
        MelleaPinpointCheckNode(
            node_id=f"{candidate_id}:reporter_page_retrieval:mellea_pinpoint_check",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaPinpointCheckOutcome.SUPPORTS,
            reasoning="The retrieved page states the proposition verbatim.",
            evidence_quote="segregated public schools are not equal",
            evidence_span=Span(20, 60),
            evidence_match_method=EvidenceQuoteMatchMethod.NORMALIZED,
            evidence_match_score=0.95,
            depends_on=(
                f"{candidate_id}:reporter_page_retrieval",
                f"{candidate_id}:reporter_page_retrieval:mellea_citing_proposition_extraction",
            ),
        ),
        YearCheckNode(
            node_id=f"{citation_id}:year_check",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=FieldCheckOutcome.MATCH,
            extracted_year="1954",
            retrieved_year="1954",
            depends_on=(candidate_id,),
        ),
        OpinionSearchCandidateAssessmentNode(
            node_id=f"{candidate_id}:opinion_search_candidate_assessment",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=SearchCandidateAssessmentOutcome.POSSIBLE_MATCH,
            candidate_index=0,
            extracted_citation="347 U.S. 483",
            extracted_case_name="Brown v. Board",
            retrieved_case_name=cluster.case_name,
            case_name_outcome=AggregatedFieldOutcome.MATCH,
            case_name_evidence="mellea",
            extracted_year="1954",
            retrieved_year="1954",
            year_outcome=AggregatedFieldOutcome.MATCH,
            extracted_court_id="scotus",
            retrieved_court_id="scotus",
            court_outcome=AggregatedFieldOutcome.MATCH,
            docket_id="42",
            depends_on=(
                semantic_id,
                f"{citation_id}:year_check",
                f"{citation_id}:court_check",
            ),
        ),
        RecapSearchCandidateAssessmentNode(
            node_id=f"{candidate_id}:recap_search_candidate_assessment",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=SearchCandidateAssessmentOutcome.POSSIBLE_MATCH,
            candidate_index=0,
            extracted_citation="347 U.S. 483",
            extracted_case_name="Brown v. Board",
            retrieved_case_name=cluster.case_name,
            case_name_outcome=AggregatedFieldOutcome.MATCH,
            case_name_evidence="mellea",
            extracted_year="1954",
            retrieved_year="1954",
            year_outcome=AggregatedFieldOutcome.MATCH,
            extracted_court_id="scotus",
            retrieved_court_id="scotus",
            court_outcome=AggregatedFieldOutcome.MATCH,
            docket_id="42",
            depends_on=(
                semantic_id,
                f"{citation_id}:year_check",
                f"{citation_id}:court_check",
            ),
        ),
        LocatorCandidateAssessmentNode(
            node_id=f"{candidate_id}:locator_candidate_assessment",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=LocatorCandidateAssessmentOutcome.MATCH,
            candidate_index=0,
            extracted_citation="347 U.S. 483",
            extracted_case_name="Brown v. Board",
            retrieved_case_name=cluster.case_name,
            case_name_outcome=AggregatedFieldOutcome.MATCH,
            case_name_evidence="mellea",
            extracted_year="1954",
            retrieved_year="1954",
            year_outcome=AggregatedFieldOutcome.MATCH,
            extracted_court_id="scotus",
            retrieved_court_id="scotus",
            court_outcome=AggregatedFieldOutcome.MATCH,
            docket_id="42",
            depends_on=(
                semantic_id,
                f"{citation_id}:year_check",
                f"{citation_id}:court_check",
            ),
        ),
        LocatorCitationSummaryNode(
            node_id=f"{citation_id}:locator_citation_summary",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=LocatorCitationSummaryOutcome.COMPLETE,
            overall_outcome=CitationSummaryAssessmentOutcome.MATCH,
            pinpoint_requires_review=None,
            candidates=(
                CitationSummaryCandidate(
                    provenance=CandidateProvenance.OPINION,
                    candidate_index=0,
                    assessment_node_id=f"{candidate_id}:locator_candidate_assessment",
                    outcome=LocatorCandidateAssessmentOutcome.MATCH,
                    extracted_citation="347 U.S. 483",
                    extracted_case_name="Brown v. Board",
                    retrieved_case_name=cluster.case_name,
                    case_name_outcome=AggregatedFieldOutcome.MATCH,
                    case_name_evidence="mellea",
                    extracted_year="1954",
                    retrieved_year="1954",
                    year_outcome=AggregatedFieldOutcome.MATCH,
                    extracted_court_id="scotus",
                    retrieved_court_id="scotus",
                    court_outcome=AggregatedFieldOutcome.MATCH,
                    docket_id="42",
                ),
            ),
            depends_on=(f"{candidate_id}:locator_candidate_assessment",),
        ),
        MelleaLocatorCandidateChoiceNode(
            node_id=f"{citation_id}:locator_citation_summary:mellea_candidate_choice",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.SELECTED,
            candidate_indices=(0,),
            selected_candidate_index=0,
            rationale="The complete candidate record supports the cited fields.",
            depends_on=(f"{citation_id}:locator_citation_summary",),
            run=ivr_run,
        ),
        LocatorIdentityResolutionNode(
            node_id=f"{citation_id}:locator_identity_resolution",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=LocatorIdentityResolutionOutcome.RESOLVED,
            selected_candidate_index=0,
            selected_assessment_node_id=f"{candidate_id}:locator_candidate_assessment",
            matching_candidate_indices=(0,),
            selection_evidence_node_id=f"{citation_id}:locator_citation_summary:mellea_candidate_choice",
            depends_on=(
                f"{citation_id}:locator_citation_summary",
                f"{citation_id}:locator_citation_summary:mellea_candidate_choice",
            ),
        ),
    )
    validation = initialized.citations[0]
    for node in nodes:
        validation = validation.append(node)
    document = type(initialized)(source=initialized.source, citations=(validation,))

    assert deserialize_validated_document(serialize_validated_document(document)) == document


def test_a_pin_cite_that_claims_no_page_survives_the_round_trip() -> None:
    """The pages are read back, not recomputed.

    `written_but_no_page` is the pin-cite reviewer saying that these characters
    claim no page. Recomputing from the text turns `74950` straight back into
    page 74,950, which throws the finding away.
    """
    claim = PinCite(span=Span(start=10, end=15), text="74950", pages=())

    adapter = TypeAdapter(PinCite)
    payload = adapter.dump_python(claim, mode="json")
    back = adapter.validate_python(payload)

    assert back == claim
    assert back.pages == ()
    assert PinCite.read("74950").pages != ()
