"""Native checkpoints reject invalid roots and report malformed pointers clearly."""

import json

import pytest
from pydantic import ValidationError

from mellea_lrc.matching.fuzziness import FuzzinessType
from mellea_lrc.model import Document, FullDocketCitation, Span
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID, latest
from mellea_lrc.model.citations.leaf_field_correction import (
    GroundedLeafFieldCorrection,
    LeafCorrectionWindow,
    LeafFieldCorrectionDecision,
    LeafFieldCorrectionProposal,
    LeafFieldCorrectionReview,
    RootValidationEvidenceReference,
)
from mellea_lrc.model.citations.reporter_opinion import RetrievedReporterOpinion
from mellea_lrc.model.citations.reporter_page_resolution import ReporterCitationPageResolution
from mellea_lrc.model.citations.reporter_pinpoint import (
    ReporterCitationPinpointEvidence,
    ReporterCitationProposition,
    ReporterCitationSupportReview,
    ReporterPinpointJudgment,
)
from mellea_lrc.model.preprocessed_document import PreprocessingMetadata
from mellea_lrc.model.source import SourceMetadata

TEXT = "No. 1:24-cv-00123; No. 2:24-cv-00456; No. 3:24-cv-00789."


def _discovered() -> Document:
    document = Document(
        text=TEXT,
        source_metadata=SourceMetadata(),
        preprocessing_metadata=PreprocessingMetadata(),
    )
    for identifier, number in zip(("a", "b", "c"), ("1:24-cv-00123", "2:24-cv-00456", "3:24-cv-00789")):
        start = TEXT.index(number) - 4
        document = document.add_citation(
            FullDocketCitation.from_locator(
                citation_id=identifier,
                substage="discover",
                source=TEXT,
                span=Span(start, start + 4 + len(number)),
                number_span=Span(start + 4, start + 4 + len(number)),
            )
        )
    return document.complete_substage("discover")


def _assign(document: Document, assignments: dict[str, str | None], substage: str) -> Document:
    for citation in document.citations:
        if citation.id in assignments:
            document = document.replace_citation(
                citation.record(substage).with_root(assignments[citation.id])
            )
    return document


@pytest.mark.parametrize(
    "assignments",
    (
        {"a": "b", "b": "a"},  # A cycle has no canonical root.
        {"a": "b", "b": "c", "c": "c"},  # Attachments must be direct, not chains.
        {"a": "b"},  # A full citation without its own root assignment is not a root.
        {"a": "b", "b": WITHDRAWN_ROOT_ID},
    ),
)
def test_invalid_root_attachments_fail_on_completion_and_native_loading(assignments):
    pending = _assign(_discovered(), assignments, "root")
    with pytest.raises(ValidationError, match="self-root"):
        pending.complete_substage("root")

    payload = pending.model_dump(mode="json")
    payload["runs"].append({"kind": "substage", "name": "root"})
    with pytest.raises(ValidationError, match="self-root"):
        Document.model_validate_json(json.dumps(payload))


def test_pending_root_assignments_can_be_completed_in_any_citation_order():
    discovered = _discovered()
    pending = _assign(discovered, {"a": "b"}, "root")
    completed = _assign(pending, {"b": "b"}, "root").complete_substage("root")

    assert tuple(root.id for root in completed.roots) == ("b",)
    assert tuple(leaf.id for leaf in completed.leaves) == ("a",)
    restored = Document.model_validate_json(completed.model_dump_json())
    assert restored == completed
    assert restored.get_substage("discover") == discovered
    assert restored.get_substage("root") == completed


def test_merging_roots_allows_pending_retargeting_and_preserves_earlier_checkpoint():
    rooted = _assign(_discovered(), {"a": "a", "b": "b", "c": "b"}, "root").complete_substage("root")
    pending = _assign(rooted, {"b": "a"}, "merge")
    with pytest.raises(ValidationError, match="self-root"):
        pending.complete_substage("merge")
    merged = _assign(pending, {"c": "a"}, "merge").complete_substage("merge")

    assert tuple(root.id for root in merged.roots) == ("a",)
    assert all(latest(citation.root_id) == "a" for citation in merged.citations)
    restored = Document.model_validate_json(merged.model_dump_json())
    assert restored.get_substage("root") == rooted
    assert restored.get_substage("merge") == merged


def test_later_root_repair_cannot_hide_an_invalid_earlier_native_checkpoint():
    pending = _assign(_discovered(), {"a": "b"}, "root")
    payload = pending.model_dump(mode="json")
    payload["citations"][1] = pending.citations[1].record("repair").with_root("b").model_dump(mode="json")
    payload["runs"].extend(({"kind": "substage", "name": "root"}, {"kind": "substage", "name": "repair"}))
    with pytest.raises(ValidationError, match="self-root"):
        Document.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("root_id", (None, WITHDRAWN_ROOT_ID))
def test_unattached_and_withdrawn_citations_need_no_canonical_root(root_id):
    document = _assign(_discovered(), {"a": root_id}, "root").complete_substage("root")
    assert document.roots == document.leaves == ()
    assert Document.model_validate_json(document.model_dump_json()) == document


def _pinpoint_citation() -> FullDocketCitation:
    citation = _discovered().citations[0].record("resolution")
    citation = citation.with_reporter_page_resolution(
        ReporterCitationPageResolution(
            node_id=citation.nodes[-1].id,
            root_id="a",
            locator_citation_id=None,
            locator_reading_index=None,
            pin_citation_id=None,
            pin_reading_index=None,
            outcome="no_pin",
            reason="No written pinpoint.",
        )
    ).record("proposition")
    citation = citation.with_reporter_proposition(
        ReporterCitationProposition(
            node_id=citation.nodes[-1].id,
            resolution_index=0,
            decision=None,
            failure_reason="No proposition.",
        )
    ).record("evidence")
    citation = citation.with_reporter_pinpoint_evidence(
        ReporterCitationPinpointEvidence(
            node_id=citation.nodes[-1].id,
            root_id="a",
            resolution_index=0,
            proposition_index=0,
            pages=(),
            outcome="no_proposition",
            reason="No proposition to assess.",
        )
    ).record("review")
    citation = citation.with_reporter_support_review(
        ReporterCitationSupportReview(
            node_id=citation.nodes[-1].id,
            evidence_index=0,
            scope="full_opinion",
            decision=None,
            failure_reason="No assessable evidence.",
        )
    ).record("judgment")
    return citation.with_reporter_pinpoint_judgment(
        ReporterPinpointJudgment(
            node_id=citation.nodes[-1].id,
            evidence_index=0,
            review_index=0,
            verdict="UNDETERMINED",
            pagination_available=False,
            correct_page=None,
            found_pages=(),
            reason="Evidence was unavailable.",
        )
    )


@pytest.mark.parametrize(
    "history",
    (
        "reporter_page_resolutions",
        "reporter_propositions",
        "reporter_pinpoint_evidence",
        "reporter_support_reviews",
        "reporter_pinpoint_judgments",
    ),
)
def test_missing_pinpoint_nodes_raise_validation_errors(history):
    payload = _pinpoint_citation().model_dump(mode="json")
    payload[history][0]["node_id"] = "missing"
    with pytest.raises(ValidationError, match="missing citation node"):
        FullDocketCitation.model_validate_json(json.dumps(payload))


def test_missing_corrected_field_node_raises_validation_error():
    citation = (
        _discovered().citations[0].record("name").with_case_name(TEXT, Span(0, 16)).record("correction")
    )
    reading = citation.case_name[0]
    citation = citation.with_leaf_field_correction_review(
        LeafFieldCorrectionReview(
            node_id=citation.nodes[-1].id,
            root_id="root",
            evidence_refs=(
                RootValidationEvidenceReference(history="locator", record_index=0, node_id="root:0"),
            ),
            windows=(LeafCorrectionWindow(field="case_name", span=reading.span),),
            decision=LeafFieldCorrectionDecision(
                fields=(
                    LeafFieldCorrectionProposal(
                        field="case_name",
                        propose_replacement=True,
                        quote=reading.quote,
                        reason="Keep grounded name.",
                    ),
                ),
                reason="Reread the source name.",
            ),
            grounded=(
                GroundedLeafFieldCorrection(
                    field="case_name",
                    quote=reading.quote,
                    span=reading.span,
                    match_type=FuzzinessType.PERFECT_MATCH,
                    similarity_percent=100,
                    edits=0,
                    reading_index=0,
                    applied=False,
                ),
            ),
        )
    )
    payload = citation.model_dump(mode="json")
    payload["case_name"][0]["node_id"] = "missing"
    with pytest.raises(ValidationError, match="missing citation node"):
        FullDocketCitation.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("missing_key", ("id", "cluster"))
def test_missing_raw_opinion_identifiers_raise_validation_errors(missing_key):
    response = {"id": 20, "cluster": 1, "plain_text": "Saved opinion."}
    del response[missing_key]
    with pytest.raises(ValidationError, match=f"missing its '{missing_key}' field"):
        RetrievedReporterOpinion.model_validate_json(
            json.dumps({"opinion_id": "20", "cluster_id": "1", "outcome": "retrieved", "response": response})
        )
