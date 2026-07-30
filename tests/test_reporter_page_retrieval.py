"""Tests for reporter-page retrieval from citation-aware opinion HTML."""

from dataclasses import dataclass

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener.opinion_models import (
    CourtListenerOpinion,
    CourtListenerOpinionCluster,
    CourtListenerOpinionClusterCitation,
)
from mellea_lrc.extraction import ExtractedCitation
from mellea_lrc.validation.pinpoint_retrieval import run_reporter_page_retrieval
from mellea_lrc.validation.pinpoint_retrieval.reporter_page import extract_reporter_page
from mellea_lrc.validation.types import (
    CandidateEvaluationNode,
    CandidateEvaluationOutcome,
    CandidateEvaluationSource,
    CitationValidation,
    ReporterPageRetrievalOutcome,
    ValidationNodeStatus,
)


@dataclass
class OpinionClient:
    """Deterministic opinion source for retrieval tests."""

    opinions: dict[str, CourtListenerOpinion]

    def get_opinion(self, opinion_id: str) -> CourtListenerOpinion:
        """Return one configured opinion."""
        return self.opinions[opinion_id]


def _validation(
    *,
    reporter: str = "F.3d",
    pin_cite: str | None = "623",
    cluster_reporter: str = "F.3d",
    sub_opinion_ids: tuple[str, ...] = ("lead", "combined"),
) -> tuple[CitationValidation, CandidateEvaluationNode]:
    extracted = ExtractedCitation(
        citation_id="cite-1",
        span=Span(0, 20),
        locator_span=Span(0, 14),
        matched_text="376 F.3d 615",
        citation=FullCaseCitation(
            volume="376",
            reporter=reporter,
            page="615",
            pin_cite=pin_cite,
        ),
    )
    cluster = CourtListenerOpinionCluster(
        cluster_id="2971299",
        citations=(CourtListenerOpinionClusterCitation("376", cluster_reporter, "615"),),
        sub_opinion_ids=sub_opinion_ids,
    )
    evaluation = CandidateEvaluationNode(
        node_id="cite-1:candidate_evaluation:1",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=CandidateEvaluationOutcome.READY,
        source=CandidateEvaluationSource.LOCATOR_LOOKUP,
        candidate_index=1,
        cluster_id=cluster.cluster_id,
        case_name=cluster.case_name,
        date_filed=cluster.date_filed,
        court_id=cluster.court_id,
        docket_id=cluster.docket_id,
        record=cluster,
        depends_on=("cite-1:lookup",),
    )
    return CitationValidation(citation=extracted), evaluation


def test_extract_reporter_page_accepts_both_courtlistener_marker_tags() -> None:
    """Marker attributes, rather than a particular HTML tag, define pages."""
    star_html = (
        '<p>prior</p><span citation-index="1" class="star-pagination" label="623">'
        "*623</span> target <b>rule</b>"
        '<span citation-index="1" label="624">*624</span> next'
    )
    page_number_html = (
        '<p>prior</p><page-number citation-index="1" label="153">*153</page-number>'
        " target rule"
        '<page-number citation-index="1" label="154">*154</page-number> next'
    )

    assert extract_reporter_page(star_html, citation_index=1, page_label="623") == ("*623 target rule")
    assert extract_reporter_page(page_number_html, citation_index=1, page_label="153") == ("*153 target rule")


def test_reporter_page_retrieval_prefers_lead_over_combined_and_dissent() -> None:
    """Returned order does not override controlling-opinion type priority."""
    validation, evaluation = _validation(
        sub_opinion_ids=("combined", "dissent", "lead"),
    )
    lead_html = (
        "infringe"
        '<page-number citation-index="1" label="623">*623</page-number>'
        "ment. Actual knowledge does not substitute for proper service."
        '<page-number citation-index="1" label="624">*624</page-number>'
    )
    client = OpinionClient(
        opinions={
            "combined": CourtListenerOpinion(
                "combined",
                "2971299",
                "010combined",
                '<span citation-index="1" label="623">*623</span> combined',
            ),
            "dissent": CourtListenerOpinion(
                "dissent",
                "2971299",
                "040dissent",
                '<span citation-index="1" label="623">*623</span> dissent',
            ),
            "lead": CourtListenerOpinion("lead", "2971299", "020lead", lead_html),
        },
    )

    node = run_reporter_page_retrieval(validation, evaluation=evaluation, client=client)

    assert node.status is ValidationNodeStatus.SUCCEEDED
    assert node.outcome is ReporterPageRetrievalOutcome.FOUND
    assert node.citation_index == 1
    assert node.evidence is not None
    assert node.evidence.opinion_id == "lead"
    assert node.evidence.text.startswith("*623ment.")
    assert node.depends_on == (evaluation.node_id,)


def test_reporter_page_retrieval_uses_unique_volume_for_reporter_alias() -> None:
    """A unique volume safely resolves a harmless reporter abbreviation alias."""
    validation, evaluation = _validation(
        reporter="Fed. App'x",
        pin_cite="459",
        cluster_reporter="F. App'x",
        sub_opinion_ids=("opinion",),
    )
    client = OpinionClient(
        opinions={
            "opinion": CourtListenerOpinion(
                "opinion",
                "2971299",
                "010combined",
                '<span citation-index="1" label="459">*459</span> evidence',
            )
        },
    )
    node = run_reporter_page_retrieval(validation, evaluation=evaluation, client=client)

    assert node.outcome is ReporterPageRetrievalOutcome.FOUND
    assert node.evidence is not None
    assert node.evidence.opinion_id == "opinion"


def test_reporter_page_retrieval_prefers_unanimous_defensively() -> None:
    """Dirty coexistence still resolves deterministically to unanimous."""
    validation, evaluation = _validation(sub_opinion_ids=("lead", "unanimous"))
    html = '<span citation-index="1" label="623">*623</span> holding'
    client = OpinionClient(
        opinions={
            "lead": CourtListenerOpinion("lead", "2971299", "020lead", html, 0),
            "unanimous": CourtListenerOpinion(
                "unanimous",
                "2971299",
                "015unamimous",
                html,
                1,
            ),
        }
    )

    node = run_reporter_page_retrieval(validation, evaluation=evaluation, client=client)

    assert node.outcome is ReporterPageRetrievalOutcome.FOUND
    assert node.evidence is not None
    assert node.evidence.opinion_id == "unanimous"


def test_reporter_page_retrieval_excludes_separate_opinions() -> None:
    """Concurrences and dissents are not selected as the base cited opinion."""
    validation, evaluation = _validation(sub_opinion_ids=("concurrence", "dissent"))
    html = '<span citation-index="1" label="623">*623</span> separate writing'
    client = OpinionClient(
        opinions={
            "concurrence": CourtListenerOpinion(
                "concurrence",
                "2971299",
                "030concurrence",
                html,
            ),
            "dissent": CourtListenerOpinion("dissent", "2971299", "040dissent", html),
        }
    )

    node = run_reporter_page_retrieval(validation, evaluation=evaluation, client=client)

    assert node.status is ValidationNodeStatus.SKIPPED
    assert node.outcome is ReporterPageRetrievalOutcome.UNAVAILABLE
    assert node.evidence is None
    assert "no accepted base-opinion type" in (node.outcome_message or "")


def test_reporter_page_retrieval_explains_missing_opinion_html() -> None:
    """An accepted opinion without HTML remains an explicit data limitation."""
    validation, evaluation = _validation(sub_opinion_ids=("combined",))
    client = OpinionClient(
        opinions={
            "combined": CourtListenerOpinion(
                "combined",
                "2971299",
                "010combined",
                "",
            ),
        }
    )

    node = run_reporter_page_retrieval(validation, evaluation=evaluation, client=client)

    assert node.outcome is ReporterPageRetrievalOutcome.UNAVAILABLE
    assert "does not provide citation-aware HTML" in (node.outcome_message or "")


def test_reporter_page_retrieval_explains_missing_reporter_marker() -> None:
    """Usable opinion text without the requested reporter marker is distinct."""
    validation, evaluation = _validation(sub_opinion_ids=("combined",))
    client = OpinionClient(
        opinions={
            "combined": CourtListenerOpinion(
                "combined",
                "2971299",
                "010combined",
                "<pre>Complete opinion text without reporter pagination.</pre>",
            ),
        }
    )

    node = run_reporter_page_retrieval(validation, evaluation=evaluation, client=client)

    assert node.outcome is ReporterPageRetrievalOutcome.UNAVAILABLE
    assert "does not mark reporter page 623" in (node.outcome_message or "")


def test_reporter_page_retrieval_skips_non_numeric_pin_cites() -> None:
    """Westlaw star pages remain explicit unsupported evidence."""
    validation, evaluation = _validation(reporter="WL", pin_cite="*3")
    client = OpinionClient(opinions={})

    node = run_reporter_page_retrieval(validation, evaluation=evaluation, client=client)

    assert node.status is ValidationNodeStatus.SKIPPED
    assert node.outcome is ReporterPageRetrievalOutcome.UNAVAILABLE
    assert "numeric reporter pin cite" in (node.outcome_message or "")
