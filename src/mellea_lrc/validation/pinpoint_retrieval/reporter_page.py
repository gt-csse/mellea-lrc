"""Retrieve a cited reporter page from CourtListener opinion HTML."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.courtlistener.client import CourtListenerError
from mellea_lrc.courtlistener.opinion_models import CourtListenerOpinionCluster
from mellea_lrc.validation.types import (
    CandidateEvaluationNode,
    CandidateEvaluationSource,
    ReporterPageEvidence,
    ReporterPageRetrievalNode,
    ReporterPageRetrievalOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.opinion_models import (
        CourtListenerOpinion,
        CourtListenerOpinionClusterCitation,
    )
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.validation.types import CitationValidation


_NUMERIC_PIN = re.compile(r"(?P<page>\d+)(?:[-\N{EN DASH}\N{EM DASH}]\d+)?")

# MVE scope: select one controlling/base opinion for a uniquely resolved
# cluster. Concurrences, dissents, and procedural opinions are deliberately
# excluded; broader multi-opinion evidence semantics can be added later.
_BASE_OPINION_TYPE_PRIORITY = {
    "015unamimous": 0,  # CourtListener's canonical value contains this typo.
    "020lead": 1,
    "025plurality": 2,
    "010combined": 3,
}


def run_reporter_page_retrieval(
    validation: CitationValidation,
    *,
    evaluation: CandidateEvaluationNode,
    client: CourtListenerServiceClient,
) -> ReporterPageRetrievalNode:
    """Retrieve numeric reporter-pin evidence for one locator-derived candidate."""
    if evaluation.source is not CandidateEvaluationSource.LOCATOR_LOOKUP or not isinstance(
        evaluation.record, CourtListenerOpinionCluster
    ):
        msg = "Reporter-page retrieval requires a locator-derived opinion cluster"
        raise ValueError(msg)
    citation = validation.citation.citation
    cluster = evaluation.record
    cluster_id = cluster.cluster_id
    dependency = (evaluation.node_id,)
    if not isinstance(citation, FullCaseCitation):
        return _unavailable(
            cluster_id=cluster_id,
            pin_cite=None,
            depends_on=dependency,
            message="Reporter-page retrieval only supports full case citations.",
        )
    reporter_citation = _reporter_citation(citation)
    pin_page = _numeric_pin_page(citation.pin_cite)
    if cluster_id is None:
        return _unavailable(
            cluster_id=None,
            reporter_citation=reporter_citation,
            pin_cite=citation.pin_cite,
            depends_on=dependency,
            message="The opinion candidate has no CourtListener cluster identifier.",
        )
    if reporter_citation is None:
        return _unavailable(
            cluster_id=cluster_id,
            pin_cite=citation.pin_cite,
            depends_on=dependency,
            message="The citation has no complete reporter locator.",
        )
    if pin_page is None:
        return _unavailable(
            cluster_id=cluster_id,
            reporter_citation=reporter_citation,
            pin_cite=citation.pin_cite,
            depends_on=dependency,
            message="The citation has no supported numeric reporter pin cite.",
        )

    try:
        citation_index = _citation_index(
            cluster.citations,
            volume=citation.volume,
            reporter=citation.reporter,
        )
        if citation_index is None:
            return _unavailable(
                cluster_id=cluster_id,
                reporter_citation=reporter_citation,
                pin_cite=citation.pin_cite,
                depends_on=dependency,
                message="The candidate cluster does not identify the cited reporter edition.",
            )
        found: list[tuple[int, int, int, ReporterPageEvidence]] = []
        has_eligible_opinion = False
        has_eligible_html = False
        for position, opinion_id in enumerate(cluster.sub_opinion_ids):
            opinion = client.get_opinion(opinion_id)
            type_priority = _BASE_OPINION_TYPE_PRIORITY.get(opinion.opinion_type)
            if type_priority is None:
                continue
            has_eligible_opinion = True
            if not opinion.html_with_citations.strip():
                continue
            has_eligible_html = True
            text = extract_reporter_page(
                opinion.html_with_citations,
                citation_index=citation_index,
                page_label=pin_page,
            )
            if text is None:
                continue
            ordering_key = opinion.ordering_key if opinion.ordering_key is not None else position
            found.append(
                (
                    type_priority,
                    ordering_key,
                    position,
                    _page_evidence(opinion, text),
                )
            )
    except CourtListenerError as exc:
        return ReporterPageRetrievalNode(
            node_id=f"{evaluation.node_id}:reporter_page_retrieval",
            status=ValidationNodeStatus.FAILED,
            outcome=ReporterPageRetrievalOutcome.FAILED,
            cluster_id=cluster_id,
            reporter_citation=reporter_citation,
            pin_cite=citation.pin_cite,
            citation_index=None,
            evidence=None,
            depends_on=dependency,
            status_message="Reporter-page retrieval failed.",
            outcome_message="CourtListener did not provide usable opinion evidence.",
            error=str(exc),
        )

    if not found:
        if not has_eligible_opinion:
            message = "The cluster contains no accepted base-opinion type for reporter-page retrieval."
        elif not has_eligible_html:
            message = (
                "The accepted base opinion does not provide citation-aware HTML for reporter-page retrieval."
            )
        else:
            message = (
                f"The accepted base-opinion HTML does not mark reporter page {pin_page} "
                "for the cited reporter edition."
            )
        return _unavailable(
            cluster_id=cluster_id,
            reporter_citation=reporter_citation,
            pin_cite=citation.pin_cite,
            citation_index=citation_index,
            depends_on=dependency,
            message=message,
        )
    evidence = min(found, key=lambda candidate: candidate[:3])[3]
    return ReporterPageRetrievalNode(
        node_id=f"{evaluation.node_id}:reporter_page_retrieval",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=ReporterPageRetrievalOutcome.FOUND,
        cluster_id=cluster_id,
        reporter_citation=reporter_citation,
        pin_cite=citation.pin_cite,
        citation_index=citation_index,
        evidence=evidence,
        depends_on=dependency,
        status_message="Reporter-page retrieval completed.",
        outcome_message=(
            f"Recovered reporter page {pin_page} from selected "
            f"{evidence.opinion_type} opinion {evidence.opinion_id}."
        ),
    )


def extract_reporter_page(
    html: str,
    *,
    citation_index: int,
    page_label: str,
) -> str | None:
    """Return text from the target reporter marker to its next page marker."""
    parser = _ReporterPageParser()
    parser.feed(html)
    parser.close()
    return parser.page(citation_index=str(citation_index), page_label=page_label)


class _ReporterPageParser(HTMLParser):
    """Collect plain text and citation-indexed reporter marker offsets."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._text: list[str] = []
        self._length = 0
        self._markers: list[tuple[str, str, int]] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        citation_index = attributes.get("citation-index")
        label = attributes.get("label")
        if citation_index is not None and label is not None:
            self._markers.append((citation_index, label, self._length))

    def handle_data(self, data: str) -> None:
        self._text.append(data)
        self._length += len(data)

    def page(self, *, citation_index: str, page_label: str) -> str | None:
        text = "".join(self._text)
        for marker_index, (index, label, start) in enumerate(self._markers):
            if index != citation_index or label != page_label:
                continue
            end = next(
                (
                    offset
                    for next_index, _next_label, offset in self._markers[marker_index + 1 :]
                    if next_index == citation_index
                ),
                len(text),
            )
            return _normalize_page_text(text[start:end])
        return None


def _citation_index(
    citations: tuple[CourtListenerOpinionClusterCitation, ...],
    *,
    volume: str | None,
    reporter: str | None,
) -> int | None:
    if volume is None or reporter is None:
        return None
    exact = [
        index
        for index, citation in enumerate(citations, start=1)
        if citation.volume == volume
        and _normalized_reporter(citation.reporter) == _normalized_reporter(reporter)
    ]
    if len(exact) == 1:
        return exact[0]
    same_volume = [index for index, citation in enumerate(citations, start=1) if citation.volume == volume]
    return same_volume[0] if len(same_volume) == 1 else None


def _numeric_pin_page(pin_cite: str | None) -> str | None:
    if pin_cite is None:
        return None
    match = _NUMERIC_PIN.fullmatch(pin_cite.strip())
    return match.group("page") if match else None


def _reporter_citation(citation: FullCaseCitation) -> str | None:
    if citation.volume is None or citation.reporter is None or citation.page is None:
        return None
    return f"{citation.volume} {citation.reporter} {citation.page}"


def _normalized_reporter(reporter: str) -> str:
    return "".join(character.lower() for character in reporter if character.isalnum())


def _normalize_page_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _page_evidence(opinion: CourtListenerOpinion, text: str) -> ReporterPageEvidence:
    return ReporterPageEvidence(
        opinion_id=opinion.opinion_id,
        opinion_type=opinion.opinion_type,
        text=text,
    )


def _unavailable(
    *,
    cluster_id: str | None,
    pin_cite: str | None,
    depends_on: tuple[str, ...],
    message: str,
    reporter_citation: str | None = None,
    citation_index: int | None = None,
) -> ReporterPageRetrievalNode:
    return ReporterPageRetrievalNode(
        node_id=f"{depends_on[0]}:reporter_page_retrieval",
        status=ValidationNodeStatus.SKIPPED,
        outcome=ReporterPageRetrievalOutcome.UNAVAILABLE,
        cluster_id=cluster_id,
        reporter_citation=reporter_citation,
        pin_cite=pin_cite,
        citation_index=citation_index,
        evidence=None,
        depends_on=depends_on,
        status_message="Reporter-page retrieval was unavailable.",
        outcome_message=message,
    )
