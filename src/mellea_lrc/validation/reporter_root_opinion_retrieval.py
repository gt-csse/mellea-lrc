"""Retrieve opinion text for admitted reporter roots with a selected cluster."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Protocol

from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.citations.reporter_lookup import ReporterExactLookupOutcome
from mellea_lrc.model.citations.reporter_opinion import (
    OpinionRetrievalOutcome,
    ReporterRootOpinionRetrieval,
    ReporterRootOpinionSource,
    RetrievedReporterOpinion,
)
from mellea_lrc.model.document import Document
from mellea_lrc.providers.courtlistener import CourtListenerClient
from mellea_lrc.providers.courtlistener.models import CourtListenerCluster

SUBSTAGE = "validate_pincite.opinion_preparation.retrieval"


class OpinionClient(Protocol):
    def get_opinion(self, opinion_id: str) -> dict[str, Any] | None: ...


def _selected_candidate(citation: FullReporterCitation) -> int | None:
    lookup = citation.reporter_exact_lookup
    if lookup is None or lookup.response is None:
        return None
    if lookup.outcome is ReporterExactLookupOutcome.UNIQUE:
        return 0
    review = citation.reporter_ambiguous_review
    if review is not None and review.decision is not None:
        return review.decision.selected_candidate_index
    resolution = citation.reporter_exact_ambiguity_resolution
    return resolution.selected_candidate_index if resolution is not None else None


def selected_reporter_root_cluster(citation: FullReporterCitation) -> tuple[int, CourtListenerCluster] | None:
    """Adapt the identity workflow's selected original cluster for default retrieval."""
    if (
        not citation.identity_judgments
        or citation.identity_judgments[-1].verdict is not IdentityVerdict.CORRECT_IDENTITY
    ):
        return None
    index = _selected_candidate(citation)
    if index is None:
        return None
    cluster = citation.reporter_exact_lookup.response.clusters[index]
    if cluster.id is None:
        raise ValueError("Selected reporter cluster has no ID")
    return index, cluster


def reporter_root_opinion_retrieval(document: Document, *, client: OpinionClient | None = None) -> Document:
    """Fetch every subopinion without altering identities, fields, or attachments.

    Correct identity is the default eligibility policy, independent of the
    source binding. A supplied binding takes precedence; otherwise the current
    identity workflow supplies the original cluster through its lookup adapter.
    A third-party admission without a selected original-case cluster does not
    qualify: the corroborating document is not the cited opinion. No pinpoint is
    required on the root; its leaves may cite distinct pages or distinct writings.
    Provider errors abort the atomic substage. A real 404 or empty text is retained
    explicitly, so neither can masquerade as successful retrieval.
    """
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if "grow_roots.root_formation.rule" not in document.substage_runs:
        raise ValueError("Form and validate reporter roots before opinion retrieval")
    cache: dict[str, dict[str, Any] | None] = {}
    with ExitStack() as stack:
        service = client
        for citation in document.roots:
            if not isinstance(citation, FullReporterCitation):
                continue
            if (
                not citation.identity_judgments
                or citation.identity_judgments[-1].verdict is not IdentityVerdict.CORRECT_IDENTITY
            ):
                continue
            recorded = citation.record(SUBSTAGE)
            source = citation.reporter_root_opinion_source
            if source is None:
                selected = selected_reporter_root_cluster(citation)
                if selected is None:
                    continue
                _, cluster = selected
                source = ReporterRootOpinionSource(node_id=recorded.nodes[-1].id, cluster=cluster)
                recorded = recorded.with_reporter_root_opinion_source(source)
            # Resolve upstream resource links to IDs, then use the configured
            # proxy. Following their canonical URLs would bypass that proxy.
            ids = source.sub_opinion_ids
            opinions: list[RetrievedReporterOpinion] = []
            for opinion_id in dict.fromkeys(ids):
                if opinion_id not in cache:
                    if service is None:
                        service = stack.enter_context(CourtListenerClient())
                    cache[opinion_id] = service.get_opinion(opinion_id)
                response = cache[opinion_id]
                # Determine availability from the actual payload, preserving it
                # unchanged. Validation also checks ID and cluster association.
                outcome = OpinionRetrievalOutcome.NOT_FOUND
                if response is not None:
                    has_text = any(
                        isinstance(response.get(key), str) and response[key].strip()
                        for key in (
                            "html_with_citations",
                            "html",
                            "html_lawbox",
                            "html_columbia",
                            "html_anon_2020",
                            "xml_harvard",
                            "plain_text",
                        )
                    )
                    outcome = (
                        OpinionRetrievalOutcome.RETRIEVED if has_text else OpinionRetrievalOutcome.EMPTY_TEXT
                    )
                opinions.append(
                    RetrievedReporterOpinion(
                        opinion_id=opinion_id,
                        cluster_id=source.cluster_id,
                        outcome=outcome,
                        response=response,
                    )
                )
            result = ReporterRootOpinionRetrieval(
                node_id=recorded.nodes[-1].id,
                cluster_id=source.cluster_id,
                sub_opinion_ids=ids,
                opinions=tuple(opinions),
            )
            document = document.replace_citation(recorded.with_reporter_root_opinion_retrieval(result))
    return document.complete_substage(SUBSTAGE)
