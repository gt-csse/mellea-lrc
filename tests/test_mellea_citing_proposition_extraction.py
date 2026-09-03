"""Tests for grounded citing-proposition extraction."""

import asyncio
import json
from types import SimpleNamespace

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import ExtractedCitation
from mellea_lrc.validation.pinpoint_retrieval.mellea_citing_proposition_extraction import (
    CONTEXT_WINDOW_CHARS,
    citing_context_span,
    run_mellea_citing_proposition_extraction,
)
from mellea_lrc.validation.types import (
    CitationValidation,
    EvidenceQuoteMatchMethod,
    MelleaCitingPropositionExtractionOutcome,
    ReporterPageEvidence,
    ReporterPageRetrievalNode,
    ReporterPageRetrievalOutcome,
    ValidationNodeStatus,
)


def test_citing_context_window_is_bounded_and_biased_before_citation() -> None:
    """The window preserves 900 preceding characters inside a 1,200-char cap."""
    document = "a" * 2_000
    citation_span = Span(1_000, 1_100)

    context_span = citing_context_span(document, citation_span)

    assert context_span == Span(100, 1_300)
    assert context_span.end - context_span.start == CONTEXT_WINDOW_CHARS
    assert context_span.start <= citation_span.start
    assert context_span.end >= citation_span.end


def test_citing_context_window_shifts_at_document_boundaries() -> None:
    """Available capacity is reused rather than returning undersized edge windows."""
    document = "a" * 2_000

    assert citing_context_span(document, Span(50, 100)) == Span(0, 1_200)
    assert citing_context_span(document, Span(1_950, 2_000)) == Span(800, 2_000)


def test_citing_proposition_node_stores_original_document_span(monkeypatch: object) -> None:
    """The model proposes text while deterministic code owns source offsets."""
    prefix = "Background. " * 100
    proposition = "The trial court has broad discretion over evidentiary rulings."
    citation_text = "United States v. Golden, 671 F.2d 369, 371 (10th Cir. 1982)"
    document = f"{prefix}{proposition} See {citation_text}."
    citation_start = document.index(citation_text)
    validation = CitationValidation(
        citation=ExtractedCitation(
            citation_id="citation-1",
            full_span=Span(citation_start, citation_start + len(citation_text)),
            locator_span=Span(
                citation_start + len("United States v. Golden, "),
                citation_start + len("United States v. Golden, 671 F.2d 369"),
            ),
            matched_text="671 F.2d 369",
            citation=FullCaseCitation(
                volume="671",
                reporter="F.2d",
                page="369",
                pin_cite="371",
            ),
        )
    )
    retrieval = ReporterPageRetrievalNode(
        node_id="citation-1:reporter_page_retrieval",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=ReporterPageRetrievalOutcome.FOUND,
        cluster_id="1",
        reporter_citation="671 F.2d 369",
        pin_cite="371",
        citation_index=1,
        evidence=ReporterPageEvidence("2", "010combined", "*371 Opinion text."),
        depends_on=("citation-1:opinion_candidate_evaluation:1",),
    )

    async def fake_run(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(
            success=True,
            result=SimpleNamespace(
                value=json.dumps(
                    {
                        "classification": "identified",
                        "reasoning": ("The sentence immediately before the citation states the proposition."),
                        "proposition_quote": proposition,
                    }
                )
            ),
        )

    config = SimpleNamespace(mellea_call_options=lambda **_kwargs: {})
    monkeypatch.setattr(
        ("mellea_lrc.validation.pinpoint_retrieval.mellea_citing_proposition_extraction.run_instruct_ivr"),
        fake_run,
    )
    monkeypatch.setattr(
        (
            "mellea_lrc.validation.pinpoint_retrieval."
            "mellea_citing_proposition_extraction.llm_api_config_from_env"
        ),
        lambda _environ: config,
    )

    node = asyncio.run(
        run_mellea_citing_proposition_extraction(
            validation,
            trigger=retrieval,
            document_text=document,
            session=object(),  # type: ignore[arg-type]
        )
    )

    assert node.status is ValidationNodeStatus.SUCCEEDED
    assert node.outcome is MelleaCitingPropositionExtractionOutcome.IDENTIFIED
    assert node.proposition == proposition
    assert node.proposition_match_method is EvidenceQuoteMatchMethod.EXACT
    assert node.proposition_span is not None
    assert document[node.proposition_span.start : node.proposition_span.end] == proposition
    assert node.context_span.end - node.context_span.start == CONTEXT_WINDOW_CHARS
    assert node.depends_on == (retrieval.node_id,)
