"""Adapters from existing document artifacts into citation-node documents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.citation_nodes.types import (
    CitationNode,
    CitationNodeDocument,
    CitationNodeInput,
)

if TYPE_CHECKING:
    from mellea_lrc.extraction.types import ExtractedDocument


def nodes_from_extracted_document(document: ExtractedDocument) -> CitationNodeDocument:
    """Create independent citation nodes from an extracted document.

    The adapter copies citation inputs into node inputs and leaves the source
    ``ExtractedDocument`` unchanged. Downstream workflows should attach evolving
    reasoning to the returned nodes rather than mutating extraction artifacts.
    """
    return CitationNodeDocument(
        text=document.text,
        nodes=tuple(
            CitationNode(
                input=CitationNodeInput(
                    citation_id=citation.citation_id,
                    citation_span=citation.citation_span,
                    # eyecite exposes ``matched_text`` as the locator string;
                    # keep that meaning explicit once we cross into nodes.
                    matched_locator_text=citation.matched_locator_text,
                    matched_citation_text=document.text[
                        citation.citation_span.start : citation.citation_span.end
                    ],
                    citation=citation.citation,
                    resolves_to=citation.resolves_to,
                )
            )
            for citation in document.citations
        ),
    )
