"""One readable composition of the independent first-pass extraction stages."""

from __future__ import annotations

from pathlib import Path

from mellea_lrc.extraction.context import (
    resolve_case_names,
    resolve_courts,
    resolve_dates,
    resolve_pin_cites,
)
from mellea_lrc.extraction.locators import find_docket_locators, find_full_reporter_locators
from mellea_lrc.extraction.roots import form_roots
from mellea_lrc.extraction.rules import ExtractionRules, stable
from mellea_lrc.extraction.structure import resolve_colocations
from mellea_lrc.model.extraction import Document
from mellea_lrc.model.preprocessed import PreprocessedDocument


def start_extraction(source: Document | PreprocessedDocument) -> Document:
    """Keep an existing checkpoint or wrap a preprocessed document."""
    return source if isinstance(source, Document) else Document.from_preprocessed(source)


async def grow_roots(
    document: Document,
    *,
    rules: ExtractionRules | None = None,
    hunt_dockets: bool = False,
) -> Document:
    """Run first-pass extraction through root formation.

    The optional model-backed hunting stage has not been rebuilt yet. It must
    run before colocation and field reading so later model reviews cannot be
    overwritten by a fresh deterministic parse.
    """
    config = rules or stable()
    document = find_full_reporter_locators(document, config)
    document = find_docket_locators(document, config)
    if hunt_dockets:
        raise NotImplementedError("Docket site hunting has not been rebuilt in the active package")
    document = resolve_colocations(document, config)
    document = resolve_case_names(document, config)
    document = resolve_courts(document, config)
    document = resolve_dates(document, config)
    document = resolve_pin_cites(document, config)
    return form_roots(document)


async def extract(
    source: Path | str | PreprocessedDocument | Document,
    *,
    rules: ExtractionRules | None = None,
) -> Document:
    """Preprocess if needed, then compose the extraction stages."""
    document = (
        start_extraction(source) if isinstance(source, PreprocessedDocument) else Document.from_source(source)
    )
    return await grow_roots(document, rules=rules)
