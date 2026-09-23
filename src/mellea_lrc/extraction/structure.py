"""Group adjacent locator occurrences without asserting they share identity."""

from __future__ import annotations

import re

from mellea_lrc.extraction.rules import ExtractionRules, stable
from mellea_lrc.model.extraction import Citation, CitationField, Colocation, Document

_SEPARATE_CITATION = re.compile(r"…|\.{2,}|\bvs?\.|\n\s*\n|\.\s+[A-Z]", re.I)


def _adjacent(text: str, left: Citation, right: Citation, maximum_gap: int) -> bool:
    assert left.locator_span is not None and right.locator_span is not None
    between = text[left.locator_span.end : right.locator_span.start]
    return (
        not _SEPARATE_CITATION.search(between)
        and sum(character.isalnum() for character in between) <= maximum_gap
    )


def resolve_colocations(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Assign groups after all locator discovery, before any context read.

    A group is a parsing boundary and candidate parallel-citation site. It is
    never itself a finding that its identifiers refer to the same case.
    """
    stage = "colocations"
    if stage in document.completed_stages:
        return document
    if not {"full_reporter_locators", "docket_locators"} & set(document.completed_stages):
        raise ValueError("Discover at least one kind of full locator before resolving colocations")
    config = rules or stable()
    groups: list[list[Citation]] = []
    for citation in document.full_locators:
        if not groups:
            groups.append([citation])
            continue
        previous = groups[-1][-1]
        reporters = {member.reporter for member in groups[-1] if member.reporter is not None}
        duplicate_reporter = citation.reporter is not None and citation.reporter in reporters
        if not duplicate_reporter and _adjacent(
            document.text, previous, citation, config.colocation_max_meaningful_gap
        ):
            groups[-1].append(citation)
        else:
            groups.append([citation])
    colocations: list[Colocation] = []
    for group in groups:
        if len(group) < 2:
            continue
        identifier = f"colocation:{group[0].id}"
        colocations.append(Colocation(id=identifier, citation_ids=tuple(item.id for item in group)))
        for item in group:
            document = document.update_fields(stage, item.id, {CitationField.COLOCATION_ID: identifier})
    return document.complete(stage, colocations=tuple(colocations))
