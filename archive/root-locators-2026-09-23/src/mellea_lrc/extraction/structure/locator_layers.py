"""Project extraction records into the locator and colocation layers.

The layers answer separate questions. A locator is one occurrence of a complete
reporter or docket identifier in the document. Repeated identifiers retain
their own spans regardless of which root they later share. A colocation is a
group of citation-record ids whose locators occupy the same citation site.
Singletons remain locators and do not become one-member colocations.

These are projections over the citation records, not a replacement for them:
the locator layer preserves the exact text span, while the colocation layer
preserves which records were grouped. Keeping both explicit makes it possible
to evaluate locator reading and grouping independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.model.document import Document
from mellea_lrc.model.locators import LocatorLayers
from mellea_lrc.model.preprocessed import PreprocessedDocument

if TYPE_CHECKING:
    from mellea_lrc.extraction.rules import ExtractionRules


def find_locators(
    document: PreprocessedDocument | Document,
    rules: ExtractionRules | None = None,
) -> LocatorLayers:
    """Run root extraction and return its two explicit locator-layer outputs.

    Every complete locator occurrence appears in locators, including repeated
    identifiers. A singleton has no colocation group. Supplying
    no rules selects eyecite defaults; pass stable(rules) for the project's
    docket and grouping rules.
    """
    from mellea_lrc.extraction.eyecite_extractor import grow_roots

    if isinstance(document, PreprocessedDocument):
        preprocessed = document
    else:
        preprocessed = PreprocessedDocument(
            source_metadata=document.source_metadata,
            text=document.text,
            preprocessing_metadata=document.preprocessing_metadata,
        )
    extracted = grow_roots(preprocessed, rules=rules)
    return LocatorLayers(
        locators=extracted.locators,
        colocations=extracted.colocations,
    )
