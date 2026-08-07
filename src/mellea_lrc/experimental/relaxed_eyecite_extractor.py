"""Eyecite extraction with layout-tolerant reporter matching.

eyecite generates every reporter regex with **literal single spaces** joining
volume, reporter, and page::

    (?P<volume>[1-9]\\d*) (?P<reporter>WL),? (?P<page>...)
                        ^                  ^

Any layout damage to those separators -- a space lost to table extraction, a
doubled space from justified text, a PDF page break landing mid-citation --
makes the citation vanish entirely rather than degrade. This module rebuilds
every extractor with those joins relaxed to ``\\s*``, which is reporter-agnostic:
it is not a fix for any particular reporter, but for how the regexes are
generated.

The result is a plain :class:`ExtractedDocument`, identical in shape to the
baseline extractor's output, so this can be swapped in without any downstream
change.

Not wired into the production pipeline.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

import ahocorasick
from eyecite import resolve_citations
from eyecite.models import CitationBase, Resource, TokenExtractor
from eyecite.tokenizers import EXTRACTORS, AhocorasickTokenizer

from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.eyecite_extractor import (
    _assign_citation_ids,
    _build_antecedent_map,
    _get_citations_with_recovered_spans,
    _to_canonical,
)
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing.plain_text import preprocess_plain_text_from_string

if TYPE_CHECKING:
    from eyecite.tokenizers import Tokenizer

    from mellea_lrc.preprocessing.types import PreprocessedDocument

# Reporter groups produced by eyecite's ``_relax_ws`` often end in ``\s*``
# themselves, so that variants like "U. S." still match. The original regex's
# literal trailing space forces such a group to give the space back on
# backtracking; replacing that space with ``\s*`` removes the pressure, and the
# group keeps it -- yielding reporter="U.S. " and a corrupted locator. The
# ``(?<!\s)`` assertion restores the pressure without requiring a space to be
# present, so the group still cannot end on whitespace. This works for the
# alternation-shaped groups too, where lifting the trailing ``\s*`` out by
# string surgery cannot reach every branch.
_JOINS: tuple[tuple[str, str], ...] = (
    (r") (?P<reporter>", r")\s*(?P<reporter>"),
    (r"),? (?P<page>", r")(?<!\s),?\s*(?P<page>"),
)


def _relax(regex: str) -> str:
    for old, new in _JOINS:
        regex = regex.replace(old, new)
    return regex


class _RelaxedTokenizer(AhocorasickTokenizer):
    """Prefiltered tokenizer that honours its own extractor list.

    ``AhocorasickTokenizer`` builds its prefilter from the module-level
    ``EXTRACTORS``, so a tokenizer constructed with replacement extractors
    silently never runs them. Rebuilding the filters from ``self.extractors``
    keeps the prefilter -- without it every one of the ~6,800 extractors runs
    against every document, which is far too slow to be usable.
    """

    def __post_init__(self) -> None:
        self.unfiltered_extractors = {e for e in self.extractors if not e.strings}
        self.case_sensitive_filter = self._filter(case_sensitive=True)
        self.case_insensitive_filter = self._filter(case_sensitive=False)

    def _filter(self, *, case_sensitive: bool) -> ahocorasick.Automaton:
        pairs = [
            (s.replace(" ", "") if case_sensitive else s.replace(" ", "").lower(), e)
            for e in self.extractors
            if e.strings and bool(e.flags & re.I) is not case_sensitive
            for s in e.strings
        ]
        return self.make_ahocorasick_filter(pairs)


def relaxed_tokenizer() -> Tokenizer:
    """Build a tokenizer whose reporter regexes tolerate any separator whitespace."""
    return _RelaxedTokenizer(
        extractors=[
            TokenExtractor(
                regex=_relax(extractor.regex),
                constructor=extractor.constructor,
                extra=extractor.extra,
                flags=extractor.flags,
                strings=extractor.strings,
            )
            for extractor in EXTRACTORS
        ]
    )


_TOKENIZER = relaxed_tokenizer()


def extract_relaxed(preprocessed: PreprocessedDocument) -> ExtractedDocument:
    """Extract canonical citations using layout-tolerant reporter matching.

    Returns an ordinary :class:`ExtractedDocument` whose spans index into
    ``preprocessed.text``.

    The baseline's whitespace-collapse step is kept even though a relaxed
    tokenizer no longer needs it to find doubled-space citations: keeping both
    backends on the same collapse-and-remap path means they stay directly
    comparable, and any normalization added there later applies to both.
    """
    text = preprocessed.text
    eyecite_citations = _get_citations_with_recovered_spans(text, tokenizer=_TOKENIZER)
    resolutions = cast(
        "dict[Resource, list[CitationBase]]",
        resolve_citations(eyecite_citations),
    )
    citation_ids = _assign_citation_ids(eyecite_citations)
    antecedent_map = _build_antecedent_map(resolutions, citation_ids)

    extracted: list[ExtractedCitation] = []
    for eyecite_citation, citation_id in citation_ids:
        span_start, span_end = eyecite_citation.full_span()
        locator_start, locator_end = eyecite_citation.span()
        extracted.append(
            ExtractedCitation(
                citation_id=citation_id,
                span=Span(start=span_start, end=span_end),
                locator_span=Span(start=locator_start, end=locator_end),
                matched_text=eyecite_citation.matched_text(),
                citation=_to_canonical(eyecite_citation),
                resolves_to=antecedent_map.get(citation_id),
            )
        )

    return ExtractedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=tuple(extracted),
        extraction_metadata=ExtractionMetadata(),
    )


def extract_relaxed_citations(text: str, *, source_path: str | None = None) -> ExtractedDocument:
    """Extract citations from raw Layer 2 text using the relaxed tokenizer."""
    return extract_relaxed(preprocess_plain_text_from_string(text, source_path=source_path))
