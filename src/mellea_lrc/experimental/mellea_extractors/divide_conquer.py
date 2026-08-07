"""Divide-and-conquer Mellea extraction strategy."""

from __future__ import annotations

import difflib
import logging
import re
from typing import TYPE_CHECKING, ClassVar

from mellea import generative

from mellea_lrc.core import Span
from mellea_lrc.experimental.mellea_extractors.base import (
    DEFAULT_MODEL_ID,
    LOCATOR,
    TRAILING_YEAR,
    FoundCitation,
    MelleaExtractorBase,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mellea.backends.model_ids import ModelIdentifier

_EXPAND_SLACK = 16
_LocalMatch = tuple[tuple[int, int], str]  # ((start, end) in the chunk, located text)
_LOGGER = logging.getLogger(__name__)


@generative
def _passage_has_full_case_citation(passage: str) -> bool:
    """Whether the passage contains at least one full case-law citation.

    A full case-law citation names the parties and a reporter locator, for
    example "Brown v. Board of Education, 347 U.S. 483". Return False when the
    passage contains none; do not guess.
    """
    ...  # noqa: PIE790


@generative
def _verbatim_case_citations(passage: str) -> list[str]:
    """Return each full case-law citation in the passage as one complete string.

    Each returned string must contain BOTH the party names AND the reporter
    locator together, exactly as they appear -- for example the single string
    "Hirabayashi v. United States, 320 U.S. 81". Never split one citation into
    separate party-name and locator entries. Copy each citation character for
    character from the passage; do not add, remove, correct, or reformat anything
    (no added years, no changed punctuation). Return an empty list when there are
    none.
    """
    ...  # noqa: PIE790


@generative
def _citation_matches_excerpt(model_citation: str, source_excerpt: str) -> bool:
    """Whether source_excerpt is the citation the model meant by model_citation.

    Return True when both refer to the same case citation and source_excerpt is a
    faithful copy of it from the source -- allowing the model's version to differ
    only by an added year or reformatted spacing. Return False otherwise.
    """
    ...  # noqa: PIE790


class MelleaDivideConquer(MelleaExtractorBase):
    """Sliding-window strategy with per-window gating, recovery, and confirmation."""

    _MODEL_OPTIONS: ClassVar[dict[str, float]] = {"temperature": 0.0}
    _MAX_CANDIDATES_PER_WINDOW = 64

    def __init__(
        self,
        model_id: ModelIdentifier = DEFAULT_MODEL_ID,
        *,
        window_size: int = 1500,
        overlap: int = 300,
        confirm: bool = True,
    ) -> None:
        """Configure the sliding window and whether to run the confirmation step."""
        super().__init__(model_id)
        if window_size <= 0:
            msg = "window_size must be positive"
            raise ValueError(msg)
        if not 0 <= overlap < window_size:
            msg = "overlap must be in [0, window_size)"
            raise ValueError(msg)
        self._window_size = window_size
        self._overlap = overlap
        self._confirm = confirm

    def _find_citations(self, text: str) -> list[FoundCitation]:
        """Slide over `text`, gate, extract, locate, confirm, then resolve overlaps."""
        candidates: list[FoundCitation] = []
        for offset, chunk in self._windows(text):
            try:
                if not self._chunk_has_citation(chunk):
                    continue
                verbatims = self._extract_verbatim(chunk)
            except Exception as error:
                _LOGGER.warning("skipping window at offset %d: %s", offset, error)
                continue
            seen: set[str] = set()
            for raw in verbatims[: self._MAX_CANDIDATES_PER_WINDOW]:
                candidate = raw.strip()
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                local = self._locate_in_chunk(chunk, candidate)
                if local is None:
                    continue
                (local_start, local_end), located = local
                try:
                    if self._confirm and not self._confirm_citation(candidate, located):
                        continue
                except Exception as error:
                    _LOGGER.warning("confirmation failed at offset %d: %s", offset, error)
                    continue
                candidates.append(
                    FoundCitation(
                        matched_text=located,
                        span=Span(offset + local_start, offset + local_end),
                    )
                )
        return self._resolve_overlaps(candidates)

    def _resolve_overlaps(self, candidates: list[FoundCitation]) -> list[FoundCitation]:
        """Keep the best of overlapping candidates by preferring the longer span."""
        ordered = sorted(candidates, key=lambda f: (-(f.span.end - f.span.start), f.span.start))
        accepted: list[FoundCitation] = []
        accepted_spans: list[tuple[int, int]] = []
        for found in ordered:
            if self._overlaps(found.span.start, found.span.end, accepted_spans):
                continue
            accepted.append(found)
            accepted_spans.append((found.span.start, found.span.end))
        accepted.sort(key=lambda found: found.span.start)
        return accepted

    def _windows(self, text: str) -> Iterator[tuple[int, str]]:
        """Yield ``(offset, chunk)`` sliding windows with `self._overlap` overlap."""
        length = len(text)
        if length == 0:
            return
        step = max(1, self._window_size - self._overlap)
        start = 0
        while start < length:
            yield start, text[start : start + self._window_size]
            if start + self._window_size >= length:
                break
            start += step

    def _locate_in_chunk(self, chunk: str, candidate: str) -> _LocalMatch | None:
        """Find `candidate` within `chunk`, tolerating the model's reformatting."""
        span = self._locate_span(chunk, candidate)
        if span:
            return (span.start, span.end), chunk[span.start : span.end]

        trimmed = TRAILING_YEAR.sub("", candidate).strip(" \"'")
        if trimmed and trimmed != candidate:
            span = self._locate_span(chunk, trimmed)
            if span:
                return (span.start, span.end), chunk[span.start : span.end]

        core = re.sub(r"\s+", " ", trimmed or candidate).strip()
        if core:
            pattern = re.escape(core).replace(r"\ ", r"\s+")
            match = re.search(pattern, chunk)
            if match:
                return (match.start(), match.end()), chunk[match.start() : match.end()]

        return self._locator_anchored(chunk, candidate)

    def _locator_anchored(self, chunk: str, candidate: str) -> _LocalMatch | None:
        """Anchor on the reporter locator, then expand left over the case name."""
        locator = LOCATOR.search(candidate)
        if locator is None:
            return None
        locator_pattern = re.escape(locator.group(0)).replace(r"\ ", r"\s+")
        anchor = re.search(locator_pattern, chunk)
        if anchor is None:
            return None
        locator_start, locator_end = anchor.start(), anchor.end()

        prefix = candidate[: locator.start()].strip().strip(",").strip()
        if not prefix:
            return (locator_start, locator_end), chunk[locator_start:locator_end]

        left_bound = max(0, locator_start - len(prefix) - _EXPAND_SLACK)
        region = chunk[left_bound:locator_start]
        block = difflib.SequenceMatcher(None, region, prefix).find_longest_match(
            0, len(region), 0, len(prefix)
        )
        substantial = block.size >= max(4, len(prefix) // 2)
        adjacent = block.a + block.size >= len(region) - 2
        start = left_bound + block.a if (substantial and adjacent) else locator_start
        while start < locator_start and not chunk[start].isalnum():
            start += 1
        return (start, locator_end), chunk[start:locator_end]

    @staticmethod
    def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
        """Whether ``[start, end)`` overlaps any already-accepted span."""
        return any(not (end <= s or start >= e) for s, e in spans)

    def _chunk_has_citation(self, chunk: str) -> bool:
        """Ask the model whether `chunk` contains any full case citation."""
        return bool(
            _passage_has_full_case_citation(self.session, passage=chunk, model_options=self._MODEL_OPTIONS)
        )

    def _extract_verbatim(self, chunk: str) -> list[str]:
        """Ask the model for every case citation in `chunk`, copied verbatim."""
        return list(_verbatim_case_citations(self.session, passage=chunk, model_options=self._MODEL_OPTIONS))

    def _confirm_citation(self, model_citation: str, source_excerpt: str) -> bool:
        """Ask the model whether `source_excerpt` is the citation it meant."""
        return bool(
            _citation_matches_excerpt(
                self.session,
                model_citation=model_citation,
                source_excerpt=source_excerpt,
                model_options=self._MODEL_OPTIONS,
            )
        )
