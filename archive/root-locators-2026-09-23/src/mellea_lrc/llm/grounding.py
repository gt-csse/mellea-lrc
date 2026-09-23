"""Reusable deterministic grounding for values proposed by a language model.

A model's text never enters a record directly.  It must resolve to one of the
source candidates supplied by the caller, whose value normally carries the
source span or canonical text.  Matching starts exact and may add whitespace
and edit-distance fallbacks through an explicit policy.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import floor
from typing import Generic, TypeVar

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from mellea_lrc.model.fuzziness import FuzzinessOption, FuzzinessType, fuzzy_literal

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class EvidenceCandidate(Generic[T]):
    """One canonical source value that a model proposal may resolve to."""

    text: str
    value: T


@dataclass(frozen=True, slots=True)
class GroundingMatch(Generic[T]):
    """A proposal resolved to one source candidate by one explicit stage."""

    candidate: EvidenceCandidate[T]
    match_type: FuzzinessType
    edits: int
    similarity_percent: float


@dataclass(frozen=True, slots=True)
class GroundedFragment(Generic[T]):
    """A quoted fragment located in one source candidate for highlighting."""

    candidate: EvidenceCandidate[T]
    text: str
    start: int
    end: int
    match_type: FuzzinessType
    edits: int
    similarity_percent: float
    # The comparison view omits verified margin line numbers. ``text`` and
    # start/end always retain the original source bytes for highlighting.
    normalized_text: str


@dataclass(frozen=True, slots=True)
class GroundingEvidence(Generic[T]):
    """A stable, ordered collection of source candidates for one grounding decision."""

    candidates: tuple[EvidenceCandidate[T], ...]

    def __init__(self, candidates: Iterable[EvidenceCandidate[T]]) -> None:
        object.__setattr__(self, "candidates", tuple(candidates))

    def fuzzy_match(
        self,
        proposed: str,
        fuzziness: FuzzinessOption,
    ) -> tuple[GroundingMatch[T], ...]:
        """Return every source candidate that satisfies the selected policy."""
        return fuzzy_match(proposed, self.candidates, fuzziness)

    def resolve_by_condition(
        self,
        condition: Callable[[EvidenceCandidate[T]], bool],
    ) -> EvidenceCandidate[T] | None:
        """Return the first source candidate satisfying a caller-owned condition."""
        return next((candidate for candidate in self.candidates if condition(candidate)), None)

    def resolve(self, proposed: str, fuzziness: FuzzinessOption) -> GroundingMatch[T] | None:
        """Return the first source-grounded match in the evidence's stable order."""
        return next(iter(self.fuzzy_match(proposed, fuzziness)), None)

    def find_fragment(
        self,
        proposed: str,
        fuzziness: FuzzinessOption,
        *,
        line_number_aware: bool = False,
    ) -> GroundedFragment[T] | None:
        """Ground a copied quote to a span inside one source candidate.

        Source candidates can be full pages or bounded excerpts. The returned
        span is relative to the candidate text and points at its actual bytes,
        including any whitespace or OCR variation in the source.
        """
        for candidate in self.candidates:
            found = _find_fragment_in_text(proposed, candidate.text, fuzziness)
            normalized_text = found[1] if found is not None else None
            if found is None and line_number_aware:
                view, original_offsets = _without_margin_line_numbers(candidate.text)
                if original_offsets:
                    aligned = _find_fragment_in_text(proposed, view, fuzziness)
                    if aligned is not None:
                        view_start, view_fragment, match = aligned
                        start = original_offsets[view_start]
                        end = original_offsets[view_start + len(view_fragment) - 1] + 1
                        found = (start, candidate.text[start:end], match)
                        normalized_text = view_fragment
            if found is None:
                continue
            start, fragment, match = found
            return GroundedFragment(
                candidate=candidate,
                text=fragment,
                start=start,
                end=start + len(fragment),
                match_type=match.match_type,
                edits=match.edits,
                similarity_percent=match.similarity_percent,
                normalized_text=normalized_text or fragment,
            )
        return None


def _find_fragment_in_text(
    proposed: str, text: str, fuzziness: FuzzinessOption
) -> tuple[int, str, GroundingMatch[None]] | None:
    fragment = fuzzy_find(proposed, text, fuzziness)
    if fragment is None:
        return None
    match = _match(proposed, EvidenceCandidate(fragment, None), fuzziness)
    if match is None:
        return None
    start = text.find(fragment)
    if start < 0:
        return None
    if match.match_type is FuzzinessType.EDIT_DISTANCE:
        refined = _best_fragment_near(proposed, text, start, fuzziness)
        if refined is not None:
            return refined
    return start, fragment, match


_MARGIN_LINE_PREFIX = re.compile(r"^( {10,})([1-9]\d?)( {2,})(?=\S)")
_MIN_MARGIN_LINE_RUN = 5


def _without_margin_line_numbers(source: str) -> tuple[str, tuple[int, ...]]:
    """Make a grounding-only view and map every retained character to source.

    A margin column must have at least five consecutively numbered nonblank
    lines with an aligned number column. Isolated numbers and digits inside citations stay
    untouched. The source itself is never rewritten, so returned spans still
    point into the exact original text.
    """
    lines = source.splitlines(keepends=True)
    numbered: list[tuple[int, int, int, int]] = []
    for index, line in enumerate(lines):
        match = _MARGIN_LINE_PREFIX.match(line)
        if match is not None:
            numbered.append((index, int(match.group(2)), match.start(3), match.end()))

    stripped: dict[int, int] = {}
    run: list[tuple[int, int, int, int]] = []

    def accept_run() -> None:
        if len(run) >= _MIN_MARGIN_LINE_RUN:
            stripped.update((index, prefix_end) for index, _, _, prefix_end in run)

    for entry in numbered:
        index, number, number_column, _ = entry
        previous = run[-1] if run else None
        continues = (
            previous is not None
            and number_column == previous[2]
            and number == previous[1] + 1
            and all(not lines[between].strip() for between in range(previous[0] + 1, index))
        )
        if not continues:
            accept_run()
            run = []
        run.append(entry)
    accept_run()

    if not stripped:
        return source, ()
    view: list[str] = []
    offsets: list[int] = []
    source_start = 0
    for index, line in enumerate(lines):
        for local_offset in range(stripped.get(index, 0), len(line)):
            view.append(line[local_offset])
            offsets.append(source_start + local_offset)
        source_start += len(line)
    return "".join(view), tuple(offsets)


def _best_fragment_near(
    proposed: str,
    source: str,
    approximate_start: int,
    fuzziness: FuzzinessOption,
) -> tuple[int, str, GroundingMatch[None]] | None:
    """Prefer the lowest-edit span near a fuzzy alignment's arbitrary tie.

    Partial alignment can shift one character left and clip the last digit of
    a citation while keeping the same similarity score. Testing nearby source
    spans ensures the highlight points at the complete printed citation.
    """
    allowance = _maximum_edits(proposed, proposed, fuzziness)
    margin = allowance + 2
    best: tuple[tuple[int, int, float, int, int], int, str, GroundingMatch[None]] | None = None
    for start in range(max(0, approximate_start - margin), min(len(source), approximate_start + margin) + 1):
        for length in range(max(1, len(proposed) - margin), len(proposed) + margin + 1):
            fragment = source[start : start + length]
            if len(fragment) != length:
                continue
            match = _match(proposed, EvidenceCandidate(fragment, None), fuzziness)
            if match is None:
                continue
            end = start + length
            clipped_token = int(end < len(source) and fragment[-1].isalnum() and source[end].isalnum())
            boundary_penalty = int(fragment[0].isspace()) + int(fragment[-1].isspace()) + clipped_token
            key = (
                match.edits,
                boundary_penalty,
                -match.similarity_percent,
                abs(length - len(proposed)),
                abs(start - approximate_start),
            )
            if best is None or key < best[0]:
                best = (key, start, fragment, match)
    return (best[1], best[2], best[3]) if best is not None else None


_WHITESPACE = re.compile(r"\s+")


def fuzzy_match(
    proposed: str,
    candidates: Iterable[EvidenceCandidate[T]],
    fuzziness: FuzzinessOption,
) -> tuple[GroundingMatch[T], ...]:
    """Match a model-proposed value against canonical source candidates.

    Candidates retain their input order.  Callers that need a different tie
    policy can inspect every returned match or use ``resolve_by_condition``.
    """
    return tuple(
        match for candidate in candidates if (match := _match(proposed, candidate, fuzziness)) is not None
    )


def fuzzy_find(proposed: str, text: str, fuzziness: FuzzinessOption) -> str | None:
    """Find a proposed fragment in source text under the shared fuzzy policy.

    ``GroundingEvidence`` compares complete candidate values, which is right
    for a model choosing from a list. Page-backed reviewers instead quote a
    fragment from a long source page. This helper keeps that operation under
    the same perfect/whitespace/edit-distance policy rather than introducing
    a one-off page matcher.
    """
    if not proposed:
        return None
    exact_start = text.find(proposed)
    if exact_start >= 0:
        return text[exact_start : exact_start + len(proposed)]
    if fuzziness.types == frozenset({FuzzinessType.PERFECT_MATCH}):
        return None
    if FuzzinessType.WHITESPACE_RELAXATION in fuzziness.types:
        whitespace = FuzzinessOption.whitespace_relaxation()
        match = re.search(fuzzy_literal(proposed, whitespace, newline=True), text)
        if match is not None:
            return match.group()
    if FuzzinessType.EDIT_DISTANCE not in fuzziness.types:
        return None
    alignment = fuzz.partial_ratio_alignment(proposed, text)
    if alignment is None or not _fragment_is_within_policy(proposed, text, alignment, fuzziness):
        return None
    return text[alignment.dest_start : alignment.dest_end]


def _fragment_is_within_policy(
    proposed: str,
    text: str,
    alignment: object,
    fuzziness: FuzzinessOption,
) -> bool:
    """Apply the shared edit threshold to RapidFuzz's best page fragment."""
    score = getattr(alignment, "score", None)
    start = getattr(alignment, "dest_start", None)
    end = getattr(alignment, "dest_end", None)
    if not isinstance(score, float) or not isinstance(start, int) or not isinstance(end, int):
        return False
    if fuzziness.similarity_percent is not None:
        return score >= fuzziness.similarity_percent
    if fuzziness.maximum_edits is None:
        return False
    candidate = text[start:end]
    compared_proposed = (
        _normalize_whitespace(proposed)
        if FuzzinessType.WHITESPACE_RELAXATION in fuzziness.types
        else proposed
    )
    compared_candidate = (
        _normalize_whitespace(candidate)
        if FuzzinessType.WHITESPACE_RELAXATION in fuzziness.types
        else candidate
    )
    return Levenshtein.distance(compared_proposed, compared_candidate) <= fuzziness.maximum_edits


def _match(
    proposed: str,
    candidate: EvidenceCandidate[T],
    fuzziness: FuzzinessOption,
) -> GroundingMatch[T] | None:
    if proposed == candidate.text:
        return GroundingMatch(candidate, FuzzinessType.PERFECT_MATCH, 0, 100.0)
    if fuzziness.types == frozenset({FuzzinessType.PERFECT_MATCH}):
        return None

    normalized_proposed = _normalize_whitespace(proposed)
    normalized_candidate = _normalize_whitespace(candidate.text)
    if FuzzinessType.WHITESPACE_RELAXATION in fuzziness.types and normalized_proposed == normalized_candidate:
        return GroundingMatch(candidate, FuzzinessType.WHITESPACE_RELAXATION, 0, 100.0)

    if FuzzinessType.EDIT_DISTANCE not in fuzziness.types:
        return None
    compared_proposed = (
        normalized_proposed if FuzzinessType.WHITESPACE_RELAXATION in fuzziness.types else proposed
    )
    compared_candidate = (
        normalized_candidate if FuzzinessType.WHITESPACE_RELAXATION in fuzziness.types else candidate.text
    )
    edits = Levenshtein.distance(compared_proposed, compared_candidate)
    maximum_edits = _maximum_edits(compared_proposed, compared_candidate, fuzziness)
    if edits > maximum_edits:
        return None
    length = max(len(compared_proposed), len(compared_candidate))
    similarity = 100.0 if length == 0 else 100.0 * (1 - edits / length)
    return GroundingMatch(candidate, FuzzinessType.EDIT_DISTANCE, edits, similarity)


def _maximum_edits(proposed: str, candidate: str, fuzziness: FuzzinessOption) -> int:
    if fuzziness.maximum_edits is not None:
        return fuzziness.maximum_edits
    assert fuzziness.similarity_percent is not None
    if fuzziness.similarity_percent == 100:
        return 0
    error_rate = 1 - fuzziness.similarity_percent / 100
    return max(1, floor(max(len(proposed), len(candidate)) * error_rate))


def _normalize_whitespace(value: str) -> str:
    """Erase layout-only whitespace for the project's relaxed-string policy.

    ``fuzzy_literal(..., whitespace_relaxation())`` accepts zero or more
    whitespace characters at token joins. Grounding must mean the same thing:
    ``1: 24-cv`` and ``1:24-cv`` differ only in layout, while changing a
    punctuation mark or digit remains a substantive mismatch.
    """
    return _WHITESPACE.sub("", value)
