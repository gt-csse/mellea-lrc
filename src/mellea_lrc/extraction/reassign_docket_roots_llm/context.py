"""Fuzzy docket neighborhoods and their source evidence for root reassignment."""

from __future__ import annotations

import json
from dataclasses import dataclass

from mellea_lrc.matching.fuzziness import FuzzinessOption
from mellea_lrc.matching.grounding import EvidenceCandidate, GroundingEvidence
from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.docket_root_llm_reassignment import DocketRootPartition
from mellea_lrc.model.document import Document

MINIMUM_SIMILARITY_PERCENT = 40.0
_FUZZINESS = FuzzinessOption.edit_distance(
    similarity_percent=MINIMUM_SIMILARITY_PERCENT, whitespace_relaxation=True
)


@dataclass(frozen=True, slots=True)
class DocketRootCandidate:
    """One root and the filing evidence relevant to same-case judgment."""

    citation_id: str
    docket_number: str
    locator: str
    case_name: str | None
    court: str | None
    date: str | None
    context: str


@dataclass(frozen=True, slots=True)
class DocketRootReviewContext:
    """A connected candidate neighborhood, indexed in source order."""

    candidates: tuple[DocketRootCandidate, ...]

    @classmethod
    def from_document(
        cls, document: Document, roots: tuple[FullDocketCitation, ...]
    ) -> DocketRootReviewContext:
        candidates: list[DocketRootCandidate] = []
        for root in roots:
            locator = root.locator[-1]
            court = root.court[-1] if root.court else None
            court_text = None
            if court is not None:
                court_text = court.quote
                if court.normalizable:
                    court_text = f"{court_text or 'inferred'} [court ID: {court.get_normalized().id}]"
            span = root.locator_span
            candidates.append(
                DocketRootCandidate(
                    citation_id=root.id,
                    docket_number=document.text[locator.number_span.start : locator.number_span.end],
                    locator=locator.quote,
                    case_name=root.case_name[-1].quote if root.case_name else None,
                    court=court_text,
                    date=root.date[-1].quote if root.date else None,
                    context=document.text[max(0, span.start - 180) : min(len(document.text), span.end + 140)],
                )
            )
        return cls(candidates=tuple(candidates))

    def partition_error(self, decision: DocketRootPartition) -> str | None:
        indices = {index for group in decision.groups for index in group}
        expected = set(range(len(self.candidates)))
        if indices != expected:
            return f"Groups must contain each candidate index from 0 through {len(self.candidates) - 1} exactly once."
        return None

    def prompt_candidates(self) -> str:
        return json.dumps(
            [
                {
                    "index": index,
                    "docket_number": candidate.docket_number,
                    "locator": candidate.locator,
                    "case_name": candidate.case_name,
                    "court": candidate.court,
                    "date": candidate.date,
                    "filing_context": candidate.context,
                }
                for index, candidate in enumerate(self.candidates)
            ],
            ensure_ascii=False,
        )


def _similar_numbers(left: str, right: str) -> bool:
    """Use the project's whitespace relaxed edit-distance policy on numbers only."""
    match = GroundingEvidence((EvidenceCandidate(right.casefold(), None),)).resolve(
        left.casefold(), _FUZZINESS
    )
    return match is not None and match.similarity_percent >= MINIMUM_SIMILARITY_PERCENT


def candidate_components(
    document: Document,
) -> tuple[tuple[FullDocketCitation, ...], ...]:
    roots = tuple(root for root in document.roots if isinstance(root, FullDocketCitation))
    parent = list(range(len(roots)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index, left in enumerate(roots):
        left_span = left.locator[-1].number_span
        left_number = document.text[left_span.start : left_span.end]
        for right_index in range(left_index + 1, len(roots)):
            right_span = roots[right_index].locator[-1].number_span
            right_number = document.text[right_span.start : right_span.end]
            if _similar_numbers(left_number, right_number):
                parent[find(right_index)] = find(left_index)

    grouped: dict[int, list[FullDocketCitation]] = {}
    for index, root in enumerate(roots):
        grouped.setdefault(find(index), []).append(root)
    return tuple(tuple(group) for group in grouped.values() if len(group) >= 2)
