"""Turn a model's copied case-name quote into source-backed citation state."""

from __future__ import annotations

import re

from mellea_lrc.llm import EvidenceCandidate, GroundingEvidence
from mellea_lrc.model.case_names import CaseName
from mellea_lrc.model.fuzziness import FuzzinessOption
from mellea_lrc.model.spans import Span

CASE_NAME_GROUNDING = FuzzinessOption.edit_distance(
    similarity_percent=90,
    whitespace_relaxation=True,
)


def ground_source_case_name(
    quote: str | None,
    *,
    before_locator: str,
    source_start: int,
    plaintiff: str | None = None,
    defendant: str | None = None,
) -> CaseName | None:
    """Locate a copied name before the target locator, retaining source bytes.

    ``before_locator`` must be an offset-preserving, target-only document view.
    Its first character is at ``source_start`` in the original document. The
    model's spelling is evidence to locate, never the value written to the
    citation; the resulting span and text come from the source view itself.
    """
    if quote is None or not quote.strip():
        return None
    # A caption repeated in nearby prose should attach to the occurrence
    # nearest this locator. Reserve edit distance for damaged source text.
    literal = re.compile(r"(?<!\w)" + r"\s+".join(re.escape(piece) for piece in quote.split()) + r"(?!\w)")
    exact = tuple(literal.finditer(before_locator))
    if exact:
        source_start_in_window, source_end_in_window = exact[-1].span()
        source_fragment = before_locator[source_start_in_window:source_end_in_window]
    else:
        grounded = GroundingEvidence((EvidenceCandidate(before_locator, None),)).find_fragment(
            quote,
            CASE_NAME_GROUNDING,
        )
        if grounded is None:
            return None
        source_start_in_window, source_end_in_window = grounded.start, grounded.end
        source_fragment = grounded.text
    trimmed = source_fragment.strip()
    if not trimmed:
        return None
    left = len(source_fragment) - len(source_fragment.lstrip())
    right = len(source_fragment) - len(source_fragment.rstrip())
    start = source_start + source_start_in_window + left
    end = source_start + source_end_in_window - right
    return CaseName(
        span=Span(start, end),
        text=trimmed,
        plaintiff=plaintiff,
        defendant=defendant,
    )
