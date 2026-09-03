"""The order the passes over a document's citations run in, and why.

Once eyecite has produced a citation for every locator it could read, several
passes refine that list. They are not independent: one of them reads an answer
another one wrote, so running them in the wrong order does not fail, it produces
a plausible wrong result -- which is the kind of mistake this project is
supposed to be bad at making.

So the order is a named sequence rather than a line of nested calls. Each stage
carries the reason it sits where it does, and a test asserts the constraints
those reasons state.

## The constraint that exists today

``colocation`` before ``post_citation``. Co-location is decided from the spans
eyecite produced. The post-citation re-read then bounds each citation's search
for a court and date at *the next citation that is not co-located with it* --
because a parallel citation is one decision in several reporters and its first
member has to read across the others to reach the single date at the end. Run
the re-read first and there are no co-location ids to bound it with, so 30
parallel citations lose the year they legitimately reach for.

## What is not a stage

Relaxing pin cites is not here. It is not a pass over citations but a swap of
eyecite's module state around the parse itself, so it lives where the parse is.
Its ordering constraint is different in kind: it has to be in effect *while*
`get_citations` runs, and it has to be off at ``Relaxation.NONE`` so the
evaluation's floor arm keeps measuring eyecite rather than us.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mellea_lrc.extraction.reading.post_citation import reread_post_citation
from mellea_lrc.extraction.structure.colocation import assign_colocation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea_lrc.extraction.types import ExtractedCitation


class Pass(Protocol):
    """One refinement over a document's citations, given the text they index."""

    def __call__(
        self, text: str, citations: Sequence[ExtractedCitation]
    ) -> tuple[ExtractedCitation, ...]: ...


def _colocation(text: str, citations: Sequence[ExtractedCitation]) -> tuple[ExtractedCitation, ...]:
    """Group citations occupying the same span. Does not read the text."""
    del text
    return assign_colocation(citations)


def _post_citation(text: str, citations: Sequence[ExtractedCitation]) -> tuple[ExtractedCitation, ...]:
    """Re-read each case citation's court and date inside its own boundary."""
    return reread_post_citation(text, citations)


@dataclass(frozen=True, slots=True)
class Stage:
    """One pass, named, with the reason it runs where it does."""

    name: str
    run: Pass
    why: str
    """Why this position. Read as a constraint, not a description."""


STAGES: tuple[Stage, ...] = (
    Stage(
        name="colocation",
        run=_colocation,
        why=(
            "Decided from the spans eyecite produced, before anything alters them, and "
            "before post_citation, which is defined in terms of its ids."
        ),
    ),
    Stage(
        name="post_citation",
        run=_post_citation,
        why=(
            "Needs co-location ids: a citation may read past a co-located neighbour for "
            "the single date a parallel citation puts at the end, and must stop at any "
            "other citation. It also trims spans, so it runs after anything reading them."
        ),
    ),
)


def refine(text: str, citations: Sequence[ExtractedCitation]) -> tuple[ExtractedCitation, ...]:
    """Run every stage over the citations, in order."""
    refined = tuple(citations)
    for stage in STAGES:
        refined = stage.run(text, refined)
    return refined
