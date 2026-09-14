"""What is true of a document and of no citation in it.

A record is a citation. Some of what a pass learns is not: a leaf it could not
grow, a site it hunted and rejected, a name standing in the text that no
citation covers. Making those into records would put a citation in the document
that the document does not hold, which is the invariant `CitationRecord` exists
to enforce -- so they go here instead, beside the citations rather than among
them.

A finding is evidence of an absence, so it carries what was read and could not
be built. `citation` is that reading: a canonical citation with no
`citation_id`, no trace and no place in the tree, because it is not part of the
document -- it is what the pass would have added if anything had let it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mellea_lrc.core.citations import CanonicalCitation
    from mellea_lrc.core.spans import Span


class FindingKind(str, Enum):
    """What kind of absence this is."""

    UNGROWN_LEAF = "ungrown_leaf"
    """A short form, `Id.`, `supra` or bare name whose root the document does not hold.

    Either the root is there and was not read, which is an extraction miss, or
    there is no root to reach, which makes the leaf a defect: a case the filing
    cites and never gives in full. Which of the two it is, is not decided here.
    """


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing a pass learned that belongs to no citation."""

    kind: FindingKind
    stage: str
    """Which stage found it: `extraction`, `identity`, `pinpoint`."""

    made_by: str
    """A rule, by module name, or a model, by the name the session reports."""

    message: str
    """One line, in the words of whatever found it."""

    span: Span | None = None
    """Where in the document it is about, when it is about a place."""

    citation: CanonicalCitation | None = None
    """The reading that could not be built, when there was one."""
