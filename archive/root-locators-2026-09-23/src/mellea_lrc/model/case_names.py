"""A case name as one value: where it is written, and what it parses to.

Four things travel together or they go out of step. **Where** the name is on the
page, **what** it says there, and the **two parties** it splits into. A reader
asked to repair a name returns all four -- it reads `Ass ' n of Specialty
Programs` on the page and reports the party as `Ass'n of Specialty Programs` --
and keeping only the span throws the repair away, which is what happened before
this existed.

The written text is kept as written, damage included, because it is what the
document holds and every span here indexes that. The parties are the repaired
reading, which is what a rule-based check compares against a record.

`plaintiff` is `None` for a case with no adverse party: `In re Flint Water
Cases` and `Ex parte Quirin` are whole names, and the single party is the
defendant, which is also how eyecite files them.
"""

from __future__ import annotations

from dataclasses import dataclass

from mellea_lrc.model.spans import Span


@dataclass(frozen=True, slots=True)
class CaseName:
    """One citation's case name, as found and as read."""

    span: Span
    """Where the name is written. Offsets index the document's own text."""

    text: str
    """The characters at `span`, as the document holds them, damage included."""

    plaintiff: str | None = None
    """The first party, repaired. `None` for a case with no adverse party."""

    defendant: str | None = None
    """The second party, repaired -- or the only party, for `In re` and `Ex parte`."""

    @property
    def parties(self) -> tuple[str | None, str | None]:
        """Both parties, for a check that wants them together."""
        return (self.plaintiff, self.defendant)
