"""What has touched one field of one citation, in order.

A citation's fields are not all read once. The rules read a case name from the
offsets eyecite gives; a reader asked about a name standing outside every
citation may then say that name belongs to *this* citation, and be right where
the rules were not. Keeping only the second answer loses which one a measurement
is measuring, and keeping only the first throws the reader's work away.

So a logged field records every touch. The first is whatever built the citation
-- the deterministic pass, or a reader that proposed the citation in the first
place -- and each later one says what changed it and why. **The last touch is
the field's value**: there is one place to read it from and no way for a stored
value and its history to disagree.

`None` is a value like any other. A citation the rules found no name for opens
its log with `None`, and a reader writing a name over it is an overwrite, which
is what makes it visible as one.

Only `case_name` is logged today. The container is keyed by field name because
the next one costs nothing then.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: What the deterministic pass is called in a log it opens.
RULES = "extraction"


@dataclass(frozen=True, slots=True)
class FieldTouch:
    """One thing that set a field, and what it set it to."""

    by: str
    """What touched it: `extraction`, or the reader that changed it."""

    value: Any
    """What the field holds after this touch, `None` included."""

    reason: str | None = None
    """One line, when whatever touched it gave one."""


class FieldLog:
    """The touches on each logged field of one citation.

    Mutable, and deliberately: a citation that has been patched is the same
    citation, so the history follows the object rather than being rebuilt
    beside it. `dataclasses.replace` copies the reference, which is why a
    citation that gains a `root_id` keeps the name a reader gave it.
    """

    __slots__ = ("_touches",)

    def __init__(self, touches: dict[str, list[FieldTouch]] | None = None) -> None:
        self._touches: dict[str, list[FieldTouch]] = {
            field: list(entries) for field, entries in (touches or {}).items()
        }

    def touch(self, field: str, value: Any, *, by: str, reason: str | None = None) -> None:
        """Record that `by` set `field` to `value`."""
        self._touches.setdefault(field, []).append(FieldTouch(by=by, value=value, reason=reason))

    def history(self, field: str) -> tuple[FieldTouch, ...]:
        """Every touch on `field`, in the order they happened."""
        return tuple(self._touches.get(field, ()))

    def latest(self, field: str) -> FieldTouch | None:
        """The touch that set the field's current value, or `None` if untouched."""
        history = self.history(field)
        return history[-1] if history else None

    def value(self, field: str, default: Any = None) -> Any:
        """What `field` holds now."""
        latest = self.latest(field)
        return latest.value if latest is not None else default

    def fields(self) -> tuple[str, ...]:
        """Every field this log has a history for."""
        return tuple(self._touches)

    def __repr__(self) -> str:
        inside = ", ".join(f"{field}={len(entries)}" for field, entries in self._touches.items())
        return f"FieldLog({inside})"
