"""Docket site hunting is intentionally unavailable.

Stable root extraction reads one generic, signaled docket envelope.  A future
hunter may propose weaker candidates only after it has a reviewed promotion
contract.  It must not revive a second docket grammar behind the caller's back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.extraction.reading.dockets import CourtCandidate

if TYPE_CHECKING:
    from mellea_lrc.extraction.types import Document


@dataclass(frozen=True, slots=True)
class SuspectedDocket:
    """Reserved candidate shape for a future independently reviewed hunter."""

    span_start: int
    span_end: int
    docket_text: str
    courts: tuple[CourtCandidate, ...]
    window: str


def suspected_dockets(document: Document) -> tuple[SuspectedDocket, ...]:
    """Raise until docket-site generation has an approved independent design."""
    del document
    raise NotImplementedError(
        "Docket site hunting is not implemented: stable root extraction owns the "
        "single docket reader, and a future hunter requires its own candidate and "
        "review contract."
    )


def docket_context(site: SuspectedDocket) -> str:
    """Describe the court candidates a future reviewer would be given."""
    if not site.courts:
        return (
            "No court string was found near this docket number. A docket number "
            "identifies a case only together with its court, so if no court is "
            "written in the window, report the court as null."
        )
    described = "; ".join(f'"{court.text}" is {court.court_name}' for court in site.courts)
    return (
        f"Court strings written near this docket number: {described}. "
        f"Use one of these exactly as written, or null if none of them is the "
        f"court of the case this docket number belongs to."
    )
