"""Domain model for one CourtListener docket."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CourtListenerDocket:
    """CourtListener docket identity and its authoritative court identifier."""

    docket_id: str
    court_id: str | None = None
    case_name: str | None = None
