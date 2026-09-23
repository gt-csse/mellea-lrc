"""Domain model and public URL helpers for CourtListener dockets."""

from dataclasses import dataclass
from urllib.parse import urljoin

COURTLISTENER_PUBLIC_ORIGIN = "https://www.courtlistener.com/"


def courtlistener_docket_url(absolute_url: str | None) -> str | None:
    """Resolve CourtListener's docket path to its canonical absolute URL."""
    return urljoin(COURTLISTENER_PUBLIC_ORIGIN, absolute_url) if absolute_url else None


@dataclass(frozen=True, slots=True)
class CourtListenerDocket:
    """CourtListener docket identity and its authoritative court identifier."""

    docket_id: str
    court_id: str | None = None
    case_name: str | None = None
