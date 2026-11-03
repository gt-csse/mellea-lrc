"""CourtListener citation lookup boundary."""

from importlib import import_module
from typing import TYPE_CHECKING

from mellea_lrc.providers.courtlistener.models import (
    CourtListenerCitationLookup,
    CourtListenerCluster,
    CourtListenerClusterCitation,
    CourtListenerDocket,
    CourtListenerSearchPage,
    CourtListenerSearchResult,
)

if TYPE_CHECKING:
    from mellea_lrc.providers.courtlistener.client import (
        CourtListenerClient,
        CourtListenerConfig,
        CourtListenerConfigurationError,
        CourtListenerError,
        CourtListenerHTTPError,
        CourtListenerPayloadError,
        CourtListenerTransportError,
    )

_CLIENT_EXPORTS = {
    "CourtListenerClient",
    "CourtListenerConfig",
    "CourtListenerConfigurationError",
    "CourtListenerError",
    "CourtListenerHTTPError",
    "CourtListenerPayloadError",
    "CourtListenerTransportError",
}


def __getattr__(name: str) -> object:
    """Load transport classes only when a caller requests the client API."""
    if name not in _CLIENT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.client"), name)
    globals()[name] = value
    return value


__all__ = [
    "CourtListenerCitationLookup",
    "CourtListenerClient",
    "CourtListenerCluster",
    "CourtListenerClusterCitation",
    "CourtListenerConfig",
    "CourtListenerConfigurationError",
    "CourtListenerDocket",
    "CourtListenerError",
    "CourtListenerHTTPError",
    "CourtListenerPayloadError",
    "CourtListenerSearchPage",
    "CourtListenerSearchResult",
    "CourtListenerTransportError",
]
