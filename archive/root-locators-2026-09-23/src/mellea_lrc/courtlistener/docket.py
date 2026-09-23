"""Docket retrieval and domain conversion."""

from mellea_lrc.courtlistener.docket_models import CourtListenerDocket
from mellea_lrc.courtlistener.docket_transport import CourtListenerDocketResponsePayload


def normalize_docket_payload(payload: object) -> CourtListenerDocket:
    """Validate one external docket payload and convert it to the domain model."""
    return CourtListenerDocketResponsePayload.model_validate(payload).to_domain()
