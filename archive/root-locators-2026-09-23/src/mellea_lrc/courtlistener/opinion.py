"""Sub-opinion domain conversion."""

from mellea_lrc.courtlistener.opinion_models import CourtListenerOpinion
from mellea_lrc.courtlistener.opinion_transport import CourtListenerOpinionResponsePayload


def normalize_opinion_payload(payload: object) -> CourtListenerOpinion:
    """Validate one external sub-opinion payload and convert it to the domain model."""
    return CourtListenerOpinionResponsePayload.model_validate(payload).to_domain()
