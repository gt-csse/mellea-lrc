"""Opinion-cluster and sub-opinion domain conversion."""

from mellea_lrc.courtlistener.opinion_models import (
    CourtListenerOpinion,
    CourtListenerOpinionCluster,
)
from mellea_lrc.courtlistener.opinion_transport import (
    CourtListenerOpinionClusterPayload,
    CourtListenerOpinionResponsePayload,
)


def normalize_opinion_cluster_payload(payload: object) -> CourtListenerOpinionCluster:
    """Validate one external cluster payload and convert it to the domain model."""
    return CourtListenerOpinionClusterPayload.model_validate(payload).to_domain()


def normalize_opinion_payload(payload: object) -> CourtListenerOpinion:
    """Validate one external sub-opinion payload and convert it to the domain model."""
    return CourtListenerOpinionResponsePayload.model_validate(payload).to_domain()
