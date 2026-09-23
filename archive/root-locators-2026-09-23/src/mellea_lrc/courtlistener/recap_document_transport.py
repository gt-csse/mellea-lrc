"""Inbound boundary for CourtListener RECAP document responses."""

from pydantic import BaseModel, ConfigDict

from mellea_lrc.courtlistener.recap_document_models import CourtListenerRecapDocument


class CourtListenerRecapDocumentPayload(BaseModel):
    """The fields needed to inspect a citation in a RECAP document body."""

    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")

    id: int | str
    plain_text: str | None = None

    def to_domain(self) -> CourtListenerRecapDocument:
        """Keep unavailable text empty so callers can defer the review."""
        return CourtListenerRecapDocument(
            document_id=str(self.id),
            plain_text=self.plain_text or "",
        )


def normalize_recap_document_payload(payload: object) -> CourtListenerRecapDocument:
    """Validate and convert one RECAP document response."""
    return CourtListenerRecapDocumentPayload.model_validate(payload).to_domain()
