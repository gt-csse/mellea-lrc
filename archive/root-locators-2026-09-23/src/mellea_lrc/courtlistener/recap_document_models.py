"""Domain model for one CourtListener RECAP document."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CourtListenerRecapDocument:
    """A RECAP document's text, separate from its containing docket."""

    document_id: str
    plain_text: str
