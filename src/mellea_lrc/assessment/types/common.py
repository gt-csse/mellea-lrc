"""Types shared across assessment domains."""

from dataclasses import dataclass, field

from mellea_lrc.core.immutable import ExtraData


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """One typed conversation turn retained as assessment provenance."""

    role: str
    content: str
    extra_data: ExtraData = field(default_factory=ExtraData)
