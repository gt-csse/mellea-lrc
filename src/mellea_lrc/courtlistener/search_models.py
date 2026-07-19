"""Domain models for CourtListener Opinion and RECAP search."""

from dataclasses import dataclass, field

from mellea_lrc.core.immutable import ExtraData


@dataclass(frozen=True, slots=True)
class CourtListenerRecapDocumentRecord:
    """One RECAP document embedded in a CourtListener search result."""

    recap_document_id: str | None = None
    docket_id: str | None = None
    entry_number: str | None = None
    document_number: str | None = None
    attachment_number: str | None = None
    description: str | None = None
    entry_date_filed: str | None = None
    pacer_doc_id: str | None = None
    filepath_local: str | None = None
    filepath_ia: str | None = None
    absolute_url: str | None = None
    snippet: str | None = None
    page_count: int | None = None
    available: bool = False
    extra_data: ExtraData = field(default_factory=ExtraData)


@dataclass(frozen=True, slots=True)
class CourtListenerSearchRecord:
    """One normalized result from a CourtListener search corpus."""

    cluster_id: str | None = None
    docket_id: str | None = None
    court_id: str | None = None
    docket_number: str | None = None
    case_name: str | None = None
    date_filed: str | None = None
    date_terminated: str | None = None
    absolute_url: str | None = None
    snippet: str | None = None
    resource_uri: str | None = None
    recap_documents: tuple[CourtListenerRecapDocumentRecord, ...] = ()
    more_docs: bool | None = None
    extra_data: ExtraData = field(default_factory=ExtraData)


@dataclass(frozen=True, slots=True)
class CourtListenerSearchResult:
    """Validated response from one CourtListener search request."""

    query: str
    search_type: str
    semantic: bool
    http_status: int | None
    count: int | None
    records: tuple[CourtListenerSearchRecord, ...]
    next_cursor: str | None = None
    previous_cursor: str | None = None
    cache: str | None = None
    key: str | None = None
    error_message: str | None = None
    extra_data: ExtraData = field(default_factory=ExtraData)

    def __post_init__(self) -> None:
        if self.search_type not in {"r", "rd", "d", "o"}:
            msg = "CourtListener search type must be one of: r, rd, d, o"
            raise ValueError(msg)
        if self.count is not None and (isinstance(self.count, bool) or self.count < 0):
            msg = "CourtListener search count must be non-negative"
            raise ValueError(msg)
