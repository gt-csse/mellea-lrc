"""Pydantic boundary models for CourtListener Opinion and RECAP search JSON."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue

from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.courtlistener.search_models import (
    CourtListenerRecapDocumentRecord,
    CourtListenerSearchRecord,
)


class _SearchPayload(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="allow")

    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    extra_data: dict[str, JsonValue] = Field(default_factory=dict)

    def collected_extra_data(self) -> ExtraData:
        """Combine explicit and previously unknown external fields."""
        values: dict[str, object] = dict(self.extra_data)
        values.update(self.__pydantic_extra__ or {})
        return ExtraData(values)


class CourtListenerRecapDocumentPayload(_SearchPayload):
    """One RECAP document returned directly or nested in a docket result."""

    id: int | str | None = None
    recap_document_id: int | str | None = Field(
        default=None,
        validation_alias=AliasChoices("recap_document_id", "recapDocumentId"),
    )
    docket_id: int | str | None = Field(
        default=None,
        validation_alias=AliasChoices("docket_id", "docketId", "cl_docket_id"),
    )
    entry_number: int | str | None = Field(
        default=None,
        validation_alias=AliasChoices("entry_number", "entryNumber"),
    )
    document_number: int | str | None = Field(
        default=None,
        validation_alias=AliasChoices("document_number", "documentNumber"),
    )
    attachment_number: int | str | None = Field(
        default=None,
        validation_alias=AliasChoices("attachment_number", "attachmentNumber"),
    )
    description: str | None = None
    entry_date_filed: str | None = Field(
        default=None,
        validation_alias=AliasChoices("entry_date_filed", "entryDateFiled"),
    )
    pacer_doc_id: int | str | None = Field(
        default=None,
        validation_alias=AliasChoices("pacer_doc_id", "pacerDocId"),
    )
    filepath_local: str | None = None
    filepath_ia: str | None = None
    absolute_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("absolute_url", "absoluteUrl"),
    )
    snippet: str | None = None
    page_count: int | None = None
    available: bool | None = None

    def to_domain(self) -> CourtListenerRecapDocumentRecord:
        """Convert validated transport data to an immutable document record."""
        return CourtListenerRecapDocumentRecord(
            recap_document_id=_string_id(self.recap_document_id or self.id),
            docket_id=_string_id(self.docket_id),
            entry_number=_string_id(self.entry_number),
            document_number=_string_id(self.document_number),
            attachment_number=_string_id(self.attachment_number),
            description=self.description,
            entry_date_filed=self.entry_date_filed,
            pacer_doc_id=_string_id(self.pacer_doc_id),
            filepath_local=self.filepath_local,
            filepath_ia=self.filepath_ia,
            absolute_url=self.absolute_url,
            snippet=self.snippet,
            page_count=self.page_count,
            available=(
                self.available
                if self.available is not None
                else bool(self.filepath_local or self.filepath_ia)
            ),
            extra_data=self.collected_extra_data(),
        )


class CourtListenerSearchRecordPayload(_SearchPayload):
    """One result from any supported CourtListener search corpus."""

    id: int | str | None = None
    cluster_id: int | str | None = Field(
        default=None,
        validation_alias=AliasChoices("cluster_id", "clusterId"),
    )
    docket_id: int | str | None = Field(
        default=None,
        validation_alias=AliasChoices("docket_id", "docketId", "cl_docket_id"),
    )
    court_id: int | str | None = Field(
        default=None,
        validation_alias=AliasChoices("court_id", "courtId", "court"),
    )
    docket_number: int | str | None = Field(
        default=None,
        validation_alias=AliasChoices("docket_number", "docketNumber"),
    )
    case_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("case_name", "caseName"),
    )
    date_filed: str | None = Field(
        default=None,
        validation_alias=AliasChoices("date_filed", "dateFiled"),
    )
    date_terminated: str | None = Field(
        default=None,
        validation_alias=AliasChoices("date_terminated", "dateTerminated"),
    )
    absolute_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("absolute_url", "absoluteUrl"),
    )
    snippet: str | None = None
    resource_uri: str | None = None
    recap_documents: list[CourtListenerRecapDocumentPayload] = Field(
        default_factory=list,
        validation_alias=AliasChoices("recap_documents", "recapDocuments"),
    )
    more_docs: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("more_docs", "moreDocs"),
    )

    def to_domain(self, search_type: str) -> CourtListenerSearchRecord:
        """Convert one validated result according to its requested corpus."""
        record_id = _string_id(self.id)
        documents = tuple(item.to_domain() for item in self.recap_documents)
        if search_type == "rd":
            document = CourtListenerRecapDocumentPayload.model_validate(self.model_dump())
            documents = (document.to_domain(),)
        return CourtListenerSearchRecord(
            cluster_id=_string_id(self.cluster_id) or (record_id if search_type == "o" else None),
            docket_id=_string_id(self.docket_id) or (
                record_id if search_type in {"r", "d"} else None
            ),
            court_id=_string_id(self.court_id),
            docket_number=_string_id(self.docket_number),
            case_name=self.case_name,
            date_filed=self.date_filed,
            date_terminated=self.date_terminated,
            absolute_url=self.absolute_url,
            snippet=self.snippet,
            resource_uri=self.resource_uri,
            recap_documents=documents,
            more_docs=self.more_docs,
            extra_data=self.collected_extra_data(),
        )


class CourtListenerSearchResponsePayload(_SearchPayload):
    """One page returned by CourtListener's search endpoint."""

    query: str | None = Field(default=None, validation_alias=AliasChoices("query", "q"))
    search_type: str | None = Field(
        default=None,
        validation_alias=AliasChoices("search_type", "type"),
    )
    semantic: bool | None = None
    http_status: int | None = None
    cache: str | None = None
    key: str | None = None
    count: int | None = None
    results: list[CourtListenerSearchRecordPayload] = Field(default_factory=list)
    next: str | None = None
    previous: str | None = None
    next_cursor: str | None = None
    previous_cursor: str | None = None
    detail: str | dict[str, JsonValue] | None = None


def _string_id(value: int | str | None) -> str | None:
    return str(value) if value is not None else None
