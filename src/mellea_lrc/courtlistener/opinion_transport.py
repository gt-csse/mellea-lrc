"""Inbound boundary models shared by CourtListener opinion-cluster responses."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from mellea_lrc.courtlistener.opinion_models import (
    CourtListenerOpinion,
    CourtListenerOpinionCluster,
    CourtListenerOpinionClusterCitation,
)

MIN_RESOURCE_URL_PARTS = 2


class CourtListenerOpinionClusterCitationPayload(BaseModel):
    """External reporter citation attached to a cluster."""

    volume: str
    reporter: str
    page: str

    def to_domain(self) -> CourtListenerOpinionClusterCitation:
        """Convert the external citation to a stable domain value."""
        return CourtListenerOpinionClusterCitation(
            volume=self.volume,
            reporter=self.reporter,
            page=self.page,
        )


class CourtListenerOpinionClusterPayload(BaseModel):
    """External CourtListener opinion-cluster payload."""

    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")

    cluster_id: int | str | None = Field(
        default=None,
        validation_alias=AliasChoices("cluster_id", "clusterId", "id"),
    )
    case_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("case_name", "caseName"),
    )
    date_filed: str | None = Field(
        default=None,
        validation_alias=AliasChoices("date_filed", "dateFiled"),
    )
    court: str | None = None
    court_id: str | None = None
    docket_id: int | str | None = Field(
        default=None,
        validation_alias=AliasChoices("docket_id", "docketId"),
    )
    citations: list[CourtListenerOpinionClusterCitationPayload] = Field(default_factory=list)
    sub_opinions: list[str] = Field(default_factory=list)

    def to_domain(self) -> CourtListenerOpinionCluster:
        """Convert transport scalars to the stable opinion-cluster model."""
        return CourtListenerOpinionCluster(
            cluster_id=str(self.cluster_id) if self.cluster_id is not None else None,
            case_name=self.case_name,
            date_filed=self.date_filed,
            court=self.court,
            court_id=self.court_id,
            docket_id=str(self.docket_id) if self.docket_id is not None else None,
            citations=tuple(citation.to_domain() for citation in self.citations),
            sub_opinion_ids=tuple(_resource_id(url, resource="opinions") for url in self.sub_opinions),
        )


class CourtListenerOpinionResponsePayload(BaseModel):
    """External CourtListener sub-opinion payload."""

    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")

    id: int | str
    cluster_id: int | str | None = Field(
        default=None,
        validation_alias=AliasChoices("cluster_id", "clusterId"),
    )
    cluster: str | None = None
    type: str
    ordering_key: int | None = None
    html_with_citations: str | None = None

    def to_domain(self) -> CourtListenerOpinion:
        """Convert an opinion response to the retrieval domain model."""
        cluster_id = (
            str(self.cluster_id)
            if self.cluster_id is not None
            else _resource_id(self.cluster, resource="clusters")
            if self.cluster
            else None
        )
        return CourtListenerOpinion(
            opinion_id=str(self.id),
            cluster_id=cluster_id,
            opinion_type=self.type,
            ordering_key=self.ordering_key,
            html_with_citations=self.html_with_citations or "",
        )


def _resource_id(url: str, *, resource: str) -> str:
    parts = url.rstrip("/").split("/")
    if len(parts) < MIN_RESOURCE_URL_PARTS or parts[-2] != resource or not parts[-1]:
        msg = f"Expected a CourtListener {resource!r} resource URL"
        raise ValueError(msg)
    return parts[-1]
