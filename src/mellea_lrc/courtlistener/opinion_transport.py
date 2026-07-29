"""Inbound boundary models shared by CourtListener opinion-cluster responses."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from mellea_lrc.courtlistener.opinion_models import CourtListenerOpinionCluster


class CourtListenerOpinionClusterPayload(BaseModel):
    """External CourtListener payload for one opinion cluster."""

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

    def to_domain(self) -> CourtListenerOpinionCluster:
        """Convert transport scalars to the stable opinion-cluster model."""
        return CourtListenerOpinionCluster(
            cluster_id=str(self.cluster_id) if self.cluster_id is not None else None,
            case_name=self.case_name,
            date_filed=self.date_filed,
            court=self.court,
            court_id=self.court_id,
            docket_id=str(self.docket_id) if self.docket_id is not None else None,
        )
