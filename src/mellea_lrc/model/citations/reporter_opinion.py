"""Reporter-root opinion retrieval, separate from selecting a cited writing."""

from __future__ import annotations

from enum import Enum
from typing import Any, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.providers.courtlistener.models import CourtListenerCluster


def opinion_resource_id(value: str | int, endpoint: str) -> str:
    """Read a REST resource ID; public opinion URLs contain cluster IDs."""
    if isinstance(value, bool):
        raise ValueError("A CourtListener resource ID cannot be a boolean")
    text = str(value)
    if text.isdecimal():
        return text
    path = urlsplit(text).path.rstrip("/")
    prefix, separator, identifier = path.rpartition("/")
    if not separator or not prefix.endswith(f"/{endpoint}") or not identifier.isdecimal():
        raise ValueError(f"Expected a CourtListener {endpoint} REST resource: {text}")
    return identifier


class OpinionRetrievalOutcome(str, Enum):
    RETRIEVED = "retrieved"
    NOT_FOUND = "not_found"
    EMPTY_TEXT = "empty_text"


class ReporterRootOpinionSource(BaseModel):
    """The original cluster bound to a root, independent of identity review.

    Retain its complete metadata so pagination and writing selection do not
    need to recover a candidate from the earlier identity workflow.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    cluster: CourtListenerCluster

    @property
    def cluster_id(self) -> str:
        if self.cluster.id is None:
            raise ValueError("An opinion source cluster needs an ID")
        return self.cluster.id

    @property
    def sub_opinion_ids(self) -> tuple[str, ...]:
        resources = self.cluster.raw_json.get("sub_opinions")
        if not isinstance(resources, list):
            raise ValueError("An opinion source cluster needs a sub_opinions list")
        return tuple(opinion_resource_id(resource, "opinions") for resource in resources)

    @model_validator(mode="after")
    def _validate_cluster(self) -> Self:
        if not self.cluster_id.isdecimal():
            raise ValueError("An opinion source cluster ID must contain decimal digits")
        self.sub_opinion_ids
        return self


class RetrievedReporterOpinion(BaseModel):
    """One complete provider response, including original HTML and page markers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    opinion_id: str
    cluster_id: str
    outcome: OpinionRetrievalOutcome
    response: dict[str, Any] | None = None

    @property
    def text_field(self) -> str | None:
        # Preserve HTML rather than flattening page markers. Later page reading
        # can interpret each format; a PDF download is a separate retrieval.
        if self.response is None:
            return None
        return next(
            (
                name
                for name in (
                    "html_with_citations",
                    "html",
                    "html_lawbox",
                    "html_columbia",
                    "html_anon_2020",
                    "xml_harvard",
                    "plain_text",
                )
                if isinstance(self.response.get(name), str) and self.response[name].strip()
            ),
            None,
        )

    @model_validator(mode="after")
    def _validate_response(self) -> Self:
        if not self.opinion_id.isdecimal() or not self.cluster_id.isdecimal():
            raise ValueError("Opinion and cluster IDs must contain decimal digits")
        if self.response is None:
            if self.outcome is not OpinionRetrievalOutcome.NOT_FOUND:
                raise ValueError("An absent response must be recorded as not_found")
        else:
            for key in ("id", "cluster"):
                if key not in self.response:
                    raise ValueError(f"Retrieved opinion response is missing its {key!r} field")
            if opinion_resource_id(self.response["id"], "opinions") != self.opinion_id:
                raise ValueError("Retrieved opinion ID differs from the requested opinion")
            if opinion_resource_id(self.response["cluster"], "clusters") != self.cluster_id:
                raise ValueError("Retrieved opinion does not belong to the selected cluster")
            expected = (
                OpinionRetrievalOutcome.RETRIEVED if self.text_field else OpinionRetrievalOutcome.EMPTY_TEXT
            )
            if self.outcome is not expected:
                raise ValueError("Opinion retrieval outcome must reflect the saved text")
        return self


class ReporterRootOpinionRetrieval(BaseModel):
    """All subopinions fetched for the root's independently bound source.

    This is retrieval, not a majority/dissent selection. A cluster can contain
    several writings, and a combined opinion can contain several opinion types.
    Type and display order do not prove which writing a pinpoint cites.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    cluster_id: str
    sub_opinion_ids: tuple[str, ...]
    opinions: tuple[RetrievedReporterOpinion, ...]

    @model_validator(mode="after")
    def _validate_members(self) -> Self:
        if not self.cluster_id.isdecimal():
            raise ValueError("Retrieval requires a decimal cluster ID")
        if tuple(dict.fromkeys(self.sub_opinion_ids)) != tuple(item.opinion_id for item in self.opinions):
            raise ValueError("Every distinct subopinion must have one ordered retrieval result")
        if any(item.cluster_id != self.cluster_id for item in self.opinions):
            raise ValueError("Every opinion must belong to the selected cluster")
        return self
