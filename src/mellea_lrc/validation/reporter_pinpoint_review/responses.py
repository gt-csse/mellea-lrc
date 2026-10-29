"""Model responses for paginated and unpaginated opinion review.

Printed locations come from the retained source index, not model-generated
numbers. The durable support decision is constructed after quote grounding.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mellea_lrc.model.citations.fields.base import require_all_json_properties
from mellea_lrc.model.citations.reporter_pinpoint import (
    OpinionEvidenceQuote as StoredOpinionEvidenceQuote,
)
from mellea_lrc.model.citations.reporter_pinpoint import OpinionSupportResult

MAX_EVIDENCE_QUOTES = 3
MAX_QUOTE_CHARACTERS = 1200


# Bound newly generated quotes in the response schema without restricting
# previously retained evidence in serialized documents.
class OpinionEvidenceQuote(StoredOpinionEvidenceQuote):
    quote: str = Field(max_length=MAX_QUOTE_CHARACTERS)


def validate_evidence_quotes(evidence: tuple[StoredOpinionEvidenceQuote, ...]) -> None:
    """Bound new review evidence before any potentially expensive fuzzy match.

    Request sentences in the prompt, but enforce size rather than guessing
    sentence boundaries around legal abbreviations. Stored documents retain
    their original evidence and are not subject to this response policy.
    """
    if len(evidence) > MAX_EVIDENCE_QUOTES:
        raise ValueError(
            f"Too many evidence quotes ({len(evidence)}); return at most {MAX_EVIDENCE_QUOTES} "
            "separate representative sentences or short fragments. Focus on one or two sentences "
            "and put the detailed explanation in reason."
        )
    for index, item in enumerate(evidence):
        if len(item.quote) > MAX_QUOTE_CHARACTERS:
            raise ValueError(
                f"Evidence quote {index + 1} is too long ({len(item.quote)} characters); "
                f"the limit is {MAX_QUOTE_CHARACTERS} characters per quote. "
                "Focus on one or two representative sentences, each in a separate evidence item; "
                "put the detailed explanation in reason. Do not copy the surrounding discussion."
            )


class PageEvidenceQuote(OpinionEvidenceQuote):
    footnote: str | None = Field(
        description="The printed footnote containing this quoted passage, if established; otherwise null."
    )

    @model_validator(mode="after")
    def _valid_footnote(self) -> Self:
        if self.footnote is not None and not self.footnote.strip():
            raise ValueError("A footnote label must be nonempty or null")
        return self


class UnpaginatedSupportResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", json_schema_extra=require_all_json_properties)

    result: OpinionSupportResult
    evidence: tuple[OpinionEvidenceQuote, ...] = Field(
        description="Up to three independent exact source quotes. Each item should contain one "
        "representative sentence or short fragment, at most 1200 characters. "
        "The items need not be adjacent; put the explanation in reason.",
        json_schema_extra={"maxItems": MAX_EVIDENCE_QUOTES},
    )
    reason: str

    @model_validator(mode="after")
    def _valid_decision(self) -> Self:
        validate_evidence_quotes(self.evidence)
        if not self.reason.strip():
            raise ValueError("Opinion support review requires a prose reason")
        if (
            self.result in {OpinionSupportResult.SUPPORTED, OpinionSupportResult.CONTRADICTED}
            and not self.evidence
        ):
            raise ValueError("Support or contradiction needs quoted opinion evidence")
        return self


class PaginatedSupportResponse(UnpaginatedSupportResponse):
    evidence: tuple[PageEvidenceQuote, ...] = Field(
        description="Up to three independent exact source quotes. Each item should contain one "
        "representative sentence or short fragment, at most 1200 characters, with a footnote label only when established. "
        "The items need not be adjacent; put the explanation in reason.",
        json_schema_extra={"maxItems": MAX_EVIDENCE_QUOTES},
    )
