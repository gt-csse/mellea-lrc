"""A saved source-only rule prediction at one citation decision node."""

from pydantic import BaseModel, ConfigDict, Field

from mellea_lrc.field_extraction.models import FieldPrediction


class RuleFieldReading(BaseModel):
    """Raw spans and the bounded reading policy that produced them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    prediction: FieldPrediction
    case_name_window: int = Field(ge=1, strict=True)
    post_locator_window: int = Field(ge=1, strict=True)
