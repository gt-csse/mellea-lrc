"""The model prediction recorded at a citation decision node."""

from pydantic import BaseModel, ConfigDict

from mellea_lrc.field_extraction.models import MLFieldPrediction


class MLFieldReading(BaseModel):
    """A single model invocation, separate from normalized field histories."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    prediction: MLFieldPrediction
