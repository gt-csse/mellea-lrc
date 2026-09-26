"""Provider-facing schema checks for reporter review decisions."""

import pytest
from pydantic import BaseModel

from mellea_lrc.model.citations.reporter_lookup import (
    ReporterAmbiguousReviewDecision,
    ReporterUniqueReviewDecision,
)


@pytest.mark.parametrize(
    "decision_type",
    (ReporterUniqueReviewDecision, ReporterAmbiguousReviewDecision),
)
def test_nested_case_name_schemas_are_strict_compatible(decision_type: type[BaseModel]) -> None:
    schema = decision_type.model_json_schema()
    assert schema["properties"]["case_name"]["$ref"] == "#/$defs/ReporterCaseNameAssessment"

    assessment = schema["$defs"]["ReporterCaseNameAssessment"]
    assert {"$ref": "#/$defs/CaseName"} in assessment["properties"]["normalized"]["anyOf"]

    for definition_name in ("ReporterCaseNameAssessment", "CaseName"):
        definition = schema["$defs"][definition_name]
        assert definition["type"] == "object"
        assert definition["additionalProperties"] is False
        assert set(definition["required"]) == set(definition["properties"]), definition_name
        assert all(
            "default" not in property_schema for property_schema in definition["properties"].values()
        ), definition_name
