"""The reusable rule baseline uses only source text and a target locator."""

from __future__ import annotations

import subprocess
import sys

import pytest

from mellea_lrc.field_extraction.models import FieldPrediction, MLFieldPrediction
from mellea_lrc.field_extraction.rule import (
    RuleFieldReader,
    read_case_name,
    read_court_and_date,
    read_pin_cite,
    read_short_case_name,
)
from mellea_lrc.parsing.case_names import CaseName, CaseNameKind
from mellea_lrc.parsing.span import Span


def span(source: str, quote: str) -> Span:
    start = source.index(quote)
    return Span(start, start + len(quote))


def quotes(source: str, prediction: FieldPrediction) -> dict[str, str | None]:
    return {
        field: source[value.start : value.end] if (value := getattr(prediction, field)) is not None else None
        for field in ("case_name", "court", "date", "pin_cite")
    }


@pytest.mark.parametrize(
    ("source", "locator", "expected"),
    [
        (
            "Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).",
            "556 U.S. 662",
            {"case_name": "Ashcroft v. Iqbal", "court": None, "date": "2009", "pin_cite": "678"},
        ),
        (
            "Alpha v. Beta, 123 F.3d 456, 459–60 (2d Cir. Mar. 4, 2000).",
            "123 F.3d 456",
            {"case_name": "Alpha v. Beta", "court": "2d Cir.", "date": "Mar. 4, 2000", "pin_cite": "459–60"},
        ),
        (
            "Smith v. Jones, No. 12-cv-1234, at *5 (S.D.N.Y. Jan. 2, 2020).",
            "No. 12-cv-1234",
            {"case_name": "Smith v. Jones", "court": "S.D.N.Y.", "date": "Jan. 2, 2020", "pin_cite": "*5"},
        ),
        (
            "Iqbal, 556 U.S. at 678–79 (S.D.N.Y. 2009).",
            "556 U.S. at 678–79",
            {"case_name": "Iqbal", "court": None, "date": None, "pin_cite": "678–79"},
        ),
    ],
)
def test_rule_predicts_exact_source_spans(source, locator, expected):
    prediction = RuleFieldReader().predict(source, span(source, locator))
    assert prediction.context_span == Span(0, len(source))
    assert quotes(source, prediction) == expected
    prediction.validate_source(source, span(source, locator))
    assert FieldPrediction.model_validate_json(prediction.model_dump_json()) == prediction


def test_rule_and_ml_share_the_same_four_field_contract():
    source = "Alpha, 123 F.3d 456 (2000)."
    raw = RuleFieldReader().predict(source, span(source, "123 F.3d 456"))
    model = MLFieldPrediction(**raw.model_dump(), model_id="org/model", revision="revision", variant="joint")
    assert isinstance(model, FieldPrediction)
    assert quotes(source, raw) == quotes(source, model)
    model.validate_source(source, span(source, "123 F.3d 456"))


def test_batch_preserves_input_order_and_never_reads_annotation():
    source = "Alpha v. Beta, 123 F.3d 456, 459 (2000). Gamma v. Delta, 456 F.3d 789 (2001)."
    first, second = span(source, "123 F.3d 456"), span(source, "456 F.3d 789")
    reader = RuleFieldReader()
    batch = reader.predict_many(source, (second, first))
    assert batch == (reader.predict(source, second), reader.predict(source, first))
    assert reader.predict_many(source, ()) == ()


@pytest.mark.parametrize("locator", [Span(0, 0), Span(0, 3), Span(4, 8)])
def test_invalid_locator_raises_before_parsing(locator):
    with pytest.raises(ValueError, match="Target locator"):
        RuleFieldReader().predict("   x", locator)


def test_invalid_prediction_cannot_silently_drop_a_field():
    with pytest.raises(ValueError, match="nonempty and inside"):
        FieldPrediction(context_span=Span(0, 3), case_name=Span(3, 4))
    with pytest.raises(ValueError, match="nonempty and inside"):
        FieldPrediction(context_span=Span(0, 3), court=Span(1, 1))
    with pytest.raises(ValueError, match="contain the target locator"):
        FieldPrediction(context_span=Span(0, 3)).validate_source("source", Span(3, 4))
    with pytest.raises(ValueError, match="blank"):
        FieldPrediction(context_span=Span(0, 3), case_name=Span(0, 1)).validate_source(" xz", Span(1, 2))


def test_bounded_helpers_keep_absolute_offsets_and_existing_rules():
    assert read_case_name("See Alpha v. Beta, ", 100, None) == Span(104, 117)
    suffix = ", 50 (S.D.N.Y. filed Jan. 2, 2020)"
    court, date = read_court_and_date(suffix, 200)
    assert court == Span(206, 214)
    assert date == Span(221, 233)
    assert read_court_and_date(". Next sentence (2000)", 0) == (None, None)
    assert read_pin_cite(", 459 n. X (2000)", 0, 16) == Span(2, 10)
    assert read_pin_cite(", 2d Cir.", 0, 9) is None
    assert read_pin_cite(", 12: prose", 0, 11) is None
    assert read_short_case_name("See Alpha, ", 50) == Span(54, 59)
    assert read_short_case_name("The Court explained Alpha, ", 0, ("Alpha",)) == Span(20, 25)


def test_case_name_normalizer_stays_strict_and_schema_keeps_nullable_parts():
    assert CaseName.from_quote("In re\nAlpha").kind is CaseNameKind.IN_RE
    assert CaseName.from_quote("Alpha v. Beta").plaintiff == "Alpha"
    assert set(CaseName.model_json_schema()["required"]) == {
        "kind",
        "plaintiff",
        "defendant",
        "subject",
        "partial",
    }
    with pytest.raises(ValueError):
        CaseName.from_quote("Alpha v.")


def test_rule_import_does_not_load_pipeline_or_ml_modules():
    code = """
import sys
from mellea_lrc.field_extraction.rule import RuleFieldReader
from mellea_lrc.parsing.span import Span
RuleFieldReader().predict('Alpha, No. 12 (2000)', Span(7, 13))
for prefix in ('mellea_lrc.model', 'mellea_lrc.extraction', 'mellea_lrc.config', 'torch', 'transformers', 'mellea', 'openai'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    subprocess.run([sys.executable, "-c", code], check=True)
