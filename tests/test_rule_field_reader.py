"""Both readers consume a prepared context; rule reading never discovers one."""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

from mellea_lrc.field_extraction import LocatorContext, MLCiteReader
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


def context(source: str, locator_quote: str, *, kind="reporter", **changes) -> LocatorContext:
    anchor = span(source, locator_quote)
    values = dict(
        text=source,
        source_span=Span(0, len(source)),
        locator=anchor,
        kind=kind,
        name_before=Span(0, anchor.start),
        court_date_after=Span(anchor.end, len(source)),
        pin_region=anchor if kind == "short_reporter" else Span(anchor.end, len(source)),
    )
    return LocatorContext(**(values | changes))


def quotes(source: str, prediction: FieldPrediction) -> dict[str, str | None]:
    return {
        field: source[value.start : value.end] if (value := getattr(prediction, field)) is not None else None
        for field in ("case_name", "court", "date", "pin_cite")
    }


@pytest.mark.parametrize(
    ("source", "locator", "kind", "expected"),
    [
        (
            "Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).",
            "556 U.S. 662",
            "reporter",
            {"case_name": "Ashcroft v. Iqbal", "court": None, "date": "2009", "pin_cite": "678"},
        ),
        (
            "Alpha v. Beta, 123 F.3d 456, 459–60 (2d Cir. Mar. 4, 2000).",
            "123 F.3d 456",
            "reporter",
            {"case_name": "Alpha v. Beta", "court": "2d Cir.", "date": "Mar. 4, 2000", "pin_cite": "459–60"},
        ),
        (
            "Smith v. Jones, No. 12-cv-1234, at *5 (S.D.N.Y. Jan. 2, 2020).",
            "No. 12-cv-1234",
            "docket",
            {"case_name": "Smith v. Jones", "court": "S.D.N.Y.", "date": "Jan. 2, 2020", "pin_cite": "*5"},
        ),
        (
            "Iqbal, 556 U.S. at 678–79 (S.D.N.Y. 2009).",
            "556 U.S. at 678–79",
            "short_reporter",
            {"case_name": "Iqbal", "court": None, "date": None, "pin_cite": "678–79"},
        ),
    ],
)
def test_rule_predicts_exact_source_spans(source, locator, kind, expected):
    prepared = context(source, locator, kind=kind)
    prediction = RuleFieldReader().predict(prepared)
    assert prediction.context_span == prepared.source_span
    assert quotes(source, prediction) == expected
    prediction.validate_source(source, prepared.locator)
    assert FieldPrediction.model_validate_json(prediction.model_dump_json()) == prediction


def test_rule_and_ml_share_the_same_four_field_contract():
    source = "Alpha, 123 F.3d 456 (2000)."
    raw = RuleFieldReader().predict(context(source, "123 F.3d 456"))
    model = MLFieldPrediction(**raw.model_dump(), model_id="org/model", revision="revision", variant="joint")
    assert isinstance(model, FieldPrediction)
    assert quotes(source, raw) == quotes(source, model)


def test_batch_preserves_prepared_input_order():
    source = "Alpha v. Beta, 123 F.3d 456, 459 (2000). Gamma v. Delta, 456 F.3d 789 (2001)."
    first, second = context(source, "123 F.3d 456"), context(source, "456 F.3d 789")
    reader = RuleFieldReader()
    assert reader.predict_many((second, first)) == (reader.predict(second), reader.predict(first))
    assert reader.predict_many(()) == ()


@pytest.mark.parametrize("anchor", [Span(0, 0), Span(0, 3), Span(4, 8)])
def test_invalid_locator_raises_before_reading(anchor):
    with pytest.raises(ValueError):
        context("   x", "x", locator=anchor)


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


def test_rule_import_does_not_load_pipeline_ml_or_locator_discovery_modules():
    code = """
import sys
from mellea_lrc.field_extraction import LocatorContext, RuleFieldReader, Span
prepared = LocatorContext(text='Alpha, No. 12 (2000)', source_span=Span(0, 20),
    locator=Span(7, 13), kind='docket', name_before=Span(0, 7),
    court_date_after=Span(13, 20), pin_region=Span(13, 20))
RuleFieldReader().predict(prepared)
for prefix in ('mellea_lrc.model', 'mellea_lrc.extraction', 'mellea_lrc.config',
               'mellea_lrc.parsing.dockets', 'mellea_lrc.parsing.reporter',
               'torch', 'transformers', 'mellea', 'openai'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_public_readers_have_only_one_identical_context_input():
    for method, expected in (("predict", ("self", "context")), ("predict_many", ("self", "contexts"))):
        rule = inspect.signature(getattr(RuleFieldReader, method)).parameters
        ml = inspect.signature(getattr(MLCiteReader, method)).parameters
        assert tuple(rule) == tuple(ml) == expected
        assert [(item.kind, item.default) for item in rule.values()] == [
            (item.kind, item.default) for item in ml.values()
        ]
    assert not inspect.signature(RuleFieldReader).parameters


@pytest.mark.parametrize("docket", ["No. 05-cv-4206", "No. 05-4206"])
def test_docket_and_reporter_use_explicit_shared_boundaries(docket):
    source = f"Alpha v. Beta, {docket}, 2006 WL 1984362, at *1 (E.D. La. July 13, 2006)."
    docket_span, reporter = span(source, docket), span(source, "2006 WL 1984362")
    contexts = (
        context(
            source,
            docket,
            kind="docket",
            court_date_after=Span(reporter.end, len(source)),
            pin_region=Span(docket_span.end, reporter.start),
        ),
        context(source, "2006 WL 1984362", name_before=Span(0, docket_span.start)),
    )
    predictions = RuleFieldReader().predict_many(contexts)
    assert [quotes(source, prediction) for prediction in predictions] == [
        {"case_name": "Alpha v. Beta", "court": "E.D. La.", "date": "July 13, 2006", "pin_cite": None},
        {"case_name": "Alpha v. Beta", "court": "E.D. La.", "date": "July 13, 2006", "pin_cite": "*1"},
    ]


def test_parallel_reporters_use_shared_group_name_court_and_date_but_local_pins():
    source = "Alpha v. Beta, 123 F.3d 456, 459, 789 F.2d 10, 12 (2d Cir. 2000)."
    first, second = span(source, "123 F.3d 456"), span(source, "789 F.2d 10")
    contexts = (
        context(
            source,
            "123 F.3d 456",
            court_date_after=Span(second.end, len(source)),
            pin_region=Span(first.end, second.start),
        ),
        context(source, "789 F.2d 10", name_before=Span(0, first.start)),
    )
    assert [quotes(source, item) for item in RuleFieldReader().predict_many(contexts)] == [
        {"case_name": "Alpha v. Beta", "court": "2d Cir.", "date": "2000", "pin_cite": "459"},
        {"case_name": "Alpha v. Beta", "court": "2d Cir.", "date": "2000", "pin_cite": "12"},
    ]


def test_explicit_short_context_preserves_aliases_and_parallel_prefix():
    source = "The Court explained Unknown, 347 U.S. at 495, 74 S. Ct. at 690."
    first = span(source, "347 U.S. at 495")
    contexts = tuple(
        context(
            source, locator, kind="short_reporter", name_before=Span(0, first.start), known_names=("Unknown",)
        )
        for locator in ("347 U.S. at 495", "74 S. Ct. at 690")
    )
    assert [quotes(source, item) for item in RuleFieldReader().predict_many(contexts)] == [
        {"case_name": "Unknown", "court": None, "date": None, "pin_cite": "495"},
        {"case_name": "Unknown", "court": None, "date": None, "pin_cite": "690"},
    ]
    assert LocatorContext.model_validate_json(contexts[0].model_dump_json()) == contexts[0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("name_before", Span(0, 30)),
        ("court_date_after", Span(0, 5)),
        ("pin_region", Span(0, 5)),
        ("court_date_after", Span(25, 100)),
        ("source_span", Span(0, 1)),
        ("known_names", (" ",)),
    ],
)
def test_impossible_context_is_rejected(field, value):
    with pytest.raises(ValueError):
        context("Alpha, 123 F.3d 456 (2000).", "123 F.3d 456", **{field: value})


def test_excerpt_predictions_preserve_absolute_offsets():
    source = "outside Ω😊 | Alpha v. Beta, 123 F.3d 456, 459 (2000). | outside"
    left, right = source.index("Alpha"), source.index(" | outside")
    locator = span(source, "123 F.3d 456")
    prepared = LocatorContext(
        text=source[left:right],
        source_span=Span(left, right),
        locator=locator,
        kind="reporter",
        name_before=Span(left, locator.start),
        court_date_after=Span(locator.end, right),
        pin_region=Span(locator.end, right),
    )
    prediction = RuleFieldReader().predict(prepared)
    assert prediction.context_span == Span(left, right)
    assert quotes(source, prediction) == {
        "case_name": "Alpha v. Beta",
        "court": None,
        "date": "2000",
        "pin_cite": "459",
    }


def test_readers_reject_legacy_inputs_and_non_context_batches():
    reader = RuleFieldReader()
    with pytest.raises(TypeError):
        reader.predict("1 U.S. 2", Span(0, 8))
    with pytest.raises(TypeError, match="LocatorContexts"):
        reader.predict_many(("1 U.S. 2",))
    with pytest.raises(TypeError, match="LocatorContexts"):
        reader.predict_many([])
