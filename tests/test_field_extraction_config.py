"""Field selection fails before work when inference and selected sources disagree."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from mellea_lrc.config.field_extraction import FieldExtractionConfig, FieldSource
from mellea_lrc.field_extraction import ml_cite

_PREFIX = "MELLEA_LRC_FIELD_EXTRACTION_"
_RULE = {
    "ML_ENABLED": "false",
    "CASE_NAME": "RULE",
    "COURT": "RULE",
    "DATE": "RULE",
    "PIN_CITE": "RULE",
    "CONTEXT_BEFORE": "260",
    "CONTEXT_AFTER": "260",
}
_CHECKPOINT = {
    "MODEL_ID": "gt-csse/ML-cite",
    "REVISION": "881db5474fafe756474034ef6bf9c86a39e9630c",
    "VARIANT": "ML-cite-Bert-augmented",
    "DEVICE": "cpu",
    "BATCH_SIZE": "2",
}


def _env(values):
    return {_PREFIX + key: value for key, value in values.items()}


def _config(**changes):
    return FieldExtractionConfig.from_env(environ=_env(_RULE | changes))


def test_all_rule_sources_need_no_checkpoint_and_do_not_load_optional_runtime(monkeypatch):
    config = _config()
    assert config == FieldExtractionConfig(
        context_before=260,
        context_after=260,
        ml_enabled=False,
        case_name="RULE",
        court="RULE",
        date="RULE",
        pin_cite="RULE",
    )
    monkeypatch.setattr(ml_cite, "get_ml_reader", lambda _: pytest.fail("rule preflight loaded ML"))
    config.validate_runtime()
    code = (
        "from mellea_lrc.config.field_extraction import FieldExtractionConfig; import sys; "
        "c = FieldExtractionConfig(context_before=260, context_after=260, ml_enabled=False, case_name='RULE', court='RULE', "
        "date='RULE', pin_cite='RULE'); c.validate_runtime(); "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("key", list(_RULE))
def test_every_policy_setting_is_required_without_code_defaults(key):
    values = _env({name: value for name, value in _RULE.items() if name != key})
    with pytest.raises(RuntimeError, match=_PREFIX + key):
        FieldExtractionConfig.from_env(environ=values)


@pytest.mark.parametrize("value", ["True", "False", "1", "0", "yes", "no", "ML", "RULE"])
def test_enabled_env_is_explicit_lowercase_boolean(value):
    with pytest.raises(ValueError, match="must be exactly true or false"):
        _config(ML_ENABLED=value)


@pytest.mark.parametrize("field", ["case_name", "court", "date", "pin_cite"])
def test_ml_source_is_rejected_if_inference_is_disabled(field):
    with pytest.raises(ValidationError, match="ML_ENABLED=true"):
        _config(**{field.upper(): "ML"})


@pytest.mark.parametrize("source", ["rule", "ml", "Rule", "Ml", "bert", "ml_cite_bert", "rules", True, 1])
def test_field_sources_do_not_accept_alternative_names_or_types(source):
    with pytest.raises(ValidationError):
        FieldExtractionConfig(
            context_before=260,
            context_after=260,
            ml_enabled=False,
            case_name=source,
            court="RULE",
            date="RULE",
            pin_cite="RULE",
        )


@pytest.mark.parametrize("field", ["CASE_NAME", "COURT", "DATE", "PIN_CITE"])
@pytest.mark.parametrize("source", ["rule", "ml", "Rule", "Ml"])
def test_each_env_field_rejects_lowercase_and_mixed_case(field, source):
    with pytest.raises(ValidationError, match=field.lower()):
        _config(**{field: source})


def test_field_source_enum_and_native_json_have_the_same_explicit_values():
    config = _config()
    assert config.case_name is config.court is config.date is config.pin_cite is FieldSource.RULE
    assert FieldExtractionConfig.model_validate_json(config.model_dump_json()) == config
    assert config.model_dump(mode="json")["case_name"] == "RULE"


@pytest.mark.parametrize("enabled", ["true", 1, 0, None])
def test_direct_configuration_also_requires_strict_boolean(enabled):
    with pytest.raises(ValidationError):
        FieldExtractionConfig(
            context_before=260,
            context_after=260,
            ml_enabled=enabled,
            case_name="RULE",
            court="RULE",
            date="RULE",
            pin_cite="RULE",
        )


@pytest.mark.parametrize("key", list(_CHECKPOINT))
def test_enabling_inference_requires_every_checkpoint_setting(key):
    values = _env(
        _RULE | {"ML_ENABLED": "true"} | {name: value for name, value in _CHECKPOINT.items() if name != key}
    )
    with pytest.raises(RuntimeError, match=_PREFIX + key):
        FieldExtractionConfig.from_env(environ=values)


def test_direct_enabled_configuration_requires_complete_checkpoint():
    with pytest.raises(ValidationError, match="ML field extraction requires"):
        FieldExtractionConfig(
            context_before=260,
            context_after=260,
            ml_enabled=True,
            case_name="RULE",
            court="RULE",
            date="RULE",
            pin_cite="RULE",
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"REVISION": "main"},
        {"BATCH_SIZE": "0"},
        {"CONTEXT_BEFORE": "-1"},
    ],
)
def test_checkpoint_constraints_are_checked_before_runtime(changes):
    with pytest.raises(ValidationError):
        _config(**(_CHECKPOINT | {"ML_ENABLED": "true"} | changes))


def test_enabled_inference_can_be_kept_while_each_field_selects_rules(monkeypatch):
    config = _config(**(_CHECKPOINT | {"ML_ENABLED": "true"}))
    calls = []
    monkeypatch.setattr(ml_cite, "get_ml_reader", lambda selected: calls.append(selected))
    config.validate_runtime()
    assert calls == [config.ml_reader_config()]
    assert config.case_name == config.court == config.date == config.pin_cite == "RULE"


def test_mixed_field_selection_is_valid_and_preflight_load_errors_propagate(monkeypatch):
    config = _config(**(_CHECKPOINT | {"ML_ENABLED": "true", "CASE_NAME": "ML", "DATE": "ML"}))
    assert config.case_name == config.date == "ML"
    assert config.court == config.pin_cite == "RULE"

    def broken(_):
        raise RuntimeError("checkpoint cannot load on selected device")

    monkeypatch.setattr(ml_cite, "get_ml_reader", broken)
    with pytest.raises(RuntimeError, match="selected device"):
        config.validate_runtime()


def test_removed_backend_is_not_a_supported_config_alias():
    with pytest.raises(ValidationError, match="backend"):
        FieldExtractionConfig(
            context_before=260,
            context_after=260,
            ml_enabled=False,
            case_name="RULE",
            court="RULE",
            date="RULE",
            pin_cite="RULE",
            backend="RULE",
        )


@pytest.mark.parametrize("key", ["CONTEXT_BEFORE", "CONTEXT_AFTER"])
def test_shared_context_windows_are_required_even_when_ml_is_disabled(key):
    values = _env({name: value for name, value in _RULE.items() if name != key})
    with pytest.raises(RuntimeError, match=_PREFIX + key):
        FieldExtractionConfig.from_env(environ=values)


def test_ml_reader_configuration_has_no_independent_context_window():
    config = _config(**(_CHECKPOINT | {"ML_ENABLED": "true"}))
    reader_config = config.ml_reader_config()
    assert not hasattr(reader_config, "context_before")
    assert not hasattr(reader_config, "context_after")
    assert config.context_before == config.context_after == 260
