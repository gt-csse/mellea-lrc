"""Field selection fails before work when inference and selected sources disagree."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.field_extraction import ml_cite

_PREFIX = "MELLEA_LRC_FIELD_EXTRACTION_"
_RULE = {
    "ML_ENABLED": "false",
    "CASE_NAME": "rule",
    "COURT": "rule",
    "DATE": "rule",
    "PIN_CITE": "rule",
}
_CHECKPOINT = {
    "MODEL_ID": "gt-csse/ML-cite",
    "REVISION": "881db5474fafe756474034ef6bf9c86a39e9630c",
    "VARIANT": "ML-cite-Bert-augmented",
    "DEVICE": "cpu",
    "CONTEXT_BEFORE": "260",
    "CONTEXT_AFTER": "260",
    "BATCH_SIZE": "2",
}


def _env(values):
    return {_PREFIX + key: value for key, value in values.items()}


def _config(**changes):
    return FieldExtractionConfig.from_env(environ=_env(_RULE | changes))


def test_all_rule_sources_need_no_checkpoint_and_do_not_load_optional_runtime(monkeypatch):
    config = _config()
    assert config == FieldExtractionConfig(
        ml_enabled=False, case_name="rule", court="rule", date="rule", pin_cite="rule"
    )
    monkeypatch.setattr(ml_cite, "get_ml_reader", lambda _: pytest.fail("rule preflight loaded ML"))
    config.validate_runtime()
    code = (
        "from mellea_lrc.config.field_extraction import FieldExtractionConfig; import sys; "
        "c = FieldExtractionConfig(ml_enabled=False, case_name='rule', court='rule', "
        "date='rule', pin_cite='rule'); c.validate_runtime(); "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("key", list(_RULE))
def test_every_policy_setting_is_required_without_code_defaults(key):
    values = _env({name: value for name, value in _RULE.items() if name != key})
    with pytest.raises(RuntimeError, match=_PREFIX + key):
        FieldExtractionConfig.from_env(environ=values)


@pytest.mark.parametrize("value", ["True", "False", "1", "0", "yes", "no", "ml", "rule"])
def test_enabled_env_is_explicit_lowercase_boolean(value):
    with pytest.raises(ValueError, match="must be exactly true or false"):
        _config(ML_ENABLED=value)


@pytest.mark.parametrize("field", ["case_name", "court", "date", "pin_cite"])
def test_ml_source_is_rejected_if_inference_is_disabled(field):
    with pytest.raises(ValidationError, match="ML_ENABLED=true"):
        _config(**{field.upper(): "ml"})


@pytest.mark.parametrize("source", ["bert", "ml_cite_bert", "ML", "rules", True, 1])
def test_field_sources_do_not_accept_alternative_names_or_types(source):
    with pytest.raises(ValidationError):
        FieldExtractionConfig(ml_enabled=False, case_name=source, court="rule", date="rule", pin_cite="rule")


@pytest.mark.parametrize("enabled", ["true", 1, 0, None])
def test_direct_configuration_also_requires_strict_boolean(enabled):
    with pytest.raises(ValidationError):
        FieldExtractionConfig(
            ml_enabled=enabled, case_name="rule", court="rule", date="rule", pin_cite="rule"
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
        FieldExtractionConfig(ml_enabled=True, case_name="rule", court="rule", date="rule", pin_cite="rule")


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
    assert config.case_name == config.court == config.date == config.pin_cite == "rule"


def test_mixed_field_selection_is_valid_and_preflight_load_errors_propagate(monkeypatch):
    config = _config(**(_CHECKPOINT | {"ML_ENABLED": "true", "CASE_NAME": "ml", "DATE": "ml"}))
    assert config.case_name == config.date == "ml"
    assert config.court == config.pin_cite == "rule"

    def broken(_):
        raise RuntimeError("checkpoint cannot load on selected device")

    monkeypatch.setattr(ml_cite, "get_ml_reader", broken)
    with pytest.raises(RuntimeError, match="selected device"):
        config.validate_runtime()


def test_removed_backend_is_not_a_supported_config_alias():
    with pytest.raises(ValidationError, match="backend"):
        FieldExtractionConfig(
            ml_enabled=False, case_name="rule", court="rule", date="rule", pin_cite="rule", backend="rule"
        )
