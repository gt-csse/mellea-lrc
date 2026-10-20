"""Model packages and substage bindings are configured only through .env."""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from mellea.backends import ModelOption

from mellea_lrc.llm import profiles as profile_config
from mellea_lrc.llm import reviewer as reviewer_runtime
from mellea_lrc.llm.config import (
    OUTPUT_MODE_OPTION,
    PROFILE_OPTION,
    LlmApiConfig,
    LlmOutputMode,
    start_mellea_session,
)
from mellea_lrc.llm.profiles import LlmProfile, load_profile


def _profile(**changes: object) -> LlmProfile:
    settings = dict(
        name="test_profile",
        model="test-model",
        api_base="https://provider.invalid/v1",
        api_key_env="TEST_MODEL_KEY",
        temperature=0.0,
        timeout_seconds=300.0,
        max_tokens=8000,
        max_attempts=3,
        output_mode=LlmOutputMode.JSON_SCHEMA,
    )
    return LlmProfile(**(settings | changes))


_PROFILE_SETTINGS = {
    "MODEL": "test-model",
    "API_BASE": "https://provider.invalid/v1",
    "API_KEY_ENV": "TEST_MODEL_KEY",
    "TEMPERATURE": "0.25",
    "TIMEOUT_SECONDS": "45",
    "MAX_TOKENS": "4321",
    "MAX_ATTEMPTS": "2",
    "OUTPUT_MODE": "prompt",
    "REASONING_EFFORT": "low",
    "SERVICE_TIER": "",
}
_SUBSTAGE = "validate_all_citation_content.review_citation_content.review_cited_pages_llm"
_BINDING = (
    "MELLEA_LRC_SUBSTAGE_VALIDATE_ALL_CITATION_CONTENT_REVIEW_CITATION_CONTENT_REVIEW_CITED_PAGES_LLM_PROFILE"
)
_PREFIX = "MELLEA_LRC_PROFILE_TEST_PROFILE_"


def _environment():
    return {_BINDING: "test_profile", **{_PREFIX + key: value for key, value in _PROFILE_SETTINGS.items()}}


def test_env_selects_a_complete_package_without_resolving_credentials():
    values = _environment()
    profile = load_profile(_SUBSTAGE, environ=values)
    assert profile == _profile(
        temperature=0.25,
        timeout_seconds=45,
        max_tokens=4321,
        max_attempts=2,
        output_mode=LlmOutputMode.PROMPT,
        reasoning_effort="low",
    )
    with pytest.raises(RuntimeError, match="TEST_MODEL_KEY"):
        profile.resolve(values)
    config = profile.resolve(values | {"TEST_MODEL_KEY": "secret"})
    assert config.profile_name == "test_profile"
    assert "secret" not in repr(profile) and "secret" not in repr(config)


def test_missing_substage_assignment_does_not_use_a_default_profile():
    values = _environment()
    del values[_BINDING]
    with pytest.raises(RuntimeError, match=_BINDING):
        load_profile(_SUBSTAGE, environ=values)


@pytest.mark.parametrize(
    "substage,old_binding",
    [
        (
            "grow_roots.discover_root_locators.hunt_docket_locators",
            "MELLEA_LRC_SUBSTAGE_GROW_ROOTS_LOCATOR_DISCOVERY_DOCKET_HUNTING_PROFILE",
        ),
        (
            "grow_leaves.grow_id_leaves.attribute_citations",
            "MELLEA_LRC_SUBSTAGE_GROW_LEAVES_ID_CITATIONS_ATTRIBUTION_PROFILE",
        ),
        (
            "validate_roots_citation_identity.lookup_reporter_roots.review_unique_llm",
            "MELLEA_LRC_SUBSTAGE_VALIDATE_ROOTS_REPORTER_LOOKUP_UNIQUE_LLM_JUDGMENT_PROFILE",
        ),
        (
            "validate_all_citation_content.review_citation_content.review_cited_pages_llm",
            "MELLEA_LRC_SUBSTAGE_VALIDATE_PINCITE_SUPPORT_REVIEW_PAGE_REVIEW_PROFILE",
        ),
    ],
)
def test_renamed_substage_profiles_require_new_binding_without_old_key_fallback(substage, old_binding):
    binding = f"MELLEA_LRC_SUBSTAGE_{substage.upper().replace('.', '_')}_PROFILE"
    values = {
        old_binding: "test_profile",
        **{_PREFIX + key: value for key, value in _PROFILE_SETTINGS.items()},
    }
    with pytest.raises(RuntimeError, match=binding):
        load_profile(substage, environ=values)
    values[binding] = "test_profile"
    assert load_profile(substage, environ=values).name == "test_profile"


@pytest.mark.parametrize(
    "setting", [key for key in _PROFILE_SETTINGS if key not in {"REASONING_EFFORT", "SERVICE_TIER"}]
)
@pytest.mark.parametrize("missing_value", [None, "", "   "])
def test_all_required_profile_settings_must_be_explicit(setting, missing_value):
    values = _environment()
    values[_PREFIX + setting] = missing_value
    with pytest.raises(RuntimeError, match=_PREFIX + setting):
        load_profile(_SUBSTAGE, environ=values)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("TEMPERATURE", "nan"),
        ("TEMPERATURE", "-1"),
        ("TEMPERATURE", "automatic"),
        ("TIMEOUT_SECONDS", "0"),
        ("TIMEOUT_SECONDS", "inf"),
        ("MAX_TOKENS", "1.5"),
        ("MAX_TOKENS", "0"),
        ("MAX_ATTEMPTS", "-1"),
        ("OUTPUT_MODE", "automatic"),
        ("REASONING_EFFORT", "automatic"),
        ("SERVICE_TIER", "rush"),
    ],
)
def test_invalid_env_settings_raise_instead_of_using_defaults(setting, value):
    with pytest.raises(ValueError, match=r"Invalid \.env profile test_profile"):
        load_profile(_SUBSTAGE, environ=_environment() | {_PREFIX + setting: value})


def test_blank_or_omitted_optional_settings_are_provider_defaults():
    values = _environment()
    for setting in ("REASONING_EFFORT", "SERVICE_TIER"):
        values[_PREFIX + setting] = "  "
    profile = load_profile(_SUBSTAGE, environ=values)
    assert profile.reasoning_effort is profile.service_tier is None
    for setting in ("REASONING_EFFORT", "SERVICE_TIER"):
        del values[_PREFIX + setting]
    assert load_profile(_SUBSTAGE, environ=values) == profile


@pytest.mark.parametrize("name", ["", "a-b", "a b", "a.b", "café", None])
def test_invalid_profile_names_raise(name):
    with pytest.raises(ValueError, match="Profile names"):
        LlmProfile.from_env(name, environ={})


def test_env_file_is_authoritative_and_reloaded_for_the_next_binding(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(_PREFIX + "MODEL", "ambient-model")
    path = tmp_path / ".env"
    values = _environment() | {"TEST_MODEL_KEY": "first-secret"}
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()))
    first = load_profile(_SUBSTAGE)
    assert first.model == "test-model"
    assert first.resolve().api_key == "first-secret"
    values[_PREFIX + "MODEL"] = "second-model"
    values["TEST_MODEL_KEY"] = "second-secret"
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()))
    second = load_profile(_SUBSTAGE)
    assert second.model == "second-model" and second.resolve().api_key == "second-secret"
    assert first.model == "test-model"
    assert __import__("os").environ[_PREFIX + "MODEL"] == "ambient-model"


def test_missing_env_file_raises_without_using_process_model_settings(monkeypatch):
    from mellea_lrc import configuration

    monkeypatch.setattr(configuration, "find_dotenv", lambda **kwargs: "")
    for key, value in _environment().items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match=r"No \.env found"):
        load_profile(_SUBSTAGE)


def test_example_defines_all_model_substages_and_no_credentials():
    from dotenv import dotenv_values

    values = dotenv_values(Path(__file__).resolve().parents[1] / ".env.example", interpolate=False)
    for module_name in _SUBSTAGES:
        module = importlib.import_module(f"mellea_lrc.{module_name}")
        profile = load_profile(module.SUBSTAGE, environ=values)
        assert not values[profile.api_key_env]
    pages = load_profile(_SUBSTAGE, environ=values)
    full = load_profile(
        "validate_all_citation_content.review_citation_content.review_full_opinions_llm", environ=values
    )
    assert pages.name == full.name == "robspark_granite4_1_30b"
    assert pages.timeout_seconds == full.timeout_seconds == 300
    assert pages.max_tokens == full.max_tokens == 8000
    assert pages.reasoning_effort is full.reasoning_effort is None
    assert LlmProfile.from_env("nrp_glm", environ=values).max_tokens == 8000


def test_profile_is_a_complete_frozen_package_and_resolves_only_its_credential() -> None:
    profile = _profile()
    config = profile.resolve({"TEST_MODEL_KEY": "test-secret"})

    assert config.model == profile.model
    assert config.api_base == profile.api_base
    assert config.api_key == "test-secret"
    assert config.temperature == profile.temperature == 0
    assert config.timeout_seconds == profile.timeout_seconds == 300
    assert config.output_mode is profile.output_mode is LlmOutputMode.JSON_SCHEMA
    assert config.reasoning_effort == profile.reasoning_effort
    assert config.service_tier == profile.service_tier
    assert config.profile_name == profile.name
    assert profile.max_tokens == 8000
    assert profile.max_attempts == 3
    with pytest.raises(FrozenInstanceError):
        profile.model = "changed"


def test_environment_model_settings_cannot_create_a_mixed_profile() -> None:
    profile = _profile()
    config = profile.resolve(
        {
            "TEST_MODEL_KEY": "correct-secret",
            "MELLEA_LRC_LLM_MODEL": "unrelated-model",
            "MELLEA_LRC_LLM_API_BASE": "https://unrelated.invalid/v1",
            "MELLEA_LRC_LLM_API_KEY": "wrong-secret",
            "MELLEA_LRC_LLM_TEMPERATURE": "0.9",
            "MELLEA_LRC_LLM_VALIDATE_ALL_CITATION_CONTENT_MODEL": "workflow-model",
            "MELLEA_LRC_LLM_VALIDATE_ALL_CITATION_CONTENT_REASONING_EFFORT": "high",
        }
    )

    assert config == profile.resolve({"TEST_MODEL_KEY": "correct-secret"})


@pytest.mark.parametrize("environ", [{}, {"TEST_MODEL_KEY": ""}, {"TEST_MODEL_KEY": "   "}])
def test_missing_profile_credential_raises_without_using_another_key(environ) -> None:
    with pytest.raises((RuntimeError, ValueError), match="TEST_MODEL_KEY"):
        _profile().resolve({"MELLEA_LRC_LLM_API_KEY": "unrelated-secret", **environ})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -1),
        ("temperature", float("nan")),
        ("temperature", True),
        ("timeout_seconds", 0),
        ("timeout_seconds", float("inf")),
        ("max_tokens", 0),
        ("max_tokens", -1),
        ("max_tokens", True),
        ("max_tokens", 1.5),
        ("max_attempts", 0),
        ("max_attempts", True),
        ("max_attempts", 1.5),
        ("reasoning_effort", "automatic"),
        ("service_tier", "rush"),
        ("output_mode", "automatic"),
    ],
)
def test_invalid_profile_settings_raise_instead_of_falling_back(field: str, value: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        replace(_profile(), **{field: value})


@pytest.mark.parametrize("field", ["name", "model", "api_base", "api_key_env"])
def test_profile_identifiers_must_be_nonempty(field: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        replace(_profile(), **{field: " "})


@pytest.mark.parametrize("mode", LlmOutputMode)
def test_profile_output_mode_is_explicit_and_reaches_ivr(mode: LlmOutputMode) -> None:
    config = _profile(output_mode=mode).resolve({"TEST_MODEL_KEY": "test-secret"})
    assert config.mellea_call_options(max_tokens=32)[OUTPUT_MODE_OPTION] is mode


def test_profile_call_options_include_only_safe_profile_metadata() -> None:
    config = _profile(reasoning_effort="low", service_tier="flex").resolve({"TEST_MODEL_KEY": "test-secret"})
    options = config.mellea_call_options(max_tokens=32)
    assert options[PROFILE_OPTION] == {
        "name": "test_profile",
        "endpoint": "https://provider.invalid/v1",
    }
    assert options[ModelOption.STREAM_TIMEOUT] is None
    assert options["temperature"] == 0
    assert options["max_tokens"] == 32
    assert options["reasoning_effort"] == "low"
    assert options["service_tier"] == "flex"
    assert "test-secret" not in repr(options)
    assert "TEST_MODEL_KEY" not in repr(options)


def test_unnamed_resolved_binding_has_no_profile_metadata_or_optional_provider_options() -> None:
    config = LlmApiConfig(
        model="test",
        api_base="https://provider.invalid/v1",
        api_key="test-secret",
        temperature=0,
        timeout_seconds=45,
        output_mode=LlmOutputMode.JSON_SCHEMA,
    )
    options = config.mellea_call_options(max_tokens=32)
    assert PROFILE_OPTION not in options
    assert "service_tier" not in options
    assert "reasoning_effort" not in options


def test_session_creation_uses_the_resolved_package_without_environment_reads(monkeypatch) -> None:
    import mellea
    import mellea.backends.openai

    profile = _profile(temperature=0.25, timeout_seconds=45, service_tier="priority")
    config = profile.resolve({"TEST_MODEL_KEY": "test-secret"})
    kwargs = {}
    backend = object()

    def make_backend(**options):
        kwargs.update(options)
        return backend

    monkeypatch.setattr(mellea.backends.openai, "OpenAIBackend", make_backend)
    monkeypatch.setattr(mellea, "MelleaSession", lambda value: value)
    assert start_mellea_session(config) is backend
    assert kwargs == {
        "model_id": profile.model,
        "base_url": profile.api_base,
        "api_key": "test-secret",
        "timeout": 45,
        "max_retries": 0,
        "model_options": {"temperature": 0.25, "service_tier": "priority"},
    }


def test_http_timeout_reaches_both_real_sdk_clients_without_stream_deadline(monkeypatch) -> None:
    import mellea.backends.openai as backend_module

    monkeypatch.setattr(backend_module, "is_vllm_server_with_structured_output", lambda **_kwargs: False)
    config = _profile().resolve({"TEST_MODEL_KEY": "offline-secret"})

    async def inspect_clients():
        session = start_mellea_session(config)
        try:
            for client in (session.backend._client, session.backend._async_client):
                assert client.timeout == 300
                assert client.max_retries == 0
            options = config.mellea_call_options(max_tokens=64)
            assert options[ModelOption.STREAM_TIMEOUT] is None
        finally:
            await session.backend._async_client.close()
            session.backend._client.close()

    asyncio.run(inspect_clients())


_REVIEWERS = (
    ("extraction.hunt_docket_locators.review", "IvrDocketReviewer"),
    ("extraction.reassign_docket_roots_llm.reviewer", "IvrDocketRootReviewer"),
    ("extraction.leaf_attribution_review.reviewer", "IvrLeafReviewer"),
    ("validation.review_unique_reporter_roots_llm.reviewer", "IvrReporterUniqueReviewer"),
    ("validation.review_ambiguous_reporter_roots_llm.reviewer", "IvrReporterAmbiguousReviewer"),
    ("validation.review_courtlistener_docket_roots_llm.reviewer", "IvrDocketLookupReviewer"),
    ("validation.review_govinfo_docket_roots_llm.reviewer", "IvrGovInfoDocketReviewer"),
    ("validation.review_locator_bodies_llm.reviewer", "IvrBodyCorroborationReviewer"),
    ("validation.review_intended_cases_llm.reviewer", "IvrIntendedCaseReviewer"),
    ("validation.select_reporter_citation_opinions.reviewer", "IvrReporterCitationOpinionReviewer"),
    ("validation.read_reporter_citation_propositions.reviewer", "IvrReporterCitationPropositionReviewer"),
    ("validation.reporter_pinpoint_review.reviewer", "IvrReporterPinpointReviewer"),
)


@pytest.mark.parametrize(("module_name", "class_name"), _REVIEWERS)
def test_reviewer_requires_a_profile_and_uses_its_session_options_and_budgets(
    monkeypatch, module_name: str, class_name: str
) -> None:
    module = importlib.import_module(f"mellea_lrc.{module_name}")
    factory = getattr(module, class_name)
    profile = _profile(
        output_mode=LlmOutputMode.PROMPT, reasoning_effort="low", max_tokens=321, max_attempts=2
    )
    config = profile.resolve({"TEST_MODEL_KEY": "test-secret"})
    resolved = []
    started = []
    session = object()

    def resolve(self):
        resolved.append(self)
        return config

    def start(binding):
        started.append(binding)
        return session

    monkeypatch.setattr(LlmProfile, "resolve", resolve)
    monkeypatch.setattr(reviewer_runtime, "start_mellea_session", start)
    with pytest.raises(TypeError):
        factory.from_profile()
    assert not hasattr(factory, "from_env")
    reviewer = factory.from_profile(profile)
    assert type(reviewer) is factory
    assert resolved == [profile]
    assert started == [config]
    assert reviewer.session is session
    assert reviewer.model_options["max_tokens"] == 321
    assert reviewer.max_attempts == 2
    assert reviewer.model_options[OUTPUT_MODE_OPTION] is LlmOutputMode.PROMPT
    assert reviewer.model_options["reasoning_effort"] == "low"
    assert reviewer.model_options[PROFILE_OPTION]["name"] == profile.name
    assert "extra_body" not in reviewer.model_options


_SUBSTAGES = (
    "extraction.hunt_docket_locators",
    "extraction.reassign_docket_roots_llm",
    "extraction.correct_leaf_fields",
    "extraction.attribute_short_reporter_citations",
    "extraction.attribute_reference_citations",
    "extraction.attribute_id_citations",
    "extraction.review_supra_attributions_llm",
    "validation.review_unique_reporter_roots_llm",
    "validation.review_ambiguous_reporter_roots_llm",
    "validation.review_courtlistener_docket_roots_llm",
    "validation.review_govinfo_docket_roots_llm",
    "validation.review_locator_bodies_llm",
    "validation.review_intended_cases_llm",
    "validation.select_reporter_citation_opinions",
    "validation.read_reporter_citation_propositions",
    "validation.review_reporter_citation_pages_llm",
    "validation.review_reporter_citation_full_opinions_llm",
)


@pytest.mark.parametrize("module_name", _SUBSTAGES)
def test_stage_passes_its_substage_to_the_lazy_profile_loader(monkeypatch, module_name: str) -> None:
    def unexpected_read():
        raise AssertionError("Importing a stage must not read .env")

    monkeypatch.setattr(profile_config, "read_env", unexpected_read)
    module = importlib.import_module(f"mellea_lrc.{module_name}")
    assert not hasattr(module, "MODEL_PROFILE")
    calls = [
        node
        for node in ast.walk(ast.parse(Path(module.__file__).read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_profile"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert not call.keywords and len(call.args) == 1
    loader = call.args[0]
    assert isinstance(loader, ast.Call) and isinstance(loader.func, ast.Name)
    assert loader.func.id == "load_profile" and not loader.keywords
    assert len(loader.args) == 1 and isinstance(loader.args[0], ast.Name)
    assert loader.args[0].id == "SUBSTAGE"


def test_pinpoint_workflow_passes_separate_page_and_full_opinion_reviewers(monkeypatch) -> None:
    from mellea_lrc.model import Document

    workflow = importlib.import_module("mellea_lrc.workflows.validate_all_citation_content")
    assert "support_reviewer" not in inspect.signature(workflow.validate_all_citation_content).parameters
    document = Document.from_source("The filing.")
    pages, full = object(), object()
    seen = []

    opinion = importlib.import_module(
        "mellea_lrc.workflows.validate_all_citation_content.prepare_root_opinions"
    )
    evidence = importlib.import_module(
        "mellea_lrc.workflows.validate_all_citation_content.prepare_citation_evidence"
    )
    support = importlib.import_module(
        "mellea_lrc.workflows.validate_all_citation_content.review_citation_content"
    )

    def complete(name):
        def run(saved, **_kwargs):
            return saved.complete_substage(name)

        return run

    def read(name):
        async def run(saved, **_kwargs):
            return saved.complete_substage(name)

        return run

    async def page_review(saved, *, reviewer):
        seen.append(("pages", reviewer))
        return saved.complete_substage(
            "validate_all_citation_content.review_citation_content.review_cited_pages_llm"
        )

    async def full_review(saved, *, reviewer):
        seen.append(("full", reviewer))
        return saved.complete_substage(
            "validate_all_citation_content.review_citation_content.review_full_opinions_llm"
        )

    for module, operation, name in (
        (
            opinion,
            "retrieve_reporter_root_opinions",
            "validate_all_citation_content.prepare_root_opinions.retrieve_opinions",
        ),
        (
            opinion,
            "index_reporter_root_opinion_pages",
            "validate_all_citation_content.prepare_root_opinions.index_pages",
        ),
        (
            evidence,
            "resolve_reporter_citation_pages",
            "validate_all_citation_content.prepare_citation_evidence.resolve_pages",
        ),
        (
            evidence,
            "prepare_reporter_citation_evidence",
            "validate_all_citation_content.prepare_citation_evidence.prepare_evidence",
        ),
        (
            support,
            "judge_reporter_citation_content",
            "validate_all_citation_content.review_citation_content.judge_citation_content",
        ),
    ):
        monkeypatch.setattr(module, operation, complete(name))
    monkeypatch.setattr(
        evidence,
        "select_reporter_citation_opinions",
        read("validate_all_citation_content.prepare_citation_evidence.select_opinions_llm"),
    )
    monkeypatch.setattr(
        evidence,
        "read_reporter_citation_propositions",
        read("validate_all_citation_content.prepare_citation_evidence.read_propositions"),
    )
    monkeypatch.setattr(support, "review_reporter_citation_pages_llm", page_review)
    monkeypatch.setattr(support, "review_reporter_citation_full_opinions_llm", full_review)
    result = asyncio.run(
        workflow.validate_all_citation_content(document, page_reviewer=pages, full_opinion_reviewer=full)
    )
    assert result.text == document.text
    assert result.stage_runs == (
        "validate_all_citation_content.prepare_root_opinions",
        "validate_all_citation_content.prepare_citation_evidence",
        "validate_all_citation_content.review_citation_content",
    )
    assert seen == [("pages", pages), ("full", full)]
