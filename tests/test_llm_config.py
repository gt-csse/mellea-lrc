"""Complete model profiles are chosen by stages and resolved without inheritance."""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from mellea.backends import ModelOption

from mellea_lrc.llm import reviewer as reviewer_runtime
from mellea_lrc.llm.config import (
    OUTPUT_MODE_OPTION,
    PROFILE_OPTION,
    LlmApiConfig,
    LlmOutputMode,
    start_mellea_session,
)
from mellea_lrc.llm.profiles import (
    NRP_GLM,
    NRP_KIMI,
    NRP_QWEN,
    OPENROUTER_LUNA,
    LlmProfile,
)


def _profile(**changes: object) -> LlmProfile:
    return LlmProfile(
        name="test-profile",
        model="test-model",
        api_base="https://provider.invalid/v1",
        api_key_env="TEST_MODEL_KEY",
        **changes,
    )


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
            "MELLEA_LRC_LLM_VALIDATE_PINCITE_MODEL": "workflow-model",
            "MELLEA_LRC_LLM_VALIDATE_PINCITE_REASONING_EFFORT": "high",
        }
    )

    assert config == profile.resolve({"TEST_MODEL_KEY": "correct-secret"})


@pytest.mark.parametrize("environ", [{}, {"TEST_MODEL_KEY": ""}, {"TEST_MODEL_KEY": "   "}])
def test_missing_profile_credential_raises_without_using_another_key(environ) -> None:
    with pytest.raises((RuntimeError, ValueError), match="TEST_MODEL_KEY"):
        _profile().resolve({"MELLEA_LRC_LLM_API_KEY": "unrelated-secret", **environ})


@pytest.mark.parametrize("profile", [OPENROUTER_LUNA, NRP_GLM, NRP_QWEN, NRP_KIMI])
def test_named_profile_resolves_as_one_package(profile: LlmProfile) -> None:
    config = profile.resolve({profile.api_key_env: "profile-secret"})
    assert config.model == profile.model
    assert config.api_base == profile.api_base
    assert config.temperature == profile.temperature
    assert config.output_mode == profile.output_mode
    assert config.profile_name == profile.name
    assert "profile-secret" not in repr(config)
    assert "profile-secret" not in repr(config.mellea_call_options(max_tokens=profile.max_tokens))


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
        "name": "test-profile",
        "endpoint": "https://provider.invalid/v1",
    }
    assert options[ModelOption.STREAM_TIMEOUT] == 300
    assert options["temperature"] == 0
    assert options["max_tokens"] == 32
    assert options["reasoning_effort"] == "low"
    assert options["service_tier"] == "flex"
    assert "test-secret" not in repr(options)
    assert "TEST_MODEL_KEY" not in repr(options)


def test_unnamed_resolved_binding_has_no_profile_metadata_or_optional_provider_options() -> None:
    config = LlmApiConfig(model="test", api_base="https://provider.invalid/v1", api_key="test-secret")
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


_REVIEWERS = (
    ("extraction.docket_site_hunting.review", "IvrDocketReviewer"),
    ("extraction.docket_root_llm_reassignment.reviewer", "IvrDocketRootReviewer"),
    ("extraction.leaf_attribution_review.reviewer", "IvrLeafReviewer"),
    ("validation.reporter_root_lookup_unique_llm_judgment.reviewer", "IvrReporterUniqueReviewer"),
    ("validation.reporter_root_lookup_ambiguous_llm_judgment.reviewer", "IvrReporterAmbiguousReviewer"),
    ("validation.docket_root_lookup_courtlistener_llm_review.reviewer", "IvrDocketLookupReviewer"),
    ("validation.docket_root_lookup_govinfo_llm_review.reviewer", "IvrGovInfoDocketReviewer"),
    ("validation.locator_body_llm_judgment.reviewer", "IvrBodyCorroborationReviewer"),
    ("validation.intended_case_llm_selection.reviewer", "IvrIntendedCaseReviewer"),
    ("validation.reporter_citation_opinion_review.reviewer", "IvrReporterCitationOpinionReviewer"),
    ("validation.reporter_citation_propositions.reviewer", "IvrReporterCitationPropositionReviewer"),
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


_STAGES = (
    "extraction.docket_site_hunting",
    "extraction.docket_root_llm_reassignment",
    "extraction.short_reporter_attribution",
    "extraction.reference_attribution",
    "extraction.id_attribution",
    "extraction.supra_attribution_llm",
    "validation.reporter_root_lookup_unique_llm_judgment",
    "validation.reporter_root_lookup_ambiguous_llm_judgment",
    "validation.docket_root_lookup_courtlistener_llm_review",
    "validation.docket_root_lookup_govinfo_llm_review",
    "validation.locator_body_llm_judgment",
    "validation.intended_case_llm_selection",
    "validation.reporter_citation_opinion_review",
    "validation.reporter_citation_propositions",
    "validation.reporter_citation_pinpoint_page_review",
    "validation.reporter_citation_full_opinion_review",
)


@pytest.mark.parametrize("module_name", _STAGES)
def test_stage_passes_its_own_profile_to_the_reviewer(module_name: str) -> None:
    module = importlib.import_module(f"mellea_lrc.{module_name}")
    assert isinstance(module.MODEL_PROFILE, LlmProfile)
    calls = [
        node
        for node in ast.walk(ast.parse(Path(module.__file__).read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_profile"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert len(call.args) == 1 and isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "MODEL_PROFILE"
    assert not call.keywords


def test_page_and_full_opinion_defaults_are_selected_independently() -> None:
    pages = importlib.import_module("mellea_lrc.validation.reporter_citation_pinpoint_page_review")
    full = importlib.import_module("mellea_lrc.validation.reporter_citation_full_opinion_review")
    assert pages.MODEL_PROFILE is NRP_GLM
    assert full.MODEL_PROFILE is NRP_QWEN


def test_pinpoint_workflow_passes_separate_page_and_full_opinion_reviewers(monkeypatch) -> None:
    from mellea_lrc.model import Document

    workflow = importlib.import_module("mellea_lrc.workflows.validate_pincite")
    assert "support_reviewer" not in inspect.signature(workflow.validate_pincite).parameters
    document = Document.from_source("The filing.")
    pages, full = object(), object()
    seen = []

    def unchanged(saved, **_kwargs):
        return saved

    async def read(saved, **_kwargs):
        return saved

    async def page_review(saved, *, reviewer):
        seen.append(("pages", reviewer))
        return saved

    async def full_review(saved, *, reviewer):
        seen.append(("full", reviewer))
        return saved

    for name in (
        "reporter_root_opinion_retrieval",
        "index_reporter_root_opinion_pages",
        "resolve_reporter_citation_pages",
        "prepare_reporter_citation_pinpoint_evidence",
        "judge_reporter_citation_pinpoints",
    ):
        monkeypatch.setattr(workflow, name, unchanged)
    monkeypatch.setattr(workflow, "review_reporter_citation_opinions", read)
    monkeypatch.setattr(workflow, "read_reporter_citation_propositions", read)
    monkeypatch.setattr(workflow, "review_reporter_citation_pinpoint_pages", page_review)
    monkeypatch.setattr(workflow, "review_reporter_citation_full_opinions", full_review)
    result = asyncio.run(workflow.validate_pincite(document, page_reviewer=pages, full_opinion_reviewer=full))
    assert result is document
    assert seen == [("pages", pages), ("full", full)]
