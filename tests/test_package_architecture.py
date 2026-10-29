"""The user-defined workflow boundary and its provider/report layers."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path

import pytest

from evaluations import score_run as evaluation
from mellea_lrc import api, extraction, providers, validation, workflows
from mellea_lrc.model import SUBSTAGE_CATALOG, WORKFLOW_CATALOG, Document

_WORKFLOWS = {
    "grow_roots",
    "validate_roots_citation_identity",
    "grow_leaves",
    "validate_all_citation_content",
}
_STAGES = {
    "discover_root_locators",
    "read_root_fields",
    "form_root_groups",
    "lookup_reporter_roots",
    "lookup_docket_roots",
    "corroborate_locator_bodies",
    "discover_intended_cases",
    "grow_short_reporter_leaves",
    "grow_reference_leaves",
    "grow_id_leaves",
    "grow_supra_leaves",
    "correct_leaf_readings",
    "prepare_root_opinions",
    "prepare_citation_evidence",
    "review_citation_content",
}
_RENAMED_APIS = {
    "review_citation_support": "review_citation_content",
    "docket_root_llm_reassignment": "reassign_docket_roots_llm",
    "reporter_root_lookup_cluster_retrieval": "retrieve_reporter_root_clusters",
    "reporter_root_lookup_docket_retrieval": "retrieve_reporter_root_dockets",
    "reporter_root_lookup_unique_rule_judgment": "judge_unique_reporter_roots_rule",
    "reporter_root_lookup_ambiguous_rule_judgment": "judge_ambiguous_reporter_roots_rule",
    "reporter_root_lookup_unique_llm_judgment": "review_unique_reporter_roots_llm",
    "reporter_root_lookup_ambiguous_llm_judgment": "review_ambiguous_reporter_roots_llm",
    "docket_root_lookup_courtlistener_retrieval": "retrieve_courtlistener_docket_roots",
    "docket_root_lookup_courtlistener_llm_review": "review_courtlistener_docket_roots_llm",
    "docket_root_lookup_govinfo_retrieval": "retrieve_govinfo_docket_roots",
    "docket_root_lookup_govinfo_llm_review": "review_govinfo_docket_roots_llm",
    "fields_aggregated_identity": "judge_root_identity_from_fields",
    "locator_body_courtlistener_opinion_retrieval": "retrieve_courtlistener_locator_opinions",
    "locator_body_courtlistener_recap_retrieval": "retrieve_courtlistener_locator_filings",
    "locator_body_govinfo_opinion_retrieval": "retrieve_govinfo_locator_opinions",
    "locator_body_llm_judgment": "review_locator_bodies_llm",
    "intended_case_courtlistener_opinion_retrieval": "retrieve_courtlistener_intended_case_opinions",
    "intended_case_courtlistener_recap_retrieval": "retrieve_courtlistener_intended_case_filings",
    "intended_case_govinfo_opinion_retrieval": "retrieve_govinfo_intended_case_opinions",
    "intended_case_llm_selection": "review_intended_cases_llm",
    "reporter_root_opinion_retrieval": "retrieve_reporter_root_opinions",
    "review_reporter_citation_opinions": "select_reporter_citation_opinions",
    "prepare_reporter_citation_pinpoint_evidence": "prepare_reporter_citation_evidence",
    "review_reporter_citation_pinpoint_pages": "review_reporter_citation_pages_llm",
    "review_reporter_citation_full_opinions": "review_reporter_citation_full_opinions_llm",
    "judge_reporter_citation_pinpoints": "judge_reporter_citation_content",
    "review_supra_attributions": "review_supra_attributions_llm",
}
_REMOVED_ATOMIC_MODULES = (
    "extraction.roots",
    "extraction.full_reporter_locator",
    "extraction.docket_locator",
    "extraction.docket_site_hunting",
    "extraction.docket_entries",
    "extraction.colocations",
    "extraction.case_names",
    "extraction.courts",
    "extraction.dates",
    "extraction.pin_cites",
    "extraction.short_reporter_locator",
    "extraction.short_reporter_colocations",
    "extraction.short_reporter_case_names",
    "extraction.short_reporter_attribution",
    "extraction.reference_citations",
    "extraction.reference_attribution",
    "extraction.id_citations",
    "extraction.id_attribution",
    "extraction.supra_citations",
    "extraction.supra_case_names",
    "extraction.supra_pin_cites",
    "extraction.supra_attribution_rule",
    "extraction.supra_attribution_llm",
    "extraction.leaf_field_corrections",
    "extraction.docket_root_llm_reassignment",
    "validation.reporter_root_lookup_cluster_retrieval",
    "validation.reporter_root_lookup_docket_retrieval",
    "validation.reporter_root_lookup_unique_rule_judgment",
    "validation.reporter_root_lookup_ambiguous_rule_judgment",
    "validation.reporter_root_lookup_unique_llm_judgment",
    "validation.reporter_root_lookup_ambiguous_llm_judgment",
    "validation.docket_root_lookup_courtlistener_retrieval",
    "validation.docket_root_lookup_courtlistener_llm_review",
    "validation.docket_root_lookup_govinfo_retrieval",
    "validation.docket_root_lookup_govinfo_llm_review",
    "validation.fields_aggregated_identity",
    "validation.body_search.locator_body_courtlistener_opinion_retrieval",
    "validation.body_search.locator_body_courtlistener_recap_retrieval",
    "validation.body_search.locator_body_govinfo_opinion_retrieval",
    "validation.locator_body_llm_judgment",
    "validation.body_search.intended_case_courtlistener_opinion_retrieval",
    "validation.body_search.intended_case_courtlistener_recap_retrieval",
    "validation.body_search.intended_case_govinfo_opinion_retrieval",
    "validation.intended_case_llm_selection",
    "validation.reporter_root_opinion_retrieval",
    "validation.reporter_root_opinion_page_index",
    "validation.reporter_citation_page_resolution",
    "validation.reporter_citation_opinion_review",
    "validation.reporter_citation_propositions",
    "validation.reporter_citation_pinpoint_evidence",
    "validation.reporter_citation_pinpoint_page_review",
    "validation.reporter_citation_full_opinion_review",
    "validation.reporter_citation_pinpoint_judgment",
)


def test_only_the_user_defined_workflows_are_exposed() -> None:
    directory = Path(workflows.__file__).parent
    assert {
        path.stem
        for path in directory.iterdir()
        if (path.is_file() and path.suffix == ".py" and path.stem != "__init__")
        or (path.is_dir() and path.name != "__pycache__")
    } == _WORKFLOWS
    assert set(workflows.__all__) == _WORKFLOWS
    assert {workflow.name for workflow in WORKFLOW_CATALOG} == _WORKFLOWS
    assert set(evaluation._WORKFLOWS) == _WORKFLOWS
    assert {
        name
        for name in api.__all__
        if getattr(getattr(api, name), "__module__", "").startswith("mellea_lrc.workflows.")
    } == _WORKFLOWS | _STAGES
    for name in _WORKFLOWS:
        module = importlib.import_module(f"mellea_lrc.workflows.{name}")
        assert getattr(api, name) is getattr(module, name)
        assert getattr(workflows, name) is getattr(module, name)
        assert getattr(api, name).__name__ == name

    for workflow in _WORKFLOWS:
        package = importlib.import_module(f"mellea_lrc.workflows.{workflow}")
        assert Path(package.__file__).name == "__init__.py"
        assert {
            path.stem for path in Path(package.__file__).parent.glob("*.py") if path.stem != "__init__"
        } == {
            stage.name.rsplit(".", 1)[1]
            for item in WORKFLOW_CATALOG
            if item.name == workflow
            for stage in item.stages
        }
        for stage_name in set(package.__all__) - {workflow}:
            assert getattr(api, stage_name) is getattr(package, stage_name)
            assert getattr(api, stage_name).__module__.startswith(f"mellea_lrc.workflows.{workflow}.")


@pytest.mark.parametrize("old_name,new_name", _RENAMED_APIS.items())
def test_replaced_public_apis_have_no_old_name_alias(old_name, new_name) -> None:
    assert new_name in api.__all__
    assert getattr(api, new_name).__name__ == new_name
    for module in (api, extraction, validation, workflows):
        assert not hasattr(module, old_name), (module.__name__, old_name)


@pytest.mark.parametrize("old_name", ["validate_roots", "validate_pincite"])
def test_replaced_workflows_have_no_public_or_module_alias(old_name) -> None:
    assert not hasattr(api, old_name)
    assert not hasattr(workflows, old_name)
    assert importlib.util.find_spec(f"mellea_lrc.workflows.{old_name}") is None


@pytest.mark.parametrize("old_name", _REMOVED_ATOMIC_MODULES)
def test_replaced_atomic_modules_have_no_import_alias(old_name) -> None:
    assert importlib.util.find_spec(f"mellea_lrc.{old_name}") is None


def test_proposition_reader_keeps_its_action_name_without_a_duplicate_prefix() -> None:
    reader = api.read_reporter_citation_propositions
    module = importlib.import_module("mellea_lrc.validation.read_reporter_citation_propositions")
    assert reader.__name__ == "read_reporter_citation_propositions"
    assert reader.__module__ == module.__name__
    assert module.read_reporter_citation_propositions is reader
    assert not hasattr(api, "read_read_reporter_citation_propositions")


def test_provider_clients_do_not_depend_on_citation_stages_or_workflows() -> None:
    directory = Path(providers.__file__).parent
    assert {path.name for path in directory.iterdir() if path.is_dir() and path.name != "__pycache__"} == {
        "courtlistener",
        "govinfo",
    }
    for file in directory.rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for module in modules:
                if module.startswith("mellea_lrc."):
                    assert (
                        module.startswith("mellea_lrc.providers.") or module == "mellea_lrc.configuration"
                    ), (file, module)


def test_source_readers_and_durable_models_do_not_import_stages() -> None:
    directory = Path(api.__file__).parent
    allowed_reading_layers = ("mellea_lrc.parsing.", "mellea_lrc.matching.")
    stage_layers = ("mellea_lrc.extraction", "mellea_lrc.validation", "mellea_lrc.workflows")
    for layer in ("parsing", "matching", "model", "llm"):
        for file in (directory / layer).rglob("*.py"):
            for node in ast.walk(ast.parse(file.read_text())):
                modules = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else []
                )
                for module in modules:
                    assert not module.startswith(stage_layers), (file, module)
                    if layer == "parsing" and module.startswith("mellea_lrc."):
                        assert module.startswith(allowed_reading_layers), (file, module)
                    if layer == "matching":
                        assert not module.startswith("mellea_lrc.preprocessing"), (file, module)


@pytest.mark.parametrize("layer", ["extraction", "validation"])
def test_direct_stage_modules_have_an_explicit_stage_boundary(layer: str) -> None:
    directory = Path(api.__file__).parent / layer
    opposite = "validation" if layer == "extraction" else "extraction"
    for file in directory.glob("*.py"):
        if file.name == "__init__.py":
            continue
        tree = ast.parse(file.read_text())
        assert any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "SUBSTAGE" for target in node.targets)
            for node in tree.body
        ), file
    for file in directory.rglob("*.py"):
        tree = ast.parse(file.read_text())
        if any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "SUBSTAGE" for target in node.targets)
            for node in tree.body
        ):
            parts = file.relative_to(Path(api.__file__).parent).with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            module = importlib.import_module(".".join(("mellea_lrc", *parts)))
            writer = getattr(module, parts[-1])
            assert module.SUBSTAGE in SUBSTAGE_CATALOG, file
            assert inspect.isfunction(writer), file
            assert writer.__name__ == parts[-1], file
            assert writer.__module__ == module.__name__, file
            assert getattr(api, parts[-1]) is writer, file
        for node in ast.walk(tree):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert not any(module.startswith(f"mellea_lrc.{opposite}") for module in modules), file


def test_scoring_writes_one_report_pair_per_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "documents"
    directory.mkdir()
    document = (
        Document.from_source("source")
        .complete_substage("grow_roots.discover_root_locators.find_full_reporter_locators")
        .complete_substage("validate_roots_citation_identity.lookup_reporter_roots.retrieve_clusters")
        .complete_substage("grow_leaves.grow_short_reporter_leaves.find_short_reporter_citations")
        .complete_substage("grow_leaves.grow_id_leaves.attribute_citations")
        .complete_substage("validate_all_citation_content.prepare_root_opinions.retrieve_opinions")
    )
    (directory / "filing.txt.json").write_text(document.model_dump_json())
    (tmp_path / "run.json").write_text(
        json.dumps({"status": "complete", "set": "primary", "filings": ["filing.txt"]})
    )
    calls: list[str] = []

    class Score:
        def __init__(self, name: str):
            self.name = name

        def as_dict(self) -> dict:
            return {"workflow": self.name}

    def scorer(name: str):
        def score(saved: Document) -> Score:
            assert saved == document
            calls.append(name)
            return Score(name)

        return score

    for name in _WORKFLOWS:
        monkeypatch.setitem(
            evaluation._WORKFLOWS, name, (scorer(name), lambda score, *, set_name: score.name)
        )
    reports = evaluation.score_run(tmp_path)
    assert set(reports) == _WORKFLOWS
    assert len(calls) == len(_WORKFLOWS) and set(calls) == _WORKFLOWS
    assert {file.name for file in tmp_path.glob("*.md")} == {
        "llm_usage.md",
        *(f"{name}.md" for name in _WORKFLOWS),
    }
    assert {file.name for file in tmp_path.glob("*.json")} == {
        "run.json",
        "llm_usage.json",
        *(f"{name}.json" for name in _WORKFLOWS),
    }


@pytest.mark.parametrize("names", [("grow_roots", "grow_roots"), ("extra_workflow",)])
def test_report_requests_reject_duplicate_or_unknown_workflows(
    tmp_path: Path, names: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        evaluation.score_run(tmp_path, names)
    assert not list(tmp_path.iterdir())
