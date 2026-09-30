"""The user-defined workflow boundary and its provider/report layers."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

from evaluations import score_run as evaluation
from mellea_lrc import api, providers, workflows
from mellea_lrc.model import Document

_WORKFLOWS = {"grow_roots", "validate_roots", "grow_leaves"}


def test_only_the_three_defined_workflows_are_exposed() -> None:
    directory = Path(workflows.__file__).parent
    assert {
        path.stem
        for path in directory.iterdir()
        if (path.is_file() and path.suffix == ".py" and path.stem != "__init__")
        or (path.is_dir() and path.name != "__pycache__")
    } == _WORKFLOWS
    assert set(workflows.__all__) == _WORKFLOWS
    assert set(evaluation._WORKFLOWS) == _WORKFLOWS
    assert {
        name
        for name in api.__all__
        if getattr(getattr(api, name), "__module__", "").startswith("mellea_lrc.workflows.")
    } == _WORKFLOWS
    for name in _WORKFLOWS:
        module = importlib.import_module(f"mellea_lrc.workflows.{name}")
        assert getattr(api, name) is getattr(module, name)
        assert getattr(workflows, name) is getattr(module, name)


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
                    assert module.startswith("mellea_lrc.providers."), (file, module)


def test_scoring_writes_one_report_pair_per_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "documents"
    directory.mkdir()
    document = Document.from_source("source").complete("36_id_attribution")
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
    assert {file.name for file in tmp_path.glob("*.md")} == {f"{name}.md" for name in _WORKFLOWS}
    assert {file.name for file in tmp_path.glob("*.json")} == {
        "run.json",
        *(f"{name}.json" for name in _WORKFLOWS),
    }


@pytest.mark.parametrize("names", [("grow_roots", "grow_roots"), ("extra_workflow",)])
def test_report_requests_reject_duplicate_or_unknown_workflows(
    tmp_path: Path, names: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        evaluation.score_run(tmp_path, names)
    assert not list(tmp_path.iterdir())
