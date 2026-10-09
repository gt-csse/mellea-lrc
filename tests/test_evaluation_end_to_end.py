"""An E2E artifact preserves four-workflow checkpoints and offline recovery."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path

import httpx
import pytest

from evaluations import __main__ as runner
from evaluations import complete_stage_boundary
from evaluations import score_run as scoring
from mellea_lrc.model import Document
from mellea_lrc.providers.courtlistener import CourtListenerHTTPError


@pytest.mark.parametrize("interruption", ["review", "opinion_http"])
@pytest.mark.parametrize("workers", [1, 3])
def test_end_to_end_composes_four_workflows_and_resumes_without_duplicate_atomic_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: str, workers: int
) -> None:
    filenames = ("001.txt", "002.txt")
    data_root = tmp_path / "data"
    source_dir = data_root / "primary" / "documents_txt"
    source_dir.mkdir(parents=True)
    annotations = data_root / "primary" / "documents"
    annotations.mkdir()
    for filename in filenames:
        (source_dir / filename).write_text("A filing without citations.\n")
        (annotations / f"{Path(filename).stem}.jsonl").write_text("{}\n")
    (data_root / "primary" / "documents.json").write_text(json.dumps({"documents": filenames}))
    results_root = tmp_path / "results"
    calls: list[tuple[str, str]] = []
    reports = []
    clients = []
    profiles = {"model-substage": {"name": "fixture", "api_key_env": "OFFLINE_CREDENTIAL"}}
    monkeypatch.setattr(runner, "_configured_model_profiles", lambda: profiles)

    def unexpected_network(*_args, **_kwargs):
        pytest.fail("E2E orchestration tests must not call real providers")

    monkeypatch.setattr(httpx.Client, "request", unexpected_network)
    monkeypatch.setattr(httpx.AsyncClient, "request", unexpected_network)

    class OfflineClient:
        def __init__(self, config):
            assert config.token is None and config.pool is None
            clients.append(self)

        def close(self):
            pass

    (tmp_path / ".env").write_text(
        "COURTLISTENER_BASE_URL=https://proxy.example/api/rest/v4/\n"
        "COURTLISTENER_TIMEOUT_SECONDS=45\n"
        "COURTLISTENER_API_TOKEN=must-not-be-used\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "CourtListenerClient", OfflineClient)
    interrupted = False
    retrieval_attempts = 0

    def workflow(name):
        async def run(document, *, checkpoint, **kwargs):
            nonlocal interrupted, retrieval_attempts
            await asyncio.sleep(0)
            if name == "grow_roots":
                assert kwargs == {"hunt_dockets": True, "review_docket_roots": True}
            elif name == "validate_roots":
                assert kwargs["retrospective_date"] is None
                assert kwargs["courtlistener_client"] in clients
                assert not kwargs.get("search_other_fields", False)
            elif name == "validate_pincite":
                assert kwargs["client"] in clients
            elif name == "grow_leaves":
                assert kwargs["review_leaves"] is True
            filename = Path(document.source_path).name
            for substage in runner._E2E_SUBSTAGES:
                if not substage.startswith(name + ".") or substage in document.substage_runs:
                    continue
                if substage == "validate_pincite.opinion_preparation.retrieval":
                    retrieval_attempts += 1
                    if interruption == "opinion_http" and not interrupted:
                        interrupted = True
                        raise CourtListenerHTTPError(
                            "CourtListener opinion retrieval returned HTTP 503",
                            failure_type="http_error",
                            upstream_status_code=503,
                        )
                calls.append((filename, substage))
                document = document.complete_substage(substage)
                checkpoint(document)
                grouped = complete_stage_boundary(document, runner._E2E_SUBSTAGES)
                if grouped != document:
                    checkpoint(grouped)
                document = grouped
                if (
                    filename == "001.txt"
                    and substage == "validate_pincite.citation_preparation.opinion_review"
                    and interruption == "review"
                    and not interrupted
                ):
                    interrupted = True
                    raise RuntimeError("Interrupted after a saved pinpoint checkpoint")
                await asyncio.sleep(0)
            return document

        return run

    for name in runner._E2E_WORKFLOWS:
        monkeypatch.setattr(runner, name, workflow(name))

    def score(run_dir, workflows):
        assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"
        assert workflows == runner._E2E_WORKFLOWS
        for filename in filenames:
            document = Document.model_validate_json((run_dir / "documents" / f"{filename}.json").read_text())
            assert document.substage_runs == runner._E2E_SUBSTAGES
            assert len(document.stage_runs) == 14
        reports.append(run_dir)
        for name in workflows:
            (run_dir / f"{name}.json").write_text("{}\n")
            (run_dir / f"{name}.md").write_text(f"# {name}\n")
        return {}

    monkeypatch.setattr(scoring, "score_run", score)
    with pytest.raises(
        RuntimeError, match=r"Interrupted after a saved pinpoint|opinion retrieval returned HTTP 503"
    ):
        asyncio.run(runner._run(data_root, results_root, None, end_to_end=True, workers=workers))
    run_dir = next(results_root.iterdir())
    artifact = run_dir / "documents" / "001.txt.json"
    checkpoint = Document.model_validate_json(artifact.read_text())
    assert checkpoint.substage_runs[-1] == (
        "validate_pincite.citation_preparation.opinion_review"
        if interruption == "review"
        else "grow_leaves.leaf_field_correction.review"
    )
    assert not reports
    saved = json.loads((run_dir / "run.json").read_text())
    assert saved["status"] == "failed" and saved["end_to_end"]
    assert saved["model_profiles"] == profiles
    assert saved["git_commit"] == runner._git_commit()
    assert saved["workers"] == workers
    assert saved["annotation_sha256"] == runner._annotation_sha256(data_root, list(filenames))
    assert saved["workflows"] == list(runner._E2E_WORKFLOWS)
    assert len(saved["substage_seconds"]["001.txt"]) == len(checkpoint.substage_runs)
    annotation_path = annotations / "001.jsonl"
    annotation_path.write_text('{}\n{"unit": "citation", "changed": true}\n')
    with pytest.raises(ValueError, match="annotation content differs"):
        asyncio.run(runner._run(data_root, results_root, None, resume_run=run_dir))
    assert Document.model_validate_json(artifact.read_text()) == checkpoint
    assert len(clients) == 1
    annotation_path.write_text("{}\n")

    monkeypatch.setattr(runner, "_configured_model_profiles", lambda: {"changed": {}})
    with pytest.raises(ValueError, match="model profiles changed"):
        asyncio.run(runner._run(data_root, results_root, None, resume_run=run_dir))
    assert Document.model_validate_json(artifact.read_text()) == checkpoint
    assert len(clients) == 1
    monkeypatch.setattr(runner, "_configured_model_profiles", lambda: profiles)

    assert asyncio.run(runner._run(data_root, results_root, None, resume_run=run_dir)) == run_dir
    expected = [(filename, substage) for filename in filenames for substage in runner._E2E_SUBSTAGES]
    assert Counter(calls) == Counter(expected)
    assert tuple(filename for filename, _ in calls[:2]) == (
        filenames if workers > 1 else ("001.txt", "001.txt")
    )
    for filename in filenames:
        assert tuple(substage for filing, substage in calls if filing == filename) == runner._E2E_SUBSTAGES
    assert all(count == 1 for count in Counter(calls).values())
    assert retrieval_attempts == (2 if interruption == "review" else 3)
    assert reports == [run_dir]
    assert list(results_root.iterdir()) == [run_dir]
    assert {path.name for path in (run_dir / "documents").iterdir()} == {
        f"{filename}.json" for filename in filenames
    }
    saved = json.loads((run_dir / "run.json").read_text())
    assert saved["status"] == "complete"
    assert all(len(saved["substage_seconds"][filename]) == 49 for filename in filenames)
    assert all(
        (run_dir / f"{workflow}.{extension}").exists()
        for workflow in runner._E2E_WORKFLOWS
        for extension in ("md", "json")
    )
    assert asyncio.run(runner._run(data_root, results_root, None, resume_run=run_dir)) == run_dir
    assert Counter(calls) == Counter(expected)
    assert len(clients) == 2


@pytest.mark.parametrize(
    "options",
    [
        {"from_roots_documents": Path("unsupported")},
        {"from_locator_review_documents": Path("unsupported")},
        {"reuse_docket_lookups": True},
        {"courtlistener_pool": "reserved"},
    ],
)
def test_end_to_end_rejects_incompatible_modes_before_creating_artifacts(
    tmp_path: Path, options: dict[str, object]
) -> None:
    roots = options.pop("from_roots_documents", None)
    with pytest.raises(ValueError, match="End-to-end mode"):
        asyncio.run(
            runner._run(tmp_path / "missing-data", tmp_path / "results", roots, end_to_end=True, **options)
        )
    assert not (tmp_path / "results").exists()


def test_scoring_rejects_changed_annotation_body_before_any_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "primary" / "documents_txt" / "example.txt"
    source.parent.mkdir(parents=True)
    source.write_text("A filing without citations.\n")
    annotation = source.parent.parent / "documents" / "example.jsonl"
    annotation.parent.mkdir()
    annotation.write_text("{}\n")
    run_dir = tmp_path / "run"
    (run_dir / "documents").mkdir(parents=True)
    document = Document.from_source(source)
    (run_dir / "documents" / "example.txt.json").write_text(document.model_dump_json())
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "set": "primary",
                "filings": ["example.txt"],
                "annotation_sha256": {"example.txt": hashlib.sha256(annotation.read_bytes()).hexdigest()},
            }
        )
    )
    annotation.write_text('{}\n{"unit": "citation", "changed": true}\n')

    def unexpected_score(_document):
        pytest.fail("Changed annotation content must be rejected before scoring")

    monkeypatch.setattr(scoring, "_WORKFLOWS", {"grow_roots": (unexpected_score, None)})
    with pytest.raises(ValueError, match="annotation content differs"):
        scoring.score_run(run_dir, ("grow_roots",))
    assert {path.name for path in run_dir.iterdir()} == {"documents", "run.json"}
