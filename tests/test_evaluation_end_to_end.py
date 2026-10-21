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
from evaluations import run_grow_leaves as leaf_runner
from evaluations import run_validate_all_citation_content as pinpoint_runner
from evaluations import score_run as scoring
from evaluations.sources import load_source, validate_source
from mellea_lrc.api import grow_roots
from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.model import Document
from mellea_lrc.model.preprocessed_document import DEFAULT_RULES, PreprocessingBackend, Rule
from mellea_lrc.model.source import SourceFormat
from mellea_lrc.providers.courtlistener import CourtListenerHTTPError

_RULE_FIELDS = FieldExtractionConfig(
    ml_enabled=False, case_name="RULE", court="RULE", date="RULE", pin_cite="RULE"
)


@pytest.fixture(autouse=True)
def explicit_rule_field_extraction(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "MELLEA_LRC_FIELD_EXTRACTION_ML_ENABLED=false\nMELLEA_LRC_FIELD_EXTRACTION_CASE_NAME=RULE\nMELLEA_LRC_FIELD_EXTRACTION_COURT=RULE\nMELLEA_LRC_FIELD_EXTRACTION_DATE=RULE\nMELLEA_LRC_FIELD_EXTRACTION_PIN_CITE=RULE\n"
    )
    monkeypatch.chdir(tmp_path)


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
    (data_root / "primary" / "documents.json").write_text(
        json.dumps({"documents": {name: {} for name in filenames}})
    )
    # This test covers workflow composition and recovery, independently of
    # rendering metadata; the source boundary is tested below with native input.
    monkeypatch.setattr(runner, "load_source", lambda path: Document.from_source(path))
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
        "MELLEA_LRC_FIELD_EXTRACTION_ML_ENABLED=false\nMELLEA_LRC_FIELD_EXTRACTION_CASE_NAME=RULE\nMELLEA_LRC_FIELD_EXTRACTION_COURT=RULE\nMELLEA_LRC_FIELD_EXTRACTION_DATE=RULE\nMELLEA_LRC_FIELD_EXTRACTION_PIN_CITE=RULE\n"
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
                assert kwargs == {
                    "hunt_dockets": True,
                    "review_docket_roots": True,
                    "field_extraction": _RULE_FIELDS,
                }
            elif name == "validate_roots_citation_identity":
                assert kwargs["retrospective_date"] is None
                assert kwargs["courtlistener_client"] in clients
                assert not kwargs.get("search_other_fields", False)
            elif name == "validate_all_citation_content":
                assert kwargs["client"] in clients
            elif name == "grow_leaves":
                assert kwargs["review_leaves"] is True
                assert kwargs["field_extraction"] == _RULE_FIELDS
            filename = Path(document.source_path).name
            for substage in runner._E2E_SUBSTAGES:
                if not substage.startswith(name + ".") or substage in document.substage_runs:
                    continue
                if substage == "validate_all_citation_content.prepare_root_opinions.retrieve_opinions":
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
                    and substage
                    == "validate_all_citation_content.prepare_citation_evidence.select_opinions_llm"
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
        "validate_all_citation_content.prepare_citation_evidence.select_opinions_llm"
        if interruption == "review"
        else "grow_leaves.correct_leaf_readings.correct_leaf_fields"
    )
    assert not reports
    saved = json.loads((run_dir / "run.json").read_text())
    assert saved["status"] == "failed" and saved["end_to_end"]
    assert saved["model_profiles"] == profiles
    assert saved["field_extraction"] == _RULE_FIELDS.model_dump(mode="json")
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
    assert all(
        len(saved["substage_seconds"][filename]) == len(runner._E2E_SUBSTAGES) for filename in filenames
    )
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


def _native_dataset(tmp_path, filenames=("001.txt",)):
    data_root = tmp_path / "native-data"
    source_dir = data_root / "primary" / "documents_txt"
    source_dir.mkdir(parents=True)
    text = "Table of Authorities\r\nTwombly, 550 U.S. 544, 556 (2007).\r\nBody: café.\r\n"
    renderings = {}
    for filename in filenames:
        raw = text.encode("utf-8")
        (source_dir / filename).write_bytes(raw)
        renderings[filename] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "length": len(text),
            "backend": "docling",
            "backend_version": "source-test-version",
            "layout_rules": ["margin_line_numbers", "repeated_furniture", "docket_stamp"],
            "table_structure": False,
            "index_spans": [{"start": 0, "end": text.index("Body:")}],
        }
    (source_dir.parent / "documents.json").write_text(json.dumps({"documents": renderings}))
    return data_root, source_dir, text, renderings


def test_native_source_preserves_official_path_exact_text_and_recorded_rules(tmp_path):
    _, source_dir, text, _ = _native_dataset(tmp_path)
    document = load_source(source_dir / "001.txt")
    assert document.text == text
    assert document.source_path == str((source_dir / "001.txt").resolve())
    assert document.source_metadata.format is SourceFormat.TEXT
    assert document.preprocessing_metadata.backend is PreprocessingBackend.DOCLING
    assert document.preprocessing_metadata.backend_version == "source-test-version"
    assert document.preprocessing_metadata.rules == DEFAULT_RULES
    assert len(document.index_spans) == 1
    assert document.citations == document.runs == ()
    validate_source(document)
    with pytest.raises(ValueError, match="preprocessing_metadata differs"):
        validate_source(Document.from_source(source_dir / "001.txt"))


@pytest.mark.parametrize(
    "change",
    [
        {"sha256": "stale"},
        {"length": 1},
        {"length": True},
        {"backend": "plain_text"},
        {"backend_version": ""},
        {"layout_rules": ["invented"]},
        {"layout_rules": ["docket_stamp", "docket_stamp"]},
        {"table_structure": "false"},
        {"index_spans": None},
        {"index_spans": [{"start": False, "end": 5}]},
        {"index_spans": [{"start": 0, "end": 9999}]},
    ],
)
def test_native_source_rejects_stale_or_malformed_rendering_metadata(tmp_path, change):
    _, source_dir, _, renderings = _native_dataset(tmp_path)
    renderings["001.txt"].update(change)
    (source_dir.parent / "documents.json").write_text(json.dumps({"documents": renderings}))
    with pytest.raises(ValueError):
        load_source(source_dir / "001.txt")


@pytest.mark.parametrize("documents", [[], {"001.txt": None}, {}])
def test_native_source_rejects_missing_rendering_records(tmp_path, documents):
    _, source_dir, _, _ = _native_dataset(tmp_path)
    (source_dir.parent / "documents.json").write_text(json.dumps({"documents": documents}))
    with pytest.raises(ValueError):
        load_source(source_dir / "001.txt")


def test_native_source_requires_explicit_index_metadata_even_when_no_toa_was_found(tmp_path):
    _, source_dir, _, renderings = _native_dataset(tmp_path)
    record = renderings["001.txt"]
    record["index_spans"] = []
    record["table_structure"] = True
    (source_dir.parent / "documents.json").write_text(json.dumps({"documents": renderings}))
    assert load_source(source_dir / "001.txt").preprocessing_metadata.rules == (
        Rule.MARGIN_LINE_NUMBERS,
        Rule.REPEATED_FURNITURE,
        Rule.DOCKET_STAMP,
        Rule.TABLE_OF_AUTHORITIES,
    )
    del record["index_spans"]
    (source_dir.parent / "documents.json").write_text(json.dumps({"documents": renderings}))
    with pytest.raises(ValueError, match="explicit valid table-of-authorities spans"):
        load_source(source_dir / "001.txt")


def test_primary_runner_uses_native_metadata_and_retains_toa_citations_for_identity(tmp_path, monkeypatch):
    data_root, source_dir, text, renderings = _native_dataset(tmp_path)
    calls = []

    async def validate(document, *, checkpoint, **_kwargs):
        assert document.text == text
        assert document.source_path == str(source_dir / "001.txt")
        assert document.roots and document.roots[0].tags
        calls.append(document)
        for substage in runner._RUN_SUBSTAGES:
            if not substage.startswith("validate_roots_citation_identity."):
                continue
            document = document.complete_substage(substage)
            checkpoint(document)
            document = complete_stage_boundary(document, runner._RUN_SUBSTAGES)
            checkpoint(document)
        return document

    def unexpected_network(*_args, **_kwargs):
        pytest.fail("Native source integration must not call providers or models")

    monkeypatch.setattr(httpx.Client, "request", unexpected_network)
    monkeypatch.setattr(httpx.AsyncClient, "request", unexpected_network)
    monkeypatch.setattr(runner, "validate_roots_citation_identity", validate)
    run_dir = asyncio.run(runner._run(data_root, tmp_path / "native-results", None))
    saved = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_bytes())
    assert saved.index_spans and saved.citations[0].tags
    assert saved.get_substage(runner._ROOT_SUBSTAGE).citations[0].tags == saved.citations[0].tags
    assert len(calls) == 1
    before = (run_dir / "run.json").read_bytes()
    renderings["001.txt"]["backend_version"] = "changed-renderer"
    (source_dir.parent / "documents.json").write_text(json.dumps({"documents": renderings}))
    with pytest.raises(ValueError, match="rendering metadata differs"):
        asyncio.run(runner._run(data_root, tmp_path / "unused", None, resume_run=run_dir))
    assert len(calls) == 1
    assert (run_dir / "run.json").read_bytes() == before


def test_primary_runner_rejects_missing_metadata_in_later_filing_before_any_work(tmp_path, monkeypatch):
    data_root, source_dir, _, renderings = _native_dataset(tmp_path, filenames=("001.txt", "002.txt"))
    del renderings["002.txt"]["index_spans"]
    (source_dir.parent / "documents.json").write_text(json.dumps({"documents": renderings}))

    def unexpected(*_args, **_kwargs):
        pytest.fail("Every rendering must validate before processing the first filing")

    monkeypatch.setattr(runner, "grow_roots", unexpected)
    monkeypatch.setattr(runner, "CourtListenerClient", unexpected)
    with pytest.raises(ValueError, match="explicit valid table-of-authorities spans"):
        asyncio.run(runner._run(data_root, tmp_path / "rejected-results", None))
    assert list((tmp_path / "rejected-results").rglob("documents/*.json")) == []


@pytest.mark.parametrize("selected_runner", [leaf_runner, pinpoint_runner])
def test_standalone_runner_rejects_missing_creation_tags_in_all_inputs_before_work(
    tmp_path, monkeypatch, selected_runner
):
    _, source_dir, _, _ = _native_dataset(tmp_path, filenames=("001.txt", "002.txt"))
    inputs = tmp_path / "native-input" / "documents"
    inputs.mkdir(parents=True)
    for filename in ("001.txt", "002.txt"):
        roots = asyncio.run(grow_roots(load_source(source_dir / filename)))
        raw = roots.model_dump(mode="json")
        if filename == "002.txt":
            raw["citations"][0]["tags"] = []
        (inputs / f"{filename}.json").write_text(json.dumps(raw))
    (inputs.parent / "run.json").write_text(
        json.dumps({"status": "complete", "set": "primary", "filings": ["001.txt", "002.txt"]})
    )

    def unexpected(*_args, **_kwargs):
        pytest.fail("All native input sources must be checked before any client or workflow call")

    monkeypatch.setattr(selected_runner, "_RESULTS_ROOT", tmp_path / "standalone-results")
    if selected_runner is leaf_runner:
        monkeypatch.setattr(selected_runner, "grow_leaves", unexpected)
        options = {"input_substage": runner._ROOT_SUBSTAGE}
    else:
        monkeypatch.setattr(selected_runner, "_resolved_model_profiles", dict)
        monkeypatch.setattr(selected_runner, "CourtListenerClient", unexpected)
        options = {}
    with pytest.raises(ValueError, match="missing source table-of-authorities tags"):
        asyncio.run(selected_runner.run(inputs, **options))
    assert not (tmp_path / "standalone-results").exists()
