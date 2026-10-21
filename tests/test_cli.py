"""The CLI composes the public workflows and preserves native checkpoints."""

from pathlib import Path

import pytest

from mellea_lrc import cli
from mellea_lrc.config import field_extraction
from mellea_lrc.field_extraction import ml_cite
from mellea_lrc.model.document import Document

WORKFLOWS = (
    "grow_roots",
    "validate_roots_citation_identity",
    "grow_leaves",
    "validate_all_citation_content",
)


def test_validate_composes_workflows_and_saves_native_checkpoints(tmp_path, monkeypatch, capsys):
    source = tmp_path / "filing.txt"
    source.write_text("See Brown v. Board of Education, 347 U.S. 483 (1954).", encoding="utf-8")
    output = tmp_path / "artifacts" / "document.json"
    called = []

    for name in WORKFLOWS:

        async def workflow(document, _name=name, **options):
            assert isinstance(document, Document)
            assert document.source_path == str(source)
            assert Document.model_validate_json(output.read_text()).text == document.text
            if _name == "grow_roots":
                assert options["hunt_dockets"] is True
                assert options["review_docket_roots"] is True
            called.append(_name)
            options["checkpoint"](document)
            return document

        monkeypatch.setattr(cli, name, workflow)

    assert cli.main(["validate", str(source), "--output", str(output)]) == 0
    assert called == list(WORKFLOWS)
    restored = Document.model_validate_json(output.read_text())
    assert restored.text == source.read_text()
    assert f"Saved {output}" in capsys.readouterr().out
    assert list(output.parent.iterdir()) == [output]


def test_help_needs_no_source_or_runtime_configuration(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli.main(["validate", "--help"])
    assert error.value.code == 0
    assert "--output" in capsys.readouterr().out


def test_missing_source_fails_before_loading_document(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid CLI input reached preprocessing")

    monkeypatch.setattr(Document, "from_source", unexpected)
    with pytest.raises(SystemExit) as error:
        cli.main(["validate", str(tmp_path / "missing.txt")])
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"CASE_NAME": "ML"}, ValueError, "ML_ENABLED=true"),
        ({"COURT": None}, RuntimeError, "FIELD_EXTRACTION_COURT"),
        ({"CASE_NAME": "rule"}, ValueError, "case_name"),
        ({"COURT": "ml"}, ValueError, "court"),
    ],
)
def test_invalid_field_policy_fails_before_preprocessing_or_output(
    tmp_path, monkeypatch, changes, error, message
):
    source = tmp_path / "source.txt"
    source.write_text("A legal document", encoding="utf-8")
    output = tmp_path / "not-created" / "document.json"
    settings = {
        "ML_ENABLED": "false",
        "CASE_NAME": "RULE",
        "COURT": "RULE",
        "DATE": "RULE",
        "PIN_CITE": "RULE",
        "CONTEXT_BEFORE": "260",
        "CONTEXT_AFTER": "260",
    } | changes
    monkeypatch.setattr(
        field_extraction,
        "read_env",
        lambda: {"MELLEA_LRC_FIELD_EXTRACTION_" + key: value for key, value in settings.items()},
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid field policy reached preprocessing")

    monkeypatch.setattr(Document, "from_source", unexpected)
    with pytest.raises(error, match=message):
        cli.main(["validate", str(source), "--output", str(output)])
    assert not output.parent.exists()


@pytest.mark.parametrize("case_name_source", ["ML", "RULE"])
def test_unloadable_enabled_model_fails_before_preprocessing_or_output(
    tmp_path, monkeypatch, case_name_source
):
    source = tmp_path / "source.txt"
    source.write_text("A legal document", encoding="utf-8")
    output = tmp_path / "not-created" / "document.json"
    settings = {
        "ML_ENABLED": "true",
        "CASE_NAME": case_name_source,
        "COURT": "RULE",
        "DATE": "RULE",
        "PIN_CITE": "RULE",
        "MODEL_ID": "test/model",
        "REVISION": "a" * 40,
        "VARIANT": "augmented",
        "DEVICE": "cpu",
        "CONTEXT_BEFORE": "260",
        "CONTEXT_AFTER": "260",
        "BATCH_SIZE": "8",
    }
    monkeypatch.setattr(
        field_extraction,
        "read_env",
        lambda: {"MELLEA_LRC_FIELD_EXTRACTION_" + key: value for key, value in settings.items()},
    )

    def broken_model(config):
        raise RuntimeError("Model checkpoint cannot load")

    def unexpected(*args, **kwargs):
        pytest.fail("Failed ML preflight reached preprocessing")

    monkeypatch.setattr(ml_cite, "get_ml_reader", broken_model)
    monkeypatch.setattr(Document, "from_source", unexpected)
    with pytest.raises(RuntimeError, match="Model checkpoint cannot load"):
        cli.main(["validate", str(source), "--output", str(output)])
    assert not output.parent.exists()


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_output_cannot_overwrite_source(tmp_path, alias):
    source = tmp_path / "source.txt"
    source.write_text("Original source", encoding="utf-8")
    output = source
    if alias == "symlink":
        output = tmp_path / "alias.json"
        output.symlink_to(source)
    elif alias == "hardlink":
        output = tmp_path / "alias.json"
        output.hardlink_to(source)
    with pytest.raises(SystemExit) as error:
        cli.main(["validate", str(source), "--output", str(output)])
    assert error.value.code == 2
    assert source.read_text() == "Original source"


def test_provider_failure_keeps_last_native_checkpoint(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.txt"
    source.write_text("A legal document", encoding="utf-8")
    output = tmp_path / "document.json"

    async def fail(document, **options):
        options["checkpoint"](document)
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(cli, "grow_roots", fail)
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        cli.main(["validate", str(source), "--output", str(output)])
    assert Document.model_validate_json(output.read_text()).text == source.read_text()
    assert "Saved " not in capsys.readouterr().out


def test_failed_checkpoint_write_preserves_previous_output(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text("A legal document", encoding="utf-8")
    output = tmp_path / "document.json"
    previous = Document.from_source("Previous artifact").model_dump_json()
    output.write_text(previous, encoding="utf-8")

    def fail_replace(self, target):
        raise OSError("Cannot replace checkpoint")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="Cannot replace checkpoint"):
        cli.main(["validate", str(source), "--output", str(output)])
    assert output.read_text() == previous
    assert sorted(path.name for path in tmp_path.iterdir()) == ["document.json", "source.txt"]
