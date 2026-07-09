"""Tests for config-driven snapshot corpus selection."""

# ruff: noqa: INP001

from pathlib import Path

import pytest

import json

from scripts.e2e_backend.snapshot_corpus import SnapshotConfig, run_document, select_documents


def _config(root: Path, *, batch_start: int = 2, batch_end: int = 4) -> SnapshotConfig:
    return SnapshotConfig(
        env_path=root / ".env",
        test_data_dir=root / "test_data",
        bookmarked_path=root / "bookmarked.txt",
        snapshot_root=root / "snapshots",
        batch_start=batch_start,
        batch_end=batch_end,
    )


def test_select_documents_uses_configured_inclusive_text_range(tmp_path: Path) -> None:
    """An omitted selector expands the configured inclusive numeric range."""
    config = _config(tmp_path)
    config.test_data_dir.mkdir()
    for index in range(2, 5):
        (config.test_data_dir / f"{index}.txt").write_text(str(index), encoding="utf-8")

    assert select_documents(config, None) == tuple(
        config.test_data_dir / f"{index}.txt" for index in range(2, 5)
    )


def test_select_documents_supports_bookmark_and_numeric_file(tmp_path: Path) -> None:
    """Single-file mode resolves only the two supported selector forms."""
    config = _config(tmp_path)
    config.test_data_dir.mkdir()
    config.bookmarked_path.write_text("bookmark", encoding="utf-8")
    numbered = config.test_data_dir / "12.txt"
    numbered.write_text("numbered", encoding="utf-8")

    assert select_documents(config, "bookmarked") == (config.bookmarked_path,)
    assert select_documents(config, "12") == (numbered,)


@pytest.mark.parametrize("selector", ["2.txt", "0", "-1", "all", "1.5"])
def test_select_documents_rejects_unsupported_file_selector(
    tmp_path: Path,
    selector: str,
) -> None:
    """Extensions, ranges, zero, and arbitrary names are not CLI selectors."""
    with pytest.raises(ValueError, match="--file"):
        select_documents(_config(tmp_path), selector)


def test_run_document_writes_citation_node_snapshot(tmp_path: Path) -> None:
    """The snapshot harness exposes the citation-node projection as its own phase."""
    config = _config(tmp_path)
    config.test_data_dir.mkdir()
    source = config.test_data_dir / "1.txt"
    source.write_text("Norton v. Shelby County, 118 U.S. 425.", encoding="utf-8")

    run_document(source, phase="citation_nodes", config=config)

    snapshot = json.loads((config.snapshot_root / "1" / "citation_nodes.json").read_text())
    assert snapshot["schema_version"] == 19
    assert snapshot["artifact_type"] == "citation_node_document"
    assert snapshot["nodes"][0]["citation_id"] == "cite-0001"
