"""The offline proposal score includes table-of-authorities docket sites."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from evaluations import docket_proposals
from evaluations.docket_proposals import score, score_set
from mellea_lrc.api import find_docket_locators, find_full_reporter_locators
from mellea_lrc.model import (
    Document,
    PreprocessingBackend,
    PreprocessingMetadata,
    SourceFormat,
    SourceMetadata,
    Span,
)


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], ("primary",)),
        (["--set", "hallucination-set-1"], ("hallucination-set-1",)),
        (["--set", "primary", "--set", "hallucination-set-1"], ("primary", "hallucination-set-1")),
    ],
)
def test_cli_set_selection_defaults_to_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str], expected: tuple[str, ...]
) -> None:
    selected: list[tuple[str, ...]] = []

    def fake_score(_data_root: Path, _run_dir: Path, names: tuple[str, ...]) -> dict:
        selected.append(names)
        return {}

    monkeypatch.setattr(docket_proposals, "score", fake_score)
    monkeypatch.setattr(sys, "argv", ["docket_proposals", "--run-dir", str(tmp_path), *options])
    docket_proposals.main()
    assert selected == [expected]


def test_score_defaults_to_primary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selected: list[str] = []

    def fake_score_set(_data_root: Path, _run_dir: Path, name: str) -> dict[str, int]:
        selected.append(name)
        return dict.fromkeys(docket_proposals.COUNT_FIELDS, 0)

    monkeypatch.setattr(docket_proposals, "score_set", fake_score_set)
    score(tmp_path, tmp_path)
    assert selected == ["primary"]


def test_score_counts_rule_proposal_and_remaining_gold_including_index(tmp_path: Path) -> None:
    source = "INDEX: Case No. 035547/2021\nSee Case No. 1:24-cv-00123. Later No. 19 Civ. 8034; Misc 77/4."
    root = tmp_path / "primary"
    text_dir = root / "documents_txt"
    text_dir.mkdir(parents=True)
    filename = "001.txt"
    (text_dir / filename).write_text(source, encoding="utf-8")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    index_end = source.index("\n")
    metadata = {
        "sha256": digest,
        "length": len(source),
        "backend": "docling",
        "index_spans": [{"start": 0, "end": index_end}],
    }
    (root / "documents.json").write_text(
        json.dumps({"documents": {filename: metadata}}),
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    source_document = Document(
        source_metadata=SourceMetadata(path=str(text_dir / filename), format=SourceFormat.TEXT),
        text=source,
        preprocessing_metadata=PreprocessingMetadata(backend=PreprocessingBackend.DOCLING),
        index_spans=(Span(0, index_end),),
    )
    document = find_docket_locators(find_full_reporter_locators(source_document))
    checkpoint = run_dir / "documents" / "primary" / f"{filename}.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(document.model_dump_json(), encoding="utf-8")

    # Evaluation needs gold, and must fail loudly if a saved prediction has none.
    with pytest.raises(FileNotFoundError):
        score_set(tmp_path, run_dir, "primary")

    annotations_dir = root / "documents"
    annotations_dir.mkdir()

    def docket_row(quote: str, *, from_offset: int = 0, parallel_wl: bool = False) -> dict[str, object]:
        start = source.index(quote, from_offset)
        return {
            "unit": "citation",
            "kind": "DocketCitation",
            "identifier": None if parallel_wl else {"docket_number": quote},
            "locator": {"start": start, "end": start + len(quote), "quote": quote},
        }

    rows = [
        {"unit": "header", "document": filename, "text": {"sha256": digest}},
        docket_row("Case No. 035547/2021"),
        docket_row("Case No. 1:24-cv-00123"),
        docket_row("No. 19 Civ. 8034", parallel_wl=True),
        docket_row("77/4"),
    ]
    (annotations_dir / "001.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = score(tmp_path, run_dir, ("primary",))

    assert result["sets"]["primary"] == {
        "documents": 1,
        "eligible_gold_docket_locators": 4,
        "rule_found": 1,
        "exact_proposed_among_rule_misses": 2,
        "remaining_misses": 1,
        "total_proposals": 2,
    }
    assert result["totals"] == result["sets"]["primary"]
