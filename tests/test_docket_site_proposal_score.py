"""The offline proposal score uses source spans and excludes index annotations."""

import hashlib
import json
from pathlib import Path

import pytest

from evaluations.docket_proposals import score, score_set
from mellea_lrc.extraction import find_docket_locators, find_full_reporter_locators
from scripts import run_docket_site_hunt as runner


def test_score_counts_rule_proposal_and_remaining_gold_after_index_exclusion(tmp_path: Path) -> None:
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
    source_document = runner._source_document(tmp_path, "primary", filename, metadata)
    document = find_docket_locators(find_full_reporter_locators(source_document))
    checkpoint = runner._checkpoint(run_dir, "primary", filename)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(document.model_dump_json(), encoding="utf-8")

    # Unlike the runner, an evaluator must load the annotation, and it must
    # fail loudly if the gold file for a saved prediction is missing.
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
        "eligible_gold_docket_locators": 3,
        "rule_found": 1,
        "exact_proposed_among_rule_misses": 1,
        "remaining_misses": 1,
        "total_proposals": 1,
    }
    assert result["totals"] == result["sets"]["primary"]
