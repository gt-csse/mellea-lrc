"""Source-grounding and split invariants for the case-name experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.case_name.build_dataset import load_examples, split_examples, write_dataset


def _quoted(text: str, quote: str) -> dict[str, object]:
    start = text.index(quote)
    return {"start": start, "end": start + len(quote), "quote": quote}


def _write_document(dataset: Path, number: int, group: str, text: str, rows: list[dict[str, object]]) -> Path:
    document = f"{number:03d}__{group}__brief.txt"
    source = dataset / "documents_txt" / document
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(text, encoding="utf-8")
    header = {
        "unit": "header",
        "dataset": dataset.name,
        "document": document,
        "text": {
            "path": f"{dataset.name}/documents_txt/{document}",
            "length": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
    }
    annotation = dataset / "documents" / f"{source.stem}.jsonl"
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text("\n".join(json.dumps(row) for row in [header, *rows]) + "\n", encoding="utf-8")
    return annotation


def _root_row(identifier: str, text: str, name: str | None, locator: str) -> dict[str, object]:
    return {
        "unit": "citation",
        "id": identifier,
        "kind": "FullCaseCitation",
        "is_root": True,
        "case_name": {
            "source": {"kind": "quoted", **_quoted(text, name)} if name else {"kind": "not_stated"},
            "normalization": {"kind": "value"} if name else {"kind": "unavailable"},
        },
        "locator": {"source": {"kind": "quoted", **_quoted(text, locator)}},
    }


def _nonroot_row(identifier: str, text: str, name: str, locator: str) -> dict[str, object]:
    return {
        "unit": "citation",
        "id": identifier,
        "kind": "DocketCitation",
        "is_root": False,
        "case_name": _quoted(text, name),
        "locator": _quoted(text, locator),
    }


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "primary"
    text1 = "Préface. Alpha v. Beta, 123 F.3d 4. End."
    text2 = "See Alpha v. Beta, No. 22-123."
    text3 = "For the order, see 456 F.2d 7."
    text4 = "Gamma v. Delta, 789 F.3d 1."
    text5 = "Omega v. Sigma, 987 F.3d 2."
    _write_document(dataset, 1, "same-case", text1, [_root_row("1-o1", text1, "Alpha v. Beta", "123 F.3d 4")])
    _write_document(
        dataset, 2, "same-case", text2, [_nonroot_row("2-o1", text2, "Alpha v. Beta", "No. 22-123")]
    )
    _write_document(dataset, 3, "other-case", text3, [_root_row("3-o1", text3, None, "456 F.2d 7")])
    _write_document(
        dataset, 4, "third-case", text4, [_root_row("4-o1", text4, "Gamma v. Delta", "789 F.3d 1")]
    )
    _write_document(
        dataset, 5, "fourth-case", text5, [_root_row("5-o1", text5, "Omega v. Sigma", "987 F.3d 2")]
    )
    return dataset


def test_load_examples_preserves_source_and_context_offsets(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    examples = load_examples(dataset, left_chars=100, right_chars=10)
    assert len(examples) == 5
    assert sum(row["case_name"] is None for row in examples) == 1
    assert {row["citation_kind"] for row in examples} == {"FullCaseCitation", "DocketCitation"}
    for row in examples:
        source = (dataset / "documents_txt" / row["document"]).read_text(encoding="utf-8")
        assert row["context_text"] == source[row["context_start"] : row["context_end"]]
        for key in ("locator", "case_name"):
            span = row[key]
            if span is None:
                continue
            assert row["context_text"][span["start"] : span["end"]] == span["quote"]
            assert source[span["document_start"] : span["document_end"]] == span["quote"]
            assert span["document_start"] == row["context_start"] + span["start"]
    assert examples[0]["case_name"]["document_start"] == len("Préface. ")


def test_builder_rejects_truncated_gold_and_stale_or_bad_quotes(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    with pytest.raises(ValueError, match="outside the context window"):
        load_examples(dataset, left_chars=2)

    annotation = next((dataset / "documents").glob("001__*.jsonl"))
    lines = annotation.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["case_name"]["source"]["quote"] = "wrong quote"
    annotation.write_text("\n".join([lines[0], json.dumps(row)]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="quoted span does not match"):
        load_examples(dataset)

    annotation.write_text("\n".join(lines) + "\n", encoding="utf-8")
    source = next((dataset / "documents_txt").glob("001__*.txt"))
    source.write_text(source.read_text(encoding="utf-8") + "changed", encoding="utf-8")
    with pytest.raises(ValueError, match="length/hash differs"):
        load_examples(dataset)


def test_splits_are_reproducible_and_keep_lawsuits_together(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    out1, out2 = tmp_path / "one", tmp_path / "two"
    manifest1 = write_dataset(dataset, out1)
    manifest2 = write_dataset(dataset, out2)
    assert manifest1 == manifest2
    for name in ("train", "dev", "test", "manifest"):
        suffix = ".json" if name == "manifest" else ".jsonl"
        assert (out1 / f"{name}{suffix}").read_bytes() == (out2 / f"{name}{suffix}").read_bytes()
    for name in ("train", "dev", "test"):
        assert (
            hashlib.sha256((out1 / f"{name}.jsonl").read_bytes()).hexdigest()
            == manifest1["split_sha256"][name]
        )
    split_groups = [set(manifest1["splits"][name]["case_groups"]) for name in ("train", "dev", "test")]
    assert all(split_groups[i].isdisjoint(split_groups[j]) for i in range(3) for j in range(i + 1, 3))
    assert manifest1["negative_training_group"] == "other-case"
    assert manifest1["splits"]["train"]["negative"] == 1
    assert sum(manifest1["splits"][name]["examples"] for name in ("train", "dev", "test")) == 5


def test_missing_name_is_not_an_implicit_negative(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    annotation = next((dataset / "documents").glob("002__*.jsonl"))
    lines = annotation.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row.pop("case_name")
    annotation.write_text("\n".join([lines[0], json.dumps(row)]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing field"):
        load_examples(dataset)


def test_uneven_groups_form_diverse_balanced_holdouts() -> None:
    sizes = [145, 87, 81, 76, 42, 41, 34, 23, 22, 19, 12, 11, 10, 10, 9, 8, 3, 2, 2, 2]
    examples = [
        {
            "id": f"{group}:{item}",
            "document": f"{group}.txt",
            "locator": {"document_start": item},
            "case_group": f"group-{group}",
            "case_name": None if group == 2 and item < 2 else {},
        }
        for group, size in enumerate(sizes)
        for item in range(size)
    ]
    splits, pinned = split_examples(examples)
    assert pinned == "group-2"
    assert sum(row["case_name"] is None for row in splits["train"]) == 2
    for name in ("dev", "test"):
        assert len({row["case_group"] for row in splits[name]}) == 4
        assert abs(len(splits[name]) - len(examples) * 0.15) <= 5
