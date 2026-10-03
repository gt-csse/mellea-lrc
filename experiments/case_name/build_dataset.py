"""Build locator-anchored case-name examples from the primary annotations.

Offsets in the annotation and source text are Unicode character offsets. The
output carries both document and context-relative offsets so tokenizers can map
the gold span without searching for a possibly repeated name string.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

FULL_KINDS = frozenset({"FullCaseCitation", "DocketCitation"})
SPLITS = ("train", "dev", "test")
DEFAULT_DATASET = Path(__file__).resolve().parents[3] / "mellea-lrc-datasets" / "primary"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_annotation(path: Path, dataset_dir: Path) -> tuple[str, list[dict[str, Any]], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty annotation file: {path}")
    header = json.loads(lines[0])
    document = f"{path.stem}.txt"
    source_path = dataset_dir / "documents_txt" / document
    expected_relative = (Path(dataset_dir.name) / "documents_txt" / document).as_posix()
    if (
        header.get("unit") != "header"
        or header.get("dataset") != dataset_dir.name
        or header.get("document") != document
        or header.get("text", {}).get("path") != expected_relative
    ):
        raise ValueError(f"Annotation header does not identify its source text: {path}")
    text = source_path.read_bytes().decode("utf-8")
    if header["text"].get("length") != len(text) or header["text"].get("sha256") != _digest(text):
        raise ValueError(f"Annotation source length/hash differs from {source_path}")
    rows = [json.loads(line) for line in lines[1:]]
    return text, rows, header["text"]["sha256"]


def _source_span(field: Any, text: str, label: str) -> dict[str, Any] | None:
    """Read native root source outcomes and direct nonroot quoted spans."""
    if not isinstance(field, dict):
        raise ValueError(f"{label}: missing field")
    source = field.get("source", field)
    if not isinstance(source, dict):
        raise ValueError(f"{label}: invalid source")
    kind = source.get("kind", "quoted" if "quote" in source else None)
    if kind == "not_stated":
        return None
    if kind != "quoted":
        raise ValueError(f"{label}: unsupported source kind {kind!r}")
    start, end, quote = source.get("start"), source.get("end"), source.get("quote")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not isinstance(quote, str)
        or not 0 <= start < end <= len(text)
        or text[start:end] != quote
    ):
        raise ValueError(f"{label}: quoted span does not match source text exactly")
    return {"document_start": start, "document_end": end, "quote": quote}


def _local_span(span: dict[str, Any], context_start: int) -> dict[str, Any]:
    return {
        "start": span["document_start"] - context_start,
        "end": span["document_end"] - context_start,
        **span,
    }


def _case_group(document: str) -> str:
    # The first component is a filing number; filings for the same lawsuit
    # share the second component. Keep those documents in one split.
    parts = Path(document).stem.split("__", 2)
    if len(parts) != 3 or not parts[1]:
        raise ValueError(f"Cannot identify filing case group from {document!r}")
    return parts[1]


def load_examples(dataset_dir: Path, left_chars: int = 384, right_chars: int = 128) -> list[dict[str, Any]]:
    """Load every full locator with an explicit, exactly grounded name outcome."""
    if left_chars < 0 or right_chars < 0:
        raise ValueError("Window character counts must be nonnegative")
    dataset_dir = dataset_dir.resolve()
    annotation_dir = dataset_dir / "documents"
    files = sorted(annotation_dir.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No primary annotation files in {annotation_dir}")
    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in files:
        text, rows, _ = _read_annotation(path, dataset_dir)
        document = f"{path.stem}.txt"
        group = _case_group(document)
        for row in rows:
            if row.get("unit") != "citation" or row.get("kind") not in FULL_KINDS:
                continue
            citation_id = row.get("id")
            if not isinstance(citation_id, str) or not citation_id:
                raise ValueError(f"{path}: full locator lacks an id")
            example_id = f"{path.stem}:{citation_id}"
            if example_id in seen_ids:
                raise ValueError(f"Duplicate citation id: {example_id}")
            seen_ids.add(example_id)
            locator = _source_span(row.get("locator"), text, f"{example_id} locator")
            name = _source_span(row.get("case_name"), text, f"{example_id} case_name")
            if locator is None:
                raise ValueError(f"{example_id}: a full locator must be quoted")
            start = max(0, locator["document_start"] - left_chars)
            end = min(len(text), locator["document_end"] + right_chars)
            if name is not None and not (start <= name["document_start"] < name["document_end"] <= end):
                raise ValueError(
                    f"{example_id}: gold case name is outside the context window "
                    f"[{start}, {end}); increase --left-chars or --right-chars"
                )
            context = text[start:end]
            local_locator = _local_span(locator, start)
            local_name = _local_span(name, start) if name is not None else None
            if context[local_locator["start"] : local_locator["end"]] != locator["quote"]:
                raise AssertionError(f"{example_id}: locator shifted incorrectly")
            if local_name is not None and context[local_name["start"] : local_name["end"]] != name["quote"]:
                raise AssertionError(f"{example_id}: name shifted incorrectly")
            examples.append(
                {
                    "id": example_id,
                    "document": document,
                    "case_group": group,
                    "citation_id": citation_id,
                    "citation_kind": row["kind"],
                    "is_root": row.get("is_root") is True,
                    "context_text": context,
                    "context_start": start,
                    "context_end": end,
                    "locator": local_locator,
                    "case_name": local_name,
                }
            )
    return sorted(examples, key=lambda x: (x["document"], x["locator"]["document_start"], x["id"]))


def split_examples(
    examples: list[dict[str, Any]], seed: int = 42, train_fraction: float = 0.7, dev_fraction: float = 0.15
) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    """Assign whole lawsuit groups, with diverse holdouts and a negative in train."""
    if not 0 < train_fraction < 1 or not 0 < dev_fraction < 1 or train_fraction + dev_fraction >= 1:
        raise ValueError("Train and dev fractions must be positive and leave a positive test fraction")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        groups[example["case_group"]].append(example)
    if len(groups) < 3:
        raise ValueError("At least three case groups are required for train/dev/test splits")
    targets = {
        "train": len(examples) * train_fraction,
        "dev": len(examples) * dev_fraction,
        "test": len(examples) * (1 - train_fraction - dev_fraction),
    }
    negative_groups = [
        group for group, records in groups.items() if any(x["case_name"] is None for x in records)
    ]
    pinned = (
        min(
            negative_groups,
            key=lambda group: (
                -sum(x["case_name"] is None for x in groups[group]),
                _digest(f"{seed}:{group}"),
            ),
        )
        if negative_groups
        else None
    )
    available = sorted(group for group in groups if group != pinned)
    # Four independent lawsuit groups per held-out split on the primary set.
    # Tiny synthetic sets use as many as their group count permits.
    holdout_groups = min(4, (len(groups) - 1) // 2)
    candidates = list(combinations(available, holdout_groups))

    def size(candidate: tuple[str, ...]) -> int:
        return sum(len(groups[group]) for group in candidate)

    # Rank single holdouts by their target size, then choose the best disjoint
    # pair. The bounded pool keeps this deterministic search fast on the corpus.
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            abs(size(candidate) - targets["dev"]),
            _digest(f"{seed}:{','.join(candidate)}"),
        ),
    )
    pool = ranked[:512]

    def best_pair(options: list[tuple[str, ...]]) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        best: tuple[float, str, tuple[str, ...], tuple[str, ...]] | None = None
        for dev_groups in options:
            dev_size = size(dev_groups)
            dev_set = set(dev_groups)
            for test_groups in options:
                if dev_set.intersection(test_groups):
                    continue
                test_size = size(test_groups)
                train_size = len(examples) - dev_size - test_size
                error = sum(
                    ((count - targets[name]) / targets[name]) ** 2
                    for name, count in (("train", train_size), ("dev", dev_size), ("test", test_size))
                )
                key = (error, _digest(f"{seed}:{','.join(dev_groups)}:{','.join(test_groups)}"))
                if best is None or key < best[:2]:
                    best = (*key, dev_groups, test_groups)
        return None if best is None else (best[2], best[3])

    pair = best_pair(pool) or best_pair(ranked)
    if pair is None:
        raise ValueError("Unable to make three nonempty group-preserving splits")
    dev_groups, test_groups = pair
    assignments = dict.fromkeys(groups, "train")
    assignments.update(dict.fromkeys(dev_groups, "dev"))
    assignments.update(dict.fromkeys(test_groups, "test"))
    result: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for group, records in groups.items():
        result[assignments[group]].extend(records)
    for split in SPLITS:
        result[split].sort(key=lambda x: (x["document"], x["locator"]["document_start"], x["id"]))
    return result, pinned


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_dataset(
    dataset_dir: Path,
    output_dir: Path,
    *,
    left_chars: int = 384,
    right_chars: int = 128,
    seed: int = 42,
    train_fraction: float = 0.7,
    dev_fraction: float = 0.15,
) -> dict[str, Any]:
    """Validate source annotations and write reproducible JSONL splits."""
    examples = load_examples(dataset_dir, left_chars=left_chars, right_chars=right_chars)
    splits, pinned = split_examples(
        examples, seed=seed, train_fraction=train_fraction, dev_fraction=dev_fraction
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    split_hashes: dict[str, str] = {}
    for name, records in splits.items():
        path = output_dir / f"{name}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        payload = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records
        ).encode("utf-8")
        split_hashes[name] = hashlib.sha256(payload).hexdigest()
        temporary.write_bytes(payload)
        temporary.replace(path)
    manifest = {
        "dataset_dir": str(dataset_dir.resolve()),
        "window": {"left_chars": left_chars, "right_chars": right_chars},
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "dev": dev_fraction,
            "test": 1 - train_fraction - dev_fraction,
        },
        "negative_training_group": pinned,
        "split_sha256": split_hashes,
        "content_sha256": _digest("".join(f"{name}:{split_hashes[name]}\n" for name in SPLITS)),
        "total": {
            "examples": len(examples),
            "positive": sum(row["case_name"] is not None for row in examples),
            "negative": sum(row["case_name"] is None for row in examples),
            "documents": len({row["document"] for row in examples}),
            "case_groups": len({row["case_group"] for row in examples}),
        },
        "splits": {
            name: {
                "examples": len(records),
                "positive": sum(row["case_name"] is not None for row in records),
                "negative": sum(row["case_name"] is None for row in records),
                "documents": sorted({row["document"] for row in records}),
                "case_groups": sorted({row["case_group"] for row in records}),
            }
            for name, records in splits.items()
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-chars", type=int, default=384)
    parser.add_argument("--right-chars", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    arguments = parser.parse_args()
    manifest = write_dataset(
        arguments.dataset_dir,
        arguments.output_dir,
        left_chars=arguments.left_chars,
        right_chars=arguments.right_chars,
        seed=arguments.seed,
        train_fraction=arguments.train_fraction,
        dev_fraction=arguments.dev_fraction,
    )
    print(json.dumps({"total": manifest["total"], "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
