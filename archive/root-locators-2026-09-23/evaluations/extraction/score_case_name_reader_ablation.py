"""Replay the case-name reader against saved locator checkpoints and gold roots.

This is an offline ablation: no provider or model calls. The saved ``case-names``
documents are the baseline, and the current reader runs on their corresponding
``case-name-input`` documents. Match by locator kind and start, then score only
annotated root occurrences whose locator was present in the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mellea_lrc.extraction.reading.case_names import reread_case_names
from mellea_lrc.model.document import Document

SETS = (
    "primary",
    "hallucination-set-1",
    "hallucination-set-2",
    "reliable-high-profile",
    "reliable-low-profile",
)


def _normalized(value: str) -> str:
    return "".join(char for char in value.casefold() if char.isalnum())


def _gold_roots(path: Path) -> dict[tuple[str, int], dict]:
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("unit") != "citation" or not row.get("is_root") or not row.get("locator"):
            continue
        key = row["kind"], row["locator"]["start"]
        if key in result:
            raise ValueError(f"Duplicate gold locator key {key} in {path}")
        result[key] = row
    return result


def _predictions(citations) -> dict[tuple[str, int], object]:
    result = {}
    for item in citations:
        if item.withdrawn:
            continue
        key = type(item.fields).__name__, item.locator_span.start
        if key in result:
            raise ValueError(f"Duplicate predicted locator key {key}")
        result[key] = item.case_name
    return result


def score(artifact_root: Path, annotation_root: Path) -> dict:
    by_set = {}
    for dataset in SETS:
        source = artifact_root / dataset / "case-name-input"
        baseline_dir = artifact_root / dataset / "case-names"
        counts = Counter()
        examples = []
        for input_path in sorted(source.glob("*.json")):
            baseline_path = baseline_dir / input_path.name
            gold_path = annotation_root / dataset / "documents" / f"{input_path.stem}.jsonl"
            if not baseline_path.exists() or not gold_path.exists():
                raise FileNotFoundError(f"Missing baseline or annotation for {input_path}")
            document = Document.model_validate(json.loads(input_path.read_text()))
            baseline = Document.model_validate(json.loads(baseline_path.read_text()))
            before = _predictions(baseline.citations)
            after = _predictions(reread_case_names(document.text, document.citations))
            for key, gold in _gold_roots(gold_path).items():
                if key not in after:
                    counts["unreached_gold_root"] += 1
                    continue
                counts["reached_gold_root"] += 1
                gold_name = gold.get("case_name")
                earlier = before.get(key)
                current = after[key]
                if gold_name is None:
                    counts["gold_name_absent"] += 1
                    if earlier is not None:
                        counts["baseline_false_name"] += 1
                    if current is not None:
                        counts["current_false_name"] += 1
                    if earlier is None and current is not None:
                        counts["new_false_name"] += 1
                        examples.append(
                            {"id": gold["id"], "issue": "new_false_name", "prediction": current.text}
                        )
                    continue
                counts["gold_name_present"] += 1
                for label, name in (("baseline", earlier), ("current", current)):
                    if name is None:
                        continue
                    counts[f"{label}_name_present"] += 1
                    if _normalized(name.text) == _normalized(gold_name["quote"]):
                        counts[f"{label}_normalized_agreement"] += 1
                    if (name.span.start, name.span.end) == (gold_name["start"], gold_name["end"]):
                        counts[f"{label}_exact_span"] += 1
                if earlier is None and current is not None:
                    counts["new_name"] += 1
                    if _normalized(current.text) == _normalized(gold_name["quote"]):
                        counts["new_name_agrees"] += 1
                    elif len(examples) < 30:
                        examples.append(
                            {
                                "id": gold["id"],
                                "issue": "new_name_disagrees",
                                "prediction": current.text,
                                "gold": gold_name["quote"],
                            }
                        )
        by_set[dataset] = {"counts": dict(sorted(counts.items())), "examples": examples}
    return {
        "description": "Offline replay of current case-name reader from saved locator checkpoints; root occurrences only.",
        "caveat": "The five corpora are not independent holdouts: some filings and citations recur across sets. Exact spans are diagnostic, not identity judgments.",
        "sets": by_set,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("data/run-artifacts/1-roots/colocation-checkpoint-from-final-locators"),
    )
    parser.add_argument("--annotations", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score(args.artifacts, args.annotations)
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
