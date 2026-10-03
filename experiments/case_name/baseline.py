"""Score the existing case-name rules on the experimental context windows.

The training records provide a wider window than the production extractor's
bounded ``before()`` window. This adapter applies the production source reader to each record's full prefix, so the baseline and a model
receive the same text. It does not train a model or modify dataset files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mellea_lrc.extraction.case_names.reader import read_case_name


def predict(record: dict[str, Any]) -> tuple[int, int] | None:
    """Return a case-name span in record-local character offsets, if found."""
    context = record["context_text"]
    locator = record["locator"]
    locator_start, locator_end = locator["start"], locator["end"]
    if context[locator_start:locator_end] != locator["quote"]:
        raise ValueError(f"{record['id']}: locator quote does not match context")
    reporter_quote = locator["quote"] if record["citation_kind"] == "FullCaseCitation" else None
    span = read_case_name(context[:locator_start], 0, reporter_quote)
    return (span.start, span.end) if span is not None else None


def score(path: Path, predictions_path: Path | None = None) -> dict[str, Any]:
    """Evaluate exact and overlapping source spans without reading gold in prediction."""
    counts = {
        "records": 0,
        "gold_names": 0,
        "predicted_names": 0,
        "exact": 0,
        "overlap": 0,
        "correct_absence": 0,
    }
    predictions: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        prediction = predict(record)
        gold = record["case_name"]
        counts["records"] += 1
        counts["gold_names"] += int(gold is not None)
        counts["predicted_names"] += int(prediction is not None)
        if gold is not None and record["context_text"][gold["start"] : gold["end"]] != gold["quote"]:
            raise ValueError(f"{path}:{line_number}: gold quote does not match context")
        exact = (prediction is None and gold is None) or (
            prediction is not None and gold is not None and prediction == (gold["start"], gold["end"])
        )
        overlap = (
            prediction is not None
            and gold is not None
            and prediction[0] < gold["end"]
            and gold["start"] < prediction[1]
        )
        predictions.append(
            {
                "id": record["id"],
                "gold": (
                    {"start": gold["start"], "end": gold["end"], "quote": gold["quote"]}
                    if gold is not None
                    else None
                ),
                "prediction": (
                    {
                        "start": prediction[0],
                        "end": prediction[1],
                        "quote": record["context_text"][prediction[0] : prediction[1]],
                    }
                    if prediction is not None
                    else None
                ),
                "exact": exact,
                "overlap": overlap,
            }
        )
        if gold is None:
            counts["correct_absence"] += int(prediction is None)
            continue
        if prediction is None:
            continue
        counts["exact"] += int(exact)
        counts["overlap"] += int(overlap)
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = predictions_path.with_suffix(predictions_path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
            encoding="utf-8",
        )
        temporary.replace(predictions_path)
    precision = counts["exact"] / counts["predicted_names"] if counts["predicted_names"] else None
    recall = counts["exact"] / counts["gold_names"] if counts["gold_names"] else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {
        "file": str(path),
        **counts,
        "exact_precision": precision,
        "exact_recall": recall,
        "exact_f1": f1,
        "overlap_recall": counts["overlap"] / counts["gold_names"] if counts["gold_names"] else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory containing build_dataset.py's dev.jsonl and test.jsonl",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test", "both"),
        default="both",
        help="Evaluate held-out splits; training records are excluded",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        help="Optional directory for per-example dev/test predictions JSONL",
    )
    args = parser.parse_args()
    splits = ("dev", "test") if args.split == "both" else (args.split,)
    for split in splits:
        path = args.data_dir / f"{split}.jsonl"
        if not path.is_file():
            parser.error(f"Missing {path}; run build_dataset.py first")
        predictions_path = (
            args.predictions_dir / f"{split}.predictions.jsonl" if args.predictions_dir else None
        )
        print(json.dumps({"split": split, **score(path, predictions_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
