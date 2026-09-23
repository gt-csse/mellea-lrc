"""Score an arm by what it did at each cited place, so the counts add up.

`evaluate.py` scores predictions against the bench by identity, which is the
right contract for a published benchmark and an awkward one for reading a
misparse. A citation read with the wrong page is charged twice -- a false
positive for the identifier it invented and a false negative for the one it
lost -- so the rows outnumber the occurrences and the arithmetic stops
reconciling with the ground truth.

This asks a different question, once per **place the document cites something**:

    correct     an identifier was reported there, and it is the right one
    incorrect   an identifier was reported there, and it names something else
    miss        nothing was reported there at all

Those three partition the ground truth, so they sum to it exactly. A fourth
count sits outside that sum because it belongs to no cited place:

    spurious    an identifier reported where the document cites nothing

The distinction the three-way split buys is the one the whole relaxation
question turns on. A **miss** reports nothing, so nothing downstream is checked
and the filing is silently under-verified. An **incorrect** reports a
well-formed citation naming a different case, so verification runs against the
wrong authority and returns a confident verdict about it. They are not the same
failure and precision and recall blur them together.

    uv run python -m evaluations.extraction.site_score \\
      --benchmark data/false-citation-bench-locator-only-v1.0/extraction.jsonl \\
      --artifact run-artifact.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """One JSON object per nonblank line."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def identity(record: dict[str, Any]) -> str | None:
    """Volume, reporter and page, with the reporter's spacing ignored.

    Converters disagree about whether a reporter is `N.C.App.` or `N.C. App.`,
    and that difference does not make it a different case.
    """
    volume, reporter, page = record.get("volume"), record.get("reporter"), record.get("page")
    if not (volume and reporter and page):
        return None
    return f"{volume}|{''.join(str(reporter).split())}|{page}"


def overlap(left: dict[str, Any], right: dict[str, Any]) -> int:
    """How many characters the two spans share."""
    a, b = left["span"], right["span"]
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


@dataclass(frozen=True, slots=True)
class Outcome:
    """One cited place and what the arm made of it."""

    verdict: str
    document: str
    expected: str
    reported: str | None
    text: str


def score(bench: list[dict], predictions: list[dict]) -> tuple[list[Outcome], list[dict]]:
    """Judge each cited place, and return whatever was reported nowhere near one.

    Pairing is by span overlap, largest first, so a prediction is matched to the
    citation it actually covers rather than to the nearest one of the same name.
    Both sides index the same text, which is what makes that legitimate.
    """
    outcomes: list[Outcome] = []
    unmatched: list[dict] = []

    by_document: dict[str, list[dict]] = {}
    for prediction in predictions:
        by_document.setdefault(prediction["document"], []).append(prediction)

    for document in sorted({record["document"] for record in bench} | set(by_document)):
        truth = [record for record in bench if record["document"] == document]
        predicted = list(by_document.get(document, []))

        pairs = sorted(
            (
                (overlap(record, prediction), index, position)
                for index, record in enumerate(truth)
                for position, prediction in enumerate(predicted)
                if overlap(record, prediction)
            ),
            reverse=True,
        )
        taken_truth: dict[int, int] = {}
        taken_prediction: set[int] = set()
        for _, index, position in pairs:
            if index in taken_truth or position in taken_prediction:
                continue
            taken_truth[index] = position
            taken_prediction.add(position)

        for index, record in enumerate(truth):
            position = taken_truth.get(index)
            expected = identity(record) or record["matched_text"]
            if position is None:
                outcomes.append(Outcome("miss", document, expected, None, record["matched_text"]))
                continue
            reported = identity(predicted[position]) or predicted[position]["matched_text"]
            verdict = "correct" if reported == expected else "incorrect"
            outcomes.append(
                Outcome(verdict, document, expected, reported, predicted[position]["matched_text"])
            )

        unmatched += [p for position, p in enumerate(predicted) if position not in taken_prediction]

    return outcomes, unmatched


def report(outcomes: list[Outcome], spurious: list[dict], limit: int) -> str:
    """The three-way split, and every failure written out."""
    counts = {
        verdict: sum(1 for o in outcomes if o.verdict == verdict)
        for verdict in ("correct", "incorrect", "miss")
    }
    total = len(outcomes)

    lines = ["| outcome | count | share |", "|---|---:|---:|"]
    for verdict in ("correct", "incorrect", "miss"):
        share = f"{counts[verdict] / total:.1%}" if total else "-"
        lines.append(f"| {verdict} | {counts[verdict]} | {share} |")
    lines += [
        f"| **cited places** | **{total}** | |",
        "",
        f"reported where nothing is cited (spurious): {len(spurious)}",
    ]

    failures = [o for o in outcomes if o.verdict != "correct"]
    if failures:
        lines += ["", "## What went wrong", ""]
        for outcome in failures[:limit]:
            text = " ".join(outcome.text.split())[:52]
            if outcome.verdict == "miss":
                lines.append(f"- **miss** {outcome.document[:26]} expected `{outcome.expected}` — {text!r}")
            else:
                lines.append(
                    f"- **incorrect** {outcome.document[:26]} expected `{outcome.expected}`, "
                    f"read `{outcome.reported}` — {text!r}"
                )
    if spurious:
        lines += ["", "## Reported where nothing is cited", ""]
        for record in spurious[:limit]:
            lines.append(f"- {record['document'][:26]} {record.get('matched_text', '')!r}")
    return "\n".join(lines)


def main() -> int:
    """Score one run artifact against one bench."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    outcomes, spurious = score(read_jsonl(args.benchmark), read_jsonl(args.artifact))
    print(report(outcomes, spurious, args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
