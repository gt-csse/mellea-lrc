"""Score every candidate margin rule against the cached corpus layout.

Two things are measured, and they pull against each other.

**Residue** is the recall side: after a rule moves its items to furniture, does
a column of line numbers still survive into the exported text? That is the
failure the rule exists to prevent, and it is read off the text rather than the
layout, because that is where it does its damage.

**Overreach** is the precision side, and it has no labels to check against, so
it is approximated from the shape of what was taken. A margin number belongs to
a long run numbered down the page. An item removed while standing outside any
such run is the rule reaching past its evidence, and a numbered list in the
body is exactly what that looks like.

    uv run python -m exploration.margin_rules.score

Needs `data/page-layout-cache.json` from `scripts/dump_page_layout.py`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from exploration.margin_rules.candidates import RULES, line_number_value

# The same residue test `scripts/regenerate_bench_text.py` uses, so a number
# here means the same thing as a number there.
_STANDALONE_NUMBERS = re.compile(r"(?m)^[ \t]*(\d{1,3})[ \t]*$")
_MIN_RUN = 4


def gutter_runs(text: str) -> list[list[int]]:
    """Runs of consecutive ascending integers left standing alone in the text."""
    numbers = [int(m.group(1)) for m in _STANDALONE_NUMBERS.finditer(text)]
    runs, current = [], []
    for value in numbers:
        if current and value == current[-1] + 1:
            current.append(value)
            continue
        if len(current) >= _MIN_RUN:
            runs.append(current)
        current = [value]
    if len(current) >= _MIN_RUN:
        runs.append(current)
    return runs


def exported(items: list[dict], removed: set[int]) -> str:
    """Approximate `export_to_text`: the body, in order, minus what was removed."""
    return "\n\n".join(
        item["text"]
        for index, item in enumerate(items)
        if index not in removed and item["layer"] == "body" and item["text"]
    )


def overreach(items: list[dict], removed: set[int]) -> list[str]:
    """Removed items that do not belong to a run numbered down a page.

    Built per page: sort every removed item by descending top edge, walk it, and
    keep the maximal ascending runs. Anything not inside a run of at least four
    was taken on position alone.
    """
    loose = []
    pages = {item["page"] for index, item in enumerate(items) if index in removed}
    for page in pages:
        on_page = sorted(
            (i for i in removed if items[i]["page"] == page),
            key=lambda i: -items[i]["t"],
        )
        run: list[int] = []
        for index in on_page:
            value = line_number_value(items[index]["text"])
            previous = line_number_value(items[run[-1]]["text"]) if run else None
            if run and value is not None and previous is not None and value > previous:
                run.append(index)
                continue
            if len(run) < _MIN_RUN:
                loose += [items[i]["text"] for i in run]
            run = [index]
        if len(run) < _MIN_RUN:
            loose += [items[i]["text"] for i in run]
    return loose


def main() -> int:
    """Run every rule over every document and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/page-layout-cache.json"))
    args = parser.parse_args()

    if not args.cache.exists():
        print(f"{args.cache} missing -- run scripts/dump_page_layout.py first")
        return 1

    corpus = json.loads(args.cache.read_text())
    print(f"{len(corpus)} documents\n")

    baseline = {entry["source"]: len(gutter_runs(exported(entry["items"], set()))) for entry in corpus}
    untouched = sum(1 for count in baseline.values() if count)
    print(f"before any rule: {untouched} documents carry a gutter\n")

    print(f"{'rule':<14}{'removed':>9}{'docs w/ residue':>18}{'residual runs':>15}{'overreach':>11}")
    results = {}
    for name, rule in RULES.items():
        removed_total = residue_docs = residue_runs = loose_total = 0
        detail = {}
        for entry in corpus:
            items = entry["items"]
            removed = rule(items)
            runs = gutter_runs(exported(items, removed))
            loose = overreach(items, removed)
            removed_total += len(removed)
            residue_runs += len(runs)
            residue_docs += 1 if runs else 0
            loose_total += len(loose)
            detail[entry["source"]] = {"removed": len(removed), "runs": len(runs), "loose": loose}
        results[name] = detail
        print(f"{name:<14}{removed_total:>9}{residue_docs:>18}{residue_runs:>15}{loose_total:>11}")

    print("\n--- documents still carrying a gutter, per rule ---")
    for name, detail in results.items():
        left = [source for source, row in detail.items() if row["runs"]]
        print(f"{name:<14}{', '.join(s[:24] for s in left) if left else '(none)'}")

    print("\n--- what each rule removes that `current` does not ---")
    for name, detail in results.items():
        if name == "current":
            continue
        gained = {
            source: row["removed"] - results["current"][source]["removed"]
            for source, row in detail.items()
            if row["removed"] != results["current"][source]["removed"]
        }
        print(f"{name:<14}{gained if gained else '(identical)'}")

    print("\n--- overreach samples ---")
    for name, detail in results.items():
        samples = [t for row in detail.values() for t in row["loose"]][:12]
        print(f"{name:<14}{samples}")

    Path("data/margin-rule-scores.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
