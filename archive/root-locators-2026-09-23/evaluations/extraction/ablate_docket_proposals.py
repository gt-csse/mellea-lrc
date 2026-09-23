"""Score the current docket proposal ceiling against saved five-corpus gold.

This is an offline candidate-generation ablation. It replays the deterministic
locator readers and site proposer over the exact text saved in a prior run; it
does not call an LLM. A gold site counted here still needs model admission.

Usage::

    uv run python evaluations/extraction/ablate_docket_proposals.py \
        --output evaluations/extraction/reports/docket-proposal-ablation.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

from mellea_lrc.extraction.adjudication.candidates.docket_sites import suspected_dockets
from mellea_lrc.extraction.locator_stages import find_docket_locators, find_full_reporter_locators
from mellea_lrc.model.citations import DocketCitation
from mellea_lrc.model.document import Document

DEFAULT_SOURCE = Path("data/run-artifacts/1-roots/docket-site-hunting-grounding-flash")


def evaluate(source: Path) -> dict[str, object]:
    """Return exact-span gold coverage and site-review volume for each set."""
    sets: dict[str, object] = {}
    for corpus in sorted(path for path in source.iterdir() if path.is_dir()):
        gold: dict[str, set[tuple[int, int]]] = defaultdict(set)
        saved_admitted = 0
        for line in (corpus / "occurrences.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row.get("annotated_gold"):
                continue
            span = row["span"]
            gold[row["document"]].add((span["start"], span["end"]))
            saved_admitted += row["outcome"] in {"gold_deterministic", "gold_site_admitted"}

        counts = {
            "documents": 0,
            "gold": 0,
            "saved_gold_admitted": saved_admitted,
            "saved_initial_site_candidates": 0,
            "rule_gold": 0,
            "rule_predictions": 0,
            "site_gold_proposed": 0,
            "site_proposals": 0,
            "combined_gold_proposed": 0,
            "combined_predictions": 0,
        }
        misses = []
        for path in sorted((corpus / "documents").glob("*.json")):
            artifact = json.loads(path.read_text(encoding="utf-8"))
            text = artifact["final"]["text"]
            name = artifact["document"]
            # Eyecite may log benign overlap diagnostics on some filings.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                document = find_docket_locators(find_full_reporter_locators(Document.from_plain_text(text)))
            rule = {
                (record.locator_span.start, record.locator_span.end)
                for record in document.citations
                if isinstance(record.fields, DocketCitation)
            }
            sites = {(site.locator_span.start, site.locator_span.end) for site in suspected_dockets(document)}
            targets = gold[name]
            counts["documents"] += 1
            counts["gold"] += len(targets)
            counts["saved_initial_site_candidates"] += len(artifact["initial"]["site_candidates"])
            counts["rule_gold"] += len(targets & rule)
            counts["rule_predictions"] += len(rule)
            counts["site_gold_proposed"] += len(targets & sites)
            counts["site_proposals"] += len(sites)
            counts["combined_gold_proposed"] += len(targets & (rule | sites))
            counts["combined_predictions"] += len(rule | sites)
            misses.extend(
                {"document": name, "span": [start, end], "text": text[start:end]}
                for start, end in sorted(targets - (rule | sites))
            )
        sets[corpus.name] = {"counts": counts, "unproposed_gold": misses}

    totals = {
        key: sum(value["counts"][key] for value in sets.values())
        for key in next(iter(sets.values()))["counts"]
    }
    return {
        "source": str(source),
        "sets": sets,
        "totals": totals,
        "interpretation": "proposal ceiling only; no site-review admissions or false-admission score",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.source)
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(content, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
