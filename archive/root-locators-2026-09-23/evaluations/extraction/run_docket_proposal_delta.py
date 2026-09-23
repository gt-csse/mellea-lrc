"""Review only newly proposed docket sites in filings with saved gold misses.

This scoped live check starts from the two deterministic locator stages, then
replays the *new* site proposals against an older, saved site's initial list.
Each review is applied immediately, so subsequent proposals see the updated
mask. Full Document checkpoints retain the model's IVR attempts and reasons.
No identity validation, court/date reading, or docket audit runs here.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from mellea_lrc.extraction.adjudication.candidates.docket_sites import suspected_dockets
from mellea_lrc.extraction.adjudication.docket_hunting import apply_docket_site_review
from mellea_lrc.extraction.adjudication.review.docket import adjudicate_docket
from mellea_lrc.extraction.locator_stages import find_docket_locators, find_full_reporter_locators
from mellea_lrc.model.document import Document

SOURCE = Path("data/run-artifacts/1-roots/docket-site-hunting-grounding-flash")
OUTPUT = Path("data/run-artifacts/1-roots/docket-proposal-delta-flash")
MISSED = {"gold_not_proposed", "gold_locator_span_mismatch"}


def _misses(corpus: Path) -> dict[str, set[tuple[int, int]]]:
    result: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for line in (corpus / "occurrences.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("annotated_gold") and row["outcome"] in MISSED:
            span = row["span"]
            result[row["document"]].add((span["start"], span["end"]))
    return result


def _source_artifact(corpus: Path, name: str) -> dict:
    return json.loads((corpus / "documents" / f"{Path(name).stem}.json").read_text(encoding="utf-8"))


async def run(source: Path, output: Path) -> dict:
    load_dotenv(".env")
    os.environ["MELLEA_LRC_LLM_MODEL"] = "z-ai/glm-5.3-flash"
    summary = {"model": os.environ["MELLEA_LRC_LLM_MODEL"], "sets": {}, "totals": {}}
    totals = defaultdict(int)
    for corpus in sorted(path for path in source.iterdir() if path.is_dir()):  # noqa: ASYNC240
        missed = _misses(corpus)
        if not missed:
            continue
        counts = defaultdict(int)
        for name, gold in sorted(missed.items()):
            checkpoint = output / corpus.name / "documents" / f"{Path(name).stem}.json"
            if checkpoint.exists():
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                Document.model_validate(payload["final_document"])
            else:
                old = _source_artifact(corpus, name)
                text = old["final"]["text"]
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    document = find_docket_locators(
                        find_full_reporter_locators(Document.from_plain_text(text))
                    )
                initial_document = document.model_dump(mode="json")
                old_sites = {
                    (item["span"]["start"], item["span"]["end"]) for item in old["initial"]["site_candidates"]
                }
                sites = suspected_dockets(document)
                target = {
                    (site.locator_span.start, site.locator_span.end)
                    for site in sites
                    if (site.locator_span.start, site.locator_span.end) not in old_sites
                }
                decisions = []
                for start, end in sorted(target):
                    site = next(
                        (
                            item
                            for item in suspected_dockets(document)
                            if (item.locator_span.start, item.locator_span.end) == (start, end)
                        ),
                        None,
                    )
                    if site is None:
                        decisions.append({"span": [start, end], "outcome": "masked_by_prior_admission"})
                        continue
                    review = await adjudicate_docket(site)
                    document = apply_docket_site_review(document, site, review)
                    decisions.append(
                        {
                            "span": [start, end],
                            "text": site.locator_text,
                            "gold": (start, end) in gold,
                            "outcome": "accepted" if review.answer is not None else "declined",
                            "reason": review.reason,
                            "model_attempts": len(review.run.attempts),
                            "model": review.run.model,
                        }
                    )
                payload = {
                    "schema_version": 1,
                    "source_artifact": str(corpus / "documents" / f"{Path(name).stem}.json"),
                    "document": name,
                    "gold_missed_spans": [list(span) for span in sorted(gold)],
                    "initial_document": initial_document,
                    "final_document": document.model_dump(mode="json"),
                    "decisions": decisions,
                }
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                temporary.replace(checkpoint)

            decisions = payload["decisions"]
            accepted = {tuple(item["span"]) for item in decisions if item["outcome"] == "accepted"}
            counts["documents"] += 1
            counts["gold"] += len(gold)
            counts["selected_proposals"] += len(decisions)
            counts["model_attempts"] += sum(item.get("model_attempts", 0) for item in decisions)
            counts["accepted_gold"] += len(accepted & gold)
            counts["false_admissions"] += len(accepted - gold)
            counts["missed_gold"] += len(gold - accepted)
            print(
                f"{corpus.name}/{name}: accepted {len(accepted & gold)}/{len(gold)} gold, "
                f"{len(accepted - gold)} false; {len(decisions)} reviews",
                flush=True,
            )
        summary["sets"][corpus.name] = dict(counts)
        for key, value in counts.items():
            totals[key] += value
    summary["totals"] = dict(totals)
    output.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.source, args.output)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
