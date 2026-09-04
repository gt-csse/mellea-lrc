r"""Does the adjudication layer find anything more, and does it reject the rest?

The layer proposes candidates the deterministic pass did not record and asks a
model about each. Two questions decide whether it is worth wiring in, and they
pull against each other:

**Does it recover anything?** `false-citation-bench-locator-only-v2.0` is
annotated and inclusive -- a locator the filing states is ground truth whether or
not any tokenizer reaches it. Extraction finds 583 of its 586. The three it
misses are the only citations there are to recover, so a recovered locator is
one of those three and nothing else can count.

**Does it reject the rest?** Most candidates are not citations. On the bench the
site generator proposes 65 windows, and the ones it proposes are dominated by
letterheads, procedural rules and legislative journals. A reviewer that accepts
those is worse than no reviewer, because a spurious locator enters the record
looking exactly like a parsed one.

Both are scored against the same ground truth, so a proposal is exactly one of:
recovered (in the annotation, not extracted), spurious (not in the annotation),
or duplicate (already extracted, which the masking should prevent).

    uv run python -m evaluations.extraction.adjudication_yield
    uv run python -m evaluations.extraction.adjudication_yield --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
from collections import Counter
from pathlib import Path

from evaluations.extraction.matrix import BENCH, BENCH_TRUTH, body
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.adjudication import (
    adjudicate_locator,
    mask_locator_spans,
    suspected_locators,
)
from mellea_lrc.extraction.adjudication.candidates import orphan_short_forms, uppercase_reporters
from mellea_lrc.llm import start_mellea_session_from_env

CONCURRENCY = 6


def _load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _quiet() -> None:
    for name in ("mellea", "mellea.backends", "httpx", "openai"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _truth() -> set[tuple[str, int, int]]:
    """Every locator the bench filings state, annotated by hand."""
    stated = set()
    for line in BENCH_TRUTH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            stated.add((record["document"], record["span"]["start"], record["span"]["end"]))
    return stated


async def _judge(session, semaphore, name, text, site):
    async with semaphore:
        try:
            return name, site, await adjudicate_locator(text, site, session=session)
        except Exception as error:  # a provider failure is a result, not a crash
            return name, site, error


async def main_async(limit: int | None) -> int:
    """Propose on the bench, review every proposal, score against the annotation."""
    truth = _truth()
    session = start_mellea_session_from_env()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    jobs, texts, extracted, proposals = [], {}, set(), Counter()
    for path in sorted(BENCH.glob("*.txt")):
        text = body(path)
        texts[path.name] = text
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        for item in document.citations:
            if isinstance(item.citation, FullCaseCitation):
                extracted.add((path.name, item.locator_span.start, item.locator_span.end))
        sites = list(suspected_locators(document))
        proposals["reporter sites"] += len(sites)
        proposals["uppercase reporters"] += len(list(uppercase_reporters(document)))
        proposals["orphan short forms"] += len(list(orphan_short_forms(document)))
        # The reviewer sees every other citation blanked, so it cannot quote
        # one the record already holds.
        masked = mask_locator_spans(document)
        jobs.extend((path.name, masked, site) for site in sites)

    if limit:
        jobs = jobs[:limit]

    print(f"candidates proposed: {dict(proposals)}")
    print(f"reviewing {len(jobs)} reporter sites\n")

    results = await asyncio.gather(
        *(_judge(session, semaphore, name, text, site) for name, text, site in jobs)
    )

    counts: Counter = Counter()
    recovered, spurious, failures = [], [], []
    for name, site, outcome in results:
        if isinstance(outcome, Exception):
            counts["call raised"] += 1
            failures.append((name[:12], site.reporter, type(outcome).__name__, str(outcome)[:300]))
            continue
        if not outcome:
            counts["declined -- no locator in the window"] += 1
            continue
        counts["accepted at least one locator"] += 1
        for found in outcome:
            key = (name, found.span.start, found.span.end)
            text = texts[name]
            if key in truth and key not in extracted:
                counts["  recovered a stated locator"] += 1
                recovered.append((name, found.text))
            elif key in extracted:
                counts["  already extracted"] += 1
            else:
                counts["  spurious"] += 1
                spurious.append(
                    (
                        name[:12],
                        found.text,
                        " ".join(text[max(0, found.span.start - 40) : found.span.end + 30].split()),
                    )
                )

    print(f"{'':<44}{'sites':>8}")
    for label, value in counts.items():
        print(f"  {label:<42}{value:>8}")
    missed = sorted(truth - extracted)
    print(f"\nlocators the bench states and extraction misses: {len(missed)}")
    for name, start, end in missed:
        print(f"    {name[:14]} [{start}:{end}] {texts[name][start:end]!r}")
    print(f"\nrecovered: {recovered}")
    if failures:
        print("\nevery failed call:")
        for row in failures:
            print(f"    {row[0]:<14}{row[1]!r:<12}{row[2]}: {row[3]}")
    print("\nspurious, all of them:")
    for row in spurious:
        print(f"    {row[0]:<14}{row[1]!r:<28}{row[2][:74]!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    _load_env()
    _quiet()
    return asyncio.run(main_async(args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
