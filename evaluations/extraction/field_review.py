r"""The matrix column a rule cannot fill: is a recorded field what the page says?

Every column in `matrix.py` is presence, absence, or a defect a rule can detect
from the document's own structure. None of them checks that a field which *is*
recorded matches the text it was read from. Nothing deterministic can: there is
no annotated field-level ground truth for either corpus.

A reader can, cheaply, on a sample. This asks one, and the question is kept
strictly inside what extraction is scored against.

## The question, and the line it does not cross

**Does the document state this value for this citation?** Nothing else.

The reviewer is told, in the instruction, not to judge whether the citation is
accurate, whether the case exists, or whether the year is really that decision's
year. A filing that writes `550 U.S. 544 (2009)` states 2009, and recording 2009
is a correct read. Whether Twombly was decided in 2009 is validation's question,
answered against validation's own ground truth, and asking it here would score
extraction for failing to repair a filing -- which is the one thing this project
must not do.

## What it produces

Per field, over a seeded sample: how often the recorded value is what the window
states, how often it differs, and how often the reviewer cannot find it there at
all. The third is the interesting one -- a value present in the record and
absent from the text is the shape of a field taken from somewhere else.

    uv run python -m evaluations.extraction.field_review --sample 40
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import logging
import os
import random
from collections import Counter
from pathlib import Path
from typing import Literal

from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict

from evaluations.extraction.matrix import BENCH, MINED, body
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)

WINDOW = 170
MAX_TOKENS = 400
CONCURRENCY = 6

INSTRUCTION = """
You are checking whether a citation parser read a document correctly.

Below is a window of text from a legal filing, and the values a parser recorded
for one citation inside it. For each value, say whether the window states that
value for that citation.

Judge only what the text says. Do not judge whether the citation is accurate,
whether the case exists, or whether the year is really that decision's year. If
the filing states a year, that year is what the parser should have recorded,
even if you believe the filing is wrong.

Use "matches" when the window states that value for this citation, allowing for
spacing and line breaks the PDF extraction introduced. Use "differs" when the
window states something else. Use "absent" when the window does not state this
value for this citation at all. Use "not_recorded" for any field that does not
appear in the recorded values below.

citation as matched: {{matched}}

window:
{{window}}

recorded values:
{{fields}}
""".strip()

# `not_recorded` exists because the provider's strict schema mode requires every
# property to be required, so a field the parser did not record still needs an
# answer. It is dropped when the verdicts are counted.
_Verdict = Literal["matches", "differs", "absent", "not_recorded"]


class FieldReview(BaseModel):
    """One verdict per field, including the ones the parser did not record."""

    model_config = ConfigDict(extra="forbid")

    volume: _Verdict
    reporter: _Verdict
    page: _Verdict
    pin_cite: _Verdict
    date: _Verdict
    court: _Verdict
    plaintiff: _Verdict
    defendant: _Verdict


def _quiet_mellea() -> None:
    """Mellea logs progress to stdout, which would sit inside the table."""
    for name in ("mellea", "mellea.backends", "httpx", "openai"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _load_env(path: Path = Path(".env")) -> None:
    """Read the API binding from `.env` when it is not already in the environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _court_name(identifier: str | None) -> str | None:
    """The court's name, so a reviewer can judge an identifier like `ca3`."""
    if not identifier:
        return None
    from eyecite.helpers import courts

    for court in courts:
        if str(court["id"]) == identifier:
            return f"{identifier} ({court.get('name') or court['citation_string']})"
    return identifier


def _recorded(citation: FullCaseCitation) -> dict[str, str]:
    """The fields worth asking about, as strings, skipping the ones not recorded."""
    values = {
        "volume": citation.volume,
        "reporter": citation.reporter.as_written if citation.reporter else None,
        "page": citation.page,
        "pin_cite": citation.pin_cite,
        "date": str(citation.date) if citation.date else None,
        "court": _court_name(citation.court),
        "plaintiff": citation.plaintiff,
        "defendant": citation.defendant,
    }
    return {key: value for key, value in values.items() if value}


def _sample(directory: Path, count: int, seed: int) -> list[tuple[str, str, dict[str, str]]]:
    """Pick citations at random, with the window a reviewer is shown."""
    picked: list[tuple[str, str, dict[str, str]]] = []
    for path in sorted(directory.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        for item in document.citations:
            if not isinstance(item.citation, FullCaseCitation):
                continue
            recorded = _recorded(item.citation)
            if len(recorded) < 3:
                continue
            window = text[max(0, item.full_span.start - WINDOW) : min(len(text), item.full_span.end + WINDOW)]
            picked.append((item.matched_text, " ".join(window.split()), recorded))
    random.Random(seed).shuffle(picked)
    return picked[:count]


async def _review(session, options, semaphore, sample) -> FieldReview | None:
    matched, window, recorded = sample
    fields = "\n".join(f"{key}: {value}" for key, value in recorded.items())
    spec = InstructIvrSpec(
        description=INSTRUCTION,
        user_variables={"matched": matched, "window": window, "fields": fields},
        output_format=FieldReview,
    )
    async with semaphore:
        # No `redirect_stdout` here. `sys.stdout` is global, so redirecting it
        # inside concurrent tasks nests wrongly and the restore leaves it
        # pointing at a dead buffer -- which silently swallowed this script's own
        # output the first time. Mellea's logger is quieted instead.
        result = await run_instruct_ivr(
            session, spec, strategy=MultiTurnStrategy(loop_budget=1), model_options=options
        )
    try:
        return FieldReview.model_validate_json(str(result.value))
    except Exception:
        return None


async def _run(directory: Path, count: int, seed: int) -> tuple[Counter, int, int]:
    samples = _sample(directory, count, seed)
    config = llm_api_config_from_env(os.environ)
    session = start_mellea_session_from_env()
    options = config.mellea_call_options(max_tokens=MAX_TOKENS)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    reviews = await asyncio.gather(*(_review(session, options, semaphore, sample) for sample in samples))
    counts: Counter = Counter()
    disputed: list[tuple[str, str, str, str, str]] = []
    for sample, review in zip(samples, reviews, strict=True):
        if review is None:
            continue
        matched, window, recorded = sample
        for field, value in recorded.items():
            verdict = getattr(review, field, None)
            if not verdict or verdict == "not_recorded":
                continue
            counts[(field, verdict)] += 1
            if verdict != "matches":
                disputed.append((field, verdict, value, matched, window))
    return counts, len(samples), sum(1 for r in reviews if r is None), disputed


def main() -> int:
    """Review a sample from each corpus and print per-field agreement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--show", type=int, default=0, help="print this many disagreements")
    args = parser.parse_args()
    _load_env()
    _quiet_mellea()

    for label, directory in (("bench", BENCH), ("mined", MINED)):
        if not directory.exists():
            continue
        counts, asked, failed, disputed = asyncio.run(_run(directory, args.sample, args.seed))
        print(f"\n## {label}: {asked} citations reviewed, {failed} calls unusable\n")
        print(f"{'field':<12}{'matches':>10}{'differs':>10}{'absent':>10}")
        for field in ("volume", "reporter", "page", "pin_cite", "date", "court", "plaintiff", "defendant"):
            row = [counts[(field, v)] for v in ("matches", "differs", "absent")]
            if sum(row):
                print(f"{field:<12}" + "".join(f"{value:>10}" for value in row))
        if args.show and disputed:
            print("\n  every disagreement, for reading:")
            for field, verdict, value, matched, window in disputed[: args.show]:
                print(f"    {field} {verdict}: recorded {value!r} for {matched!r}")
                print(f"      {window[:150]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
