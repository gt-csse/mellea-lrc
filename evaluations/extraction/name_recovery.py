"""Score the case-name recovery layer against the citation tree.

`case_name_sites` proposes every case named where the deterministic pass read
no citation, and `adjudicate_case_name` says which of three things each one is.
This runs both over a corpus and reports what the readings are worth, against
what `extraction-v3.0` says is there.

The ground truth for each site is the row whose span it covers:

    a `citation` row of kind `ReferenceCitation` with no locator
                     a bare short form. The reading should be `short_form` and
                     the root it names should be the row's `root_id`. A site
                     whose root is the citation written beside it comes back as
                     `misread_citation` instead, which is the same finding in
                     different words, so both count as recovered and only the
                     root is scored where one was named

    a `nonconforming_citation` row
                     a case offered as authority the filing never locates. The
                     reading should be `uncited_case`

    anything else    a name the dataset does not annotate: the filing's own
                     caption, a heading, or part of a citation that was read.
                     `misread_citation` and `not_a_citation` are both defensible
                     there, so what is counted is the two answers that are not:
                     `short_form` naming a root nowhere near it, a citation
                     invented out of a name, and `uncited_case`, a defect
                     reported against a filing that does not have one

Every metric is counted against the ground truth's own denominator, and detail
lines are indented with `- `, the same as `tree.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluations.extraction.tree import read_documents
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.adjudication.candidates.case_name_sites import case_name_sites
from mellea_lrc.extraction.adjudication.review.case_name import (
    AdjudicatedCaseName,
    Reading,
    adjudicate_case_name,
)
from mellea_lrc.llm import start_mellea_session_from_env


def annotation(rows: list[dict[str, Any]], start: int, end: int) -> dict[str, Any] | None:
    """The annotated row a site covers, if the dataset holds one."""
    for row in rows:
        name = row.get("named_as") or row.get("case_name")
        if name and name["start"] < end and start < name["end"]:
            return row
    return None


def expected(row: dict[str, Any] | None) -> Reading | None:
    """What the dataset says this site is, or `None` when it annotates none."""
    if row is None:
        return None
    if row["unit"] == "nonconforming_citation":
        return Reading.UNCITED_CASE
    if row["unit"] == "citation" and row["kind"] == "ReferenceCitation" and "locator" not in row:
        return Reading.SHORT_FORM
    return None


async def review(
    dataset: Path,
    corpus: Path,
    relaxation: Relaxation,
    limit: int | None,
    out: Path | None = None,
    *,
    ordered: bool = True,
) -> tuple[Counter[str], dict[str, list[str]]]:
    """Run every site in the corpus past a reader and score the answers."""
    counts: Counter[str] = Counter()
    detail: dict[str, list[str]] = {
        "declined": [],
        "recovered": [],
        "root_right": [],
        "uncited_right": [],
        "invented": [],
        "false_defect": [],
    }
    session = start_mellea_session_from_env()
    for header, body in read_documents(dataset):
        text = (corpus / header["document"]).read_text(encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=relaxation)
        by_locator = {
            (row["locator"]["start"], row["locator"]["end"]): row for row in body if row.get("locator")
        }
        sites = list(case_name_sites(document))
        if limit is not None:
            sites = sites[:limit]
        for site in sites:
            row = annotation(body, site.span.start, site.span.end)
            want = expected(row)
            counts["site"] += 1
            if want is Reading.SHORT_FORM:
                counts["short_form"] += 1
            elif want is Reading.UNCITED_CASE:
                counts["nonconforming"] += 1
            else:
                counts["unannotated"] += 1

            answer = await adjudicate_case_name(document, site, session=session, ordered=ordered)
            if out is not None:
                _record(out, header, text, site, row, want, answer)
            if answer is None:
                counts["declined"] += 1
                detail["declined"].append(
                    f"{header['document'][:3]} {text[site.span.start : site.span.end][:40]!r}"
                )
                continue
            _score(counts, detail, header, text, site, row, want, answer, by_locator, document)
    return counts, detail


def _record(
    out: Path,
    header: dict[str, Any],
    text: str,
    site: Any,
    row: dict[str, Any] | None,
    want: Reading | None,
    answer: AdjudicatedCaseName | None,
) -> None:
    """Append one answer, so a run can be re-scored without calling a model again."""
    with out.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "document": header["document"],
                    "site": {"start": site.span.start, "end": site.span.end},
                    "text": text[site.span.start : site.span.end],
                    "annotated": None if row is None else {"id": row["id"], "unit": row["unit"]},
                    "expected": None if want is None else want.value,
                    "answer": None
                    if answer is None
                    else {
                        "reading": answer.reading.value,
                        "name": answer.name,
                        "span": {"start": answer.span.start, "end": answer.span.end},
                        "citation_id": answer.citation_id,
                        "root_id": answer.root_id,
                        "reason": answer.reason,
                    },
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _score(
    counts: Counter[str],
    detail: dict[str, list[str]],
    header: dict[str, Any],
    text: str,
    site: Any,
    row: dict[str, Any] | None,
    want: Reading | None,
    answer: AdjudicatedCaseName,
    by_locator: dict[tuple[int, int], dict[str, Any]],
    document: Any,
) -> None:
    """Count one answer against what the dataset says the site is."""
    where = f"{header['document'][:3]} {text[site.span.start : site.span.end][:40]!r}"
    if want is Reading.SHORT_FORM:
        if answer.reading not in {Reading.SHORT_FORM, Reading.MISREAD_CITATION}:
            detail["recovered"].append(f"{where} read as {answer.reading.value}")
            return
        counts["recovered"] += 1
        if answer.reading is not Reading.SHORT_FORM:
            detail["root_right"].append(f"{where} read as misread_citation, so no root was named")
            return
        counts["root_named"] += 1
        chosen = next((c for c in document.citations if c.citation_id == answer.root_id), None)
        annotated = by_locator.get(
            (chosen.locator_span.start, chosen.locator_span.end) if chosen else (-1, -1)
        )
        if annotated is not None and annotated["id"] == row["root_id"]:
            counts["root_right"] += 1
        else:
            named = annotated["id"] if annotated else "a citation the dataset has no row for"
            detail["root_right"].append(f"{where} -> {named}, annotated {row['root_id']}")
        return
    if want is Reading.UNCITED_CASE:
        if answer.reading is Reading.UNCITED_CASE:
            counts["uncited_right"] += 1
        else:
            detail["uncited_right"].append(f"{where} read as {answer.reading.value}")
        return
    if answer.reading is Reading.SHORT_FORM:
        counts["invented"] += 1
        detail["invented"].append(f"{where} read as a short form of a distant root")
    elif answer.reading is Reading.UNCITED_CASE:
        counts["false_defect"] += 1
        detail["false_defect"].append(f"{where} read as a case the filing never cites")


def report(counts: Counter[str], detail: dict[str, list[str]]) -> str:
    """A line per metric, and its disagreements under it."""
    lines = [
        f"{'sites':<22}{counts['site']}",
        f"- {'bare short forms':<20}{counts['short_form']}",
        f"- {'uncited cases':<20}{counts['nonconforming']}",
        f"- {'not annotated':<20}{counts['unannotated']}",
        f"{'declined':<22}{counts['declined']}/{counts['site']}",
        *[f"- {line}" for line in detail["declined"]],
        f"{'recovered':<22}{counts['recovered']}/{counts['short_form']}",
        *[f"- {line}" for line in detail["recovered"]],
        f"{'root_right':<22}{counts['root_right']}/{counts['root_named']}",
        *[f"- {line}" for line in detail["root_right"]],
        f"{'uncited_right':<22}{counts['uncited_right']}/{counts['nonconforming']}",
        *[f"- {line}" for line in detail["uncited_right"]],
        f"{'invented':<22}{counts['invented']}/{counts['unannotated']}",
        *[f"- {line}" for line in detail["invented"]],
        f"{'false_defect':<22}{counts['false_defect']}/{counts['unannotated']}",
        *[f"- {line}" for line in detail["false_defect"]],
    ]
    return "\n".join(lines)


def main() -> None:
    """Run the command-line reviewer."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True, help="extraction-v3.0/documents/")
    parser.add_argument("--documents", type=Path, required=True, help="The text those spans index.")
    parser.add_argument(
        "--relaxation", default="FULL", choices=[level.name for level in Relaxation], help="Tokenizer."
    )
    parser.add_argument("--limit", type=int, default=None, help="Sites per document, for a short run.")
    parser.add_argument("--out", type=Path, default=None, help="Write every answer to this JSONL.")
    parser.add_argument(
        "--unordered",
        action="store_true",
        help="Let a short form stand before the full citation it refers to.",
    )
    args = parser.parse_args()

    if args.out is not None and args.out.exists():
        args.out.unlink()
    counts, detail = asyncio.run(
        review(
            args.dataset,
            args.documents,
            Relaxation[args.relaxation],
            args.limit,
            args.out,
            ordered=not args.unordered,
        )
    )
    print(report(counts, detail))


if __name__ == "__main__":
    main()
