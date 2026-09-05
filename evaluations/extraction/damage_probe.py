r"""Damage no whitespace rule can repair, and what each stage does with it.

The relaxations widen separators. They read `550  U.S.  544` and `367  P.3d  at
74` because the damage is *between* the parts. They cannot read `550 US 544`,
because the damage is *inside* one -- a missing period is not whitespace, and
widening a gap never puts a character back.

So this is a synthetic document of damage of that second kind, one case per
line, with what each citation actually is recorded beside it. Three questions
are asked of every case, in the order the pipeline asks them:

1.  **Does extraction miss it?** If not, the case is not testing what it claims,
    and the probe says so rather than scoring a rule that already works.
2.  **Is a candidate proposed, and by which stage?** The strict stage searches
    for the gazetteer's own spellings, so damage *inside* a reporter is
    invisible to it; the fuzzy stage catches what reduces to a reporter once
    punctuation stops mattering. A case that reaches no stage cannot be
    recovered however good the reviewer is, and that is a property of the layer
    worth stating plainly.
3.  **Does a re-read settle it without a call?** Capitalisation is the only
    damage a widened rule can still forgive, and those sites never reach a
    reviewer.
4.  **Does the reviewer recover it, and does eyecite then parse it?** A
    recovered locator is only worth having if it becomes an ordinary citation,
    so the reviewer's repaired parts are substituted back and re-read. And, for
    the cases that are not citations at all, does the reviewer refuse?

The document is small and hand-made. It measures what the layer *can* do, not
how often that happens -- for how often, see
`exploration/notes/does-adjudication-earn-its-place.md`, which runs the same
layer over a corpus with annotated ground truth.

    uv run python -m evaluations.extraction.damage_probe
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from mellea_lrc.core.citations import FullCaseCitation, ShortCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.adjudication import (
    adjudicate_locator,
    mask_locator_spans,
    promote_locator,
    reread_site,
    suspected_locators,
)
from mellea_lrc.llm import start_mellea_session_from_env


@dataclass(frozen=True, slots=True)
class Case:
    """One damaged citation, and what it would be if it were read."""

    label: str
    sentence: str
    expect: tuple[str, str, str] | None
    """Volume, reporter and page -- or None when the case is not a citation."""


CASES: tuple[Case, ...] = (
    # -- damage inside the reporter, which no separator rule reaches ----------
    Case(
        "periods gone from the reporter",
        "The Court so held. Roe v. Wade, 410 US 113, 116 (1973).",
        ("410", "US", "113"),
    ),
    Case("commas for periods", "See Ashcroft v. Iqbal, 556 U,S, 662, 678 (2009).", ("556", "U,S,", "662")),
    Case(
        "period gone from the series",
        "See Doe v. Megless, 654 F3d 404, 408 (3d Cir. 2011).",
        ("654", "F3d", "404"),
    ),
    # -- damage inside a number, which is not a separator either --------------
    Case(
        "letter l for 1 in the page",
        "See DCD Programs v. Leighton, 833 F.2d l83, 186 (9th Cir. 1987).",
        ("833", "F.2d", "l83"),
    ),
    Case("letter O for 0 in the volume", "See Cates v. Wilson, 32O N.C. 1, 4 (1987).", ("32O", "N.C.", "1")),
    # -- a short form damaged the same way ------------------------------------
    Case("short form, periods gone", "The standard is settled. Iqbal, 556 US at 678.", ("556", "US", "678")),
    # -- controls: the rules should already read these ------------------------
    Case("clean, a control", "See Brown v. Board, 347 U.S. 483, 495 (1954).", ("347", "U.S.", "483")),
    Case("doubled spaces, a control", "See Twombly,  550  U.S.  544,  570  (2007).", ("550", "U.S.", "544")),
    # -- not citations: the reviewer must refuse ------------------------------
    Case("a statute", "Sanctions are sought under 28 U.S.C. § 1927 and the inherent power.", None),
    Case("a letterhead", "Counsel of record, P.O. Box 944 Corrales, NM 87048 (505) 220-5691.", None),
    Case(
        "a procedural rule", "Disclosure is governed by FED. R. CIV. P. 26 and this District's LCR 15.", None
    ),
    Case("a docket number", "The related matter is No. 1:19-CV-362 (E.D.N.Y. filed Jan. 3, 2019).", None),
)

DOCUMENT = "\n\n".join(f"{index}. {case.sentence}" for index, case in enumerate(CASES, start=1))


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


def _case_of(offset: int) -> Case | None:
    """Which case a document offset falls in."""
    for index, case in enumerate(CASES, start=1):
        start = DOCUMENT.find(f"{index}. {case.sentence}")
        if start <= offset < start + len(case.sentence) + len(str(index)) + 2:
            return case
    return None


async def main_async() -> int:
    """Report what extraction, the generator and the reviewer each do."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(DOCUMENT, relaxation=Relaxation.FULL)

    read: dict[str, str] = {}
    for item in document.citations:
        if isinstance(item.citation, FullCaseCitation | ShortCaseCitation):
            case = _case_of(item.locator_span.start)
            if case:
                read[case.label] = item.matched_text

    masked = mask_locator_spans(document)
    sites = list(suspected_locators(document))
    proposed: dict[str, list] = {}
    for site in sites:
        case = _case_of(site.span_start)
        if case:
            proposed.setdefault(case.label, []).append(site)

    session = start_mellea_session_from_env()
    recovered: dict[str, list[str]] = {}
    reread: dict[str, list[str]] = {}
    for label, group in proposed.items():
        for site in group:
            settled = reread_site(DOCUMENT, site)
            if settled is not None:
                reread.setdefault(label, []).append(settled.matched_text)
                continue
            try:
                found = await adjudicate_locator(masked, site, session=session)
            except Exception as error:
                recovered.setdefault(label, []).append(f"<raised {type(error).__name__}>")
                continue
            for item in found:
                # The reviewer's answer is not the end of it: what enters a
                # record is a citation object, and only eyecite makes one.
                citation = promote_locator(DOCUMENT, item)
                parsed = "parsed" if citation is not None else "UNPARSED"
                repaired = " [repaired]" if item.repaired else ""
                recovered.setdefault(label, []).append(
                    f"{item.volume} {item.reporter} {item.page}{repaired} -> {parsed}"
                )

    print(f"{'case':<32}{'expected':<20}{'rules read':<18}{'stages':<16}{'re-read':<14}reviewer")
    print("-" * 132)
    for case in CASES:
        expected = " ".join(case.expect) if case.expect else "-- not a citation --"
        group = proposed.get(case.label, [])
        stages = ",".join(sorted({site.stage.value for site in group})) or "none"
        print(
            f"{case.label:<32}{expected:<20}{read.get(case.label, '')!s:<18}"
            f"{f'{stages} ({len(group)})':<16}{reread.get(case.label, '') or ''!s:<14}"
            f"{recovered.get(case.label, [])}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    _load_env()
    _quiet()
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
