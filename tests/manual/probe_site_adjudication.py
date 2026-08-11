"""Probe the site adjudicator on synthetic damaged windows.

Not a pytest test: this calls a live model, so it is run by hand.

    uv run --env-file .env python tests/manual/probe_site_adjudication.py

Every case here is invented. No text, citation or span comes from the
benchmark, so a passing run says the layer handles a damage class rather than
that it has memorised an answer.
"""

from __future__ import annotations

import asyncio
import sys

from mellea_lrc.experimental.grounded_adjudication.site_adjudication import adjudicate_site
from mellea_lrc.experimental.grounded_adjudication.site_hunting import SuspectedSite

# (label, window text, reporter flagged by the hunter, expected triple or None)
CASES: list[tuple[str, str, str, tuple[str, str, str] | None]] = [
    (
        "clean",
        "See Alder v. Fenwick Holdings, 731 P.2d 902, 907 (Or. 1987) (per curiam).",
        "P.2d",
        ("731", "P.2d", "902"),
    ),
    (
        "doubled spaces",
        "Quoting Marek  v.  Sandoval  Grain  Co. ,  412   P.2d   88 ,  91  (Kan.  1971).",
        "P.2d",
        ("412", "P.2d", "88"),
    ),
    (
        "space before page lost",
        "| Trenholm v. Vasquez Bros. LLC, 1998 WL7654321 (D. Or. Mar. 2, 1998)   | 14 |",
        "WL",
        ("1998", "WL", "7654321"),
    ),
    (
        "page break inside locator",
        "accord Prine v. Halloway Trust, 604 P.2d\n\n318 , 322 (Idaho 1980), and see",
        "P.2d",
        ("604", "P.2d", "318"),
    ),
    (
        "space inside reporter token",
        "citing Okoye v. Brandt Freight, 58  Cal. Rptr .3d  411 (Ct. App. 2007) (dictum).",
        "Cal. Rptr. 3d",
        ("58", "Cal. Rptr. 3d", "411"),
    ),
    (
        "margin line numbers, no page",
        "See Delgado v. Ruiz Textile , 214 P.2d\n\n1\n\n2\n\n3\n\n4\n\n5\n\n6\n\n7",
        "P.2d",
        None,
    ),
    (
        "volume absent entirely",
        "| Whitaker v. Ossining Dev. Corp., WL7654321 (S.D. Cal. 2001) …………… | 9 |",
        "WL",
        None,
    ),
    (
        "street address decoy",
        "MERRIWEATHER & COLE LLP\n\n- 4 902 Harrison Street\n\n- 5 Portland, OR 97204",
        "Harrison",
        None,
    ),
    (
        "statute, out of scope",
        "brought under 42 U.S.C. § 1983 and 28 U.S.C. § 1331 for the reasons stated.",
        "U.S.C.",
        None,
    ),
]


async def main() -> int:
    failures = 0
    for label, window, reporter, expected in CASES:
        site = SuspectedSite(span_start=0, span_end=len(window), reporter=reporter, window=window)
        found = await adjudicate_site(window, site, context=0)
        got = [(f.volume, f.reporter, f.page) for f in found]
        if expected is None:
            ok = not found
            want = "nothing"
        else:
            ok = len(found) == 1 and got[0] == expected
            want = str(expected)
        failures += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<26} want {want:<28} got {got}")
        if found:
            for f in found:
                print(f"            quoted {f.text!r}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(min(asyncio.run(main()), 1))
