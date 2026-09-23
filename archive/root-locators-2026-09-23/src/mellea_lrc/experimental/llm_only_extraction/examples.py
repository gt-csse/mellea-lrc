"""Few-shot locator examples for small-model extraction prompts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LocatorExample:
    """One worked example: source text, and the locator to extract from it."""

    source: str
    locator: str


LOCATOR_EXAMPLES: tuple[LocatorExample, ...] = (
    LocatorExample(
        source="Hirabayashi v. United States, 320 U.S. 81",
        locator="320 U.S. 81",
    ),
    LocatorExample(
        source="Chastleton Corporation v. Sinclair, 264 U.S. 543, 547",
        locator="264 U.S. 543",
    ),
    LocatorExample(
        source="Block v. Hirsh, 256 U.S. 135, 154-5",
        locator="256 U.S. 135",
    ),
    LocatorExample(
        source=(
            "District courts retain broad discretion to control their dockets and "
            '"[i]n the exercise of that power they may impose sanctions including, '
            'where appropriate, default or dismissal." Thompson v. Hous. Auth. of '
            "City of Los Angeles, 782 F.2d 829, 831 (9th Cir.1986) (per curiam);"
        ),
        locator="782 F.2d 829",
    ),
    LocatorExample(
        source=(
            "Hartsel Springs Ranch of Colorado, Inc. v. Bluegreen Corp., "
            '"the fact that plaintiff was denied leave to amend does not give '
            "h[er] the right to file a second lawsuit based on the same\n\n"
            "689\n\n"
            'facts." 296 F.3d 982, 989 (10th Cir.2002)'
        ),
        locator="296 F.3d 982",
    ),
    LocatorExample(
        source=("Adams v. Cal. Dep't of Health Servs., No. CV-03-8920 (C.D. Cal. filed Dec. 8, 2003)"),
        locator="No. CV-03-8920",
    ),
)

# Parked until the locator policy for these two shapes is decided.
UNRESOLVED_LOCATOR_EXAMPLES: tuple[str, ...] = (
    "Link v. Wabash R.R., 370 U.S. 626, 629-30, 82 S.Ct. 1386, 8 L.Ed.2d 734 (1962)",
    (
        "Dismissal under [Federal Rule of Civil Procedure] 12(b)(6) is appropriate only where "
        "the complaint lacks a cognizable legal theory or sufficient facts to support a "
        'cognizable legal theory." Mendiondo v. Centinela Hosp. Med. Ctr., 521 F.3d 1097, 1104 '
        '(9th Cir. 2008). A complaint must contain "a short and plain statement of the claim '
        'showing that the pleader is entitled to relief." Fed. R. Civ. P. 8(a)(2).'
    ),
)


def render_locator_examples(examples: tuple[LocatorExample, ...] = LOCATOR_EXAMPLES) -> str:
    """Render the examples as a labelled block for a few-shot prompt."""
    return "\n\n".join(f"Text: {example.source}\nLocator: {example.locator}" for example in examples)
