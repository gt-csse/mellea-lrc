"""Unit tests for locator parsing and deterministic span grounding."""

from pathlib import Path

import pytest

from mellea_lrc.experimental.llm_only_extraction.examples import (
    LOCATOR_EXAMPLES,
    render_locator_examples,
)
from mellea_lrc.experimental.llm_only_extraction.locators import (
    find_locators,
    parse_locator,
)
from mellea_lrc.experimental.llm_only_extraction.matching import (
    MatchMethod,
    locate_locator,
    locate_text,
)


@pytest.fixture
def korematsu_text() -> str:
    """Return the Korematsu text."""
    return (Path(__file__).parent / "data" / "Korematsu_v_US.txt").read_text()


def load_citations(file_name: str) -> list[tuple[str, list[int]]]:
    """Return each fixture citation with its expected [start, end] span."""
    path = Path(__file__).parent / "data" / file_name
    rows = [
        [part.strip() for part in line.split("---")] for line in path.read_text().splitlines() if line.strip()
    ]
    return [(citation, [int(offset) for offset in spans.split()]) for citation, spans in rows]


# --- locator grammar ---


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Hirabayashi v. United States, 320 U.S. 81", "320 U.S. 81"),
        # The pin cite is not part of the locator.
        ("Chastleton Corporation v. Sinclair, 264 U.S. 543, 547", "264 U.S. 543"),
        # Series markers stay attached to the reporter; the page does not.
        ("Thompson v. Hous. Auth., 782 F.2d 829, 831 (9th Cir.1986)", "782 F.2d 829"),
        ("8 L.Ed.2d 734 (1962)", "8 L.Ed.2d 734"),
        # Nineteenth-century nominative reporters.
        ("United States v. Russell, 13 Wall. 623, 627-8", "13 Wall. 623"),
        # Spaced reporter abbreviation.
        ("320 U. S. 81", "320 U. S. 81"),
        # No reporter: the docket number is the locator.
        ("Adams v. Cal. Dep't, No. CV-03-8920 (C.D. Cal. 2003)", "No. CV-03-8920"),
        # PDF extraction eats the space before a Westlaw page routinely.
        ("Doe v. Roe, 2010 WL4722279, at *3", "2010 WL4722279"),
        ("Doe v. Roe, 2010 WL 4722279", "2010 WL 4722279"),
        # The Federal Appendix carries an apostrophe.
        ("Smith v. Jones, 30 F. App'x 931, 933", "30 F. App'x 931"),
        ("Smith v. Jones, 534 Fed. App'x 454", "534 Fed. App'x 454"),
        # A series marker whose period was stranded by the extraction.
        ("Matter of Doe, 58  N.Y .2d  916", "58  N.Y .2d  916"),
    ],
)
def test_parse_locator_extracts_the_locator(source: str, expected: str) -> None:
    """The parser keeps volume, reporter, and first page, and nothing else."""
    locator = parse_locator(source)
    assert locator is not None
    assert locator.text == expected


@pytest.mark.parametrize(
    "source",
    [
        "Hirabayashi v. United States",  # no locator at all
        "320 U.S.",  # volume and reporter, page missing
        "320 U.S. *.",  # page replaced by noise
        "Fed. R. Civ. P. 8(a)(2)",  # procedural rule, not a citation
        "",
    ],
)
def test_parse_locator_rejects_incomplete_shapes(source: str) -> None:
    """A string missing any locator component is refused, not partially matched."""
    assert parse_locator(source) is None


def test_locator_span_points_into_the_parsed_string() -> None:
    """The reported span slices the locator back out of its source."""
    source = "Blockburger v. United States, 284 U.S. 299, 304"
    locator = parse_locator(source)
    assert locator is not None
    assert source[locator.span.start : locator.span.end] == locator.text


def test_find_locators_returns_every_parallel_citation() -> None:
    """Parallel reporters each yield their own locator, in order."""
    parallel = "370 U.S. 626, 629-30, 82 S.Ct. 1386, 8 L.Ed.2d 734 (1962)"
    assert [item.text for item in find_locators(parallel)] == [
        "370 U.S. 626",
        "82 S.Ct. 1386",
        "8 L.Ed.2d 734",
    ]


def test_locator_key_ignores_reporter_spelling_but_not_digits() -> None:
    """Reporter style folds together; volume and page never do.

    The folding must reach every non-alphanumeric character, because the
    benchmark compares identifiers that way: ``F. App'x`` and ``F.Appx`` are
    the same reporter and must not be scored as different citations.
    """
    assert parse_locator("320 U.S. 81").key == parse_locator("320 U. S. 81").key
    assert parse_locator("30 F. App'x 931").key == parse_locator("30 F.Appx 931").key
    assert parse_locator("2010 WL4722279").key == parse_locator("2010 WL 4722279").key
    assert parse_locator("320 U.S. 81").key != parse_locator("320 U.S. 82").key
    assert parse_locator("320 U.S. 81").key != parse_locator("3320 U.S. 81").key


# --- match tiers ---


def test_exact_match_is_preferred(korematsu_text: str) -> None:
    """A verbatim proposal grounds at the exact tier with a full score."""
    located = locate_text(korematsu_text, "Hirabayashi v. United States, 320 U.S. 81")
    assert located is not None
    assert located.method is MatchMethod.EXACT
    assert located.score == 1.0
    assert (located.span.start, located.span.end) == (4120, 4161)


@pytest.mark.parametrize(
    "proposal",
    [
        "Sterling    v. Constantin, 287  U.S. 378, 401",  # collapsed whitespace
        "sterling v. constantin, 287 U.S. 378, 401",  # letter case
        "Sterling v. Constantin, 287 U.S. 378, 401",  # already exact
    ],
)
def test_surface_drift_grounds_deterministically(korematsu_text: str, proposal: str) -> None:
    """Whitespace and case drift resolve without reaching the fuzzy tier."""
    located = locate_text(korematsu_text, proposal)
    assert located is not None
    assert located.method in {MatchMethod.EXACT, MatchMethod.NORMALIZED}
    assert located.score == 1.0
    assert (located.span.start, located.span.end) == (36572, 36613)


def test_located_text_is_sliced_from_the_source(korematsu_text: str) -> None:
    """The returned text is document text, never the model's proposal."""
    proposal = "Sterling    v. Constantin, 287  U.S. 378, 401"
    located = locate_text(korematsu_text, proposal)
    assert located is not None
    assert located.text != proposal
    assert located.text == korematsu_text[located.span.start : located.span.end]


def test_absent_text_grounds_nowhere(korematsu_text: str) -> None:
    """A citation that is not in the document returns None rather than a guess."""
    assert locate_text(korematsu_text, "Fabricated v. Nobody, 999 U.S. 111") is None


@pytest.mark.parametrize(
    "proposal",
    [
        "Hirabayashi v. United States, 3320 U.S. 81",  # volume corrupted
        "Hirabayashi v. United States, 320 U.S. 82",  # page corrupted
        "United States v. Russell, 13 Wall. 23, 627-8",  # page truncated
    ],
)
def test_fuzzy_tier_refuses_digit_drift(korematsu_text: str, proposal: str) -> None:
    """Digits are identity: a corrupted number is never fuzzed back into a match."""
    assert locate_text(korematsu_text, proposal) is None


def test_cursor_walks_successive_occurrences(korematsu_text: str) -> None:
    """Passing the previous end as `start` resolves repeats to distinct spans."""
    proposal = "Hirabayashi v. United States, 320 U.S. 81"
    spans = []
    cursor = 0
    for _ in range(4):
        located = locate_text(korematsu_text, proposal, start=cursor)
        assert located is not None
        spans.append((located.span.start, located.span.end))
        cursor = located.span.end
    assert spans == [(4120, 4161), (17952, 17993), (20999, 21040), (54613, 54654)]
    assert locate_text(korematsu_text, proposal, start=cursor) is None


# --- locator grounding ---


def test_locate_locator_grounds_the_locator_only(korematsu_text: str) -> None:
    """Grounding a full citation returns the span of its locator."""
    located = locate_locator(korematsu_text, "Hirabayashi v. United States, 320 U.S. 81")
    assert located is not None
    assert located.text == "320 U.S. 81"


def test_locate_locator_tolerates_party_name_corruption(korematsu_text: str) -> None:
    """A damaged party name does not prevent grounding an intact locator.

    Party names are validation's concern (`exact_case_name_check`); the
    extractor's job is the locator, and this locator is undamaged.
    """
    located = locate_locator(korematsu_text, "Hirabasii v. United States, 320 U.S. 81")
    assert located is not None
    assert located.text == "320 U.S. 81"


@pytest.mark.parametrize(
    "proposal",
    [
        "Hirabayashi v. United States, 320 U.SS. 81",  # reporter corrupted
        "Hirabayashi v. United States, 3320 U.SS. 81",  # volume corrupted
        "Bridge Co. v. United States, 21 U.S. 177",  # volume truncated
        "UnitedStates v.. Russell, 13 Wall. 23, 627-8",  # page truncated
        "Hirabayashi v. United States, 320 U.S. *.",  # page destroyed
    ],
)
def test_locate_locator_refuses_corrupted_locators(korematsu_text: str, proposal: str) -> None:
    """Every corrupted-locator row of the fixture is refused.

    These are the rows of ``Incorrect_Citations_Korematsu_v_US.txt`` whose
    *locator* is damaged. The file's remaining rows carry intact locators with
    damaged surroundings (stray punctuation, added parentheses); whether those
    should ground is an open product question, so they are not asserted here.
    """
    assert locate_locator(korematsu_text, proposal) is None


def test_every_correct_fixture_citation_grounds(korematsu_text: str) -> None:
    """All sixteen known-good citations ground at their recorded first occurrence."""
    for citation, (start, end) in load_citations("Correct_Citations_Korematsu_v_US.txt"):
        located = locate_text(korematsu_text, citation)
        assert located is not None, citation
        assert (located.span.start, located.span.end) == (start, end), citation


# --- prompt examples ---


def test_examples_agree_with_the_parser() -> None:
    """Each hand-written example answer matches what the parser derives.

    This keeps the few-shot prompt and the deterministic grammar from drifting
    apart: an example the parser disagrees with is teaching the model something
    the extractor will then reject.
    """
    for example in LOCATOR_EXAMPLES:
        locator = parse_locator(example.source)
        assert locator is not None, example.source
        assert locator.text == example.locator, example.source


def test_rendered_examples_carry_every_pair() -> None:
    """The rendered prompt block contains each example's source and answer."""
    rendered = render_locator_examples()
    for example in LOCATOR_EXAMPLES:
        assert example.locator in rendered
    assert rendered.count("Locator:") == len(LOCATOR_EXAMPLES)
