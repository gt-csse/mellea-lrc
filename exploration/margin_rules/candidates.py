"""Candidate margin-detection rules, written against cached page layout.

Each rule takes one document's text items -- plain dicts of text, label, page
and bounding box, as `scripts/dump_page_layout.py` records them -- and returns
the indices it judges to be margin numbers. Working on dicts rather than a
`DoclingDocument` keeps a variant to a few lines and lets the whole corpus be
scored without reconverting anything.

The shipped rule is `current`. Everything after it is an experiment, and the
comment above each says what evidence it adds and what it risks.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from statistics import median

Item = dict
Items = list[Item]

MAX_LINE_NUMBER = 99
ALIGNMENT_SLACK = 5.0


# --- shared primitives -------------------------------------------------------


def line_number_value(text: str) -> int | None:
    """The integer a margin number would carry, if this text is one at all."""
    stripped = text.strip().strip("-").strip()
    if not stripped.isdigit():
        return None
    value = int(stripped)
    return value if 1 <= value <= MAX_LINE_NUMBER else None


def by_page(items: Items) -> dict[int, list[int]]:
    """Indices of the items on each page."""
    pages: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        pages[item["page"]].append(index)
    return pages


def prose_edge(items: Items, indices: list[int]) -> float | None:
    """Median left edge of the page's non-numeric items."""
    lefts = [items[i]["l"] for i in indices if line_number_value(items[i]["text"]) is None]
    return median(lefts) if lefts else None


def right_aligned_columns(items: Items, indices: list[int]) -> list[list[int]]:
    """Group numeric items into columns sharing a right edge."""
    columns: list[list[int]] = []
    for index in sorted(indices, key=lambda i: items[i]["r"]):
        if columns and items[index]["r"] - items[columns[-1][0]]["r"] <= ALIGNMENT_SLACK:
            columns[-1].append(index)
        else:
            columns.append([index])
    return columns


def numeric_indices(items: Items, indices: list[int]) -> list[int]:
    """The page's bare-integer items."""
    return [i for i in indices if line_number_value(items[i]["text"]) is not None]


def ascends_down_the_page(items: Items, column: list[int]) -> bool:
    """Whether the column's values increase as it descends the page.

    A margin is numbered top to bottom. A numeric column in a table carries
    arbitrary values, and a column of identical page numbers does not ascend at
    all. This is the cheapest evidence that separates the two, and it needs no
    position at all -- which is what makes it safe to lean on when the position
    test is the thing being loosened.
    """
    ordered = sorted(column, key=lambda i: -items[i]["t"])
    values = [line_number_value(items[i]["text"]) for i in ordered]
    return all(a is not None and b is not None and b > a for a, b in pairwise(values))


# --- R0: what ships today ----------------------------------------------------


def current(items: Items, *, min_numbers: int = 5) -> set[int]:
    """A right-aligned column of >= 5 bare integers, entirely left of the prose."""
    found: set[int] = set()
    for page, indices in by_page(items).items():
        edge = prose_edge(items, indices)
        if edge is None:
            continue
        for column in right_aligned_columns(items, numeric_indices(items, indices)):
            if len(column) >= min_numbers and items[column[0]]["r"] <= edge:
                found.update(column)
    return found


# --- R1: require the sequence to ascend --------------------------------------


def ascending(items: Items, *, min_numbers: int = 5) -> set[int]:
    """`current`, plus the column must be numbered down the page.

    Pure precision: it can only remove candidates `current` would have taken.
    Worth measuring on its own so the cost of the constraint is known before
    anything is relaxed in exchange for it.
    """
    found: set[int] = set()
    for page, indices in by_page(items).items():
        edge = prose_edge(items, indices)
        if edge is None:
            continue
        for column in right_aligned_columns(items, numeric_indices(items, indices)):
            if (
                len(column) >= min_numbers
                and items[column[0]]["r"] <= edge
                and ascends_down_the_page(items, column)
            ):
                found.update(column)
    return found


# --- R2: let the document vouch for its own margin ---------------------------


def cross_page(items: Items, *, min_numbers: int = 5, min_pages: int = 3, confirmed_min: int = 2) -> set[int]:
    """Find the margin's x-position from the pages that show it clearly, then
    apply it to the pages that do not.

    A pleading-paper margin is printed at the same place on every page. Where
    Docling absorbs line numbers into the text beside them, a page can be left
    with too few bare integers to clear the count threshold -- and the current
    rule then keeps the survivors in the body, which is precisely the residue
    that lands inside a citation.

    So: take the right edges of every column the strict test accepts, keep
    those recurring on at least `min_pages` pages, and on every other page
    accept a much shorter ascending column standing at the same edge.
    """
    pages = by_page(items)
    edges: dict[int, list[float]] = defaultdict(list)
    strict: set[int] = set()

    for page, indices in pages.items():
        edge = prose_edge(items, indices)
        if edge is None:
            continue
        for column in right_aligned_columns(items, numeric_indices(items, indices)):
            if len(column) >= min_numbers and items[column[0]]["r"] <= edge:
                strict.update(column)
                edges[round(items[column[0]]["r"] / ALIGNMENT_SLACK)].append(page)

    confirmed = {key for key, seen in edges.items() if len({*seen}) >= min_pages}
    if not confirmed:
        return strict

    found = set(strict)
    for page, indices in pages.items():
        for column in right_aligned_columns(items, numeric_indices(items, indices)):
            key = round(items[column[0]]["r"] / ALIGNMENT_SLACK)
            if key in confirmed and len(column) >= confirmed_min and ascends_down_the_page(items, column):
                found.update(column)
    return found


# --- R3: side-agnostic placement ---------------------------------------------


def either_margin(items: Items, *, min_numbers: int = 5) -> set[int]:
    """`ascending`, but the column may stand clear of the prose on either side.

    The current test is `column.r <= median prose left`, which can only see a
    left margin. Nothing about a line-number column requires it to be on the
    left, and a rule that names a jurisdiction's paper stock in its geometry is
    not general. This asks only that the column sit outside the prose block.
    """
    found: set[int] = set()
    for page, indices in by_page(items).items():
        prose = [i for i in indices if line_number_value(items[i]["text"]) is None]
        if not prose:
            continue
        left_edge = median(items[i]["l"] for i in prose)
        right_edge = median(items[i]["r"] for i in prose)
        for column in right_aligned_columns(items, numeric_indices(items, indices)):
            outside = items[column[0]]["r"] <= left_edge or items[column[0]]["l"] >= right_edge
            if len(column) >= min_numbers and outside and ascends_down_the_page(items, column):
                found.update(column)
    return found


# --- R4: everything ----------------------------------------------------------


def combined(items: Items) -> set[int]:
    """Cross-page confirmation, ascending sequence, either side."""
    return cross_page(items) | either_margin(items)


RULES = {
    "current": current,
    "ascending": ascending,
    "cross_page": cross_page,
    "either_margin": either_margin,
    "combined": combined,
}
