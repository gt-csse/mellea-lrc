"""Integrity checks for the maintained bookmark research fixture."""

# ruff: noqa: INP001

import json
from pathlib import Path


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures/bookmarked"
BOOKMARK_SCHEMA_VERSION = 2


def test_bookmark_fixture_json_and_text_are_synchronized() -> None:
    """The committed aggregate text must be the exact projection of JSON."""
    store = json.loads((FIXTURE_DIR / "bookmarks.json").read_text(encoding="utf-8"))
    blocks = []
    for bookmark in store["bookmarks"]:
        lines = [bookmark["citation"]["matched_text"]]
        if bookmark["comment"]:
            lines.extend(("", f"> {bookmark['comment']}"))
        lines.extend(("", f"Context: {bookmark['citation']['context']}"))
        blocks.append("========\n" + "\n".join(lines))
    joined_blocks = "\n\n".join(blocks)
    expected = f"{joined_blocks}\n" if blocks else ""

    assert store["schema_version"] == BOOKMARK_SCHEMA_VERSION
    assert (FIXTURE_DIR / "bookmarked.txt").read_text(encoding="utf-8") == expected


def test_bookmark_fixture_identities_are_unique() -> None:
    """Bookmark and provenance identities must remain unique in the corpus."""
    store = json.loads((FIXTURE_DIR / "bookmarks.json").read_text(encoding="utf-8"))
    bookmark_ids = [bookmark["bookmark_id"] for bookmark in store["bookmarks"]]
    provenance_ids = [
        provenance["provenance_id"]
        for bookmark in store["bookmarks"]
        for provenance in bookmark["provenances"]
    ]

    assert len(bookmark_ids) == len(set(bookmark_ids))
    assert len(provenance_ids) == len(set(provenance_ids))
