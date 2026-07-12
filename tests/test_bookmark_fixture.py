"""Integrity checks for the maintained bookmark research fixture."""

# ruff: noqa: INP001

import hashlib
import json
from pathlib import Path


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures/bookmarked"
BOOKMARK_SCHEMA_VERSION = 2
MINIMUM_DATE_EXTRACTION_RECORDS = 30
SET_DIR = FIXTURE_DIR / "sets"


def test_bookmark_fixture_json_and_text_are_synchronized() -> None:
    """The committed aggregate text must be the exact projection of JSON."""
    store = json.loads((FIXTURE_DIR / "bookmarks.json").read_text(encoding="utf-8"))
    blocks = [bookmark["citation"]["context"] for bookmark in store["bookmarks"]]
    joined_blocks = "\n\n\n\n".join(blocks)
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


def test_bookmark_ids_follow_the_frontend_identity_contract() -> None:
    """Fixture identities must resolve through the frontend status endpoint."""
    store = json.loads((FIXTURE_DIR / "bookmarks.json").read_text(encoding="utf-8"))
    for bookmark in store["bookmarks"]:
        citation = bookmark["citation"]
        identity = {
            "matched_citation_text": " ".join(citation["matched_citation_text"].split()),
            "context": " ".join(citation["context"].split()),
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        expected = f"citation:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
        assert bookmark["bookmark_id"] == expected


def test_named_bookmark_store_text_is_its_json_projection() -> None:
    """Each named fixture is self-contained rather than a view over another store."""
    for json_path in SET_DIR.glob("bookmark-*.json"):
        store = json.loads(json_path.read_text(encoding="utf-8"))
        text_path = json_path.with_suffix(".txt")
        assert text_path.is_file()
        blocks = [bookmark["citation"]["context"] for bookmark in store["bookmarks"]]
        assert text_path.read_text(encoding="utf-8") == "\n\n\n\n".join(blocks) + "\n"


def test_date_extraction_collection_has_reviewed_minimum_corpus_size() -> None:
    """Keep the date corpus broad enough to expose non-local regressions."""
    store = json.loads((SET_DIR / "bookmark-date-extraction.json").read_text(encoding="utf-8"))
    assert len(store["bookmarks"]) >= MINIMUM_DATE_EXTRACTION_RECORDS
