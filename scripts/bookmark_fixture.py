"""Maintain bookmark comments without hand-editing the authoritative JSON."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).parents[1]
FIXTURE_DIR = ROOT / "fixtures" / "bookmarked"


def main() -> None:
    """Replace one comment or add one bookmark, then regenerate the text projection."""
    if len(sys.argv) > 1 and sys.argv[1] not in {"comment", "add", "-h", "--help"}:
        _main_legacy_comment()
        return

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    comment_parser = subparsers.add_parser("comment", help="Replace an existing comment")
    comment_parser.add_argument("id", help="Full bookmark ID or an unambiguous prefix")
    comment_parser.add_argument("comment", help="Replacement comment; use an empty string to clear it")

    add_parser = subparsers.add_parser("add", help="Add a bookmark citation context")
    add_parser.add_argument("--matched-citation-text", required=True, help="Matched full citation text")
    add_parser.add_argument("--context", required=True, help="Local citation context")
    add_parser.add_argument("--comment", required=True, help="Initial bookmark comment")
    add_parser.add_argument("--source-path", required=True, help="Path relative to fixtures/bookmarked")
    add_parser.add_argument("--citation-span-start", required=True, type=int, help="Source citation span start")
    add_parser.add_argument("--citation-span-end", required=True, type=int, help="Source citation span end")
    add_parser.add_argument("--seen-at", default=None, help="UTC ISO timestamp; defaults to now")
    args = parser.parse_args()

    json_path = FIXTURE_DIR / "bookmarks.json"
    store = json.loads(json_path.read_text(encoding="utf-8"))

    if args.command is None:
        parser.error("expected a command: comment or add")
    if args.command == "comment":
        _replace_comment(store, args.id, args.comment)
    elif args.command == "add":
        _add_bookmark(
            store,
            matched_citation_text=args.matched_citation_text,
            context=args.context,
            comment=args.comment,
            source_path=args.source_path,
            citation_span_start=args.citation_span_start,
            citation_span_end=args.citation_span_end,
            seen_at=args.seen_at,
        )
    else:
        raise AssertionError(args.command)

    json_path.write_text(f"{json.dumps(store, indent=2)}\n", encoding="utf-8")
    (FIXTURE_DIR / "bookmarked.txt").write_text(render_text(store), encoding="utf-8")


def _main_legacy_comment() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("id", help="Full bookmark ID or an unambiguous prefix")
    parser.add_argument("comment", help="Replacement comment; use an empty string to clear it")
    args = parser.parse_args()

    json_path = FIXTURE_DIR / "bookmarks.json"
    store = json.loads(json_path.read_text(encoding="utf-8"))
    _replace_comment(store, args.id, args.comment)
    json_path.write_text(f"{json.dumps(store, indent=2)}\n", encoding="utf-8")
    (FIXTURE_DIR / "bookmarked.txt").write_text(render_text(store), encoding="utf-8")


def _replace_comment(store: dict[str, Any], bookmark_id: str, comment: str) -> None:
    matches = [item for item in store["bookmarks"] if item["bookmark_id"].startswith(bookmark_id)]
    if len(matches) != 1:
        message = f"Expected one bookmark for {bookmark_id!r}; found {len(matches)}"
        raise SystemExit(message)

    matches[0]["comment"] = comment.strip() or None
    # Deliberately leave updated_at alone: this command regularizes research notes
    # in a committed fixture and should produce reproducible diffs.


def _add_bookmark(
    store: dict[str, Any],
    *,
    matched_citation_text: str,
    context: str,
    comment: str,
    source_path: str,
    citation_span_start: int,
    citation_span_end: int,
    seen_at: str | None,
) -> None:
    timestamp = seen_at or _utc_now()
    bookmark_id = _identity("citation", matched_citation_text, context)
    provenance_id = _identity(
        "provenance",
        source_path,
        str(citation_span_start),
        str(citation_span_end),
        context,
    )
    if any(item["bookmark_id"] == bookmark_id for item in store["bookmarks"]):
        raise SystemExit(f"Bookmark already exists: {bookmark_id}")
    store["bookmarks"].append(
        {
            "bookmark_id": bookmark_id,
            "citation": {
                "matched_citation_text": matched_citation_text,
                "context": context,
            },
            "comment": comment.strip() or None,
            "provenances": [
                {
                    "source_path": source_path,
                    "source_format": "text",
                    "citation_span": {
                        "start": citation_span_start,
                        "end": citation_span_end,
                    },
                    "provenance_id": provenance_id,
                    "seen_at": timestamp,
                }
            ],
            "created_at": timestamp,
            "updated_at": timestamp,
        }
    )


def _identity(prefix: str, *parts: str) -> str:
    canonical = "\n".join(_normalize(part) for part in parts)
    return f"{prefix}:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def render_text(store: dict[str, object]) -> str:
    """Render the exact aggregate fixture projection."""
    blocks: list[str] = []
    for bookmark in store["bookmarks"]:  # type: ignore[index]
        lines = [bookmark["citation"]["matched_citation_text"]]  # type: ignore[index]
        if bookmark["comment"]:  # type: ignore[index]
            lines.extend(("", f"> {bookmark['comment']}"))  # type: ignore[index]
        lines.extend(("", f"Context: {bookmark['citation']['context']}"))  # type: ignore[index]
        blocks.append("========\n" + "\n".join(lines))
    joined = "\n\n".join(blocks)
    return f"{joined}\n" if blocks else ""


if __name__ == "__main__":
    main()
