"""Maintain bookmark comments without hand-editing the authoritative JSON."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
FIXTURE_DIR = ROOT / "fixtures" / "bookmarked"
SET_DIR = FIXTURE_DIR / "sets"


def main() -> None:
    """Replace one comment or add one bookmark, then regenerate the text projection."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    comment_parser = subparsers.add_parser("comment", help="Replace an existing comment")
    comment_parser.add_argument("id", help="Full bookmark ID or an unambiguous prefix")
    comment_parser.add_argument("comment", help="Replacement comment; use an empty string to clear it")
    comment_parser.add_argument("--set", dest="set_name", help="Named bookmark set; omit for the default store")

    add_parser = subparsers.add_parser("add", help="Add a bookmark citation context")
    add_parser.add_argument("--matched-citation-text", required=True, help="Matched full citation text")
    add_parser.add_argument("--context", required=True, help="Local citation context")
    add_parser.add_argument("--comment", required=True, help="Initial bookmark comment")
    add_parser.add_argument("--source-path", required=True, help="Path relative to fixtures/bookmarked")
    add_parser.add_argument("--citation-span-start", required=True, type=int, help="Source citation span start")
    add_parser.add_argument("--citation-span-end", required=True, type=int, help="Source citation span end")
    add_parser.add_argument("--seen-at", default=None, help="UTC ISO timestamp; defaults to now")
    add_parser.add_argument("--set", dest="set_name", help="Named bookmark set; omit for the default store")
    readme_parser = subparsers.add_parser(
        "render-set-readme",
        help="Regenerate the per-bookmark expected-results README for one named set",
    )
    readme_parser.add_argument("name", help="Named bookmark set")
    args = parser.parse_args()

    selected_set_name = getattr(args, "set_name", None)
    if args.command == "render-set-readme":
        selected_set_name = args.name
    json_path, text_path = _store_paths(selected_set_name)
    if json_path.exists():
        store = json.loads(json_path.read_text(encoding="utf-8"))
    elif args.command == "add" and getattr(args, "set_name", None) is not None:
        store = {"schema_version": 2, "bookmarks": []}
    else:
        message = f"Bookmark store does not exist: {json_path}"
        raise SystemExit(message)

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
    elif args.command == "render-set-readme":
        _render_named_store_readme(args.name, store)
    else:
        raise AssertionError(args.command)

    json_path.write_text(f"{json.dumps(store, indent=2)}\n", encoding="utf-8")
    text_path.write_text(render_text(store), encoding="utf-8")
    set_name = selected_set_name
    if set_name is not None:
        _render_named_store_readme(set_name, store)


def _store_paths(set_name: str | None) -> tuple[Path, Path]:
    """Return the JSON/text pair for one bookmark store."""
    if set_name is None:
        return FIXTURE_DIR / "bookmarks.json", FIXTURE_DIR / "bookmarked.txt"
    if not set_name.isascii() or not set_name.replace("-", "").isalnum():
        message = f"Unsafe bookmark set name: {set_name}"
        raise SystemExit(message)
    SET_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"bookmark-{set_name}"
    return SET_DIR / f"{stem}.json", SET_DIR / f"{stem}.txt"


def _render_named_store_readme(set_name: str, store: dict[str, Any]) -> None:
    """Document each bookmark's expected observable result beside its store."""
    lines = [
        f"# bookmark-{set_name} expected results",
        "",
        "Each row is an e2e expectation for one bookmarked citation. It does not ",
        "state authority identity or proposition support.",
        "",
        "| Bookmark | Citation | Expected result |",
        "| --- | --- | --- |",
    ]
    for bookmark in store["bookmarks"]:
        citation = " ".join(bookmark["citation"]["matched_citation_text"].split())
        comment = bookmark["comment"] or "No expectation recorded."
        expected = comment.split("Expected date recovery:", maxsplit=1)[-1].strip()
        lines.append(f"| `{bookmark['bookmark_id'][:17]}` | {citation} | {expected} |")
    (SET_DIR / f"bookmark-{set_name}.README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


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
    bookmark_id = _bookmark_identity(matched_citation_text, context)
    provenance_id = _provenance_identity(
        source_path=source_path,
        source_format="text",
        citation_span_start=citation_span_start,
        citation_span_end=citation_span_end,
    )
    if any(item["bookmark_id"] == bookmark_id for item in store["bookmarks"]):
        message = f"Bookmark already exists: {bookmark_id}"
        raise SystemExit(message)
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


def _bookmark_identity(matched_citation_text: str, context: str) -> str:
    canonical = _stable_json(
        {
            "matched_citation_text": _normalize(matched_citation_text),
            "context": _normalize(context),
        }
    )
    return f"citation:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _provenance_identity(
    *,
    source_path: str,
    source_format: str,
    citation_span_start: int,
    citation_span_end: int,
) -> str:
    canonical = _stable_json(
        {
            "source_path": source_path,
            "source_format": source_format,
            "citation_span": {
                "start": citation_span_start,
                "end": citation_span_end,
            },
        }
    )
    return f"provenance:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _stable_json(value: object) -> str:
    """Match the frontend's sorted, compact JSON identity encoding."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def render_text(store: dict[str, object]) -> str:
    """Render clean extraction input from bookmarked source contexts.

    Research comments and record delimiters belong only in ``bookmarks.json``.
    Including them here can create citations that were never bookmarked or make
    a delimiter look like part of a case name to the extraction model.
    """
    blocks = [
        bookmark["citation"]["context"]  # type: ignore[index]
        for bookmark in store["bookmarks"]  # type: ignore[index]
    ]
    joined = "\n\n\n\n".join(blocks)
    return f"{joined}\n" if blocks else ""


if __name__ == "__main__":
    main()
