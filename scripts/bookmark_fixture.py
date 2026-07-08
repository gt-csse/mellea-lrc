"""Maintain bookmark comments without hand-editing the authoritative JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
FIXTURE_DIR = ROOT / "fixtures" / "bookmarked"


def main() -> None:
    """Replace one comment and regenerate the text projection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("id", help="Full bookmark ID or an unambiguous prefix")
    parser.add_argument("comment", help="Replacement comment; use an empty string to clear it")
    args = parser.parse_args()

    json_path = FIXTURE_DIR / "bookmarks.json"
    store = json.loads(json_path.read_text(encoding="utf-8"))
    matches = [item for item in store["bookmarks"] if item["bookmark_id"].startswith(args.id)]
    if len(matches) != 1:
        message = f"Expected one bookmark for {args.id!r}; found {len(matches)}"
        raise SystemExit(message)

    matches[0]["comment"] = args.comment.strip() or None
    # Deliberately leave updated_at alone: this command regularizes research notes
    # in a committed fixture and should produce reproducible diffs.
    json_path.write_text(f"{json.dumps(store, indent=2)}\n", encoding="utf-8")
    (FIXTURE_DIR / "bookmarked.txt").write_text(render_text(store), encoding="utf-8")


def render_text(store: dict[str, object]) -> str:
    """Render the exact aggregate fixture projection."""
    blocks: list[str] = []
    for bookmark in store["bookmarks"]:  # type: ignore[index]
        lines = [bookmark["citation"]["matched_text"]]  # type: ignore[index]
        if bookmark["comment"]:  # type: ignore[index]
            lines.extend(("", f"> {bookmark['comment']}"))  # type: ignore[index]
        lines.extend(("", f"Context: {bookmark['citation']['context']}"))  # type: ignore[index]
        blocks.append("========\n" + "\n".join(lines))
    joined = "\n\n".join(blocks)
    return f"{joined}\n" if blocks else ""


if __name__ == "__main__":
    main()
