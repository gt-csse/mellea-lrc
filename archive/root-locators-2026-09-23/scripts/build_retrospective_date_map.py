"""Build a strict, source-evidenced date map for a selected evaluation slice.

Examples:
    python scripts/build_retrospective_date_map.py --source data/primary/documents_txt/001__...txt --output dates.json
    python scripts/build_retrospective_date_map.py --sources-file selected.txt --output dates.json

An entire set can be requested with ``--set``. The command fails if any selected
filing lacks an exact document date; it never substitutes a filing/service date.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "evaluations" / "retrospective_dates.json"


def _record_key(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        candidate = candidate.relative_to(ROOT)
    return candidate.as_posix()


def _verify_record(record: dict[str, object]) -> None:
    path = ROOT / str(record["source_txt"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["source_sha256"]:
        raise ValueError(f"Source hash changed: {path}")
    text = raw.decode("utf-8")
    for evidence in record["evidence"]:
        span = evidence["span"]
        if text[span["start"] : span["end"]] != evidence["quote"]:
            raise ValueError(f"Date evidence no longer matches: {path}")


def build_map(inventory: dict[str, object], sources: list[str], sets: list[str]) -> dict[str, str]:
    records = inventory["documents"]
    by_path = {_record_key(str(record["source_txt"])): record for record in records}
    if len(by_path) != len(records):
        raise ValueError("Duplicate source path in retrospective date inventory")
    selected = set()
    for corpus in sets:
        matches = [path for path, record in by_path.items() if record["set"] == corpus]
        if not matches:
            raise ValueError(f"Unknown or empty set: {corpus}")
        selected.update(matches)
    for source in sources:
        path = _record_key(source)
        if path not in by_path:
            raise ValueError(f"Source is not in retrospective date inventory: {source}")
        selected.add(path)
    if not selected:
        raise ValueError("Select at least one --set, --source, or --sources-file entry")

    missing = [path for path in sorted(selected) if by_path[path]["retrospective_date"] is None]
    if missing:
        raise ValueError("No explicit document date for selected source(s):\n" + "\n".join(missing))

    output = {}
    for path in sorted(selected):
        record = by_path[path]
        _verify_record(record)
        value = str(record["retrospective_date"])
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError(f"Invalid ISO date for {path}: {value}")
        basename = Path(path).name
        if basename in output:
            raise ValueError(f"Selected sources share filename {basename}; use distinct filenames")
        output[basename] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", action="append", default=[], help="Select every filing in a corpus")
    parser.add_argument("--source", action="append", default=[], help="Select one source text path")
    parser.add_argument("--sources-file", type=Path, help="Text file with one source text path per line")
    parser.add_argument("--output", required=True, type=Path, help="Write basename-to-ISO-date JSON here")
    args = parser.parse_args()

    sources = list(args.source)
    if args.sources_file is not None:
        sources.extend(line.strip() for line in args.sources_file.read_text().splitlines() if line.strip())
    try:
        result = build_map(json.loads(INVENTORY.read_text()), sources, args.set)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(result)} retrospective dates to {args.output}")


if __name__ == "__main__":
    main()
