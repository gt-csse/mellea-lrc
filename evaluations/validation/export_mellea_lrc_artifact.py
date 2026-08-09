"""Flatten serialized Mellea-LRC runs into the generic identity-evaluation JSONL format."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _normalized_part(value: str) -> str:
    return _NON_ALPHANUMERIC.sub("-", value.casefold()).strip("-")


def _locator_id(citation: dict[str, Any]) -> str | None:
    """Return the stable locator identity when a full reporter locator is available."""
    values = (
        citation["citation_type"],
        citation.get("volume"),
        citation.get("reporter"),
        citation.get("page"),
    )
    if not all(values[1:]):
        return None
    return "-".join(_normalized_part(str(value)) for value in values)


def _document_id(path: Path) -> str:
    """Return the benchmark's document number for one serialized run file.

    Runs are produced over the published corpus, whose files are named
    ``006__coomer-v-lindell...``. The benchmark keys occurrences on the leading
    number, so take that prefix; a bare ``6.json`` is accepted unchanged.
    """
    return path.stem.split("__", 1)[0]


def _record_id(document_id: str, source: dict[str, Any]) -> str:
    """Build the same occurrence ID used by the identity benchmark."""
    locator_id = _locator_id(source["citation"])
    identity = locator_id or _normalized_part(source["citation"]["citation_type"])
    span = source["locator_span"]
    return f"cite:{document_id}:{identity}:{span['start']}-{span['end']}"


def export(artifact_dir: Path) -> list[dict[str, Any]]:
    """Convert one directory of per-document serialized runs into generic rows."""
    records: list[dict[str, Any]] = []
    paths = list(artifact_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"{artifact_dir}: no serialized JSON files found")
    if invalid := [path.name for path in paths if not _document_id(path).isdigit()]:
        raise ValueError(
            f"{artifact_dir}: serialized file names must start with the benchmark's "
            f"document number, not {', '.join(sorted(invalid))}"
        )
    for path in sorted(paths, key=lambda item: int(_document_id(item))):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            sources = {source["citation_id"]: source for source in document["source"]["citations"]}
            for citation in document["citations"]:
                source = sources[citation["citation_id"]]
                records.append(
                    {
                        "id": _record_id(_document_id(path), source),
                        "locator_id": _locator_id(source["citation"]),
                        "locator_span": source["locator_span"],
                        "verdict": (citation.get("aggregation") or {}).get("overall_outcome")
                        or "unavailable",
                    }
                )
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"{path}: not a serialized Mellea-LRC validation artifact") from error
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("serialized artifacts produce duplicate identity-benchmark IDs")
    return records


def main() -> None:
    """Write the generic run artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir", type=Path, required=True, help="Directory of numbered serialized run JSON files."
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL run artifact.")
    args = parser.parse_args()

    records = export(args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    print(f"Wrote {len(records)} identity-evaluation records to {args.output}")


if __name__ == "__main__":
    main()
