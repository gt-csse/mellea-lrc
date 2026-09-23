"""Run the explicit leaf-growth stage over serialized Documents.

Leaf growth is deliberately a separate, cheap checkpoint after root identity.
It reads the roots that the incoming document currently holds, preserves every
identity trace and withdrawal, and writes only the structural leaf graph and
unattributed-leaf findings.  It makes no CourtListener or model request.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from mellea_lrc.api import Document, grow_leaves
from mellea_lrc.model.citations import is_leaf


def _atomic_json(path: Path, value: object) -> None:
    """Write one complete checkpoint or leave the preceding file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    """Return the exact source-checkpoint hash recorded in the manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run(
    *,
    documents: Path,
    output: Path,
    start: int,
    limit: int,
    resume: bool,
) -> dict[str, object]:
    """Grow leaves for one deterministic slice of serialized Documents."""
    paths = sorted(documents.glob("*.json"))[start : start + limit]  # noqa: ASYNC240
    if len(paths) != limit:
        msg = f"{documents} has {len(paths)} artifacts in slice start={start}, limit={limit}"
        raise ValueError(msg)

    counts: Counter[str] = Counter()
    results: list[dict[str, str]] = []
    for index, path in enumerate(paths, start=1):
        result_path = output / "documents" / path.name
        if resume and result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            document = Document.model_validate(payload)
        else:
            document = Document.model_validate(json.loads(path.read_text(encoding="utf-8")))
            document = await grow_leaves(document)
            payload = document.model_dump(mode="json")
            Document.model_validate(payload)
            _atomic_json(result_path, payload)
        counts["roots"] += sum(record.is_root for record in document.active_citations)
        counts["leaves"] += sum(is_leaf(record.fields) for record in document.active_citations)
        counts["findings"] += len(document.findings)
        results.append(
            {
                "input_document": path.name,
                "input_document_sha256": _sha256(path),
                "result": str(result_path.relative_to(output)),
            }
        )
        print(f"{index}/{len(paths)} {path.stem}: {len(document.citations)} citations")

    return {
        "artifact_type": "leaf_growth",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "input_documents": str(documents),
        "start_index": start,
        "document_count": len(results),
        "documents": results,
        "counts": dict(sorted(counts.items())),
    }


def main() -> None:
    """Run leaf growth and write its resumable artifact manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True, help="Input serialized Document directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output stage-artifact directory.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.limit < 1:
        parser.error("--start must be non-negative and --limit must be positive")
    manifest = asyncio.run(
        run(
            documents=args.documents,
            output=args.output,
            start=args.start,
            limit=args.limit,
            resume=args.resume,
        )
    )
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
