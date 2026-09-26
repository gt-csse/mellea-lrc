"""Build resumable root and exact reporter-lookup prediction artifacts.

Run from the repository root::

    uv run python -m evaluations.run_reporter_root_lookup \
        --sets primary --run-dir local/reporter-root-lookup

The runner reads only manifest-listed source text and index masks. It saves
``root_documents`` immediately after ``grow_roots`` and ``documents`` after
``reporter_root_lookup``. A provider error leaves completed documents
and the current document's root checkpoint available for the same command to
resume. Use ``--through roots`` to create only the extraction checkpoint.
Annotations are read later by the independent scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evaluations.annotations import SETS
from mellea_lrc.api import Document, grow_roots, reporter_root_lookup
from mellea_lrc.extraction.roots import STAGE as ROOTS_STAGE
from mellea_lrc.model import Span
from mellea_lrc.validation.reporter_root_lookup import STAGE as LOOKUP_STAGE


def _save(path: Path, content: str) -> None:
    """Replace a complete artifact atomically; interrupted writes stay hidden."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _source_document(data_root: Path, name: str, filename: str, metadata: dict[str, Any]) -> Document:
    if Path(filename).name != filename or Path(filename).suffix != ".txt":
        raise ValueError(f"Invalid manifest filename: {filename}")
    text_dir = "documents_txt" if name == "primary" else "filings_txt"
    source_path = data_root / name / text_dir / filename
    document = Document.from_source(source_path)
    digest = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
    if digest != metadata["sha256"] or len(document.text) != metadata["length"]:
        raise ValueError(f"{source_path}: source differs from documents.json")
    return Document.model_validate(
        {
            **document.model_dump(mode="python"),
            "index_spans": tuple(
                Span(start=raw["start"], end=raw["end"]) for raw in metadata.get("index_spans", ())
            ),
        }
    )


def _checkpoint(path: Path, stage: str, source: Document) -> Document:
    document = Document.model_validate_json(path.read_text(encoding="utf-8"))
    if document.stage_runs[-1:] != (stage,):
        raise ValueError(f"{path}: expected final stage {stage}")
    if (
        document.text != source.text
        or document.index_spans != source.index_spans
        or document.source_metadata != source.source_metadata
    ):
        raise ValueError(f"{path}: saved checkpoint differs from its manifest source")
    return document


def _persist_document(path: Path, document: Document) -> None:
    payload = document.model_dump_json()
    if Document.model_validate_json(payload) != document:
        raise ValueError(f"{path}: document did not survive serialization")
    _save(path, payload)


def _run_spec(data_root: Path) -> str:
    return (
        json.dumps(
            {
                "source_data_root": str(data_root.resolve()),
                "root_rules": "stable",
                "hunt_dockets": False,
                "court_docket_fetch": True,
                "checkpoints": [ROOTS_STAGE, LOOKUP_STAGE],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


async def run_documents(
    data_root: Path, run_dir: Path, sets: Iterable[str], *, through: str = "lookup"
) -> dict[str, int]:
    """Persist complete Document checkpoints and resume by document."""
    names = tuple(dict.fromkeys(sets))
    if not names or any(name not in SETS for name in names):
        raise ValueError("Select one or more known annotated sets")
    if through not in {"roots", "lookup"}:
        raise ValueError(f"Unsupported checkpoint: {through}")
    specification = _run_spec(data_root)
    spec_path = run_dir / "run.json"
    if spec_path.exists():
        if spec_path.read_text(encoding="utf-8") != specification:
            raise ValueError(f"{spec_path}: run settings differ from this invocation")
    else:
        _save(spec_path, specification)

    counts = {"roots_created": 0, "roots_reused": 0, "lookup_created": 0, "lookup_reused": 0}
    for name in names:
        manifest_path = data_root / name / "documents.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["documents"]
        for filename, metadata in sorted(manifest.items()):
            source = _source_document(data_root, name, filename, metadata)
            root_path = run_dir / "root_documents" / name / f"{filename}.json"
            if root_path.exists():
                roots = _checkpoint(root_path, ROOTS_STAGE, source)
                counts["roots_reused"] += 1
            else:
                roots = await grow_roots(source, hunt_dockets=False)
                if roots.stage_runs[-1:] != (ROOTS_STAGE,):
                    raise ValueError(f"{name}/{filename}: root extraction did not finish")
                _persist_document(root_path, roots)
                counts["roots_created"] += 1
            if through == "roots":
                continue

            lookup_path = run_dir / "documents" / name / f"{filename}.json"
            if lookup_path.exists():
                exact = _checkpoint(lookup_path, LOOKUP_STAGE, source)
                if exact.get_stage(ROOTS_STAGE) != roots:
                    raise ValueError(f"{lookup_path}: saved lookup stage has a different root checkpoint")
                counts["lookup_reused"] += 1
                continue
            try:
                exact = reporter_root_lookup(roots)
            except Exception:
                print(
                    f"Reporter lookup stopped at {name}/{filename}; rerun this command to resume.",
                    file=sys.stderr,
                )
                raise
            _persist_document(lookup_path, exact)
            counts["lookup_created"] += 1
            print(f"Saved reporter lookup: {name}/{filename}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sets", nargs="+", choices=SETS, default=["primary"])
    parser.add_argument("--through", choices=("roots", "lookup"), default="lookup")
    args = parser.parse_args()

    import asyncio

    counts = asyncio.run(run_documents(args.data_root, args.run_dir, args.sets, through=args.through))
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
