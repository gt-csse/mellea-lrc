"""Regenerate cumulative root-identity checkpoints from clean source text.

The dated portion of the retrospective inventory supplies per-document hard
cutoffs. Undated documents still run, but the manifest marks them explicitly;
their identity decisions must not be treated as retrospective evaluation.

Example:
    uv run python -m scripts.regenerate_corpus_artifacts \
      --dataset primary \
      --artifacts data/run-artifacts/54-schema21-primary-v1 \
      --source-provenance data/sources/courtlistener/filings.json

Add ``--roots-only`` to regenerate the root-formation layer without retrieval.
Use ``--resume`` with the same arguments to retain verified completed files
after an interrupted provider run.
Use ``--root-checkpoints`` to resume identity from a separately generated
root-formation corpus without repeating locator hunting.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.govinfo import GovInfoClient
from mellea_lrc.llm import start_mellea_session_from_env
from mellea_lrc.model.document import Document
from mellea_lrc.serialization.source_provenance import read_source_provenance
from scripts.run_root_identity_pipeline import run

DATASET_SOURCES = {
    "primary": Path("data/primary/documents_txt"),
    "hallucination-set-1": Path("data/hallucination-set-1/filings_txt"),
    "hallucination-set-2": Path("data/hallucination-set-2/filings_txt"),
    "reliable-high-profile": Path("data/reliable-high-profile/filings_txt"),
    "reliable-low-profile": Path("data/reliable-low-profile/filings_txt"),
}
FINAL_STAGE = "shared_body_evidence_review"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source_files(dataset: str) -> tuple[Path, ...]:
    directory = DATASET_SOURCES[dataset]
    sources = tuple(
        source for source in sorted(directory.glob("*.txt")) if "(before clean)" not in source.name
    )
    if not sources:
        raise ValueError(f"No clean source files found in {directory}")
    return sources


def _retrospective_dates(
    inventory_path: Path, dataset: str, sources: tuple[Path, ...]
) -> dict[str, date | None]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    rows = [row for row in inventory["documents"] if row["set"] == dataset]
    by_name = {Path(row["source_txt"]).name: row for row in rows}
    if len(by_name) != len(rows) or set(by_name) != {source.name for source in sources}:
        raise ValueError(f"Retrospective inventory does not match clean {dataset} sources")
    dates: dict[str, date | None] = {}
    for source in sources:
        row = by_name[source.name]
        if _sha256(source.read_bytes()) != row["source_sha256"]:
            raise ValueError(f"Retrospective inventory source hash differs: {source}")
        value = row["retrospective_date"]
        dates[source.name] = date.fromisoformat(value) if value else None
    return dates


def _completed_documents(directory: Path, stage: str) -> dict[str, Document]:
    result: dict[str, Document] = {}
    for path in sorted((directory / stage).glob("*.json")):
        document = Document.model_validate(json.loads(path.read_text(encoding="utf-8")))
        source = document.source_metadata.path
        if source is None:
            raise ValueError(f"Final checkpoint has no source path: {path}")
        name = Path(source).name
        if name in result:
            raise ValueError(f"Duplicate final checkpoints for {name}")
        result[name] = document
    return result


def _root_checkpoint_paths(directory: Path, sources: tuple[Path, ...]) -> dict[str, Path]:
    """Index native root checkpoints by the exact source basename."""
    paths: dict[str, Path] = {}
    for path in sorted((directory / "root_formation").glob("*.json")):
        document = Document.model_validate(json.loads(path.read_text(encoding="utf-8")))
        source = document.source_metadata.path
        if source is None or document.passes[-1:] != ("root_formation",):
            raise ValueError(f"Not a source-backed root checkpoint: {path}")
        name = Path(source).name
        if name in paths:
            raise ValueError(f"Duplicate root checkpoints for {name}")
        paths[name] = path
    if set(paths) != {source.name for source in sources}:
        raise ValueError("Root checkpoints do not cover the clean corpus exactly")
    return paths


def _write_manifest(directory: Path, payload: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "corpus-manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


async def _run(args: argparse.Namespace) -> None:
    sources = _source_files(args.dataset)
    dates = _retrospective_dates(args.date_inventory, args.dataset, sources)
    provenance_details = (
        read_source_provenance(args.source_provenance)[1] if args.source_provenance is not None else {}
    )
    if args.artifacts.exists() and any(args.artifacts.iterdir()) and not args.resume:
        raise ValueError(f"Artifact directory already contains files: {args.artifacts}; pass --resume")
    final_stage = "root_formation" if args.roots_only else FINAL_STAGE
    completed = _completed_documents(args.artifacts, final_stage) if args.resume else {}
    root_paths = (
        _root_checkpoint_paths(args.root_checkpoints, sources) if args.root_checkpoints is not None else {}
    )
    if set(completed) - {source.name for source in sources}:
        raise ValueError("Artifact directory contains checkpoints from another corpus")

    manifest: dict[str, object] = {
        "dataset": args.dataset,
        "schema_version": 21,
        "final_stage": final_stage,
        "source_directory": str(DATASET_SOURCES[args.dataset].resolve()),
        "retrospective_inventory": str(args.date_inventory.resolve()),
        "retrospective_inventory_sha256": _sha256(args.date_inventory.read_bytes()),
        "source_provenance": str(args.source_provenance.resolve()) if args.source_provenance else None,
        "source_provenance_sha256": provenance_details.get("source_provenance_sha256"),
        "root_checkpoints": str(args.root_checkpoints.resolve()) if args.root_checkpoints else None,
        "model": os.environ.get("MELLEA_LRC_LLM_MODEL"),
        "courtlistener_pool": os.environ.get("MELLEA_LRC_COURTLISTENER_POOL"),
        "documents": {
            source.name: {
                "source_sha256": _sha256(source.read_bytes()),
                "retrospective_date": dates[source.name].isoformat() if dates[source.name] else None,
                "retrospective_evaluable": dates[source.name] is not None,
                "root_checkpoint_sha256": None,
                "status": "pending",
            }
            for source in sources
        },
    }
    rows = manifest["documents"]
    assert isinstance(rows, dict)
    prior_manifest = args.artifacts / "corpus-manifest.json"
    if args.resume and prior_manifest.exists():
        previous = json.loads(prior_manifest.read_text(encoding="utf-8"))
        for key in (
            "dataset",
            "schema_version",
            "final_stage",
            "retrospective_inventory_sha256",
            "source_provenance",
            "source_provenance_sha256",
            "model",
            "courtlistener_pool",
        ):
            prior_value = previous.get(key, FINAL_STAGE if key == "final_stage" else None)
            if prior_value != manifest[key]:
                raise ValueError(f"Resume configuration changed: {key}")
        previous_roots = previous.get("root_checkpoints")
        if previous_roots is not None and previous_roots != manifest["root_checkpoints"]:
            raise ValueError("Resume configuration changed: root_checkpoints")
        for name in completed:
            old_row = previous["documents"].get(name)
            if old_row is None or any(
                old_row.get(key) != rows[name][key]
                for key in (
                    "source_sha256",
                    "retrospective_date",
                    "retrospective_evaluable",
                )
            ):
                raise ValueError(f"Resume source or date changed: {name}")
            rows[name]["root_checkpoint_sha256"] = old_row.get("root_checkpoint_sha256")
    client = CourtListenerClient()
    govinfo = GovInfoClient()
    session = start_mellea_session_from_env()
    for index, source in enumerate(sources, start=1):
        row = rows[source.name]
        assert isinstance(row, dict)
        if source.name in completed:
            saved = completed[source.name]
            if _sha256(saved.text.encode("utf-8")) != row["source_sha256"]:
                raise ValueError(f"Final checkpoint has different source text: {source.name}")
            row["status"] = "complete"
            print(f"{index}/{len(sources)} {source.name}: reused schema-21 checkpoint", flush=True)
            continue
        _write_manifest(args.artifacts, manifest)
        print(f"{index}/{len(sources)} {source.name}: running", flush=True)
        if root_paths:
            row["root_checkpoint_sha256"] = _sha256(root_paths[source.name].read_bytes())
            _write_manifest(args.artifacts, manifest)
        try:
            document = await run(
                source.resolve(),
                artifacts=args.artifacts,
                root_checkpoint=root_paths.get(source.name),
                client=client,
                govinfo_client=govinfo,
                session=session,
                stop_after_root_formation=args.roots_only,
                retrospective_date=dates[source.name],
                source_provenance=args.source_provenance,
            )
        except Exception:
            row["status"] = "interrupted"
            _write_manifest(args.artifacts, manifest)
            raise
        if Document.model_validate(document.model_dump(mode="json")) != document:
            raise ValueError(f"Final checkpoint does not round-trip: {source.name}")
        row["status"] = "complete"
        _write_manifest(args.artifacts, manifest)
        print(f"{index}/{len(sources)} {source.name}: {len(document.citations)} citations", flush=True)
    print(f"Completed {len(sources)} {args.dataset} documents", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SOURCES), required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--date-inventory", type=Path, default=Path("evaluations/retrospective_dates.json"))
    parser.add_argument("--source-provenance", type=Path)
    parser.add_argument("--root-checkpoints", type=Path, help="Root-formation corpus to reuse for identity.")
    parser.add_argument(
        "--roots-only", action="store_true", help="Save root formation without identity retrieval."
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_dotenv(".env")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
