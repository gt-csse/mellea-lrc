"""Review deferred roots against grounded evidence already saved for another root.

Example:
    uv run python scripts/run_shared_body_checkpoint.py \
      --checkpoint data/run-artifacts/43-primary-current-body-resolution-v1/root_body_corroboration_resolution \
      --artifacts data/run-artifacts/44-primary-shared-body-review-v1

This stage makes no CourtListener or GovInfo request. It restores each complete
Document, runs the standalone shared-evidence review, and persists its result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from mellea_lrc.api import Document, review_shared_body_evidence
from mellea_lrc.llm import start_mellea_session_from_env
from mellea_lrc.serialization import artifact_directory, serialize

if TYPE_CHECKING:
    from mellea import MelleaSession


@serialize()
async def review_shared_body_document(document: Document, *, session: MelleaSession) -> Document:
    """Save one fully resumable Document after the independent root review."""
    return await review_shared_body_evidence(document, session=session)


def _checkpoint_paths(checkpoint: Path, documents: tuple[str, ...]) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(checkpoint.glob("*.json"))
        if not documents or any(path.stem == name or path.stem.startswith(f"{name}__") for name in documents)
    )


def _write_manifest(artifacts: Path, checkpoint: Path, count: int) -> None:
    manifest = {
        "source_checkpoint": str(checkpoint),
        "document_count": count,
        "provider_requests": 0,
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


async def run_checkpoint(checkpoint: Path, artifacts: Path, documents: tuple[str, ...]) -> None:
    load_dotenv()
    paths = _checkpoint_paths(checkpoint, documents)
    if not paths:
        raise ValueError(f"No serialized Documents matched in {checkpoint}")
    session = start_mellea_session_from_env()
    with artifact_directory(artifacts):
        for index, path in enumerate(paths, start=1):
            target = artifacts / "shared_body_evidence_review" / path.name
            if target.is_file():
                print(f"{index}/{len(paths)} {path.stem}: reused {target}", flush=True)
                continue
            document = Document.model_validate(json.loads(path.read_text(encoding="utf-8")))
            await review_shared_body_document(document, session=session)
            print(f"{index}/{len(paths)} {path.stem}: {target}", flush=True)
    _write_manifest(artifacts, checkpoint, len(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--document", action="append", default=[], dest="documents")
    args = parser.parse_args()
    asyncio.run(run_checkpoint(args.checkpoint, args.artifacts, tuple(args.documents)))


if __name__ == "__main__":
    main()
