"""Resume the reserved reporter-date checkpoint from serialized Documents.

Example:
    uv run python scripts/run_exact_reporter_date_checkpoint.py \
      --checkpoint data/run-artifacts/44-primary-shared-body-review-v1/shared_body_evidence_review \
      --artifacts data/run-artifacts/47-primary-reporter-dates-v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from mellea_lrc.api import Document, review_full_reporter_exact_dates
from mellea_lrc.courtlistener import CourtListenerClient, CourtListenerConfig
from mellea_lrc.serialization import artifact_directory, serialize


@serialize()
async def review_saved_reporter_dates(document: Document, *, client: CourtListenerClient) -> Document:
    """Persist the date boundary without changing a prior identity judgment."""
    return await review_full_reporter_exact_dates(document, client=client)


def _checkpoint_paths(checkpoint: Path, documents: tuple[str, ...]) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(checkpoint.glob("*.json"))
        if not documents
        or any(path.stem == selector or path.stem.startswith(f"{selector}__") for selector in documents)
    )


def _write_manifest(artifacts: Path, checkpoint: Path, count: int, pool: str) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "run-manifest.json").write_text(
        json.dumps(
            {"source_checkpoint": str(checkpoint), "document_count": count, "courtlistener_pool": pool},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


async def run_checkpoint(
    checkpoint: Path,
    artifacts: Path,
    *,
    documents: tuple[str, ...] = (),
    courtlistener_pool: str = "main",
) -> None:
    """Review selected saved Documents without replaying earlier stages."""
    load_dotenv()
    paths = _checkpoint_paths(checkpoint, documents)
    if not paths:
        raise ValueError(f"No serialized Documents matched in {checkpoint}")
    config = replace(
        CourtListenerConfig.from_env(),
        token=None,
        pool=courtlistener_pool,
        retry_short_rate_limits=True,
    )
    client = CourtListenerClient(config=config)
    with artifact_directory(artifacts):
        for index, path in enumerate(paths, start=1):
            target = artifacts / "full_reporter_exact_date_review" / path.name
            if target.is_file():
                print(f"{index}/{len(paths)} {path.stem}: reused {target}", flush=True)
                continue
            document = Document.model_validate(json.loads(path.read_text(encoding="utf-8")))
            await review_saved_reporter_dates(document, client=client)
            print(f"{index}/{len(paths)} {path.stem}: {target}", flush=True)
    _write_manifest(artifacts, checkpoint, len(paths), courtlistener_pool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--document", action="append", default=[], dest="documents")
    parser.add_argument("--courtlistener-pool", choices=("main", "reserved"), default="main")
    args = parser.parse_args()
    asyncio.run(
        run_checkpoint(
            args.checkpoint,
            args.artifacts,
            documents=tuple(args.documents),
            courtlistener_pool=args.courtlistener_pool,
        )
    )


if __name__ == "__main__":
    main()
