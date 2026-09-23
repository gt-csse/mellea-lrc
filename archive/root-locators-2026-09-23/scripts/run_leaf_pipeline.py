"""Resume a root Document and persist each chosen leaf-stage checkpoint.

Example:
    uv run python scripts/run_leaf_pipeline.py \\
      --checkpoint run-artifacts/root_formation/filing.json \\
      --artifacts run-artifacts/leaves \\
      --hunt-leaf-names --validate-pin-cite-sites

Root identity is deliberately outside this runner. It accepts any serialized
Document after root formation, including one whose roots were not validated.
The two model-assisted reviews are opt-in and make no CourtListener calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from mellea_lrc.api import (
    Document,
    grow_leaves,
    hunt_leaf_case_names,
    validate_pin_cite_sites,
)
from mellea_lrc.serialization import artifact_directory, serialize


@serialize()
async def grow_leaves_document(document: Document) -> Document:
    """Attach deterministic leaves to the available roots."""
    return await grow_leaves(document)


@serialize()
async def hunt_leaf_case_names_document(document: Document) -> Document:
    """Optionally admit grounded bare-name leaves or repair adjacent names."""
    return await hunt_leaf_case_names(document)


@serialize()
async def validate_pin_cite_sites_document(document: Document) -> Document:
    """Optionally review uncertain page claims on grown roots and leaves."""
    return await validate_pin_cite_sites(document)


async def run(
    checkpoint: Path,
    *,
    artifacts: Path,
    hunt_leaf_names: bool = False,
    validate_pin_sites: bool = False,
) -> Document:
    """Resume from one root checkpoint without retrieval or implicit reviews."""
    payload = json.loads(await asyncio.to_thread(checkpoint.read_text, encoding="utf-8"))
    document = Document.model_validate(payload)
    with artifact_directory(artifacts):
        document = await grow_leaves_document(document)
        if hunt_leaf_names:
            document = await hunt_leaf_case_names_document(document)
        if validate_pin_sites:
            document = await validate_pin_cite_sites_document(document)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="Serialized root Document JSON.")
    parser.add_argument("--artifacts", type=Path, required=True, help="Directory for stage checkpoints.")
    parser.add_argument("--hunt-leaf-names", action="store_true", help="Review unread bare case names.")
    parser.add_argument(
        "--validate-pin-cite-sites", action="store_true", help="Review uncertain pin-cite sites."
    )
    args = parser.parse_args(argv)
    load_dotenv()
    asyncio.run(
        run(
            args.checkpoint,
            artifacts=args.artifacts,
            hunt_leaf_names=args.hunt_leaf_names,
            validate_pin_sites=args.validate_pin_cite_sites,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
