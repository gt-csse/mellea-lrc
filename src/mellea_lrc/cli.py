"""Command-line entrypoint for validating a legal document."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from mellea_lrc.api import (
    Document,
    FieldExtractionConfig,
    grow_leaves,
    grow_roots,
    validate_all_citation_content,
    validate_roots_citation_identity,
)


async def _validate(source: Path, output: Path) -> None:
    fields = FieldExtractionConfig.from_env()
    fields.validate_runtime()
    output.parent.mkdir(parents=True, exist_ok=True)

    def save(document: Document) -> None:
        # Replace only a complete JSON file, preserving the last checkpoint if
        # serialization or writing fails. The temporary file shares its filesystem.
        with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as directory:
            temporary = Path(directory) / output.name
            temporary.write_text(document.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(output)

    document = Document.from_source(source)
    save(document)
    print("Running grow_roots...", flush=True)
    document = await grow_roots(
        document,
        hunt_dockets=True,
        review_docket_roots=True,
        field_extraction=fields,
        checkpoint=save,
    )
    print("Running validate_roots_citation_identity...", flush=True)
    document = await validate_roots_citation_identity(document, checkpoint=save)
    print("Running grow_leaves...", flush=True)
    document = await grow_leaves(document, field_extraction=fields, checkpoint=save)
    print("Running validate_all_citation_content...", flush=True)
    document = await validate_all_citation_content(document, checkpoint=save)
    save(document)
    print(f"Saved {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mellea-lrc", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate a PDF or text document")
    validate.add_argument("source", type=Path, help="Path to the source document")
    validate.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("document.json"),
        help="Native Document JSON (default: document.json)",
    )
    args = parser.parse_args(argv)
    if not args.source.is_file():
        parser.error(f"Source is not a file: {args.source}")
    if args.output.exists() and args.output.samefile(args.source):
        parser.error("Output must be different from the source document")
    if args.output.exists() and not args.output.is_file():
        parser.error(f"Output is not a file: {args.output}")
    asyncio.run(_validate(args.source, args.output))
    return 0
