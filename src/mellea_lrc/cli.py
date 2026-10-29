"""Command-line entrypoint for validating a legal document."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from mellea_lrc import execution


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
    resume = commands.add_parser("resume", help="Continue a saved Document with its adjacent .run directory")
    resume.add_argument("document", type=Path, help="Path to the saved Document JSON")
    args = parser.parse_args(argv)
    if args.command == "resume":
        if not args.document.is_file():
            parser.error(f"Document is not a file: {args.document}")
        asyncio.run(execution.resume(args.document))
        return 0
    if not args.source.is_file():
        parser.error(f"Source is not a file: {args.source}")
    if args.output.exists() and args.output.samefile(args.source):
        parser.error("Output must be different from the source document")
    if args.output.exists() and not args.output.is_file():
        parser.error(f"Output is not a file: {args.output}")
    asyncio.run(execution.validate(args.source, args.output))
    return 0
