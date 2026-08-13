"""Command-line entrypoint for the validation pipeline.

One command, running both layers end to end -- the citations are parsed out of
the source and then checked against CourtListener::

    mellea-lrc validate "See Brown v. Board of Education, 347 U.S. 483, 495 (1954)."
    mellea-lrc validate --from-file filing.pdf

The source is read as text unless ``--from-file`` says it names a document. The
serialized result is written as JSON, to ``--output`` when given and to stdout
otherwise. CourtListener and model credentials are read from the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mellea_lrc.extraction import ExtractedDocument, extract_from_plain_text, extract_from_raw_document
from mellea_lrc.serialization import serialize_validated_document
from mellea_lrc.validation import validate_document


def _parse(source: str, *, from_file: bool) -> ExtractedDocument:
    """Parse the citations out of a document on disk, or out of the text itself."""
    if from_file:
        return extract_from_raw_document(Path(source))
    return extract_from_plain_text(source)


def _validate(args: argparse.Namespace) -> int:
    """Parse the source, then check every citation it contains."""
    document = _parse(args.source, from_file=args.from_file)
    print(f"Parsed {len(document.full_citations)} citations; validating", file=sys.stderr)
    validated = asyncio.run(validate_document(document))

    text = json.dumps(serialize_validated_document(validated), indent=2, ensure_ascii=False)
    if args.output is None:
        sys.stdout.write(text + "\n")
    else:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the validation pipeline."""
    parser = argparse.ArgumentParser(prog="mellea-lrc", description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser(
        "validate",
        help="Parse the citations in a source, then check them against CourtListener.",
        description="Parse the citations in a source, then check them against CourtListener.",
    )
    validate.add_argument("source", help="The text to check, or a document path with --from-file.")
    origin = validate.add_mutually_exclusive_group()
    origin.add_argument(
        "--from-text",
        dest="from_file",
        action="store_false",
        default=False,
        help="Read the source as text itself. This is the default.",
    )
    origin.add_argument(
        "--from-file",
        dest="from_file",
        action="store_true",
        help="Read the source as a path to a document (PDF, DOCX, or .txt).",
    )
    validate.add_argument("-o", "--output", type=Path, help="Write JSON here instead of stdout.")
    validate.set_defaults(handler=_validate)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
