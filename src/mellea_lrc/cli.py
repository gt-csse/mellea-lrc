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

from mellea_lrc.api import (
    Document,
    find_docket_locators,
    find_full_reporter_locators,
    form_roots,
    lookup_full_reporter_locators_exact,
    resolve_case_names,
    resolve_colocations,
    resolve_courts,
    resolve_dates,
    resolve_full_reporter_locator_ambiguities,
    stable,
    validate_unique_full_reporter_locator_identities,
)


def _parse(source: str, *, from_file: bool) -> Document:
    """Run the explicit locator-to-root chain used before identity validation."""
    document = Document.from_source(Path(source) if from_file else source)
    rules = stable()
    document = find_full_reporter_locators(document, rules=rules)
    document = find_docket_locators(document, rules=rules)
    document = resolve_colocations(document, rules=rules)
    document = resolve_case_names(document, rules=rules)
    document = resolve_courts(document, rules=rules)
    document = resolve_dates(document, rules=rules)
    return form_roots(document)


def _validate(args: argparse.Namespace) -> int:
    """Parse the source, then check every citation it contains."""
    document = _parse(args.source, from_file=args.from_file)
    print(
        f"Formed {sum(citation.is_root for citation in document.active_citations)} roots; validating",
        file=sys.stderr,
    )
    document = asyncio.run(lookup_full_reporter_locators_exact(document))
    document = asyncio.run(validate_unique_full_reporter_locator_identities(document))
    document = asyncio.run(resolve_full_reporter_locator_ambiguities(document))

    text = json.dumps(document.serialize(), indent=2, ensure_ascii=False)
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
