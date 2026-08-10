"""Run Mellea-LRC validation over a corpus and serialize one artifact per document.

Produces the input ``export_mellea_lrc_artifact.py`` expects: a directory of
serialized ``ValidatedDocument`` JSON files, named after the benchmark document
so the adapter can recover its number.

The serialized run is the expensive artifact — it costs CourtListener and model
calls — so it is kept on disk and re-scored for free.

This script is Mellea-LRC-only. It is not part of the public contract.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evaluations.extraction.run import read_body
from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.extraction import run_extraction_from_text
from mellea_lrc.llm import start_mellea_session_from_env
from mellea_lrc.serialization.validated_document import serialize_validated_document
from mellea_lrc.validation import validate_document


def read_corpus(documents: Path) -> list[tuple[str, str]]:
    """Read every ``.txt`` in a directory as ``(stem, body)`` pairs."""
    paths = sorted(documents.glob("*.txt"))
    if not paths:
        raise ValueError(f"{documents}: no .txt documents found")
    return [(path.stem, read_body(path)) for path in paths]


async def validate_corpus(corpus: list[tuple[str, str]]) -> list[tuple[str, dict]]:
    """Validate an already-read corpus, returning one serialized run per document."""
    client = CourtListenerClient()
    session = start_mellea_session_from_env()
    runs: list[tuple[str, dict]] = []
    for stem, body in corpus:
        extracted = run_extraction_from_text(body, source_path=stem)
        validated = await validate_document(extracted, client=client, session=session)
        runs.append((stem, serialize_validated_document(validated)))
        print(f"  {stem[:40]:<40} {len(validated.citations):>4} citations")
    return runs


def write_runs(output: Path, runs: list[tuple[str, dict]]) -> int:
    """Write each serialized run under the document's own name."""
    output.mkdir(parents=True, exist_ok=True)
    for stem, payload in runs:
        (output / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sum(len(payload["citations"]) for _, payload in runs)


def main() -> None:
    """Run the command-line producer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents", type=Path, required=True, help="Directory of benchmark documents (documents_txt/)."
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory to write serialized runs into.")
    args = parser.parse_args()

    runs = asyncio.run(validate_corpus(read_corpus(args.documents)))
    citations = write_runs(args.output, runs)
    print(f"Validated {citations} citations across {len(runs)} documents into {args.output}")


if __name__ == "__main__":
    main()
