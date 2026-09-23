"""Run one extraction arm over a corpus and write the public run artifact.

This writes the evaluator's input format directly: a run produces public
``Occurrence`` rows, and ``evaluate.py`` scores them. There is no serialization
step and no adapter.

To score a system that is not one of these arms, either register your own in
``ARMS`` or skip this script and write the JSONL yourself — the format is two
fields, and nothing here is privileged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluations.extraction.arms import ARMS
from evaluations.extraction.occurrences import Occurrence, write_run_artifact


def read_body(path: Path) -> str:
    """Return the document text. A file is its text, so an offset is an offset."""
    return path.read_text(encoding="utf-8")


def read_corpus(documents: Path) -> list[tuple[str, str]]:
    """Read every ``.txt`` in a directory as ``(filename, body)`` pairs."""
    paths = sorted(documents.glob("*.txt"))
    if not paths:
        raise ValueError(f"{documents}: no .txt documents found")
    return [(path.name, read_body(path)) for path in paths]


def run_corpus(arm: str, corpus: list[tuple[str, str]]) -> list[Occurrence]:
    """Run one arm over an already-read corpus."""
    spec = ARMS[arm]
    found: list[Occurrence] = []
    for name, body in corpus:
        occurrences = spec.run(name, body)
        found.extend(occurrences)
        print(f"  {name[:40]:<40} {len(occurrences):>4} occurrences")
    return found


def main() -> None:
    """Run the command-line producer."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="arms:\n" + "\n".join(f"  {name:<26} {spec.components}" for name, spec in ARMS.items()),
    )
    parser.add_argument("--arm", required=True, choices=list(ARMS), help="Which arm to run.")
    parser.add_argument(
        "--documents", type=Path, required=True, help="Directory of benchmark documents (documents_txt/)."
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL run artifact.")
    args = parser.parse_args()

    spec = ARMS[args.arm]
    print(f"{args.arm}: {spec.components}")
    if spec.needs_model:
        print("  this arm calls a model; set MELLEA_LRC_LLM_* in the environment")
    occurrences = run_corpus(args.arm, read_corpus(args.documents))
    count = write_run_artifact(args.output, occurrences)
    print(f"Wrote {count} occurrences to {args.output}")


if __name__ == "__main__":
    main()
