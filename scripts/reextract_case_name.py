"""Run the standalone case-name re-extraction workflow.

Example:
    uv run --group llm python scripts/reextract_case_name.py \
      --context-file local/bookmarked/bookmarked.txt \
      --extracted-case-name "<NO_EXTRACTED_CASE_NAME>" \
      --courtlistener-case-name "Brown v. Board" \
      --attempts 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from mellea_lrc.assessment.reextraction import reextract_case_name
from mellea_lrc.llm import start_mellea_session_from_env


def main() -> None:
    """Parse CLI args, run re-extraction, and print JSON."""
    args = _parse_args()
    load_dotenv()
    context = _read_context(args.context_file)
    session = start_mellea_session_from_env()
    result = asyncio.run(
        reextract_case_name(
            session,
            document_context=context,
            extracted_case_name=args.extracted_case_name,
            courtlistener_case_name=args.courtlistener_case_name,
            attempts=args.attempts,
        )
    )
    print(json.dumps(result.to_json(), indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-file",
        type=Path,
        help="Path containing local citation context. Reads stdin when omitted.",
    )
    parser.add_argument(
        "--extracted-case-name",
        default=None,
        help="Current extracted case name. Omit for no extracted case name.",
    )
    parser.add_argument(
        "--courtlistener-case-name",
        required=True,
        help="CourtListener case name to compare against.",
    )
    parser.add_argument("--attempts", type=int, default=3, help="Maximum re-extraction attempts.")
    return parser.parse_args()


def _read_context(path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    return sys.stdin.read()


if __name__ == "__main__":
    main()
