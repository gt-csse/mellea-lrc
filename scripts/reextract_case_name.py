"""Run the standalone case-name re-extraction workflow.

Keep this script aligned with retrieval case-name preparation: both workflows
should bind extracted names to the target locator, prefer copied text before the
locator, and debug prompt changes by rendering the raw Mellea instruct prompt.
Re-extraction differs only by using CourtListener's retrieved case name as an
extra identity cue; it is still not allowed to copy text from that cue.

Example:
    uv run --group llm python scripts/reextract_case_name.py \
      --context-file fixtures/bookmarked/bookmarked.txt \
      --citation-locator "999 U.S. 999" \
      --extracted-case-name "<NO_EXTRACTED_CASE_NAME>" \
      --courtlistener-case-name "Brown v. Board"

"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mellea_lrc.assessment import reextract_case_name
from mellea_lrc.core.env import load_env_file
from mellea_lrc.llm import start_mellea_session_from_env


def main() -> None:
    """Parse CLI args, run re-extraction, and print JSON."""
    args = _parse_args()
    load_env_file(Path(".env"))
    context = _read_context(args.context_file)
    session = start_mellea_session_from_env()
    result = asyncio.run(
        reextract_case_name(
            session,
            document_context=context,
            extracted_case_name=args.extracted_case_name,
            courtlistener_case_name=args.courtlistener_case_name,
            citation_locator=args.citation_locator,
        )
    )
    sys.stdout.write(f"{json.dumps(result.to_json(), indent=2)}\n")


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
    parser.add_argument(
        "--citation-locator",
        default=None,
        help="Matched locator text for the target citation, used to bind the copied case name.",
    )
    return parser.parse_args()


def _read_context(path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    return sys.stdin.read()


if __name__ == "__main__":
    main()
