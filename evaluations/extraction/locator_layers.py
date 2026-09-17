"""Run the isolated locator, colocation, and docket-audit evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluations.extraction.eval_colocation import eval_colocation
from evaluations.extraction.eval_docket_audit import eval_docket_audit
from evaluations.extraction.eval_locators import eval_locators
from evaluations.extraction.locator_eval_common import DEFAULT_DATA, grow_annotated_corpus


def evaluate(annotations: Path, texts_root: Path) -> dict[str, Any]:
    """Run the shared grow-roots layer, then combine the isolated scores."""
    corpus = grow_annotated_corpus(annotations, texts_root)
    locator_score = eval_locators(corpus)
    colocation_score = eval_colocation(corpus)
    audit_score = eval_docket_audit(corpus)
    return {
        **locator_score,
        "colocation_groups": colocation_score["colocation_groups"],
        "docket_audit": audit_score["docket_audit"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_DATA / "annotation-v4.0" / "documents",
    )
    parser.add_argument("--texts-root", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.annotations, args.texts_root), indent=2))


if __name__ == "__main__":
    main()
