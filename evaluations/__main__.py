"""Run the first two grow_roots stage evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluations.grow_roots import evaluate_set, render_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--set", action="append", dest="sets", default=None)
    parser.add_argument(
        "--run-dir", type=Path, help="Read saved Documents instead of rerunning the two stages"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sets = args.sets or ["primary"]
    results = [evaluate_set(args.data_root, set_name, run_dir=args.run_dir) for set_name in sets]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report = render_markdown(results)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
