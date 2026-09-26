"""Resume saved ambiguous reporter rule checks through model review.

Run from the repository root::

    uv run python -m evaluations.run_reporter_root_lookup_ambiguous_llm \
        --input-run-dir local/reporter-root-lookup-ambiguous \
        --run-dir local/reporter-root-lookup-ambiguous-llm

Each output Document retains its lookup and rule-review checkpoints. An
interrupted run resumes at the first filing without a saved model review.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mellea_lrc.validation.reporter_root_lookup_ambiguous_llm import (
    STAGE,
    reporter_root_lookup_ambiguous_llm,
)

from evaluations.annotations import SETS
from evaluations.run_reporter_root_lookup import _checkpoint, _persist_document, _save, _source_document
from mellea_lrc.validation.reporter_root_lookup_ambiguous import STAGE as AMBIGUOUS_STAGE


def _run_spec(data_root: Path, input_run_dir: Path, run_dir: Path) -> str:
    """Validate the saved rule-review run and describe the new checkpoint."""
    if input_run_dir.resolve() == run_dir.resolve():
        raise ValueError("The model-review run needs its own output directory")
    input_spec = json.loads((input_run_dir / "run.json").read_text(encoding="utf-8"))
    if input_spec.get("source_data_root") != str(data_root.resolve()):
        raise ValueError("Input run uses a different source dataset")
    if input_spec.get("checkpoints", [])[-1:] != [AMBIGUOUS_STAGE]:
        raise ValueError("Input run must end at the ambiguous reporter rule checkpoint")
    return (
        json.dumps(
            {
                **input_spec,
                "checkpoints": [*input_spec["checkpoints"], STAGE],
                "input_run_dir": str(input_run_dir.resolve()),
                "input_stage": AMBIGUOUS_STAGE,
                "output_stage": STAGE,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


async def run_documents(
    data_root: Path,
    input_run_dir: Path,
    run_dir: Path,
    sets: tuple[str, ...] = ("primary",),
) -> dict[str, int]:
    """Review immutable rule-stage Documents and commit each filing."""
    names = tuple(dict.fromkeys(sets))
    if not names or any(name not in SETS for name in names):
        raise ValueError("Select one or more known annotated sets")
    spec = _run_spec(data_root, input_run_dir, run_dir)
    spec_path = run_dir / "run.json"
    if spec_path.exists():
        if spec_path.read_text(encoding="utf-8") != spec:
            raise ValueError(f"{spec_path}: run settings differ from this invocation")
    else:
        _save(spec_path, spec)

    counts = {"review_created": 0, "review_reused": 0}
    for name in names:
        manifest = json.loads((data_root / name / "documents.json").read_text(encoding="utf-8"))["documents"]
        for filename, metadata in sorted(manifest.items()):
            source = _source_document(data_root, name, filename, metadata)
            input_path = input_run_dir / "documents" / name / f"{filename}.json"
            before = _checkpoint(input_path, AMBIGUOUS_STAGE, source)
            output_path = run_dir / "documents" / name / f"{filename}.json"
            if output_path.exists():
                saved = _checkpoint(output_path, STAGE, source)
                if saved.get_stage(AMBIGUOUS_STAGE) != before:
                    raise ValueError(f"{output_path}: ambiguous rule checkpoint differs from input")
                counts["review_reused"] += 1
                continue
            try:
                reviewed = await reporter_root_lookup_ambiguous_llm(before)
            except Exception:
                print(
                    f"Ambiguous reporter model review stopped at {name}/{filename}; rerun to resume.",
                    file=sys.stderr,
                )
                raise
            _persist_document(output_path, reviewed)
            counts["review_created"] += 1
            print(f"Saved ambiguous reporter model review: {name}/{filename}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--input-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sets", nargs="+", choices=SETS, default=["primary"])
    args = parser.parse_args()
    counts = asyncio.run(run_documents(args.data_root, args.input_run_dir, args.run_dir, tuple(args.sets)))
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
