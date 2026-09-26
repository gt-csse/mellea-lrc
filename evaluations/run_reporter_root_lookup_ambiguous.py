"""Resume saved reporter lookups through the rule-only ambiguous branch.

Run from the repository root::

    uv run python -m evaluations.run_reporter_root_lookup_ambiguous \
        --input-run-dir local/reporter-root-lookup \
        --run-dir local/reporter-root-lookup-ambiguous

Each output Document retains the complete lookup checkpoint and can
be scored by ``evaluations.evaluate_run`` without further provider calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluations.annotations import SETS
from evaluations.run_reporter_root_lookup import _checkpoint, _persist_document, _save, _source_document
from mellea_lrc.validation.reporter_root_lookup import STAGE as LOOKUP_STAGE
from mellea_lrc.validation.reporter_root_lookup_ambiguous import STAGE, reporter_root_lookup_ambiguous


def run_documents(
    data_root: Path,
    input_run_dir: Path,
    run_dir: Path,
    sets: tuple[str, ...] = ("primary",),
) -> dict[str, int]:
    """Use immutable lookup documents and resume safely by filing."""
    names = tuple(dict.fromkeys(sets))
    if not names or any(name not in SETS for name in names):
        raise ValueError("Select one or more known annotated sets")
    if input_run_dir.resolve() == run_dir.resolve():
        raise ValueError("The ambiguity run needs its own output directory")
    input_spec = json.loads((input_run_dir / "run.json").read_text(encoding="utf-8"))
    if input_spec.get("source_data_root") != str(data_root.resolve()):
        raise ValueError("Input run uses a different source dataset")
    spec = (
        json.dumps(
            {
                **input_spec,
                "checkpoints": [*input_spec["checkpoints"], STAGE],
                "input_run_dir": str(input_run_dir.resolve()),
                "input_stage": LOOKUP_STAGE,
                "output_stage": STAGE,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    spec_path = run_dir / "run.json"
    if spec_path.exists():
        if spec_path.read_text(encoding="utf-8") != spec:
            raise ValueError(f"{spec_path}: run settings differ from this invocation")
    else:
        _save(spec_path, spec)
    counts = {"ambiguity_created": 0, "ambiguity_reused": 0}
    for name in names:
        manifest = json.loads((data_root / name / "documents.json").read_text(encoding="utf-8"))["documents"]
        for filename, metadata in sorted(manifest.items()):
            source = _source_document(data_root, name, filename, metadata)
            lookup_path = input_run_dir / "documents" / name / f"{filename}.json"
            lookup = _checkpoint(lookup_path, LOOKUP_STAGE, source)
            output_path = run_dir / "documents" / name / f"{filename}.json"
            if output_path.exists():
                saved = _checkpoint(output_path, STAGE, source)
                if saved.get_stage(LOOKUP_STAGE) != lookup:
                    raise ValueError(f"{output_path}: lookup checkpoint differs from input")
                counts["ambiguity_reused"] += 1
                continue
            resolved = reporter_root_lookup_ambiguous(lookup)
            _persist_document(output_path, resolved)
            counts["ambiguity_created"] += 1
            print(f"Saved ambiguity review: {name}/{filename}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--input-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sets", nargs="+", choices=SETS, default=["primary"])
    args = parser.parse_args()
    counts = run_documents(args.data_root, args.input_run_dir, args.run_dir, tuple(args.sets))
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
