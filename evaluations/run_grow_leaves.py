"""Resume the primary corpus from native Documents and append the leaf workflow."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from mellea_lrc.api import (
    Document,
    attribute_id_citations,
    attribute_leaves_rule,
    find_id_citations,
    find_reference_citations,
    find_short_reporter_citations,
    find_supra_citations,
    resolve_leaf_case_names,
    resolve_leaf_pin_cites,
    review_id_attributions,
    review_leaf_attributions,
)

_SET = "primary"
_RESULTS_ROOT = Path(__file__).resolve().parent / "results" / _SET
_STAGES = (
    ("28_short_reporter_citations", find_short_reporter_citations),
    ("29_supra_citations", find_supra_citations),
    ("30_id_citations", find_id_citations),
    ("31_reference_citations", find_reference_citations),
    ("32_leaf_case_names", resolve_leaf_case_names),
    ("33_leaf_pin_cites", resolve_leaf_pin_cites),
    ("34_leaf_attribution_rule", attribute_leaves_rule),
    ("35_leaf_attribution_llm", review_leaf_attributions),
    ("36_id_attribution", attribute_id_citations),
    ("37_id_attribution_llm", review_id_attributions),
)


def _write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


async def run(
    input_documents: Path | None,
    *,
    input_stage: str = "23_locator_body_llm_judgment",
    review_leaves: bool = True,
    resume_run: Path | None = None,
) -> Path:
    if resume_run is None:
        if input_documents is None:
            raise ValueError("Specify saved input Documents")
        parent = json.loads((input_documents.parent / "run.json").read_text())
        if parent["set"] != _SET or parent["status"] != "complete":
            raise ValueError("Leaf input must be a complete primary run")
        run_dir = _RESULTS_ROOT / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_dir.mkdir(parents=True)
        (run_dir / "documents").mkdir()
        record = {
            "set": _SET,
            "workflow": "grow_leaves",
            "status": "running",
            "filings": parent["filings"],
            "input_documents": str(input_documents),
            "input_stage": input_stage,
            "review_leaves": review_leaves,
            "input_sha256": {
                name: hashlib.sha256((input_documents / f"{name}.json").read_bytes()).hexdigest()
                for name in parent["filings"]
            },
        }
    else:
        run_dir = resume_run
        record = json.loads((run_dir / "run.json").read_text())
        if record.get("workflow") != "grow_leaves":
            raise ValueError("Resume is not a grow_leaves run")
        input_documents = Path(record["input_documents"])
        input_stage = record["input_stage"]
        review_leaves = record["review_leaves"]
    stages = [(name, call) for name, call in _STAGES if not name.endswith("_llm") or review_leaves]
    # Any completed leaf checkpoint can be an input too. The one cumulative
    # Document supplies both the old history and the remaining stage boundary.
    names = [name for name, _ in stages]
    if input_stage in names:
        stages = stages[names.index(input_stage) + 1 :]
    record["status"] = "running"
    _write(run_dir / "run.json", json.dumps(record, indent=2) + "\n")
    try:
        for index, name in enumerate(record["filings"], 1):
            source = input_documents / f"{name}.json"
            if hashlib.sha256(source.read_bytes()).hexdigest() != record["input_sha256"][name]:
                raise ValueError(f"Leaf input changed: {name}")
            original = Document.model_validate_json(source.read_text()).get_stage(input_stage)
            artifact = run_dir / "documents" / f"{name}.json"
            document = Document.model_validate_json(artifact.read_text()) if artifact.exists() else original
            if document.get_stage(input_stage) != original:
                raise ValueError(f"Saved leaf input differs: {name}")
            finished = document.stage_runs[len(original.stage_runs) :]
            if finished != tuple(stage for stage, _ in stages[: len(finished)]):
                raise ValueError(f"Unexpected leaf checkpoint: {name}")
            for stage, call in stages[len(finished) :]:
                document = await call(document) if stage.endswith("_llm") else call(document)
                _write(artifact, document.model_dump_json(indent=2) + "\n")
                print(f"{index}/{len(record['filings'])} {name}: {stage}", flush=True)
            print(f"{index}/{len(record['filings'])} {name}: complete", flush=True)
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        _write(run_dir / "run.json", json.dumps(record, indent=2) + "\n")
        raise
    record["status"] = "complete"
    record.pop("error", None)
    _write(run_dir / "run.json", json.dumps(record, indent=2) + "\n")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-documents", type=Path)
    parser.add_argument("--input-stage", default="23_locator_body_llm_judgment")
    parser.add_argument("--rule-only", action="store_true")
    parser.add_argument("--resume-run", type=Path)
    args = parser.parse_args()
    if bool(args.input_documents) == bool(args.resume_run):
        parser.error("Choose input Documents or resume a leaf run")
    print(
        asyncio.run(
            run(
                args.input_documents.resolve() if args.input_documents else None,
                input_stage=args.input_stage,
                review_leaves=not args.rule_only,
                resume_run=args.resume_run.resolve() if args.resume_run else None,
            )
        )
    )


if __name__ == "__main__":
    main()
