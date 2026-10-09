"""Evaluate saved workflow checkpoints without rerunning extraction or validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluations.grow_leaves import render_grow_leaves, score_grow_leaves
from evaluations.grow_roots import render_grow_roots, score_grow_roots
from evaluations.validate_pincite import render_validate_pincite, score_validate_pincite
from evaluations.validate_roots import render_validate_roots, score_validate_roots
from mellea_lrc.api import Document
from mellea_lrc.model.execution import get_workflow

_WORKFLOWS = {
    "grow_roots": (score_grow_roots, render_grow_roots),
    "validate_roots": (score_validate_roots, render_validate_roots),
    "grow_leaves": (score_grow_leaves, render_grow_leaves),
    "validate_pincite": (score_validate_pincite, render_validate_pincite),
}


def score_run(run_dir: Path, workflows: tuple[str, ...] | None = None) -> dict[str, str]:
    """Write each requested workflow's JSON and Markdown beside one saved run."""
    if workflows is not None:
        if len(workflows) != len(set(workflows)):
            raise ValueError("Each workflow can have only one report")
        unknown = set(workflows).difference(_WORKFLOWS)
        if unknown:
            raise ValueError(f"Unknown workflows: {', '.join(sorted(unknown))}")
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if record["status"] != "complete":
        raise ValueError(f"Cannot score an incomplete run: {run_dir}")
    filenames = record["filings"]
    if {path.name for path in (run_dir / "documents").glob("*.json")} != {
        f"{filename}.json" for filename in filenames
    }:
        raise ValueError("Run Documents do not match its document list")
    documents = tuple(
        Document.model_validate_json((run_dir / "documents" / f"{filename}.json").read_text(encoding="utf-8"))
        for filename in filenames
    )
    if not documents:
        raise ValueError("Cannot score a run without Documents")
    if "annotation_sha256" in record:
        annotation_hashes = {}
        for filename, document in zip(filenames, documents, strict=True):
            if document.source_path is None:
                raise ValueError("Annotation provenance requires an official Document source")
            source = Path(document.source_path).resolve()
            annotation = source.parent.parent / "documents" / f"{source.stem}.jsonl"
            annotation_hashes[filename] = hashlib.sha256(annotation.read_bytes()).hexdigest()
        if annotation_hashes != record["annotation_sha256"]:
            raise ValueError("Run annotation content differs from saved provenance")
    if workflows is None:
        workflows = tuple(
            name
            for name in _WORKFLOWS
            if all(
                get_workflow(name).stages[0].substages[0].name in document.substage_runs
                for document in documents
            )
        )
    rendered: dict[str, str] = {}
    for workflow in workflows:
        scorer, renderer = _WORKFLOWS[workflow]
        total = scorer(documents[0])
        for document in documents[1:]:
            total += scorer(document)
        (run_dir / f"{workflow}.json").write_text(
            json.dumps({"set": record["set"], **total.as_dict()}, indent=2) + "\n",
            encoding="utf-8",
        )
        report = renderer(total, set_name=record["set"])
        (run_dir / f"{workflow}.md").write_text(report, encoding="utf-8")
        rendered[workflow] = report
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--workflow", choices=tuple(_WORKFLOWS), action="append")
    args = parser.parse_args()
    workflows = tuple(dict.fromkeys(args.workflow)) if args.workflow else None
    for name, report in score_run(args.run_dir.resolve(), workflows).items():
        print(f"{name}:\n{report}", end="")


if __name__ == "__main__":
    main()
