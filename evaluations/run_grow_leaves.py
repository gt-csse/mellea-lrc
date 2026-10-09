"""Resume a corpus from native Documents and append the leaf workflow."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

from evaluations import complete_stage_boundary
from mellea_lrc.api import (
    Document,
    attribute_id_citations,
    attribute_reference_citations,
    attribute_short_reporter_citations,
    attribute_supra_citations_rule,
    correct_leaf_fields,
    find_id_citations,
    find_reference_citations,
    find_short_reporter_citations,
    find_supra_citations,
    resolve_short_reporter_case_names,
    resolve_short_reporter_colocations,
    resolve_supra_case_names,
    resolve_supra_pin_cites,
    review_supra_attributions,
)
from mellea_lrc.workflows import _require_workflow_prefix

_SETS = frozenset(
    {"primary", "hallucination-set-1", "hallucination-set-2", "reliable-high-profile", "reliable-low-profile"}
)
_RESULTS_ROOT = Path(__file__).resolve().parent / "results"
_SUBSTAGES = (
    ("grow_leaves.short_reporter_citations.discovery", find_short_reporter_citations),
    ("grow_leaves.short_reporter_citations.colocations", resolve_short_reporter_colocations),
    ("grow_leaves.short_reporter_citations.case_names", resolve_short_reporter_case_names),
    ("grow_leaves.short_reporter_citations.attribution", attribute_short_reporter_citations),
    ("grow_leaves.reference_citations.discovery", find_reference_citations),
    ("grow_leaves.reference_citations.attribution", attribute_reference_citations),
    ("grow_leaves.id_citations.discovery", find_id_citations),
    ("grow_leaves.id_citations.attribution", attribute_id_citations),
    ("grow_leaves.supra_citations.discovery", find_supra_citations),
    ("grow_leaves.supra_citations.case_names", resolve_supra_case_names),
    ("grow_leaves.supra_citations.pin_cites", resolve_supra_pin_cites),
    ("grow_leaves.supra_citations.rule_attribution", attribute_supra_citations_rule),
    ("grow_leaves.supra_citations.llm_attribution", review_supra_attributions),
    ("grow_leaves.leaf_field_correction.review", correct_leaf_fields),
)


def _write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


async def run(
    input_documents: Path | None,
    *,
    input_substage: str = "validate_roots.locator_body_corroboration.llm_judgment",
    review_leaves: bool = True,
    stop_after: str | None = None,
    resume_run: Path | None = None,
) -> Path:
    if resume_run is None:
        if input_documents is None:
            raise ValueError("Specify saved input Documents")
        parent = json.loads((input_documents.parent / "run.json").read_text())
        set_name = parent.get("set")
        if set_name not in _SETS:
            raise ValueError(f"Unsupported corpus: {set_name}")
        if parent.get("status") != "complete":
            raise ValueError("Leaf input must be a complete run")
        run_dir = _RESULTS_ROOT / set_name / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        record = {
            "set": set_name,
            "workflow": "grow_leaves",
            "status": "running",
            "filings": parent["filings"],
            "input_documents": str(input_documents),
            "input_substage": input_substage,
            "review_leaves": review_leaves,
            "stop_after": stop_after,
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
        if record.get("set") not in _SETS:
            raise ValueError(f"Unsupported corpus: {record.get('set')}")
        input_documents = Path(record["input_documents"])
        parent = json.loads((input_documents.parent / "run.json").read_text())
        if parent.get("set") != record["set"]:
            raise ValueError("Leaf input corpus differs from the saved run")
        if parent.get("status") != "complete":
            raise ValueError("Leaf input must be a complete run")
        if parent.get("filings") != record["filings"]:
            raise ValueError("Leaf input filings differ from the saved run")
        input_substage = record["input_substage"]
        review_leaves = record["review_leaves"]
        if stop_after is not None and stop_after != record.get("stop_after"):
            raise ValueError("Resume must use the saved stop substage")
        stop_after = record.get("stop_after")
    stages = [
        (name, call)
        for name, call in _SUBSTAGES
        if name != "grow_leaves.supra_citations.llm_attribution" or review_leaves
    ]
    # Any completed leaf checkpoint can be an input too. The one cumulative
    # Document supplies both the old history and the remaining substage boundary.
    names = [name for name, _ in stages]
    enabled = tuple(names)
    all_names = tuple(name for name, _ in _SUBSTAGES)
    omitted = () if review_leaves else ("grow_leaves.supra_citations.llm_attribution",)
    if stop_after is not None:
        if stop_after not in names:
            raise ValueError(f"Stop substage is not enabled: {stop_after}")
        if input_substage in all_names and all_names.index(stop_after) <= all_names.index(input_substage):
            raise ValueError("Stop substage must follow the input substage")
        stages = stages[: names.index(stop_after) + 1]
    if input_substage in all_names:
        stages = [
            (name, call) for name, call in stages if all_names.index(name) > all_names.index(input_substage)
        ]
    if resume_run is None:
        run_dir.mkdir(parents=True)
        (run_dir / "documents").mkdir()
    record["status"] = "running"
    _write(run_dir / "run.json", json.dumps(record, indent=2) + "\n")
    try:
        for index, name in enumerate(record["filings"], 1):
            source = input_documents / f"{name}.json"
            if hashlib.sha256(source.read_bytes()).hexdigest() != record["input_sha256"][name]:
                raise ValueError(f"Leaf input changed: {name}")
            original = Document.model_validate_json(source.read_text()).get_substage(input_substage)
            _require_workflow_prefix(original, "grow_leaves", omitted_substages=omitted)
            original = complete_stage_boundary(original, (*enabled, *original.substage_runs))
            artifact = run_dir / "documents" / f"{name}.json"
            document = Document.model_validate_json(artifact.read_text()) if artifact.exists() else original
            _require_workflow_prefix(document, "grow_leaves", omitted_substages=omitted)
            document = complete_stage_boundary(document, (*enabled, *document.substage_runs))
            last = original.runs[-1]
            recover = document.get_stage if last.kind == "stage" else document.get_substage
            if recover(last.name) != original:
                raise ValueError(f"Saved leaf input differs: {name}")
            finished = document.substage_runs[len(original.substage_runs) :]
            if finished != tuple(substage for substage, _ in stages[: len(finished)]):
                raise ValueError(f"Unexpected leaf checkpoint: {name}")
            for substage, call in stages[len(finished) :]:
                if call in (
                    attribute_short_reporter_citations,
                    attribute_reference_citations,
                    attribute_id_citations,
                    correct_leaf_fields,
                ):
                    result = call(document, review=review_leaves)
                else:
                    result = call(document)
                document = await result if inspect.isawaitable(result) else result
                document = complete_stage_boundary(document, enabled)
                _write(artifact, document.model_dump_json(indent=2) + "\n")
                print(f"{index}/{len(record['filings'])} {name}: {substage}", flush=True)
            _write(artifact, document.model_dump_json(indent=2) + "\n")
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
    parser.add_argument("--input-substage", default="validate_roots.locator_body_corroboration.llm_judgment")
    parser.add_argument("--rule-only", action="store_true")
    parser.add_argument("--stop-after", choices=tuple(name for name, _ in _SUBSTAGES))
    parser.add_argument("--resume-run", type=Path)
    args = parser.parse_args()
    if bool(args.input_documents) == bool(args.resume_run):
        parser.error("Choose input Documents or resume a leaf run")
    print(
        asyncio.run(
            run(
                args.input_documents.resolve() if args.input_documents else None,
                input_substage=args.input_substage,
                review_leaves=not args.rule_only,
                stop_after=args.stop_after,
                resume_run=args.resume_run.resolve() if args.resume_run else None,
            )
        )
    )


if __name__ == "__main__":
    main()
