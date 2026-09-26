"""Combine independent reporter model-review checkpoints without another model run.

The unique and ambiguous reviewers work on disjoint reporter roots. This
command verifies their common lookup checkpoint, appends only the saved unique
review histories to the ambiguous-review Document, and saves one cumulative
checkpoint. It never copies an unverified provider result between citations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluations.annotations import SETS
from evaluations.run_reporter_root_lookup import _checkpoint, _persist_document, _save, _source_document
from mellea_lrc.model import Document, FullReporterCitation
from mellea_lrc.validation.stage_names import (
    REPORTER_ROOT_LOOKUP,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
    REPORTER_ROOT_LOOKUP_UNIQUE_LLM,
)

AMBIGUOUS_STAGE = REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM
UNIQUE_STAGE = REPORTER_ROOT_LOOKUP_UNIQUE_LLM


def combine_documents(ambiguous: Document, unique: Document) -> Document:
    """Append disjoint unique-review histories after the ambiguous checkpoint."""
    if ambiguous.stage_runs[-1] != AMBIGUOUS_STAGE or unique.stage_runs[-1] != UNIQUE_STAGE:
        raise ValueError("Expected completed ambiguous and unique reporter review checkpoints")
    if ambiguous.get_stage(REPORTER_ROOT_LOOKUP) != unique.get_stage(REPORTER_ROOT_LOOKUP):
        raise ValueError("Reporter review branches do not share an identical lookup checkpoint")
    routed = {
        root.id
        for root in unique.get_stage(REPORTER_ROOT_LOOKUP).roots
        if isinstance(root, FullReporterCitation)
        and root.identity_judgments
        and root.identity_judgments[-1].next_stage == UNIQUE_STAGE
    }
    reviewed = {
        citation.id
        for citation in unique.citations
        if isinstance(citation, FullReporterCitation) and citation.reporter_unique_review is not None
    }
    if reviewed != routed:
        raise ValueError("Unique review branch does not cover exactly its routed reporter roots")
    by_id = {citation.id: citation for citation in ambiguous.citations}
    combined = ambiguous
    for reviewed in unique.citations:
        stage_nodes = [index for index, node in enumerate(reviewed.nodes) if node.stage == UNIQUE_STAGE]
        if not stage_nodes:
            continue
        if not isinstance(reviewed, FullReporterCitation) or len(stage_nodes) != 1:
            raise ValueError("Only one unique review node per full reporter root may be combined")
        prior = by_id.get(reviewed.id)
        if prior is None or reviewed._through_node_count(stage_nodes[0]) != prior:
            raise ValueError(f"Unique review changed its lookup input: {reviewed.id}")
        combined = combined.replace_citation(reviewed)
    combined = combined.complete(UNIQUE_STAGE)
    if combined.get_stage(AMBIGUOUS_STAGE) != ambiguous:
        raise ValueError("Combining unique reviews changed the ambiguous checkpoint")
    if combined.get_stage(REPORTER_ROOT_LOOKUP) != unique.get_stage(REPORTER_ROOT_LOOKUP):
        raise ValueError("Combining reviews changed their common lookup checkpoint")
    return combined


def _run_spec(data_root: Path, ambiguous_dir: Path, unique_dir: Path) -> str:
    ambiguous_spec = json.loads((ambiguous_dir / "run.json").read_text(encoding="utf-8"))
    unique_spec = json.loads((unique_dir / "run.json").read_text(encoding="utf-8"))
    source_root = str(data_root.resolve())
    if (
        ambiguous_spec.get("source_data_root") != source_root
        or unique_spec.get("source_data_root") != source_root
    ):
        raise ValueError("Reporter review branches use different source datasets")
    if ambiguous_spec.get("checkpoints", [])[-1:] != [AMBIGUOUS_STAGE]:
        raise ValueError("Ambiguous branch must end at its model-review checkpoint")
    if unique_spec.get("checkpoints", [])[-1:] != [UNIQUE_STAGE]:
        raise ValueError("Unique branch must end at its model-review checkpoint")
    if ambiguous_spec["checkpoints"][:2] != unique_spec["checkpoints"][:2]:
        raise ValueError("Reporter review branches do not share the same root and lookup stages")
    return (
        json.dumps(
            {
                "root_rules": ambiguous_spec["root_rules"],
                "hunt_dockets": ambiguous_spec["hunt_dockets"],
                "court_docket_fetch": ambiguous_spec["court_docket_fetch"],
                "source_data_root": source_root,
                "checkpoints": [*ambiguous_spec["checkpoints"], UNIQUE_STAGE],
                "combined_from": {
                    "ambiguous": str(ambiguous_dir.resolve()),
                    "unique": str(unique_dir.resolve()),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def run_documents(
    data_root: Path,
    ambiguous_dir: Path,
    unique_dir: Path,
    run_dir: Path,
    sets: tuple[str, ...] = ("primary",),
) -> dict[str, int]:
    """Persist one exact, resumable cumulative Document per source filing."""
    names = tuple(dict.fromkeys(sets))
    if not names or any(name not in SETS for name in names):
        raise ValueError("Select one or more known annotated sets")
    if len({ambiguous_dir.resolve(), unique_dir.resolve(), run_dir.resolve()}) != 3:
        raise ValueError("Input and output run directories must be distinct")
    spec = _run_spec(data_root, ambiguous_dir, unique_dir)
    spec_path = run_dir / "run.json"
    if spec_path.exists():
        if spec_path.read_text(encoding="utf-8") != spec:
            raise ValueError(f"{spec_path}: run settings differ from this invocation")
    else:
        _save(spec_path, spec)
    counts = {"created": 0, "reused": 0}
    for name in names:
        manifest = json.loads((data_root / name / "documents.json").read_text(encoding="utf-8"))["documents"]
        for filename, metadata in sorted(manifest.items()):
            source = _source_document(data_root, name, filename, metadata)
            ambiguous = _checkpoint(
                ambiguous_dir / "documents" / name / f"{filename}.json", AMBIGUOUS_STAGE, source
            )
            unique = _checkpoint(unique_dir / "documents" / name / f"{filename}.json", UNIQUE_STAGE, source)
            output_path = run_dir / "documents" / name / f"{filename}.json"
            if output_path.exists():
                saved = _checkpoint(output_path, UNIQUE_STAGE, source)
                if saved.get_stage(AMBIGUOUS_STAGE) != ambiguous or saved != combine_documents(
                    ambiguous, unique
                ):
                    raise ValueError(f"{output_path}: combined checkpoint differs from its inputs")
                counts["reused"] += 1
                continue
            _persist_document(output_path, combine_documents(ambiguous, unique))
            counts["created"] += 1
            print(f"Combined reporter reviews: {name}/{filename}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--ambiguous-run-dir", type=Path, required=True)
    parser.add_argument("--unique-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sets", nargs="+", choices=SETS, default=["primary"])
    args = parser.parse_args()
    counts = run_documents(
        args.data_root,
        args.ambiguous_run_dir,
        args.unique_run_dir,
        args.run_dir,
        tuple(args.sets),
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
