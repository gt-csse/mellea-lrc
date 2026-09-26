"""Score the primary grow_roots workflow and persist its Documents and report."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evaluations.grow_roots import WorkflowScore, render_grow_roots, score_grow_roots
from mellea_lrc.api import Document, grow_roots

_SET = "primary"
_DATA_ROOT = Path(__file__).resolve().parents[2] / "mellea-lrc-datasets"
_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / _SET


async def _run(data_root: Path, output_dir: Path, saved_documents: Path | None) -> WorkflowScore:
    manifest = json.loads((data_root / _SET / "documents.json").read_text(encoding="utf-8"))["documents"]
    total: WorkflowScore | None = None
    for filename in sorted(manifest):
        artifact = (saved_documents / f"{filename}.json") if saved_documents is not None else None
        if artifact is not None:
            document = Document.model_validate_json(artifact.read_text(encoding="utf-8"))
        else:
            document = await grow_roots(Document.from_source(data_root / _SET / "documents_txt" / filename))
            artifact = output_dir / "documents" / f"{filename}.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")
        score = score_grow_roots(document)
        total = score if total is None else total + score
    if total is None:
        raise ValueError(f"No documents in {_SET}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_DATA_ROOT)
    parser.add_argument("--saved-documents", type=Path, help="Directory of serialized Documents to score")
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT_DIR)
    args = parser.parse_args()
    score = asyncio.run(_run(args.data_root.resolve(), args.output_dir, args.saved_documents))
    result = {"set": _SET, **score.as_dict()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = render_grow_roots(score, set_name=_SET)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
