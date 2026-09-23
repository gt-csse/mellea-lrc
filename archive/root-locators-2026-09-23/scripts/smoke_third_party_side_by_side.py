"""Run one live third-party review against saved source and citing excerpts.

This deliberately skips provider fetching while exercising the real Mellea
schema, grounding, correction, judgment, and Document serialization paths.
The cited excerpt is saved in an earlier resolution artifact; its span starts
at zero here, so the output's document_span indexes that saved excerpt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from mellea_lrc.llm import start_mellea_session_from_env
from mellea_lrc.model.document import Document
from mellea_lrc.model.record import Question
from mellea_lrc.validation import body_search
from mellea_lrc.validation.types import RootBodySearchSource

if TYPE_CHECKING:
    from mellea_lrc.model.record import CitationRecord


def _read_document(path: Path) -> Document:
    return Document.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _record(document: Document, citation_id: str) -> CitationRecord:
    return next(record for record in document.citations if record.citation_id == citation_id)


async def run(
    source_path: Path,
    evidence_path: Path,
    citation_id: str,
    output_path: Path,
    evidence_index: int | None,
) -> dict[str, object]:
    """Review one source root against one archived independent citation."""
    load_dotenv()
    source_document = _read_document(source_path)
    source_record = _record(source_document, citation_id)
    archived_record = _record(_read_document(evidence_path), citation_id)
    archived_review = next(
        node for node in archived_record.trace if node.node_id.endswith(":root_body:third_party_review")
    )
    if evidence_index is None:
        evidence_index = archived_review.details.get("selected_evidence_index")
    raw_evidence = archived_review.details.get("evidence")
    if not isinstance(raw_evidence, list):
        raise ValueError("Archived review contains no citing excerpt")
    selected = next((item for item in raw_evidence if item.get("index") == evidence_index), None)
    if selected is None:
        raise ValueError(f"No archived evidence has index {evidence_index}")
    excerpt = body_search._ThirdPartyEvidence(
        index=selected["index"],
        source=RootBodySearchSource(selected["source"]),
        citing_id=selected["citing_id"],
        citing_case_name=selected["citing_case_name"],
        excerpt=selected["excerpt"],
        excerpt_start=0,
        origin=f"saved_excerpt:{selected['origin']}",
    )
    # The search checkpoint already holds the provider candidates. Supply the
    # saved excerpt as the fetched body so this smoke test spends no API quota.
    body_search._third_party_evidence = lambda *args, **kwargs: (excerpt,)
    scoped_document = replace(source_document, citations=(source_record,))
    resolved = await body_search.resolve_root_body_corroboration(
        scoped_document,
        session=start_mellea_session_from_env(),
        client=object(),  # type: ignore[arg-type]
        govinfo_client=object(),  # type: ignore[arg-type]
    )
    record = resolved.citations[0]
    review = next(node for node in record.trace if node.node_id.endswith(":root_body:third_party_review"))
    ivr = review.details.get("ivr")
    summary: dict[str, object] = {
        "source_checkpoint": str(source_path),
        "evidence_checkpoint": str(evidence_path),
        "citation_id": citation_id,
        "evidence_index": evidence_index,
        "citation_quote": review.details.get("citation_quote"),
        "source_fields": review.details.get("source_fields"),
        "cited_fields": review.details.get("cited_fields"),
        "field_comparisons": review.details.get("field_comparisons"),
        "citation_grounding": review.details.get("citation_grounding"),
        "reason": review.message,
        "review_outcome": review.outcome,
        "identity_outcome": record.judgement(Question.IDENTITY).outcome,
        "field_updates": [update.field.value for update in record.field_updates],
        "model": ivr.get("model") if isinstance(ivr, dict) else None,
        "ivr_success": ivr.get("success") if isinstance(ivr, dict) else None,
        "ivr_attempts": len(ivr.get("attempts", ())) if isinstance(ivr, dict) else None,
        "error": review.details.get("error"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(
        output_path.write_text,
        json.dumps(resolved.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    await asyncio.to_thread(
        output_path.with_suffix(".summary.json").write_text,
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--citation-id", required=True)
    parser.add_argument("--evidence-index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = asyncio.run(run(args.source, args.evidence, args.citation_id, args.output, args.evidence_index))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
