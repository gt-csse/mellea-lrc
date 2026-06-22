"""Run the E2E pipeline from typed snapshots without the frontend."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mellea_lrc.assessment import (
    CitationAssessment,
    DocumentAssessment,
    ModifiedExtractedCitation,
    assess_case_name_exact_match,
    assess_year_exact_match,
    build_extracted_case_name,
    find_text_span_near_full_span,
    get_extended_span_text,
)
from mellea_lrc.assessment.types import CaseNameAssessmentStatus
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import run_extraction
from mellea_lrc.preprocessing import preprocess_plain_text
from mellea_lrc.serialization import (
    deserialize_document_validation,
    serialize_document_assessment,
    serialize_document_validation,
)
from mellea_lrc.validation import validate_extraction
from mellea_lrc.validation.types import CitationValidation, DocumentValidation, ValidationStatus
from scripts.e2e_backend.pipeline import _start_mellea_session_from_env

if TYPE_CHECKING:
    from mellea_lrc.extraction.types import DocumentExtraction

DEFAULT_INPUT = Path("local/test_data/test-1.txt")
DEFAULT_SNAPSHOT_DIR = Path("local/snapshots/e2e")


def main() -> None:
    """Run extraction, reusable validation, and optional Mellea assessment."""
    args = _parse_args()
    _load_dotenv(Path(".env"))

    input_path = args.input
    snapshot_dir = args.snapshot_dir
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    validation_path = snapshot_dir / f"{input_path.stem}.document_validation.json"
    assessment_path = snapshot_dir / f"{input_path.stem}.document_assessment.json"

    extraction = run_extraction(preprocess_plain_text(input_path))
    validation = _load_or_create_validation(
        validation_path,
        extraction,
        refresh=args.refresh_validation,
    )

    _emit_json(
        {
            "input": str(input_path),
            "validation_snapshot": str(validation_path),
            "assessment_snapshot": str(assessment_path),
            "citations": len(extraction.citations),
            "validations": _validation_counts(validation),
        }
    )

    if not args.assess_mellea:
        return

    assessment = _run_assessment(validation, max_mellea=args.max_mellea)
    assessment_path.write_text(
        json.dumps(serialize_document_assessment(assessment), indent=2),
        encoding="utf-8",
    )
    _emit(f"wrote assessment snapshot: {assessment_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--refresh-validation", action="store_true")
    parser.add_argument("--assess-mellea", action="store_true")
    parser.add_argument("--max-mellea", type=int, default=None)
    return parser.parse_args()


def _load_or_create_validation(
    path: Path,
    extraction: DocumentExtraction,
    *,
    refresh: bool,
) -> DocumentValidation:
    if path.exists() and not refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return deserialize_document_validation(payload)

    validation = validate_extraction(extraction)
    path.write_text(json.dumps(serialize_document_validation(validation), indent=2), encoding="utf-8")
    return validation


def _run_assessment(validation: DocumentValidation, *, max_mellea: int | None) -> DocumentAssessment:
    from mellea_lrc.assessment.mellea import assess_case_name_with_mellea  # noqa: PLC0415

    validations_by_id = {item.citation_id: item for item in validation.validations}
    assessments: list[CitationAssessment] = []
    modified_citations: list[ModifiedExtractedCitation] = []
    reassessments: list[CitationAssessment] = []
    session = None
    mellea_calls = 0

    for citation in validation.citations:
        citation_validation = validations_by_id.get(citation.citation_id)
        if citation_validation is None:
            continue
        if not isinstance(citation.citation, FullCaseCitation):
            continue
        if citation_validation.status != ValidationStatus.FOUND:
            continue

        extracted_case_name = build_extracted_case_name(citation.citation)
        courtlistener_case_name = _first_cluster_case_name(citation_validation)
        year_assess = assess_year_exact_match(
            citation_id=citation.citation_id,
            extracted_year=citation.citation.year,
            courtlistener_year=_first_cluster_year(citation_validation),
        )
        exact = assess_case_name_exact_match(
            citation_id=citation.citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
        )
        if exact.status != CaseNameAssessmentStatus.NEEDS_ASSESSMENT:
            assessments.append(
                CitationAssessment(
                    citation_id=citation.citation_id,
                    case_assess=exact,
                    year_assess=year_assess,
                )
            )
            continue
        if max_mellea is not None and mellea_calls >= max_mellea:
            continue

        session = session or _start_mellea_session_from_env()
        context = get_extended_span_text(validation.text, citation.span)
        _emit_json(
            {
                "mellea_call": mellea_calls + 1,
                "citation_id": citation.citation_id,
                "matched_text": citation.matched_text,
                "extracted_case_name": extracted_case_name,
                "courtlistener_case_name": courtlistener_case_name,
                "context": context,
            }
        )
        run = assess_case_name_with_mellea(
            session,
            citation_id=citation.citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            document_context=context,
        )
        mellea_calls += 1
        assessments.append(
            CitationAssessment(
                citation_id=citation.citation_id,
                case_assess=run.assessment,
                year_assess=year_assess,
            )
        )
        if run.modified_citation is not None:
            modified_citations.append(
                ModifiedExtractedCitation.from_proposal(
                    run.modified_citation,
                    citation_id=citation.citation_id,
                    span=find_text_span_near_full_span(
                        validation.text,
                        run.modified_citation.extracted_case_name or "",
                        citation.span,
                    ),
                )
            )
        if run.reassessment is not None:
            reassessments.append(
                CitationAssessment(
                    citation_id=citation.citation_id,
                    case_assess=run.reassessment,
                    year_assess=year_assess,
                )
            )

    return DocumentAssessment(
        preprocessed=validation.preprocessed,
        citations=validation.citations,
        validations=validation.validations,
        assessments=tuple(assessments),
        modified_citations=tuple(modified_citations),
        reassessments=tuple(reassessments),
    )


def _validation_counts(validation: DocumentValidation) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in validation.validations:
        counts[item.status.value] = counts.get(item.status.value, 0) + 1
    return counts


def _first_cluster_case_name(validation: CitationValidation) -> str | None:
    if not validation.clusters:
        return None
    case_name = validation.clusters[0].get("case_name") or validation.clusters[0].get("caseName")
    return str(case_name) if isinstance(case_name, str) and case_name else None


def _first_cluster_year(validation: CitationValidation) -> str | None:
    if not validation.clusters:
        return None
    date_filed = validation.clusters[0].get("date_filed") or validation.clusters[0].get("dateFiled")
    return str(date_filed)[:4] if isinstance(date_filed, str) and date_filed else None


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def _emit_json(payload: object) -> None:
    _emit(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
