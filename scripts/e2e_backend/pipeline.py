"""Assembled extraction and retrieval API for the Modal E2E backend."""

from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from mellea_lrc.assessment import (
    AssessedCitationAssessment,
    AssessedDocument,
    CitationAssessment,
    run_assessment_async,
)
from mellea_lrc.citation_nodes import (
    citation_node_document_to_json,
    nodes_from_extracted_document,
    with_assessment_steps,
    with_jurisdiction_steps,
    with_retrieval_steps,
)
from mellea_lrc.core.citations import FullCaseCitation, Reporter, UnknownCitation, citation_kind
from mellea_lrc.core.documents import SourceFormat, SourceMetadata
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import run_extraction
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing.docling import build_docling_converter, is_docling_supported_format
from mellea_lrc.preprocessing.types import (
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
)
from mellea_lrc.jurisdiction_inference import (
    InferredDocument,
    Jurisdiction,
    infer_jurisdiction,
)
from mellea_lrc.jurisdiction_inference.leads import evaluate_court_inference, evaluate_reporter_inference
from mellea_lrc.jurisdiction_inference.types import (
    CourtInference,
    CourtInferenceStatus,
    ReporterInference,
    ReporterInferenceStatus,
)
from mellea_lrc.retrieval import run_retrieval, run_retrieval_async
from mellea_lrc.retrieval.types import (
    CitationRetrieval,
    RetrievedDocument,
    RetrievalMetadata,
    RetrievalStatus,
)
from mellea_lrc.serialization import (
    deserialize_citation_retrieval,
    serialize_citation_assessment,
    serialize_citation_retrieval,
)
from mellea_lrc.serialization.json import serialize_jurisdiction

if TYPE_CHECKING:
    from collections.abc import Callable

    from mellea_lrc.retrieval import CourtListenerAccessClient

JsonDict = dict[str, Any]


class DoclingDocument(Protocol):
    """Protocol for the Docling document export surface used here."""

    def export_to_text(self) -> str:
        """Export extracted document content as plain text."""


class DoclingResult(Protocol):
    """Protocol for a Docling conversion result."""

    document: DoclingDocument


class DoclingConverter(Protocol):
    """Protocol for the Docling converter surface used here."""

    def convert(self, source: object) -> DoclingResult:
        """Convert a document source."""


class E2EBackend:
    """Well-defined API for text/PDF extraction, retrieval, and assessment."""

    def __init__(
        self,
        *,
        converter_factory: Callable[[], DoclingConverter] | None = None,
    ) -> None:
        self._converter_factory = converter_factory or _build_converter
        self._converter: DoclingConverter | None = None

    def review_text(self, text: str, *, retrieve: bool = True) -> dict[str, Any]:
        """Run the frontend review API for plain text input."""
        return review_preprocessed(_text_to_preprocessed(text), retrieve=retrieve)

    async def review_text_async(self, text: str, *, retrieve: bool = True) -> dict[str, Any]:
        """Run the frontend review API with async retrieval preparation."""
        return await review_preprocessed_async(_text_to_preprocessed(text), retrieve=retrieve)

    def review_preprocessed_document(self, preprocessed: PreprocessedDocument) -> dict[str, Any]:
        """Serialize a preprocessed document snapshot for the frontend review UI."""
        return review_preprocessed_document(preprocessed)

    def review_document_extraction(self, extraction: ExtractedDocument) -> dict[str, Any]:
        """Serialize an extraction snapshot for the frontend review UI."""
        return review_document_extraction(extraction)

    def review_citation_node_document(self, payload: dict[str, object]) -> dict[str, Any]:
        """Serialize a citation-node snapshot for the frontend review UI."""
        return review_citation_node_document(payload)

    def review_document_inference(self, inferred: InferredDocument) -> dict[str, Any]:
        """Serialize a jurisdiction-inferred document for the frontend review UI."""
        return review_document_inference(inferred)

    def review_document_retrieval(self, retrieval: RetrievedDocument) -> dict[str, Any]:
        """Serialize a retrieval snapshot for the frontend review UI."""
        return review_document_retrieval(retrieval)

    def extract_text(self, text: str) -> dict[str, Any]:
        """Extract frontend review citations from plain text without retrieval."""
        return review_preprocessed(_text_to_preprocessed(text), retrieve=False)

    def review_pdf_bytes(
        self,
        content: bytes,
        filename: str,
        *,
        retrieve: bool = True,
    ) -> dict[str, Any]:
        """Run the frontend review API for PDF bytes."""
        if content[:4] != b"%PDF":
            msg = f"{filename} is not a PDF"
            raise ValueError(msg)
        return review_preprocessed(
            _pdf_to_preprocessed(self._get_converter(), content, filename),
            retrieve=retrieve,
        )

    async def review_document_bytes_async(
        self,
        content: bytes,
        filename: str,
        *,
        retrieve: bool = True,
    ) -> dict[str, Any]:
        """Run the frontend review API for uploaded bytes with async retrieval preparation."""
        if Path(filename).suffix.lower() == ".txt":
            return await review_preprocessed_async(
                _text_file_to_preprocessed(content, filename),
                retrieve=retrieve,
            )
        return await review_preprocessed_async(
            _document_to_preprocessed(self._get_converter(), content, filename),
            retrieve=retrieve,
        )

    def review_document_bytes(
        self,
        content: bytes,
        filename: str,
        *,
        retrieve: bool = True,
    ) -> dict[str, Any]:
        """Run the frontend review API for an uploaded document."""
        if Path(filename).suffix.lower() == ".txt":
            return review_preprocessed(
                _text_file_to_preprocessed(content, filename),
                retrieve=retrieve,
            )
        return review_preprocessed(
            _document_to_preprocessed(self._get_converter(), content, filename),
            retrieve=retrieve,
        )

    def extract_document_bytes(
        self,
        content: bytes,
        filename: str,
    ) -> dict[str, Any]:
        """Extract frontend review citations from an uploaded document without retrieval."""
        if Path(filename).suffix.lower() == ".txt":
            return review_preprocessed(_text_file_to_preprocessed(content, filename), retrieve=False)
        return review_preprocessed(
            _document_to_preprocessed(self._get_converter(), content, filename),
            retrieve=False,
        )

    def retrieve_review_payload(
        self,
        payload: dict[str, object],
        *,
        client: CourtListenerAccessClient | None = None,
    ) -> dict[str, Any]:
        """Attach CourtListener retrieval to an existing frontend review payload."""
        return retrieve_review_payload(payload, client=client)

    async def retrieve_review_payload_async(
        self,
        payload: dict[str, object],
        *,
        client: CourtListenerAccessClient | None = None,
    ) -> dict[str, Any]:
        """Attach retrieval with mandatory async case-name preparation."""
        return await retrieve_review_payload_async(payload, client=client)

    def retrieve_review_citation_payload(
        self,
        payload: dict[str, object],
        *,
        client: CourtListenerAccessClient | None = None,
    ) -> dict[str, Any]:
        """Retrieve one frontend review citation."""
        return retrieve_review_citation_payload(payload, client=client)

    async def retrieve_review_citation_payload_async(
        self,
        payload: dict[str, object],
        *,
        client: CourtListenerAccessClient | None = None,
    ) -> dict[str, Any]:
        """Retrieve one frontend review citation with mandatory async preparation."""
        return await retrieve_review_citation_payload_async(payload, client=client)

    async def assess_review_payload(self, payload: dict[str, object]) -> dict[str, Any]:
        """Attach Mellea-assisted assessment to an existing retrieved review payload."""
        return await assess_review_payload(payload)

    def review_document_assessment(self, assessment: AssessedDocument) -> dict[str, Any]:
        """Serialize a cached assessment artifact for the frontend review UI."""
        return review_document_assessment(assessment)

    def _get_converter(self) -> DoclingConverter:
        if self._converter is None:
            self._converter = self._converter_factory()
        return self._converter


def review_preprocessed(
    preprocessed: PreprocessedDocument,
    *,
    retrieve: bool = True,
    client: CourtListenerAccessClient | None = None,
) -> dict[str, Any]:
    """Run extraction, jurisdiction inference, optional retrieval, and frontend review."""
    extraction = run_extraction(preprocessed)
    inferred = infer_jurisdiction(extraction)
    retrieval = _run_retrieval(inferred, retrieve=retrieve, client=client)

    return {
        "document": {
            "text": inferred.text,
            "source_path": inferred.source_path,
            "source_format": inferred.source_metadata.format.value,
            "backend": inferred.preprocessing_metadata.backend.value,
        },
        "citations": _citation_payloads(inferred, retrieval),
        "jurisdictions": _jurisdictions_payload(inferred),
        "retrieval": _retrieval_payload(retrieval),
        "node_graph": _node_graph_payload(inferred, retrieval=retrieval),
        "stats": _stats(inferred, retrieval),
    }


async def review_preprocessed_async(
    preprocessed: PreprocessedDocument,
    *,
    retrieve: bool = True,
    client: CourtListenerAccessClient | None = None,
) -> dict[str, Any]:
    """Run extraction, inference, and async retrieval with mandatory preparation."""
    extraction = run_extraction(preprocessed)
    inferred = infer_jurisdiction(extraction)
    retrieval = await _run_retrieval_async(inferred, retrieve=retrieve, client=client)

    return {
        "document": {
            "text": inferred.text,
            "source_path": inferred.source_path,
            "source_format": inferred.source_metadata.format.value,
            "backend": inferred.preprocessing_metadata.backend.value,
        },
        "citations": _citation_payloads(inferred, retrieval),
        "jurisdictions": _jurisdictions_payload(inferred),
        "retrieval": _retrieval_payload(retrieval),
        "node_graph": _node_graph_payload(inferred, retrieval=retrieval),
        "stats": _stats(inferred, retrieval),
    }


def review_preprocessed_document(preprocessed: PreprocessedDocument) -> dict[str, Any]:
    """Serialize a preprocessed document snapshot without running extraction."""
    return {
        "document": {
            "text": preprocessed.text,
            "source_path": preprocessed.source_metadata.path,
            "source_format": preprocessed.source_metadata.format.value,
            "backend": preprocessed.preprocessing_metadata.backend.value,
        },
        "citations": [],
        "retrieval": None,
        "assessment": None,
        "node_graph": None,
        "stats": {
            "chars": len(preprocessed.text),
            "citation_spans": 0,
            "full_citations": 0,
        },
    }


def _jurisdictions_payload(document: InferredDocument) -> list[dict[str, Any]]:
    """Serialize the document's jurisdiction inferences for the frontend."""
    return [serialize_jurisdiction(j) for j in document.jurisdictions]


def review_document_extraction(extraction: ExtractedDocument) -> dict[str, Any]:
    """Serialize a typed extraction artifact as a frontend review payload."""
    return {
        "document": {
            "text": extraction.text,
            "source_path": extraction.source_path,
            "source_format": extraction.source_metadata.format.value,
            "backend": extraction.preprocessing_metadata.backend.value,
        },
        "citations": _citation_payloads(extraction, None),
        "jurisdictions": None,
        "retrieval": None,
        "assessment": None,
        "node_graph": _node_graph_payload(extraction),
        "stats": _stats(extraction, None),
    }


def review_citation_node_document(payload: dict[str, object]) -> dict[str, Any]:
    """Serialize a citation-node snapshot as an extraction-level review payload."""
    text = str(payload.get("text") or "")
    nodes = _list_field(payload.get("nodes"))
    citations = [_citation_payload_from_node(node) for node in nodes if isinstance(node, dict)]
    return {
        "document": {
            "text": text,
            "source_path": None,
            "source_format": SourceFormat.UNKNOWN.value,
            "backend": PreprocessingBackend.PLAIN_TEXT.value,
        },
        "citations": citations,
        "jurisdictions": None,
        "retrieval": None,
        "assessment": None,
        "node_graph": payload,
        "stats": {
            "chars": len(text),
            "citation_spans": len(citations),
            "full_citations": sum(1 for item in citations if str(item.get("kind")).startswith("Full")),
        },
    }


def review_document_inference(inferred: InferredDocument) -> dict[str, Any]:
    """Serialize a typed jurisdiction-inferred artifact as a frontend review payload."""
    return {
        "document": {
            "text": inferred.text,
            "source_path": inferred.source_path,
            "source_format": inferred.source_metadata.format.value,
            "backend": inferred.preprocessing_metadata.backend.value,
        },
        "citations": _citation_payloads(inferred, None),
        "jurisdictions": _jurisdictions_payload(inferred),
        "retrieval": None,
        "assessment": None,
        "node_graph": _node_graph_payload(inferred),
        "stats": _stats(inferred, None),
    }


def review_document_retrieval(retrieval: RetrievedDocument) -> dict[str, Any]:
    """Serialize a typed retrieval artifact as a frontend review payload."""
    return {
        "document": {
            "text": retrieval.text,
            "source_path": retrieval.source_path,
            "source_format": retrieval.source_metadata.format.value,
            "backend": retrieval.preprocessing_metadata.backend.value,
        },
        "citations": _citation_payloads(retrieval, retrieval),
        "jurisdictions": _jurisdictions_payload(retrieval),
        "retrieval": _retrieval_payload(retrieval),
        "assessment": None,
        "node_graph": _node_graph_payload(retrieval, retrieval=retrieval),
        "stats": _stats(retrieval, retrieval),
    }


def retrieve_review_payload(
    payload: dict[str, object],
    *,
    client: CourtListenerAccessClient | None = None,
) -> dict[str, Any]:
    """Attach retrieval to an existing frontend review payload without re-extracting."""
    extraction = _extraction_from_review_payload(payload)
    retrieval = _run_retrieval(extraction, retrieve=True, client=client)
    output = _copy_review_payload(payload)
    retrieval_by_id = {item.citation_id: _retrieval_item_payload(item) for item in retrieval.retrievals}
    citations = _review_citations(output)
    for citation in citations:
        citation["retrieval"] = retrieval_by_id.get(str(citation.get("id")))
    output["citations"] = citations
    output["retrieval"] = _retrieval_payload(retrieval)
    output["node_graph"] = _node_graph_payload(retrieval, retrieval=retrieval)
    output["stats"] = _merge_stats(output.get("stats"), _stats(extraction, retrieval))
    return output


async def retrieve_review_payload_async(
    payload: dict[str, object],
    *,
    client: CourtListenerAccessClient | None = None,
) -> dict[str, Any]:
    """Attach retrieval to an existing frontend review payload with LLM preparation."""
    extraction = _extraction_from_review_payload(payload)
    retrieval = await _run_retrieval_async(extraction, retrieve=True, client=client)
    output = _copy_review_payload(payload)
    retrieval_by_id = {item.citation_id: _retrieval_item_payload(item) for item in retrieval.retrievals}
    citations = _review_citations(output)
    for citation in citations:
        citation["retrieval"] = retrieval_by_id.get(str(citation.get("id")))
    output["citations"] = citations
    output["retrieval"] = _retrieval_payload(retrieval)
    output["node_graph"] = _node_graph_payload(retrieval, retrieval=retrieval)
    output["stats"] = _merge_stats(output.get("stats"), _stats(extraction, retrieval))
    return output


def retrieve_review_citation_payload(
    payload: dict[str, object],
    *,
    client: CourtListenerAccessClient | None = None,
) -> dict[str, Any]:
    """Retrieve a single frontend review citation payload."""
    citation = payload.get("citation")
    if not isinstance(citation, dict):
        msg = "citation is required"
        raise TypeError(msg)

    extracted_citation = _extracted_citation_from_review_item(citation)
    retrieval_text = extracted_citation.matched_text or "citation"
    retrieval_citation = ExtractedCitation(
        citation_id=extracted_citation.citation_id,
        span=Span(start=0, end=len(retrieval_text)),
        matched_text=extracted_citation.matched_text,
        citation=extracted_citation.citation,
        resolves_to=None,
    )
    extraction = ExtractedDocument(
        source_metadata=SourceMetadata(),
        text=retrieval_text,
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend.PLAIN_TEXT,
        ),
        citations=(retrieval_citation,),
        extraction_metadata=ExtractionMetadata(),
    )
    retrieval = _run_retrieval(extraction, retrieve=True, client=client)
    if retrieval is None or not retrieval.retrievals:
        msg = "citation retrieval did not produce a result"
        raise ValueError(msg)
    return _retrieval_item_payload(retrieval.retrievals[0])


async def retrieve_review_citation_payload_async(
    payload: dict[str, object],
    *,
    client: CourtListenerAccessClient | None = None,
) -> dict[str, Any]:
    """Retrieve one frontend review citation with LLM preparation."""
    citation = payload.get("citation")
    if not isinstance(citation, dict):
        msg = "citation is required"
        raise TypeError(msg)

    extracted_citation = _extracted_citation_from_review_item(citation)
    retrieval_text = extracted_citation.matched_text or "citation"
    retrieval_citation = ExtractedCitation(
        citation_id=extracted_citation.citation_id,
        span=Span(start=0, end=len(retrieval_text)),
        matched_text=extracted_citation.matched_text,
        citation=extracted_citation.citation,
        resolves_to=None,
    )
    extraction = ExtractedDocument(
        source_metadata=SourceMetadata(),
        text=retrieval_text,
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend.PLAIN_TEXT,
        ),
        citations=(retrieval_citation,),
        extraction_metadata=ExtractionMetadata(),
    )
    retrieval = await _run_retrieval_async(extraction, retrieve=True, client=client)
    if retrieval is None or not retrieval.retrievals:
        msg = "citation retrieval did not produce a result"
        raise ValueError(msg)
    return _retrieval_item_payload(retrieval.retrievals[0])


async def assess_review_payload(payload: dict[str, object]) -> dict[str, Any]:
    """Attach Mellea-assisted case-name assessment to an existing review payload."""
    output = _copy_review_payload(payload)
    citations = _review_citations(output)
    extraction = _extraction_from_review_payload(output)
    retrieval = _retrieval_from_review_payload(output, extraction)
    document_assessment = await run_assessment_async(retrieval)
    assessment_by_id = {item.citation_id: item for item in document_assessment.assessments}
    for citation in citations:
        citation["assessment"] = _assessment_payload(assessment_by_id[str(citation.get("id"))])

    output["citations"] = citations
    output["assessment"] = _assessment_payload_document(document_assessment)
    output["node_graph"] = _node_graph_payload(
        document_assessment,
        retrieval=document_assessment,
        assessments=document_assessment.assessments,
    )
    output["stats"] = _merge_stats(
        output.get("stats"),
        {
            "assessed": _assessment_status_counts(document_assessment.assessments).get("assessed", 0),
            "case_name_counts": _assessment_case_name_counts(document_assessment.assessments),
            "court_counts": _assessment_court_counts(document_assessment.assessments),
            "year_counts": _assessment_year_counts(document_assessment.assessments),
        },
    )
    return output


def review_document_assessment(assessment: AssessedDocument) -> dict[str, Any]:
    """Serialize a typed assessment artifact as a frontend review payload."""
    output = review_document_retrieval(assessment)
    output["assessment"] = _assessment_payload_document(assessment)
    output["stats"] = _merge_stats(
        output.get("stats"),
        {
            "assessed": _assessment_status_counts(assessment.assessments).get("assessed", 0),
            "case_name_counts": _assessment_case_name_counts(assessment.assessments),
            "court_counts": _assessment_court_counts(assessment.assessments),
            "year_counts": _assessment_year_counts(assessment.assessments),
        },
    )
    assessment_by_id = {item.citation_id: _assessment_payload(item) for item in assessment.assessments}
    citations = _review_citations(output)
    for citation in citations:
        citation["assessment"] = assessment_by_id.get(str(citation.get("id")))
    output["citations"] = citations
    output["node_graph"] = _node_graph_payload(assessment, retrieval=assessment, assessments=assessment.assessments)
    return output


def _build_converter() -> DoclingConverter:
    return build_docling_converter(enable_pdf_ocr=True)  # type: ignore[return-value]


def _pdf_to_preprocessed(
    converter: DoclingConverter,
    content: bytes,
    filename: str,
) -> PreprocessedDocument:
    return _docling_to_preprocessed(converter, content, filename)


def _document_to_preprocessed(
    converter: DoclingConverter,
    content: bytes,
    filename: str,
) -> PreprocessedDocument:
    path = Path(filename)
    if not is_docling_supported_format(path):
        msg = f"Unsupported document format: {path.suffix or '<none>'}"
        raise ValueError(msg)
    return _docling_to_preprocessed(converter, content, filename)


def _docling_to_preprocessed(
    converter: DoclingConverter,
    content: bytes,
    filename: str,
) -> PreprocessedDocument:
    from docling.datamodel.base_models import DocumentStream  # noqa: PLC0415

    source = DocumentStream(name=filename, stream=BytesIO(content))
    result = converter.convert(source)
    return PreprocessedDocument(
        source_metadata=SourceMetadata(
            path=filename,
            format=_source_format(filename),
        ),
        text=result.document.export_to_text(),
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend.DOCLING,
        ),
    )


def _text_to_preprocessed(text: str) -> PreprocessedDocument:
    from mellea_lrc.preprocessing import preprocess_plain_text_from_string  # noqa: PLC0415

    return preprocess_plain_text_from_string(text)


def _text_file_to_preprocessed(content: bytes, filename: str) -> PreprocessedDocument:
    return PreprocessedDocument(
        source_metadata=SourceMetadata(path=filename, format=SourceFormat.TEXT),
        text=content.decode("utf-8"),
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend.PLAIN_TEXT,
        ),
    )


def _source_format(filename: str) -> SourceFormat:
    suffix = Path(filename).suffix.lower()
    return {
        ".pdf": SourceFormat.PDF,
        ".docx": SourceFormat.DOCX,
        ".pptx": SourceFormat.PPTX,
        ".xlsx": SourceFormat.XLSX,
        ".html": SourceFormat.HTML,
        ".htm": SourceFormat.HTML,
        ".md": SourceFormat.MARKDOWN,
    }.get(suffix, SourceFormat.UNKNOWN)


def _run_retrieval(
    document: ExtractedDocument,
    *,
    retrieve: bool,
    client: CourtListenerAccessClient | None,
) -> RetrievedDocument | None:
    if not retrieve:
        return None
    inferred = document if isinstance(document, InferredDocument) else infer_jurisdiction(document)
    return run_retrieval(
        inferred,
        client_mode="custom" if client is not None else "deployed",
        client=client,
    )


async def _run_retrieval_async(
    document: ExtractedDocument,
    *,
    retrieve: bool,
    client: CourtListenerAccessClient | None,
    mellea_concurrency: int | None = 5,
) -> RetrievedDocument | None:
    if not retrieve:
        return None
    inferred = document if isinstance(document, InferredDocument) else infer_jurisdiction(document)
    return await run_retrieval_async(
        inferred,
        client_mode="custom" if client is not None else "deployed",
        client=client,
        mellea_concurrency=mellea_concurrency,
    )


def _citation_payloads(
    extraction: ExtractedDocument,
    retrieval: RetrievedDocument | None,
) -> list[dict[str, Any]]:
    retrieval_by_id = {item.citation_id: item for item in retrieval.retrievals} if retrieval else {}
    return [
        {
            "id": item.citation_id,
            "start": item.span.start,
            "end": item.span.end,
            "matched_text": item.matched_text,
            "kind": citation_kind(item.citation).value,
            "fields": _citation_fields(item.citation),
            "resolves_to": item.resolves_to,
            "retrieval": _single_retrieval_payload(retrieval_by_id.get(item.citation_id)),
        }
        for item in extraction.citations
    ]


def _node_graph_payload(
    extraction: ExtractedDocument,
    *,
    retrieval: RetrievedDocument | None = None,
    assessments: tuple[CitationAssessment, ...] = (),
) -> dict[str, object]:
    node_document = nodes_from_extracted_document(extraction)
    if isinstance(extraction, InferredDocument):
        node_document = with_jurisdiction_steps(node_document, extraction)
    if retrieval is not None:
        node_document = with_retrieval_steps(node_document, retrieval)
    if assessments:
        node_document = with_assessment_steps(node_document, assessments)
    return citation_node_document_to_json(node_document)


def _citation_payload_from_node(node: dict[str, object]) -> dict[str, Any]:
    node_input = node.get("input") if isinstance(node.get("input"), dict) else {}
    span = node_input.get("span") if isinstance(node_input.get("span"), dict) else {}
    citation = node_input.get("citation") if isinstance(node_input.get("citation"), dict) else {}
    return {
        "id": str(node_input.get("citation_id") or node.get("citation_id") or ""),
        "start": _optional_int(span.get("start")) or 0,
        "end": _optional_int(span.get("end")) or 0,
        "matched_text": str(node_input.get("matched_text") or ""),
        "kind": str(citation.get("type") or citation.get("kind") or "UnknownCitation"),
        "fields": {
            key: value
            for key, value in citation.items()
            if key not in {"type", "kind"} and _has_citation_field_value(value)
        },
        "resolves_to": _optional_str(node_input.get("resolves_to")),
        "retrieval": None,
        "assessment": None,
    }


def _citation_fields(citation: object) -> dict[str, object]:
    return {key: value for key, value in asdict(citation).items() if _has_citation_field_value(value)}


def _has_citation_field_value(value: object) -> bool:
    return value is not None and value not in ("", ())


def _single_retrieval_payload(item: CitationRetrieval | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return _retrieval_item_payload(item)


def _retrieval_payload(retrieval: RetrievedDocument | None) -> dict[str, Any] | None:
    if retrieval is None:
        return None
    return {
        "retrievals": [_retrieval_item_payload(item) for item in retrieval.retrievals],
        "counts": {
            "total": len(retrieval.retrievals),
            "found": len(retrieval.found),
        },
    }


def _retrieval_from_review_payload(
    payload: dict[str, object],
    extraction: ExtractedDocument,
) -> RetrievedDocument:
    retrieval_by_id = {
        item.citation_id: item
        for item in (
            _citation_retrieval_from_review_item(citation) for citation in _review_citations(payload)
        )
        if item is not None
    }
    return RetrievedDocument(
        source_metadata=extraction.source_metadata,
        text=extraction.text,
        preprocessing_metadata=extraction.preprocessing_metadata,
        citations=extraction.citations,
        extraction_metadata=extraction.extraction_metadata,
        retrievals=tuple(
            retrieval_by_id[item.citation_id]
            for item in extraction.citations
            if item.citation_id in retrieval_by_id
        ),
        retrieval_metadata=RetrievalMetadata(client_mode="custom", source="review_payload"),
        jurisdictions=tuple(
            Jurisdiction(
                reporter_inference=evaluate_reporter_inference(item.citation.reporter) if hasattr(item.citation, "reporter") else ReporterInference(reporter=None, status=ReporterInferenceStatus.MISSING_REPORTER, mlz_jurisdictions=()),
                court_inference=evaluate_court_inference(item.citation.court) if hasattr(item.citation, "court") else CourtInference(extracted_court=None, status=CourtInferenceStatus.MISSING_COURT, courts_db_classification=None),
            )
            for item in extraction.citations
        ),
    )


def _retrieval_item_payload(item: CitationRetrieval) -> dict[str, Any]:
    payload = dict(serialize_citation_retrieval(item))
    payload["case_names"] = list(item.case_names)
    return payload


def _citation_retrieval_from_review_item(item: JsonDict) -> CitationRetrieval | None:
    retrieval = item.get("retrieval")
    if not isinstance(retrieval, dict):
        return None
    retrieval_payload = dict(retrieval)
    retrieval_payload.pop("case_names", None)
    retrieval_payload["citation_id"] = str(retrieval.get("citation_id") or item.get("id") or "")
    return deserialize_citation_retrieval(retrieval_payload)


def _extraction_from_review_payload(payload: dict[str, object]) -> ExtractedDocument:
    text = _review_document_text(payload)
    citations = tuple(_extracted_citation_from_review_item(item) for item in _review_citations(payload))
    return ExtractedDocument(
        source_metadata=SourceMetadata(
            path=_review_source_path(payload),
            format=_review_source_format(payload),
        ),
        text=text,
        preprocessing_metadata=PreprocessingMetadata(
            backend=_review_preprocessing_backend(payload),
        ),
        citations=citations,
        extraction_metadata=ExtractionMetadata(),
    )


def _extracted_citation_from_review_item(item: JsonDict) -> ExtractedCitation:
    fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
    reporter: Reporter | None = None
    raw_reporter = fields.get("reporter")
    if isinstance(raw_reporter, dict):
        reporter = Reporter(**raw_reporter)
    elif isinstance(raw_reporter, str):
        reporter = Reporter(
            edition=raw_reporter,
            short_name=raw_reporter,
            name="",
            cite_type="",
            is_scotus=False,
            source="",
        )
    citation = (
        FullCaseCitation(
            plaintiff=_optional_str(fields.get("plaintiff")),
            defendant=_optional_str(fields.get("defendant")),
            volume=_optional_str(fields.get("volume")),
            reporter=reporter,
            page=_optional_str(fields.get("page")),
            pin_cite=_optional_str(fields.get("pin_cite")),
            extra=_optional_str(fields.get("extra")),
            year=_optional_str(fields.get("year")),
            court=_optional_str(fields.get("court")),
            parenthetical=_optional_str(fields.get("parenthetical")),
        )
        if item.get("kind") == "FullCaseCitation"
        else UnknownCitation()
    )
    return ExtractedCitation(
        citation_id=str(item.get("id") or ""),
        span=Span(start=_int_field(item.get("start")), end=_int_field(item.get("end"))),
        matched_text=str(item.get("matched_text") or ""),
        citation=citation,
        resolves_to=_optional_str(item.get("resolves_to")),
    )


def _assessment_payload(item: CitationAssessment) -> JsonDict:
    return dict(serialize_citation_assessment(item))


def _assessment_payload_document(assessment: AssessedDocument) -> JsonDict:
    return {
        "assessments": [_assessment_payload(item) for item in assessment.assessments],
        "assessment_complete": assessment.assessment_complete,
        "status_counts": _assessment_status_counts(assessment.assessments),
        "case_name_followup_status_counts": _case_name_followup_status_counts(assessment.assessments),
        "case_name_counts": _assessment_case_name_counts(assessment.assessments),
        "court_counts": _assessment_court_counts(assessment.assessments),
        "year_counts": _assessment_year_counts(assessment.assessments),
    }


def _assessment_case_name_counts(assessments: tuple[CitationAssessment, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in assessments:
        if isinstance(item, AssessedCitationAssessment):
            status = item.result.case_name.initial.status.value
            counts[status] = counts.get(status, 0) + 1
    return counts


def _assessment_court_counts(assessments: tuple[CitationAssessment, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in assessments:
        if isinstance(item, AssessedCitationAssessment):
            status = item.result.court.status.value
            counts[status] = counts.get(status, 0) + 1
    return counts


def _assessment_year_counts(assessments: tuple[CitationAssessment, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in assessments:
        if isinstance(item, AssessedCitationAssessment):
            status = item.result.year.status.value
            counts[status] = counts.get(status, 0) + 1
    return counts


def _assessment_status_counts(assessments: tuple[CitationAssessment, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in assessments:
        counts[item.status.value] = counts.get(item.status.value, 0) + 1
    return counts


def _case_name_followup_status_counts(
    assessments: tuple[CitationAssessment, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in assessments:
        if isinstance(item, AssessedCitationAssessment):
            status = item.result.case_name.followup.status.value
            counts[status] = counts.get(status, 0) + 1
    return counts


def _copy_review_payload(payload: dict[str, object]) -> JsonDict:
    return {
        "document": dict(payload.get("document")) if isinstance(payload.get("document"), dict) else {},
        "citations": _review_citations(payload),
        "jurisdictions": payload.get("jurisdictions"),
        "retrieval": payload.get("retrieval"),
        "assessment": payload.get("assessment"),
        "stats": dict(payload.get("stats")) if isinstance(payload.get("stats"), dict) else {},
    }


def _review_citations(payload: dict[str, object]) -> list[JsonDict]:
    citations = payload.get("citations")
    if not isinstance(citations, list):
        return []
    return [dict(item) for item in citations if isinstance(item, dict)]


def _review_document_text(payload: dict[str, object]) -> str:
    document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    text = document.get("text")
    if not isinstance(text, str) or not text:
        msg = "review payload is missing document.text"
        raise ValueError(msg)
    return text


def _review_source_path(payload: dict[str, object]) -> str | None:
    document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    return _optional_str(document.get("source_path"))


def _review_source_format(payload: dict[str, object]) -> SourceFormat:
    document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    try:
        return SourceFormat(str(document.get("source_format") or SourceFormat.UNKNOWN.value))
    except ValueError:
        return SourceFormat.UNKNOWN


def _review_preprocessing_backend(payload: dict[str, object]) -> PreprocessingBackend:
    document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    try:
        return PreprocessingBackend(str(document.get("backend") or PreprocessingBackend.PLAIN_TEXT.value))
    except ValueError:
        return PreprocessingBackend.PLAIN_TEXT


def _merge_stats(existing: object, updates: dict[str, int]) -> dict[str, int]:
    stats = dict(existing) if isinstance(existing, dict) else {}
    for key, value in updates.items():
        stats[key] = value
    return stats


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return _int_field(value)
    except (TypeError, ValueError):
        return None


def _list_field(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _int_field(value: object) -> int:
    if isinstance(value, int):
        return value
    return int(str(value))


def _stats(
    extraction: ExtractedDocument,
    retrieval: RetrievedDocument | None,
) -> dict[str, int]:
    label_count = len(extraction.citations)
    stats = {
        "chars": len(extraction.text),
        "citation_spans": label_count,
        "full_citations": len(extraction.full_citations),
    }
    if retrieval is not None:
        stats["retrieved"] = sum(
            1 for item in retrieval.retrievals if item.status != RetrievalStatus.SKIPPED
        )
        stats["found"] = len(retrieval.found)
    return stats
