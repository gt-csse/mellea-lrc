"""GovInfo body search saves only excerpts from dated, individual opinion PDFs."""

from __future__ import annotations

import json
from datetime import date

import httpx

from mellea_lrc.model.citations.body_evidence import BodySource
from mellea_lrc.model.citations.full_reporter import FullReporterCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.preprocessed_document import PreprocessingMetadata
from mellea_lrc.model.source import SourceMetadata
from mellea_lrc.model.span import Span
from mellea_lrc.providers.govinfo import GovInfoClient, GovInfoConfig
from mellea_lrc.validation.body_search.locator_body_govinfo_opinion_retrieval import (
    SUBSTAGE,
    locator_body_govinfo_opinion_retrieval,
)


def _document() -> Document:
    source = "Acme v. Smith, 30 F.3d 100 (2d Cir. 1994)."
    start = source.index("30 F.3d 100")
    root = FullReporterCitation.from_locator(
        citation_id="root:1", substage="01_extract", source=source, span=Span(start, start + 11)
    )
    document = Document(
        source_metadata=SourceMetadata(),
        text=source,
        preprocessing_metadata=PreprocessingMetadata(),
        citations=(root,),
    ).complete_substage("01_extract")
    document = document.replace_citation(root.record("02_roots").with_root(root.id))
    return document.complete_substage("02_roots")


def _document_with_case_name() -> Document:
    document = _document()
    source = document.text
    start = source.index("Acme v. Smith")
    recorded = (
        document.roots[0]
        .record("03_case_name")
        .with_case_name(source, Span(start, start + len("Acme v. Smith")))
    )
    return document.replace_citation(recorded).complete_substage("03_case_name")


def _pdf(text: str) -> bytes:
    """Create a minimal one-page PDF with an ordinary extractable text stream."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET\n".encode()
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream",
    )
    result = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(result)
    result.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode())
    result.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return bytes(result)


def _client(respond: httpx.MockTransport) -> GovInfoClient:
    return GovInfoClient(
        GovInfoConfig(base_url="https://api.govinfo.gov/", timeout_seconds=45, api_key="test-secret"),
        session=httpx.Client(transport=respond),
    )


def test_govinfo_stage_fetches_individual_granule_pdf_and_uses_granule_date() -> None:
    assert SUBSTAGE == "validate_roots.locator_body_corroboration.govinfo_opinion_retrieval"
    requests: list[httpx.Request] = []
    old_id = "USCOURTS-nyd-1_20-cv-1"
    new_id = "USCOURTS-nyd-1_20-cv-2"
    old_granule = "opinion-old"
    new_granule = "opinion-new"
    pdf_url = f"https://api.govinfo.gov/packages/{old_id}/granules/{old_granule}/pdf"

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/search":
            assert request.method == "POST"
            assert json.loads(request.content)["resultLevel"] == "default"
            return httpx.Response(
                200,
                json={
                    "count": 2,
                    "results": [
                        {
                            "packageId": new_id,
                            "granuleId": new_granule,
                            "dateIssued": "2020-01-01",
                            "title": "new package",
                        },
                        {
                            "packageId": old_id,
                            "granuleId": old_granule,
                            "dateIssued": "2030-01-01",
                            "title": "old package",
                        },
                    ],
                    "otherMetadata": {"keep": True},
                },
            )
        if path == f"/packages/{new_id}/granules/{new_granule}/summary":
            return httpx.Response(
                200,
                json={
                    "packageId": new_id,
                    "granuleId": new_granule,
                    "dateIssued": "2026-01-01",
                    "download": {
                        "pdfLink": f"https://api.govinfo.gov/packages/{new_id}/granules/{new_granule}/pdf"
                    },
                },
            )
        if path == f"/packages/{old_id}/granules/{old_granule}/summary":
            return httpx.Response(
                200,
                json={
                    "packageId": old_id,
                    "granuleId": old_granule,
                    "dateIssued": "2024-06-01",
                    "download": {"pdfLink": pdf_url},
                    "otherMetadata": ["keep"],
                },
            )
        if path == f"/packages/{old_id}/granules/{old_granule}/pdf":
            return httpx.Response(200, content=_pdf("A later court cites 30 F.3d 100 in a different case."))
        raise AssertionError(f"Unexpected request: {request.url}")

    after = locator_body_govinfo_opinion_retrieval(
        _document(), client=_client(httpx.MockTransport(respond)), retrospective_date=date(2025, 1, 1)
    )
    record = after.roots[0].body_searches[0]

    assert after.substage_runs[-1] == SUBSTAGE
    assert record.source is BodySource.GOVINFO_OPINION
    assert record.retrospective_date == date(2025, 1, 1)
    assert record.attempts[0].pages[0]["otherMetadata"] == {"keep": True}
    assert record.discovery_pages == ()
    assert len(record.evidence) == 1
    evidence = record.evidence[0]
    assert evidence.body_id == old_granule
    assert evidence.parent_id == old_id
    assert evidence.url == pdf_url
    assert evidence.issued_on == date(2024, 6, 1)
    assert evidence.date_basis == "granule.dateIssued"
    assert evidence.anchor_kind == "locator"
    assert evidence.excerpt[evidence.anchor_span.start : evidence.anchor_span.end] == "30 F.3d 100"
    assert evidence.metadata["granule_summary"]["otherMetadata"] == ["keep"]
    assert [(item.failure_type, item.item_id) for item in record.failures] == [
        ("ineligible_date", new_granule)
    ]
    assert sum(request.url.path.endswith("/pdf") for request in requests) == 1
    assert requests[-1].url.params["api_key"] == "test-secret"
    assert not any(request.url.path.endswith("/granules") for request in requests)


def test_govinfo_stage_records_scanned_and_no_locator_pdfs_without_evidence() -> None:
    package = "USCOURTS-nyd-1_20-cv-1"
    granules = ("scanned", "unrelated")

    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/search":
            return httpx.Response(
                200,
                json={
                    "count": 2,
                    "results": [{"packageId": package, "granuleId": item} for item in granules],
                },
            )
        if path.endswith("/summary"):
            granule_id = path.split("/")[-2]
            return httpx.Response(
                200,
                json={
                    "packageId": package,
                    "granuleId": granule_id,
                    "dateIssued": "2024-01-01",
                    "download": {
                        "pdfLink": f"https://api.govinfo.gov/packages/{package}/granules/{granule_id}/pdf"
                    },
                },
            )
        if path.endswith("/scanned/pdf"):
            return httpx.Response(200, content=_pdf(""))
        if path.endswith("/unrelated/pdf"):
            return httpx.Response(200, content=_pdf("An unrelated opinion with no reported citation."))
        raise AssertionError(path)

    after = locator_body_govinfo_opinion_retrieval(_document(), client=_client(httpx.MockTransport(respond)))
    record = after.roots[0].body_searches[0]

    assert record.evidence == ()
    assert record.discovery_pages == ()
    assert {(failure.failure_type, failure.item_id) for failure in record.failures} == {
        ("no_extractable_text", "scanned"),
        ("no_locator", "unrelated"),
    }


def test_govinfo_stage_keeps_search_failure_in_atomic_checkpoint() -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"message": "test-secret temporarily unavailable"})

    after = locator_body_govinfo_opinion_retrieval(_document(), client=_client(httpx.MockTransport(respond)))
    record = after.roots[0].body_searches[0]

    assert after.substage_runs[-1] == SUBSTAGE
    assert len(record.attempts) == 1
    assert record.attempts[0].pages == ()
    assert record.attempts[0].failure.failure_type == "http_error"
    assert "test-secret" not in record.attempts[0].failure.message
    assert record.evidence == ()


def test_govinfo_locator_query_does_not_promote_name_only_body_to_evidence() -> None:
    package = "USCOURTS-nyd-1_20-cv-1"
    granule = "opinion-other"
    queries: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/search":
            query = json.loads(request.content)["query"]
            queries.append(query)
            return httpx.Response(
                200,
                json={
                    "count": 1,
                    "results": [{"packageId": package, "granuleId": granule}],
                },
            )
        if path.endswith("/summary"):
            return httpx.Response(
                200,
                json={
                    "packageId": package,
                    "granuleId": granule,
                    "dateIssued": "2024-01-01",
                    "download": {
                        "pdfLink": f"https://api.govinfo.gov/packages/{package}/granules/{granule}/pdf"
                    },
                },
            )
        if path.endswith("/pdf"):
            return httpx.Response(
                200, content=_pdf("This later decision discusses Acme v. Smith and distinguishes it.")
            )
        raise AssertionError(path)

    after = locator_body_govinfo_opinion_retrieval(
        _document_with_case_name(), client=_client(httpx.MockTransport(respond))
    )
    record = after.roots[0].body_searches[0]

    assert queries == ['collection:uscourts and "30 F.3d 100"']
    assert len(record.attempts) == 1
    assert record.evidence == ()
    assert [(failure.failure_type, failure.item_id) for failure in record.failures] == [
        ("no_locator", granule)
    ]


def test_govinfo_stage_paginates_granules_and_keeps_summary_failure() -> None:
    package = "USCOURTS-nyd-1_20-cv-1"
    marks: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/search":
            return httpx.Response(200, json={"count": 1, "results": [{"packageId": package}]})
        if path.endswith("/granules"):
            mark = request.url.params["offsetMark"]
            marks.append(mark)
            if mark == "*":
                return httpx.Response(
                    200,
                    json={
                        "count": 2,
                        "granules": [{"granuleId": "failed"}],
                        "nextPage": f"https://api.govinfo.gov/packages/{package}/granules?offsetMark=second%2Bpage",
                        "pageMetadata": "first",
                    },
                )
            assert mark == "second+page"
            return httpx.Response(
                200, json={"count": 2, "granules": [{"granuleId": "available"}], "pageMetadata": "second"}
            )
        if path.endswith("/failed/summary"):
            return httpx.Response(503, json={"message": "temporarily unavailable"})
        if path.endswith("/available/summary"):
            return httpx.Response(
                200,
                json={
                    "packageId": package,
                    "granuleId": "available",
                    "dateIssued": "2024-01-01",
                    "download": {
                        "pdfLink": f"https://api.govinfo.gov/packages/{package}/granules/available/pdf"
                    },
                },
            )
        if path.endswith("/available/pdf"):
            return httpx.Response(200, content=_pdf("A later court cited 30 F.3d 100."))
        raise AssertionError(path)

    after = locator_body_govinfo_opinion_retrieval(_document(), client=_client(httpx.MockTransport(respond)))
    record = after.roots[0].body_searches[0]

    assert marks == ["*", "second+page"]
    assert [page["pageMetadata"] for page in record.discovery_pages] == ["first", "second"]
    assert record.evidence[0].body_id == "available"
    assert [(item.failure_type, item.item_id, item.status_code) for item in record.failures] == [
        ("http_error", "failed", 503)
    ]


def test_govinfo_stage_caps_pdf_fetches_for_large_default_result_sets() -> None:
    package = "USCOURTS-nyd-1_20-cv-1"
    pdf_requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal pdf_requests
        path = request.url.path
        if path == "/search":
            assert json.loads(request.content)["resultLevel"] == "default"
            return httpx.Response(
                200,
                json={
                    "count": 100_000,
                    "results": [
                        {"packageId": package, "granuleId": f"opinion-{index}"} for index in range(10)
                    ],
                },
            )
        if path.endswith("/summary"):
            granule = path.split("/")[-2]
            return httpx.Response(
                200,
                json={
                    "packageId": package,
                    "granuleId": granule,
                    "dateIssued": "2024-01-01",
                    "download": {
                        "pdfLink": f"https://api.govinfo.gov/packages/{package}/granules/{granule}/pdf"
                    },
                },
            )
        if path.endswith("/pdf"):
            pdf_requests += 1
            return httpx.Response(200, content=_pdf("This document has no matching citation."))
        raise AssertionError(path)

    after = locator_body_govinfo_opinion_retrieval(_document(), client=_client(httpx.MockTransport(respond)))
    record = after.roots[0].body_searches[0]

    assert pdf_requests == 8
    assert len(record.attempts[0].pages) == 1
    assert record.evidence == ()
    assert any(failure.failure_type == "fetch_limit" for failure in record.failures)


def test_govinfo_direct_search_date_cannot_replace_granule_issue_date() -> None:
    package = "USCOURTS-nyd-1_20-cv-1"
    granule = "opinion-1"

    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/search":
            return httpx.Response(
                200,
                json={
                    "count": 1,
                    "results": [{"packageId": package, "granuleId": granule, "dateIssued": "2024-01-01"}],
                },
            )
        if path.endswith("/summary"):
            return httpx.Response(
                200,
                json={
                    "packageId": package,
                    "granuleId": granule,
                    "download": {
                        "pdfLink": f"https://api.govinfo.gov/packages/{package}/granules/{granule}/pdf"
                    },
                },
            )
        raise AssertionError("Undated granule must not be downloaded")

    after = locator_body_govinfo_opinion_retrieval(
        _document(), client=_client(httpx.MockTransport(respond)), retrospective_date=date(2025, 1, 1)
    )
    record = after.roots[0].body_searches[0]

    assert record.evidence == ()
    assert [(failure.failure_type, failure.item_id) for failure in record.failures] == [
        ("ineligible_date", granule)
    ]
