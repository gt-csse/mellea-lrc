"""Find independent USCOURTS opinion citations in individual GovInfo granules."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import date
from io import BytesIO
from typing import Literal, Protocol

from pypdf import PdfReader

from mellea_lrc.govinfo import GovInfoClient, GovInfoError, GovInfoGranulesPage, GovInfoSearchPage
from mellea_lrc.model.citations import FullCitationVariant
from mellea_lrc.model.citations.body_evidence import (
    BodyEvidence,
    BodyEvidenceFailure,
    BodySearch,
    BodySearchAttempt,
    BodySource,
)
from mellea_lrc.model.citations.field_body_evidence import FieldBodySearch
from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search.common import (
    eligible_on,
    evidence_date,
    locator_text,
    make_body_evidences,
    roots_for_body_search,
)

STAGE = "22_locator_body_govinfo_opinion_retrieval"
SEARCH_PAGE_SIZE = 25
MAX_SEARCH_PAGES = 2
MAX_FALLBACK_PACKAGES = 2
MAX_CANDIDATES = 24
GRANULE_PAGE_SIZE = 100
MAX_GRANULE_PAGES = 5
MAX_GRANULES = 40
MAX_PDF_DOWNLOADS = 8
MAX_EVIDENCE = 12


class GovInfoBodyClient(Protocol):
    """The four GovInfo calls needed for granule-level body evidence."""

    def search(
        self,
        query: str,
        *,
        offset_mark: str = "*",
        page_size: int = 100,
        result_level: Literal["package", "default"] = "package",
    ) -> GovInfoSearchPage: ...

    def list_granules(
        self, package_id: str, *, offset_mark: str = "*", page_size: int = 100
    ) -> GovInfoGranulesPage: ...

    def get_granule_summary(self, package_id: str, granule_id: str) -> dict: ...

    def download_pdf(self, pdf_url: str) -> bytes: ...


def _failure(error: GovInfoError, *, item_id: str | None = None) -> BodyEvidenceFailure:
    return BodyEvidenceFailure(
        failure_type=error.failure_type,
        message=str(error) or type(error).__name__,
        item_id=item_id,
        status_code=error.upstream_status_code,
    )


def _problem(failure_type: str, message: str, *, item_id: str | None = None) -> BodyEvidenceFailure:
    return BodyEvidenceFailure(failure_type=failure_type, message=message, item_id=item_id)


def _query(phrase: str) -> str:
    escaped = phrase.replace("\\", "\\\\").replace('"', '\\"')
    return f'collection:uscourts and "{escaped}"'


def _search_results(service: GovInfoBodyClient, query: str) -> tuple[BodySearchAttempt, list[dict]]:
    pages: list[dict] = []
    results: list[dict] = []
    failure: BodyEvidenceFailure | None = None
    mark = "*"
    seen_marks = {mark}
    for _ in range(MAX_SEARCH_PAGES):
        try:
            page = service.search(query, offset_mark=mark, page_size=SEARCH_PAGE_SIZE, result_level="default")
        except GovInfoError as error:
            failure = _failure(error)
            break
        pages.append(page.raw_json)
        for result in page.results:
            package_id = result.get("packageId")
            if isinstance(package_id, str) and package_id.startswith("USCOURTS-"):
                results.append(result)
        next_mark = page.next_offset_mark
        if next_mark is None:
            break
        if next_mark in seen_marks:
            failure = _problem("repeated_offset_mark", "GovInfo search repeated an offset mark")
            break
        seen_marks.add(next_mark)
        mark = next_mark
    else:
        failure = _problem("page_limit", "GovInfo search reached its page limit")
    return BodySearchAttempt(query=query, pages=tuple(pages), failure=failure), results


def _granules(
    service: GovInfoBodyClient, package_id: str
) -> tuple[list[dict], list[dict], list[BodyEvidenceFailure]]:
    pages: list[dict] = []
    found: list[dict] = []
    failures: list[BodyEvidenceFailure] = []
    seen_ids: set[str] = set()
    mark = "*"
    seen_marks = {mark}
    for _ in range(MAX_GRANULE_PAGES):
        try:
            page = service.list_granules(package_id, offset_mark=mark, page_size=GRANULE_PAGE_SIZE)
        except GovInfoError as error:
            failures.append(_failure(error, item_id=package_id))
            break
        except ValueError:
            failures.append(
                _problem("invalid_package_id", "GovInfo package ID is invalid", item_id=package_id)
            )
            break
        pages.append(page.raw_json)
        for index, granule in enumerate(page.granules):
            granule_id = granule.get("granuleId")
            if not isinstance(granule_id, str) or not granule_id:
                failures.append(
                    _problem("missing_granule_id", "GovInfo granule has no ID", item_id=package_id)
                )
                continue
            if granule_id in seen_ids:
                continue
            seen_ids.add(granule_id)
            found.append(granule)
            if len(found) >= MAX_GRANULES:
                if (
                    index < len(page.granules) - 1
                    or page.next_offset_mark is not None
                    or page.count > len(found)
                ):
                    failures.append(
                        _problem(
                            "granule_limit", "GovInfo package reached its granule limit", item_id=package_id
                        )
                    )
                return pages, found, failures
        next_mark = page.next_offset_mark
        if next_mark is None:
            break
        if next_mark in seen_marks:
            failures.append(
                _problem(
                    "repeated_offset_mark", "GovInfo granules repeated an offset mark", item_id=package_id
                )
            )
            break
        seen_marks.add(next_mark)
        mark = next_mark
    else:
        failures.append(_problem("page_limit", "GovInfo granules reached a page limit", item_id=package_id))
    return pages, found, failures


def _granule_issue_date(
    summary: dict, entry: dict, entry_date_basis: str | None
) -> tuple[date | None, str | None]:
    dates = [
        ("granule.granuleDate", summary.get("granuleDate")),
        ("granule.dateIssued", summary.get("dateIssued")),
    ]
    if entry_date_basis is not None:
        dates.append((entry_date_basis, entry.get("dateIssued")))
    for source, raw in dates:
        issued_on = evidence_date(raw)
        if issued_on is not None:
            return issued_on, source
    return None, None


def _pdf_text(data: bytes) -> str:
    reader = PdfReader(BytesIO(data), strict=False)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _fetch_granule(
    service: GovInfoBodyClient,
    *,
    package_id: str,
    search_result: dict,
    entry: dict,
    entry_date_basis: str | None,
    anchor_text: str,
    anchor_kind: Literal["locator", "case_name"],
    source_text: str,
    retrospective_date: date | None,
) -> tuple[tuple[BodyEvidence, ...], BodyEvidenceFailure | None, bool]:
    granule_id = entry["granuleId"]
    try:
        summary = service.get_granule_summary(package_id, granule_id)
    except GovInfoError as error:
        return (), _failure(error, item_id=granule_id), False
    except ValueError:
        return (), _problem("invalid_granule_id", "GovInfo granule ID is invalid", item_id=granule_id), False

    issued_on, date_basis = _granule_issue_date(summary, entry, entry_date_basis)
    if not eligible_on(issued_on, retrospective_date):
        reason = (
            "GovInfo granule has no precise issue date"
            if issued_on is None
            else "GovInfo granule postdates the retrospective cutoff"
        )
        return (), _problem("ineligible_date", reason, item_id=granule_id), False

    download = summary.get("download")
    pdf_url = download.get("pdfLink") if isinstance(download, dict) else None
    if not isinstance(pdf_url, str) or not pdf_url:
        return (), _problem("missing_pdf_link", "GovInfo granule has no PDF link", item_id=granule_id), False
    try:
        pdf = service.download_pdf(pdf_url)
    except GovInfoError as error:
        return (), _failure(error, item_id=granule_id), True
    except ValueError:
        return (
            (),
            _problem("invalid_pdf_link", "GovInfo granule has an invalid PDF link", item_id=granule_id),
            True,
        )
    try:
        body_text = _pdf_text(pdf)
    except Exception:
        return (
            (),
            _problem(
                "pdf_extraction_error", "Could not extract text from GovInfo granule PDF", item_id=granule_id
            ),
            True,
        )
    if not body_text.strip():
        return (
            (),
            _problem(
                "no_extractable_text", "GovInfo granule PDF has no extractable text", item_id=granule_id
            ),
            True,
        )
    evidence = make_body_evidences(
        body_id=granule_id,
        parent_id=package_id,
        url=pdf_url,
        issued_on=issued_on,
        date_basis=date_basis,
        metadata={
            "search_result": search_result,
            "granule_entry": entry,
            "granule_summary": summary,
        },
        body_text=body_text,
        locator=anchor_text,
        anchor_kind=anchor_kind,
        source_text=source_text,
    )
    if not evidence:
        missing_anchor = "locator" if anchor_kind == "locator" else "case_name"
        return (
            (),
            _problem(
                f"no_{missing_anchor}",
                f"Fetched GovInfo granule does not contain the cited {missing_anchor.replace('_', ' ')}",
                item_id=granule_id,
            ),
            True,
        )
    return evidence, None, True


def _search_root(
    document: Document,
    root: FullCitationVariant,
    service: GovInfoBodyClient,
    retrospective_date: date | None,
    *,
    query_text: str | None = None,
    query_parties: tuple[str, str] | None = None,
    anchor_kind: Literal["locator", "case_name"] = "locator",
) -> BodySearch | FieldBodySearch:
    anchor_text = locator_text(root) if query_text is None else query_text
    attempts: list[BodySearchAttempt] = []
    discovery_pages: list[dict] = []
    evidence: list[BodyEvidence] = []
    failures: list[BodyEvidenceFailure] = []
    seen_granules: set[tuple[str, str]] = set()
    fallback_packages: set[str] = set()
    pdf_downloads = 0
    limit_reached = False
    if anchor_text:
        query = (
            f"collection:uscourts and {query_parties[0]} and {query_parties[1]}"
            if query_parties is not None
            else _query(anchor_text)
        )
        attempt, results = _search_results(service, query)
        attempts.append(attempt)
        candidate_limit = MAX_CANDIDATES // 2 if query_parties is not None else MAX_CANDIDATES
        fetch_limit = MAX_PDF_DOWNLOADS // 2 if query_parties is not None else MAX_PDF_DOWNLOADS
        for search_result in results:
            package_id = search_result["packageId"]
            granule_id = search_result.get("granuleId")
            if isinstance(granule_id, str) and granule_id:
                granules = [search_result]
                entry_date_basis = None
            else:
                if package_id in fallback_packages:
                    continue
                if len(fallback_packages) >= MAX_FALLBACK_PACKAGES:
                    failures.append(
                        _problem("package_limit", "GovInfo granule listing reached its package limit")
                    )
                    continue
                fallback_packages.add(package_id)
                pages, granules, granule_failures = _granules(service, package_id)
                entry_date_basis = "granule_list.dateIssued"
                discovery_pages.extend(pages)
                failures.extend(granule_failures)
                if not granules and not granule_failures:
                    failures.append(
                        _problem("no_granules", "GovInfo package has no granules", item_id=package_id)
                    )
            for entry in granules:
                key = (package_id, entry["granuleId"])
                if key in seen_granules:
                    continue
                if len(seen_granules) >= candidate_limit:
                    failures.append(_problem("candidate_limit", "GovInfo search reached its granule limit"))
                    limit_reached = True
                    break
                if pdf_downloads >= fetch_limit or len(evidence) >= MAX_EVIDENCE:
                    failures.append(_problem("fetch_limit", "GovInfo opinion fetch reached its limit"))
                    limit_reached = True
                    break
                seen_granules.add(key)
                excerpts, failure, downloaded = _fetch_granule(
                    service,
                    package_id=package_id,
                    search_result=search_result,
                    entry=entry,
                    entry_date_basis=entry_date_basis,
                    anchor_text=anchor_text,
                    anchor_kind=anchor_kind,
                    source_text=document.text,
                    retrospective_date=retrospective_date,
                )
                pdf_downloads += int(downloaded)
                evidence.extend(excerpts[: MAX_EVIDENCE - len(evidence)])
                if failure is not None:
                    failures.append(failure)
            if limit_reached:
                break
    search_type = BodySearch if anchor_kind == "locator" else FieldBodySearch
    return search_type(
        node_id=root.nodes[-1].id,
        source=BodySource.GOVINFO_OPINION,
        retrospective_date=retrospective_date,
        **({"query_name": anchor_text} if anchor_kind == "case_name" else {}),
        attempts=tuple(attempts),
        discovery_pages=tuple(discovery_pages),
        evidence=tuple(evidence),
        failures=tuple(failures),
    )


def locator_body_govinfo_opinion_retrieval(
    document: Document,
    *,
    client: GovInfoBodyClient | None = None,
    retrospective_date: date | None = None,
) -> Document:
    """Save locator matches from fetched USCOURTS opinion granules."""
    if STAGE in document.stage_runs:
        raise ValueError("GovInfo opinion body search has already completed")
    with ExitStack() as stack:
        service = client
        for root in roots_for_body_search(document):
            recorded = root.record(STAGE)
            if service is None:
                service = stack.enter_context(GovInfoClient())
            result = _search_root(document, recorded, service, retrospective_date)
            if not isinstance(result, BodySearch):
                raise ValueError("A locator search must return locator evidence")
            document = document.replace_citation(recorded.with_body_search(result))
    return document.complete(STAGE)
