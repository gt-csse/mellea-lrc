"""Small, typed client for GovInfo's U.S. Courts Opinions search service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from types import MappingProxyType
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

import requests

DEFAULT_BASE_URL = "https://api.govinfo.gov/"
DEFAULT_API_KEY = "DEMO_KEY"
DEFAULT_USER_AGENT = "mellea-lrc (+https://github.com/gt-csse/mellea-lrc)"
_PACKAGE_TEXT_MAX_GRANULES = 5
_PACKAGE_TEXT_MAX_PAGES_PER_GRANULE = 32
_PACKAGE_TEXT_MAX_PDF_BYTES = 10_000_000


@dataclass(frozen=True, slots=True)
class GovInfoConfig:
    """Configuration for the GovInfo API Search Service."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY

    @classmethod
    def from_env(cls) -> GovInfoConfig:
        """Load the optional GovInfo API key, using its public demo key by default."""
        return cls(
            base_url=os.getenv("GOVINFO_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.getenv("GOVINFO_API_KEY", "").strip() or DEFAULT_API_KEY,
        )


class GovInfoError(RuntimeError):
    """A structured failure returned while querying the GovInfo API."""

    def __init__(
        self,
        message: str,
        *,
        failure_type: str,
        upstream_status_code: int | None = None,
        retryable: bool = False,
        url: str | None = None,
        upstream_detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.upstream_status_code = upstream_status_code
        self.retryable = retryable
        self.url = url
        self.upstream_detail = upstream_detail


@dataclass(frozen=True, slots=True)
class GovInfoSearchResult:
    """A preserved page of USCOURTS package results."""

    query: str
    count: int
    results: tuple[dict[str, object], ...]
    next_offset_mark: str | None


class GovInfoClient:
    """Search Federal opinions by the case number GovInfo indexes."""

    def __init__(
        self,
        config: GovInfoConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or GovInfoConfig.from_env()
        self.session = session or requests.Session()

    def search_uscourts_docket(
        self,
        docket_number: str,
        *,
        court_id: str | None,
        page_size: int,
    ) -> GovInfoSearchResult:
        """Return USCOURTS packages matching the stated docket without alteration.

        This is intentionally an archive lookup, not another docket grammar:
        the query retains the filing's literal docket text and uses an extracted
        court only as an optional narrowing condition.
        """
        query = govinfo_uscourts_docket_query(docket_number, court_id=court_id)
        return self.search_uscourts(query, page_size=page_size)

    def search_uscourts(self, query: str, *, page_size: int) -> GovInfoSearchResult:
        """Search USCOURTS packages with a caller-supplied, auditable query.

        Higher-level stages build the small, fixed family of query forms from
        source-grounded fields; this client only carries the exact query to the
        archive. It does not parse or normalize docket numbers.
        """
        url = urljoin(self.config.base_url.rstrip("/") + "/", "search")
        body = {
            "query": query,
            "pageSize": page_size,
            "offsetMark": "*",
            "resultLevel": "package",
            "sorts": [{"field": "score", "sortOrder": "DESC"}],
        }
        try:
            response = self.session.request(
                "POST",
                url,
                params={"api_key": self.config.api_key},
                json=body,
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=45,
            )
        except requests.Timeout as exc:
            raise GovInfoError(
                "GovInfo request timed out",
                failure_type="upstream_timeout",
                retryable=True,
                url=url,
            ) from exc
        except requests.RequestException as exc:
            raise GovInfoError(
                "GovInfo request failed before a response was received",
                failure_type="upstream_request_error",
                retryable=True,
                url=url,
                upstream_detail=str(exc),
            ) from exc
        if response.status_code >= 400:
            raise GovInfoError(
                f"GovInfo returned HTTP {response.status_code}",
                failure_type="upstream_http_error",
                upstream_status_code=response.status_code,
                retryable=response.status_code in {429, 500, 502, 503, 504},
                url=response.url,
                upstream_detail=_response_detail(response),
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise GovInfoError(
                "GovInfo returned invalid JSON",
                failure_type="upstream_invalid_response",
                upstream_status_code=response.status_code,
                retryable=False,
                url=response.url,
                upstream_detail=response.text[:1000],
            ) from exc
        return _search_result(payload, query=query, url=response.url)

    def get_package_text(
        self,
        package_id: str,
        *,
        retrospective_date: date | None = None,
    ) -> str:
        """Read bounded text from a package's official opinion granules.

        USCOURTS usually supplies PDF granules rather than text or HTML. Each
        granule is read separately so one package can contain multiple opinions.
        An absent download or an image-only PDF contributes no text.

        A package can contain opinions issued on different dates. When an
        evaluation cutoff is supplied, only a granule whose own ``dateIssued``
        is a valid date on or before the cutoff may contribute text. An absent
        or invalid date excludes that granule rather than risking later evidence.
        """
        package = quote(package_id, safe="")
        summary = self._get_json(f"packages/{package}/summary")
        if not isinstance(summary, dict):
            raise GovInfoError(
                "GovInfo returned an invalid package summary",
                failure_type="upstream_invalid_response",
                retryable=False,
            )
        granules_link = summary.get("granulesLink")
        if not isinstance(granules_link, str) or not granules_link:
            return ""
        listing = self._get_json_link(
            granules_link,
            params={"offsetMark": "*", "pageSize": _PACKAGE_TEXT_MAX_GRANULES},
        )
        if not isinstance(listing, dict) or not isinstance(listing.get("granules"), list):
            raise GovInfoError(
                "GovInfo returned an invalid granule listing",
                failure_type="upstream_invalid_response",
                retryable=False,
            )
        texts: list[str] = []
        for granule in listing["granules"][:_PACKAGE_TEXT_MAX_GRANULES]:
            if not isinstance(granule, dict):
                continue
            link = granule.get("granuleLink")
            if not isinstance(link, str) or not link:
                continue
            granule_summary = self._get_json_link(link)
            if not isinstance(granule_summary, dict):
                raise GovInfoError(
                    "GovInfo returned an invalid granule summary",
                    failure_type="upstream_invalid_response",
                    retryable=False,
                )
            if retrospective_date is not None and not _granule_issued_by(granule_summary, retrospective_date):
                continue
            download = granule_summary.get("download")
            if not isinstance(download, dict):
                continue
            content = self._download_granule_text(download)
            if content.strip():
                texts.append(content.strip())
        return "\n\n".join(texts)

    def _download_granule_text(self, download: dict[str, object]) -> str:
        for name in ("txtLink", "htmlLink", "htmLink", "pdfLink"):
            link = download.get(name)
            if not isinstance(link, str) or not link:
                continue
            response = self._get_link(link)
            if name == "pdfLink":
                content = _pdf_text(response.content)
            elif name in {"htmlLink", "htmLink"} or "html" in response.headers.get("Content-Type", ""):
                content = _html_text(response.text)
            else:
                content = response.text
            if content.strip():
                return content
        return ""

    def _get_json(self, path: str) -> object:
        url = urljoin(self.config.base_url.rstrip("/") + "/", path)
        response = self._get(url)
        try:
            return response.json()
        except ValueError as exc:
            raise GovInfoError(
                "GovInfo returned invalid JSON",
                failure_type="upstream_invalid_response",
                upstream_status_code=response.status_code,
                retryable=False,
                url=response.url,
                upstream_detail=response.text[:1000],
            ) from exc

    def _get_json_link(self, link: str, *, params: dict[str, object] | None = None) -> object:
        response = self._get(self._linked_url(link), params=params)
        try:
            return response.json()
        except ValueError as exc:
            raise GovInfoError(
                "GovInfo returned invalid JSON",
                failure_type="upstream_invalid_response",
                upstream_status_code=response.status_code,
                retryable=False,
                url=response.url,
                upstream_detail=response.text[:1000],
            ) from exc

    def _get_link(self, link: str) -> requests.Response:
        return self._get(self._linked_url(link))

    def _linked_url(self, link: str) -> str:
        """Keep API links on the configured API base, including proxy bases."""
        parts = urlsplit(link)
        if parts.hostname not in {"api.govinfo.gov", urlsplit(self.config.base_url).hostname}:
            raise GovInfoError(
                "GovInfo returned an unexpected download host",
                failure_type="upstream_invalid_response",
                retryable=False,
            )
        package_path = parts.path.partition("/packages/")
        if not package_path[1]:
            raise GovInfoError(
                "GovInfo returned an unexpected package link",
                failure_type="upstream_invalid_response",
                retryable=False,
            )
        return urljoin(self.config.base_url.rstrip("/") + "/", "packages/" + package_path[2])

    def _get(self, url: str, *, params: dict[str, object] | None = None) -> requests.Response:
        try:
            response = self.session.request(
                "GET",
                url,
                params={"api_key": self.config.api_key, **(params or {})},
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=45,
            )
        except requests.Timeout as exc:
            raise GovInfoError(
                "GovInfo request timed out",
                failure_type="upstream_timeout",
                retryable=True,
                url=url,
            ) from exc
        except requests.RequestException as exc:
            raise GovInfoError(
                "GovInfo request failed before a response was received",
                failure_type="upstream_request_error",
                retryable=True,
                url=url,
                upstream_detail=str(exc),
            ) from exc
        if response.status_code >= 400:
            raise GovInfoError(
                f"GovInfo returned HTTP {response.status_code}",
                failure_type="upstream_http_error",
                upstream_status_code=response.status_code,
                retryable=response.status_code in {429, 500, 502, 503, 504},
                url=response.url,
                upstream_detail=_response_detail(response),
            )
        return response


class _ReadableHtml(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br", "p", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _granule_issued_by(granule_summary: dict[str, object], cutoff: date) -> bool:
    """Only a granule's own valid issue date proves eligibility for text review."""
    raw = granule_summary.get("dateIssued")
    if not isinstance(raw, str) or len(raw) != 10:
        return False
    try:
        issued = date.fromisoformat(raw)
    except ValueError:
        return False
    return issued.isoformat() == raw and issued <= cutoff


def _html_text(content: str) -> str:
    parser = _ReadableHtml()
    parser.feed(content)
    return "".join(parser.parts)


def _pdf_text(content: bytes) -> str:
    if not content:
        return ""
    if len(content) > _PACKAGE_TEXT_MAX_PDF_BYTES:
        raise GovInfoError(
            "GovInfo opinion PDF exceeds the text-review limit",
            failure_type="resource_limit",
            retryable=False,
        )
    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(content)
        try:
            pages: list[str] = []
            for index in range(min(len(document), _PACKAGE_TEXT_MAX_PAGES_PER_GRANULE)):
                page = document[index]
                try:
                    text_page = page.get_textpage()
                    try:
                        pages.append(text_page.get_text_range())
                    finally:
                        text_page.close()
                finally:
                    page.close()
            return "\n\n".join(page for page in pages if page.strip())
        finally:
            document.close()
    except ImportError as exc:
        raise GovInfoError(
            "PDF text extraction is unavailable",
            failure_type="text_extraction_unavailable",
            retryable=False,
        ) from exc
    except Exception as exc:
        raise GovInfoError(
            "GovInfo returned an unreadable opinion PDF",
            failure_type="upstream_invalid_response",
            retryable=False,
        ) from exc


def govinfo_uscourts_case_name_query(terms: tuple[str, ...], *, court_id: str | None) -> str:
    """Build a USCOURTS title query from source-grounded case-name terms."""
    if not terms:
        msg = "GovInfo case-name search needs at least one term"
        raise ValueError(msg)
    clauses = ["collection:uscourts", f"title:({' AND '.join(_query_term(term) for term in terms)})"]
    if court_id:
        clauses.append(f"courtCode:{court_id}")
    return " ".join(clauses)


def govinfo_uscourts_docket_query(docket_number: str, *, court_id: str | None) -> str:
    """Build the documented USCOURTS case-number query from source text."""
    escaped = docket_number.replace("\\", "\\\\").replace('"', '\\"')
    clauses = ["collection:uscourts", f'casenumber:("{escaped}")']
    if court_id:
        clauses.append(f"courtCode:{court_id}")
    return " ".join(clauses)


def govinfo_uscourts_body_query(locator: str) -> str:
    """Build a literal USCOURTS full-index query from text written in a filing.

    Unlike the metadata helpers above, this deliberately names no metadata
    field.  GovInfo evaluates the phrase against its package search index, so a
    returned package is body-corroboration evidence rather than a title or
    case-number lookup.
    """
    escaped = locator.replace("\\", "\\\\").replace('"', '\\"')
    return f'collection:uscourts "{escaped}"'


def _query_term(value: str) -> str:
    escaped = value.strip().replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def govinfo_package_url(package_id: str | None) -> str | None:
    """Return the public details page for a USCOURTS package."""
    return f"https://www.govinfo.gov/app/details/{package_id}" if package_id else None


def govinfo_package_candidate(result: dict[str, object]) -> dict[str, object]:
    """Expose a package record without treating its deposit date as a decision date.

    A USCOURTS package represents a case and can contain more than one opinion.
    Its search-result ``dateIssued`` belongs to the returned package record; it
    does not identify the particular order cited in a filing. Keep that value
    for provenance, but do not map it to the generic ``decisionDate`` field
    used by citation-date validation.
    """
    package_id = _string(result.get("packageId"))
    court_id, docket_number = _package_identity(package_id)
    return {
        **result,
        "govinfo_package_id": package_id,
        "caseName": _string(result.get("title")),
        "court_id": court_id,
        "docketNumber": docket_number,
        "packageDateIssued": _string(result.get("dateIssued")),
    }


def _search_result(payload: object, *, query: str, url: str) -> GovInfoSearchResult:
    if not isinstance(payload, dict):
        raise GovInfoError(
            "GovInfo returned a non-object search response",
            failure_type="upstream_invalid_response",
            retryable=False,
            url=url,
            upstream_detail=payload,
        )
    count = payload.get("count")
    results = payload.get("results")
    if not isinstance(count, int) or count < 0 or not isinstance(results, list):
        raise GovInfoError(
            "GovInfo returned an invalid search response shape",
            failure_type="upstream_invalid_response",
            retryable=False,
            url=url,
            upstream_detail=payload,
        )
    normalized: list[dict[str, object]] = []
    for item in results:
        if not isinstance(item, dict):
            raise GovInfoError(
                "GovInfo returned a non-object search result",
                failure_type="upstream_invalid_response",
                retryable=False,
                url=url,
                upstream_detail=item,
            )
        normalized.append(_freeze_json(item))
    offset_mark = payload.get("offsetMark")
    return GovInfoSearchResult(
        query=query,
        count=count,
        results=tuple(normalized),
        next_offset_mark=offset_mark if isinstance(offset_mark, str) else None,
    )


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _package_identity(package_id: str | None) -> tuple[str | None, str | None]:
    """Read Court code and literal stored case number from a USCOURTS package ID."""
    if package_id is None or not package_id.startswith("USCOURTS-"):
        return None, None
    remainder = package_id.removeprefix("USCOURTS-")
    court_id, separator, encoded_docket = remainder.partition("-")
    if not separator or not court_id or not encoded_docket:
        return None, None
    return court_id, encoded_docket.replace("_", ":")


def _response_detail(response: requests.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return response.text[:1000]


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None
