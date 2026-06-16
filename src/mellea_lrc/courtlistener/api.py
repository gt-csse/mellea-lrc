"""FastAPI app factory for CourtListener access service routes."""

from collections.abc import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from mellea_lrc.courtlistener.cache import R2Cache
from mellea_lrc.courtlistener.client import (
    CourtListenerClient,
    CourtListenerError,
    CourtListenerRateLimiter,
)
from mellea_lrc.courtlistener.lookup import citation_lookup_envelope_dict

HTTP_NOT_FOUND = 404
HTTP_METHOD_NOT_ALLOWED = 405


def create_api(client_factory: Callable[[], CourtListenerClient] | None = None) -> FastAPI:  # noqa: C901
    """Create the CourtListener access API without tying it to any deployment host."""
    api = FastAPI(title="CourtListener Access", version="0.1.0")
    cache = R2Cache.from_env() if client_factory is None else None
    rate_limiter = CourtListenerRateLimiter() if client_factory is None else None

    def client() -> CourtListenerClient:
        if client_factory is not None:
            return client_factory()
        return CourtListenerClient(cache=cache, rate_limiter=rate_limiter)

    @api.exception_handler(CourtListenerError)
    def courtlistener_error_handler(_: Request, exc: CourtListenerError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.to_public_dict()},
        )

    @api.exception_handler(StarletteHTTPException)
    def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code in {HTTP_NOT_FOUND, HTTP_METHOD_NOT_ALLOWED}:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": _bad_route_detail(request, exc.status_code)},
            )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "UP", "service": "courtlistener-access"}

    @api.get("/dockets/resolve")
    def docket_resolve(
        court_id: str,
        docket_number: str,
        cursor: str | None = None,
    ) -> dict[str, object]:
        return client().resolve_docket(
            court_id=court_id,
            docket_number=docket_number,
            cursor=cursor,
        )

    @api.get("/dockets/{cl_docket_id:int}")
    def docket(cl_docket_id: int) -> dict[str, object]:
        return client().get_docket(cl_docket_id)

    @api.get("/docket-entries/search")
    def docket_entries(
        cl_docket_id: str,
        entry_number: str | None = None,
        cursor: str | None = None,
        order_by: str | None = None,
    ) -> dict[str, object]:
        return client().search_docket_entries(
            cl_docket_id,
            entry_number,
            cursor=cursor,
            order_by=order_by,
        )

    @api.get("/recap-documents/search")
    def recap_documents_search(
        cl_docket_entry_id: str | None = None,
        cl_docket_id: str | None = None,
        entry_number: str | None = None,
        cursor: str | None = None,
        order_by: str | None = None,
    ) -> dict[str, object]:
        try:
            return client().search_recap_documents(
                cl_docket_entry_id=cl_docket_entry_id,
                cl_docket_id=cl_docket_id,
                entry_number=entry_number,
                cursor=cursor,
                order_by=order_by,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.get("/recap-documents/{recap_document_id:int}")
    def recap_document(recap_document_id: int) -> dict[str, object]:
        return client().get_recap_document(recap_document_id)

    @api.get("/recap-documents/{recap_document_id:int}/download-url")
    def recap_document_download_url(recap_document_id: int) -> dict[str, object]:
        return client().get_recap_document_download_url(recap_document_id)

    @api.get("/courts")
    def courts() -> dict[str, object]:
        return client().list_courts()

    @api.get("/courts/{court_id}")
    def court(court_id: str) -> dict[str, object]:
        return client().get_court(court_id)

    @api.get("/search")
    def search(q: str, type: str, cursor: str | None = None) -> dict[str, object]:  # noqa: A002
        try:
            return client().search(q=q, search_type=type, cursor=cursor)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/citation-lookup")
    async def citation_lookup(request: Request) -> dict[str, object]:
        form = await request.form()

        def require_field(name: str) -> str:
            value = form.get(name)
            if not isinstance(value, str) or not value.strip():
                raise HTTPException(status_code=400, detail=f"{name} is required")
            return value.strip()

        lookup = client().lookup_citation(
            volume=require_field("volume"),
            reporter=require_field("reporter"),
            page=require_field("page"),
        )
        return citation_lookup_envelope_dict(lookup)

    return api


def _bad_route_detail(request: Request, status_code: int) -> dict[str, object]:
    message = (
        "HTTP method is not allowed for this backend route"
        if status_code == HTTP_METHOD_NOT_ALLOWED
        else "No matching backend route"
    )
    return {
        "failure_type": "bad_route",
        "message": message,
        "retryable": False,
        "method": request.method,
        "path": request.url.path,
    }
