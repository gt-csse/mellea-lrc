"""Read the opaque cursor from CourtListener search pagination links."""

from urllib.parse import parse_qs, urlparse


def cursor_from_url(url: str) -> str | None:
    """Return one nonempty cursor, or None for an unusable pagination link."""
    values = parse_qs(urlparse(url).query).get("cursor", ())
    return values[0] if len(values) == 1 and values[0] else None
