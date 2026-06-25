"""Grounding checks for LLM-proposed citation fields."""


def is_in_context(value: str, document_context: str) -> bool:
    """Return whether a proposed value is grounded in the source context."""
    return _normalize_whitespace(value) in _normalize_whitespace(document_context)


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())
