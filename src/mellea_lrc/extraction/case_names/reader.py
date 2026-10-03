"""Case-name source reading local to the full-citation field stage."""

import re

from mellea_lrc.model.citations.fields.case_name import CaseName, CaseNameKind
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.reporters import full_reporter_readings

_CASE = re.compile(
    r"(?:In re|Ex parte|Matter of)\s+[^,;\n]{2,100}|[A-Z][^,;\n]{0,100}?\s+v\.\s+[^,;\n]{1,100}"
)
_PROCEDURAL_NEAR_SITE = re.compile(
    r"(?<!\w)(?P<name>(?:In\s+re|Ex\s+parte|Matter\s+of)\s+[^;\n]{2,100}?)\s*,\s*$",
    re.I,
)
_SIGNAL = re.compile(r"^(?:See(?: also)?|Cf\.|But see|Accord|Compare)\s+", re.I)
_NAME_TOKEN = re.compile(r"[\w.'’&-]+")
_VERSUS = re.compile(r"\s+v\.\s+")
_PARTIAL_NAME = re.compile(
    r"(?<!\w)(?P<name>[A-Z][\w.'’&/-]*"
    r"(?:\s+(?:[A-Z][\w.'’&/-]*|of|the|and|for|cases)){1,12}"
    r"(?:,\s+(?:Inc\.|Corp\.|LLC))?)\s*,\s*$"
)


def _reporter_name_span(reporter_quote: str | None, before: str, start: int) -> Span | None:
    """Use eyecite's parsed parties to anchor a written name, not surrounding prose."""
    if reporter_quote is None:
        return None
    excerpt = before + reporter_quote
    parsed = next(
        (item for item in full_reporter_readings(excerpt) if item.span == (len(before), len(excerpt))),
        None,
    )
    if parsed is None:
        return None
    metadata = parsed.citation.metadata
    if not metadata.plaintiff or not metadata.defendant:
        return None
    plaintiff_tokens = _NAME_TOKEN.findall(metadata.plaintiff)
    defendant_tokens = _NAME_TOKEN.findall(metadata.defendant)
    if not plaintiff_tokens or not defendant_tokens:
        return None
    first, last = plaintiff_tokens[0], defendant_tokens[-1]
    for separator in reversed(tuple(_VERSUS.finditer(before))):
        left = before[: separator.start()]
        right = before[separator.end() :]
        starts = tuple(re.finditer(rf"(?<!\w){re.escape(first)}(?!\w)", left, re.I))
        end = re.search(rf"(?<!\w){re.escape(last)}(?!\w)", right, re.I)
        if not starts or end is None:
            continue
        local_start = starts[-1].start()
        local_end = separator.end() + end.end()
        if local_end - local_start > 100:
            continue
        try:
            CaseName.from_quote(before[local_start:local_end])
        except ValueError:
            continue
        return Span(start + local_start, start + local_end)
    return None


def _reporter_partial_span(reporter_quote: str | None, before: str, start: int) -> Span | None:
    """Ground a lone party read by eyecite without treating it as complete."""
    if reporter_quote is None:
        return None
    excerpt = before + reporter_quote
    parsed = next(
        (item for item in full_reporter_readings(excerpt) if item.span == (len(before), len(excerpt))),
        None,
    )
    if parsed is None:
        return None
    metadata = parsed.citation.metadata
    if bool(metadata.plaintiff) == bool(metadata.defendant):
        return None
    one_party = metadata.plaintiff or metadata.defendant
    if one_party is None or len(one_party) > 100:
        return None
    expression = r"\s+".join(re.escape(token) for token in one_party.split())
    match = re.search(rf"(?<!\w)({expression})\s*,?\s*$", before, re.I)
    if match is None:
        return None
    quote = before[match.start(1) : match.end(1)]
    try:
        if CaseName.from_quote(quote).kind is CaseNameKind.PARTIAL:
            return Span(start + match.start(1), start + match.end(1))
    except ValueError:
        return None
    return None


def _partial_name_span(before: str, start: int) -> Span | None:
    """Read a nearby multiword fragment when no complete name was found."""
    match = _PARTIAL_NAME.search(before)
    if match is None:
        return None
    name = match.group("name")
    signal = _SIGNAL.match(name)
    offset = signal.end() if signal else 0
    name = name[offset:]
    if not name:
        return None
    try:
        if CaseName.from_quote(name).kind is not CaseNameKind.PARTIAL:
            return None
    except ValueError:
        return None
    return Span(start + match.start("name") + offset, start + match.end("name"))


def _procedural_name_span(before: str, start: int) -> Span | None:
    match = _PROCEDURAL_NEAR_SITE.search(before)
    if match is None:
        return None
    name = match.group("name")
    try:
        if CaseName.from_quote(name).kind not in {CaseNameKind.IN_RE, CaseNameKind.EX_PARTE}:
            return None
    except ValueError:
        return None
    return Span(start + match.start("name"), start + match.end("name"))


def read_case_name(before: str, start: int, reporter_quote: str | None) -> Span | None:
    """Read one name from a caller-bounded prefix, with unchanged source offsets."""
    span = _reporter_name_span(reporter_quote, before, start) or _procedural_name_span(before, start)
    if span is not None:
        return span
    matches = tuple(_CASE.finditer(before))
    if not matches:
        return _partial_name_span(before, start) or _reporter_partial_span(reporter_quote, before, start)
    match = matches[-1]
    name = match.group().rstrip(" ,")
    signal = _SIGNAL.match(name)
    offset = signal.end() if signal else 0
    name = name[offset:]
    if not name:
        return None
    return Span(start + match.start() + offset, start + match.start() + offset + len(name))
