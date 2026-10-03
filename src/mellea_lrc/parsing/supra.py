"""Named supra source reading shared with field normalization."""

import re
from dataclasses import dataclass

from mellea_lrc.matching.literal import fuzzy_literal
from mellea_lrc.parsing.events import events
from mellea_lrc.parsing.pin_cite import pin_after

# A preceding optional volume may touch the marker when whitespace is absent;
# a preceding letter still makes this part of another word.
_SUPRA = re.compile(rf"(?<![^\W\d]){fuzzy_literal('supra', whitespace=True, newline=True)}(?!\w)", re.I)
_AT = re.compile(rf"\s*[,.]?\s*{fuzzy_literal('at ', whitespace=True, newline=True)}(?=\d|[*¶])", re.I)
_NOTE = re.compile(rf"\s*,?\s*{fuzzy_literal('note ', whitespace=True, newline=True)}\d+", re.I)
_NAME = re.compile(
    r"(?P<name>[A-Z][\w.'’&/-]*(?:(?:\s+|\s*,\s*)[\w.'’&/-]+){0,20}?)"
    r"\s*(?:,|\s)\s*(?P<volume>\d+)?\s*$"
)
_SIGNAL = re.compile(r"(?:See(?:\s+also)?|But\s+see|Cf\.|Accord|Compare)\s+", re.I)


@dataclass(frozen=True, slots=True)
class SupraReading:
    """Source offsets for one named supra, shared with its field normalizer."""

    span: tuple[int, int]
    antecedent_span: tuple[int, int]
    volume: int | None
    pin_span: tuple[int, int] | None


def supra_readings(source: str, *, names: tuple[str, ...] = ()) -> tuple[SupraReading, ...]:
    """Read source-shaped supra references without requiring an existing root.

    Known source names improve boundaries but never decide attribution. This
    service also reads an isolated quote during normalization, so discovery
    and normalization use the same relaxed marker and adjacent-pin grammar.
    A bare ``supra`` is not a case-name anchor: internal cross-references must
    not become case citations merely because the keyword was recognized.
    """
    markers = tuple(_SUPRA.finditer(source))
    source_events = events(source)
    readings: list[SupraReading] = []
    for marker in markers:
        # Internal section references have a word immediately after supra,
        # rather than the supported punctuation, at-pin or note join. This is
        # a source-shape boundary, not a list of section names or stopwords.
        following = source[marker.end() :]
        if (
            following
            and re.match(r"\s+\w", following)
            and not (_AT.match(following) or _NOTE.match(following))
        ):
            continue
        start = max(0, marker.start() - 160)
        for event in source_events:
            _, finish = event.span_with_pincite()
            if finish <= marker.start():
                start = max(start, finish)
        prefix = source[start : marker.start()]
        known: list[tuple[tuple[int, int], int | None]] = []
        for name in names:
            expression = (
                rf"(?<!\w){fuzzy_literal(name, whitespace=True, newline=True)}(?!\w)"
                r"\s*,?\s*(?P<volume>\d+)?\s*$"
            )
            if match := re.search(expression, prefix, re.I):
                finish = match.start("volume") if match.group("volume") else len(prefix)
                finish = len(prefix[:finish].rstrip(" ,\t\r\n"))
                known.append(
                    (
                        (start + match.start(), start + finish),
                        int(match.group("volume")) if match.group("volume") else None,
                    )
                )
        if known:
            name_span, volume = min(known, key=lambda item: item[0][0])
        elif match := _NAME.search(prefix):
            name_start = match.start("name")
            if signal := _SIGNAL.match(match.group("name") + " "):
                name_start += signal.end()
            if name_start >= match.end("name") or not prefix[name_start].isupper():
                continue
            name_span = (start + name_start, start + match.end("name"))
            volume = int(match.group("volume")) if match.group("volume") else None
        else:
            continue
        end = marker.end()
        if note := _NOTE.match(source[end:]):
            end += note.end()
        limit = min(len(source), end + 100)
        limit = min(
            limit,
            next((item.start() for item in markers if item.start() > marker.start()), limit),
            min(
                (event.span()[0] for event in source_events if event.span()[0] > marker.start()),
                default=limit,
            ),
        )
        pin = None
        if at := _AT.match(source[end:limit]):
            pin = pin_after(source, end + at.end(), limit)
            if pin is not None:
                end = pin[1]
        if period := re.match(r"\s*\.", source[end:limit]):
            end += period.end()
        readings.append(SupraReading((name_span[0], end), name_span, volume, pin))
    return tuple(readings)
