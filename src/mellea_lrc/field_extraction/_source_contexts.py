"""Prepare rule reading boundaries from unchanged source and supplied anchors."""

from __future__ import annotations

from dataclasses import dataclass

from eyecite.models import ReferenceCitation, ShortCaseCitation

from mellea_lrc.field_extraction._case_names import read_case_name, read_short_case_name
from mellea_lrc.field_extraction.models import FieldContext
from mellea_lrc.parsing.case_names import CaseName
from mellea_lrc.parsing.colocations import colocation_groups
from mellea_lrc.parsing.dockets import docket_readings
from mellea_lrc.parsing.events import events
from mellea_lrc.parsing.reporters import full_reporter_readings, short_reporter_readings
from mellea_lrc.parsing.span import Span


@dataclass(frozen=True)
class _Site:
    span: Span
    kind: str
    edition: str | None = None

    @property
    def id(self) -> str:
        return f"{self.span.start}:{self.span.end}"


def source_contexts(
    source: str,
    locators: tuple[Span, ...],
    *,
    case_name_window: int,
    post_locator_window: int,
    colocation_max_meaningful_gap: int,
) -> tuple[FieldContext, ...]:
    """Include supplied anchors, including dockets discovered by a caller.

    The standalone reader recognizes the same narrow docket grammar as locator
    discovery. An opaque/hunted docket supplied as a target is also a boundary;
    caller-provided contexts can account for any additional known locators.
    """
    full = full_reporter_readings(source)
    short = short_reporter_readings(source)
    by_span = {
        reading.span: _Site(Span(*reading.span), "reporter", reading.citation.groups.get("reporter"))
        for reading in full
    }
    for reading in docket_readings(source):
        if not any(Span(*reading.span).overlaps(site.span) for site in by_span.values()):
            by_span[reading.span] = _Site(Span(*reading.span), "docket")
    for reading in short:
        by_span.setdefault(
            reading.span,
            _Site(Span(*reading.span), "short_reporter", reading.citation.groups.get("reporter")),
        )
    for locator in locators:
        key = (locator.start, locator.end)
        if key not in by_span:
            # Parsing the bare anchor identifies short reporters whose extended
            # span differs from discovery in a larger source. Other supplied
            # anchors are docket locators; callers need not adopt our grammar.
            quote = source[locator.start : locator.end]
            short_anchor = any(item.span == (0, len(quote)) for item in short_reporter_readings(quote))
            by_span[key] = _Site(locator, "short_reporter" if short_anchor else "docket")
    sites = tuple(sorted(by_span.values(), key=lambda site: site.span.start))
    full_sites = tuple(site for site in sites if site.kind != "short_reporter")
    short_sites = tuple(site for site in sites if site.kind == "short_reporter")
    full_groups = colocation_groups(
        source,
        tuple((site.id, site.span, site.edition) for site in full_sites),
        colocation_max_meaningful_gap,
    )
    known_names: set[str] = set()
    for reading in full:
        start = max(0, reading.span[0] - case_name_window)
        name = read_case_name(source[start : reading.span[0]], start, source[slice(*reading.span)])
        if name is not None:
            try:
                parsed = CaseName.from_quote(source[name.start : name.end])
            except ValueError:
                continue
            known_names.update(
                value
                for value in (parsed.as_citation(), parsed.plaintiff, parsed.defendant, parsed.subject)
                if value
            )
    # Other authorities bound a short citation's name, just as they do before
    # other leaf types have been materialized in the document pipeline.
    source_events = events(source) if short_sites else ()
    short_barriers = [
        Span(*event.span()) for event in source_events if not isinstance(event, ShortCaseCitation)
    ]
    short_barriers.extend(site.span for site in full_sites)
    for site in short_sites:
        previous = max((other.span.end for other in sites if other.span.end <= site.span.start), default=0)
        previous = max(
            previous,
            max(
                (
                    event.span_with_pincite()[1]
                    for event in source_events
                    if not isinstance(event, ReferenceCitation)
                    and event.span_with_pincite()[1] <= site.span.start
                ),
                default=0,
            ),
        )
        start = max(previous, site.span.start - case_name_window)
        name = read_short_case_name(source[start : site.span.start], start, tuple(sorted(known_names)))
        if name is not None:
            short_barriers.append(name)
    short_groups = colocation_groups(
        source,
        tuple((site.id, site.span, site.edition) for site in short_sites),
        colocation_max_meaningful_gap,
        barriers=short_barriers,
    )
    groups = {identifier: group for group in (*full_groups, *short_groups) for identifier in group}
    by_id = {site.id: site for site in sites}
    result = []
    for locator in locators:
        site = by_span[(locator.start, locator.end)]
        members = tuple(by_id[identifier] for identifier in groups.get(site.id, (site.id,)))
        first = min(member.span.start for member in members)
        last = max(member.span.end for member in members)
        member_ids = {member.id for member in members}
        boundaries = sites if site.kind == "short_reporter" else full_sites
        previous = max(
            (
                other.span.end
                for other in boundaries
                if other.id not in member_ids and other.span.end <= first
            ),
            default=0,
        )
        if site.kind == "short_reporter":
            previous = max(
                previous,
                max(
                    (
                        event.span_with_pincite()[1]
                        for event in source_events
                        if not isinstance(event, ReferenceCitation) and event.span_with_pincite()[1] <= first
                    ),
                    default=0,
                ),
            )
        following = min(
            (
                other.span.start
                for other in boundaries
                if other.id not in member_ids and other.span.start >= last
            ),
            default=len(source),
        )
        next_locator = min(
            (other.span.start for other in full_sites if other.span.start >= locator.end),
            default=len(source),
        )
        result.append(
            FieldContext(
                kind=site.kind,
                name_before=Span(max(previous, first - case_name_window), first),
                court_date_after=Span(last, min(following, last + post_locator_window)),
                pin_region=locator
                if site.kind == "short_reporter"
                else Span(locator.end, min(next_locator, locator.end + post_locator_window)),
                known_names=tuple(sorted(known_names)) if site.kind == "short_reporter" else (),
            )
        )
    return tuple(result)
