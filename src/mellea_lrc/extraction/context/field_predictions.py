"""Prepare source-only context for either field reader; never predict fields."""

from eyecite.models import ReferenceCitation as EyeciteReferenceCitation

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.context.leaves import aliases
from mellea_lrc.field_extraction.models import LocatorContext
from mellea_lrc.model.citation_windows import after, before
from mellea_lrc.model.citations import latest
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.events import events
from mellea_lrc.parsing.span import Span


def _root_field_contexts(
    document: Document, rules: ExtractionRules, fields: FieldExtractionConfig
) -> tuple[LocatorContext, ...]:
    """Describe existing locator groups without extracting or normalizing fields."""
    contexts = []
    for citation in document.full_locators:
        prefix, start = before(document, citation, rules.case_name_window)
        suffix, suffix_start = after(document, citation, rules.post_locator_window)
        site = citation.locator_span
        next_start = min(
            (
                item.locator_span.start
                for item in document.full_locators
                if item.locator_span.start >= site.end
            ),
            default=len(document.text),
        )
        contexts.append(
            _locator_context(
                document.text,
                site,
                fields,
                kind=citation.kind,
                name_before=Span(start, start + len(prefix)),
                court_date_after=Span(suffix_start, suffix_start + len(suffix)),
                pin_region=Span(site.end, min(next_start, site.end + rules.post_locator_window)),
            )
        )
    return tuple(contexts)


def _short_field_contexts(document: Document, fields: FieldExtractionConfig) -> tuple[LocatorContext, ...]:
    """Give both readers the same group boundary and currently known name aliases."""
    rules = stable()
    groups = {group.id: group.citation_ids for group in document.short_reporter_colocations}
    by_id = {citation.id: citation for citation in document.short_reporters}
    source_events = events(document.text)
    known_names = tuple(aliases(document))
    contexts = []
    for citation in document.short_reporters:
        group_id = latest(citation.colocation_id)
        first = by_id[groups[group_id][0]] if group_id is not None else citation
        first_span = first.short_locator_span
        start = max(0, first_span.start - rules.case_name_window)
        for previous in document.citations:
            if previous.site_span.end <= first_span.start:
                start = max(start, previous.site_span.end)
        for event in source_events:
            if not isinstance(event, EyeciteReferenceCitation):
                _, end = event.span_with_pincite()
                if end <= first_span.start:
                    start = max(start, end)
        site = citation.short_locator_span
        contexts.append(
            _locator_context(
                document.text,
                site,
                fields,
                kind="short_reporter",
                name_before=Span(start, first_span.start),
                court_date_after=Span(site.end, site.end),
                pin_region=site,
                known_names=known_names,
            )
        )
    return tuple(contexts)


def _locator_context(
    source: str,
    locator: Span,
    fields: FieldExtractionConfig,
    *,
    kind: str,
    name_before: Span,
    court_date_after: Span,
    pin_region: Span,
    known_names: tuple[str, ...] = (),
) -> LocatorContext:
    """Crop once for both readers, retaining complete shared locator-group bounds."""
    bounds = (locator, name_before, court_date_after, pin_region)
    start = min(max(0, locator.start - fields.context_before), *(span.start for span in bounds))
    end = max(min(len(source), locator.end + fields.context_after), *(span.end for span in bounds))
    return LocatorContext(
        text=source[start:end],
        source_span=Span(start, end),
        locator=locator,
        kind=kind,
        name_before=name_before,
        court_date_after=court_date_after,
        pin_region=pin_region,
        known_names=known_names,
    )
