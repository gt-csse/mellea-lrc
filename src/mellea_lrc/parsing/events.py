"""Citation token stream, including noncase boundaries for Id. chronology."""

from functools import lru_cache

from eyecite import get_citations
from eyecite.models import CitationBase, IdToken, SupraToken, TokenExtractor

from mellea_lrc.matching.literal_to_regex import fuzzy_literal
from mellea_lrc.parsing.markers import ID_MARKER
from mellea_lrc.parsing.reporters import _reporter_tokenizer, _ReporterTokenizer


@lru_cache(maxsize=1)
def _leaf_tokenizer() -> _ReporterTokenizer:
    """Retain eyecite's event stream, with relaxed Id/supra marker tokens."""
    extractors = []
    for extractor in _reporter_tokenizer().extractors:
        constructor = getattr(extractor.constructor, "__self__", None)
        if constructor is IdToken:
            regex, strings = ID_MARKER, ["id", "ibid"]
        elif constructor is SupraToken:
            regex = rf"(?<![^\W\d])({fuzzy_literal('supra', whitespace=True, newline=True)})(?!\w)"
            strings = ["supra"]
        else:
            extractors.append(extractor)
            continue
        extractors.append(
            TokenExtractor(
                regex=regex,
                constructor=extractor.constructor,
                extra=extractor.extra,
                flags=extractor.flags,
                strings=strings,
            )
        )
    return _ReporterTokenizer(extractors=extractors)


@lru_cache(maxsize=32)
def events(source: str) -> tuple[CitationBase, ...]:
    """Keep noncase/unknown events: they are required for correct Id chronology."""
    return tuple(get_citations(source, tokenizer=_leaf_tokenizer()))
