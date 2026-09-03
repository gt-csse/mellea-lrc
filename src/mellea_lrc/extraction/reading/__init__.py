"""What eyecite is made to read, and where each correction to it lives.

eyecite reads clean text well. This project reads text a PDF converter produced,
and every module here is one answer to a way that goes wrong -- kept together
because they are the same kind of thing, and kept out of
:mod:`~mellea_lrc.extraction.structure` because deciding what a citation *says*
is not deciding what several citations *mean together*.

:mod:`~mellea_lrc.extraction.reading.relaxation`
    The reporter joins. eyecite writes a literal space between volume, reporter
    and page; extraction leaves several, or none, or a page break. This is the
    one setting a caller chooses, and the only one that changes whether a
    citation is found at all.

:mod:`~mellea_lrc.extraction.reading.pin_cites`
    The same defect inside a pin cite, twice: the separator and the range
    hyphen. Applied by swapping module state, because eyecite offers no seam.

:mod:`~mellea_lrc.extraction.reading.post_citation`
    Where the scan for a court and date has to stop. Left alone it takes them
    from the next citation.

:mod:`~mellea_lrc.extraction.reading.courts`
    Turning what the parenthetical says into a court, which courts-db does by
    one hand-entered spelling per court.

:mod:`~mellea_lrc.extraction.reading.dockets`
    Not a correction but an addition: a case identified by docket number rather
    than by a reporter page, taught to eyecite as an extractor of its own.

Each is measured against the corpora in `exploration/notes/`, and none of them
rewrites the text. Spans index the document as written, which is what makes the
result checkable.
"""
