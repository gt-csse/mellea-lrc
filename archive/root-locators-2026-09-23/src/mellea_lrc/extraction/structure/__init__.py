"""What a document's citations mean together, once each has been read.

Reading a citation answers what it says. These answer how several of them
relate, and neither module changes a citation or looks at the text again.

:mod:`~mellea_lrc.extraction.structure.colocation`
    Citations occupying the same span. A filing citing a case in parallel writes
    one citation and several identifiers for it, and eyecite returns them
    unlinked. This reports the coincidence and refuses to call it identity --
    `Brown, 347 U.S. 483, 349 U.S. 294 (1955)` has one name, one date and two
    decisions.

:mod:`~mellea_lrc.extraction.structure.citation_tree`
    Citations referring to the same authority. Every short form, `Id.` and
    supra traced to the full citation that introduced it, so an authority is
    identified once and each return visit keeps its own pin cite as its own
    checkable claim.

Both are consumed rather than acted on: co-location is a candidate signal that
validation settles by resolving, and an occurrence the tree cannot place is
reported rather than guessed at.
"""
