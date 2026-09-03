# Extraction is scored against the document, never against the reporter

The boundary this layer works inside, written down because it is easy to drift
across and the drift is invisible in the numbers.

## The rule

**A parsed field is right when it matches what the page states**, read as the
writer meant it under citation convention, with the flexibility a converter's
damage requires. Nothing in this layer asks whether what the writer stated is
*true*.

If a filing writes `Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2009)`, extraction
recording 2009 is **correct**. Whether that is the year of that decision is
validation's question, on validation's own ground truth, which is a separate
layer of annotation and not this one's to build.

Getting that backwards would be worse than sloppy. A tool built to surface false
citations has to recover a wrong citation faithfully; scoring extraction against
a reporter would penalise it for doing exactly that, and would reward a parser
that quietly repaired the filing into something defensible.

## What "wrong" means in this layer's measurements

Internally contradicted by the same document, and nothing else. Every defect
column in `evaluations/extraction/matrix.py` is of that kind:

- **a date taken from another case** -- the span demonstrably reaches over an
  unrelated citation and takes the parenthetical belonging to it. The evidence
  is two citations and their offsets, both on the page.
- **a pin cite lost** -- a page eyecite recognised as part of the citation and
  filed under `extra`, where nothing looks for it.
- **a court written, not recorded** -- a parenthetical naming a court beside a
  citation carrying none.

None of these needs a reporter, an annotation or a lookup. They are structural
contradictions, which is why the matrix could measure 68 to 1, 21 to 0 and 50 to
14 on a corpus nobody has labelled.

## What was rejected for crossing the line

The reporter **edition date range**. `Edition` carries a start and end year, so
`695 F. Supp. 2d` dated 1975 looks checkable. It was dropped for being unreliable
-- seven false alarms and no true ones -- but the better reason is that the
question is not this layer's. "Could this citation exist?" is answered by an
external authority, and an extraction that declined to record a stated year
because a database disbelieved it would be deciding the filing was wrong.

Note the contrast with the pin-cite range check, which *is* inside the line: a
pin cite below the authority's own first page contradicts two numbers the
document itself states. It says an attribution is impossible, not that a writer
erred.

### What the edition range actually detects

Every flag it raises was read. On the mined corpus there are nine, and in all
nine the reporter **as written matches the reporter recorded**::

    167 F.R.D. 649 (S.D.N.Y. 1996)     F.R.D.    db range 2001-
    94 F.R.D. 631, 637 (D. Kan. 1982)  F.R.D.    db range 2001-
    807 F. Supp. 109, 110 (D. Kan. 1992)  F. Supp.  db range 1932-1988
    960 F.Supp. 253 (D. Kan. 1997)     F. Supp.  db range 1932-1988

F.R.D. began in the 1930s and F. Supp. ran to 1998. The citations are sound and
the ranges are wrong, so the signal is about reporters-db rather than about any
document. There is nothing here for this layer to capture, and nothing a
reviewer could be usefully asked.

### The version of this that would be ours, and why it is empty

A **misread reporter** is an extraction defect: if `695 F. Supp. 2d 1149` lost
its `2d`, the recorded reporter would be wrong, and a year outside the edition's
range would be the clue.

The year is not needed to see it. The unread `2d` would still be sitting between
the reporter and the page, on the same line, in the document -- so the check is
document-internal and needs neither a database nor a reader. Run across all
2,607 case citations in both corpora: **zero reporter suffixes left unread.**

So the answer to "can only a reviewer catch this" is no, in both directions. The
extraction-relevant version has a deterministic check, and it finds nothing. The
other version is not extraction's question.

## What this means for annotation

Extraction annotation records **what the page states** -- the locator spans of
`false-citation-bench-locator-only-v2.0`, and the scope and attribution
judgements of the tree bench, which are readings of the document and not of any
authority.

Validation ground truth is a different layer of annotation. Where notes here
suggest a model should see a retrieved case name beside the filing's rendering,
that is a suggestion about validation's design and about when the case-name
problem becomes tractable -- not a proposal to score extraction against a
retrieval.
