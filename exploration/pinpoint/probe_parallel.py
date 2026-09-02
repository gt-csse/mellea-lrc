"""Does the tree treat a parallel citation as one authority or three?

Three. And eyecite already holds the evidence needed to say otherwise.

`FullCaseCitation.is_parallel_citation` fires on these: each parallel reporter
takes the plaintiff, defendant and year from the citation before it, which is
why all three below carry `defendant='St. Amant'`. But it is a metadata helper,
not a data model -- it copies names and returns None. Nothing links the three
objects, and the tree, resolving each on its own identifier, makes three
authorities out of one case.

Their **full spans are the signal**: 11-78, 11-78, 12-78 for the same sentence.
Citations whose full spans coincide are one authority, and that is cheaper and
firmer evidence than reading a parallel reporter out of another citation's
`extra`.
"""

import contextlib
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from exploration.pinpoint.survey_extra import body
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.citation_tree import build_citation_tree

TEXT = "St. Amant v. Thompson, 390 U.S. 727, 731, 88 S.Ct. 1323, 20 L.Ed.2d 262 (1968). Id. at 733."

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    document = extract_from_plain_text(TEXT, relaxation=Relaxation.FULL)
    tree = build_citation_tree(document)

print(f"synthetic: {len(tree.authorities)} authorities from one parallel citation")
for authority in tree.authorities:
    print(f"  authority {authority.root.matched_text!r}  occurrences={len(authority.occurrences)}")

print("\n--- the corpus: is the third parallel reporter extracted? ---")
path = next(Path("data/false-citation-bench-locator-only-v2.0/documents_txt").glob("006*.txt"))
text = body(path)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    document = extract_from_plain_text(text, relaxation=Relaxation.FULL)

for needle in ("L.Ed.2d 262", "L.Ed.2d 125", "S.Ct. 1323", "S.Ct. 209"):
    written = len(re.findall(re.escape(needle), text))
    extracted = sum(1 for c in document.citations if needle in " ".join(c.matched_text.split()))
    print(f"  {needle!r:<18} written {written}, extracted {extracted}")

print("\n--- how many corpus authorities are a parallel reporter? ---")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    tree = build_citation_tree(document)
parallel_reporters = ("S.Ct.", "L.Ed.", "L. Ed.")
hits = [
    a
    for a in tree.authorities
    if any(r in str(getattr(a.root.citation, "reporter", "")) for r in parallel_reporters)
]
print(f"  {len(hits)} of {len(tree.authorities)} authorities in document 006")
for authority in hits:
    print(f"    {authority.root.matched_text!r} occurrences={len(authority.occurrences)}")
