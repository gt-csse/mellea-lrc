"""Generators: cheap, deterministic, and each answering one question.

Every generator takes an :class:`~mellea_lrc.extraction.types.ExtractedDocument`
and yields :class:`~mellea_lrc.adjudication.types.Candidate` objects. None of
them decides anything -- deciding is the reviewer's job, and a generator that
decides is a rule that should have been in extraction instead.

They differ enormously in yield, and the difference is the design. A narrow
generator built from a pattern we cannot prove general is affordable to review
*because* it is narrow: `uppercase_reporters` proposed two candidates across 103
documents. A broad one is not: `reporter_sites` proposed 185 across 77, nearly
all of them letterheads. See
`exploration/notes/candidates-and-adjudication.md`.
"""

from mellea_lrc.adjudication.candidates.orphan_short_forms import orphan_short_forms
from mellea_lrc.adjudication.candidates.uppercase_reporters import uppercase_reporters

__all__ = ["orphan_short_forms", "uppercase_reporters"]
