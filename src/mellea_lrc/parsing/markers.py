"""Relaxed source markers used by Id. and supra readers."""

from mellea_lrc.matching.literal import fuzzy_literal

# Share source recognition with normalization: keep offsets in the original
# text, and relax only whitespace around the written marker. A missing period
# needs an explicit pinpoint join; ordinary uses of the word "ID" are not sites.
_DOTTED_ID = "|".join(fuzzy_literal(marker, whitespace=True, newline=True) for marker in ("Id.", "Ibid."))
_ID_PIN_JOIN = fuzzy_literal("at ", whitespace=True, newline=True)
ID_MARKER = rf"(?<!\w)((?:{_DOTTED_ID})(?:\s*,)?|(?:id|ibid)(?=\s+{_ID_PIN_JOIN}(?=[\d*¶])))"
