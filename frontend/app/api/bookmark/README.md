# Maintained bookmark fixture

This subsystem intentionally lives in the frontend demo application rather
than `src/mellea_lrc`.

It maintains two synchronized, committable artifacts:

- `fixtures/bookmarked/bookmarked.txt` is the aggregate plain-text test fixture;
- `fixtures/bookmarked/bookmarks.json` stores extraction-level citation identity
  and one or more provenance observations.

Citation identity is derived from the extracted citation kind and sorted field
values. It does not use a retrieval or assessment result. Re-bookmarking the
same identity from the same source is a no-op. Bookmarking it from a new source
adds provenance without adding another TXT block.

The schema-versioned JSON file is authoritative. The TXT fixture is regenerated
from it whenever the store changes; there is no legacy migration path.

Set `MELLEA_LRC_BOOKMARK_DIR` to test against an isolated directory. The default
fixture should be committed whenever a case or its investigation comment changes.
