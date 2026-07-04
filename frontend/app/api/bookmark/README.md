# Local bookmark store

This subsystem intentionally lives in the frontend demo application rather
than `src/mellea_lrc`.

It maintains two synchronized local artifacts:

- `local/bookmarked/bookmarked.txt` is the aggregate plain-text test fixture;
- `local/bookmarked/bookmarks.json` stores extraction-level citation identity
  and one or more provenance observations.

Citation identity is derived from the extracted citation kind and sorted field
values. It does not use a retrieval or assessment result. Re-bookmarking the
same identity from the same source is a no-op. Bookmarking it from a new source
adds provenance without adding another TXT block.

When JSON is absent, existing TXT blocks are migrated as explicit legacy
records. Their text remains available, but they cannot be matched to an
extracted citation until a structured bookmark for that citation is created.

Set `MELLEA_LRC_BOOKMARK_DIR` to test against an isolated directory.
