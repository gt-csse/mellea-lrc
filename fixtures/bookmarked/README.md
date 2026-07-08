# Bookmark research fixture

`bookmarks.json` is authoritative. `bookmarked.txt` is its generated,
human-readable projection and is also used as snapshot input. A bookmark is a
citation context, not a retrieval verdict: its identity is derived from
normalized `matched_text` plus the surrounding context. Repeated observations
belong in `provenances`.

## Comments

Comments are short research conclusions. Prefer this stable shape:

```text
Finding: what the pipeline/search returned.
Evaluation: whether the evidence identifies the cited case, and why.
Next: the smallest useful follow-up (omit when resolved).
```

Do not paste raw API payloads into a comment. Candidate summaries belong in
the retrieval snapshot; comments should interpret that evidence.

Update a comment by an unambiguous bookmark-ID prefix so neither JSON nor the
generated text fixture needs to be edited by hand:

```bash
uv run python scripts/bookmark_fixture.py citation:43ae974f \
  $'Finding: opinion 0; RECAP 1.\nEvaluation: the RECAP docket matches the cited parties.'
```

Run `uv run pytest tests/test_bookmark_fixture.py` after changes.
