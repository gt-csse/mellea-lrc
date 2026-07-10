# Bookmark research fixture

`bookmarks.json` is authoritative. `bookmarked.txt` is its generated,
human-readable projection and is also used as snapshot input. A bookmark is a
citation context, not a retrieval verdict: its identity is derived from
normalized `matched_citation_text` plus the surrounding context. Repeated
observations belong in `provenances`.

## Comments

Comments are short retrieval and authority-identity research conclusions.
Proposition support and citation appropriateness belong to a later evaluation
layer, not this fixture's current comments. A useful comment records:

```text
Identity: VERIFIED, UNRESOLVED, NO CANDIDATE, NOT SEARCHED, or LINKAGE DEFECT.
Evidence: the parties, court, docket, date, and decisional-document cues used.
Next: the smallest missing identity-resolution step (omit when resolved).
```

Do not paste raw API payloads or proposition-support judgments into a comment.
Candidate summaries belong in the retrieval snapshot; comments should interpret
only the authority-identity evidence.

Update a comment by an unambiguous bookmark-ID prefix so neither JSON nor the
generated text fixture needs to be edited by hand:

```bash
uv run python scripts/bookmark_fixture.py comment citation:43ae974f \
  $'Finding: opinion 0; RECAP 1.\nEvaluation: the RECAP docket matches the cited parties.'
```

The legacy shorthand still works:

```bash
uv run python scripts/bookmark_fixture.py citation:43ae974f \
  $'Finding: opinion 0; RECAP 1.\nEvaluation: the RECAP docket matches the cited parties.'
```

Add a newly observed citation context through the same hook so bookmark IDs,
provenance IDs, and the text projection stay synchronized:

```bash
uv run python scripts/bookmark_fixture.py add \
  --matched-citation-text "Lampe v. United States , 18 F. App'x 744 (10th Cir. 2001)" \
  --context "Lampe v. United States , 18 F. App'x 744 (10th Cir. 2001)" \
  --source-path "../local/test_data/13.txt" \
  --citation-span-start 2481 \
  --citation-span-end 2635 \
  --comment $'Identity: LINKAGE DEFECT. ...'
```

Run `uv run pytest tests/test_bookmark_fixture.py` after changes.
