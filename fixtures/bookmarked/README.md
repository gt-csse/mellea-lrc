# Bookmark research fixture

Each named `sets/bookmark-<name>.json` is authoritative and contains the
research comments. Its matching `sets/bookmark-<name>.txt` is the generated
extraction-input projection: it contains only
the copied source contexts, separated by whitespace, so fixture metadata cannot
be misread as citation text. A bookmark is a
citation context, not a retrieval verdict: its identity is derived from
normalized `matched_citation_text` plus the surrounding context. Repeated
observations belong in `provenances`. The Python hook and frontend share the
same sorted-JSON SHA-256 identity contract; the fixture test enforces it.

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
uv run python scripts/bookmark_fixture.py comment --set research citation:43ae974f \
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

Named research fixtures are independent bookmark stores. Each has its own
authoritative `bookmark-<name>.json`, generated `bookmark-<name>.txt`, and
expected-results README in `sets/`. Add directly to a named set:

```bash
uv run python scripts/bookmark_fixture.py add --set date-recovery \
  --matched-citation-text "..." --context "..." --comment "..." \
  --source-path "../local/test_data/6.txt" --citation-span-start 1 --citation-span-end 2
```

Run a named fixture as one document:

```bash
uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus \
  --file date-extraction --phase assessment
```

Run `uv run pytest tests/test_bookmark_fixture.py` after changes.
