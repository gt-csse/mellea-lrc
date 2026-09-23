# Handoff: extraction agent, 2026-09-15

Untracked on purpose. Do not commit this file.

## Where things are

- Code: this worktree, branch `extraction/case-name-recovery`, HEAD `6b5b5808`
  (relaxed `Id.`). Clean tree.
- Data: `~/CodingProjects/mellea-lrc-datasets` (branch `main`, shared with the
  validation agent: **commit by explicit paths only**).
- Rules to read first: `agents/extraction-annotator.md` (the Log and Open list),
  `annotation-v4.0/PROTOCOL.md`. Messages to validation go in `agents/NOTE.md`.
- You annotate **extraction fields only**. Never write `validation` or
  `out_of_scope_pincite` rows.
- Never mention Claude/AI in commits, PRs, code or docs. No co-author trailers.
  Never post to GitHub or teammates on the user's behalf.

## Done today

- Draft PR #87 (remove Label Studio) and draft PR #88 (right-margin line
  numbers, `docs/Preprocessing.md`, counts removed, real-page tests removed).
  Both await the user.
- `extraction-eval-4`: seven filings annotated, 1,147 rows (datasets `4285e48`,
  `154e01b`). Note to validation written.
- Relaxed `Id.` (`Id .` with a space before the period): code `6b5b5808`;
  eval-3 annotation fixes `5730ed7`. Measured cost is in the annotator note,
  entry "`Id .` is read now, and what it costs": +21 real, +51 not a case.

## Open, in order

1. **Docket reader fix. Waiting on the user's go-ahead.** On eval-4, roots are 387/397 · 387/390.
   Of the 10 missed roots, 9 are docket numbers written beside a WL/LEXIS number, and 1 is
   `580 F. Supp 3d 114` (the reporter has no period). The research is done:
   Bluebook 10.8.1(a) writes the docket "as it appears on court documents", so
   shape lists are incomplete by design. Proposal the user has not yet approved:
   - A docket beside a database identifier is found **by position**: a `No.` signal,
     then text up to the comma before an eyecite-read WL/LEXIS locator, with the
     courts-db court in the parenthetical after the identifiers. This covers two
     identifiers in a row (Ashton, Purnell) and `CV-15-00077`, `S-12-2817`,
     `CV 19-7532`, `C 10-04368 RS`, `2:21-CV-00403MJH`.
   - A standalone docket keeps the shape rules, but adopts juriscraper's district
     pattern `(\d{1,2}:)?\d\d-[a-zA-Z]{1,4}-\d{1,10}` (which fixes `23-cv-19`).
   - A bare `19-6658` with no `No.`: left out unless measured safe.
   Code: `src/mellea_lrc/extraction/reading/dockets.py`. Measure the way the
   `Id.` change was measured: dump every citation before and after over the
   corpus and all four held-out sets, classify each new hit against the
   annotations, and report the true and false counts. Scratch scripts from that
   run: `/private/tmp/claude-501/-Users-woodygoodenough-CodingProjects-mellea-lrc/14551910-eea4-4cdc-8b08-2cdf4fcb85c1/scratchpad/idrelax/`
   (`dump.py`, `compare.py`).
2. **Remind the user** where the "does this `Id.` return to a case?" model
   check should run. The user leans towards pincite validation; the alternatives
   are extraction's leaf prune or reassign. Memory file: `project_id_antecedent_check.md`.
3. The two false roots are short forms written without `at` (`Singh , 123 F.4th 95`,
   `Triplex , 900 S.W.2d 721`); the third is `2006 WL` split by footnote 6. No fix proposed yet.
4. Older sets lack 21 docket numbers written beside WL/LEXIS (corpus 9, eval-1 1,
   eval-2 5, eval-3 6). Eval-4 has them. Not added: each would be a new root for
   validation. The user's call.
5. Validation's last eval-3 note lists unfixed rows: `63113826_21-o104`,
   `4234232_20220812-o149`, `-o060`, `63113826_21-o050`, and three front-cut
   names in `23-235_303109`. Only if the user asks.
6. Small cleanups: `courtlistener/client.py` imports `requests`, which no
   dependency group declares; 12 files on this branch fail `ruff format --check`
   (they already did).

## How to run

    D=~/CodingProjects/mellea-lrc-datasets
    uv run python -m evaluations.extraction.tree --dataset $D/extraction-eval-4/documents \
        --documents $D/evaluation_set_4/filings_txt --arms eyecite augmented grown [--detail]
    python3 tools/check_extraction.py extraction-eval-4   # from $D
    python3 tools/check_evidence.py                       # from $D

## Development checkout update — 2026-09-15

Continue pipeline development in `/Users/woodygoodenough/CodingProjects/mellea-lrc-e2e`,
branch `codex/pipeline-e2e`. It starts from `validation/record-operations` at
`b68f89b6` and integrates this checkout's committed history and uncommitted
locator/colocation/docket-audit work. The original extraction changes remain
here as a preserved copy. Read the new checkout's untracked `handoff.md` for
integration status and open work.

The current task owns development across the pipeline. The other worker's
scope is annotation and independent testing. Operation interfaces belong in
the shared core and are stage-neutral; stages supply provenance and order.
