# The citation tree: short forms as their own claims

A brief for the agent taking this over. The module is on this branch and its
tests pass against `main`. It is **not wired into the pipeline**, and wiring it
in is the work.

## 1. Ownership

**Taken up 1 September 2026**, by the worktree that owns `extraction/` —
`~/CodingProjects/mellea-lrc-preprocess-and-extract`, which also holds
`extraction/relaxation` and the three `preprocess/*` branches.

One agent holds both, for the reason in section 5: relaxation decides which
citations exist, and the tree groups whatever the extractor emits. The two do
not share a file or an import, so they look independent and are not — and
section 5 now has the measurement showing how they interact, which is not how
this note originally predicted.

Section 6 is untouched and is still the work.

## 2. What the module does

A filing does not cite an authority once. It cites it in full, then returns to
it as `Id. at 570`, `550 U.S. at 563`, or by party name — and each return visit
usually names a **different page** and attaches a **different proposition** to
it.

`build_citation_tree` resolves every citation, transitively, to the full
citation that introduced its authority. `Id.` may point at a short form that
points at the full citation, and the chain is followed.

## 3. Why it is worth doing

**Validating only full citations checks one claim per authority and skips every
other claim made about it** — which is the larger part of what a brief actually
asserts. An authority cited at pages 563, 570 and 578 is three distinct claims
about three distinct pages, each verifiable on its own and each capable of
being wrong on its own.

Two properties follow:

- Identity is resolved once per authority, not once per occurrence. Ten
  references to one case cost one lookup.
- Pinpoint claims multiply, which is the point.

This connects to a gap measured elsewhere. The reference dataset labels three
citation-shaped defect classes at roughly even weight, and **wrong pincite is
the largest at 39%** — the one class the hallucination miner cannot see at all,
because an order quotes the citation and not the passage. See section 8.9 of
`exploration/AUDIT.md` on `experiment/general-explorations`. The tree is the
structure that makes those claims addressable.

## 4. What the module already gets right

**Nothing is invented.** A citation eyecite could not attribute is reported
rather than guessed at, because attaching a claim to the wrong authority checks
it against the wrong page.

**Two failures that look alike in a count mean opposite things**, and they are
reported separately:

- *out of scope* — positive evidence the citation names something other than a
  case: a statute, a journal article, an unparseable span, or an `id.`
  resolving to one of those. On false-citation-bench this is 252 of 894, and
  every one is correct behaviour.
- *unattributed* — no such evidence. A case citation that could not be traced,
  or a reference that reached no antecedent. On the same corpus this is 17 at
  the default relaxation, and it is the number that measures the tree. See
  section 5: it does not move with the relaxation level, and an earlier figure
  of 20 was measured against an extractor configuration that no longer exists.

Reporting those as one figure would read as a 30% failure rate for what is, in
case citations, one. Read individually the 17 are defensible: one real short
form quoted inside another case's parenthetical and never given in full, and
sixteen `Id.` of which most carry a paragraph pin cite and are probably
references into a pleading's own numbered allegations. That evidence is
deliberately not acted on, because several state courts number opinion
paragraphs in the public-domain format.

## 5. How relaxation changes these numbers

`extraction/relaxation` defines three tokenizer levels — `NONE`, `BOUNDED`
(the default) and `FULL` — differing in how much whitespace damage a citation
may carry and still be found. `ExtractionMetadata.relaxation` records which one
ran.

The tree resolves a short form by finding the full citation that introduced the
authority. **If a level finds a full citation that another level misses, a short
form that had no antecedent acquires one.** So the *unattributed* count in
section 4 is a property of the tree and the relaxation level together, not of
the tree alone.

That was the reasoning. **It has now been measured, and the second half of it
is wrong.** Over the 26 published `false-citation-bench` documents, building the
tree on top of each level:

| level | authorities | occurrences | pinpoint claims | unattributed | out of scope |
|---|---:|---:|---:|---:|---:|
| `NONE` | 369 | 577 | 261 | 17 | 252 |
| `BOUNDED` | 390 | 635 | 270 | 17 | 252 |
| `FULL` | 391 | 635 | 269 | **19** | 251 |

**Relaxation does not lower the unattributed count.** `BOUNDED` finds 21 more
authorities and 58 more occurrences than `NONE`, and resolves not one
previously-unattributed short form: the two sets are identical citation for
citation, not merely equal in size.

The reason is that whitespace damage was never what stranded them. The 17 are
16 `Id.` — 15 of them in document 016 — and one `383 U.S. at 85` quoted inside
another case's parenthetical and never given in full. None is a short form
whose full citation was lost to a line break, which is the case the prediction
assumed. Section 4's reading of them stands; relaxation has no purchase on it.

The number to quote for the tree is therefore **17, at `NONE` or `BOUNDED`
alike**. Section 4 previously said 20; that figure was measured against the old
production extractor, eyecite plus a whitespace repair, which no longer exists
as a configuration.

### What this measurement cannot tell you

**It was taken on text the margin rule never touched, so the `FULL` row does
not describe `FULL` as it would actually run.**

`data/false-citation-bench/documents_txt/` was exported by a Docling run
predating `preprocess/margin-line-numbers`. That rule works on the Docling
document during PDF conversion; it cannot reach an already-exported `.txt`. So
every margin gutter in the published corpus is still in the text that produced
the table above.

That accounts for both citations `FULL` loses, which are in document 022 — one
of the eight bench documents that carry a gutter:

- `214 F.3d at 1068` — a short form that no longer matches its own authority
- an `id.` immediately after it, which chained off that short form

Under `FULL` on gutter-carrying text, `214 F.3d 1058` is read as `214 F.3d 1`,
taking its page from the margin line numbers, so the later short form does not
resolve to it. On margin-adjusted text that misparse does not happen: the
margin branch's own measurement records `FULL` *gaining* `214 F.3d 1058` there,
correctly, and losing nothing. Both of these would go away.

So the `FULL` row measures a condition the preprocessing exists to remove. Read
`NONE` and `BOUNDED` from that table; do not read `FULL` from it.

### The idea that survives, unproven

The mechanism is still sound in principle: a short form that stops matching the
full citation it plainly refers to is evidence the full citation was misparsed,
and a wrong page is otherwise invisible to any check that counts citations —
one identifier in, one identifier out, count unchanged. That is exactly why the
relaxation errors had to be found by reading results rather than by a metric.

But the example above does not demonstrate it, because that corruption was an
artefact of unpreprocessed text.

The test that would demonstrate it is `FULL`'s three **non-margin** errors,
which occurred in text the margin rule had already cleaned:

- `206 P. 327` → `206 P.3 27`
- `607 F.3d 355` → `130 S.Ct. 607`
- `Fed. R. Civ. P. 11(b)(2)` → `1 Fed.R. 1`

Two of the three destroy a full citation that had parsed correctly, so if any
short form in those filings refers back to it, the tree should strand it. That
has not been checked, and the documents are in the mined corpus rather than the
bench. Until it is, "the tree detects relaxation corruption" is a hypothesis.

Reproducing any of this needs both branches in one tree; they compose without
modification, and the combined test suites pass together.

## 6. What to build

1. **Wire the tree into extraction**, so a `ValidatedDocument` carries
   occurrences grouped under their authority rather than a flat list.
2. **Make each occurrence its own pinpoint claim.** The identity lookup happens
   once for the authority; the pinpoint check runs per occurrence, against that
   occurrence's own page and its own proposition.
3. **Measure what that adds.** How many additional checkable claims does a real
   filing contain once short forms are counted? On the corpus, how many of them
   fail? That number has not been measured and is a finding either way.

## 7. What conflicts with main, and what does not

This branch adds three files and modifies none:

    src/mellea_lrc/extraction/citation_tree.py
    tests/test_citation_tree.py
    exploration/notes/citation-tree-handoff.md

`citation_tree.py` imports only from `mellea_lrc.core.citations`. It does not
touch `extraction/__init__.py`, `pipeline.py`, `types.py` or
`eyecite_extractor.py` — the files `extraction/relaxation` rewrites. The two
branches overlap on no line, so the migration is a clean cherry-pick.

The original commit also touched
`src/mellea_lrc/validation/pinpoint_retrieval/reporter_page.py`, and **that file
has since been reworked on `main` by PR #80** (locator parsing and grounding).
Cherry-picking the original commits onto `main` conflicts there. This branch
therefore carries only the self-contained module and its tests; the pinpoint
side needs to be redone against the current `reporter_page.py` rather than
lifted. The original commits, if you want to read what was done there, are
`9ed7161` and `85e3310` on `experiment/general-explorations`.

## 8. What depends on this

The pinpoint work is blocked on the tree.
`exploration/notes/pinpoint-boundary-handoff.md` on
`pinpoint/reasoning-boundary` names it as a prerequisite: the tree is what
turns one authority into several checkable claims about several pages, which is
the unit that branch's per-occurrence checks operate on.

Section 6 item 1 is what unblocks it. It is worth doing before the measurement
in item 3, because another branch is waiting on the structure and not on the
number.

## 9. Standing constraints

Nothing is committed or pushed to `origin`; work goes to `woody-fork`. No
dataset is pushed anywhere — everything under `local/` is git-ignored.
