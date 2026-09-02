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

## 5a. Every attribution, read by hand

All 70 attributed secondary citations on `locator-only-v2.0` were read against
the authority the tree gave them. **Sixty-eight are right.** Two are not, and
they are different failures.

**A misattribution, document 022.** The filing reads:

```
The cited case, Doe v. Commonwealth's Attorney, 403 F. Supp. 1199
(E.D. Va. 1975), is inapposite ... Moreover, Advanced Textile itself
granted anonymity for civil labor claims ... Id. at 1072-73.
```

The claim is about Advanced Textile, `214 F.3d 1058`; it lands on
`403 F. Supp. 1199`. Page 1072 is 127 *below* that case's first page and sits
inside Advanced Textile, whose other pin cites in this filing are 1068 and
1071-72.

**The cause is a missing reference, not a broken `Id.`** Between the two, the
prose names "Advanced Textile" — a party-name reference to `214 F.3d 1058` —
and eyecite extracts nothing there. Had that reference been produced it would
have been the nearest preceding citation and the chain would have been right.
`Id.` resolution is positional, so it silently absorbs the miss.

That miss looks systematic rather than incidental. The document names Advanced
Textile **eleven times**; the whole corpus holds **two** `ReferenceCitation`s,
neither in this document. The captured defendant is "Advanced Textile **Corp.**"
and the prose writes "Advanced Textile", which is the likely reason the
reference matcher never fires.

**A spurious reference, document 016.** `Chen Zhi 32` is a section heading
followed by a numbered paragraph — `D. The Arrest and Extradition of Chen Zhi`
then `32.` — read as a reference to the case's own defendant. It attaches to
the right authority, so it costs nothing here, but it is not a citation.

The 15 docket-rooted `Id.` chains were checked in document order rather than by
authority, because that is where an error would hide. The Indictment at 13237
owns the eleven that follow it, the Forfeiture Complaint at 16999 takes the
next seven, and the switch happens exactly where the second docket is
introduced.

### The detector, and what it cost to get right

`exploration/locator_recall/check_pin_range.py` finds this arithmetically: a
pin cite below the authority's first page cannot be a page of it. 36
occurrences carry a comparable pin cite and exactly one fails.

Two things had to be right, and both were wrong on the first attempt:

- **It must not run on Westlaw or LEXIS.** Their page is a document number and
  their pin cite is a star page; `2024 WL 1076736, at *6` is not page 6 of
  anything. Before that test it flagged 36 sound citations.
- **It must read the pin cite from the text, not from the parse.** On this very
  case eyecite records `pin_cite=None`, having discarded `at 1072 -73` while
  still making the attribution. A check on the parser's own output cannot see
  the parser's own mistake.

### Backward re-checking, proposed and not built

The detector says an attribution is impossible. It does not say what the right
answer is, and the tree currently has no way to ask.

**When an `Id.` attribution fails a check like this, re-read the context and
look for the antecedent that fits.** Here the answer is present and cheap: a
case is named by party in the intervening prose, another authority in the same
filing has a page range that contains 1072, and both point at `214 F.3d 1058`.
A backward pass over the preceding text — party names, and authorities whose
page range admits the pin cite — would recover it.

This is worth doing carefully rather than soon. Rewriting an attribution on a
heuristic risks turning one wrong answer into a different wrong answer, and the
failure is rare: one in seventy here. What makes it worth building anyway is
that the *mechanism* is ordinary — a brief discussing two cases in succession
and returning to the first by name is normal legal writing — so the rate on
this corpus is probably not the rate everywhere. Measure it on a larger corpus
before deciding how hard to try.

The safe intermediate is to report rather than repair: mark the occurrence as
suspect, leave it attributed where it is, and let a consumer decline to make a
pinpoint claim on it. A wrong page sent to verification returns a confident
verdict about the wrong case, which is the failure this project exists to
prevent.

### Reassignment over a closed set

A design for the repair, and it is better than the heuristic pass above.

**Attach every occurrence to the first primary.** An authority is introduced
once, by the first full citation that names it; everything after — a repeated
full citation, a short form, an `id.`, a party reference — is an occurrence
hanging off that one. The tree already has this shape: `Authority.root` *is*
the first primary and `Authority.occurrences` *is* the list. Nothing new has to
be built to hold it.

**Then a failed attribution becomes a choice, not a search.** When the pin-cite
range check rejects an `Id.`, the question put to a model is not "what does this
refer to" but "which of these authorities does this refer to", over the
first primaries the document has already established. It reads the context and
picks one, or declines.

That is the contract `adjudicate_docket` already uses: courts written near a
docket are resolved against courts-db and offered as a closed set, so the model
picks or declines and cannot invent a court. The same shape applies here, and
it inherits the same property — **the model never decides what is in the
document, only which of the things already in it a reference points at.**

**And it avoids the trap the other route walks into.** Recovering the document
022 case by extracting name references would mean treating `Advanced Textile` in
running prose as a citation. It carries no pin cite; eyecite is right not to
make one. Sometimes a name is just a name — a party discussed, a company in the
facts, a person — and a rule that turns prose names into citations buys this one
attribution at the cost of a false-positive class with no natural bound.

Reassignment needs none of that. The prose mentioning "Advanced Textile" is
*context for the model to read*, not a citation to be extracted. The candidate
set comes from citations the document undeniably makes.

Three properties worth stating before anyone builds it:

- **It is invoked on failure, so the cost is bounded.** One call per rejected
  attribution, and the corpus has one.
- **It is checkable.** The answer must be one of the offered first primaries,
  and the pin cite must fall within that authority's page range — the same
  arithmetic that raised the flag can confirm the repair.
- **Declining must be a real option.** An `Id.` whose antecedent is out of
  scope — the opposing brief, the filing's own complaint — has no right answer
  in the candidate set, and the model must be able to say so rather than pick
  the nearest case.

## 6. What to build

1. **Wire the tree into extraction**, so a `ValidatedDocument` carries
   occurrences grouped under their authority rather than a flat list.
2. **Make each occurrence its own pinpoint claim.** The identity lookup happens
   once for the authority; the pinpoint check runs per occurrence, against that
   occurrence's own page and its own proposition.
3. **Measure what that adds.** How many additional checkable claims does a real
   filing contain once short forms are counted? On the corpus, how many of them
   fail? That number has not been measured and is a finding either way.
4. **Extract party-name references properly.** Section 5a: eleven mentions of
   "Advanced Textile" in one filing produce no `ReferenceCitation`, and the
   corpus holds two in total. That gap is what makes `Id.` fall through, so it
   is upstream of the misattribution and probably of others not yet found.
5. **Then backward re-checking as reassignment over a closed set**, on the
   terms in section 5a. Report the suspect attribution first; repair it by
   asking a model to choose among the first primaries the document already
   establishes, with declining allowed. That needs no name references to be
   invented, which is what makes it safe.

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
