# Proofreading `data/false-citation-bench-plus-pincite-v2.0`

Eleven filings, 62 annotated defects, 97 spans, ten files in `orders_txt/`. Every
annotation was read against the document it points at and against the file the
`order` field names.

## What holds

The mechanical layer is sound and needs no work.

    annotation files                11   = manifest `filings`
    entries                         62   = manifest `citations`
    spans                           97   = manifest `spans_verified`
    documents_txt[start:end]==term  97/97, 0 mismatches
    per-document counts             agree with the manifest everywhere
    kind census                     agrees with `citations_by_kind`

`reporter_citation` also agrees with the span text everywhere a span carries a
digit: 72 of 72 compare equal once punctuation and spacing are removed.

Two filings are complete and accurately quoted: `72299304_24_1` (9 entries, 9
table rows in the order) and `72299304_28` (7 entries, 7 rows). `70607460_15`
records all six findings in the order's footnote 10.

## 1. Three filings have no court order behind them

This is the finding that matters, because the README's first sentence about the
data is that `orders_txt/` holds "the court orders: the ground truth", and its
first check is "Does the order say it? `ruling_evidence` should be the court's
words."

| file | what it actually is |
|---|---|
| `62980057_439813347` | **Plaintiff Frankie Johnson's Response** to the motion (Doc 186), signed by plaintiff's counsel |
| `69412014_446417376` | **A Rule 11 letter from opposing counsel**, filed as "Exhibit A" (Doc 57-2) |
| `69713591_440567315` | **Plaintiffs' Notification of Non-Existent Legal Authority and Rule 12(f) Motion to Strike** (Doc 24) |

None is signed by a judge and none adjudicates anything. **9 of the 62 entries**
— every entry for `62980057_174`, `69412014_46` and `69713591_23` — rest on an
adversary's accusation rather than a court's finding. That is not a weaker
version of the same ground truth; it is a different kind of claim, and a bench
built on it measures whether a party alleged a defect, not whether one was found.

A fourth is a partial case. `68658788_462710593` is Doc 79-**1**, an exhibit: a
three-column table headed *Fabricated Case / Plaintiff's Use / Court's Research*.
The third column is written in the court's voice ("the Court cannot find", "as
cited by Billups"), so it is very likely the court's own work product — but the
order it was attached to is not in the corpus, and nothing in the dataset says
which document Doc 79 is.

## 2. `ruling_evidence` is composed, not quoted

Of 62 evidence strings, **9 appear verbatim** in the file the `order` field
names. The rest are edited, and the editing is unmarked. Only one entry
(`67673016_163` #1) uses an ellipsis where it elides.

Four patterns, all present more than once:

*   **A "See" string is pruned to the one case that matters.** `70495110_84` #3
    quotes the order as `... See Zetwick v. County of Yolo, 66 F. Supp. 3d 1274
    (E.D. Cal. 2014) (spanning reporter pages 1274 to 1287).` The order reads
    `See Hua Yao Yang v. U.S. Att'y Gen., 538 F. App'x 873 (11th Cir. 2013)
    (denying a petition ... ); Zetwick v. County of Yolo, 66 F. Supp. 3d 1274
    (E.D. Cal. 2014) (granting summary judgment in a sexual harassment lawsuit
    and spanning reporter pages 1274 to 1287), rev'd and remanded, ...`. A whole
    citation and half a parenthetical are gone with no ellipsis. Same in #6
    (Phillips: `Lowe Inv. Corp.` dropped), #7 (Brydger: `Whitley` dropped) and #1
    (Sun Healthcare: every court-and-date parenthetical dropped, and "spanning
    **reporter** pages" silently shortened to "spanning pages").
*   **Non-contiguous sentences are joined as continuous prose.** All six
    `70607460_15` entries open with the same two-sentence preamble; in the order
    the first sentence is body text on one page and the second is the last
    sentence of footnote 10.
*   **Sentences are reworded.** `70607460_15` #1 gives `That quotation does not
    appear in the opinion.`; the order says `But it does not appear in the
    opinion.` #3 appends `in that opinion` to a sentence that ends at `does not
    exist`. `70495110_84` #11 turns the order's `despite Donaldson explicitly
    stating that ...` into `Donaldson explicitly states that ...` and appends a
    pin cite the sentence does not carry.
*   **Paraphrase, which the README forbids outright.** `15663311_149` #2 gives
    `the Westlaw citation leads elsewhere`. The order says `the Westlaw citation
    provided is not of any case, and is instead a Department of Defense Language
    Testing Program`. `62980057_174` #2 gives `But the so-cited case does not
    exist`, a sentence that appears nowhere in its source file.

One quotation is silently corrected: `67673016_163` #2 writes `FMLA` where the
order has the typo `FLMA`.

**The substance is right in every case checked.** The court (or the party) did
find the defect the entry describes. What fails is locatability: a consumer
cannot find `ruling_evidence` in the file the annotation names, which is the one
property that makes the evidence field checkable rather than trusted.

## 3. Five filings record only some of the defects the order names

Every unrecorded defect below was confirmed present in the filing's own
`documents_txt`, so none of these is a "could not locate" case. `located` equals
`citation_count` in all eleven, so nothing in the data reports the gap.

| filing | named in the source, in this filing | recorded | not recorded |
|---|---:|---:|---|
| `15663311_149` | 6 | 2 | Leary v. Daeschner (fabricated quote), Yeschick v. Mineta, Century Prod. (non-existent), In re Nat'l Prescription Opiate |
| `62980057_174` | 4 | 2 | Greer v. Warden (2020 WL 3060362), Wilson v. Jackson (2006 WL 8438651) |
| `67673016_163` | 7 | 4 | Boumehdi, Stutler, Green v. Brennan |
| `68658788_65` | 4 | 2 | Jackson v. Cal-W. Packaging Corp., Etienne v. Spanish Lake |
| `70495110_84` | 12 | 11 | United States v. Valencia-Trujillo, 573 F.3d 1171 (the order's section A) |

`15663311_149` is the sharpest: the order lists seven problematic citations, six
of them in this filing, and the annotation records two. Excluding the seventh
(Inge) is *correct* — the order places it in Hild's reply brief, a different
document, and it appears here only inside another case's parenthetical.

## 4. One kind assignment contradicts the dataset's own rule

The README: "`wrong_pincite` means the case exists and the page is wrong or
absent. A case that does not exist is `non_existent` even when the court also
mentions a page."

`62980057_174` records the same citation twice: #0 `non_existent` (the case at
`539 F. App'x 937` is Williams v. Morahan, not United States v. Baker) and #1
`wrong_pincite` (`The pincite of page 943 ... does not exist`). By the stated
rule the second should not be a separate `wrong_pincite`.

Two other pairs are *not* violations and should stay: `15663311_149` #0/#1 and
`68658788_65` #0/#1 record two genuinely different defects in one citation — a
holding that is not there and a page that is not there.

Two kinds are stretched rather than wrong. `70495110_84` #5 files "relies on a
pleading standard Twombly overruled" as `misrepresented_holding`; superseded
authority is not one of the four kinds. `70607460_15` #5 files Pape as
`wrong_pincite` on an order that says only `the citation ("231 Kan. 595") is not
correct`, which does not say the page rather than the case is wrong.

## 5. `69412014_46` #2 records a citation the document does not contain

`reporter_citation` is `2006 WL 678577`. Neither that string nor `678577`
appears anywhere in `documents_txt/69412014_46.txt`. The filing prints
`Blackwell v. Eskin, 916 A.2d 1123` three times and `Blackwell v. Eskin, 2006
Phila. Ct. Com. Pl. LEXIS 125` once. The README defines `reporter_citation` as
"the citation as the filing prints it", so this is a field taken from the source
letter rather than from the document the spans index.

## 6. Twenty-five spans land on a case name, not a citation

Five entries span party names: `68658788_65` #2 and #3 (eight `Jackson v.
Gautreaux` each), `69412014_46` #1 and #2, `69713591_23` #2, `70607460_15` #3.
They pass the build's check, because that compares `term` to the text and `term`
is itself the case name — but the README states the check as "Do the spans land
on the citation?", and a party name is not one. For `68658788_65` this is
defensible (the court's finding is about eight short-form references), and for
`70607460_15` #3 the filing gives Cochrane no reporter citation at all. For
`69412014_46` #2 it compounds finding 5: three of the four spans sit at a
citation (`916 A.2d 1123`) different from the one recorded.

## 7. The manifest's margin field is the wrong type and contradicts the README

`preprocessing.margin_line_numbers_dropped` is `true` for all eleven documents.
The project's own `PreprocessingMetadata` declares it `int | None` — a **count**,
where `None` means the rule did not run and is deliberately not the same as zero.
A boolean is neither, and read as a count `true` is 1.

The README says "The margin rule removed nothing from any of the eleven", which
would be `0`. One of the two is wrong and the data cannot say which.

## 8. Smaller things

*   `15663311_149` has `court_id: ""` and `docket_number: ""` in both the
    annotation and the manifest. The order states them: E.D. Mich.,
    No. 19-cv-11512.
*   `manifest.json` asserts `"ground_truth": "the court order named every defect
    recorded here"`. Findings 1 and 3 both contradict it.
*   Nine of the ten `orders_txt/` files carry no `--- Plain text ---` marker,
    unlike `documents_txt/`. Not wrong, but a reader written against one will
    not work on the other.

## What to do with it

The dataset is usable for **extraction** work as it stands: the text is v2.0, the
spans verify exactly, and 62 real citations with hand-checked offsets is worth
having. Nothing in findings 1–4 touches whether a span is where it says it is.

It is not yet usable as **validation ground truth**. Findings 1 and 3 are the
blockers — 9 entries are an adversary's allegation rather than an adjudication,
and five filings record between a third and a half of what their source names, so
neither precision nor recall against it means what it appears to mean. Finding 2
makes every entry unverifiable without re-reading the source by hand, which is
what this pass had to do.

The cheapest repair order: replace or relabel the three party filings; add a
`source_type` field so the distinction cannot be lost again; requote every
`ruling_evidence` from its source with ellipses where text is elided; record the
unrecorded defects, or add a field saying the annotation is a subset and of what.
