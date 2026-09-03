r"""Scope for the 154 unattributed occurrences that carried no evidence of their own.

Read one by one against the text, in the order
`exploration.generality.read_unattributed` prints them. The question for each is
the only one that matters here: **does this refer to a court case?** Not which
case -- whether there is a case at all. An `Id.` pointing at a declaration, a
transcript, an opposition brief or a book is not a citation this project checks,
so it belongs outside the denominator rather than inside the numerator.

The other 98 unattributed occurrences are not listed: a paragraph or
page-and-line pin cite is evidence of its own and they were classified by it.

## What they turned out to be

**Out of scope, 97.** Overwhelmingly the record. Two sanctions proceedings
account for 61 of them on their own -- `( Id. )` after `(Doc. 17.)`, meaning
counsel's response and its exhibits -- and one First Amendment brief runs 28
`Id.` off expert declarations and the complaint. The rest are opposition briefs
cited as `Opp. 21`, a book the brief quotes, and `supra` pointing at the filing's
own argument sections rather than at any authority.

**In scope, 49.** Real case references the resolver could not place: short forms
whose full citation is elsewhere in the document (`833 F.2d at 186`,
`556 U.S. at 678`, `33 F.4th at 695`), and `Id.` chains discussing a case
(`Id. at *15`, `Id., 153 P.3d at 1240`). One is a case identified only by docket,
which has no reporter authority to attach to.

**Uncertain, 8.** Kept separate rather than forced. Three kinds: a case citation
quoted from inside another party's brief, where whether the filing cites it or
merely reproduces it is a judgement; `Id., p. 1 & n.1`, where `p.` suggests a
document page but could be a case; and two `id.` in a brief that is itself about
citations, where the referent may be the case or the filing that miscited it.
"""

from __future__ import annotations

# Indexes as `read_unattributed` prints them.
IN_SCOPE = frozenset(
    {
        *range(2, 6),  # 69197386_173, short forms of DCD Programs and Webb
        37,  # 89 F.4th at 1068, Animal Legal Defense Fund
        *range(38, 43),  # the `Id.` chain discussing it
        44,
        45,  # 69206960_30, short forms
        *range(89, 92),  # 69694686_18, a chain about another sanctions case
        94,  # 463 F. Supp. 3d at 1073
        104,  # 44 F.3d at 1265
        105,  # 556 U.S. at 678
        106,  # Arizona Student Doe, a case identified only by docket
        *range(117, 120),  # 69979017_207, Rule 5(b)(2)(E) consent
        122,  # 473 F. Supp. 3d at 154
        *range(123, 126),  # 70583449_97, the same passage in another filing
        *range(129, 133),  # 70607460_15, the Voorhees chain
        *range(134, 144),  # 70764936_25, Twombly, Iqbal and the Bates chain
        *range(145, 149),  # (144 sits in a scrambled passage; see UNCERTAIN)
        150,  # 376 U.S. at 279
        151,
        152,  # 71920595_40
        153,
        154,  # 72050145_17, supra to cases cited earlier
    }
)

UNCERTAIN = frozenset(
    {
        107,  # `see also id. (listing non-existent case ...)` -- the brief, or the case?
        108,  # a safe-harbor proposition; reads like a case, sits among filings
        109,  # `See id. (describing Ahanchian)`
        120,
        121,  # case citations quoted from inside Def.'s Objection
        128,  # `Id., p. 1 & n.1` -- a document page, or a case at page 1
        133,  # a citation quoted from Doc. 40 as the subject of discussion
        144,  # inside a passage extraction scrambled
    }
)

TOTAL = 154
OUT_OF_SCOPE = frozenset(range(1, TOTAL + 1)) - IN_SCOPE - UNCERTAIN

# The 98 not listed here, classified by their own pin cite.
WITH_RECORD_EVIDENCE = 98
