"""Re-anchor the hand annotations onto v2.0 text and publish them as two labels.

The 26 filings were annotated citation by citation against the sanctioning
court's own words. Those annotations were offsets into the **v1** rendering, and
verified against nothing else -- 79 of 79 against v1, 0 of 79 against every later
rendering. This moves them onto the v2.0 text the extraction bench uses, so one
corpus carries both stages' ground truth in one coordinate space.

**Anchoring is diff-based, not a search.** `SpanUpdater` maps each v1 offset into
v2.0, and the annotation's own text is then matched near that point. 50 of the 79
strings occur once in the document and need no disambiguation; the other 27 occur
two to five times, and every one of them resolved to a candidate the mapping had
already landed on -- distance zero -- so none is a guess. Two differ between
renderings by a single optically confused character and were read by hand.

**Two labels, and they are disjoint.**

    WRONG_IDENTITY   the locator identifies no case, or identifies a case whose
                     fields disagree with the filing, or a different case sits
                     at the page
    WRONG_PINCITE    the locator identifies one case and the fields agree, so
                     identity is sound and what is wrong is what the case is
                     cited for

A refuted identity carries no pincite label: with no case at the locator there is
nothing for a proposition to be checked against. Where a court found both, the
entry is `WRONG_IDENTITY` and the pincite finding is kept as evidence with a
`comment` saying why it is not a second label.

**One cited authority in one filing is one data point**, and an authority is its
**first occurrence** -- the full citation that introduced the case, which is what
`authorities.jsonl` records as `first_primary`. Three annotations name a case the
filing never cites, giving only a court and a year; they are their own data
points with no authority, because inheriting a neighbour's would file a finding
under the wrong case.

    uv run python -m scripts.build_validation_bench
"""

from __future__ import annotations

import argparse
import json
import re
from bisect import bisect_right
from collections import Counter
from pathlib import Path

from eyecite.annotate import SpanUpdater

V1 = Path("data/false-citation-bench")
V20 = Path("data/extraction-v2.0")
OUT = Path("data/validation-v2.0")

NAME_TO_CITE = 60
"""How far after a case name its citation may begin and still be the same one."""

WRONG_IDENTITY = "WRONG_IDENTITY"
WRONG_PINCITE = "WRONG_PINCITE"
LABEL = {"unverifiable_authority": WRONG_IDENTITY, "misrepresented_authority": WRONG_PINCITE}

BY_HAND = {
    # One optically confused character apart between the two renderings, read
    # from both files. v2.0 is right about the first and wrong about the second,
    # and the dataset records what the document says either way.
    "11-1-alford-v-motors-ins-corp": "Alford v. Motors Ins. Corp., 104 N.C. App. 537 (1991)",
    "11-6-kusulas-v-geico": "Kusulas v. GE/CO",
}

CORRECTIONS: dict[str, tuple[str | None, str]] = {
    # From `false-citation-bench/audit-identity-vs-misrepresentation.md`, which
    # checked every label against the classification rule. A label the rule
    # contradicts is corrected here rather than carried forward.
    "23-1-in-re-soundview-elite-ltd": (
        WRONG_PINCITE,
        "503 B.R. 571 is that case, decided 2014-01-23 in that court, so its identity is sound. "
        "The court's correction points to 543 B.R. 78 (2016), a different Soundview opinion, for "
        "the proposition.",
    ),
    "9-3-roe-v-bernabei-katz-pllc": (
        None,
        "Dropped: the page holds United States v. Abu Khatallah, so there is nothing for a "
        "proposition to be checked against and the identity finding is the whole of it.",
    ),
}

UNREACHABLE = {
    # Named with no locator the filing states. They must not inherit a
    # neighbour's authority: doing so files a court's finding under a case it
    # was not about.
    "11-6-kusulas-v-geico": "The filing states no locator at all, only a name in prose.",
    "14-2-akiachak-native-community-v-united-states-department-of": ("Name and court, no reporter citation."),
    "14-3-chugach-natives-inc-v-doyon-ltd": "Name and court, no reporter citation.",
}


def _annotations() -> list[dict]:
    return [p for p in sorted((V1 / "annotations").glob("*.json")) if p.name != "manifest.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    authorities = {
        a["authority_id"]: a
        for a in (json.loads(x) for x in (V20 / "authorities.jsonl").read_text().splitlines() if x.strip())
    }
    occurrences: dict[str, list[dict]] = {}
    for line in (V20 / "occurrences.jsonl").read_text().splitlines():
        if line.strip():
            o = json.loads(line)
            occurrences.setdefault(o["document"], []).append(o)

    counts: Counter = Counter()
    findings: list[dict] = []
    for p in _annotations():
        document = p.stem + ".txt"
        src = (V1 / "documents_txt" / (p.stem + ".txt")).read_text(encoding="utf-8", errors="replace")
        dst = (V20 / "documents_txt" / (p.stem + ".txt")).read_text(encoding="utf-8", errors="replace")
        updater = SpanUpdater(src, dst)
        for ann in json.load(open(p))["annotations"]:
            label, note = CORRECTIONS.get(ann["id"], (LABEL[ann["type"]], ""))
            if label is None:
                counts["dropped by the audit"] += 1
                continue
            if ann["id"] in CORRECTIONS:
                counts["label corrected by the audit"] += 1

            needle = BY_HAND.get(ann["id"], ann["citation_span"]["text"])
            hint = updater.update(ann["citation_span"]["start"], bisect_right)
            pattern = re.compile(r"\s+".join(re.escape(word) for word in needle.split()))
            candidates = [m.span() for m in pattern.finditer(dst)]
            if not candidates:
                msg = f"{ann['id']} does not resolve in {document}"
                raise SystemExit(msg)
            start, end = min(candidates, key=lambda c: abs(c[0] - hint))
            counts["anchored uniquely" if len(candidates) == 1 else "anchored by the mapped offset"] += 1

            authority_id = None
            if ann["id"] not in UNREACHABLE:
                here = occurrences.get(document, [])
                # The annotation usually spans the case name, and a citation's
                # full span does not always reach back over it -- a short form's
                # does not. So containment first, then the citation that begins
                # just after the name. The second test is what tells a case the
                # filing cites from one it only names: the three that have no
                # citation within reach are the three the audit calls
                # unreachable, found independently. A docket citation carries no
                # authority in the tree, so it lands here with none, which is
                # what the audit says about it.
                inside = [o for o in here if o["span"]["start"] <= start and end <= o["span"]["end"]]
                # The annotation is sometimes the wider of the two, quoting the
                # name and the citation together, so the locator sits inside it.
                holds = [o for o in here if start <= o["locator_span"]["start"] < end]
                adjacent = [o for o in here if 0 <= o["locator_span"]["start"] - end <= NAME_TO_CITE]
                chosen = inside or holds or adjacent
                if chosen:
                    authority_id = min(chosen, key=lambda o: o["span"]["end"] - o["span"]["start"])[
                        "authority_id"
                    ]
            findings.append(
                {
                    "document": document,
                    "annotation_id": ann["id"],
                    "label": label,
                    "authority_id": authority_id,
                    "span": {"start": start, "end": end, "text": dst[start:end]},
                    "cited_authority": ann.get("cited_authority"),
                    "reporter_citation": ann.get("reporter_citation"),
                    "proposition": (ann.get("proposition_span") or {}).get("text"),
                    "ruling_evidence": ann.get("reason"),
                    "note": note or UNREACHABLE.get(ann["id"], ""),
                }
            )

    grouped: dict[tuple[str, str], list[dict]] = {}
    for f in findings:
        key = (f["document"], f["authority_id"] or f"name:{f['annotation_id']}")
        grouped.setdefault(key, []).append(f)

    entries = []
    for (document, key), group in sorted(grouped.items()):
        labels = {f["label"] for f in group}
        label = WRONG_IDENTITY if WRONG_IDENTITY in labels else WRONG_PINCITE
        counts[label] += 1
        authority = authorities.get(group[0]["authority_id"] or "")
        comment = " ".join(f["note"] for f in group if f["note"]).strip()
        if len(labels) > 1:
            counts["identity refuted a citation also faulted on its pincite"] += 1
            comment = (
                "The court also faulted this citation on its pincite. No pincite label is "
                "recorded: the case is not what the locator identifies, so nothing it is cited "
                "for can be checked. The finding is kept as evidence below. " + comment
            ).strip()
        entries.append(
            {
                "document": document,
                "label": label,
                "cited_authority": group[0]["cited_authority"],
                "reporter_citation": group[0]["reporter_citation"],
                # The authority is its first occurrence: the full citation that
                # introduced the case in this filing.
                "authority": (
                    {
                        "citation": authority["first_primary"]["matched_text"],
                        "span": authority["first_primary"]["span"],
                    }
                    if authority
                    else None
                ),
                "flagged_spans": [f["span"] for f in group],
                "comment": comment,
                "evidence": [
                    {
                        "label": f["label"],
                        "proposition": f["proposition"] or "",
                        "ruling_evidence": f["ruling_evidence"] or "",
                    }
                    for f in group
                ],
            }
        )

    dataset = {
        "name": "validation-v2.0",
        "schema": "mellea-lrc/false-citation/two-label/v1",
        "unit": "one cited authority in one filing, the authority being its first occurrence",
        "labels": [WRONG_IDENTITY, WRONG_PINCITE],
        "text": "documents_txt is `../extraction-v2.0/documents_txt`; spans index it directly",
        "document_count": len({e["document"] for e in entries}),
        "entry_count": len(entries),
        "label_counts": {
            WRONG_IDENTITY: sum(1 for e in entries if e["label"] == WRONG_IDENTITY),
            WRONG_PINCITE: sum(1 for e in entries if e["label"] == WRONG_PINCITE),
        },
        "entries": entries,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "annotations.json").write_text(
        json.dumps(dataset, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(findings)} findings -> {len(entries)} entries at {args.out}/annotations.json")
    for label, count in sorted(counts.items(), key=lambda i: -i[1]):
        print(f"  {label:<52}{count:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
