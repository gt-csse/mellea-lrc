"""Run the rule-based extractor over a bench and write artifacts for later stages.

This is extraction's own output and nothing else. **No adjudication runs**: no
candidate is proposed, no model is called, no locator is repaired. What comes out
is what the deterministic pass reads, which is the only thing that can be
measured against the bench annotations without measuring a reviewer as well.

One JSON file per document, written through
:func:`~mellea_lrc.serialization.serialize_extracted_document`, so a consumer
loads a real ``ExtractedDocument`` -- spans, citation objects, co-location and
authority ids -- rather than re-parsing text. Every artifact is read back and
compared with what produced it before the run is reported, because an artifact
that does not round-trip is worse than no artifact.

Spans index the document **body**: the plain-text loader splits the RECAP-style
header off and keeps it in ``source_metadata.header``, which is the same text the
bench annotations are anchored to.

    uv run python -m scripts.extract_bench
    uv run python -m scripts.extract_bench --relaxation bounded --out data/extraction-bounded
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from mellea_lrc.core.citations import citation_kind
from mellea_lrc.extraction import Relaxation
from mellea_lrc.extraction.pipeline import extract_from_raw_document
from mellea_lrc.extraction.types import ExtractedDocument
from mellea_lrc.serialization import deserialize_extracted_document, serialize_extracted_document
from mellea_lrc.serialization.extracted_document import SCHEMA_VERSION

BENCH = Path("data/false-citation-bench-v2.0/documents_txt")
OUT = Path("data/extraction-v2.0")
TREE = Path("data/false-citation-bench-tree-v2.0")
"""Citation-tree ground truth over the same text, used for the build report.

Optional. The artifact is the same either way -- this only lets the directory
say how the run scored, so a consumer does not have to find that out for
itself.
"""


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _commit() -> str | None:
    """The revision that produced these artifacts, when the tree is a checkout."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=BENCH, help="directory of .txt documents")
    parser.add_argument("--out", type=Path, default=OUT, help="directory to write artifacts into")
    parser.add_argument(
        "--relaxation",
        choices=[level.value for level in Relaxation],
        default=Relaxation.FULL.value,
        help="how much separator damage a citation may carry and still be read",
    )
    args = parser.parse_args()
    relaxation = Relaxation(args.relaxation)

    paths = sorted(args.bench.glob("*.txt"))
    if not paths:
        print(f"no documents in {args.bench}")
        return 1

    documents = args.out / "documents"
    documents.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    kinds: Counter[str] = Counter()
    docs_by_name: dict[str, ExtractedDocument] = {}
    for path in paths:
        # eyecite writes to stdout on some inputs; the artifact is the output.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_raw_document(path, relaxation=relaxation)
        payload = serialize_extracted_document(document)

        # An artifact nobody can load is not an artifact. Reading it back here
        # means the next stage's failure is its own, not this file's.
        recovered = deserialize_extracted_document(json.loads(json.dumps(payload)))
        if recovered != document:
            print(f"round-trip failed for {path.name}")
            return 1

        target = documents / f"{path.stem}.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        docs_by_name[path.name] = document
        counts = Counter(citation_kind(item.citation).value for item in document.citations)
        kinds.update(counts)
        entries.append(
            {
                "document": path.name,
                "artifact": str(target.relative_to(args.out)),
                "text_sha256_16": _digest(document.text),
                "text_length": len(document.text),
                "citations": len(document.citations),
                "by_kind": dict(sorted(counts.items())),
            }
        )

    manifest = {
        "artifact_type": "extraction_run",
        "schema_version": SCHEMA_VERSION,
        "source": str(args.bench),
        "relaxation": relaxation.value,
        # Stated rather than implied. A later stage reading these must know that
        # nothing here was proposed by a generator or confirmed by a reader.
        "adjudication": False,
        "produced_by": "scripts/extract_bench.py",
        "commit": _commit(),
        "eyecite_version": _package_version("eyecite"),
        "reporters_db_version": _package_version("reporters-db"),
        "documents": len(entries),
        "citations": sum(int(entry["citations"]) for entry in entries),
        "by_kind": dict(sorted(kinds.items())),
        "entries": entries,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = _build_report(manifest, docs_by_name)
    (args.out / "build-report.md").write_text(report, encoding="utf-8")

    print(f"{len(entries)} documents -> {args.out}")
    print(f"relaxation={relaxation.value}  adjudication=off  schema_version={SCHEMA_VERSION}")
    print(f"{manifest['citations']} citations")
    for kind, count in sorted(kinds.items(), key=lambda item: -item[1]):
        print(f"  {kind:<22}{count:>6}")
    return 0


def _score_against_tree(documents: dict[str, ExtractedDocument]) -> str:
    """How the run reads against the citation-tree ground truth, when it is there.

    Two questions, and they are separate. **Did extraction find the occurrence?**
    -- an annotated locator span matched exactly. **Did it attribute it to the
    right authority?** -- the citation its `authority_id` points at is the one
    the annotation says introduced the case. A layer that finds every citation
    and files half of them under the wrong case is not doing better than one
    that finds fewer.
    """
    occurrences = TREE / "occurrences.jsonl"
    authorities = TREE / "authorities.jsonl"
    if not (occurrences.exists() and authorities.exists()):
        return ""

    def _read(path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    roots = {record["authority_id"]: record["first_primary"]["span"] for record in _read(authorities)}
    scoped = [record for record in _read(occurrences) if record["scope"] == "in_scope"]

    found = attributed = agree = 0
    missed: list[str] = []
    for record in scoped:
        document = documents.get(str(record["document"]))
        if document is None:
            continue
        locator = (record["locator_span"]["start"], record["locator_span"]["end"])
        whole = (record["span"]["start"], record["span"]["end"])
        by_locator = {(c.locator_span.start, c.locator_span.end): c for c in document.citations}
        citation = by_locator.get(locator) or next(
            (c for c in document.citations if (c.full_span.start, c.full_span.end) == whole), None
        )
        if citation is None:
            missed.append(f"{record['document'][:14]} {record['matched_text']!r}")
            continue
        found += 1
        if record["authority_id"] is None:
            continue
        attributed += 1
        by_id = {c.citation_id: c for c in document.citations}
        root = by_id.get(citation.authority_id or "")
        wanted = roots[record["authority_id"]]
        if root is not None and (root.locator_span.start, root.locator_span.end) == (
            wanted["start"],
            wanted["end"],
        ):
            agree += 1

    lines = [
        "",
        "## Against the citation-tree ground truth",
        "",
        f"    in-scope occurrences        {len(scoped)}",
        f"      found by extraction       {found}",
        f"      of those, attributable    {attributed}   (the rest the annotation leaves open)",
        f"      attributed correctly      {agree}",
        "",
    ]
    if missed:
        lines.append("Not found:")
        lines.extend(f"    {item}" for item in missed)
        lines.append("")
    return "\n".join(lines)


def _build_report(manifest: dict[str, object], documents: dict[str, ExtractedDocument]) -> str:
    """A description of the directory, for whoever reads it next."""
    kinds = manifest["by_kind"]
    assert isinstance(kinds, dict)
    rows = "\n".join(f"    {kind:<22}{count:>6}" for kind, count in sorted(kinds.items()))
    return f"""# Extraction run over `{manifest["source"]}`

Rule-based extraction only. **No adjudication ran**: no candidate was proposed,
no model was called, no locator was repaired. Every citation here was read by the
deterministic pass, which is what makes these artifacts comparable with the bench
annotations without also measuring a reviewer.

    documents      {manifest["documents"]}
    citations      {manifest["citations"]}
    relaxation     {manifest["relaxation"]}
    eyecite        {manifest["eyecite_version"]}
    reporters-db   {manifest["reporters_db_version"]}
    commit         {manifest["commit"]}

{rows}

## What a file holds

`documents/<name>.json` is one serialized `ExtractedDocument` at schema version
{manifest["schema_version"]}. Load it with
`mellea_lrc.serialization.deserialize_extracted_document` and you get the object
extraction produced, not a re-parse: citation objects with `Reporter` and
`CitationDate` values, `full_span` and `locator_span`, `resolves_to`,
`authority_id` and `colocation_id`.

**Spans index `text`, which is the document body.** The plain-text loader splits
the RECAP-style header off and keeps it in `source_metadata.header`; the bench
annotations are anchored to the same body.

`manifest.json` lists every artifact with the SHA-256 prefix of the text it was
built from, so a consumer can tell whether the text under it has changed.
{_score_against_tree(documents)}"""


if __name__ == "__main__":
    raise SystemExit(main())
