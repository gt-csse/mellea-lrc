"""Tests for the extraction evaluator."""

import json
from pathlib import Path

import pytest

from evaluations.extraction.evaluate import evaluate

_DOC = "001__example.txt"


def _locator(start: int, end: int, volume: str, reporter: str, page: str, document: str = _DOC) -> dict:
    return {
        "document": document,
        "span": {"start": start, "end": end},
        "volume": volume,
        "reporter": reporter,
        "page": page,
    }


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_scores_agreement_on_identifier_and_overlap(tmp_path: Path) -> None:
    bench = _write(
        tmp_path / "bench.jsonl",
        [
            _locator(10, 22, "556", "U.S.", "662"),
            _locator(30, 41, "550", "U.S.", "544"),
            _locator(50, 62, "731", "P.2d", "902"),
        ],
    )
    artifact = _write(
        tmp_path / "run.jsonl",
        [_locator(10, 22, "556", "U.S.", "662"), _locator(30, 41, "550", "U.S.", "544")],
    )

    counts, disagreements = evaluate(bench, artifact)

    assert counts["expected"] == 3
    assert counts["true_positive"] == 2
    assert counts["false_negative"] == 1
    assert counts["false_positive"] == 0
    assert [d["reason"] for d in disagreements] == ["false_negative"]


def test_a_near_miss_span_still_matches(tmp_path: Path) -> None:
    """Where a citation's edges lie is convention once the identifier agrees."""
    bench = _write(tmp_path / "bench.jsonl", [_locator(10, 22, "556", "U.S.", "662")])
    artifact = _write(tmp_path / "run.jsonl", [_locator(10, 23, "556", "U.S.", "662")])

    counts, _ = evaluate(bench, artifact)

    assert counts["true_positive"] == 1
    assert counts["false_positive"] == 0


def test_a_disjoint_span_does_not_match(tmp_path: Path) -> None:
    """Overlap is required: the same authority cited elsewhere is another occurrence."""
    bench = _write(tmp_path / "bench.jsonl", [_locator(10, 22, "556", "U.S.", "662")])
    artifact = _write(tmp_path / "run.jsonl", [_locator(900, 912, "556", "U.S.", "662")])

    counts, _ = evaluate(bench, artifact)

    assert counts["true_positive"] == 0
    assert counts["false_negative"] == 1
    assert counts["false_positive"] == 1


def test_reporter_spelling_is_normalized(tmp_path: Path) -> None:
    """``F.Supp.2d`` and ``F. Supp. 2d`` name one reporter; damage is not scored."""
    bench = _write(tmp_path / "bench.jsonl", [_locator(10, 26, "798", "F. Supp. 2d", "1215")])
    artifact = _write(tmp_path / "run.jsonl", [_locator(10, 26, "798", "F.Supp.2d", "1215")])

    counts, _ = evaluate(bench, artifact)

    assert counts["true_positive"] == 1


def test_a_wrong_page_is_not_the_same_citation(tmp_path: Path) -> None:
    bench = _write(tmp_path / "bench.jsonl", [_locator(10, 22, "214", "F.3d", "1058")])
    artifact = _write(tmp_path / "run.jsonl", [_locator(10, 20, "214", "F.3d", "1")])

    counts, _ = evaluate(bench, artifact)

    assert counts["true_positive"] == 0
    assert counts["false_positive"] == 1
    assert counts["false_negative"] == 1


def test_one_benchmark_occurrence_is_claimed_once(tmp_path: Path) -> None:
    """Reporting one citation twice is not credited twice."""
    bench = _write(tmp_path / "bench.jsonl", [_locator(10, 22, "556", "U.S.", "662")])
    artifact = _write(
        tmp_path / "run.jsonl",
        [_locator(10, 22, "556", "U.S.", "662"), _locator(12, 20, "556", "U.S.", "662")],
    )

    counts, _ = evaluate(bench, artifact)

    assert counts["true_positive"] == 1
    assert counts["false_positive"] == 1


def _docket(start: int, end: int, text: str, **court: str) -> dict:
    return {"document": _DOC, "span": {"start": start, "end": end}, "matched_text": text, **court}


def _bench_docket(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "bench.jsonl",
        [_docket(10, 25, "No. 1:19-CV-362", court="M.D.N.C", court_id="ncmd")],
    )


def test_docket_numbers_ignore_a_leading_label(tmp_path: Path) -> None:
    """ "No.", "Case No." and the like name no part of the docket."""
    artifact = _write(tmp_path / "run.jsonl", [_docket(14, 25, "1:19-cv-362", court="M.D.N.C.")])

    counts, _ = evaluate(_bench_docket(tmp_path), artifact)

    assert counts["true_positive"] == 1


def test_a_docket_court_may_be_named_or_resolved(tmp_path: Path) -> None:
    """The citation string and the courts-db id are both acceptable."""
    for court in ({"court": "M.D.N.C."}, {"court_id": "ncmd"}):
        artifact = _write(tmp_path / "run.jsonl", [_docket(10, 25, "No. 1:19-CV-362", **court)])
        counts, _ = evaluate(_bench_docket(tmp_path), artifact)
        assert counts["true_positive"] == 1, court


def test_a_docket_in_the_wrong_court_is_a_different_case(tmp_path: Path) -> None:
    """The same number exists in many districts, so the court is part of the identity."""
    artifact = _write(
        tmp_path / "run.jsonl",
        [_docket(10, 25, "No. 1:19-CV-362", court="E.D.N.Y", court_id="nyed")],
    )

    counts, _ = evaluate(_bench_docket(tmp_path), artifact)

    assert counts["true_positive"] == 0
    assert counts["false_positive"] == 1
    assert counts["false_negative"] == 1


def test_a_docket_without_a_court_does_not_match(tmp_path: Path) -> None:
    """A docket number alone does not reach one case."""
    artifact = _write(tmp_path / "run.jsonl", [_docket(10, 25, "No. 1:19-CV-362")])

    counts, _ = evaluate(_bench_docket(tmp_path), artifact)

    assert counts["true_positive"] == 0
    assert counts["false_negative"] == 1


def test_the_same_identifier_in_different_documents_stays_distinct(tmp_path: Path) -> None:
    bench = _write(
        tmp_path / "bench.jsonl",
        [
            _locator(10, 22, "556", "U.S.", "662", document="001__a.txt"),
            _locator(10, 22, "556", "U.S.", "662", document="002__b.txt"),
        ],
    )
    artifact = _write(tmp_path / "run.jsonl", [_locator(10, 22, "556", "U.S.", "662", document="001__a.txt")])

    counts, _ = evaluate(bench, artifact)

    assert counts["true_positive"] == 1
    assert counts["false_negative"] == 1
    assert counts["false_positive"] == 0


def test_rejects_an_occurrence_with_no_identifier(tmp_path: Path) -> None:
    """A span alone says where something is, not which authority it names."""
    bench = _write(tmp_path / "bench.jsonl", [_locator(10, 22, "556", "U.S.", "662")])
    artifact = _write(tmp_path / "run.jsonl", [{"document": _DOC, "span": {"start": 10, "end": 22}}])

    with pytest.raises(ValueError, match="volume, reporter and page"):
        evaluate(bench, artifact)


def test_rejects_a_malformed_span(tmp_path: Path) -> None:
    bench = _write(tmp_path / "bench.jsonl", [_locator(10, 22, "556", "U.S.", "662")])
    artifact = _write(tmp_path / "run.jsonl", [_locator(22, 10, "556", "U.S.", "662")])

    with pytest.raises(ValueError, match="non-empty half-open span"):
        evaluate(bench, artifact)


def test_rejects_duplicate_occurrences(tmp_path: Path) -> None:
    bench = _write(tmp_path / "bench.jsonl", [_locator(10, 22, "556", "U.S.", "662")])
    artifact = _write(
        tmp_path / "run.jsonl",
        [_locator(10, 22, "556", "U.S.", "662"), _locator(10, 22, "556", "U.S.", "662")],
    )

    with pytest.raises(ValueError, match="duplicate occurrence"):
        evaluate(bench, artifact)
