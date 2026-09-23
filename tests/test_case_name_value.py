"""Case names retain their structure after normalization."""

import pytest

from mellea_lrc.model.citations.fields.case_name import CaseName, CaseNameKind


@pytest.mark.parametrize(
    ("quote", "kind", "plaintiff", "defendant", "subject", "citation"),
    [
        (
            "  Acme, Inc.\n v.  U.S. Dept. ",
            CaseNameKind.ADVERSARIAL,
            "Acme, Inc.",
            "U.S. Dept.",
            None,
            "Acme, Inc. v. U.S. Dept.",
        ),
        (
            "In  re   Motors Liquidation Co.",
            CaseNameKind.IN_RE,
            None,
            None,
            "Motors Liquidation Co.",
            "In re Motors Liquidation Co.",
        ),
        ("ex PARTE  Young", CaseNameKind.EX_PARTE, None, None, "Young", "Ex parte Young"),
    ],
)
def test_case_name_from_quote(
    quote: str,
    kind: CaseNameKind,
    plaintiff: str | None,
    defendant: str | None,
    subject: str | None,
    citation: str,
) -> None:
    name = CaseName.from_quote(quote)
    assert (name.kind, name.plaintiff, name.defendant, name.subject) == (
        kind,
        plaintiff,
        defendant,
        subject,
    )
    assert name.as_citation() == citation


@pytest.mark.parametrize(
    "quote",
    ["", "   ", "A v. ", " v. B", "A vs. B", "A v. B v. C", "A v. ,", "In re ", "Ex parte ", "A B"],
)
def test_case_name_rejects_unparsed_or_incomplete_quote(quote: str) -> None:
    with pytest.raises(ValueError):
        CaseName.from_quote(quote)


@pytest.mark.parametrize(
    "parts",
    [
        {"kind": CaseNameKind.ADVERSARIAL, "plaintiff": "A"},
        {"kind": CaseNameKind.ADVERSARIAL, "plaintiff": "A", "defendant": "B", "subject": "C"},
        {"kind": CaseNameKind.ADVERSARIAL, "plaintiff": " A", "defendant": "B"},
        {"kind": CaseNameKind.ADVERSARIAL, "plaintiff": "A v. B", "defendant": "C"},
        {"kind": CaseNameKind.IN_RE, "subject": " "},
        {"kind": CaseNameKind.IN_RE, "subject": "X", "plaintiff": "A"},
        {"kind": CaseNameKind.EX_PARTE, "subject": "X  Y"},
        {"kind": CaseNameKind.EX_PARTE, "defendant": "B"},
    ],
)
def test_case_name_model_rejects_invalid_shape_or_format(parts: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CaseName(**parts)
