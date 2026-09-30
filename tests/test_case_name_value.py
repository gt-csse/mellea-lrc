"""Case names retain complete forms and grounded partial forms."""

import pytest

from mellea_lrc.model import Span
from mellea_lrc.model.citations.fields.case_name import CaseName, CaseNameField, CaseNameKind


@pytest.mark.parametrize(
    ("quote", "kind", "plaintiff", "defendant", "subject", "partial", "citation"),
    [
        (
            "  Acme, Inc.\n v.  U.S. Dept. ",
            CaseNameKind.ADVERSARIAL,
            "Acme, Inc.",
            "U.S. Dept.",
            None,
            None,
            "Acme, Inc. v. U.S. Dept.",
        ),
        (
            "In  re   Motors Liquidation Co.",
            CaseNameKind.IN_RE,
            None,
            None,
            "Motors Liquidation Co.",
            None,
            "In re Motors Liquidation Co.",
        ),
        ("ex PARTE  Young", CaseNameKind.EX_PARTE, None, None, "Young", None, "Ex parte Young"),
        ("Matter of M4 Enters., Inc.", CaseNameKind.IN_RE, None, None, "M4 Enters., Inc.", None, "In re M4 Enters., Inc."),
        ("Troxel v.Granville", CaseNameKind.ADVERSARIAL, "Troxel", "Granville", None, None, "Troxel v. Granville"),
        ("Ridgewood Bd. V. Zebra", CaseNameKind.ADVERSARIAL, "Ridgewood Bd.", "Zebra", None, None, "Ridgewood Bd. v. Zebra"),
        ("Gucci  America", CaseNameKind.PARTIAL, None, None, None, "Gucci America", "Gucci America"),
    ],
)
def test_case_name_from_quote(
    quote: str,
    kind: CaseNameKind,
    plaintiff: str | None,
    defendant: str | None,
    subject: str | None,
    partial: str | None,
    citation: str,
) -> None:
    name = CaseName.from_quote(quote)
    assert (name.kind, name.plaintiff, name.defendant, name.subject, name.partial) == (
        kind,
        plaintiff,
        defendant,
        subject,
        partial,
    )
    assert name.as_citation() == citation


@pytest.mark.parametrize(
    "quote",
    ["", "   ", "A v. ", " v. B", "A v. B v. C", "A v. ,", "In re ", "Ex parte ", "A v."],
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
        {"kind": CaseNameKind.PARTIAL, "partial": " "},
        {"kind": CaseNameKind.PARTIAL, "partial": "One", "plaintiff": "Another"},
        {"kind": CaseNameKind.PARTIAL, "partial": "One v. Another"},
        {"kind": CaseNameKind.PARTIAL, "partial": "In re One"},
        {"kind": CaseNameKind.ADVERSARIAL, "plaintiff": "A", "defendant": "B", "partial": "A"},
    ],
)
def test_case_name_model_rejects_invalid_shape_or_format(parts: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CaseName(**parts)


def test_not_stated_is_a_typed_outcome_without_citation_text() -> None:
    name = CaseName(kind=CaseNameKind.NOT_STATED)
    assert name.model_dump(mode="json") == {
        "kind": "not_stated",
        "plaintiff": None,
        "defendant": None,
        "subject": None,
        "partial": None,
    }
    with pytest.raises(ValueError, match="no citation text"):
        name.as_citation()
    with pytest.raises(ValueError, match="no printed parts"):
        CaseName(kind=CaseNameKind.NOT_STATED, partial="Smith")
    with pytest.raises(ValueError, match="cannot be a quoted field"):
        CaseNameField.from_model("Smith", Span(0, 5), name, node_id="cite:node:0")
