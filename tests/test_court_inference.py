import pytest
from eyecite import get_citations

from mellea_lrc.assessment.fields.court import get_reporter_inference
from mellea_lrc.reporter_jurisdiction.types import ReporterJurisdictionStatus


@pytest.mark.parametrize(
    ("reporter", "court"),
    (
        ("U.S.", "scotus"),
        ("S. Ct.", "scotus"),
        ("L. Ed.", "scotus"),
        ("L. Ed. 2d", "scotus"),
        ("U.S. LEXIS", "scotus"),
        ("T.C.", "tax"),
        ("B.T.A.", "bta"),
        ("Fed. Cl.", "uscfc"),
        ("Cl. Ct.", "uscfc"),
        ("Ct. Int'l Trade", "cit"),
        ("Cust. Ct.", "cusc"),
        ("C.C.P.A.", "ccpa"),
        ("Vet. App.", "cavc"),
        ("C.M.A.", "cma"),
    ),
)
def test_infer_court_from_exclusive_reporter(reporter: str, court: str) -> None:
    inference = get_reporter_inference(reporter)
    assert inference.status is ReporterJurisdictionStatus.EXHAUSTIVE_REPORTER
    assert inference.exact_court_id == court


@pytest.mark.parametrize(
    ("text", "reporter"),
    (
        ("1 U.S. LEXIS 1 (1980)", "U.S. LEXIS"),
        ("100 T.C. 1 (1993)", "T.C."),
        ("1 B.T.A. 1 (1924)", "B.T.A."),
        ("100 Fed. Cl. 1 (2011)", "Fed. Cl."),
        ("1 Cl. Ct. 1 (1983)", "Cl. Ct."),
        ("1 Ct. Int'l Trade 1 (1980)", "Ct. Int'l Trade"),
        ("1 Cust. Ct. 1 (1938)", "Cust. Ct."),
        ("1 C.C.P.A. 1 (1930)", "C.C.P.A."),
        ("1 Vet. App. 1 (1990)", "Vet. App."),
        ("1 M.S.P.R. 1 (1979)", "M.S.P.R."),
        ("1 C.M.A. 1 (1951)", "C.M.A."),
    ),
)
def test_eyecite_does_not_infer_exclusive_reporter_court(
    text: str,
    reporter: str,
) -> None:
    citations = get_citations(text)

    assert len(citations) == 1
    assert citations[0].groups["reporter"] == reporter
    assert citations[0].metadata.court is None


@pytest.mark.parametrize(
    "reporter",
    (None, "", "F.3d", "F. Supp. 3d", "B.R.", "M.J.", "U.S.L.W."),
)
def test_does_not_infer_court_from_non_exhaustive_reporter(reporter: str | None) -> None:
    inference = get_reporter_inference(reporter)
    assert inference.exact_court_id is None
