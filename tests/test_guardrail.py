"""Guardrail tests.

These matter more than the retrieval tests: the guardrail is the only thing
standing between a confident wrong answer and a user acting on it.
"""

import pytest

from app.guardrail import check
from app.index import Doc


def make_docs():
    return [
        Doc(nct_id="NCT01234567",
            title="Trastuzumab in HER2-positive Breast Cancer",
            conditions="HER2-positive Breast Cancer",
            interventions="DRUG: Trastuzumab",
            status="RECRUITING", url="https://example.org/1"),
        Doc(nct_id="NCT07654321",
            title="Pembrolizumab for Advanced Melanoma",
            conditions="Melanoma",
            interventions="DRUG: Pembrolizumab",
            status="RECRUITING", url="https://example.org/2"),
    ]


def test_fabricated_nct_id_is_caught():
    """A cited id absent from the context is a fabrication, regardless of overlap."""
    docs = make_docs()
    answer = "Consider [NCT09999999], a trastuzumab trial for HER2-positive breast cancer."

    report = check(answer, docs)

    assert "NCT09999999" in report.hallucinated_ids
    assert report.grounded is False, "a fabricated id must never pass"


def test_valid_citation_with_grounded_text_passes():
    docs = make_docs()
    answer = ("[NCT01234567] studies Trastuzumab in HER2-positive Breast Cancer "
              "and is RECRUITING.")

    report = check(answer, docs)

    assert report.hallucinated_ids == []
    assert report.overlap > 0.8
    assert report.grounded is True


def test_ungrounded_answer_fails_on_overlap():
    """Correct-looking prose invented from nowhere should score low."""
    docs = make_docs()
    answer = ("Patients typically receive six cycles of carboplatin followed by "
              "autologous stem cell transplantation with mandatory genomic "
              "sequencing before enrolment.")

    report = check(answer, docs)

    assert report.overlap < 0.5
    assert report.grounded is False


def test_empty_answer_is_not_grounded():
    assert check("", make_docs()).grounded is False


def test_stopwords_do_not_inflate_the_score():
    """An answer of pure filler must not look grounded."""
    docs = make_docs()
    report = check("The it is and of to in for on with that this.", docs)
    assert report.overlap == 0.0


@pytest.mark.parametrize("threshold,expected", [(0.0, True), (0.99, False)])
def test_threshold_is_respected(threshold, expected):
    docs = make_docs()
    answer = "[NCT01234567] examines Trastuzumab alongside an unrelated invented regimen."
    assert check(answer, docs, threshold=threshold).grounded is expected
