from app.analyzer import analyze

SAMPLE = (
    "The city should invest in bus lanes because they move more people "
    "per hour than car lanes. According to the transit authority, the "
    "pilot corridor in Berlin cut average commute times by 18%."
)


def test_report_structure():
    report = analyze(SAMPLE)
    for key in (
        "text_stats", "linguistic", "semantic", "argument",
        "fallacies", "bias", "textdna_score", "summary",
    ):
        assert key in report


def test_entities_found():
    report = analyze(SAMPLE)
    texts = {e["text"] for e in report["linguistic"]["entities"]}
    assert "Berlin" in texts


def test_score_bounds():
    report = analyze(SAMPLE)
    overall = report["textdna_score"]["overall"]
    assert 0.0 <= overall <= 100.0
    for value in report["textdna_score"]["sub_scores"].values():
        assert 0.0 <= value <= 100.0


def test_semantic_backend_reported():
    report = analyze(SAMPLE)
    # tests force the tfidf backend via conftest
    assert report["semantic"]["backend"] == "tfidf"
    assert report["semantic"]["coherence"] is not None


def test_related_sentences_more_coherent_than_unrelated():
    related = analyze(
        "The dog chased the ball across the park. "
        "The dog caught the ball and ran back happily."
    )
    unrelated = analyze(
        "The dog chased the ball across the park. "
        "Quantum entanglement challenges classical notions of locality."
    )
    assert (
        related["semantic"]["mean_similarity"]
        > unrelated["semantic"]["mean_similarity"]
    )


def test_single_sentence_coherence_is_null():
    report = analyze("Just one sentence here.")
    assert report["semantic"]["coherence"] is None
    # neutral fallback keeps the composite score well-defined
    assert 0.0 <= report["textdna_score"]["overall"] <= 100.0


def test_fallacy_lowers_soundness():
    clean = analyze("The committee published its report on Tuesday.")
    loaded = analyze("You're either with us or against us, there is no middle ground.")
    assert (
        loaded["textdna_score"]["sub_scores"]["soundness"]
        < clean["textdna_score"]["sub_scores"]["soundness"]
    )
