from app import detectors


def types_for(text):
    sentences = [s.strip() for s in text.replace("!", ".").split(".") if s.strip()]
    return {f["type"] for f in detectors.detect_fallacies(sentences)}


def test_ad_hominem_detected():
    assert "ad_hominem" in types_for("Don't listen to him, he is just a liar.")


def test_slippery_slope_detected():
    text = "If we allow this change, soon everything will be permitted."
    assert "slippery_slope" in types_for(text)


def test_false_dilemma_detected():
    assert "false_dilemma" in types_for("You're either with us or against us.")


def test_bandwagon_detected():
    assert "bandwagon" in types_for("Everyone knows this product works.")


def test_whataboutism_detected():
    assert "whataboutism" in types_for("But what about the other factory?")


def test_neutral_text_has_no_fallacies():
    text = (
        "The library opens at nine in the morning. "
        "It closes at eight in the evening."
    )
    assert types_for(text) == set()


def test_bias_flags_loaded_language():
    text = (
        "The corrupt regime is destroying our country with its disastrous "
        "and rigged policies, and the pathetic media repeats the propaganda."
    )
    result = detectors.detect_bias(text, word_count=22)
    assert result["is_biased"] is True
    assert "corrupt" in result["loaded_terms_found"]


def test_bias_leaves_neutral_text_alone():
    text = "The report describes rainfall levels recorded during the spring season."
    result = detectors.detect_bias(text, word_count=len(text.split()))
    assert result["is_biased"] is False


def test_argument_roles_claim_and_premise():
    sentences = [
        "The city should build more cycle lanes.",
        "Because cycling reduces congestion, according to transport data.",
    ]
    labelled = detectors.classify_sentences(sentences)
    summary = detectors.argument_summary(labelled)
    assert summary["is_argumentative"] is True
    assert summary["claim_count"] >= 1
    assert summary["premise_count"] >= 1


def test_non_argumentative_text():
    sentences = ["The cat slept on the mat.", "Outside, it was raining."]
    summary = detectors.argument_summary(detectors.classify_sentences(sentences))
    assert summary["is_argumentative"] is False
