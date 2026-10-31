"""TextDNA analysis pipeline.

Layers, mirroring the original TextDNA concept:

1. linguistic  -- spaCy tokenisation, POS tags, dependency relations, NER
2. semantic    -- sentence embeddings, coherence and drift between
                  consecutive sentences
3. argument    -- claim / premise role classification from cue patterns
4. cognitive   -- fallacy and bias detectors (transparent rules, see
                  detectors.py)
5. generative  -- the composite TextDNA Score and a plain-language
                  summary generated from the findings

Everything runs locally on CPU. The embedding backend actually used
(sentence-transformers or the TF-IDF fallback) is reported in the output.
"""

from __future__ import annotations

from collections import Counter

import spacy

from . import detectors, embeddings, scoring

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def analyze(text: str) -> dict:
    nlp = get_nlp()
    doc = nlp(text)

    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    tokens = [tok for tok in doc if not tok.is_space]
    words = [tok for tok in tokens if not tok.is_punct]
    word_count = len(words)

    # --- linguistic layer -------------------------------------------------
    entities = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        }
        for ent in doc.ents
    ]
    pos_counts = Counter(tok.pos_ for tok in tokens)
    dep_counts = Counter(tok.dep_ for tok in tokens)
    lexical_diversity = (
        len({tok.lemma_.lower() for tok in words}) / word_count if word_count else 0.0
    )
    sentence_word_counts = [
        len([t for t in sent if not t.is_punct and not t.is_space])
        for sent in doc.sents
        if sent.text.strip()
    ]

    # --- semantic layer ---------------------------------------------------
    semantic = embeddings.coherence_stats(sentences)

    # --- argument layer ---------------------------------------------------
    labelled = detectors.classify_sentences(sentences)
    argument = detectors.argument_summary(labelled)

    # --- cognitive layer --------------------------------------------------
    fallacies = detectors.detect_fallacies(sentences)
    bias = detectors.detect_bias(text, word_count)

    # --- generative layer: score + summary --------------------------------
    sub_scores = {
        "coherence": round(
            scoring.coherence_score(semantic["mean_similarity"]), 2
        ),
        "argumentation": round(scoring.argumentation_score(argument), 2),
        "neutrality": round(scoring.neutrality_score(bias["bias_score"]), 2),
        "soundness": round(scoring.soundness_score(fallacies), 2),
        "structure": round(
            scoring.structure_score(sentence_word_counts, lexical_diversity), 2
        ),
    }
    overall = scoring.composite(sub_scores)
    summary = _summarise(
        overall, sub_scores, fallacies, bias, argument, semantic
    )

    return {
        "text_stats": {
            "characters": len(text),
            "words": word_count,
            "sentences": len(sentences),
            "lexical_diversity": round(lexical_diversity, 4),
            "avg_sentence_length_words": round(
                sum(sentence_word_counts) / len(sentence_word_counts), 2
            )
            if sentence_word_counts
            else 0.0,
        },
        "linguistic": {
            "entities": entities,
            "pos_counts": dict(pos_counts),
            "top_dependencies": dep_counts.most_common(8),
        },
        "semantic": semantic,
        "argument": argument,
        "fallacies": fallacies,
        "bias": bias,
        "textdna_score": {
            "overall": overall,
            "sub_scores": sub_scores,
            "weights": scoring.WEIGHTS,
        },
        "summary": summary,
    }


def _summarise(overall, sub_scores, fallacies, bias, argument, semantic) -> str:
    parts = [f"TextDNA Score {overall}/100."]
    if argument["is_argumentative"]:
        parts.append(
            f"The text is argumentative: {argument['claim_count']} claim "
            f"sentence(s) supported by {argument['premise_count']} premise "
            f"sentence(s), {argument['evidence_sentence_count']} with "
            "evidence markers."
        )
    else:
        parts.append("The text does not show a clear claim-plus-premise argument structure.")
    if fallacies:
        types = sorted({f["type"] for f in fallacies})
        parts.append(
            f"{len(fallacies)} possible fallacy signal(s) detected "
            f"({', '.join(types)}); treat these as prompts for a closer "
            "read, not verdicts."
        )
    else:
        parts.append("No fallacy patterns from the built-in rule set were detected.")
    if bias["is_biased"]:
        parts.append(
            f"Loaded or absolutist language is present (bias score "
            f"{bias['bias_score']}); terms include "
            f"{', '.join(bias['loaded_terms_found'][:5]) or 'absolutist phrasing'}."
        )
    else:
        parts.append("Language is comparatively neutral by the loaded-language measure.")
    if semantic["coherence"] is not None:
        parts.append(
            f"Semantic coherence between consecutive sentences is "
            f"{semantic['coherence']} (backend: {semantic['backend']})."
        )
    return " ".join(parts)
