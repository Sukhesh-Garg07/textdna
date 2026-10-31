"""Composite TextDNA Score.

The score is a transparent weighted blend of five sub-scores, each in
[0, 100]. The weights and mappings are fixed constants, documented here
and in the README, so any score can be reproduced and audited by hand.

Sub-scores:
- coherence:     semantic similarity between consecutive sentences.
- argumentation: presence and balance of claims, premises and evidence.
- neutrality:    inverse of the bias (loaded-language) score.
- soundness:     penalty per distinct fallacy type detected.
- structure:     readability basics -- sentence-length regularity and
                 lexical diversity.

Composite weights: coherence .25, argumentation .25, neutrality .20,
soundness .20, structure .10.
"""

from __future__ import annotations

WEIGHTS = {
    "coherence": 0.25,
    "argumentation": 0.25,
    "neutrality": 0.20,
    "soundness": 0.20,
    "structure": 0.10,
}


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def coherence_score(mean_similarity: float | None) -> float:
    """Map a mean cosine similarity to 0-100.

    Similarities for on-topic consecutive sentences typically land in
    0.1-0.8 (dense embeddings) or 0.0-0.6 (TF-IDF). A linear ramp from
    0.0 -> 0 and 0.75 -> 100 is a simple, documented compromise.
    Single-sentence texts get a neutral 50 (nothing to compare).
    """
    if mean_similarity is None:
        return 50.0
    return _clamp((mean_similarity / 0.75) * 100.0)


def argumentation_score(argument: dict) -> float:
    claims = argument["claim_count"]
    premises = argument["premise_count"]
    evidence = argument["evidence_sentence_count"]
    score = 0.0
    if claims:
        score += 35.0
    if premises:
        score += 35.0
    if claims and premises:
        # balance bonus: neither side dominates completely
        ratio = min(claims, premises) / max(claims, premises)
        score += 15.0 * ratio
    score += min(15.0, evidence * 5.0)
    return _clamp(score)


def neutrality_score(bias_score: float) -> float:
    return _clamp(100.0 - bias_score * 2.0)


def soundness_score(fallacy_findings: list[dict]) -> float:
    distinct_types = {f["type"] for f in fallacy_findings}
    return _clamp(100.0 - 25.0 * len(distinct_types) - 5.0 * len(fallacy_findings))


def structure_score(sentence_word_counts: list[int], lexical_diversity: float) -> float:
    if not sentence_word_counts:
        return 0.0
    avg_len = sum(sentence_word_counts) / len(sentence_word_counts)
    # Sentences of roughly 8-30 words read as well-formed prose.
    if 8 <= avg_len <= 30:
        length_part = 60.0
    elif 4 <= avg_len < 8 or 30 < avg_len <= 45:
        length_part = 40.0
    else:
        length_part = 20.0
    diversity_part = _clamp(lexical_diversity * 100.0) * 0.4
    return _clamp(length_part + diversity_part)


def composite(sub_scores: dict[str, float]) -> float:
    total = sum(sub_scores[name] * weight for name, weight in WEIGHTS.items())
    return round(_clamp(total), 2)
