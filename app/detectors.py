"""Transparent rule-based detectors for fallacies, bias and argument roles.

Design principle: every finding must be explainable. Each detector is a
set of documented lexical / regex patterns with a plain-language
explanation of *why* the pattern is a signal. No black-box classifier is
used for fallacy or bias detection, so a user can always audit why a text
was flagged. The known cost of this approach is recall: paraphrased or
subtle fallacies will be missed. The evaluation harness in ``eval/``
measures exactly how well these rules do on a small starter set.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Fallacy rules
# ---------------------------------------------------------------------------
# Each rule: (fallacy_name, [regex patterns], explanation)
FALLACY_RULES: list[tuple[str, list[str], str]] = [
    (
        "ad_hominem",
        [
            r"\b(you|he|she|they)(?:'re| are| is)?\s+(just\s+)?(a\s+)?(stupid|idiot|moron|liar|fraud|corrupt|crook|clown|fool|hypocrite|incompetent)\b",
            r"\bdon'?t listen to (him|her|them|that \w+)\b",
            r"\bwhat (does|would) (he|she|they) know about\b",
            r"\bconsider the source\b",
            r"\btypical of (him|her|them|those people)\b",
        ],
        "Attacks the person making the argument instead of the argument itself.",
    ),
    (
        "straw_man",
        [
            r"\bso (you'?re|basically you'?re) saying\b",
            r"\bmy opponent (wants|believes|thinks|claims)\b",
            r"\bthey (just\s+)?want to (ban|destroy|abolish|take away) (all|every|your)\b",
            r"\bin other words,? (you|they) (want|believe|think)\b",
        ],
        "Restates the opposing position in an exaggerated form that is easier to attack.",
    ),
    (
        "slippery_slope",
        [
            r"\bnext thing you know\b",
            r"\bbefore long\b",
            r"\bif we (allow|let|permit|accept)\b.{0,80}\b(soon|then|eventually|next)\b",
            r"\bwill (inevitably|soon) lead to\b",
            r"\bfirst they\b.{0,60}\bthen (they|it)\b",
            r"\ba slippery slope\b",
        ],
        "Claims one step will cascade into extreme consequences without justifying each link.",
    ),
    (
        "false_dilemma",
        [
            r"\byou'?re either\b.{0,60}\bor\b",
            r"\beither you('re| are)\b.{0,60}\bor you('re| are)\b",
            r"\bthere are only two (choices|options|sides)\b",
            r"\bno middle ground\b",
            r"\bif you'?re not with us,? you'?re against us\b",
        ],
        "Presents only two options when more may exist.",
    ),
    (
        "bandwagon",
        [
            r"\bevery(one|body) (knows|agrees|believes|is doing)\b",
            r"\bmillions of people can'?t be wrong\b",
            r"\beveryone else is\b",
            r"\bjoin (everyone|the majority|us) (and|in)\b",
            r"\bthe whole (country|world) (knows|agrees)\b",
        ],
        "Treats popularity as evidence that a claim is true.",
    ),
    (
        "hasty_generalization",
        [
            r"\b(all|every) (politicians|men|women|millennials|boomers|liberals|conservatives|immigrants|rich people|poor people|journalists|bankers|lawyers|students) (are|always|never)\b",
            r"\bthose people (are|always|never)\b",
            r"\bthey (are all|all just)\b",
            r"\bevery single (one of them|politician|lawyer)\b",
        ],
        "Draws a conclusion about a whole group from an unrepresentative sample.",
    ),
    (
        "whataboutism",
        [
            r"\bbut what about\b",
            r"\bwhat about (the|when|how)\b",
        ],
        "Deflects criticism by pointing at a different wrongdoing instead of answering it.",
    ),
    (
        "appeal_to_vague_authority",
        [
            r"\bexperts agree\b",
            r"\bscientists (say|agree|confirm)\b(?! (in|at|from) (the )?[A-Z])",
            r"\bstudies show\b(?! (in|published))",
            r"\bresearch proves\b",
            r"\bit is (well )?known that experts\b",
        ],
        "Leans on unnamed experts or studies with no checkable source.",
    ),
]

_COMPILED_FALLACY_RULES = [
    (name, [re.compile(p, re.IGNORECASE) for p in patterns], why)
    for name, patterns, why in FALLACY_RULES
]


def detect_fallacies(sentences: list[str]) -> list[dict]:
    """Run every fallacy rule over every sentence. Returns findings."""
    findings: list[dict] = []
    for idx, sentence in enumerate(sentences):
        for name, patterns, why in _COMPILED_FALLACY_RULES:
            for pattern in patterns:
                match = pattern.search(sentence)
                if match:
                    findings.append(
                        {
                            "type": name,
                            "sentence_index": idx,
                            "sentence": sentence,
                            "matched_text": match.group(0),
                            "explanation": why,
                        }
                    )
                    break  # one finding per rule per sentence is enough
    return findings


# ---------------------------------------------------------------------------
# Bias: loaded language + absolutist framing
# ---------------------------------------------------------------------------
LOADED_TERMS = {
    "corrupt", "crooked", "stupid", "idiotic", "moronic", "pathetic",
    "disgusting", "inhumane", "evil", "greedy", "radical", "extremist",
    "regime", "destroying", "destroyed", "devastating", "catastrophic",
    "disaster", "disastrous", "rigged", "stolen", "fraud", "fraudulent",
    "liar", "lying", "incompetent", "clueless", "brainwashed", "sheep",
    "propaganda", "hoax", "scam", "shameful", "outrageous", "insane",
    "absurd", "ridiculous", "hateful", "vile", "toxic", "poisoning",
    "betrayal", "betrayed", "traitor", "enemy of the people",
}

ABSOLUTE_TERMS = {
    "always", "never", "everyone", "everybody", "nobody", "no one",
    "all", "none", "every", "completely", "totally", "utterly",
}


def detect_bias(text: str, word_count: int) -> dict:
    """Score loaded / absolutist language density.

    bias_score is 0-100. It combines loaded-term density (per 100 words)
    and absolute-term density with fixed, documented weights. A text is
    labelled biased at score >= 12 -- roughly: sustained loaded language,
    not a single strong word.
    """
    lowered = f" {text.lower()} "
    loaded_hits = sorted(
        {term for term in LOADED_TERMS if f" {term}" in lowered or f"{term} " in lowered}
    )
    # count occurrences for density, not just presence
    loaded_count = sum(lowered.count(term) for term in LOADED_TERMS)
    absolute_count = sum(
        len(re.findall(rf"\b{re.escape(term)}\b", lowered))
        for term in ABSOLUTE_TERMS
    )
    per_hundred = max(word_count, 1) / 100.0
    loaded_density = loaded_count / per_hundred
    absolute_density = absolute_count / per_hundred
    score = min(100.0, loaded_density * 6.0 + absolute_density * 1.5)
    return {
        "bias_score": round(score, 2),
        "is_biased": score >= 12.0,
        "threshold": 12.0,
        "loaded_terms_found": loaded_hits,
        "loaded_term_count": loaded_count,
        "absolute_term_count": absolute_count,
        "loaded_density_per_100_words": round(loaded_density, 3),
        "absolute_density_per_100_words": round(absolute_density, 3),
        "method": "lexicon density of loaded and absolutist terms (transparent, auditable)",
    }


# ---------------------------------------------------------------------------
# Argument roles: claim / premise cue classification per sentence
# ---------------------------------------------------------------------------
CLAIM_CUES = [
    r"\bi believe\b", r"\bshould\b", r"\bmust\b",
    r"\btherefore\b", r"\bthus\b", r"\bin conclusion\b", r"\bclearly\b",
    r"\bought to\b", r"\bit is (essential|necessary|important) (that|to)\b",
    r"\bthe (best|only) (way|solution|option)\b", r"\bneeds? to be\b",
]
PREMISE_CUES = [
    r"\bbecause\b", r"\bsince\b", r"\bgiven that\b", r"\bfor example\b",
    r"\bfor instance\b", r"\baccording to\b", r"\bdata (shows?|suggests?)\b",
    r"\bstudies (show|found|suggest|indicate)\b",
    r"\bresearch (shows|found|indicates)\b", r"\bevidence (shows|suggests)\b",
    r"\bstatistics (show|suggest)\b", r"\ba \d{4} (study|report|survey)\b",
    r"\bin fact\b", r"\bthe fact that\b",
]
EVIDENCE_CUES = [
    r"\baccording to\b", r"\bdata\b", r"\bstudy\b", r"\bstudies\b",
    r"\bresearch\b", r"\bsurvey\b", r"\bstatistics\b", r"\b\d+(\.\d+)?\s?%\b",
    r"\breported\b",
]

_CLAIM_RES = [re.compile(p, re.IGNORECASE) for p in CLAIM_CUES]
_PREMISE_RES = [re.compile(p, re.IGNORECASE) for p in PREMISE_CUES]
_EVIDENCE_RES = [re.compile(p, re.IGNORECASE) for p in EVIDENCE_CUES]


def classify_sentences(sentences: list[str]) -> list[dict]:
    """Label each sentence claim / premise / evidence-bearing / other."""
    labelled = []
    for idx, sentence in enumerate(sentences):
        claim_hits = [p.pattern for p in _CLAIM_RES if p.search(sentence)]
        premise_hits = [p.pattern for p in _PREMISE_RES if p.search(sentence)]
        has_evidence = any(p.search(sentence) for p in _EVIDENCE_RES)
        if claim_hits and premise_hits:
            role = "claim+premise"
        elif claim_hits:
            role = "claim"
        elif premise_hits:
            role = "premise"
        else:
            role = "other"
        labelled.append(
            {
                "index": idx,
                "sentence": sentence,
                "role": role,
                "has_evidence_marker": has_evidence,
                "claim_cues": len(claim_hits),
                "premise_cues": len(premise_hits),
            }
        )
    return labelled


def argument_summary(labelled: list[dict]) -> dict:
    claims = [s for s in labelled if "claim" in s["role"]]
    premises = [s for s in labelled if "premise" in s["role"]]
    evidence = [s for s in labelled if s["has_evidence_marker"]]
    return {
        "is_argumentative": bool(claims) and bool(premises),
        "claim_count": len(claims),
        "premise_count": len(premises),
        "evidence_sentence_count": len(evidence),
        "sentences": labelled,
    }
